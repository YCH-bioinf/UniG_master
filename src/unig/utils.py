"""Utility functions and classes"""

import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
import os
import json
from anndata import AnnData
import anndata as ad
from pathlib import Path
from .settings import settings
import torch
from anndata.io import read_hdf
from tqdm import tqdm
from typing import Optional




def discretize(adata, layer=None, n_bins=5, max_bins=100):
    """Discretize continuous values

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    layer: str, optional (default: None)
        The layer used to perform discretization
    n_bins: int, optional (default: 5)
        The number of bins to produce.
        It must be smaller than max_bins.
    max_bins: int, optional (default: 100)
        The number of bins used in the initial approximation.

    Returns
    -------
    updates adata with the following fields
    .layer['unig'] : array_like
        The matrix of discretized values to build the unig graph.
    .uns['disc'] : dict
        bin_edges: The edges of each bin.
        bin_count: The number of values in each bin.
        hist_edges: The edges of each bin in the initial approximation.
        hist_count: The number of values in each bin for the initial approximation.
    """
    if layer is None:
        X = csr_matrix(adata.X)
    else:
        X = csr_matrix(adata.layers[layer])
    nonzero_cont = X.data

    hist_count, hist_edges = np.histogram(nonzero_cont, bins=max_bins, density=False)
    hist_centroids = (hist_edges[0:-1] + hist_edges[1:])/2

    kmeans = KMeans(n_clusters=n_bins, random_state=2021, n_init='auto').fit(
        hist_centroids.reshape(-1, 1), sample_weight=hist_count)
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())

    padding = (hist_edges[-1] - hist_edges[0])/(max_bins*10)
    bin_edges = np.array(
        [hist_edges[0]-padding] +
        list((cluster_centers[0:-1] + cluster_centers[1:])/2) +
        [hist_edges[-1]+padding])
    nonzero_disc = np.digitize(nonzero_cont, bin_edges).reshape(-1,)
    bin_count = np.unique(nonzero_disc, return_counts=True)[1]

    adata.layers['unig'] = X.copy()
    adata.layers['unig'].data = nonzero_disc
    adata.uns['disc'] = dict()
    adata.uns['disc']['bin_edges'] = bin_edges
    adata.uns['disc']['bin_count'] = bin_count
    adata.uns['disc']['hist_edges'] = hist_edges
    adata.uns['disc']['hist_count'] = hist_count


def SpaNeighbor(adata, k=10):
    """Compute spatial neighborhood graph"""
    if 'spatial' not in adata.obsm:
       raise KeyError(
        "The AnnData object does not contain 'spatial' coordinates in .obsm. "
        "Please make sure that .obsm['spatial'] is available."
    )
    coords = adata.obsm['spatial']
    cell_names = adata.obs_names

    dist_mat = pairwise_distances(coords, metric='euclidean')
    n_cells = dist_mat.shape[0]
    adj_mat = np.zeros((n_cells, n_cells), dtype=int)

    for i in range(n_cells):
        nearest_idx = np.argsort(dist_mat[i])[1:k+1]
        adj_mat[i, nearest_idx] = 1

    # Undirected graph
    adj_mat = np.maximum(adj_mat, adj_mat.T)
    adj_df = pd.DataFrame(adj_mat, index=cell_names, columns=cell_names)
    adata_CC = AnnData(adj_df)
    return adata_CC


def read_embedding(path_emb=None, path_entity=None, convert_alias=True,
                   path_entity_alias=None, prefix=None, num_epochs=None):
    """Read in entity embeddings from pbg training

    Parameters
    ----------
    path_emb: str, optional (default: None)
        Path to directory for pbg embedding model
        If None, .settings.pbg_params['checkpoint_path'] will be used.
    path_entity: str, optional (default: None)
        Path to entity name file
    prefix: list, optional (default: None)
        A list of entity type prefixes to include.
        By default, it reads in the embeddings of all entities.
    convert_alias: bool, optional (default: True)
        If True, it will convert entity aliases to the original indices
    path_entity_alias: str, optional (default: None)
        Path to entity alias file
    num_epochs: int, optional (default: None)
        The embedding result associated with num_epochs to read in

    Returns
    -------
    dict_adata: dict
        A dictionary of anndata objects of shape (#entities x #dimensions)
    """
    pbg_params = settings.pbg_params
    if path_emb is None:
        path_emb = pbg_params['checkpoint_path']
    if path_entity is None:
        path_entity = pbg_params['entity_path']
    if num_epochs is None:
        num_epochs = pbg_params["num_epochs"]
    if prefix is None:
        prefix = []
    assert isinstance(prefix, list), "prefix must be list"
    
    if convert_alias:
        if path_entity_alias is None:
            path_entity_alias = Path(path_emb).parent.as_posix()
        df_entity_alias = pd.read_csv(
            os.path.join(path_entity_alias, 'entity_alias.txt'),
            header=0, index_col=0, sep='\t')
        df_entity_alias['id'] = df_entity_alias.index
        df_entity_alias.index = df_entity_alias['alias'].values

    dict_adata = dict()
    for x in os.listdir(path_emb):
        if x.startswith('embeddings'):
            entity_type = x.split('_')[1]
            if (len(prefix) == 0) or (entity_type in prefix):
                # Try .h5 first (torchbiggraph), then .h5ad (custom trainer)
                h5_path = os.path.join(path_emb, f'embeddings_{entity_type}_0.v{num_epochs}.h5')
                h5ad_path = os.path.join(path_emb, f'embeddings_{entity_type}_0.v{num_epochs}.h5ad')
                if os.path.exists(h5_path):
                    adata = read_hdf(h5_path, key="embeddings")
                    with open(os.path.join(path_entity, f'entity_names_{entity_type}_0.json'), "rt") as tf:
                        names_entity = json.load(tf)
                    if convert_alias:
                        names_entity = df_entity_alias.loc[names_entity, 'id'].tolist()
                    adata.obs.index = names_entity
                elif os.path.exists(h5ad_path):
                    adata = ad.read_h5ad(h5ad_path)
                else:
                    continue
                dict_adata[entity_type] = adata
              
    return dict_adata



def rand_projections(
    embedding_dim,
    num_samples=50,
    device='cpu'
):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """

    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor).to(device)


def _sliced_wasserstein_distance(
    encoded_samples,
    distribution_samples,
    num_projections=50,
    p=2,
    device='cpu'
):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """

    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections).to(device)
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1).to(device))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()


def sliced_wasserstein_distance(
    encoded_samples,
    transformed_samples,
    num_projections=50,
    p=2,
    device='cpu'
):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive batch size from encoded samples
    # draw random samples from latent space prior distribution

    # approximate mean wasserstein_distance between encoded and prior distributions
    # for each random projection
    swd = _sliced_wasserstein_distance(encoded_samples, transformed_samples, num_projections, p, device)
    return swd


def ot_loss_swd(
    mat_pred: torch.Tensor,
    mat_target: torch.Tensor,
    num_projections: int = 50,
    p: int = 2,
) -> torch.Tensor:
    """
    Computes Sliced Wasserstein Distance between two matrices (treated as empirical distributions).

    This function treats the rows of the input matrices as samples from two distributions
    and computes the Sliced Wasserstein Distance between them.

    Args:
        mat_pred (torch.Tensor): The predicted matrix, with shape (n_samples, n_features).
        mat_target (torch.Tensor): The target matrix, with shape (n_samples, n_features).
        num_projections (int): The number of projections to use for SWD.
        p (int): The power for the Wasserstein distance metric.

    Returns:
        torch.Tensor: A scalar tensor representing the SWD loss.
    """
    device = mat_pred.device
    mat_target = mat_target.to(device)

    # The sliced_wasserstein_distance function expects (n_samples, embedding_dim),
    # which corresponds to (n_cells, n_features) in this context.
    swd = sliced_wasserstein_distance(
        mat_pred,
        mat_target,
        num_projections=num_projections,
        p=p,
        device=device
    )
    return swd


from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix, find
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix, find
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import anndata as ad

# def infer_edges_seurat(adata_ref,
#                        adata_query,
#                        feature='spatially_variable',
#                        n_components=20,
#                        k=20,
#                        anchor_percentile=90.0,
#                        layer=None):
#     """Infer edges between reference and query observations using a Seurat-like
#     PCA projection and anchor filtering approach (inspired by Seurat v4 rPCA).

#     Parameters
#     ----------
#     adata_ref: `AnnData`
#         Annotated reference data matrix.
#     adata_query: `AnnData`
#         Annotated query data matrix.
#     feature: `str`, optional (default: 'spatially_variable')
#         Feature used for edges inference. The data type of `.var[feature]`
#         needs to be `bool`. If None, all shared features are used.
#     n_components: `int`, optional (default: 20)
#         The number of components for PCA.
#     k: `int`, optional (default: 20)
#         The number of nearest neighbors to consider for MNN search.
#     anchor_percentile: `float`, optional (default: 90.0)
#         The percentile for filtering MNN anchors based on their distance.
#         Anchors with a distance above this percentile will be removed.
#         Set to 100.0 to disable filtering.
#     layer: `str`, optional (default: None)
#         The layer used to perform edge inference. If None, `.X` will be used.

#     Returns
#     -------
#     adata_ref_query: `AnnData`
#         An AnnData object where `obs` are from `adata_ref` and `var` are from
#         `adata_query`. The `.X` slot stores similarity scores of the filtered
#         anchors, and `.layers['connectivities']` stores the binary connectivity matrix.
#     """
#     # 1. Select shared features
#     if feature is None:
#         feature_ref = adata_ref.var_names
#     else:
#         mask_ref = adata_ref.var.get(feature, False)
#         feature_ref = adata_ref.var_names[mask_ref]
#     feature_query = adata_query.var_names
#     feature_shared = list(set(feature_ref).intersection(set(feature_query)))
#     print(f'#shared features: {len(feature_shared)}')

#     if layer is None:
#         X_ref = adata_ref[:, feature_shared].X
#         X_query = adata_query[:, feature_shared].X
#     else:
#         X_ref = adata_ref[:, feature_shared].layers[layer]
#         X_query = adata_query[:, feature_shared].layers[layer]

#     # 2. Find shared low-dimensional space using PCA projection
#     print(f'Performing PCA on reference and projecting query data ...')
#     # Fit PCA on the reference data
#     pca = PCA(n_components=n_components)
#     # Transform both reference and query data into the reference's PCA space
#     try:
#         X_ref_c = pca.fit_transform(X_ref.toarray())
#         X_query_c = pca.transform(X_query.toarray())
#     except Exception as e:
#         print(f"Error during PCA, check if data has sufficient variance: {e}")
#         return None

#     # 3. L2-normalize the PCA embeddings
#     X_ref_c = normalize(X_ref_c, norm='l2', axis=1)
#     X_query_c = normalize(X_query_c, norm='l2', axis=1)

#     # 4. Find Mutual Nearest Neighbors (MNNs) in the aligned PCA space
#     print('Searching for mutual nearest neighbors (anchors) ...')
#     # Find neighbors of query cells in reference
#     conn_query_in_ref, dist_query_in_ref = _knn(
#         X_ref=X_ref_c, X_query=X_query_c, k=k)
#     # Find neighbors of reference cells in query
#     conn_ref_in_query, _ = _knn(
#         X_ref=X_query_c, X_query=X_ref_c, k=k)

#     # MNNs are pairs where each cell is a neighbor of the other
#     mnn_matrix = conn_query_in_ref.multiply(conn_ref_in_query.T)
#     ref_idx, query_idx, _ = find(mnn_matrix)
#     print(f'Found {len(ref_idx)} initial anchors (MNNs).')

#     if len(ref_idx) == 0:
#         print("No anchors found. Cannot proceed.")
#         return None

#     # 5. Filter anchors based on distance in PCA space
#     mnn_distances = dist_query_in_ref[ref_idx, query_idx].A.flatten()
#     if len(mnn_distances) > 0 and anchor_percentile < 100.0:
#         dist_cutoff = np.percentile(mnn_distances, anchor_percentile)
#         keep_mask = mnn_distances <= dist_cutoff
#         final_ref_idx = ref_idx[keep_mask]
#         final_query_idx = query_idx[keep_mask]
#         final_distances = mnn_distances[keep_mask]
#         print(f'Kept {len(final_ref_idx)} anchors after filtering at {anchor_percentile:.1f} percentile (distance cutoff: {dist_cutoff:.4f}).')
#     else:
#         final_ref_idx, final_query_idx, final_distances = ref_idx, query_idx, mnn_distances
#         dist_cutoff = np.inf
#         print('Kept all initial anchors (filtering disabled or no anchors to filter).')


#     # 6. Construct result AnnData object
#     shape = (adata_ref.shape[0], adata_query.shape[0])
#     # Similarity score is inversely related to distance
#     similarity_scores = 1 / (1 + final_distances)

#     conn_matrix = csr_matrix(
#         (np.ones_like(final_ref_idx), (final_ref_idx, final_query_idx)),
#         shape=shape)
#     sim_matrix = csr_matrix(
#         (similarity_scores, (final_ref_idx, final_query_idx)),
#         shape=shape)

#     adata_ref_query = ad.AnnData(X=sim_matrix,
#                                  obs=adata_ref.obs,
#                                  var=adata_query.obs)
#     adata_ref_query.layers['connectivities'] = conn_matrix
#     adata_ref_query.obsm['ref_pca'] = X_ref_c
#     adata_ref_query.varm['query_pca_proj'] = X_query_c
#     adata_ref_query.uns['anchor_stats'] = {
#         'initial_anchors': len(ref_idx),
#         'filtered_anchors': len(final_ref_idx),
#         'distance_cutoff': dist_cutoff,
#     }
#     return adata_ref_query


def _knn(X_ref,
         X_query=None,
         k=20,
         leaf_size=40,
         metric='euclidean'):
    """Calculate K nearest neigbors for each row.
    """
    if X_query is None:
        X_query = X_ref.copy()
    kdt = KDTree(X_ref, leaf_size=leaf_size, metric=metric)
    kdt_d, kdt_i = kdt.query(X_query, k=k, return_distance=True)
    # kdt_i = kdt_i[:, 1:]  # exclude the point itself
    # kdt_d = kdt_d[:, 1:]  # exclude the point itself
    sp_row = np.repeat(np.arange(kdt_i.shape[0]), kdt_i.shape[1])
    sp_col = kdt_i.flatten()
    sp_conn = np.repeat(1, len(sp_row))
    sp_dist = kdt_d.flatten()
    mat_conn_ref_query = csr_matrix(
        (sp_conn, (sp_row, sp_col)),
        shape=(X_query.shape[0], X_ref.shape[0])).T
    mat_dist_ref_query = csr_matrix(
        (sp_dist, (sp_row, sp_col)),
        shape=(X_query.shape[0], X_ref.shape[0])).T
    return mat_conn_ref_query, mat_dist_ref_query


def infer_edges(adata_ref,
                adata_query,
                feature='variable',
                n_components=20,
                random_state=42,
                layer=None,
                k=20,
                metric='euclidean',
                leaf_size=40,
                **kwargs):
    """Infer edges between reference and query observations

    Parameters
    ----------
    adata_ref: `AnnData`
        Annotated reference data matrix.
    adata_query: `AnnData`
        Annotated query data matrix.
    feature: `str`, optional (default: None)
        Feature used for edges inference.
        The data type of `.var[feature]` needs to be `bool`
    n_components: `int`, optional (default: 20)
        The number of components used in `randomized_svd`
        for comparing reference and query observations
    random_state: `int`, optional (default: 42)
        The seed used for truncated randomized SVD
    n_top_edges: `int`, optional (default: None)
        The number of edges to keep
        If specified, `percentile` will be ignored
    percentile: `float`, optional (default: 0.01)
        The percentile of edges to keep
    k: `int`, optional (default: 5)
        The number of nearest neighbors to consider within each dataset
    metric: `str`, optional (default: 'euclidean')
        The metric to use when calculating distance between
        reference and query observations
    layer: `str`, optional (default: None)
        The layer used to perform edge inference
        If None, `.X` will be used.
    kwargs:
        Other keyword arguments are passed down to `randomized_svd()`

    Returns
    -------
    adata_ref_query: `AnnData`
        Annotated relation matrix betwewn reference and query observations
        Store reference entity as observations and query entity as variables
    """
    if feature is None:
        feature_ref = adata_ref.var_names
    else:
        mask_ref = adata_ref.var[feature]
        feature_ref = adata_ref.var_names[mask_ref]
    feature_query = adata_query.var_names
    feature_shared = list(set(feature_ref).intersection(set(feature_query)))
    print(f'#shared features: {len(feature_shared)}')
    if layer is None:
        X_ref = adata_ref[:, feature_shared].X
        X_query = adata_query[:, feature_shared].X
    else:
        X_ref = adata_ref[:, feature_shared].layers[layer]
        X_query = adata_query[:, feature_shared].layers[layer]

    if any(X_ref.sum(axis=1) == 0) or any(X_query.sum(axis=1) == 0):
        raise ValueError(
            f'Some nodes contain zero expressed {feature} features.\n'
            f'Please try to include more {feature} features.')

    print('Performing randomized SVD ...')
    mat = X_ref @ X_query.T

    U, Sigma, VT = randomized_svd(mat,
                                  n_components=n_components,
                                  random_state=random_state,
                                  **kwargs)
    svd_data = np.vstack((U, VT.T))
    X_svd_ref = svd_data[:U.shape[0], :]
    X_svd_query = svd_data[-VT.shape[1]:, :]
    X_svd_ref = X_svd_ref / (X_svd_ref**2).sum(-1, keepdims=True)**0.5
    X_svd_query = X_svd_query / (X_svd_query**2).sum(-1, keepdims=True)**0.5

    # print('Searching for neighbors within each dataset ...')
    # knn_conn_ref, knn_dist_ref = _knn(
    #     X_ref=X_svd_ref,
    #     k=k,
    #     leaf_size=leaf_size,
    #     metric=metric)
    # knn_conn_query, knn_dist_query = _knn(
    #     X_ref=X_svd_query,
    #     k=k,
    #     leaf_size=leaf_size,
    #     metric=metric)

    print('Searching for mutual nearest neighbors ...')
    knn_conn_ref_query, knn_dist_ref_query = _knn(
        X_ref=X_svd_ref,
        X_query=X_svd_query,
        k=k,
        leaf_size=leaf_size,
        metric=metric)
    knn_conn_query_ref, knn_dist_query_ref = _knn(
        X_ref=X_svd_query,
        X_query=X_svd_ref,
        k=k,
        leaf_size=leaf_size,
        metric=metric)

    sum_conn_ref_query = knn_conn_ref_query + knn_conn_query_ref.T
    id_x, id_y, values = find(sum_conn_ref_query > 1)
    print(f'{len(id_x)} edges are selected')
    conn_ref_query = csr_matrix(
        (values*1, (id_x, id_y)),
        shape=(knn_conn_ref_query.shape))
    dist_ref_query = csr_matrix(
        (knn_dist_ref_query[id_x, id_y].A.flatten(), (id_x, id_y)),
        shape=(knn_conn_ref_query.shape))
    # it's easier to distinguish zeros (no connection vs zero distance)
    # using similarity scores
    sim_ref_query = csr_matrix(
        (1/(dist_ref_query.data+1), dist_ref_query.nonzero()),
        shape=(dist_ref_query.shape))  # similarity scores

    # print('Computing similarity scores ...')
    # dist_ref_query = pairwise_distances(X_svd_ref,
    #                                     X_svd_query,
    #                                     metric=metric)
    # sim_ref_query = 1/(1+dist_ref_query)
    # # remove low similarity entries to save memory
    # sim_ref_query = np.where(
    #     sim_ref_query < np.percentile(sim_ref_query, pct_keep*100),
    #     0, sim_ref_query)
    # sim_ref_query = csr_matrix(sim_ref_query)

    adata_ref_query = ad.AnnData(X=sim_ref_query,
                                 obs=adata_ref.obs,
                                 var=adata_query.obs)
    adata_ref_query.layers['unig'] = conn_ref_query
    adata_ref_query.obsm['svd'] = X_svd_ref
    # adata_ref_query.obsp['conn'] = knn_conn_ref
    # adata_ref_query.obsp['dist'] = knn_dist_ref
    adata_ref_query.varm['svd'] = X_svd_query
    # adata_ref_query.varp['conn'] = knn_conn_query
    # adata_ref_query.varp['dist'] = knn_dist_query
    return adata_ref_query



"""Predict gene scores based on chromatin accessibility"""

import numpy as np
import pandas as pd
import anndata as ad
import io
import pybedtools
from scipy.sparse import (
    coo_matrix,
    csr_matrix
)
import pkgutil

def _uniquify(seq, sep='-'):
    """Uniquify a list of strings.

    Adding unique numbers to duplicate values.

    Parameters
    ----------
    seq : `list` or `array-like`
        A list of values
    sep : `str`
        Separator

    Returns
    -------
    seq: `list` or `array-like`
        A list of updated values
    """

    dups = {}

    for i, val in enumerate(seq):
        if val not in dups:
            # Store index of first occurrence and occurrence value
            dups[val] = [i, 1]
        else:
            # Increment occurrence value, index value doesn't matter anymore
            dups[val][1] += 1

            # Use stored occurrence value
            seq[i] += (sep+str(dups[val][1]))

    return seq


class GeneScores:
    """A class used to represent gene scores

    Attributes
    ----------

    Methods
    -------

    """
    def __init__(self,
                 adata,
                 genome,
                 gene_anno=None,
                 tss_upstream=1e5,
                 tss_downsteam=1e5,
                 gb_upstream=5000,
                 cutoff_weight=1,
                 use_top_pcs=True,
                 use_precomputed=True,
                 use_gene_weight=True,
                 min_w=1,
                 max_w=5,
                 local_file_path=None):
        """
        Parameters
        ----------
        adata: `Anndata`
            Input anndata
        genome : `str`
            The genome name
        """
        self.adata = adata
        self.genome = genome
        self.gene_anno = gene_anno
        self.tss_upstream = tss_upstream
        self.tss_downsteam = tss_downsteam
        self.gb_upstream = gb_upstream
        self.cutoff_weight = cutoff_weight
        self.use_top_pcs = use_top_pcs
        self.use_precomputed = use_precomputed
        self.use_gene_weight = use_gene_weight
        self.min_w = min_w
        self.max_w = max_w
        self.local_file_path = local_file_path

    def _read_gene_anno(self):
        """Read in gene annotation

        Parameters
        ----------

        Returns
        -------

        """
        assert (self.genome in ['hg19', 'hg38', 'mm9', 'mm10']),\
            "`genome` must be one of ['hg19','hg38','mm9','mm10']"

        # Prefer an explicitly supplied annotation file, otherwise fall back
        # to the gene_anno directory bundled with this project source tree.
        local_file_path = self.local_file_path
        if local_file_path is None:
            local_file_path = Path(__file__).resolve().parent.parent / \
                'gene_anno' / f'{self.genome}_genes.bed'
        local_file_path = Path(local_file_path)
        if not local_file_path.exists():
            raise FileNotFoundError(
                f"Gene annotation file not found: {local_file_path}. "
                "Pass `local_file_path=` or `gene_anno=` to gene_scores()."
            )

        with open(local_file_path, 'rb') as f:
            bin_str = f.read()
            
        gene_anno = pd.read_csv(io.BytesIO(bin_str),
                                encoding='utf8',
                                sep='\t',
                                header=None,
                                names=['chr', 'start', 'end',
                                       'symbol', 'strand'])
        self.gene_anno = gene_anno
        return self.gene_anno

    def _extend_tss(self, pbt_gene):
        """Extend transcription start site in both directions

        Parameters
        ----------

        Returns
        -------

        """
        ext_tss = pbt_gene
        if ext_tss['strand'] == '+':
            ext_tss.start = max(0, ext_tss.start - self.tss_upstream)
            ext_tss.end = max(ext_tss.end, ext_tss.start + self.tss_downsteam)
        else:
            ext_tss.start = max(0, min(ext_tss.start,
                                       ext_tss.end - self.tss_downsteam))
            ext_tss.end = ext_tss.end + self.tss_upstream
        return ext_tss

    def _extend_genebody(self, pbt_gene):
        """Extend gene body upstream

        Parameters
        ----------

        Returns
        -------

        """
        ext_gb = pbt_gene
        if ext_gb['strand'] == '+':
            ext_gb.start = max(0, ext_gb.start - self.gb_upstream)
        else:
            ext_gb.end = ext_gb.end + self.gb_upstream
        return ext_gb

    def _weight_genes(self):
        """Weight genes

        Parameters
        ----------

        Returns
        -------

        """
        gene_anno = self.gene_anno
        gene_size = gene_anno['end'] - gene_anno['start']
        w = 1/gene_size
        w_scaled = (self.max_w-self.min_w) * (w-min(w)) / (max(w)-min(w)) \
            + self.min_w
        return w_scaled

    def cal_gene_scores(self):
        """Calculate gene scores

        Parameters
        ----------

        Returns
        -------

        """
        adata = self.adata
        if self.gene_anno is None:
            gene_ann = self._read_gene_anno()
        else:
            gene_ann = self.gene_anno

        df_gene_ann = gene_ann.copy()
        df_gene_ann.index = _uniquify(df_gene_ann['symbol'].values)
        if self.use_top_pcs:
            mask_p = adata.var['top_pcs']
        else:
            mask_p = pd.Series(True, index=adata.var_names)
        df_peaks = adata.var[mask_p][['chr', 'start', 'end']].copy()

        if 'gene_scores' not in adata.uns_keys():
            print('Gene scores are being calculated for the first time')
            print('`use_precomputed` has been ignored')
            self.use_precomputed = False

        if self.use_precomputed:
            print('Using precomputed overlap')
            df_overlap_updated = adata.uns['gene_scores']['overlap'].copy()
        else:
            # add the fifth column
            # so that pybedtool can recognize the sixth column as the strand
            df_gene_ann_for_pbt = df_gene_ann.copy()
            df_gene_ann_for_pbt['score'] = 0
            df_gene_ann_for_pbt = df_gene_ann_for_pbt[['chr', 'start', 'end',
                                                       'symbol', 'score',
                                                       'strand']]
            df_gene_ann_for_pbt['id'] = range(df_gene_ann_for_pbt.shape[0])

            df_peaks_for_pbt = df_peaks.copy()
            df_peaks_for_pbt['id'] = range(df_peaks_for_pbt.shape[0])

            pbt_gene_ann = pybedtools.BedTool.from_dataframe(
                df_gene_ann_for_pbt
                )
            pbt_gene_ann_ext = pbt_gene_ann.each(self._extend_tss)
            pbt_gene_gb_ext = pbt_gene_ann.each(self._extend_genebody)

            pbt_peaks = pybedtools.BedTool.from_dataframe(df_peaks_for_pbt)

            # peaks overlapping with extended TSS
            pbt_overlap = pbt_peaks.intersect(pbt_gene_ann_ext,
                                              wa=True,
                                              wb=True)
            df_overlap = pbt_overlap.to_dataframe(
                names=[x+'_p' for x in df_peaks_for_pbt.columns]
                + [x+'_g' for x in df_gene_ann_for_pbt.columns])
            # peaks overlapping with gene body
            pbt_overlap2 = pbt_peaks.intersect(pbt_gene_gb_ext,
                                               wa=True,
                                               wb=True)
            df_overlap2 = pbt_overlap2.to_dataframe(
                names=[x+'_p' for x in df_peaks_for_pbt.columns]
                + [x+'_g' for x in df_gene_ann_for_pbt.columns])

          
            if 'symbol_g' not in df_overlap.columns:
                 raise KeyError(f"'symbol_g' not found in df_overlap columns. Available columns: {df_overlap.columns}. "
                                f"df_peaks_cols: {df_peaks_for_pbt.columns}, df_gene_ann_cols: {df_gene_ann_for_pbt.columns}")

            # add distance and weight for each overlap
            df_overlap_updated = df_overlap.copy()
            df_overlap_updated['dist'] = 0

            for i, x in enumerate(df_overlap['symbol_g'].unique()):
                # peaks within the extended TSS
                df_overlap_x = \
                    df_overlap[df_overlap['symbol_g'] == x].copy()
                # peaks within the gene body
                df_overlap2_x = \
                    df_overlap2[df_overlap2['symbol_g'] == x].copy()
                # peaks that are not intersecting with the promoter
                # and gene body of gene x
                id_overlap = df_overlap_x.index[
                    ~np.isin(df_overlap_x['id_p'], df_overlap2_x['id_p'])]
                mask_x = (df_gene_ann['symbol'] == x)
                range_x = df_gene_ann[mask_x][['start', 'end']].values\
                    .flatten()
                if df_overlap_x['strand_g'].iloc[0] == '+':
                    df_overlap_updated.loc[id_overlap, 'dist'] = pd.concat(
                        [abs(df_overlap_x.loc[id_overlap, 'start_p']
                             - (range_x[1])),
                         abs(df_overlap_x.loc[id_overlap, 'end_p']
                             - max(0, range_x[0]-self.gb_upstream))],
                        axis=1, sort=False).min(axis=1)
                else:
                    df_overlap_updated.loc[id_overlap, 'dist'] = pd.concat(
                        [abs(df_overlap_x.loc[id_overlap, 'start_p']
                             - (range_x[1]+self.gb_upstream)),
                         abs(df_overlap_x.loc[id_overlap, 'end_p']
                             - (range_x[0]))],
                        axis=1, sort=False).min(axis=1)

                n_batch = int(df_gene_ann_for_pbt.shape[0]/5)
                if i % n_batch == 0:
                    print(f'Processing: {i/df_gene_ann_for_pbt.shape[0]:.1%}')
            df_overlap_updated['dist'] = df_overlap_updated['dist']\
                .astype(float)

            adata.uns['gene_scores'] = dict()
            adata.uns['gene_scores']['overlap'] = df_overlap_updated.copy()

        df_overlap_updated['weight'] = np.exp(
            -(df_overlap_updated['dist'].values/self.gb_upstream))
        mask_w = (df_overlap_updated['weight'] < self.cutoff_weight)
        df_overlap_updated.loc[mask_w, 'weight'] = 0
        # construct genes-by-peaks matrix
        mat_GP = csr_matrix(coo_matrix((df_overlap_updated['weight'],
                                       (df_overlap_updated['id_g'],
                                        df_overlap_updated['id_p'])),
                                       shape=(df_gene_ann.shape[0],
                                              df_peaks.shape[0])))
        # adata_GP = ad.AnnData(X=csr_matrix(mat_GP),
        #                       obs=df_gene_ann,
        #                       var=df_peaks)
        # adata_GP.layers['weight'] = adata_GP.X.copy()
        if self.use_gene_weight:
            gene_weights = self._weight_genes()
            gene_scores = adata[:, mask_p].X * \
                (mat_GP.T.multiply(gene_weights))
        else:
            gene_scores = adata[:, mask_p].X * mat_GP.T
        adata_CG_atac = ad.AnnData(gene_scores,
                                   obs=adata.obs.copy(),
                                   var=df_gene_ann.copy())
        return adata_CG_atac


def gene_scores(adata,
                genome,
                gene_anno=None,
                tss_upstream=1e5,
                tss_downsteam=1e5,
                gb_upstream=5000,
                cutoff_weight=1,
                use_top_pcs=True,
                use_precomputed=True,
                use_gene_weight=True,
                min_w=1,
                max_w=5,
                local_file_path=None):
    """Calculate gene scores

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    genome : `str`
        Reference genome. Choose from {'hg19', 'hg38', 'mm9', 'mm10'}
    gene_anno : `pandas.DataFrame`, optional (default: None)
        Dataframe of gene annotation.
        If None, built-in gene annotation will be used depending on `genome`;
        If provided, custom gene annotation will be used instead.
    tss_upstream : `int`, optional (default: 1e5)
        The number of base pairs upstream of TSS
    tss_downsteam : `int`, optional (default: 1e5)
        The number of base pairs downstream of TSS
    gb_upstream : `int`, optional (default: 5000)
        The number of base pairs upstream by which gene body is extended.
        Peaks within the extended gene body are given the weight of 1.
    cutoff_weight : `float`, optional (default: 1)
        Weight cutoff for peaks
    use_top_pcs : `bool`, optional (default: True)
        If True, only peaks associated with top PCs will be used
    use_precomputed : `bool`, optional (default: True)
        If True, overlap bewteen peaks and genes
        (stored in `adata.uns['gene_scores']['overlap']`) will be imported
    use_gene_weight : `bool`, optional (default: True)
        If True, for each gene, the number of peaks assigned to it
        will be rescaled based on gene size
    min_w : `int`, optional (default: 1)
        The minimum weight for each gene.
        Only valid if `use_gene_weight` is True
    max_w : `int`, optional (default: 5)
        The maximum weight for each gene.
        Only valid if `use_gene_weight` is True

    Returns
    -------
    adata_new: AnnData
        Annotated data matrix.
        Stores #cells x #genes gene score matrix

    updates `adata` with the following fields.
    overlap: `pandas.DataFrame`, (`adata.uns['gene_scores']['overlap']`)
        Dataframe of overlap between peaks and genes
    """
    GS = GeneScores(adata,
                    genome,
                    gene_anno=gene_anno,
                    tss_upstream=tss_upstream,
                    tss_downsteam=tss_downsteam,
                    gb_upstream=gb_upstream,
                    cutoff_weight=cutoff_weight,
                    use_top_pcs=use_top_pcs,
                    use_precomputed=use_precomputed,
                    use_gene_weight=use_gene_weight,
                    min_w=min_w,
                    max_w=max_w,
                    local_file_path=local_file_path)
    adata_CG_atac = GS.cal_gene_scores()
    return adata_CG_atac



# followed STAGATE
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm=None, random_seed=2020):
    """
    Performs R's Mclust via rpy2.

    Args:
        adata (AnnData): AnnData object with embedding stored in `.obsm`.
        num_cluster (int): Desired number of clusters.
        modelNames (str): Covariance structure model in Mclust.
        used_obsm (str): Key in `.obsm` to use for clustering.
        random_seed (int): Random seed for reproducibility.

    Returns:
        AnnData: Annotated object with added `mclust` cluster label.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    import rpy2.rinterface as ri
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    # rpy2.robjects.numpy2ri.activate() is deprecated
    
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
                                                                           
    if used_obsm is None:
            data_for_clust = adata.X
    else:
        if used_obsm not in adata.obsm:
            raise KeyError(f"{used_obsm} not found in adata.obsm")
        data_for_clust = adata.obsm[used_obsm]
   
    import scipy.sparse as sp
    if sp.issparse(data_for_clust):
        data_for_clust = data_for_clust.toarray()
    
    if data_for_clust.ndim == 1:
        data_for_clust = data_for_clust.reshape(-1, 1)
        
    data_for_clust = np.ascontiguousarray(data_for_clust, dtype=np.float64)

    if data_for_clust.shape[1] > 500:
        print(f"Warning: Clustering on {data_for_clust.shape[1]} features. This might be slow or cause errors. Consider using PCA (used_obsm='X_pca').")
    
    if np.any(np.isnan(data_for_clust)):
        print("Warning: Input data contains NaNs. Replacing with 0.")
        data_for_clust = np.nan_to_num(data_for_clust)

    # Explicitly convert to R matrix to avoid dimension issues
    nr, nc = data_for_clust.shape
    rvec = robjects.FloatVector(data_for_clust.ravel())
    
    # Use localconverter to ensure we get R objects back, not numpy arrays
    # This prevents the "Error in dimnames(x) <- dn" issue caused by numpy2ri being active
    from rpy2.robjects import conversion
    from rpy2.robjects import default_converter
    
    with conversion.localconverter(default_converter):
        r_data = robjects.r.matrix(rvec, nrow=nr, ncol=nc, byrow=True)
        
        # Explicitly set column names to avoid "Error in dimnames(x) <- dn"
        # This is critical when numpy2ri has been active
        col_names = robjects.StrVector([f"V{i}" for i in range(nc)])
        r_data.colnames = col_names
        
        # The key fix is here: explicitly convert Python's None to R's NULL
        # Use kwargs to ensure arguments are passed correctly to R function
        res = rmclust(r_data, G=num_cluster, modelNames=modelNames)
        
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int').astype('category')
    return adata


from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors
import numpy as np

def find_mnn_triplets(
    E_c1: np.ndarray,
    E_c2: np.ndarray,
    k: int = 50,
) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], list]]:
    """
    Finds Mutual Nearest Neighbors (MNNs) between two sets of embeddings
    using Faiss for accelerated search, and then samples random negatives.
    Falls back to scikit-learn if Faiss is not available.

    Parameters
    ----------
    E_c1
        Embeddings for cell batch 1 (shape: [n_cells_1, dim]). Used for anchors and negatives.
    E_c2
        Embeddings for cell batch 2 (shape: [n_cells_2, dim]). Used for positives.
    k
        Number of neighbors to consider for MNN search.

    Returns
    -------
    A tuple containing:
      - A tuple of three numpy arrays: (anchors, positives, negatives).
      - A list of the raw (c1_idx, c2_idx) MNN pairs.
    Indices are local to the input arrays E_c1 and E_c2.
    Returns None if no MNNs are found or an error occurs.
    """
    if E_c1.shape[0] == 0 or E_c2.shape[0] == 0:
        print(f"Empty input arrays for MNN search: E_c1={E_c1.shape}, E_c2={E_c2.shape}")
        return None

    try:
        try:
            import faiss
            print("Using Faiss for accelerated k-NN search.")
            
            # Ensure data is float32 and contiguous for Faiss
            E_c1_faiss = np.ascontiguousarray(E_c1, dtype='float32')
            E_c2_faiss = np.ascontiguousarray(E_c2, dtype='float32')
            dim = E_c1_faiss.shape[1]

            # Find k-nearest neighbors of C2 embeddings in C1
            actual_k_c1 = min(k, E_c1.shape[0])
            index_c1 = faiss.IndexFlatL2(dim)
            index_c1.add(E_c1_faiss)
            _, indices_c2_in_c1 = index_c1.search(E_c2_faiss, actual_k_c1)

            # Find k-nearest neighbors of C1 embeddings in C2
            actual_k_c2 = min(k, E_c2.shape[0])
            index_c2 = faiss.IndexFlatL2(dim)
            index_c2.add(E_c2_faiss)
            _, indices_c1_in_c2 = index_c2.search(E_c1_faiss, actual_k_c2)
            
        except ImportError:
            print("Faiss not found. Falling back to scikit-learn for k-NN search.")
            # Find k-nearest neighbors of C2 embeddings in C1
            actual_k_c1 = min(k, E_c1.shape[0])
            nbrs_c1 = NearestNeighbors(n_neighbors=actual_k_c1, algorithm='auto').fit(E_c1)
            _, indices_c2_in_c1 = nbrs_c1.kneighbors(E_c2)

            # Find k-nearest neighbors of C1 embeddings in C2
            actual_k_c2 = min(k, E_c2.shape[0])
            nbrs_c2 = NearestNeighbors(n_neighbors=actual_k_c2, algorithm='auto').fit(E_c2)
            _, indices_c1_in_c2 = nbrs_c2.kneighbors(E_c1)

        print(f"Computed k-NN with actual k values: C1={actual_k_c1}, C2={actual_k_c2}")

        # Find mutual nearest neighbors
        mnn_pairs = []
        for i in range(E_c1.shape[0]):
            for j in indices_c1_in_c2[i]:
                if i in indices_c2_in_c1[j]:
                    # Always return (index_in_c1, index_in_c2)
                    mnn_pairs.append((i, j))
        
        if not mnn_pairs:
            return None

        return mnn_pairs

    except Exception as e:
        import traceback
        print(f"An error occurred during MNN search: {e}")
        traceback.print_exc()
        return None
