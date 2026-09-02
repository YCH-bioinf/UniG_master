from pathlib import Path
from collections import Counter
from collections.abc import Mapping, MutableMapping
import glob
import math
import multiprocessing
import os
import pickle
import random
import re
import shutil

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

if "object" not in np.__dict__:
    np.object = object
if "float" not in np.__dict__:
    np.float = float

from arboreto.algo import grnboost2
from arboreto.utils import load_tf_names
from ctxcore.genesig import GeneSignature
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.aucell import aucell
from pyscenic.binarization import binarize, derive_threshold
from pyscenic.prune import prune2df, df2regulons
from pyscenic.utils import modules_from_adjacencies

np.seterr(divide="ignore", invalid="ignore")


def install_numpy_compat_aliases():
    """Add NumPy aliases needed by older dependencies."""
    if "object" not in np.__dict__:
        np.object = object
    if "float" not in np.__dict__:
        np.float = float


def register_anndata_null_reader():
    """Register a reader for legacy AnnData null datasets."""
    import h5py
    from anndata._io.specs import IOSpec, _REGISTRY

    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def read_null(elem, *, _reader):
            return None
    except Exception as exc:
        message = str(exc).lower()
        if "already" not in message and "duplicate" not in message:
            raise
        read_null = None

    return read_null


_H5AD_KEY_MAP_UNS = "_unig_h5ad_key_map"


def _encode_h5ad_key(key):
    """Encode AnnData uns keys so they can be written to h5ad."""
    key = str(key)
    return key.replace("%", "%25").replace("/", "%2F")


def _decode_h5ad_key(key):
    """Decode keys produced by _encode_h5ad_key."""
    key = str(key)
    return key.replace("%2F", "/").replace("%25", "%")


def sanitize_uns_keys_for_h5ad(adata, copy=False):
    """Return an AnnData object whose uns mapping keys are safe for h5ad writes.

    AnnData writes nested ``uns`` dictionaries as HDF5 groups, where forward
    slashes are interpreted as path separators and are therefore invalid in
    keys. UniG CCC LR-pair names can contain slashes, so they are encoded before
    saving and recorded in ``adata.uns['_unig_h5ad_key_map']``. The accessor
    helpers in this module resolve either the original or encoded names.
    """
    if copy:
        adata = adata.copy()

    key_map = {}
    _sanitize_mapping_for_h5ad(adata.uns, path=(), key_map=key_map)
    if key_map:
        existing = adata.uns.get(_H5AD_KEY_MAP_UNS, {})
        if isinstance(existing, Mapping):
            merged = {str(k): dict(v) for k, v in existing.items()}
            for group, mapping in key_map.items():
                merged.setdefault(group, {}).update(mapping)
            key_map = merged
        adata.uns[_H5AD_KEY_MAP_UNS] = key_map
    return adata


def _sanitize_mapping_for_h5ad(mapping, path, key_map):
    if not isinstance(mapping, MutableMapping):
        return

    for key in list(mapping.keys()):
        value = mapping[key]
        if key == _H5AD_KEY_MAP_UNS:
            continue

        safe_key = _encode_h5ad_key(key)
        if safe_key != str(key):
            safe_key = _deduplicate_mapping_key(mapping, key, safe_key)
            mapping[safe_key] = mapping.pop(key)
            group = "||".join(path) if path else "__uns__"
            key_map.setdefault(group, {})[safe_key] = str(key)
            key = safe_key
            value = mapping[key]

        if isinstance(value, MutableMapping):
            _sanitize_mapping_for_h5ad(value, path + (str(key),), key_map)


def _deduplicate_mapping_key(mapping, old_key, safe_key):
    if safe_key == old_key or safe_key not in mapping:
        return safe_key

    idx = 1
    candidate = f"{safe_key}%{idx}"
    while candidate in mapping:
        idx += 1
        candidate = f"{safe_key}%{idx}"
    return candidate


def _uns_feature_items(adata, weight_key):
    group = adata.uns[weight_key]
    original_names = _uns_original_key_map(adata, weight_key)
    for key, value in group.items():
        feature = original_names.get(str(key), _decode_h5ad_key(key))
        yield feature, value


def _resolve_uns_feature_key(adata, weight_key, feature):
    group = adata.uns[weight_key]
    if feature in group:
        return feature

    feature_str = str(feature)
    encoded = _encode_h5ad_key(feature_str)
    if encoded in group:
        return encoded

    original_names = _uns_original_key_map(adata, weight_key)
    for stored_key, original in original_names.items():
        if str(original) == feature_str and stored_key in group:
            return stored_key

    decoded_matches = [key for key in group.keys() if _decode_h5ad_key(key) == feature_str]
    if decoded_matches:
        return decoded_matches[0]

    candidates = [original for original in original_names.values() if feature_str in str(original)]
    if not candidates:
        candidates = [key for key in group.keys() if feature_str in _decode_h5ad_key(key)]
    raise KeyError(f"{feature!r} not found in {weight_key}. Candidates: {candidates[:10]}")


def _uns_original_key_map(adata, weight_key):
    key_map = adata.uns.get(_H5AD_KEY_MAP_UNS, {})
    if not isinstance(key_map, Mapping):
        return {}
    mapping = key_map.get(str(weight_key), {})
    if not isinstance(mapping, Mapping):
        return {}
    return {str(stored_key): str(original) for stored_key, original in mapping.items()}




# Integrated UniG CCC core implementation.
def detect_outliers_z_score(data, 
                            threshold=3):
    """
    使用Z-score方法检测outlier
    """
    z_scores = (data - np.mean(data)) / np.std(data)
    return np.where(np.abs(z_scores) > threshold)[0]

def find_radius(adata,  
                scope=6):
    ## 计算每个点周围是scope个邻居的时候，半径是多少
    n = adata.shape[0]
    coords = adata.obsm['spatial']
    distances = cdist(coords, coords)
    nearby_dis = []
    for i in range(n):
        sorted_indices = np.argsort(distances[i])
        nearby_indices = sorted_indices[1:scope + 1]  # 第0个是它自己，不需要计算
        nearby_dis.append([distances[i][a] for a in nearby_indices])
    new_nearby_dis = np.concatenate(nearby_dis)
    # 示例
    outliers = detect_outliers_z_score(new_nearby_dis)
    new_nearby_dis = np.delete(new_nearby_dis, outliers)
    radius = np.max(new_nearby_dis)
    print('The radius is: ' + str(radius))
    adata.uns['radius'] = {'radius':radius}
    
    
def subset_anndata(adata, cell_type_list, key='cell_type'):
    return adata[adata.obs[key].isin(cell_type_list)]

def check_elements(list1, 
                   list2, 
                   number):
    if number == 'all':
        ## 检查list1是否全部存在于list2中，返回True/False
        set2 = set(list2)
        return set2.issuperset(list1)
    else:
        set1 = set(list1)
        set2 = set(list2)
        return len(set1.intersection(set2)) >= number


def grep_exist_LR(adata,
                  DB_interaction,
                  DB_complex,
                  if_hvg=True,
                  n_top_genes=None,
                  min_mean=0.0125,
                  max_mean=3,
                  min_disp=0.5,
                  max_disp=np.inf):
    ## 检查Database中的LR对对应的基因是否都存在于adata的var中，如果一个LR对中有一个基因不存在于adata的var中，则删除
    adata.uns['DB_complex'] = DB_complex
    
    DB_complex_index = DB_complex.index.tolist()
    LR_noexist = []
    if if_hvg == True:
        sc.pp.highly_variable_genes(adata, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp, max_disp=max_disp, n_top_genes=n_top_genes)
        hvggene = list(adata[:, adata.var.highly_variable].var.index)
    allgene = adata.var.index.tolist()
    LR_dict = {}
    
    for i in DB_interaction.index:
        list_tmp = []
        ligand = DB_interaction.loc[i,'ligand']
        receptor = DB_interaction.loc[i,'receptor']
        ligand_annotation = DB_interaction.loc[i,'ligand_annotation']

        if ligand in DB_complex_index:
            sub = DB_complex.loc[ligand].tolist()
            ligands_sub = [i for i in sub if str(i) != 'nan']
        else:
            ligands_sub = ligand
        if receptor in DB_complex_index:
            sub = DB_complex.loc[receptor].tolist()
            receptor_sub = [i for i in sub if str(i) != 'nan']
        else:
            receptor_sub = receptor

        if type(ligands_sub) == str:
            list_tmp.append(ligands_sub)
        else:
            list_tmp = list_tmp + ligands_sub

        if type(receptor_sub) == str:
            list_tmp.append(receptor_sub)
        else:
            list_tmp = list_tmp + receptor_sub
        
        if 'pathway' in DB_interaction.columns:
            pathway_name = DB_interaction.loc[i,'pathway']
            if check_elements(list_tmp, allgene,'all'):
                if if_hvg == True:
                    if check_elements(list_tmp, hvggene, 1):
                        LR_dict[i] = {'ligand':ligand, 'receptor':receptor,'l_annotation':ligand_annotation,'pathway':pathway_name}
                else:
                    LR_dict[i] = {'ligand':ligand, 'receptor':receptor,'l_annotation':ligand_annotation,'pathway':pathway_name}
                    
        else:
            if check_elements(list_tmp, allgene,'all'):
                if if_hvg == True:
                    if check_elements(list_tmp, hvggene, 1):
                        LR_dict[i] = {'ligand':ligand, 'receptor':receptor,'l_annotation':ligand_annotation}
                else:
                    LR_dict[i] = {'ligand':ligand, 'receptor':receptor,'l_annotation':ligand_annotation}
            
    adata.uns['LR_pair_information'] = LR_dict
    
    
    geneLR = DB_interaction.loc[list(LR_dict.keys())]['ligand'].tolist() + DB_interaction.loc[list(LR_dict.keys())]['receptor'].tolist()
    #geneall = DB_geneinfo['Symbol'].tolist()
    complexset = list(set([lr for lr in geneLR if lr in list(DB_complex.index)]))
    geneset = list(set(geneLR) - set(complexset))
    DB_complex_keep = DB_complex.loc[complexset]
    complex_sub = DB_complex_keep.stack().tolist()
    complex_sub = list(set(complex_sub))
    geneLR = list(set(complex_sub + geneset))
    
    LR_gene_com_keep = {'geneset':geneset,
                       'complexset':complexset,
                       'complex_sub':complex_sub,
                       'geneall':geneLR}
    
    adata.uns['LR_gene_complex_information'] = LR_gene_com_keep
    print('The number of keep LR pair is '+str(len(LR_dict.keys())))
    
def progress_bar(progress, names):
    bar_length = 30
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percentage = int(progress * 100)
    print(f'{names}: |{bar}| {percentage}%', end="\r")

def gmean(expression_levels):
    # 计算四分位数
    q1 = np.percentile(expression_levels, 25)
    q2 = np.percentile(expression_levels, 50)
    q3 = np.percentile(expression_levels, 75)
    # 计算平均值
    average = (1/2) * q2 + (1/4) * (q1 + q3)
    return average

def get_gene_expression(mat,
                        gene_index, ## index 和 gene的对应关系
                        gene,
                        dic,
                        key_name=None):
    ## 输入一个gene，获取其对应的基因表达值
    ## 如果输入的是一个list，则获取几个基因的几何平均表达
    if isinstance(gene, str):
        dic[gene] = mat[:, gene_index[gene]]
    elif isinstance(gene, list):
        gene_indices = np.array([gene_index[g] for g in gene])
        mat_sub = mat[:, gene_indices]
        dic[key_name] = np.apply_along_axis(gmean, axis=1, arr=mat_sub)

        
def get_LR_gene_exp_pool(mat, gene_index, all_complex, geneLR_exp_dict, DB_complex):
    """模拟进程池"""
    while True:
        try:
            com = all_complex.get(False)
            sub = DB_complex.loc[com].tolist()
            receptor_sub = [i for i in sub if str(i) != 'nan']
            get_gene_expression(mat,gene_index, receptor_sub, geneLR_exp_dict, key_name=com, )
            
        except Exception:
            if all_complex.empty():
                break
    
                
def get_LR_gene_exp(adata,
                    threads=40):
    ## 该函数计算了所有受体和配体的基因表达，如果受配体是复合物，则使用几何平均值计算平均表达。
    ## 返回一个dict，里面是受体配体基因或复合物的每个细胞的表达值
    
    geneset = adata.uns['LR_gene_complex_information']['geneset']
    complexset = adata.uns['LR_gene_complex_information']['complexset']
    DB_complex = adata.uns['DB_complex']
    
    try:
        mat = adata.X.toarray()
    except AttributeError:
        mat = adata.X
    gl = adata.var.index
    gene_index = {name: i for i, name in enumerate(gl)}
    geneLR_exp_dict = {}
    for gene in geneset:
        get_gene_expression(mat, gene_index, gene, geneLR_exp_dict)
    p_list = []
    # 先将多进程所要执行的任务的所有参数放入队列中
    all_task = multiprocessing.Queue()
    for com in complexset:
        all_task.put(com)
    # 结果存储
    com_dict = multiprocessing.Manager().dict()
    
    # 启动多进程

    for i in range(threads):
        p = multiprocessing.Process(target=get_LR_gene_exp_pool, args=(mat, gene_index, all_task, com_dict, DB_complex,))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    
    com_dict = dict(com_dict)
    geneLR_exp_dict.update(com_dict)
    
    adata.uns['LR_gene_complex_exp'] = geneLR_exp_dict
    
    
def get_close_gene(adata,
                   background_number=100):
    ## 找出与每个受体配体基因或复合物的平均表达值最接近的100个close基因，然后计算这些基因的表达值
    ## 返回两个dict，一个是每个受体配体基因或复合物最接近的100个close基因，第二个是这些close基因的平均表达
    geneLR_exp_dict = adata.uns['LR_gene_complex_exp']
    
    try:
        mat = adata.X.toarray()
    except AttributeError:
        mat = adata.X
    gl = adata.var.index
    gene_index = {name: i for i, name in enumerate(gl)}
    
    
    geneLR = adata.uns['LR_gene_complex_information']['geneall']
    
    all_genes = [item for item in adata.var_names.tolist()
                 if not (item.startswith("MT-") or item.startswith("MT_"))]
    geneLR = [item for item in geneLR
                 if not (item.startswith("MT-") or item.startswith("MT_"))]
    means = adata.to_df()[all_genes].mean().sort_values()

    geneLR_close_means_gene = {}
    for key in geneLR_exp_dict.keys():
        means_exp = np.mean(geneLR_exp_dict[key])
        selected_lf = (abs(means - means_exp).sort_values().drop(geneLR)[:background_number].index.tolist())
        random.shuffle(selected_lf)
        geneLR_close_means_gene[key] = selected_lf

    close_gene_list = []
    for key in geneLR_close_means_gene.keys():
        close_gene_list = close_gene_list + geneLR_close_means_gene[key]
    close_gene_list = list(set(close_gene_list))

    close_gene_exp_dict = {}
    for gene in close_gene_list:
        get_gene_expression(mat,gene_index, gene, close_gene_exp_dict)
        
    adata.uns['LR_close_gene'] = geneLR_close_means_gene
    adata.uns['LR_close_gene_exp'] = close_gene_exp_dict
    #del(adata.uns['LR_gene_complex_information'])
    
def get_true_weight_matirx(adata,
                           method, ## 计算lr分数的方法
                           scope=6, ## 非分泌性受配体扩散范围到其周围的几个点
                           min_exp=0.01):
    
    n = adata.shape[0]
    coords = adata.obsm['spatial']
    distances = cdist(coords, coords)
    adata.obsm['distances'] = distances
    radius = adata.uns['radius']['radius']
    
    LR_weight_dict = {}
    #LR_neighbors_dict = {}
    LR_dict = adata.uns['LR_pair_information']
    
    geneLR_exp_dict = adata.uns['LR_gene_complex_exp']
    #del(adata.uns['LR_gene_complex_exp'])
    
    for key in LR_dict.keys():
        LR_weight_dict[key] = np.zeros((n, n),dtype='float32')
        #LR_neighbors_dict[key] = np.zeros((n, n))

    LR_secreted_dict = {}
    LR_unsecreted_dict = {}
    for key in LR_dict.keys():
        if LR_dict[key]['l_annotation'] == 'Secreted':
            LR_secreted_dict[key] = LR_dict[key]
        else:
            LR_unsecreted_dict[key] = LR_dict[key]

    Wr = (3*radius/2)
    print('Now processing the unsecreted')
    for i in range(n):
        neighbors = np.where(distances[i] < radius)[0]
        #neighbors = np.delete(neighbors, np.where(neighbors == i))
        if len(neighbors) == 0:
            continue
        distance_factor = np.exp(-(distances[i, neighbors] / Wr)**2)  # 距离衰减因子，使用指数函数
        for key in LR_unsecreted_dict.keys():
            ligand_name = LR_unsecreted_dict[key]['ligand']
            receptor_name = LR_unsecreted_dict[key]['receptor']
            ligand = geneLR_exp_dict[ligand_name]
            receptor = geneLR_exp_dict[receptor_name]

            score_l = ligand[i] * distance_factor
            indx = np.where(score_l < min_exp)[0]
            neighbors_nonsig = neighbors[indx]
            neighbors_sig = np.delete(neighbors, indx)
            LR_weight_dict[key][i, neighbors_nonsig] = 0
            distance_factor_sig = np.delete(distance_factor, indx)
            if method == 'Hill':
                scores_tmp = ligand[i] * receptor[neighbors_sig] * distance_factor_sig
                scores = scores_tmp / (0.5+scores_tmp)
            if method == 'normal':
                scores = ligand[i] * receptor[neighbors_sig] * distance_factor_sig
            if method =='square_root':
                scores_tmp = ligand[i] * receptor[neighbors_sig] * distance_factor_sig
                scores = np.sqrt(scores_tmp)
            
            non_zero_positions = np.where(scores != 0)
            neighbors_sig_nozero = neighbors_sig[non_zero_positions]
            LR_weight_dict[key][i, neighbors_sig] = scores
            #LR_neighbors_dict[key][i, neighbors_sig_nozero] = 1
        progress_bar((i+1)/n, 'Computed weight matrix process')
    
    for key in LR_unsecreted_dict.keys():
        LR_weight_dict[key] = csr_matrix(LR_weight_dict[key])
        #LR_neighbors_dict[key] = csr_matrix(LR_neighbors_dict[key])

    print('\nNow processing the secreted')
    num = 1 
    for key in LR_secreted_dict.keys():
        ligand_name = LR_dict[key]['ligand']
        receptor_name = LR_dict[key]['receptor']
        ligand = geneLR_exp_dict[ligand_name]
        receptor = geneLR_exp_dict[receptor_name]
        distance_radius = np.where(ligand == 0, 0, np.sqrt(-np.log(min_exp/ligand)) * Wr)
        for i in range(n):
            neighbors = np.where(distances[i] < distance_radius[i])[0]
            #neighbors = np.delete(neighbors, np.where(neighbors == i))
            if len(neighbors) == 0:
                continue
            distance_factor = np.exp(-(distances[i, neighbors] / Wr)**2)  # 距离衰减因子，使用指数函数
            
            if method == 'Hill':
                scores_tmp = ligand[i] * receptor[neighbors] * distance_factor
                scores = scores_tmp / (0.5+scores_tmp)
            if method == 'normal':
                scores = ligand[i] * receptor[neighbors] * distance_factor
            if method =='square_root':
                scores_tmp = ligand[i] * receptor[neighbors] * distance_factor
                scores = np.sqrt(scores_tmp)
                
            non_zero_positions = np.where(scores != 0)
            neighbors_nozero = neighbors[non_zero_positions]
            LR_weight_dict[key][i, neighbors] = scores
            #LR_neighbors_dict[key][i, neighbors_nozero] = 1
            
        LR_weight_dict[key] = csr_matrix(LR_weight_dict[key])
        #LR_neighbors_dict[key] = csr_matrix(LR_neighbors_dict[key])
        progress_bar(num/len(LR_secreted_dict.keys()), 'Computed weight matrix process')
        num = num + 1
    
    adata.uns['LR_cell_weight'] = LR_weight_dict
    #adata.uns['Cell_neighbors'] = LR_neighbors_dict
    
    
def find_key_position(dictionary, key):
    keys = list(dictionary.keys())
    if key in keys:
        position = keys.index(key)
        return position
    else:
        return None


_PERMUTATION_CONTEXT = None


def _transform_lr_score(raw_score, method):
    if method == 'Hill':
        return raw_score / (0.5 + raw_score)
    if method == 'normal':
        return raw_score
    if method == 'square_root':
        return np.sqrt(raw_score)
    raise ValueError(
        f"Unsupported LR score method '{method}'. "
        "Choose from 'Hill', 'normal', or 'square_root'."
    )


def _calculate_fake_weight_sparse(LR_name, context):
    radius = context['radius']
    Wr = 3 * radius / 2
    distances = context['distances']
    geneLR_close_means_gene = context['geneLR_close_means_gene']
    close_gene_exp_dict = context['close_gene_exp_dict']
    LR_inf_dict = context['LR_inf_dict']
    true_weight = context['LR_true_weight_dict'][LR_name].tocoo()
    method = context['method']
    min_exp = context['min_exp']
    cutoff = context['cutoff']
    batch_size = context['batch_size']

    matrix_shape = true_weight.shape
    if true_weight.nnz == 0:
        empty = csr_matrix(matrix_shape, dtype='float32')
        return LR_name, empty, empty.copy(), 0

    sender_idx = true_weight.row
    receiver_idx = true_weight.col
    true_scores = true_weight.data
    edge_distances = distances[sender_idx, receiver_idx]
    distance_factor = np.exp(-(edge_distances / Wr) ** 2)

    ligand_key = LR_inf_dict[LR_name]['ligand']
    receptor_key = LR_inf_dict[LR_name]['receptor']
    ligand_fake_names = geneLR_close_means_gene[ligand_key]
    receptor_fake_names = geneLR_close_means_gene[receptor_key]

    available_backgrounds = min(
        len(ligand_fake_names),
        len(receptor_fake_names),
    )
    is_secreted = LR_inf_dict[LR_name]['l_annotation'] == 'Secreted'
    requested_backgrounds = available_backgrounds if is_secreted else 100
    n_permutations = min(requested_backgrounds, available_backgrounds)

    if n_permutations == 0:
        raise ValueError(f"No matched background genes are available for LR '{LR_name}'.")

    count_less = np.zeros(true_weight.nnz, dtype=np.uint32)

    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        ligand_batch = np.asarray([
            close_gene_exp_dict[name][sender_idx]
            for name in ligand_fake_names[start:stop]
        ])
        receptor_batch = np.asarray([
            close_gene_exp_dict[name][receiver_idx]
            for name in receptor_fake_names[start:stop]
        ])

        if is_secreted:
            with np.errstate(divide='ignore', invalid='ignore'):
                diffusion_radius = np.where(
                    ligand_batch == 0,
                    0,
                    np.sqrt(-np.log(min_exp / ligand_batch)) * Wr,
                )
            valid_edges = edge_distances[np.newaxis, :] < diffusion_radius
        else:
            valid_edges = (
                ligand_batch * distance_factor[np.newaxis, :]
            ) >= min_exp

        raw_scores = (
            ligand_batch
            * receptor_batch
            * distance_factor[np.newaxis, :]
        )
        fake_scores = np.where(
            valid_edges,
            _transform_lr_score(raw_scores, method),
            0,
        )
        count_less += np.sum(
            fake_scores < true_scores[np.newaxis, :],
            axis=0,
            dtype=np.uint32,
        )

    edge_pvalues = 1 - count_less.astype(np.float64) / n_permutations
    keep = edge_pvalues < cutoff

    filtered_weight = csr_matrix(
        (
            true_scores[keep],
            (sender_idx[keep], receiver_idx[keep]),
        ),
        shape=matrix_shape,
        dtype='float32',
    )
    neighbors = csr_matrix(
        (
            np.ones(np.count_nonzero(keep), dtype='float32'),
            (sender_idx[keep], receiver_idx[keep]),
        ),
        shape=matrix_shape,
        dtype='float32',
    )
    return LR_name, filtered_weight, neighbors, n_permutations


def _permutation_worker(LR_name):
    return _calculate_fake_weight_sparse(LR_name, _PERMUTATION_CONTEXT)


def permutation_test(adata,
                     method,
                     threads=40,
                     min_exp=0.01,
                     cutoff=0.05,
                     batch_size=16):
    global _PERMUTATION_CONTEXT

    if not 0 <= cutoff <= 1:
        raise ValueError("'cutoff' must be between 0 and 1.")
    if batch_size < 1:
        raise ValueError("'batch_size' must be at least 1.")

    LR_names = list(adata.uns['LR_pair_information'].keys())
    _PERMUTATION_CONTEXT = {
        'radius': adata.uns['radius']['radius'],
        'distances': adata.obsm['distances'],
        'geneLR_close_means_gene': adata.uns['LR_close_gene'],
        'close_gene_exp_dict': adata.uns['LR_close_gene_exp'],
        'LR_inf_dict': adata.uns['LR_pair_information'],
        'LR_true_weight_dict': adata.uns['LR_cell_weight'],
        'method': method,
        'min_exp': min_exp,
        'cutoff': cutoff,
        'batch_size': batch_size,
    }

    filtered_weights = {}
    LR_neighbors_dict = {}
    permutation_counts = {}
    worker_count = max(1, min(int(threads), len(LR_names)))

    if worker_count == 1:
        results = map(_permutation_worker, LR_names)
        pool = None
    else:
        mp_context = multiprocessing.get_context('fork')
        pool = mp_context.Pool(processes=worker_count)
        results = pool.imap_unordered(_permutation_worker, LR_names)

    try:
        for num, (LR_name, weight, neighbors, n_permutations) in enumerate(
            results,
            start=1,
        ):
            filtered_weights[LR_name] = weight
            LR_neighbors_dict[LR_name] = neighbors
            permutation_counts[LR_name] = n_permutations
            progress_bar(
                num / len(LR_names),
                'Permutation test and filtering process',
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _PERMUTATION_CONTEXT = None

    adata.uns['LR_cell_weight'] = {
        LR_name: filtered_weights[LR_name] for LR_name in LR_names
    }
    adata.uns['Cell_neighbors'] = {
        LR_name: LR_neighbors_dict[LR_name] for LR_name in LR_names
    }
    adata.uns['LR_permutation_count'] = {
        LR_name: permutation_counts[LR_name] for LR_name in LR_names
    }
        
        
def run_scenic(adata, DB_interaction, DATABASES_GLOB, MOTIF_ANNOTATIONS_FNAME):
    exp_matrix = pd.DataFrame(adata.X.toarray(), columns=adata.var.index, index=adata.obs.index)
    tf_names = []
    for i in range(len(list(DB_interaction['TF']))):
        if type(list(DB_interaction['TF'])[i]) is not float:
            tf_names = tf_names + list(DB_interaction['TF'])[i].split(', ') 
    tf_names = list(set(tf_names))
    db_fnames = glob.glob(DATABASES_GLOB)
    def name(fname):
        return os.path.splitext(os.path.basename(fname))[0]
    dbs = [RankingDatabase(fname=fname, name=name(fname)) for fname in db_fnames]
    # Run grnboost2
    print('Phase I: Inference of co-expression modules')
    adjacencies = grnboost2(exp_matrix, tf_names=tf_names, verbose=True)
    # Create modules from a dataframe containing weighted adjacencies between a TF and its target genes.
    modules = list(modules_from_adjacencies(adjacencies, exp_matrix, min_genes=10))
    print('Phase II: Prune modules for targets with cis regulatory footprints')
    # Calculate a list of enriched motifs and the corresponding target genes for all modules.
    # ``dask_multiprocessing`` may launch fresh Python interpreters, which do
    # not see the NumPy compatibility alias above.  pySCENIC's own custom
    # multiprocessing backend forks these workers and is intended for local
    # ranking databases; its enrichment results are identical.
    df = prune2df(
        dbs,
        modules,
        MOTIF_ANNOTATIONS_FNAME,
        num_workers=20,
        client_or_address="custom_multiprocessing",
    )
    regulons = df2regulons(df)
    print('Phase III: Cellular regulon enrichment matrix')
    auc_mtx = aucell(exp_matrix, regulons, num_workers=40)
    ## binarize
    thrs = []
    num = 1
    for regulon in auc_mtx.columns:
        thrs.append(derive_threshold(auc_mtx, regulon))
        progress_bar(num/len(auc_mtx.columns), 'Binarize')
        num = num + 1
    thresholds = pd.Series(index=auc_mtx.columns, data=thrs)
    auc_bin_mtx = (auc_mtx > thresholds).astype(int)
    # save result
    scenic_res_dict = {}
    scenic_res_dict['auc_mtx'] = auc_mtx
    scenic_res_dict['auc_bin_mtx'] = auc_bin_mtx
    scenic_res_dict['auc_thresholds'] = thresholds
    adata.uns['scenic_res'] = scenic_res_dict
    adata.uns['scenic_res']['auc_thresholds'] = dict(adata.uns['scenic_res']['auc_thresholds'])


def add_intracellular_signals(adata, DB_interaction, DATABASES_GLOB, MOTIF_ANNOTATIONS_FNAME, if_stringent):
	print('Run SCENIC')
	run_scenic(adata, DB_interaction, DATABASES_GLOB, MOTIF_ANNOTATIONS_FNAME)
	auc_mtx = adata.uns['scenic_res']['auc_mtx']
	auc_mtx = auc_mtx.loc[adata.obs.index]
	auc_bin_mtx = adata.uns['scenic_res']['auc_bin_mtx']
	thresholds = adata.uns['scenic_res']['auc_thresholds']
	auc_mtx[auc_mtx < thresholds] = 0
	regulons_list = [r.split('(')[0] for r in auc_mtx.columns]
	regulons_dict = {}
	for lr in DB_interaction.index:
		if type(DB_interaction.loc[lr,'TF']) is not float:
			if len(list(set(DB_interaction.loc[lr,'TF'].split(', ')) & set(regulons_list))) > 0:
				regulons_dict[lr] = list(set(DB_interaction.loc[lr,'TF'].split(', ')) & set(regulons_list))
	if if_stringent:
		num = 1
		for lr in adata.uns['LR_cell_weight'].keys():
			if lr in regulons_dict.keys():
				lr_weight = adata.uns['LR_cell_weight'][lr].toarray()
				auc_mtx_lr = auc_mtx[[r+'(+)' for r in regulons_dict[lr]]]
				row_sums = auc_mtx_lr.sum(axis=1)
				transformed_sums = row_sums.apply(lambda x: x / (x + 0.5))
				transformed_sums = np.expand_dims(transformed_sums, axis=1)
				multiplier_str = np.where(transformed_sums>0, transformed_sums+1, transformed_sums)
				lr_weight_str = lr_weight * (multiplier_str.T)
				adata.uns['LR_cell_weight'][lr] = csr_matrix(lr_weight_str)
			else:
				lr_weight = adata.uns['LR_cell_weight'][lr].toarray()
				adata.uns['LR_cell_weight'][lr] = csr_matrix(np.zeros(lr_weight.shape))
			progress_bar(num/len(adata.uns['LR_cell_weight'].keys()), 'Add intracellular signals')
			num = num + 1
	else:
		num = 1
		for lr in adata.uns['LR_cell_weight'].keys():
			if lr in regulons_dict.keys():
				lr_weight = adata.uns['LR_cell_weight'][lr].toarray()
				auc_mtx_lr = auc_mtx[[r+'(+)' for r in regulons_dict[lr]]]
				row_sums = auc_mtx_lr.sum(axis=1)
				transformed_sums = row_sums.apply(lambda x: x / (x + 0.5))
				transformed_sums = np.expand_dims(transformed_sums, axis=1)
				multiplier_nonstr = transformed_sums + 1
				lr_weight_nonstr = lr_weight * (multiplier_nonstr.T)
				adata.uns['LR_cell_weight'][lr] = csr_matrix(lr_weight_nonstr)
			progress_bar(num/len(adata.uns['LR_cell_weight'].keys()), 'Add intracellular signals')
			num = num + 1
	adata.uns['scenic_res']['regulons_dict'] = regulons_dict


def _tf_name_candidates(name):
    cleaned = re.sub(r'^\s*M_', '', str(name))
    cleaned = re.sub(r'\(var\.\d+\)$', '', cleaned)
    cleaned = re.sub(r'\(\+\)$', '', cleaned)
    cleaned = cleaned.strip()
    candidates = [cleaned]
    if '_' in cleaned:
        candidates.extend([part for part in cleaned.split('_') if part])
    return list(dict.fromkeys(candidates))


def _normalize_tf_activity_names(tf_activity):
    renamed = {}
    for col in tf_activity.columns:
        for candidate in _tf_name_candidates(col):
            renamed.setdefault(candidate, []).append(col)

    normalized_series = {}
    for tf, columns in renamed.items():
        if len(columns) == 1:
            normalized_series[tf] = tf_activity[columns[0]]
        else:
            normalized_series[tf] = tf_activity[columns].max(axis=1)
    return pd.DataFrame(normalized_series, index=tf_activity.index)


def _global_minmax(df):
    min_value = df.values.min()
    max_value = df.values.max()
    if max_value == min_value:
        return pd.DataFrame(0, index=df.index, columns=df.columns, dtype='float32')
    return ((df - min_value) / (max_value - min_value)).astype('float32')


def _build_unig_regulons(adata, unig_network_path, min_regulon_targets):
    network = pd.read_csv(unig_network_path)
    required_columns = {'source', 'target'}
    missing_columns = required_columns - set(network.columns)
    if missing_columns:
        raise ValueError(
            f"UniG network is missing required columns: {sorted(missing_columns)}"
        )

    adata_genes = set(adata.var_names)
    tf_to_targets = {}
    for tf, sub in network.groupby('source'):
        targets = sorted(set(sub['target'].dropna().astype(str)) & adata_genes)
        if len(targets) >= min_regulon_targets:
            tf_to_targets[str(tf)] = targets

    if len(tf_to_targets) == 0:
        raise ValueError(
            "No UniG regulons passed the target overlap filter. "
            "Check the network target gene names and adata.var_names."
        )

    signatures = [
        GeneSignature(f"{tf}(+)", targets)
        for tf, targets in tf_to_targets.items()
    ]
    return tf_to_targets, signatures


def _read_unig_tf_activity(adata, unig_tf_activity_path):
    tf_activity_adata = sc.read_h5ad(unig_tf_activity_path)
    missing_cells = adata.obs_names.difference(tf_activity_adata.obs_names)
    if len(missing_cells) > 0:
        raise ValueError(
            "UniG TF activity matrix does not cover all adata.obs_names. "
            f"Missing {len(missing_cells)} cells; first examples: "
            f"{missing_cells[:5].tolist()}"
        )

    tf_activity_adata = tf_activity_adata[adata.obs_names, :].copy()
    try:
        tf_activity_matrix = tf_activity_adata.X.toarray()
    except AttributeError:
        tf_activity_matrix = tf_activity_adata.X

    tf_activity = pd.DataFrame(
        tf_activity_matrix,
        index=tf_activity_adata.obs_names,
        columns=tf_activity_adata.var_names,
    )
    tf_activity = _normalize_tf_activity_names(tf_activity)
    return _global_minmax(tf_activity)


def add_unig_intracellular_signals(adata,
                                   DB_interaction,
                                   unig_network_path,
                                   unig_tf_activity_path,
                                   min_regulon_targets=10,
                                   if_stringent=False):
    if unig_network_path is None or unig_tf_activity_path is None:
        raise ValueError(
            "'unig_network_path' and 'unig_tf_activity_path' must be specified "
            "when intra_method='unig'."
        )

    print('Run UniG GRN intracellular signals')
    tf_to_targets, signatures = _build_unig_regulons(
        adata,
        unig_network_path,
        min_regulon_targets,
    )

    try:
        exp_matrix = pd.DataFrame(
            adata.X.toarray(),
            columns=adata.var.index,
            index=adata.obs.index,
        )
    except AttributeError:
        exp_matrix = pd.DataFrame(
            adata.X,
            columns=adata.var.index,
            index=adata.obs.index,
        )

    auc_mtx = aucell(exp_matrix, signatures, num_workers=32)
    auc_mtx = auc_mtx.loc[adata.obs.index]
    auc_mtx.columns = [col.split('(')[0] for col in auc_mtx.columns]
    auc_mtx_clean = auc_mtx.T.groupby(level=0).max().T
    auc_mtx_output = auc_mtx_clean.rename(
        columns={tf: f"{tf}(+)" for tf in auc_mtx_clean.columns}
    )

    tf_activity_norm = _read_unig_tf_activity(adata, unig_tf_activity_path)
    tf_activity_norm = tf_activity_norm.T.groupby(level=0).max().T
    regulatory_support_clean = auc_mtx_clean.copy()
    regulatory_support_output = regulatory_support_clean.rename(
        columns={tf: f"{tf}(+)" for tf in regulatory_support_clean.columns}
    )

    support_tfs = set(regulatory_support_clean.columns)
    regulons_dict = {}
    for lr in DB_interaction.index:
        if 'TF' in DB_interaction.columns and pd.notna(DB_interaction.loc[lr, 'TF']):
            lr_tfs = [tf.strip() for tf in DB_interaction.loc[lr, 'TF'].split(', ')]
            matched_tfs = sorted(set(lr_tfs) & support_tfs)
            if len(matched_tfs) > 0:
                regulons_dict[lr] = matched_tfs

    lr_names = list(adata.uns['LR_cell_weight'].keys())
    for num, lr in enumerate(lr_names, start=1):
        if lr in regulons_dict:
            lr_weight = adata.uns['LR_cell_weight'][lr].toarray()
            row_sums = regulatory_support_clean[regulons_dict[lr]].sum(axis=1)
            transformed_sums = row_sums.apply(lambda x: x / (x + 0.5))
            transformed_sums = np.expand_dims(transformed_sums, axis=1)
            multiplier = transformed_sums + 1
            adata.uns['LR_cell_weight'][lr] = csr_matrix(lr_weight * multiplier.T)
        elif if_stringent:
            lr_weight = adata.uns['LR_cell_weight'][lr].toarray()
            adata.uns['LR_cell_weight'][lr] = csr_matrix(np.zeros(lr_weight.shape))
        progress_bar(num/len(lr_names), 'Add UniG intracellular signals')

    adata.uns['GRN_res'] = {
        'auc_mtx': auc_mtx_output,
        'tf_activity_global_minmax': tf_activity_norm,
        'regulatory_support_mtx': regulatory_support_output,
        'regulons_dict': regulons_dict,
        'unig_regulon_targets': {
            tf: tf_to_targets[tf]
            for tf in regulatory_support_clean.columns
            if tf in tf_to_targets
        },
    }
    
    
    
def filter_cell(adata, 
                key,
                if_self=True):
    ## 过滤掉没有与其他细胞有相互作用的细胞
    ## 同时也可以过滤自分泌的细胞，如果只关注细胞间相互作用的话
    n = adata.shape[0]
    LR_cell_weight = adata.uns['LR_cell_weight']
    cw = np.zeros((n, n),dtype='float32')
    for key in LR_cell_weight.keys():
        cw = cw + LR_cell_weight[key].toarray()
    
    if if_self == False:
        obs_meta = adata.obs.copy()
        obs_meta.index = range(n)
        unique_values = obs_meta[key].unique()
        for ct in unique_values:
            index_ct = list(obs_meta.index[obs_meta[key]==ct])
            mask = np.zeros_like(cw, dtype=bool)
            # 将指定行和列的布尔值设置为True
            for idx in index_ct:
                mask[index_ct,idx] = True
            # 将布尔数组中对应位置为True的元素设置为0
            cw[mask] = 0
    
    index_keep = []
    for i in range(n):
        score_cell = np.sum(cw[:, i]) + np.sum(cw[i, :])
        if score_cell != 0:
            index_keep.append(i)

    for key in adata.uns['LR_cell_weight']:
        cw = adata.uns['LR_cell_weight'][key].toarray()
        cn = adata.uns['Cell_neighbors'][key].toarray()
        cw_sel = cw[index_keep, :]
        cw_sel = cw_sel[:, index_keep]
        cn_sel = cn[index_keep, :]
        cn_sel = cn_sel[:, index_keep]
        adata.uns['LR_cell_weight'][key] = csr_matrix(cw_sel) 
        adata.uns['Cell_neighbors'][key] = csr_matrix(cn_sel) 

    return adata[index_keep,]

def aggregate_matrix(adata,
                     key):
    LR_dict = adata.uns['LR_pair_information']
    LR_weight_dict = adata.uns['LR_cell_weight']
    LR_neighbors_dict = adata.uns['Cell_neighbors']
    
    ct_interaction_dict = {}
    ct_interaction_per_dict = {}
    ct_interaction_edgenum_dict = {}
    cell_type_list = adata.obs[key].tolist()
    cell_type = list(set(cell_type_list))
    adata.uns[key+'_list'] = cell_type
    ct_n = len(cell_type)
    for key in LR_weight_dict.keys():
        weight_direct_matrix =  LR_weight_dict[key].toarray()
        neighbors_direct_matrix =  LR_neighbors_dict[key].toarray()
        interaction = np.zeros((ct_n, ct_n))
        interaction_per = np.zeros((ct_n, ct_n))
        interaction_edgenum = np.zeros((ct_n, ct_n))
        for l in range(ct_n):
            l_ct = cell_type[l]
            index_l = np.where(np.array(cell_type_list)==l_ct)
            weight_direct_matrix_tmp = weight_direct_matrix[index_l]
            neighbors_direct_matrix_tmp = neighbors_direct_matrix[index_l]
            merged_row = np.sum(weight_direct_matrix_tmp, axis=0)
            neighbors_merged_row = np.sum(neighbors_direct_matrix_tmp, axis=0)
            for r in range(ct_n):
                r_ct = cell_type[r]
                interaction[l,r] = np.sum(merged_row[np.where(np.array(cell_type_list)==r_ct)])
                edge_num = np.sum(neighbors_merged_row[np.where(np.array(cell_type_list)==r_ct)])
                if edge_num == 0:
                    interaction_per[l,r] = interaction[l,r]
                else:
                    interaction_per[l,r] = interaction[l,r]/edge_num
                interaction_edgenum[l,r] = edge_num
        ct_interaction_dict[key] = interaction
        ct_interaction_per_dict[key] = interaction_per
        ct_interaction_edgenum_dict[key] = interaction_edgenum
    
    ## 记录针对每个LR对，不同细胞类型之间的相互作用程度
    adata.uns['LR_celltype_weight'] = ct_interaction_dict
    adata.uns['LR_celltype_mean_weight'] = ct_interaction_per_dict
    adata.uns['LR_celltype_edge_num'] = ct_interaction_edgenum_dict
    
    result_weight = np.zeros_like(next(iter(ct_interaction_dict.values())))
    result_weight_per = np.zeros_like(next(iter(ct_interaction_per_dict.values())))
    result_count = np.zeros_like(next(iter(ct_interaction_dict.values())))
    result_edge = np.zeros_like(next(iter(ct_interaction_edgenum_dict.values())))
    for array in ct_interaction_dict.values():
        result_weight += array
    for array in ct_interaction_dict.values():
        result_count += (array != 0)
    for array in ct_interaction_per_dict.values():
        result_weight_per += array

    # 提取ct_interaction_edgenum_dict字典中每个array对应位置最大的值，组成新的数组，目的是看每个细胞类型之间有多少相互作用的细胞对
    edge_arrays = list(ct_interaction_edgenum_dict.values())
    result_edge = np.mean(edge_arrays,axis=0)
    result_edge = np.array(result_edge)
    
    adata.uns['LR_celltype_aggregate_weight'] = {'weight':result_weight,
                                                 'weight_per':result_weight_per,
                                                 'count':result_count,
                                                 'edge_num':result_edge}
    
    #celltype_pathway_level_cal
    if 'pathway' in list(LR_dict[list(LR_dict.keys())[0]].keys()):
        pathway_list = list(set([LR_dict[lr]['pathway'] for lr in LR_dict.keys()]))
        pathway_list = [p for p in pathway_list if str(p) != 'nan']
        pathway_interaction_mean_weight_dict = {}
        pathway_interaction_weight_dict = {}
        pathway_interaction_count_dict = {}
        pathway_interaction_edge_dict = {}
        for pathway in pathway_list:
            pathway_lr = [lr for lr in LR_dict.keys() if LR_dict[lr]['pathway'] == pathway]

            LR_pathway_mean_dict = {lr:adata.uns['LR_celltype_mean_weight'][lr] for lr in pathway_lr}
            LR_pathway_dict = {lr:adata.uns['LR_celltype_weight'][lr] for lr in pathway_lr}
            LR_pathway_edge_dict = {lr:adata.uns['LR_celltype_edge_num'][lr] for lr in pathway_lr}
            presult_weight_mean = np.zeros_like(next(iter(LR_pathway_mean_dict.values())))
            presult_weight = np.zeros_like(next(iter(LR_pathway_dict.values())))
            presult_edge = np.zeros_like(next(iter(LR_pathway_edge_dict.values())))
            presult_count = np.zeros_like(next(iter(LR_pathway_dict.values())))

            for array in LR_pathway_mean_dict.values():
                presult_weight_mean += array
            pathway_interaction_mean_weight_dict[pathway] = presult_weight_mean

            for array in LR_pathway_dict.values():
                presult_weight += array
            pathway_interaction_weight_dict[pathway] = presult_weight
            
            for array in LR_pathway_dict.values():
                presult_count += (array != 0)
            pathway_interaction_count_dict[pathway] = presult_count
            
            # 提取ct_interaction_edgenum_dict字典中每个array对应位置最大的值，组成新的数组，目的是看每个细胞类型之间有多少相互作用的细胞对
            edge_arrays = list(LR_pathway_edge_dict.values())
            presult_edge = np.mean(edge_arrays,axis=0)
            presult_edge = np.array(presult_edge)
            pathway_interaction_edge_dict[pathway] = presult_edge

        adata.uns['LR_pathway_celltype_weight'] = pathway_interaction_weight_dict
        adata.uns['LR_pathway_celltype_mean_weight'] = pathway_interaction_mean_weight_dict
        adata.uns['LR_pathway_celltype_edge_num'] = pathway_interaction_edge_dict
        adata.uns['LR_pathway_celltype_count'] = pathway_interaction_count_dict
    del(adata.uns['DB_complex'])
    
    ##singlecell_pathway_level_cal
    def dict_array_add(dic, key=None):
        if key==None:
            key = list(dic.keys())
        # 初始化结果数组，以第一个要相加的数组为基准
        result_array = dic[key[0]].copy()
        # 遍历其他要相加的数组，并相应位置相加
        for k in key[1:]:
            result_array += dic[k]
    
        return csr_matrix(result_array)

    if 'pathway' in list(LR_dict[list(LR_dict.keys())[0]].keys()):
        pathway_list = list(set([LR_dict[lr]['pathway'] for lr in LR_dict.keys()]))
        pathway_list = [p for p in pathway_list if str(p) != 'nan']
        LR_pathway_cell_weight = {}
        
        for pw in pathway_list:
            pw_lr = [lr for lr in LR_dict.keys() if LR_dict[lr]['pathway'] == pw]
            pw_dict = {}
            for pl in pw_lr:
                pw_dict[pl] = adata.uns['LR_cell_weight'][pl].toarray() 
            if len(pw_dict) > 0:
                LR_pathway_cell_weight[pw] = dict_array_add(pw_dict)

        adata.uns['LR_pathway_cell_weight'] = LR_pathway_cell_weight



def spatial_cell_communication_run(adata, 
                                   DB_interaction,
                                   DB_complex, 
                                   method, 
                                   ct_key='cell_type',
                                   cell_type=None,
                                   if_hvg=True,   
                                   if_filter=False,
                                   if_self=True,
                                   if_intra=True,
                                   if_stringent=False,
                                   intra_method='scenic',
                                   unig_network_path=None,
                                   unig_tf_activity_path=None,
                                   min_regulon_targets=10,
                                   DATABASES_GLOB=None,
                                   MOTIF_ANNOTATIONS_FNAME=None,
                                   hvg_n_top_genes=None,      
                                   hvg_min_mean=0.0125, 
                                   hvg_max_mean=3,    
                                   hvg_min_disp=0.5,     
                                   hvg_max_disp=np.inf,
                                   background_number=100,
                                   threads=50, 
                                   scope=6,
                                   min_exp=0.1,
                                   cutoff=0.05):
    if if_intra:
        if intra_method not in {'scenic', 'unig'}:
            raise ValueError("'intra_method' must be 'scenic' or 'unig'")
        if intra_method == 'scenic' and (DATABASES_GLOB is None or MOTIF_ANNOTATIONS_FNAME is None):
            raise ValueError("'DATABASES_GLOB' and 'MOTIF_ANNOTATIONS_FNAME' must be specified when 'if_intra' is True")
        if intra_method == 'unig' and (unig_network_path is None or unig_tf_activity_path is None):
            raise ValueError("'unig_network_path' and 'unig_tf_activity_path' must be specified when intra_method='unig'")
    
    print('##################################################################')
    print('Now start to calulate radius')
    find_radius(adata,scope=scope)
    
    if cell_type != None:
        print('##################################################################')
        print('Now start to subset anndata')
        adata = subset_anndata(adata, cell_type, key=ct_key)
    

    print('##################################################################')
    print('Now start to get LR pair')
    grep_exist_LR(adata,
                  DB_interaction,
                  DB_complex,
                  if_hvg=if_hvg,
                  n_top_genes=hvg_n_top_genes,
                  min_mean=hvg_min_mean,
                  max_mean=hvg_max_mean,
                  min_disp=hvg_min_disp,
                  max_disp=hvg_max_disp)

    print('##################################################################')
    print('get_LR_gene_exp')
    get_LR_gene_exp(adata,
                    threads=threads)

    print('##################################################################')
    print('get_close_gene')
    get_close_gene(adata,
                   background_number=background_number)

    print('##################################################################')
    print('Now start to get true weight matirx')
    get_true_weight_matirx(adata,
                           method, ## 计算lr分数的方法
                           scope=scope, ## 非分泌性受配体扩散范围到其周围的几个点
                           min_exp=min_exp)

    print('\n##################################################################')
    print('Now start to permutation test')
    permutation_test(adata, 
                     method, 
                     threads=threads,
                     min_exp=min_exp,
                     cutoff=cutoff)
    if if_intra:
        print('\n##################################################################')
        print('Now  start to add intracellular signals')
        if intra_method == 'scenic':
            add_intracellular_signals(adata, DB_interaction, DATABASES_GLOB, MOTIF_ANNOTATIONS_FNAME, if_stringent)
        else:
            add_unig_intracellular_signals(
                adata,
                DB_interaction,
                unig_network_path=unig_network_path,
                unig_tf_activity_path=unig_tf_activity_path,
                min_regulon_targets=min_regulon_targets,
                if_stringent=if_stringent,
            )
    if if_filter:
        print('\n##################################################################')
        print('Now  start to filter cell')
        adata = filter_cell(adata, key=ct_key, if_self=if_self)
        print('##################################################################')
    else:
        print('\n##################################################################')
    print('Now  start to aggregate')
    aggregate_matrix(adata, ct_key)
    print('Spatial cell communication finished!')
    return adata


def load_unig_ccc_databases(stcase_db_dir, species="Human"):
    """Load STCaseDB interaction and complex tables for UniG """
    db_dir = Path(stcase_db_dir)
    interaction = pd.read_csv(db_dir / f"STCaseDB_{species}.csv", index_col=0)
    complex_table = pd.read_csv(db_dir / f"STCaseDB_{species}_Complex.csv", index_col=0)
    return interaction, complex_table


def attach_unig_clusters_to_spatial_adata(
    spatial_adata,
    unig_adata_dict,
    entity_key="C",
    cluster_key="mclust",
    output_key="type",
):
    """Attach UniG cluster labels to a spatial RNA AnnData object."""
    adata_c = unig_adata_dict[entity_key]
    adata_c = adata_c[spatial_adata.obs_names, :]
    spatial_adata = spatial_adata.copy()
    spatial_adata.obs[output_key] = adata_c.obs[cluster_key].values
    return spatial_adata


def run_unig_ccc(
    *args,
    ct_key="type",
    method="Hill",
    cell_type=None,
    if_hvg=False,
    if_filter=False,
    if_self=True,
    if_intra=True,
    if_stringent=False,
    intra_method="unig",
    unig_network_path=None,
    unig_tf_activity_path=None,
    min_regulon_targets=10,
    background_number=1000,
    threads=10,
    scope=6,
    min_exp=0.1,
    cutoff=0.01,
):
    """Run UniG spatial cell-cell communication with the integrated CCC engine.

    Preferred usage is ``run_unig_ccc(adata, db_interaction, db_complex, ...)``.
    The previous positional form ``run_unig_ccc(ccc_module, adata, db_interaction,
    db_complex, ...)`` is still accepted for old notebooks.
    """
    if len(args) == 3:
        spatial_adata, db_interaction, db_complex = args
    elif len(args) == 4 and hasattr(args[0], "ccci"):
        _, spatial_adata, db_interaction, db_complex = args
    else:
        raise TypeError(
            "run_unig_ccc expects (adata, db_interaction, db_complex) or the "
            "legacy (ccc_module, adata, db_interaction, db_complex) form."
        )

    ccc_adata = spatial_cell_communication_run(
        spatial_adata,
        db_interaction,
        db_complex,
        method=method,
        ct_key=ct_key,
        cell_type=cell_type,
        if_hvg=if_hvg,
        if_filter=if_filter,
        if_self=if_self,
        if_intra=if_intra,
        if_stringent=if_stringent,
        intra_method=intra_method,
        unig_network_path=unig_network_path,
        unig_tf_activity_path=unig_tf_activity_path,
        min_regulon_targets=min_regulon_targets,
        background_number=background_number,
        threads=threads,
        scope=scope,
        min_exp=min_exp,
        cutoff=cutoff,
    )
    return sanitize_uns_keys_for_h5ad(ccc_adata)


def lr_cell_level_score(adata, lr_pair, lr_weight_key="LR_cell_weight"):
    """Return a spot-by-spot communication matrix for one LR pair."""
    lr_key = _resolve_uns_feature_key(adata, lr_weight_key, lr_pair)
    matrix = adata.uns[lr_weight_key][lr_key]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return pd.DataFrame(matrix, index=adata.obs_names, columns=adata.obs_names)


def lr_celltype_weight(adata, lr_pair, weight_key="LR_celltype_weight"):
    """Return sender-by-receiver cell-type communication for one LR pair."""
    type_list = [str(x) for x in adata.uns["type_list"]]
    lr_key = _resolve_uns_feature_key(adata, weight_key, lr_pair)
    return pd.DataFrame(adata.uns[weight_key][lr_key], index=type_list, columns=type_list)


def lr_celltype_mean_weight(adata, lr_pair, weight_key="LR_celltype_mean_weight"):
    """Return sender-by-receiver mean communication for one LR pair."""
    type_list = [str(x) for x in adata.uns["type_list"]]
    lr_key = _resolve_uns_feature_key(adata, weight_key, lr_pair)
    return pd.DataFrame(adata.uns[weight_key][lr_key], index=type_list, columns=type_list)


def celltype_communication_matrix(adata, weight_key="LR_celltype_weight"):
    """Aggregate all LR cell-type matrices into one sender-by-receiver matrix."""
    type_list = [str(x) for x in adata.uns["type_list"]]
    total_mat = np.zeros((len(type_list), len(type_list)), dtype=float)
    for matrix in adata.uns[weight_key].values():
        total_mat += np.asarray(matrix, dtype=float)
    return pd.DataFrame(total_mat, index=type_list, columns=type_list)


def plot_celltype_communication_heatmap(
    comm_mtx,
    out_pdf=None,
    title="Cell type communication",
    cmap="coolwarm",
    figsize=(6.5, 5.8),
):
    """Plot a log1p cell-type communication heatmap."""
    import matplotlib.pyplot as plt

    plot_mtx = np.log1p(comm_mtx)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plot_mtx.values, cmap=cmap, aspect="equal")

    ax.set_xticks(np.arange(comm_mtx.shape[1]))
    ax.set_yticks(np.arange(comm_mtx.shape[0]))
    ax.set_xticklabels(comm_mtx.columns)
    ax.set_yticklabels(comm_mtx.index)
    ax.set_xlabel("Receiver cluster")
    ax.set_ylabel("Sender cluster")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log1p(total communication abundance)")
    fig.tight_layout()

    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_chord_diagram_nature(
    comm_mtx,
    out_pdf=None,
    out_svg=None,
    title="Cell type communication",
    min_weight=None,
    top_n_edges=None,
    remove_self=True,
    sector_size="strength",
    gap_deg=3.0,
    ribbon_alpha=0.55,
    figsize=(6.2, 6.2),
    label_size=8.5,
    title_size=11,
    ring_width=0.055,
    font_family="Arial",
    color_adata=None,
    color_key="type",
    fallback_cmap="tab20",
):
    """Plot a chord diagram for aggregated cell-type communication."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    plt.rcParams["font.family"] = font_family
    mat = comm_mtx.copy().astype(float)
    labels = [str(x) for x in mat.index]
    if remove_self:
        shared = mat.index.intersection(mat.columns)
        for node in shared:
            mat.loc[node, node] = 0.0

    edges = _prepare_comm_edges(mat, min_weight=min_weight, top_n_edges=top_n_edges, remove_self=False)
    if not edges:
        raise ValueError("No positive communication edges remain after filtering.")

    edge_df = pd.DataFrame(edges, columns=["sender", "receiver", "weight"])
    displayed = pd.DataFrame(0.0, index=labels, columns=labels)
    for sender, receiver, weight in edges:
        if sender in displayed.index and receiver in displayed.columns:
            displayed.loc[sender, receiver] = weight

    out_strength = displayed.sum(axis=1)
    in_strength = displayed.sum(axis=0)
    total_strength = out_strength + in_strength
    if sector_size == "equal":
        sector_weights = pd.Series(1.0, index=labels)
    elif sector_size == "outgoing":
        sector_weights = out_strength
    elif sector_size == "incoming":
        sector_weights = in_strength
    else:
        sector_weights = total_strength
    fallback = max(float(total_strength.max()) * 0.02, 1e-12)
    sector_weights = sector_weights.replace(0, np.nan).fillna(fallback)

    n_labels = len(labels)
    available = 360.0 - gap_deg * n_labels
    sector_angles = sector_weights / sector_weights.sum() * available

    sectors = {}
    current = 90.0
    for label, width in zip(labels, sector_angles):
        start = current
        end = current - width
        sectors[label] = (start, end)
        current = end - gap_deg

    color_map = _scanpy_category_color_map(color_adata, color_key, labels, fallback_cmap=fallback_cmap)
    source_seg, target_seg = _allocate_chord_segments(labels, sectors, edge_df)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    for _, row in edge_df.sort_values("weight", ascending=True).iterrows():
        sender = row["sender"]
        receiver = row["receiver"]
        s0, s1 = source_seg[(sender, receiver)]
        t0, t1 = target_seg[(sender, receiver)]
        ax.add_patch(
            _make_ribbon(
                s0,
                s1,
                t0,
                t1,
                r=0.90,
                facecolor=color_map[sender],
                alpha=ribbon_alpha,
            )
        )

    outer_r = 1.00
    for label in labels:
        start, end = sectors[label]
        ax.add_patch(
            Wedge(
                (0, 0),
                outer_r,
                end,
                start,
                width=ring_width,
                facecolor=color_map[label],
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
        )
        mid = (start + end) / 2
        x_pos, y_pos = _pol2cart(mid, 1.10)
        rot, ha = _text_rotation(mid)
        ax.text(
            x_pos,
            y_pos,
            label,
            ha=ha,
            va="center",
            rotation=rot,
            rotation_mode="anchor",
            fontsize=label_size,
            color="#222222",
        )

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    if title:
        ax.set_title(title, fontsize=title_size, pad=12)

    fig.tight_layout()
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight", transparent=True)
    plt.show()
    return fig, ax


def receiver_lr_spot_matrix(adata, lr_weight_key="LR_cell_weight", dtype=np.float32):
    """Build an LR-pair-by-receiver-spot matrix."""
    return receiver_feature_spot_matrix(adata, lr_weight_key, dtype=dtype)


def receiver_pathway_spot_matrix(
    adata,
    pathway_weight_key="LR_pathway_cell_weight",
    dtype=np.float32,
):
    """Build a pathway-by-receiver-spot matrix."""
    return receiver_feature_spot_matrix(adata, pathway_weight_key, dtype=dtype)


def receiver_feature_spot_matrix(adata, weight_key, dtype=np.float32):
    """Build a feature-by-receiver-spot matrix from spot-by-spot weights."""
    feature_items = list(_uns_feature_items(adata, weight_key))
    feature_names = [feature for feature, _matrix in feature_items]
    spot_names = adata.obs_names.astype(str).to_list()
    receiver_rows = np.zeros((len(feature_names), adata.n_obs), dtype=dtype)

    for i, (_feature, matrix) in enumerate(feature_items):
        if sparse.issparse(matrix):
            receiver_total = np.asarray(matrix.sum(axis=0)).ravel()
        else:
            receiver_total = np.asarray(matrix, dtype=dtype).sum(axis=0)
        receiver_rows[i, :] = receiver_total.astype(dtype, copy=False)

    return pd.DataFrame(receiver_rows, index=feature_names, columns=spot_names)


def make_receiver_lr_adata(lr_spot_mtx, adata, cluster_key="type"):
    """Represent receiver LR abundance as spots by LR pairs."""
    return make_receiver_feature_adata(
        lr_spot_mtx,
        adata,
        cluster_key=cluster_key,
        feature_col="LR_pair",
        label_func=format_lr_name,
        label_col="LR_label",
    )


def make_receiver_pathway_adata(pathway_spot_mtx, adata, cluster_key="type"):
    """Represent receiver pathway abundance as spots by pathways."""
    return make_receiver_feature_adata(
        pathway_spot_mtx,
        adata,
        cluster_key=cluster_key,
        feature_col="pathway",
    )


def make_receiver_feature_adata(
    feature_spot_mtx,
    adata,
    cluster_key="type",
    feature_col="feature",
    label_func=None,
    label_col=None,
):
    """Represent receiver feature abundance as an AnnData matrix."""
    receiver_adata = sc.AnnData(feature_spot_mtx.T.astype(np.float32))
    receiver_adata.obs_names = feature_spot_mtx.columns.astype(str)
    receiver_adata.var_names = feature_spot_mtx.index.astype(str)
    receiver_adata.obs[cluster_key] = adata.obs.loc[receiver_adata.obs_names, cluster_key].astype("category")
    receiver_adata.var[feature_col] = receiver_adata.var_names
    if label_func is not None and label_col is not None:
        receiver_adata.var[label_col] = [label_func(x) for x in receiver_adata.var_names]
    return receiver_adata


def summarize_receiver_by_cluster(feature_spot_mtx, adata, cluster_key="type"):
    """Average receiver feature abundance for each cluster."""
    cluster_labels = adata.obs[cluster_key]
    if hasattr(cluster_labels, "cat"):
        cluster_order = [str(x) for x in cluster_labels.cat.categories]
    else:
        cluster_order = [str(x) for x in pd.Index(cluster_labels.astype(str)).drop_duplicates()]

    cluster_labels = cluster_labels.astype(str)
    mean_rows = []
    pct_rows = []
    for cluster in cluster_order:
        spots = adata.obs_names[cluster_labels.to_numpy() == cluster].astype(str).to_list()
        if not spots:
            continue
        sub = feature_spot_mtx.loc[:, spots]
        mean_rows.append(sub.mean(axis=1).rename(cluster))
        pct_rows.append((sub > 0).mean(axis=1).rename(cluster))

    return pd.concat(mean_rows, axis=1), pd.concat(pct_rows, axis=1)


def summarize_lr_receiver_by_cluster(lr_spot_mtx, adata, cluster_key="type"):
    """Average LR receiver abundance for each cluster."""
    return summarize_receiver_by_cluster(lr_spot_mtx, adata, cluster_key=cluster_key)


def summarize_pathway_receiver_by_cluster(pathway_spot_mtx, adata, cluster_key="type"):
    """Average pathway receiver abundance for each cluster."""
    return summarize_receiver_by_cluster(pathway_spot_mtx, adata, cluster_key=cluster_key)


def rank_cluster_specific_lr_pairs_scanpy(
    lr_spot_mtx,
    adata,
    cluster_key="type",
    top_n=20,
    method="wilcoxon",
    min_pct=0.05,
    max_pval_adj=0.05,
    min_logfoldchange=0.0,
    fill_top_n_with_unfiltered=False,
):
    """Rank receiver LR markers with Scanpy group-vs-rest marker logic."""
    return rank_cluster_specific_features_scanpy(
        lr_spot_mtx,
        adata,
        cluster_key=cluster_key,
        feature_col="LR_pair",
        top_n=top_n,
        method=method,
        min_pct=min_pct,
        max_pval_adj=max_pval_adj,
        min_logfoldchange=min_logfoldchange,
        fill_top_n_with_unfiltered=fill_top_n_with_unfiltered,
        label_func=format_lr_name,
        label_col="LR_label",
    )


def rank_cluster_specific_pathways_scanpy(
    pathway_spot_mtx,
    adata,
    cluster_key="type",
    top_n=20,
    method="wilcoxon",
    min_pct=0.05,
    max_pval_adj=0.05,
    min_logfoldchange=0.0,
    fill_top_n_with_unfiltered=False,
):
    """Rank receiver pathway markers with Scanpy group-vs-rest marker logic."""
    return rank_cluster_specific_features_scanpy(
        pathway_spot_mtx,
        adata,
        cluster_key=cluster_key,
        feature_col="pathway",
        top_n=top_n,
        method=method,
        min_pct=min_pct,
        max_pval_adj=max_pval_adj,
        min_logfoldchange=min_logfoldchange,
        fill_top_n_with_unfiltered=fill_top_n_with_unfiltered,
    )


def rank_cluster_specific_features_scanpy(
    feature_spot_mtx,
    adata,
    cluster_key="type",
    feature_col="feature",
    top_n=20,
    method="wilcoxon",
    min_pct=0.05,
    max_pval_adj=0.05,
    min_logfoldchange=0.0,
    fill_top_n_with_unfiltered=False,
    label_func=None,
    label_col=None,
):
    """Rank receiver communication markers with Scanpy group-vs-rest logic."""
    receiver_adata = make_receiver_feature_adata(
        feature_spot_mtx,
        adata,
        cluster_key=cluster_key,
        feature_col=feature_col,
        label_func=label_func,
        label_col=label_col,
    )
    sc.pp.log1p(receiver_adata)
    sc.tl.rank_genes_groups(
        receiver_adata,
        groupby=cluster_key,
        reference="rest",
        method=method,
        pts=True,
        use_raw=False,
        n_genes=receiver_adata.n_vars,
    )

    marker_df = sc.get.rank_genes_groups_df(receiver_adata, group=None)
    marker_df = marker_df.rename(
        columns={
            "group": "cluster",
            "names": feature_col,
            "pct_nz_group": "pct_receiver_spots_positive",
            "pct_nz_reference": "pct_receiver_spots_positive_rest",
        }
    )
    marker_df["cluster"] = marker_df["cluster"].astype(str)
    marker_df[feature_col] = marker_df[feature_col].astype(str)
    if label_func is not None and label_col is not None:
        marker_df[label_col] = marker_df[feature_col].map(label_func)

    raw_mean, raw_pct = summarize_receiver_by_cluster(
        feature_spot_mtx,
        adata,
        cluster_key=cluster_key,
    )
    marker_df = _merge_receiver_mean_and_pct(marker_df, raw_mean, raw_pct, feature_col)
    marker_df = marker_df.sort_values(
        ["cluster", "scores", "logfoldchanges", "mean_receiver_abundance"],
        ascending=[True, False, False, False],
    )

    keep = marker_df["pct_receiver_spots_positive"].fillna(1.0) >= min_pct
    if np.isfinite(max_pval_adj):
        keep &= marker_df["pvals_adj"].fillna(0.0) <= max_pval_adj
    if np.isfinite(min_logfoldchange):
        keep &= marker_df["logfoldchanges"].fillna(0.0) >= min_logfoldchange

    if fill_top_n_with_unfiltered:
        selected = []
        for _cluster, cluster_df in marker_df.groupby("cluster", sort=False):
            filtered = cluster_df.loc[keep.loc[cluster_df.index]].head(top_n)
            if len(filtered) < top_n:
                filler = cluster_df.loc[
                    ~cluster_df[feature_col].isin(filtered[feature_col])
                ].head(top_n - len(filtered))
                filtered = pd.concat([filtered, filler], axis=0)
            selected.append(filtered)
        marker_df = pd.concat(selected, ignore_index=True)
    else:
        marker_df = (
            marker_df.loc[keep]
            .groupby("cluster", group_keys=False, sort=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    return marker_df, receiver_adata, raw_mean, raw_pct


def format_lr_name(lr_name):
    """Format an LR pair name for display."""
    return str(lr_name).replace("COMPLEX:", "").replace("|", "->")


def plot_cluster_specific_lr_marker_heatmap(
    mean_by_cluster,
    marker_table,
    adata,
    cluster_key="type",
    out_pdf=None,
    out_svg=None,
    cmap="RdBu_r",
    vmin=-2.0,
    vmax=2.0,
):
    """Plot cluster-specific receiver LR markers."""
    selected_table = marker_table.drop_duplicates(["cluster", "LR_pair"]).copy()
    selected = selected_table["LR_pair"].drop_duplicates().to_list()
    if not selected:
        raise ValueError("No cluster-specific LR pairs passed the current filters.")

    heatmap_raw = np.log1p(mean_by_cluster.loc[selected])
    heatmap_z = _row_zscore(heatmap_raw).clip(vmin, vmax)
    label_map = selected_table.drop_duplicates("LR_pair").set_index("LR_pair")["LR_label"].to_dict()
    row_labels = [label_map.get(x, format_lr_name(x)) for x in heatmap_z.index]
    row_group_map = selected_table.drop_duplicates("LR_pair").set_index("LR_pair")["cluster"].to_dict()
    row_groups = [row_group_map.get(x, "") for x in heatmap_z.index]

    return _draw_marker_heatmap(
        heatmap_z,
        row_labels=row_labels,
        row_groups=row_groups,
        adata=adata,
        cluster_key=cluster_key,
        title="Cluster-specific receiver L-R communication markers",
        ylabel="Receiver L-R pair",
        colorbar_label="Row z-score of log1p(mean receiver abundance)",
        out_pdf=out_pdf,
        out_svg=out_svg,
        fig_width=7.0,
        row_height=0.125,
        min_fig_height=4.2,
        max_fig_height=11.0,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        row_label_size=5.2,
    )


def plot_cluster_specific_pathway_marker_heatmap(
    mean_by_cluster,
    marker_table,
    adata,
    cluster_key="type",
    out_pdf=None,
    out_svg=None,
    cmap="RdBu_r",
    vmin=-2.0,
    vmax=2.0,
):
    """Plot cluster-specific receiver pathway markers."""
    selected_table = marker_table.drop_duplicates(["cluster", "pathway"]).copy()
    selected = selected_table["pathway"].drop_duplicates().to_list()
    if not selected:
        raise ValueError("No cluster-specific pathways passed the current filters.")

    heatmap_raw = np.log1p(mean_by_cluster.loc[selected])
    heatmap_z = _row_zscore(heatmap_raw).clip(vmin, vmax)
    row_group_map = selected_table.drop_duplicates("pathway").set_index("pathway")["cluster"]
    row_groups = row_group_map.reindex(heatmap_z.index).fillna("").to_list()

    return _draw_marker_heatmap(
        heatmap_z,
        row_labels=heatmap_z.index.to_list(),
        row_groups=row_groups,
        adata=adata,
        cluster_key=cluster_key,
        title="Cluster-specific receiver pathway communication markers",
        ylabel="Receiver pathway",
        colorbar_label="Row z-score of log1p(mean receiver abundance)",
        out_pdf=out_pdf,
        out_svg=out_svg,
        fig_width=6.8,
        row_height=0.28,
        min_fig_height=3.6,
        max_fig_height=9.0,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        row_label_size=7.2,
    )


def extract_cluster_markers(
    marker_df,
    cluster_value="1",
    feature_col="LR_pair",
    label_col=None,
    top_n=20,
):
    """Extract the top receiver communication markers for one cluster."""
    df = marker_df.copy()
    df["cluster"] = df["cluster"].astype(str)
    out = df.loc[df["cluster"] == str(cluster_value)].copy()
    if out.empty:
        available = sorted(df["cluster"].astype(str).unique())
        raise ValueError(f"No markers found for cluster {cluster_value!r}. Available clusters: {available}")

    sort_cols = [c for c in ["scores", "logfoldchanges", "mean_receiver_abundance"] if c in out.columns]
    out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(top_n).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    if label_col is None:
        label_col = feature_col
    if "label" not in out.columns:
        out["label"] = out[label_col].astype(str)
    if "communication_abundance" not in out.columns and "mean_receiver_abundance" in out.columns:
        out["communication_abundance"] = out["mean_receiver_abundance"]
    return out


def plot_cluster_marker_score_bars(
    lr_markers,
    pathway_markers,
    cluster_value="1",
    top_n=20,
    out_pdf=None,
    cmap="YlGnBu",
):
    """Plot LR and pathway receiver marker scores for one cluster."""
    import matplotlib.pyplot as plt

    lr_logfc = lr_markers["logfoldchanges"].astype(float).to_numpy()
    pathway_logfc = pathway_markers["logfoldchanges"].astype(float).to_numpy()
    all_logfc = np.concatenate([lr_logfc[np.isfinite(lr_logfc)], pathway_logfc[np.isfinite(pathway_logfc)]])
    if len(all_logfc) > 0 and np.nanmax(all_logfc) > np.nanmin(all_logfc):
        norm = plt.Normalize(vmin=np.nanmin(all_logfc), vmax=np.nanmax(all_logfc))
    else:
        norm = plt.Normalize(vmin=0, vmax=1)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4), constrained_layout=True)
    _plot_marker_score_bar(
        lr_markers,
        label_col="label",
        title=f"Cluster {cluster_value} receiver top {top_n} L-R markers",
        ax=axes[0],
        top_n=top_n,
        norm=norm,
        cmap=cmap,
    )
    _plot_marker_score_bar(
        pathway_markers,
        label_col="label",
        title=f"Cluster {cluster_value} receiver top {top_n} pathway markers",
        ax=axes[1],
        top_n=top_n,
        norm=norm,
        cmap=cmap,
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.82, pad=0.015)
    cbar.set_label("logfoldchanges", rotation=90)

    if out_pdf is not None:
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.show()
    return fig, axes


def add_lr_receiver_total(adata, lr_pair, score_key=None, lr_weight_key="LR_cell_weight"):
    """Add receiver total score for one LR pair to adata.obs."""
    if score_key is None:
        score_key = f"{_safe_obs_key(lr_pair)}_receiver_total"
    return add_receiver_total_score(adata, lr_pair, score_key, lr_weight_key)


def add_pathway_receiver_total(
    adata,
    pathway,
    score_key=None,
    pathway_weight_key="LR_pathway_cell_weight",
):
    """Add receiver total score for one pathway to adata.obs."""
    if score_key is None:
        score_key = f"{_safe_obs_key(pathway)}_pathway_receiver_total"
    return add_receiver_total_score(adata, pathway, score_key, pathway_weight_key)


def add_receiver_total_score(adata, feature, score_key, weight_key):
    """Add the receiver total score for one communication feature to adata.obs."""
    feature_key = _resolve_uns_feature_key(adata, weight_key, feature)
    matrix = adata.uns[weight_key][feature_key]
    if sparse.issparse(matrix):
        receiver_total = np.asarray(matrix.sum(axis=0)).ravel()
    else:
        receiver_total = np.asarray(matrix).sum(axis=0)
    adata.obs[score_key] = receiver_total
    return score_key


def plot_spatial_receiver_score(
    adata,
    score_key,
    basis="spatial",
    size=75,
    title=None,
    cmap="coolwarm",
    save=None,
):
    """Plot a receiver communication score on spatial coordinates."""
    return sc.pl.embedding(
        adata,
        basis=basis,
        color=[score_key],
        size=size,
        title=title or score_key,
        cmap=cmap,
        save=save,
    )


def _merge_receiver_mean_and_pct(marker_df, raw_mean, raw_pct, feature_col):
    long_mean = (
        raw_mean.stack()
        .rename("mean_receiver_abundance")
        .reset_index()
        .rename(columns={"level_0": feature_col, "level_1": "cluster"})
    )
    long_mean[feature_col] = long_mean[feature_col].astype(str)
    long_mean["cluster"] = long_mean["cluster"].astype(str)
    marker_df = marker_df.merge(long_mean, on=[feature_col, "cluster"], how="left")

    if "pvals_adj" not in marker_df.columns:
        marker_df["pvals_adj"] = np.nan
    if "logfoldchanges" not in marker_df.columns:
        marker_df["logfoldchanges"] = np.nan
    if "pct_receiver_spots_positive" not in marker_df.columns:
        long_pct = (
            raw_pct.stack()
            .rename("pct_receiver_spots_positive")
            .reset_index()
            .rename(columns={"level_0": feature_col, "level_1": "cluster"})
        )
        long_pct[feature_col] = long_pct[feature_col].astype(str)
        long_pct["cluster"] = long_pct["cluster"].astype(str)
        marker_df = marker_df.merge(long_pct, on=[feature_col, "cluster"], how="left")
    return marker_df


def _prepare_comm_edges(comm_mtx, min_weight=None, top_n_edges=None, remove_self=True):
    mat = comm_mtx.copy().astype(float)
    if remove_self:
        shared = mat.index.intersection(mat.columns)
        for node in shared:
            mat.loc[node, node] = 0.0

    edges = []
    for sender in mat.index:
        for receiver in mat.columns:
            weight = float(mat.loc[sender, receiver])
            if np.isfinite(weight) and weight > 0:
                edges.append((str(sender), str(receiver), weight))

    if min_weight is not None:
        edges = [edge for edge in edges if edge[2] >= min_weight]
    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    if top_n_edges is not None:
        edges = edges[: int(top_n_edges)]
    return edges


def _row_zscore(df):
    values = df.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, keepdims=True)
    z = (values - mean) / np.where(std == 0, 1.0, std)
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def _draw_marker_heatmap(
    heatmap_z,
    row_labels,
    row_groups,
    adata,
    cluster_key,
    title,
    ylabel,
    colorbar_label,
    out_pdf=None,
    out_svg=None,
    fig_width=7.0,
    row_height=0.135,
    min_fig_height=4.0,
    max_fig_height=11.0,
    cmap="RdBu_r",
    vmin=-2.0,
    vmax=2.0,
    row_label_size=5.6,
    col_label_size=8.0,
):
    import matplotlib.pyplot as plt

    n_rows, n_cols = heatmap_z.shape
    fig_height = min(max_fig_height, max(min_fig_height, 1.15 + n_rows * row_height))
    left = 0.24 if n_rows > 45 else 0.22
    right = 0.89
    bottom = 0.075
    top = 0.93
    strip_h = 0.018
    strip_gap = 0.010

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([left, bottom, right - left, top - bottom - strip_h - strip_gap])
    ax_colors = fig.add_axes([left, top - strip_h, right - left, strip_h])
    cax = fig.add_axes([0.925, bottom, 0.030, top - bottom - strip_h - strip_gap])

    cluster_order = heatmap_z.columns.to_list()
    color_map = _scanpy_category_color_map(adata, cluster_key, cluster_order, fallback_cmap="tab20")
    rgb = np.array([plt.matplotlib.colors.to_rgb(color_map[c]) for c in cluster_order])[np.newaxis, :, :]
    ax_colors.imshow(rgb, aspect="auto", interpolation="nearest")
    ax_colors.set_xlim(-0.5, n_cols - 0.5)
    ax_colors.set_xticks(np.arange(n_cols))
    ax_colors.set_xticklabels(cluster_order, rotation=45, ha="right", fontsize=col_label_size)
    ax_colors.set_yticks([])
    ax_colors.tick_params(axis="x", length=0, pad=1)
    for spine in ax_colors.spines.values():
        spine.set_visible(False)

    im = ax.imshow(
        heatmap_z.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(cluster_order, rotation=45, ha="right", fontsize=col_label_size)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=row_label_size)
    ax.set_xlabel("Receiver cell type", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    ax.tick_params(length=0, pad=1)

    group_counts = pd.Series(row_groups).value_counts(sort=False).to_numpy()
    for boundary in np.cumsum(group_counts)[:-1]:
        ax.axhline(boundary - 0.5, color="white", lw=1.0)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(colorbar_label, rotation=90, fontsize=8)
    cbar.ax.tick_params(labelsize=8, length=2)

    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight")
    plt.show()
    return fig, ax


def _plot_marker_score_bar(df, label_col, title, ax, top_n=20, norm=None, cmap="YlGnBu"):
    import matplotlib.pyplot as plt

    plot_df = df.head(top_n).copy()
    values = plot_df["scores"].astype(float).to_numpy()
    logfc = plot_df.get("logfoldchanges", pd.Series(np.nan, index=plot_df.index)).astype(float).to_numpy()

    cmap_obj = plt.get_cmap(cmap)
    if norm is None:
        if np.isfinite(logfc).any() and np.nanmax(logfc) > np.nanmin(logfc):
            norm = plt.Normalize(vmin=np.nanmin(logfc), vmax=np.nanmax(logfc))
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
    colors = cmap_obj(norm(logfc)) if np.isfinite(logfc).any() else cmap_obj(np.linspace(0.25, 0.85, len(plot_df)))

    x_pos = np.arange(len(plot_df))
    ax.bar(x_pos, values, color=colors, edgecolor="white", linewidth=0.5, width=0.78)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_df[label_col].astype(str), rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Scanpy marker score")
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    return ax


def _scanpy_category_color_map(adata, color_key, labels, fallback_cmap="tab20"):
    import matplotlib.pyplot as plt

    labels = [str(label) for label in labels]
    fallback_palette = plt.get_cmap(fallback_cmap)(np.linspace(0, 1, max(len(labels), 3)))
    fallback = {label: fallback_palette[i] for i, label in enumerate(labels)}

    if adata is None or color_key not in getattr(adata, "obs", {}):
        return fallback

    color_key_uns = f"{color_key}_colors"
    if color_key_uns not in adata.uns:
        return fallback

    values = adata.obs[color_key]
    if hasattr(values, "cat"):
        categories = [str(x) for x in values.cat.categories]
    else:
        categories = [str(x) for x in pd.Categorical(values).categories]

    colors = list(adata.uns[color_key_uns])
    scanpy_map = {category: colors[i] for i, category in enumerate(categories) if i < len(colors)}
    return {label: scanpy_map.get(label, fallback[label]) for label in labels}


def _allocate_chord_segments(labels, sectors, edge_df):
    source_seg = {}
    target_seg = {}
    for label in labels:
        start, end = sectors[label]
        width = start - end

        outgoing = edge_df.loc[edge_df["sender"] == label]
        out_total = outgoing["weight"].sum()
        cursor = start
        for _, row in outgoing.iterrows():
            span = width * row["weight"] / out_total if out_total > 0 else 0
            source_seg[(row["sender"], row["receiver"])] = (cursor, cursor - span)
            cursor -= span

        incoming = edge_df.loc[edge_df["receiver"] == label]
        in_total = incoming["weight"].sum()
        cursor = start
        for _, row in incoming.iterrows():
            span = width * row["weight"] / in_total if in_total > 0 else 0
            target_seg[(row["sender"], row["receiver"])] = (cursor, cursor - span)
            cursor -= span
    return source_seg, target_seg


def _make_ribbon(theta1a, theta1b, theta2a, theta2b, r=0.92, alpha=0.55, facecolor="gray"):
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    p1 = _pol2cart(theta1a, r)
    p2 = _pol2cart(theta1b, r)
    p3 = _pol2cart(theta2a, r)
    p4 = _pol2cart(theta2b, r)
    center = np.array([0.0, 0.0])
    verts = [p1, center, center, p4, p3, center, center, p2, p1]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(
        MplPath(verts, codes),
        facecolor=facecolor,
        edgecolor="none",
        lw=0.0,
        alpha=alpha,
        zorder=1,
    )


def _pol2cart(theta_deg, r):
    theta = np.deg2rad(theta_deg)
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def _text_rotation(theta_deg):
    if 90 < theta_deg < 270:
        return theta_deg + 180, "right"
    return theta_deg, "left"


def _safe_obs_key(value):
    return (
        str(value)
        .replace("COMPLEX:", "")
        .replace("|", "_")
        .replace("->", "_")
        .replace(":", "_")
        .replace("-", "_")
    )
