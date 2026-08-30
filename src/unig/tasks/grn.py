import anndata as ad
import numpy as np
import os
import pandas as pd
from scipy import sparse
from anndata import AnnData
from pyfaidx import Fasta
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def shuffle_peak_motif(adata_atac, adata_pm, genome="mm10", n_background=50, seed=42):
    """Generate a background peak-motif matrix matched by GC and accessibility."""
    fasta_path = (
        f"/home/nas3/biod/yangchenghui/proj2/UniG_master/genomes/{genome}/{genome}.fa"
    )

    print("Step 1: Computing peak accessibility.")
    peaks = adata_atac.var_names.tolist()
    if sparse.issparse(adata_atac.X):
        accessibility = np.asarray(adata_atac.X.mean(axis=0)).ravel()
    else:
        accessibility = np.mean(adata_atac.X, axis=0)

    print("Step 2: Computing GC content.")
    gc_content = _compute_peak_gc_content(peaks, fasta_path)

    print("Step 3: Finding background peaks.")
    background_indices = _find_background_peaks(
        gc_content,
        accessibility,
        n_background=n_background,
    )
    print(f"Background indices shape: {background_indices.shape}")

    print("Step 4: Loading peak-motif matrix.")
    adata_pm = adata_pm[peaks, :].copy()
    peak_motif_sparse = sparse.csc_matrix(adata_pm.X)
    peak_names = adata_pm.obs_names.copy()
    motif_names = adata_pm.var_names.copy()
    n_peaks, n_motifs = peak_motif_sparse.shape

    print("Step 5: Generating background peak-motif matrix.")
    rng = np.random.default_rng(seed)
    rows = []
    cols = []

    for motif_idx in tqdm(range(n_motifs), desc="Processing motifs"):
        motif_values = peak_motif_sparse[:, motif_idx].toarray().ravel()
        background_probs = motif_values[background_indices].mean(axis=1)
        active_peak_indices = np.where(rng.random(n_peaks) < background_probs)[0]

        if len(active_peak_indices) == 0:
            continue

        rows.extend(active_peak_indices)
        cols.extend([motif_idx] * len(active_peak_indices))

    data = np.ones(len(rows), dtype=np.int8)
    background_matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(n_peaks, n_motifs),
    ).tocsr()

    adata_pm_background = AnnData(
        X=background_matrix,
        obs=pd.DataFrame(index=peak_names),
        var=pd.DataFrame(index=motif_names),
    )

    print(f"Finished background matrix: {adata_pm_background.shape}")
    return adata_pm_background


def _compute_peak_gc_content(peaks, fasta_path):
    if not os.path.exists(fasta_path):
        print("Warning: reference genome FASTA not found. Using GC=0.5 for all peaks.")
        return np.full(len(peaks), 0.5, dtype=float)

    print(f"Loading genome from {fasta_path}.")
    genome_fasta = Fasta(fasta_path)
    chrom_set = set(genome_fasta.keys())
    gc_content = []

    for peak in tqdm(peaks, desc="Calculating GC"):
        try:
            chrom, start, end = _parse_peak_interval(peak)
            chrom = _resolve_chrom_name(chrom, chrom_set)
            if chrom is None:
                gc_content.append(0.5)
                continue

            seq = str(genome_fasta[chrom][start:end]).upper()
            gc_content.append(_gc_fraction(seq))
        except (IndexError, TypeError, ValueError):
            gc_content.append(0.5)

    return np.asarray(gc_content, dtype=float)


def _parse_peak_interval(peak):
    if ":" in peak:
        chrom, coords = peak.split(":", maxsplit=1)
        start, end = coords.split("-", maxsplit=1)
    else:
        chrom, start, end = peak.split("-", maxsplit=2)

    return chrom, int(start), int(end)


def _resolve_chrom_name(chrom, chrom_set):
    if chrom in chrom_set:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in chrom_set:
        return chrom[3:]
    if not chrom.startswith("chr") and f"chr{chrom}" in chrom_set:
        return f"chr{chrom}"
    return None


def _gc_fraction(seq):
    if len(seq) == 0:
        return 0.5
    return (seq.count("G") + seq.count("C")) / len(seq)


def _find_background_peaks(gc_content, accessibility, n_background):
    log_accessibility = np.log1p(accessibility)
    norm_gc = _zscore_or_zero(gc_content)
    norm_acc = _zscore_or_zero(log_accessibility)
    features = np.vstack([norm_gc, norm_acc]).T

    n_neighbors = min(n_background + 1, features.shape[0])
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree")
    neighbors.fit(features)
    _, indices = neighbors.kneighbors(features)

    return indices[:, 1:]


def _zscore_or_zero(values):
    values = np.asarray(values, dtype=float)
    std = np.std(values)
    if std == 0:
        return np.zeros_like(values)
    return (values - np.mean(values)) / std


def get_celltype_cres(df,
                    threshold=0.95, 
                    min_score_floor=0.2, 
                    max_score_floor=0.5):
    """
    First normalize Score at Gene level (Min-Max),
    then select Top CREs and plot gene cutoff distribution.
    Finally output results keeping original Score.
    """
    filtered_rows = []
    gene_cutoffs = []
    
    # 1. Copy and backup original Score
    df = df.copy()
    df['celltype'] = df['celltype'].astype(str)
    df['raw_score'] = df['score']  # Backup original score
    print("Performing Gene-level Min-Max Normalization...")
    
    # 2. Define normalization function
    def min_max_norm(x):
        if x.max() == x.min():
            return np.ones_like(x) # If max equals min, set all to 1 (or 0, depending on needs, setting to 1 to keep here)
        return (x - x.min()) / (x.max() - x.min())

    # Transform score column grouped by gene
    # transform keeps index unchanged, assign back to original DataFrame
    df['score'] = df.groupby('gene')['score'].transform(min_max_norm)

    print(f"Filtering by gene group (Top {(1-threshold):.0%})...")
    print(f"Normalized Score limits: Min={min_score_floor}, Max Cap={max_score_floor}")

    # === Core Filtering (using normalized score) ===
    for gene, group in df.groupby('gene'):
        cutoff = 0
        raw_cutoff = group['score'].quantile(threshold)
        
        # 1. Apply minimum floor (prevent noise)
        if min_score_floor is not None:
            cutoff = max(raw_cutoff, min_score_floor)
        else:
            cutoff = raw_cutoff
            
        # 2. Apply maximum cap (prevent over-killing)
        if max_score_floor is not None:
            cutoff = min(cutoff, max_score_floor)
        
        mask = (group['score'] > cutoff)
        selected_group = group[mask]
        
        # Record Cutoff for this gene (normalized value)
        gene_cutoffs.append({'gene': gene, 'cutoff': cutoff})
            
        if not selected_group.empty:
            filtered_rows.append(selected_group)

    # === Merge and Output ===
    if filtered_rows:
        filtered_df = pd.concat(filtered_rows, ignore_index=True)
        
        # 3. Restore Original Score
        # Assign raw_score back to score column, and drop raw_score column
        filtered_df['score'] = filtered_df['raw_score']
        filtered_df = filtered_df.drop(columns=['raw_score'])
        
    else:
        filtered_df = pd.DataFrame(columns=df.columns).drop(columns=['raw_score'])
        
    if not filtered_df.empty:
        cre_dict = {}
        for ct, group in filtered_df.groupby('celltype'):
            idx = group.index
            filtered_df.loc[idx, 'score'] = group['score'] / group['score'].max()

            cre_dict[ct] = group['peak'].unique().tolist()
            print(f"CellType {ct}: Selected {len(cre_dict[ct])} validated peaks")

    return cre_dict, filtered_df

def correct_TF_activity(adata_C, adata_M, adata_C_bg, adata_M_bg):
    """Correct motif activity by subtracting background reconstructed activity."""
    common_cells = adata_C.obs_names.intersection(adata_C_bg.obs_names)
    common_motifs = adata_M.obs_names.intersection(adata_M_bg.obs_names)

    if len(common_cells) == 0:
        raise ValueError("No shared cells between adata_C and adata_C_bg.")
    if len(common_motifs) == 0:
        raise ValueError("No shared motifs between adata_M and adata_M_bg.")

    if len(common_cells) != adata_C.n_obs or len(common_motifs) != adata_M.n_obs:
        print("Warning: cell or motif indices do not fully match. Aligning inputs.")

    adata_C = adata_C[common_cells].copy()
    adata_C_bg = adata_C_bg[common_cells].copy()
    adata_M = adata_M[common_motifs].copy()
    adata_M_bg = adata_M_bg[common_motifs].copy()

    raw_activity = adata_C.X @ adata_M.X.T
    background_activity = adata_C_bg.X @ adata_M_bg.X.T

    if sparse.issparse(raw_activity):
        raw_activity = raw_activity.toarray()
    if sparse.issparse(background_activity):
        background_activity = background_activity.toarray()

    corrected_activity = np.asarray(raw_activity) - np.asarray(background_activity)

    return AnnData(
        X=corrected_activity,
        obs=adata_C.obs.copy(),
        var=adata_M.obs.copy(),
        obsm=adata_C.obsm.copy(),
    )

def TF_binding_peak_strenth(adata_PM, adata_P, adata_M):
    cos_sim = (cosine_similarity(adata_P.X, adata_M.X)+1)/2
    adata_PM_unig = ad.AnnData(
        X=cos_sim,
        obs=pd.DataFrame(index=adata_P.obs_names),   # peaks
        var=pd.DataFrame(index=adata_M.obs_names)    # motifs
    )
    adata_PM_unig = adata_PM_unig[adata_PM.obs_names, adata_PM.var_names]
    assert (adata_PM.obs_names == adata_PM_unig.obs_names).all()
    assert (adata_PM.var_names == adata_PM_unig.var_names).all()

    PM = adata_PM.X
    CS = adata_PM_unig.X

    if not isinstance(PM, np.ndarray):
        PM = PM.toarray()
    if not isinstance(CS, np.ndarray):
        CS = CS.toarray()
    X_new = CS * (PM == 1)
    adata_PM_gated = ad.AnnData(
        X=X_new,
        obs=adata_PM.obs.copy(),
        var=adata_PM.var.copy()
    )
    return adata_PM_gated


def fast_pearson_corr(A, B):
    """
    Helper function: Fast calculation of Pearson correlation coefficients between matrix columns.
    """
    if sparse.issparse(A): A = A.toarray()
    if sparse.issparse(B): B = B.toarray()
    n = A.shape[0]
    if n < 2: return np.zeros((A.shape[1], B.shape[1]))
    A_centered = A - A.mean(axis=0)
    B_centered = B - B.mean(axis=0)
    A_std = np.std(A, axis=0) + 1e-8
    B_std = np.std(B, axis=0) + 1e-8
    A_norm = A_centered / A_std
    B_norm = B_centered / B_std
    return np.dot(A_norm.T, B_norm) / n

def celltype_exp(adata):
    """
    Calculate cell-type specific average expression with normalization.
    """
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
        
    labels = adata.obs['label'].values
    fea_names = adata.var_names

    # 1. min-shift
    min_fea = X.min(axis=0)
    min_fea[min_fea > 0] = 0
    X_pos = X - min_fea
     
    # 2. cell-type mean
    df = pd.DataFrame(X_pos, index=labels, columns=fea_names)
    fea_ct_mean = df.groupby(level=0).mean()

    # 3. max-normalize across cell types
    fea_ct_mean = fea_ct_mean / fea_ct_mean.max(axis=0)
    fea_ct_mean = fea_ct_mean.fillna(0)
    return fea_ct_mean

def calculate_tf_re_binding_potential(
    ct_str,
    ct_cre_df,
    ct_tf_df,
    ct_peak_exp,
    adata_pm,
    adata_motif_expression,
    adata_peak_expression,
    peak_to_idx,
    motif_to_idx,
    peak_names,
    motif_names,
    cell_type_col='label'
):
    """
    Part 1: Calculate TF-RE (TF-Peak) Binding Potential
    TF binding potential = (TF-RE binding affinity) * (TF expression) * (peak expression) * (TF-peak PCC)
    """
    
    # 1. Determine Valid Peaks and Active TFs
    valid_peaks = ct_cre_df['peak'].unique()
    valid_peaks = [p for p in valid_peaks if p in peak_to_idx]
    if not valid_peaks: return None
    valid_peak_indices = [peak_to_idx[p] for p in valid_peaks]
    
    active_tfs = ct_tf_df['TF'].values
    active_tf_indices = [motif_to_idx[tf] for tf in active_tfs if tf in motif_to_idx]
    if not active_tf_indices: return None
    filtered_tfs = [motif_names[i] for i in active_tf_indices]

    # 2. Calculate TF-Peak PCC
    pcc_long = None
    cells_in_ct = adata_motif_expression.obs[
        adata_motif_expression.obs[cell_type_col].astype(str) == ct_str
    ].index
    common_cells = cells_in_ct.intersection(adata_peak_expression.obs_names)
    
    if len(common_cells) > 5:
        valid_tfs_adata = [t for t in filtered_tfs if t in adata_motif_expression.var_names]
        valid_peaks_adata = [p for p in valid_peaks if p in adata_peak_expression.var_names]
        if valid_tfs_adata and valid_peaks_adata:
            tf_mat = adata_motif_expression[common_cells, valid_tfs_adata].X
            peak_mat = adata_peak_expression[common_cells, valid_peaks_adata].X
            pcc_matrix = fast_pearson_corr(peak_mat, tf_mat)
            pcc_df = pd.DataFrame(pcc_matrix, index=valid_peaks_adata, columns=valid_tfs_adata)
            pcc_df.index.name = 'peak'
            pcc_long = pcc_df.reset_index().melt(id_vars='peak', var_name='TF', value_name='tf_peak_pcc')

    # 3. Extract Binding (TF-RE binding affinity)
    pm_matrix = adata_pm.X
    if not sparse.issparse(pm_matrix): pm_matrix = sparse.csr_matrix(pm_matrix)
    
    sub_pm_matrix = pm_matrix[valid_peak_indices, :]
    peak_tf_binding = sub_pm_matrix[:, active_tf_indices]
    coo = peak_tf_binding.tocoo()
    peak_tf_long = pd.DataFrame({
        'peak': [valid_peaks[r] for r in coo.row],
        'TF': [filtered_tfs[c] for c in coo.col],
        'tf_binding': coo.data
    })
    
    # 4. Merge Components to Calculate Potential
    tf_peak_df = pd.merge(peak_tf_long, ct_tf_df[['TF', 'TF_expression']], on='TF', how='inner')
    tf_peak_df = pd.merge(tf_peak_df, ct_peak_exp, on='peak', how='left')
    tf_peak_df['peak_expression'] = tf_peak_df['peak_expression'].fillna(0)
    
    if pcc_long is not None:
        tf_peak_df = pd.merge(tf_peak_df, pcc_long, on=['peak', 'TF'], how='left')
    else:
        tf_peak_df['tf_peak_pcc'] = 0
        
    tf_peak_df['tf_peak_pcc'] = tf_peak_df['tf_peak_pcc'].fillna(0)
    tf_peak_df['pcc_score'] = (tf_peak_df['tf_peak_pcc']+1)/2
    
    # Calculate Potential
    tf_peak_df['binding_potential'] = (tf_peak_df['tf_binding'] * \
                                      tf_peak_df['TF_expression'] * \
                                      tf_peak_df['peak_expression'] * \
                                      tf_peak_df['pcc_score']
                                      )
                                      
    return tf_peak_df


def build_celltype_specific_tf_gene_network(
    gene_cre_df, 
    adata_pm, 
    TF_ct_exp, 
    peak_ct_exp,
    TG_ct_exp,              # Ensure cell-type average expression matrix for Target Genes is passed
    adata_motif_expression, 
    adata_peak_expression,
    adata_gene_expression,
    cell_type_col='label',
):
    print("Building Peak-TF Index...")
    peak_names = adata_pm.obs_names.values
    motif_names = adata_pm.var_names.values
    peak_to_idx = {peak: i for i, peak in enumerate(peak_names)}
    motif_to_idx = {motif: i for i, motif in enumerate(motif_names)}
    
    results = []
    all_merged_dfs = [] 
    celltypes = TF_ct_exp.index.tolist()
    
    for ct in tqdm(celltypes, desc="Processing Cell Types"):
        ct_str = str(ct)
        
        # --- Prepare Basic Data ---
        
        # 1. Peak Average Expression
        if ct not in peak_ct_exp.index:
            if ct_str in peak_ct_exp.index: ct_peak_series = peak_ct_exp.loc[ct_str]
            else: continue
        else: ct_peak_series = peak_ct_exp.loc[ct]
        ct_peak_exp = ct_peak_series.reset_index()
        ct_peak_exp.columns = ['peak', 'peak_expression']

        # 2. TF Average Expression
        if ct not in TF_ct_exp.index:
            if ct_str in TF_ct_exp.index: ct_tf_series = TF_ct_exp.loc[ct_str]
            else: continue
        else: ct_tf_series = TF_ct_exp.loc[ct]
        ct_tf_df = ct_tf_series.reset_index()
        ct_tf_df.columns = ['TF', 'TF_expression']

        # 3. Target Gene Average Expression
        if ct not in TG_ct_exp.index:
            if ct_str in TG_ct_exp.index: ct_gene_series = TG_ct_exp.loc[ct_str]
            else: continue 
        else: ct_gene_series = TG_ct_exp.loc[ct]
        ct_gene_df = ct_gene_series.reset_index()
        ct_gene_df.columns = ['gene', 'gene_expression']

        # 4. CRE Data
        ct_cre_df = gene_cre_df[gene_cre_df['celltype'].astype(str) == ct_str] 
        if ct_tf_df.empty or ct_cre_df.empty: continue

        # ==========================================================
        # Part 1: Calculate TF-RE Binding Potential
        # ==========================================================
        tf_peak_potential_df = calculate_tf_re_binding_potential(
            ct_str, ct_cre_df, ct_tf_df, ct_peak_exp, 
            adata_pm, adata_motif_expression, adata_peak_expression,
            peak_to_idx, motif_to_idx, peak_names, motif_names, cell_type_col
        )
        
        if tf_peak_potential_df is None or tf_peak_potential_df.empty:
            continue

        # ==========================================================
        # Part 2: TF-TG Network Construction (Binding Potential + Cis-Score)
        # ==========================================================
        
        # 1. Merge
        merged_df = pd.merge(ct_cre_df[['gene', 'peak', 'score']], tf_peak_potential_df, on='peak', how='inner')
        
        # Calculate Intensity
        merged_df['intensity'] = (
                merged_df['binding_potential']
                + merged_df['score']
                - (merged_df['binding_potential'] - merged_df['score']).abs() / (2 ** 0.5)
            )
        
        # 2. Aggregate to TF-Gene
        summary_df = merged_df.groupby(['TF', 'gene']).agg(
            regulatory_score=('intensity', 'sum'),
            n_peaks=('peak', 'count') 
        ).reset_index()

        # ==========================================================
        # Part 3: Multiply by TF-TG PCC Similarity
        # ==========================================================
        if not summary_df.empty:
            # 1. Find cells under this cell type
            cells_idx = adata_motif_expression.obs[
                adata_motif_expression.obs[cell_type_col].astype(str) == ct_str
            ].index
            common_cells = cells_idx.intersection(adata_gene_expression.obs_names)
            
            if len(common_cells) > 10:
                # 2. Prepare matrices
                valid_tfs = [t for t in summary_df['TF'].unique() if t in adata_motif_expression.var_names]
                valid_genes = [g for g in summary_df['gene'].unique() if g in adata_gene_expression.var_names]
                
                if valid_tfs and valid_genes:
                    tf_mat = adata_motif_expression[common_cells, valid_tfs].X
                    gene_mat = adata_gene_expression[common_cells, valid_genes].X
                    
                    # 3. Calculate PCC (Gene x TF)
                    pcc_matrix = fast_pearson_corr(gene_mat, tf_mat)
                    
                    # 4. Build mapping table
                    pcc_df = pd.DataFrame(pcc_matrix, index=valid_genes, columns=valid_tfs)
                    pcc_long = pcc_df.reset_index().melt(id_vars='index', var_name='TF', value_name='tf_gene_pcc')
                    pcc_long.rename(columns={'index': 'gene'}, inplace=True)
                    
                    # 5. Merge and calculate
                    summary_df = pd.merge(summary_df, pcc_long, on=['TF', 'gene'], how='left')
                    summary_df['tf_gene_pcc'] = summary_df['tf_gene_pcc'].fillna(0)
                    
                    # Map PCC from [-1, 1] to [0, 1]
                    summary_df['tf_gene_pcc_score'] = (summary_df['tf_gene_pcc'] + 1) / 2
    
                    # Multiply PCC weight
                    summary_df['regulatory_score'] = summary_df['regulatory_score'] * summary_df['tf_gene_pcc_score']
            else:
                summary_df['tf_gene_pcc_score'] = 0.5 # Default neutral weight

        # ==========================================================
        # Part 4: Multiply by Target Gene Average Expression
        # ==========================================================
        if not summary_df.empty:
            # Merge Gene Expression
            summary_df = pd.merge(summary_df, ct_gene_df, on='gene', how='left')
            summary_df['gene_expression'] = summary_df['gene_expression'].fillna(0)
            
            # Multiply by gene expression
            summary_df['regulatory_score'] = summary_df['regulatory_score'] * summary_df['gene_expression']

        summary_df['celltype'] = ct
        results.append(summary_df)
        all_merged_dfs.append(merged_df)

    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df = final_df.sort_values(by='TF')
        final_df = final_df[final_df['regulatory_score'] > 0]
        final_merged_df = pd.concat(all_merged_dfs, ignore_index=True) if all_merged_dfs else pd.DataFrame()
        return final_df, final_merged_df
    else:
        return pd.DataFrame(), pd.DataFrame()
    
    
def trans_GRN(
    adata_C,
    adata_G,
    adata_P,
    adata_M,
    adata_TF_expression,
    gene_CRE_df,
    adata_PM,
    emb_cluster_key="celltype",
    min_cell_count=10,
    threshold=0.95,
    min_score_floor=0.05,
    max_score_floor=0.5,
    output_dir=None,
):
    print("Starting GRN Inference Pipeline...")
    
    # 1. Filter Cell Types
    print("Filtering cell types...")
    adata_C.obs[emb_cluster_key] = adata_C.obs[emb_cluster_key].astype(str)
    celltype_counts = adata_C.obs[emb_cluster_key].value_counts()
    keep_celltypes = celltype_counts[celltype_counts >= min_cell_count].index.tolist()
    removed_celltypes = set(celltype_counts.index) - set(keep_celltypes)
    print(f"Keeping cell types ({len(keep_celltypes)}): {keep_celltypes}")
    if removed_celltypes:
        print(f"Removed cell types (<{min_cell_count} cells): {removed_celltypes}")
        
    adata_C = adata_C[adata_C.obs[emb_cluster_key].isin(keep_celltypes)].copy()

    common_cells = adata_C.obs_names[
        adata_C.obs_names.isin(adata_TF_expression.obs_names)
    ]
    if len(common_cells) == 0:
        raise ValueError("No shared cells between adata_C and adata_TF_expression.")
    if len(common_cells) != adata_C.n_obs:
        print(
            "Warning: adata_C and adata_TF_expression cells do not fully match. "
            "Aligning inputs."
        )

    Cell_emd = adata_C[common_cells].copy()
    Gene_emd = adata_G.copy()
    Peak_emd = adata_P.copy()
    Motif_emd = adata_M.copy()
    adata_motif_expression = adata_TF_expression[common_cells, :].copy()
    adata_motif_expression.obs["label"] = Cell_emd.obs[emb_cluster_key].astype(str).values
    if "spatial" in Cell_emd.obsm.keys():
        adata_motif_expression.obsm["spatial"] = Cell_emd.obsm["spatial"]

    # 2. Reconstruct Expression Matrices
    print("Reconstructing expression matrices from embeddings...")
    Peak_expression_matrix = Cell_emd.X @ Peak_emd.X.T  # Cell_emd * Peak_emd.T
    TG_expression_matrix = Cell_emd.X @ Gene_emd.X.T  # Cell_emd * Gene_emd.T

    # Store results in new AnnData objects
    adata_peak_expression = AnnData(X=Peak_expression_matrix)
    adata_peak_expression.obs_names = Cell_emd.obs_names
    adata_peak_expression.var_names = Peak_emd.obs_names
    adata_peak_expression.obs["label"] = list(Cell_emd.obs[emb_cluster_key])
    adata_peak_expression.obs["label"] = adata_peak_expression.obs["label"].astype(str)
    if "spatial" in Cell_emd.obsm.keys():
        adata_peak_expression.obsm["spatial"] = Cell_emd.obsm["spatial"]

    adata_TG_expression = AnnData(X=TG_expression_matrix)
    adata_TG_expression.obs_names = Cell_emd.obs_names
    adata_TG_expression.var_names = Gene_emd.obs_names
    adata_TG_expression.obs["label"] = list(Cell_emd.obs[emb_cluster_key])
    adata_TG_expression.obs["label"] = adata_TG_expression.obs["label"].astype(str)
    if "spatial" in Cell_emd.obsm.keys():
        adata_TG_expression.obsm["spatial"] = Cell_emd.obsm["spatial"]

    adata_motif_expression = adata_motif_expression[adata_motif_expression.obs['label'].isin(keep_celltypes)].copy()
    adata_TG_expression = adata_TG_expression[adata_TG_expression.obs['label'].isin(keep_celltypes)].copy()
    adata_peak_expression = adata_peak_expression[adata_peak_expression.obs['label'].isin(keep_celltypes)].copy()

    adata_PM = adata_PM.copy()
    pm_tf_names = pd.Index(adata_PM.var_names.astype(str))
    expression_tf_names = pd.Index(adata_motif_expression.var_names.astype(str))
    motif_tf_names = pd.Index(Motif_emd.obs_names.astype(str))
    common_tfs = pm_tf_names.intersection(expression_tf_names)
    if len(common_tfs) == 0:
        prefixed_pm_names = pd.Index(
            [
                name if name.startswith("M_") else f"M_{name}"
                for name in pm_tf_names
            ]
        )
        if len(prefixed_pm_names.intersection(expression_tf_names)) > 0:
            adata_PM.var_names = prefixed_pm_names
            pm_tf_names = prefixed_pm_names
            common_tfs = pm_tf_names.intersection(expression_tf_names)

    if len(motif_tf_names.intersection(expression_tf_names)) == 0:
        prefixed_motif_names = pd.Index(
            [
                name if name.startswith("M_") else f"M_{name}"
                for name in motif_tf_names
            ]
        )
        if len(prefixed_motif_names.intersection(expression_tf_names)) > 0:
            Motif_emd.obs_names = prefixed_motif_names
            motif_tf_names = prefixed_motif_names

    common_tfs = common_tfs.intersection(motif_tf_names)
    if len(common_tfs) == 0:
        raise ValueError(
            "No shared TF names among adata_PM, adata_TF_expression, and adata_M."
        )
    if (
        len(common_tfs) != adata_motif_expression.n_vars
        or len(common_tfs) != adata_PM.n_vars
        or len(common_tfs) != Motif_emd.n_obs
    ):
        print(
            "Warning: TF names do not fully match across adata_PM, "
            "adata_TF_expression, and adata_M. Aligning inputs."
        )
        adata_motif_expression = adata_motif_expression[:, common_tfs].copy()
        adata_PM = adata_PM[:, common_tfs].copy()
        Motif_emd = Motif_emd[common_tfs, :].copy()
    
    # 3. Filter CREs (FGOT Result)
    print("Filtering CREs based on cell-type specificity...")
    gene_CRE_df["celltype"] = gene_CRE_df["celltype"].astype(str)
    gene_CRE_df = gene_CRE_df[gene_CRE_df['celltype'].isin(keep_celltypes)]
    celltype_cre_dict, filtered_gene_CRE_df = get_celltype_cres(
        gene_CRE_df,
        threshold=threshold,
        min_score_floor=min_score_floor,
        max_score_floor=max_score_floor
    )

    # 4. Calculate Cell-Type Average Expressions
    print("Calculating cell-type specific average expressions...")
    TF_ct_exp = celltype_exp(adata_motif_expression)
    peak_ct_exp = celltype_exp(adata_peak_expression)
    TG_ct_exp = celltype_exp(adata_TG_expression)

    # 5. Calculate TF Binding Strength
    print("Calculating TF binding strength on peaks...")
    adata_PM = TF_binding_peak_strenth(adata_PM, Peak_emd, Motif_emd)

    # 6. Build TF-Gene Network
    print("Building TF-Gene Regulatory Network...")
    tf_gene_network, merged_df = build_celltype_specific_tf_gene_network(
        filtered_gene_CRE_df, 
        adata_PM, 
        TF_ct_exp=TF_ct_exp,
        peak_ct_exp=peak_ct_exp,
        TG_ct_exp=TG_ct_exp,
        adata_motif_expression=adata_motif_expression,
        adata_peak_expression=adata_peak_expression,
        adata_gene_expression = adata_TG_expression,
    )
    
    # Select relevant columns
    tf_gene_network = tf_gene_network[["TF","gene","regulatory_score","celltype"]]
    
    if output_dir is not None:
        print(f"Saving results to {output_dir}...")
        tf_gene_network.to_csv(f"{output_dir}/tf_gene_network.csv", index=False)

    print("GRN Inference Completed.")
    return tf_gene_network
