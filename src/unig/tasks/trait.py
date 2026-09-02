from pathlib import Path
from typing import Literal
from bisect import bisect_left, bisect_right
import json
import re

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats import gmean, rankdata
from sklearn.neighbors import NearestNeighbors

from .liftover_peaks import ensure_peak_lift_mapping, lift_peak_colon_to_regions


def find_neighbors(coordinates, n_neighbors):
    """Find nearest spatial neighbors from coordinate values."""
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(coordinates)
    _, indices = neighbors.kneighbors(coordinates, return_distance=True)
    return indices


def build_spatial_net_dict(adata, annotation_col=None, n_neighbors=101):
    """Build a spatial neighbor dictionary, optionally within each annotation."""
    coordinates = adata.obsm["spatial"]
    spatial_net = {}
    global_indices = np.arange(adata.n_obs)

    if annotation_col is None:
        indices = find_neighbors(coordinates, n_neighbors)
        return {cell_idx: neighbors for cell_idx, neighbors in enumerate(indices)}

    for annotation in adata.obs[annotation_col].unique():
        mask = adata.obs[annotation_col] == annotation
        annotation_indices = global_indices[mask]
        annotation_coordinates = coordinates[mask]
        k = min(len(annotation_indices), n_neighbors)

        neighbors = NearestNeighbors(n_neighbors=k).fit(annotation_coordinates)
        _, local_indices = neighbors.kneighbors(annotation_coordinates)

        for local_idx, local_neighbors in enumerate(local_indices):
            global_idx = annotation_indices[local_idx]
            spatial_net[global_idx] = annotation_indices[local_neighbors]

    return spatial_net


def compute_regional_score(
    cell_idx,
    spatial_net_dict,
    latent_emb,
    ranks,
    frac_whole,
    expression_bool,
    n_latent_neighbors=21,
):
    """Compute gene marker scores for one spatial region."""
    spatial_neighbors = spatial_net_dict.get(cell_idx, [])
    if len(spatial_neighbors) == 0:
        return np.zeros(ranks.shape[1])

    cell_latent = latent_emb[cell_idx, :].reshape(1, -1)
    neighbor_latents = latent_emb[spatial_neighbors, :]
    similarities = (cell_latent @ neighbor_latents.T).ravel()

    k = min(len(similarities), n_latent_neighbors)
    top_neighbor_idx = np.argsort(-similarities)[:k]
    selected_neighbors = spatial_neighbors[top_neighbor_idx]

    ranks_region = ranks[selected_neighbors, :]
    regional_rank = gmean(ranks_region, axis=0)
    regional_rank[regional_rank <= 1] = 0

    frac_focal = expression_bool[selected_neighbors, :].mean(axis=0)
    frac_ratio = frac_focal / frac_whole
    frac_ratio[frac_ratio <= 1] = 0
    frac_ratio[frac_ratio > 1] = 1

    marker_score = np.exp(regional_rank * frac_ratio) - 1
    return marker_score.astype(np.float16, copy=False)


def run_cal_GSS(
    adata_C,
    adata_TG_expression,
    workdir,
    sample_name="GBM",
    species="hg38",
    homolog_file=None,
    n_spatial_neighbors=101,
    n_latent_neighbors=21,
    annotation_col="label",
):
    """Calculate gsMap-style gene specificity scores using UniG embeddings."""
    print("Preparing data...")
    adata = adata_TG_expression.copy()
    sc.pp.filter_genes(adata, min_cells=1)
    print(f"Expression source: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"Embedding source: {adata_C.n_obs} cells x {adata_C.n_vars} latent dims")

    adata = _convert_gene_symbols_if_needed(adata, species, homolog_file)
    adata = _attach_unig_latent_embedding(adata, adata_C)

    print(f"Aligned cells: {adata.n_obs}; genes retained for GSS: {adata.n_vars}")
    print(f"latent_embedding shape: {adata.obsm['latent_embedding'].shape}")

    if "spatial" not in adata.obsm:
        raise ValueError("Spatial coordinates not found in adata.obsm['spatial'].")

    print("Building spatial graph...")
    spatial_net_dict = build_spatial_net_dict(
        adata,
        annotation_col=annotation_col,
        n_neighbors=n_spatial_neighbors,
    )

    print("Computing global ranks and expression background...")
    expression = _to_dense(adata.X)
    n_cells, n_genes = expression.shape

    ranks = np.zeros((n_cells, n_genes), dtype=np.float32)
    for cell_idx in _tqdm_range(n_cells, desc="Ranking cells"):
        ranks[cell_idx, :] = rankdata(expression[cell_idx, :], method="average")

    global_rank = gmean(ranks, axis=0)
    expression_bool = expression.astype(bool)
    frac_whole = expression_bool.mean(axis=0) + 1e-12
    ranks = ranks / (global_rank + 1e-12)

    print("Calculating marker scores...")
    marker_scores = np.zeros((n_cells, n_genes), dtype=np.float32)
    latent_emb = adata.obsm["latent_embedding"]
    if latent_emb.shape[0] != n_cells:
        raise ValueError(
            f"latent rows ({latent_emb.shape[0]}) != expression cells ({n_cells})."
        )

    for cell_idx in _tqdm_range(n_cells, desc="Scoring"):
        marker_scores[cell_idx, :] = compute_regional_score(
            cell_idx,
            spatial_net_dict,
            latent_emb,
            ranks,
            frac_whole,
            expression_bool,
            n_latent_neighbors=n_latent_neighbors,
        )

    workdir = Path(workdir)
    latent_to_gene_dir = workdir / sample_name / "latent_to_gene"
    find_latent_dir = workdir / sample_name / "find_latent_representations"
    latent_to_gene_dir.mkdir(parents=True, exist_ok=True)
    find_latent_dir.mkdir(parents=True, exist_ok=True)

    gene_names = pd.Index(adata.var_names).astype(str)
    non_mito_mask = ~(gene_names.str.startswith("MT-") | gene_names.str.startswith("mt-"))
    if np.sum(~non_mito_mask) > 0:
        print(f"Removed mitochondrial genes: {np.sum(~non_mito_mask)}")

    marker_score_table = pd.DataFrame(
        marker_scores[:, non_mito_mask].T,
        index=gene_names[non_mito_mask].to_numpy(),
        columns=adata.obs_names,
    )
    marker_score_table.index.name = "HUMAN_GENE_SYM"
    marker_score_table.reset_index(inplace=True)

    feather_path = latent_to_gene_dir / f"{sample_name}_gene_marker_score.feather"
    h5ad_path = find_latent_dir / f"{sample_name}_add_latent.h5ad"
    marker_score_table.to_feather(feather_path)
    adata.write(h5ad_path)

    print(f"Feather saved to: {feather_path}")
    print(f"H5AD saved to: {h5ad_path}")
    return marker_score_table, adata



def patch_plink_next_snps():
    """Patch gsMap PlinkBEDFile.nextSNPs for newer bitarray decode behavior."""
    from gsMap.utils.generate_r2_matrix import PlinkBEDFile, normalized_snps

    def _patched_nextSNPs(self, b, minorRef=None):
        try:
            b = int(b)
            if b <= 0:
                raise ValueError("b must be > 0")
        except TypeError as exc:
            raise TypeError("b must be an integer") from exc

        if self._currentSNP + b > self.m:
            message = "{b} SNPs requested, {k} SNPs remain"
            raise ValueError(message.format(b=b, k=(self.m - self._currentSNP)))

        current = self._currentSNP
        n_samples = self.n
        n_bytes_per_snp = self.nru
        raw_slice = self.geno[
            2 * current * n_bytes_per_snp : 2 * (current + b) * n_bytes_per_snp
        ]

        decoded = raw_slice.decode(self._bedcode)
        if not isinstance(decoded, (list, tuple, np.ndarray)):
            decoded = list(decoded)

        values = np.array(decoded, dtype="float32").reshape((b, n_bytes_per_snp)).T
        values = values[0:n_samples, :]
        snps = normalized_snps(values, b, minorRef, self.freq, self._currentSNP)

        self._currentSNP += b
        return snps

    PlinkBEDFile.nextSNPs = _patched_nextSNPs
    print("[PATCH] PlinkBEDFile.nextSNPs compatibility patch applied.")


def generate_ldscore(
    workdir,
    sample_name,
    abc_enhancer_bed,
    bfile_root,
    keep_snp_root,
    gtf_annotation_file,
    chrom="all",
    gene_window_size=50000,
    spots_per_chunk=3000,
    ld_wind=1,
    ld_unit="CM",
):
    """Run gsMap LD score generation with enhancer annotations."""
    from gsMap.config import GenerateLDScoreConfig
    from gsMap.generate_ldscore import run_generate_ldscore

    abc_enhancer_bed = Path(abc_enhancer_bed)
    if not abc_enhancer_bed.exists():
        raise FileNotFoundError(f"ABC enhancer bed not found: {abc_enhancer_bed}")

    config = GenerateLDScoreConfig(
        workdir=str(workdir),
        sample_name=sample_name,
        chrom=chrom,
        bfile_root=bfile_root,
        keep_snp_root=keep_snp_root,
        gtf_annotation_file=gtf_annotation_file,
        gene_window_size=gene_window_size,
        enhancer_annotation_file=str(abc_enhancer_bed),
        gene_window_enhancer_priority="gene_window_first",
        snp_multiple_enhancer_strategy="max_mkscore",
        spots_per_chunk=spots_per_chunk,
        ld_wind=ld_wind,
        ld_unit=ld_unit,
        additional_baseline_annotation=None,
    )

    print("[RUN] generate_ldscore")
    run_generate_ldscore(config)
    print(f"[DONE] LD score generated at: {config.ldscore_save_dir}")
    return config


def batch_spatial_ldsc(
    trait_sumstats,
    workdir,
    sample_name,
    w_file,
    num_processes=16,
    skip_existing=True,
):
    """Run gsMap spatial LDSC for a set of traits."""
    from gsMap.config import SpatialLDSCConfig
    from gsMap.spatial_ldsc_multiple_sumstats import run_spatial_ldsc

    workdir = Path(workdir)
    ldsc_dir = workdir / sample_name / "spatial_ldsc"
    ldsc_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    for trait_name, sumstats_file in trait_sumstats.items():
        out_file = ldsc_dir / f"{sample_name}_{trait_name}.csv.gz"
        outputs[trait_name] = out_file

        if skip_existing and out_file.exists():
            print(f"[SKIP] Spatial LDSC already exists for {trait_name}: {out_file}")
            continue

        print(f"[RUN] Spatial LDSC for {trait_name}")
        config = SpatialLDSCConfig(
            workdir=str(workdir),
            sample_name=sample_name,
            trait_name=trait_name,
            sumstats_file=sumstats_file,
            w_file=w_file,
            num_processes=num_processes,
            ldscore_save_format="feather",
        )
        run_spatial_ldsc(config)

        if not out_file.exists():
            raise FileNotFoundError(
                f"Spatial LDSC finished but expected output was not found: {out_file}"
            )
        print(f"[DONE] Spatial LDSC for {trait_name}: {out_file}")

    return outputs


def batch_cauchy_combination(
    trait_sumstats,
    workdir,
    sample_name,
    annotation_col="label",
    skip_existing=True,
):
    """Run gsMap Cauchy combination for each trait and save a batch summary."""
    from gsMap.cauchy_combination_test import run_Cauchy_combination
    from gsMap.config import CauchyCombinationConfig

    workdir = Path(workdir)
    latent_h5ad_path = (
        workdir / sample_name / "find_latent_representations" / f"{sample_name}_add_latent.h5ad"
    )
    if not latent_h5ad_path.exists():
        raise FileNotFoundError(f"h5ad not found: {latent_h5ad_path}.")

    ldsc_dir = workdir / sample_name / "spatial_ldsc"
    cauchy_dir = workdir / sample_name / "cauchy_combination"
    cauchy_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    tables = []

    for trait_name in trait_sumstats:
        ldsc_file = ldsc_dir / f"{sample_name}_{trait_name}.csv.gz"
        if not ldsc_file.exists():
            raise FileNotFoundError(f"Spatial LDSC output missing for {trait_name}: {ldsc_file}")

        out_file = cauchy_dir / f"{sample_name}_{trait_name}.Cauchy.csv.gz"
        outputs[trait_name] = out_file

        if skip_existing and out_file.exists():
            print(f"[SKIP] Cauchy result already exists for {trait_name}: {out_file}")
            result = pd.read_csv(out_file)
        else:
            print(f"[RUN] Cauchy combination for {trait_name}")
            config = CauchyCombinationConfig(
                workdir=str(workdir),
                sample_name=sample_name,
                trait_name=trait_name,
                annotation=annotation_col,
            )
            result = run_Cauchy_combination(config)
            if not out_file.exists():
                raise FileNotFoundError(
                    f"Cauchy finished but expected output was not found: {out_file}"
                )

        result = result.copy()
        result.insert(0, "trait", trait_name)
        tables.append(result)

    summary = pd.concat(tables, ignore_index=True)
    summary_path = cauchy_dir / f"{sample_name}_batch_Cauchy_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[DONE] Batch Cauchy summary saved to: {summary_path}")
    return summary, outputs


def safe_run_visualize(config, output_vis_dir):
    """Run gsMap visualization and save HTML/CSV outputs."""
    from gsMap.visualize import draw_scatter, load_ldsc, load_st_coord

    ldsc = load_ldsc(Path(config.ldsc_save_dir) / f"{config.sample_name}_{config.trait_name}.csv.gz")
    adata = sc.read_h5ad(config.hdf5_with_latent_path)
    space_coord_concat = load_st_coord(adata, ldsc, annotation=config.annotation)
    fig = draw_scatter(
        space_coord_concat,
        title=config.fig_title,
        fig_style=config.fig_style,
        point_size=config.point_size,
        width=config.fig_width,
        height=config.fig_height,
        annotation=config.annotation,
    )

    output_vis_dir = Path(output_vis_dir)
    output_vis_dir.mkdir(parents=True, exist_ok=True)
    html_file = output_vis_dir / f"{config.sample_name}_{config.trait_name}.html"
    csv_file = output_vis_dir / f"{config.sample_name}_{config.trait_name}.csv"
    fig.write_html(str(html_file))
    space_coord_concat.to_csv(str(csv_file))
    return html_file, csv_file


def batch_visualization(
    trait_sumstats,
    workdir,
    sample_name,
    annotation_col="label",
    skip_existing=True,
    point_size=5,
    fig_width=900,
    fig_height=700,
):
    """Generate gsMap HTML/CSV visualizations for each trait."""
    from gsMap.config import VisualizeConfig

    workdir = Path(workdir)
    ldsc_dir = workdir / sample_name / "spatial_ldsc"
    visualize_dir = workdir / sample_name / "visualize"
    outputs = {}

    for trait_name in trait_sumstats:
        ldsc_file = ldsc_dir / f"{sample_name}_{trait_name}.csv.gz"
        if not ldsc_file.exists():
            raise FileNotFoundError(f"Spatial LDSC output missing for {trait_name}: {ldsc_file}")

        html_file = visualize_dir / f"{sample_name}_{trait_name}.html"
        csv_file = visualize_dir / f"{sample_name}_{trait_name}.csv"
        outputs[trait_name] = {"html": html_file, "csv": csv_file}

        if skip_existing and html_file.exists() and csv_file.exists():
            print(f"[SKIP] Visualization already exists for {trait_name}: {html_file}")
            continue

        print(f"[RUN] Visualization for {trait_name}")
        config = VisualizeConfig(
            workdir=str(workdir),
            sample_name=sample_name,
            trait_name=trait_name,
            annotation=annotation_col,
            fig_title=f"{sample_name} Spatial LDSC ({trait_name})",
            point_size=point_size,
            fig_width=fig_width,
            fig_height=fig_height,
        )
        outputs[trait_name]["html"], outputs[trait_name]["csv"] = safe_run_visualize(
            config,
            visualize_dir,
        )

    return outputs


def plot_trait_spatial_score(
    trait_name,
    workdir,
    sample_name,
    output_path=None,
    color_col="logp",
    size=75,
    cmap="coolwarm",
    vmin=0,
    vmax=16,
):
    """Plot one trait from a gsMap visualization CSV on spatial coordinates."""
    import matplotlib.pyplot as plt

    workdir = Path(workdir)
    latent_h5ad_path = (
        workdir / sample_name / "find_latent_representations" / f"{sample_name}_add_latent.h5ad"
    )
    csv_file = workdir / sample_name / "visualize" / f"{sample_name}_{trait_name}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Visualization CSV not found for {trait_name}: {csv_file}")

    adata = sc.read_h5ad(latent_h5ad_path)
    result = pd.read_csv(csv_file, index_col=0)
    adata.obs[color_col] = result[color_col]

    fig = sc.pl.embedding(
        adata,
        basis="spatial",
        color=[color_col],
        size=size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title=f"{sample_name} Spatial LDSC ({trait_name})",
        show=False,
        return_fig=True,
    )
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[-1]
        cbar_ax.set_yticks([vmin, (vmin + vmax) / 2, vmax])
        cbar_ax.set_yticklabels([str(vmin), str((vmin + vmax) / 2), str(vmax)])

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    return fig, adata


def plot_cauchy_heatmap(
    workdir,
    sample_name,
    trait_order=None,
    cluster_order=None,
    p_col="p_cauchy",
    cluster_col="annotation",
    cmap="Reds",
    vmax=20.0,
):
    """Create a cluster-by-trait heatmap from Cauchy combination results."""
    import matplotlib.pyplot as plt

    workdir = Path(workdir)
    cauchy_dir = workdir / sample_name / "cauchy_combination"
    output_prefix = cauchy_dir / f"{sample_name}_cluster_trait_cauchy_neglog10P_heatmap"

    if trait_order is None:
        cauchy_files = sorted(cauchy_dir.glob(f"{sample_name}_*.Cauchy.csv.gz"))
    else:
        cauchy_files = [cauchy_dir / f"{sample_name}_{trait}.Cauchy.csv.gz" for trait in trait_order]
        missing = [path for path in cauchy_files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing Cauchy files:\n" + "\n".join(str(path) for path in missing)
            )

    rows = []
    for path in cauchy_files:
        trait = _trait_name_from_cauchy_path(path, sample_name)
        table = pd.read_csv(path)
        missing_cols = {cluster_col, p_col} - set(table.columns)
        if missing_cols:
            raise ValueError(f"{path} missing columns: {sorted(missing_cols)}")

        for _, row in table.iterrows():
            p_value = float(row[p_col])
            rows.append(
                {
                    "trait": trait,
                    "cluster": _format_cluster(row[cluster_col]),
                    "p_cauchy": p_value,
                    "neg_log10_p": -np.log10(np.clip(p_value, np.nextafter(0, 1), 1.0)),
                }
            )

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise ValueError(f"No {sample_name}_*.Cauchy.csv.gz files found in {cauchy_dir}.")

    if cluster_order is None:
        cluster_order = sorted(long_df["cluster"].astype(str).unique(), key=lambda x: int(float(x)))
    else:
        available_clusters = set(long_df["cluster"].astype(str).unique())
        cluster_order = [cluster for cluster in cluster_order if cluster in available_clusters]
        cluster_order += sorted(
            available_clusters - set(cluster_order),
            key=lambda x: int(float(x)),
        )

    available_traits = list(long_df["trait"].unique())
    if trait_order is None:
        trait_order = sorted(available_traits)
    else:
        trait_order = [trait for trait in trait_order if trait in available_traits]

    matrix = long_df.pivot(index="cluster", columns="trait", values="neg_log10_p")
    matrix = matrix.loc[cluster_order, trait_order]

    long_df.to_csv(f"{output_prefix}.long.csv", index=False)
    matrix.to_csv(f"{output_prefix}.matrix.csv")

    n_clusters, n_traits = matrix.shape
    cell_size = 0.48
    fig_width = max(7, n_traits * cell_size + 3.0)
    fig_height = max(4, n_clusters * cell_size + 1.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_values = matrix.to_numpy(dtype=float)
    finite_values = plot_values[np.isfinite(plot_values)]
    if finite_values.size == 0:
        raise ValueError("Heatmap matrix has no finite values to plot.")

    im = ax.imshow(plot_values, cmap=cmap, aspect="equal", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(n_traits))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(n_clusters))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Trait")
    ax.set_ylabel("Cluster")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xticks(np.arange(-0.5, n_traits, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_clusters, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_ticks([0, vmax / 2, vmax])
    cbar.set_ticklabels([f"{tick:.2f}" for tick in [0, vmax / 2, vmax]])
    cbar.set_label("-log10(P)")

    ax.set_title(f"{sample_name} cluster x trait Cauchy enrichment")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.png", dpi=300)
    fig.savefig(f"{output_prefix}.pdf")
    plt.show()

    print(f"clusters={matrix.shape[0]}")
    print(f"traits={matrix.shape[1]}")
    print(f"matrix_csv={output_prefix}.matrix.csv")
    print(f"long_csv={output_prefix}.long.csv")
    print(f"png={output_prefix}.png")
    print(f"pdf={output_prefix}.pdf")
    return matrix, long_df, fig


def normalize_tf_name(value):
    """Normalize TF names used by regulatory tables."""
    return str(value).strip()


def normalize_gene_name(value):
    """Normalize gene names used by regulatory tables."""
    return str(value).strip()


def parse_region(region):
    """Parse chr-start-end or chr:start-end into chrom, start, and end."""
    region = str(region).strip()
    match = re.match(r"^(chr[^:\-]+)[:-](\d+)-(\d+)$", region)
    if not match:
        raise ValueError(f"Cannot parse region: {region}")

    chrom, start, end = match.group(1), int(match.group(2)), int(match.group(3))
    if end < start:
        start, end = end, start
    return chrom, start, end


def region_to_grs_region(region):
    """Convert a genomic region to chr-start-end format."""
    chrom, start, end = parse_region(region)
    return f"{chrom}-{start}-{end}"


def safe_neglog10_pval(p_values):
    """Convert p-values to finite -log10(p) scores."""
    p_values = pd.to_numeric(p_values, errors="coerce")
    p_values = p_values.clip(lower=np.finfo(float).tiny, upper=1.0)
    return -np.log10(p_values)



def peak_colon_to_regions(peak: str) -> str:
    """chr15:74375198-74375698 -> chr15-74375198-74375698."""
    peak = str(peak).strip()
    if peak.startswith("chr"):
        chrom, rest = peak.split(":", 1)
    else:
        chrom, rest = peak.split(":", 1)
        chrom = f"chr{chrom}"
    start, end = rest.split("-", 1)
    return f"{chrom}-{start}-{end}"


def weight_to_pval(weight: float, peak_score: float, eps: float = 1e-6) -> float:
    """Map TF weight × peak score to pseudo Pval (legacy)."""
    strength = float(weight) * float(peak_score)
    return float(np.clip(1.0 / (1.0 + strength), eps, 1.0))


def peak_score_to_pval(score: float, eps: float = 1e-300) -> float:
    """Map FGOT score to pseudo Pval: P = 10^(-exp(score)); Strength = -log10(Pval)."""
    score = max(float(score), 0.0)
    pval = 10.0 ** (-np.exp(score))
    return float(np.clip(pval, eps, 1.0))


def _edge_pval(weight: float, peak_score: float, use_peak_score_only: bool, eps: float = 1e-6) -> float:
    if use_peak_score_only:
        return peak_score_to_pval(peak_score, eps=eps)
    return weight_to_pval(weight, peak_score, eps=eps)


def _aggregate_tf_target(grn_ct: pd.DataFrame) -> pd.DataFrame:
    return grn_ct.groupby(["TF", "Target"], as_index=False).agg(
        Regions=("Regions", lambda xs: ";".join(sorted(set(";".join(xs).split(";"))))),
        Pval=("Pval", "min"),
    )


def _filter_grn_by_min_targets(grn: pd.DataFrame, min_targets: int) -> tuple[pd.DataFrame, list[str]]:
    tf_target_counts = grn.groupby("TF")["Target"].nunique()
    valid_tfs = tf_target_counts[tf_target_counts >= min_targets].index
    grn = grn[grn["TF"].isin(valid_tfs)].reset_index(drop=True)
    return grn, sorted(grn["TF"].unique())


def extract_tf_peak_pairs(
    pm_h5ad: str | Path,
    peaks: set[str] | None = None,
    tfs: set[str] | None = None,
    min_tf_binding: float = 0.0,
) -> pd.DataFrame:
    """Extract (peak, TF, tf_binding) from adata_PM; keep pairs with binding > min_tf_binding."""
    import anndata as ad
    from scipy import sparse

    pm = ad.read_h5ad(pm_h5ad)
    obs_names = pm.obs_names.astype(str).tolist()
    var_names = pm.var_names.astype(str).tolist()
    var_set = set(var_names)

    row_idx = list(range(len(obs_names)))
    if peaks is not None:
        peak_set = {str(p) for p in peaks}
        row_idx = [i for i, p in enumerate(obs_names) if p in peak_set]
    if not row_idx:
        return pd.DataFrame(columns=["peak", "TF", "tf_binding"])

    col_map: dict[str, int] = {}
    if tfs is not None:
        for tf in {str(t) for t in tfs}:
            if tf in var_set:
                col_map[tf] = var_names.index(tf)
            elif f"M_{tf}" in var_set:
                col_map[tf] = var_names.index(f"M_{tf}")
        col_idx = sorted(set(col_map.values()))
    else:
        col_map = {v: i for i, v in enumerate(var_names)}
        col_idx = list(range(len(var_names)))

    if not col_idx:
        return pd.DataFrame(columns=["peak", "TF", "tf_binding"])

    X = pm.X[np.array(row_idx)][:, np.array(col_idx)]
    if sparse.issparse(X):
        X = X.tocoo()
    else:
        X = sparse.coo_matrix(np.asarray(X, dtype=float))

    idx_to_peak = [obs_names[i] for i in row_idx]
    col_to_tf = {col_idx.index(j): tf for tf, j in col_map.items()}

    rows: list[dict] = []
    for r, c, val in zip(X.row, X.col, X.data):
        if float(val) > min_tf_binding:
            rows.append(
                {
                    "peak": idx_to_peak[r],
                    "TF": col_to_tf.get(c, var_names[col_idx[c]]),
                    "tf_binding": float(val),
                }
            )
    return pd.DataFrame(rows)


def _peaks_hg38_to_regions_hg19(
    peaks_hg38: set[str] | list[str],
    mapping: dict[str, str],
) -> tuple[str | None, int]:
    """Map hg38 colon peaks to hg19 trait GRS Regions string; return (regions, n_dropped)."""
    regions: set[str] = set()
    dropped = 0
    for peak in peaks_hg38:
        peak_hg19 = mapping.get(str(peak))
        if not peak_hg19:
            dropped += 1
            continue
        regions.add(lift_peak_colon_to_regions(peak_hg19))
    if not regions:
        return None, dropped
    return ";".join(sorted(regions)), dropped


def build_trait_grn(
    tf_gene_csv: str | Path,
    gene_peak_csv: str | Path,
    min_targets: int = 10,
    valid_genes: set[str] | None = None,
    grn_mode: Literal["global", "per_celltype", "shared_structure"] = "global",
    use_peak_score_only: bool = True,
    pm_h5ad: str | Path | None = None,
    min_tf_binding: float = 0.0,
    lift_mapping_path: str | Path | None = None,
    output_genome: Literal["hg19"] = "hg19",
) -> tuple[pd.DataFrame, list[str], dict, pd.DataFrame]:
    """
    Join TF-gene with gene-peak (hg38); optionally filter by adata_PM TF-peak binding.
    In grn_mode='global', cell-type labels are ignored during the join so all
    clusters contribute to one global regulon/peak-gene strength table.
    Output Regions are lifted to hg19 for downstream SNP-to-peak scoring.

    Returns (grn_df, tf_names, meta, grn_long).
    """
    tf_gene = pd.read_csv(tf_gene_csv)
    gene_peak = pd.read_csv(gene_peak_csv)

    tf_gene["cell_type"] = tf_gene["cell_type"].astype(str)
    gene_peak["celltype"] = gene_peak["celltype"].astype(str)

    if valid_genes is not None:
        tf_gene = tf_gene[tf_gene["target"].isin(valid_genes)]

    if grn_mode == "global":
        tf_gene_join = (
            tf_gene[["source", "target", "weight"]]
            .groupby(["source", "target"], as_index=False)
            .agg(weight=("weight", "max"))
        )
        gene_peak_join = (
            gene_peak[["gene", "peak", "score"]]
            .groupby(["gene", "peak"], as_index=False)
            .agg(score=("score", "max"))
        )
        merged = tf_gene_join.merge(
            gene_peak_join,
            left_on="target",
            right_on="gene",
            how="inner",
        )
        merged["cell_type"] = "global"
    else:
        merged = tf_gene.merge(
            gene_peak,
            left_on=["target", "cell_type"],
            right_on=["gene", "celltype"],
            how="inner",
        )
    if merged.empty:
        raise ValueError("TF-gene × gene-peak join produced no rows; check inputs and cell_type alignment.")

    n_after_gene_peak = len(merged)
    tf_peak_filter = "none"
    if pm_h5ad is not None:
        peaks_in = set(merged["peak"].astype(str))
        tfs_in = set(merged["source"].astype(str))
        tf_peak = extract_tf_peak_pairs(
            pm_h5ad,
            peaks=peaks_in,
            tfs=tfs_in,
            min_tf_binding=min_tf_binding,
        )
        if tf_peak.empty:
            raise ValueError("TF-peak filter via adata_PM removed all rows; check PM h5ad and TF names.")
        merged = merged.merge(
            tf_peak,
            left_on=["source", "peak"],
            right_on=["TF", "peak"],
            how="inner",
        )
        if merged.empty:
            raise ValueError("TF-gene × gene-peak × TF-peak join produced no rows.")
        tf_peak_filter = "adata_PM"

    n_triplet_rows = len(merged)
    merged["Pval"] = merged.apply(
        lambda r: _edge_pval(r["weight"], r["score"], use_peak_score_only), axis=1
    )

    lift_mapping: dict[str, str] = {}
    n_peaks_dropped_lift = 0
    if output_genome == "hg19":
        if lift_mapping_path is None:
            raise ValueError("lift_mapping_path required when output_genome='hg19'")
        lift_mapping = ensure_peak_lift_mapping(
            set(merged["peak"].astype(str)),
            lift_mapping_path,
        )

    rows = []
    for (tf, target, ct), sub in merged.groupby(["source", "target", "cell_type"]):
        peaks_hg38 = set(sub["peak"].astype(str))
        if output_genome == "hg19":
            regions, dropped = _peaks_hg38_to_regions_hg19(peaks_hg38, lift_mapping)
            n_peaks_dropped_lift += dropped
            if regions is None:
                continue
        else:
            regions = ";".join(sorted({peak_colon_to_regions(p) for p in peaks_hg38}))
        pval = float(sub["Pval"].min())
        rows.append({"TF": tf, "Target": target, "cell_type": ct, "Regions": regions, "Pval": pval})

    grn_long = pd.DataFrame(rows)
    if grn_long.empty:
        raise ValueError("No GRN rows after TF-peak filter and/or hg19 lift.")

    # Global union first, then filter TFs by total target count.
    grn = _aggregate_tf_target(grn_long)
    grn, tf_names = _filter_grn_by_min_targets(grn, min_targets)
    grn_long = grn_long[grn_long["TF"].isin(tf_names)].reset_index(drop=True)

    meta = {
        "n_edges": len(grn),
        "n_tf": len(tf_names),
        "n_celltypes": int(grn_long["cell_type"].nunique()),
        "tfs_with_edges_by_celltype": grn_long.groupby("cell_type")["TF"].nunique().astype(int).to_dict(),
        "min_targets": min_targets,
        "min_targets_scope": "global",
        "grn_mode": grn_mode,
        "use_peak_score_only": use_peak_score_only,
        "pval_mapping": "pow10_neg_exp",
        "pval_formula": "10^(-exp(score))",
        "peak_join_genome": "hg38",
        "regions_output_genome": output_genome,
        "tf_peak_filter": tf_peak_filter,
        "min_tf_binding": min_tf_binding,
        "n_rows_after_gene_peak_join": n_after_gene_peak,
        "n_triplet_rows_before_agg": n_triplet_rows,
        "n_peaks_dropped_lift_fail": n_peaks_dropped_lift,
        "lift_mapping_path": str(lift_mapping_path) if lift_mapping_path else None,
    }
    return grn, tf_names, meta, grn_long


def save_trait_grn_by_celltype(
    grn_long: pd.DataFrame,
    out_dir: str | Path,
    min_targets: int = 10,
) -> dict:
    """Write per-cell-type GRN files under grn_by_celltype/{ct}/."""
    out_dir = Path(out_dir)
    base = out_dir / "grn_by_celltype"
    base.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"celltypes": {}, "min_targets": min_targets}

    for ct in sorted(grn_long["cell_type"].astype(str).unique()):
        sub = grn_long[grn_long["cell_type"].astype(str) == ct]
        grn_ct, tf_names = _filter_grn_by_min_targets(_aggregate_tf_target(sub), min_targets)
        if len(tf_names) == 0:
            continue
        ct_dir = base / str(ct)
        ct_dir.mkdir(parents=True, exist_ok=True)
        grn_ct.to_csv(ct_dir / "grn_trait.csv", index=False)
        (ct_dir / "grn_tf_names.txt").write_text("\n".join(tf_names) + "\n")
        manifest["celltypes"][str(ct)] = {
            "n_tf": len(tf_names),
            "n_edges": len(grn_ct),
            "grn_path": str(ct_dir / "grn_trait.csv"),
        }

    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def save_trait_grn_shared_structure(
    grn_long: pd.DataFrame,
    grn_global: pd.DataFrame,
    tf_names: list[str],
    out_dir: str | Path,
    min_targets: int = 10,
) -> dict:
    """Shared target universe for TFs present in each cell type.

    A TF is written for a cell type only if that TF has at least one real edge
    in that cell type. For those TFs, the TF-target-Regions topology is expanded
    to the global union observed for the TF. Missing target evidence within a
    present TF is encoded as Pval=1.0, i.e. zero peak-gene strength after
    -log10(Pval).
    """
    out_dir = Path(out_dir)
    base = out_dir / "grn_by_celltype"
    base.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"celltypes": {}, "min_targets": min_targets, "mode": "shared_structure"}

    pval_by_ct: dict[str, dict[tuple[str, str], float]] = {}
    for ct in sorted(grn_long["cell_type"].astype(str).unique()):
        sub = grn_long[grn_long["cell_type"].astype(str) == ct]
        agg = _aggregate_tf_target(sub)
        pval_by_ct[str(ct)] = {
            (str(r.TF), str(r.Target)): float(r.Pval) for r in agg.itertuples(index=False)
        }

    tf_by_ct = {
        str(ct): sorted(set(sub["TF"].astype(str)))
        for ct, sub in grn_long.groupby(grn_long["cell_type"].astype(str))
    }

    for ct in sorted(grn_long["cell_type"].astype(str).unique()):
        ct_tfs = tf_by_ct.get(str(ct), [])
        grn_ct = grn_global[grn_global["TF"].astype(str).isin(ct_tfs)].copy()
        lookup = pval_by_ct.get(str(ct), {})
        grn_ct["Pval"] = grn_ct.apply(
            lambda r: lookup.get((str(r["TF"]), str(r["Target"])), 1.0),
            axis=1,
        )
        ct_dir = base / str(ct)
        ct_dir.mkdir(parents=True, exist_ok=True)
        grn_ct.to_csv(ct_dir / "grn_trait.csv", index=False)
        (ct_dir / "grn_tf_names.txt").write_text("\n".join(ct_tfs) + "\n")
        manifest["celltypes"][str(ct)] = {
            "n_tf": len(ct_tfs),
            "n_edges": len(grn_ct),
            "grn_path": str(ct_dir / "grn_trait.csv"),
        }

    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def save_trait_grn_outputs(
    grn: pd.DataFrame,
    tf_names: list[str],
    meta: dict,
    out_dir: str | Path,
    grn_long: pd.DataFrame | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grn_path = out_dir / "grn_trait.csv"
    grn.to_csv(grn_path, index=False)
    (out_dir / "grn_tf_names.txt").write_text("\n".join(tf_names) + "\n")

    grn_mode = meta.get("grn_mode", "global")
    min_targets = int(meta.get("min_targets", 10))
    if grn_mode == "shared_structure" and grn_long is not None:
        manifest = save_trait_grn_shared_structure(
            grn_long, grn, tf_names, out_dir, min_targets=min_targets
        )
        meta = {**meta, "shared_structure_manifest": manifest}
    elif grn_mode == "per_celltype" and grn_long is not None:
        manifest = save_trait_grn_by_celltype(grn_long, out_dir, min_targets=min_targets)
        meta = {**meta, "per_celltype_manifest": manifest}

    (out_dir / "grn_meta.json").write_text(json.dumps(meta, indent=2))
    return grn_path


def load_bim_snp_map(bim_path: str | Path) -> pd.DataFrame:
    """Load plink bim as SNP -> CHR, POS."""
    bim = pd.read_csv(
        bim_path,
        sep=r"\s+",
        header=None,
        names=["CHR", "SNP", "CM", "POS", "A1", "A2"],
    )
    bim["CHR"] = bim["CHR"].astype(str).str.replace("^chr", "", regex=True)
    return bim[["SNP", "CHR", "POS", "A1", "A2"]].drop_duplicates("SNP")


def format_snp_info_for_grs(
    sumstats_file: str | Path,
    bim_prefix: str | Path,
    out_path: str | Path,
    sample_size: int | None = None,
) -> Path:
    """
    Build SNP summary input: CHR, POS, ES, SE, LP, AF, SZ, SNP.

    Supports sumstats with SNP, A1, A2, Z, N (scz file) or BETA/SE/P variants.
    """
    sumstats_file = Path(sumstats_file)
    out_path = Path(out_path)
    bim_prefix = Path(bim_prefix)

    if str(sumstats_file).endswith(".gz"):
        ss = pd.read_csv(sumstats_file, sep=r"\s+|\t", engine="python", compression="gzip")
    else:
        ss = pd.read_csv(sumstats_file, sep=r"\s+|\t", engine="python")

    ss.columns = [c.upper() for c in ss.columns]
    if "RSID" in ss.columns and "SNP" not in ss.columns:
        ss = ss.rename(columns={"RSID": "SNP"})

    bim_files = list(bim_prefix.parent.glob(f"{bim_prefix.name}*.bim"))
    if not bim_files:
        bim_files = [Path(f"{bim_prefix}.bim")]
    bim_parts = [load_bim_snp_map(p) for p in bim_files if p.exists()]
    if not bim_parts:
        raise FileNotFoundError(f"No bim files for prefix {bim_prefix}")
    bim = pd.concat(bim_parts, ignore_index=True).drop_duplicates("SNP")

    df = ss.merge(bim, on="SNP", how="inner")
    if df.empty:
        raise ValueError("No SNPs overlap between sumstats and bim reference.")

    if "Z" in df.columns:
        from scipy.stats import norm

        z = df["Z"].astype(float)
        if "N" in df.columns:
            df["SE"] = 1.0 / np.sqrt(df["N"].astype(float))
            df["ES"] = z * df["SE"]
        else:
            df["SE"] = 0.01
            df["ES"] = z * df["SE"]
        if "P" not in df.columns:
            df["P"] = 2 * norm.sf(np.abs(z))
    if "BETA" in df.columns:
        df["ES"] = df["BETA"]
    if "SE" in df.columns:
        pass
    elif "Z" in df.columns and "N" in df.columns:
        df["SE"] = 1.0 / np.sqrt(df["N"])

    if "P" not in df.columns and "Z" in df.columns:
        from scipy.stats import norm

        df["P"] = 2 * norm.sf(np.abs(df["Z"].astype(float)))

    df["LP"] = -np.log10(df["P"].clip(lower=1e-300))
    df["AF"] = 0.5
    if sample_size is None and "N" in df.columns:
        sample_size = int(df["N"].median())
    df["SZ"] = sample_size if sample_size else 100000

    out = df[["CHR", "POS", "ES", "SE", "LP", "AF", "SZ", "SNP"]].copy()
    out.to_csv(out_path, sep="\t", index=False)
    return out_path


def prepare_magma_snploc(bim_prefix: str | Path, out_path: str | Path) -> Path:
    """Build MAGMA snp-loc file from plink bim (SNP, CHR, BP)."""
    bim_prefix = Path(bim_prefix)
    bim_files = sorted(bim_prefix.parent.glob(f"{bim_prefix.name}*.bim"))
    if not bim_files:
        bim_files = [Path(f"{bim_prefix}.bim")]
    parts = [load_bim_snp_map(p) for p in bim_files if p.exists()]
    bim = pd.concat(parts, ignore_index=True).drop_duplicates("SNP")
    out = bim[["SNP", "CHR", "POS"]].rename(columns={"POS": "BP"})
    out.to_csv(out_path, sep="\t", index=False, header=False)
    return Path(out_path)


def prepare_magma_pval_file(
    sumstats_file: str | Path,
    bim_prefix: str | Path,
    out_path: str | Path,
) -> Path:
    """MAGMA --pval format: SNP, P, N (tab-separated, header)."""
    sumstats_file = Path(sumstats_file)
    out_path = Path(out_path)
    bim_prefix = Path(bim_prefix)

    if str(sumstats_file).endswith(".gz"):
        ss = pd.read_csv(sumstats_file, sep=r"\s+|\t", engine="python", compression="gzip")
    else:
        ss = pd.read_csv(sumstats_file, sep=r"\s+|\t", engine="python")
    ss.columns = [c.upper() for c in ss.columns]
    if "RSID" in ss.columns:
        ss = ss.rename(columns={"RSID": "SNP"})

    bim_files = sorted(bim_prefix.parent.glob(f"{bim_prefix.name}*.bim"))
    if not bim_files:
        single = Path(f"{bim_prefix}.bim")
        bim_files = [single] if single.exists() else []
    parts = [load_bim_snp_map(p) for p in bim_files if p.exists()]
    if not parts:
        raise FileNotFoundError(f"No bim files for prefix {bim_prefix}")
    bim = pd.concat(parts, ignore_index=True).drop_duplicates("SNP")
    df = ss.merge(bim[["SNP"]], on="SNP", how="inner")

    if "P" not in df.columns and "Z" in df.columns:
        from scipy.stats import norm

        df["P"] = 2 * norm.sf(np.abs(df["Z"].astype(float)))
    if "N" not in df.columns:
        df["N"] = 100000

    df[["SNP", "P", "N"]].to_csv(out_path, sep="\t", index=False)
    return out_path


def run_magma_gene_analysis(
    workdir: str | Path,
    trait_name: str,
    magma_bin: str | Path,
    bfile_root: str | Path,
    gene_loc: str | Path,
    snp_loc: str | Path,
    pval_file: str | Path,
    skip_existing: bool = True,
) -> Path:
    """Run MAGMA gene-based analysis for trait localization inputs."""
    import shutil
    import subprocess

    workdir = Path(workdir)
    trait_dir = workdir / trait_name
    out_dir = trait_dir / "magma"
    out_dir.mkdir(parents=True, exist_ok=True)

    magma_bin = Path(magma_bin)
    gene_loc = Path(gene_loc)
    snp_loc = Path(snp_loc)
    pval_file = Path(pval_file)
    final_out = trait_dir / "magma.genes.out"
    nested_out = out_dir / "magma.genes.out"
    annot_out = out_dir / f"{trait_name}.genes.annot"

    if skip_existing and final_out.exists():
        print(f"[SKIP] MAGMA output exists: {final_out}")
        return final_out
    if skip_existing and nested_out.exists():
        shutil.copy2(nested_out, final_out)
        print(f"[SKIP] MAGMA output exists: {nested_out}")
        return final_out

    required_paths = [magma_bin, gene_loc, snp_loc, pval_file]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing MAGMA input(s):\n" + "\n".join(map(str, missing)))

    print("[MAGMA] Step 1: SNP -> gene annotation")
    subprocess.run(
        [
            str(magma_bin),
            "--snp-loc",
            str(snp_loc),
            "--annotate",
            "window=10,10",
            "--gene-loc",
            str(gene_loc),
            "--out",
            str(out_dir / trait_name),
        ],
        check=True,
    )

    n_value = "100000"
    with pval_file.open() as handle:
        next(handle, None)
        second_line = next(handle, "").strip().split()
        if len(second_line) >= 3 and second_line[2]:
            n_value = second_line[2]

    print("[MAGMA] Step 2: Gene-based association")
    subprocess.run(
        [
            str(magma_bin),
            "--bfile",
            str(bfile_root),
            "--pval",
            str(pval_file),
            f"N={n_value}",
            "--gene-annot",
            str(annot_out),
            "--out",
            str(out_dir / "magma"),
        ],
        check=True,
    )

    if not nested_out.exists():
        raise FileNotFoundError(f"MAGMA finished but output was not found: {nested_out}")
    shutil.copy2(nested_out, final_out)
    print(f"[OK] {final_out}")
    return final_out

def read_magma_with_symbol(magma_out, gene_loc, map_csv=None):
    """Read MAGMA gene scores and map Entrez IDs to gene symbols."""
    magma = pd.read_csv(magma_out, sep=r"\s+", comment="#")
    required = {"GENE", "P"}
    missing = required - set(magma.columns)
    if missing:
        raise ValueError(f"MAGMA output missing columns: {sorted(missing)}")

    loc = pd.read_csv(
        gene_loc,
        sep="\t",
        header=None,
        names=["GENE", "CHR", "START", "STOP", "STRAND", "SYMBOL"],
        dtype={"GENE": str, "SYMBOL": str},
    )
    loc = loc[["GENE", "SYMBOL"]].dropna().drop_duplicates("GENE")
    if map_csv is not None:
        Path(map_csv).parent.mkdir(parents=True, exist_ok=True)
        loc.to_csv(map_csv, index=False)

    magma["GENE"] = magma["GENE"].astype(str)
    magma = magma.merge(loc, on="GENE", how="left")
    magma["gene_risk"] = safe_neglog10_pval(magma["P"])
    magma = magma[np.isfinite(magma["gene_risk"]) & magma["SYMBOL"].notna()].copy()
    magma = magma.sort_values("gene_risk", ascending=False).drop_duplicates("SYMBOL")
    return magma[["GENE", "SYMBOL", "P", "gene_risk"]]


def build_peak_snp_signal(regions, snp_info, buffer=500):
    """Return peak-level SNP signal from SNP summary input."""
    peaks = pd.DataFrame({"Regions": sorted(set(regions))})
    coords = peaks["Regions"].map(parse_region)
    peaks[["chrom", "start", "end"]] = pd.DataFrame(coords.tolist(), index=peaks.index)

    snp = pd.read_csv(snp_info, sep="\t")
    required = {"CHR", "POS", "LP", "SNP"}
    missing = required - set(snp.columns)
    if missing:
        raise ValueError(f"snp_info missing columns: {sorted(missing)}")

    snp = snp[["CHR", "POS", "LP", "SNP"]].copy()
    snp["chrom"] = "chr" + snp["CHR"].astype(str).str.replace("chr", "", regex=False)
    snp["POS"] = pd.to_numeric(snp["POS"], errors="coerce")
    snp["LP"] = pd.to_numeric(snp["LP"], errors="coerce")
    snp = snp[np.isfinite(snp["POS"]) & np.isfinite(snp["LP"])]

    snp_by_chr = {}
    for chrom, sub in snp.groupby("chrom", sort=False):
        sub = sub.sort_values("POS")
        snp_by_chr[chrom] = (
            sub["POS"].to_numpy(),
            sub["LP"].to_numpy(),
            sub["SNP"].astype(str).to_numpy(),
        )

    rows = []
    for row in peaks.itertuples(index=False):
        positions, lp_values, snp_ids = snp_by_chr.get(row.chrom, (None, None, None))
        peak_snp_signal = np.nan
        top_snp = None
        n_snp = 0

        if positions is not None:
            left = bisect_left(positions, row.start - buffer)
            right = bisect_right(positions, row.end + buffer)

            if right > left:
                values = lp_values[left:right]
                ids = snp_ids[left:right]
                ok = np.isfinite(values)
                if ok.any():
                    values = values[ok]
                    ids = ids[ok]
                    best = int(np.argmax(values))
                    peak_snp_signal = float(values[best])
                    top_snp = ids[best]
                    n_snp = int(values.size)

        rows.append((row.Regions, peak_snp_signal, top_snp, n_snp))

    return pd.DataFrame(
        rows,
        columns=["Regions", "peak_snp_signal", "top_snp", "n_overlap_snps"],
    )


def select_cluster_tf_targets(
    tf_gene,
    cluster_id,
    edge_weight_col="weight",
    min_targets=5,
):
    """Select TF-target edges for one cluster after TF-level target-count filtering."""
    if not isinstance(tf_gene, pd.DataFrame):
        tf_gene = pd.read_csv(tf_gene)

    required = {"source", "target", "cell_type"}
    missing = required - set(tf_gene.columns)
    if missing:
        raise ValueError(f"TF-gene table missing columns: {sorted(missing)}")

    tf_gene = tf_gene.copy()
    tf_gene["source"] = tf_gene["source"].map(normalize_tf_name)
    tf_gene["target"] = tf_gene["target"].map(normalize_gene_name)
    tf_gene["cell_type"] = tf_gene["cell_type"].astype(str)

    if edge_weight_col in tf_gene.columns:
        tf_gene["TF_target_weight"] = pd.to_numeric(tf_gene[edge_weight_col], errors="coerce")
        weight_source = edge_weight_col
    else:
        tf_gene["TF_target_weight"] = 1.0
        weight_source = "constant_1_missing_weight_column"

    tf_gene["TF_target_weight"] = tf_gene["TF_target_weight"].replace([np.inf, -np.inf], np.nan)
    invalid_weight = int(tf_gene["TF_target_weight"].isna().sum())
    tf_gene["TF_target_weight"] = tf_gene["TF_target_weight"].fillna(1.0)

    negative_weight = int((tf_gene["TF_target_weight"] < 0).sum())
    if negative_weight:
        print(f"[WARN] {negative_weight} negative TF-target weights found; using absolute values.")
        tf_gene["TF_target_weight"] = tf_gene["TF_target_weight"].abs()

    raw_targets = (
        tf_gene[tf_gene["cell_type"].eq(str(cluster_id))][
            ["source", "target", "TF_target_weight"]
        ]
        .dropna(subset=["source", "target"])
        .rename(columns={"source": "TF", "target": "Target"})
        .groupby(["TF", "Target"], as_index=False)["TF_target_weight"]
        .max()
        .reset_index(drop=True)
    )
    if raw_targets.empty:
        raise ValueError(f"No raw TF-target edges found for cluster {cluster_id}.")

    target_counts = raw_targets.groupby("TF")["Target"].nunique().sort_values(ascending=False)
    eligible_tfs = set(target_counts[target_counts > min_targets].index)
    excluded_tf_counts = target_counts[target_counts <= min_targets]

    cluster_targets = (
        raw_targets[raw_targets["TF"].isin(eligible_tfs)]
        .drop_duplicates(["TF", "Target"])
        .reset_index(drop=True)
    )
    if cluster_targets.empty:
        raise ValueError(f"No cluster {cluster_id} TFs have > {min_targets} unique targets.")

    targets_by_tf = {
        tf: set(sub["Target"])
        for tf, sub in cluster_targets.groupby("TF", sort=False)
    }

    return {
        "tf_gene_input": tf_gene,
        "raw_cluster_tf_targets": raw_targets,
        "cluster_tf_targets": cluster_targets,
        "cluster_targets_by_tf": targets_by_tf,
        "tested_tfs": sorted(cluster_targets["TF"].unique()),
        "cluster_target_counts": target_counts,
        "excluded_tf_counts": excluded_tf_counts,
        "weight_source": weight_source,
        "invalid_weight": invalid_weight,
        "negative_weight": negative_weight,
    }


def build_cluster_fgot_grn(
    cluster_tf_targets,
    gene_peak,
    lift_mapping,
    cluster_id,
):
    """Build cluster-specific TF-target-peak rows for weighted eRegulon scoring."""
    if not isinstance(gene_peak, pd.DataFrame):
        gene_peak = pd.read_csv(gene_peak)
    if not isinstance(lift_mapping, pd.DataFrame):
        lift_mapping = pd.read_csv(lift_mapping, sep="\t")

    for col in ["gene", "peak", "score", "celltype"]:
        if col not in gene_peak.columns:
            raise ValueError(f"gene-peak table missing column: {col}")
    required_lift = {"peak_hg38", "peak_hg19", "status"}
    missing_lift = required_lift - set(lift_mapping.columns)
    if missing_lift:
        raise ValueError(f"liftover table missing columns: {sorted(missing_lift)}")

    gene_peak = gene_peak.copy()
    gene_peak["gene"] = gene_peak["gene"].map(normalize_gene_name)
    gene_peak["score"] = pd.to_numeric(gene_peak["score"], errors="coerce")
    gene_peak["celltype"] = gene_peak["celltype"].astype(str)

    cluster_target_union = set(cluster_tf_targets["Target"])
    lift_ok = lift_mapping[lift_mapping["status"].astype(str).str.lower().eq("ok")].copy()
    gene_peak_hg19 = gene_peak[gene_peak["gene"].isin(cluster_target_union)].copy()
    gene_peak_hg19 = gene_peak_hg19.merge(
        lift_ok[["peak_hg38", "peak_hg19"]],
        left_on="peak",
        right_on="peak_hg38",
        how="inner",
    )
    gene_peak_hg19["Regions"] = gene_peak_hg19["peak_hg19"].map(region_to_grs_region)

    cluster_gene_peak = gene_peak_hg19[gene_peak_hg19["celltype"].eq(str(cluster_id))].copy()
    cluster_strength = (
        cluster_gene_peak.groupby(["gene", "Regions"], as_index=False)["score"]
        .max()
        .rename(columns={"score": "score_for_strength"})
    )
    cluster_strength["has_cluster_gene_peak"] = True
    cluster_strength["gene_peak_strength_expected"] = np.exp(cluster_strength["score_for_strength"])
    cluster_strength["Pval"] = np.power(10.0, -cluster_strength["gene_peak_strength_expected"])

    cluster_grn = cluster_tf_targets.merge(
        cluster_strength[
            [
                "gene",
                "Regions",
                "Pval",
                "score_for_strength",
                "has_cluster_gene_peak",
                "gene_peak_strength_expected",
            ]
        ],
        left_on="Target",
        right_on="gene",
        how="inner",
    )
    cluster_grn = (
        cluster_grn.drop(columns=["gene"])
        .dropna(subset=["TF", "Target", "Regions", "Pval"])
        .drop_duplicates()
    )

    cluster_targets_by_tf = {
        tf: set(sub["Target"])
        for tf, sub in cluster_tf_targets.groupby("TF", sort=False)
    }
    grn_targets_by_tf = {
        tf: set(sub["Target"])
        for tf, sub in cluster_grn.groupby("TF", sort=False)
    }
    missing_target_tfs = {
        tf: sorted(targets - grn_targets_by_tf.get(tf, set()))
        for tf, targets in cluster_targets_by_tf.items()
        if targets - grn_targets_by_tf.get(tf, set())
    }

    return {
        "cluster_grn": cluster_grn,
        "cluster_targets_by_tf": cluster_targets_by_tf,
        "cluster_grn_targets_by_tf": grn_targets_by_tf,
        "missing_target_tfs": missing_target_tfs,
        "cluster_gene_peak_strength": cluster_strength,
        "cluster_target_union": cluster_target_union,
    }


def target_grs_from_grn(grn, gene_risk, peak_signal):
    """Score each TF-target pair using gene risk, SNP signal, and gene-peak strength."""
    df = grn.copy()
    df["TF"] = df["TF"].map(normalize_tf_name)
    df["Target"] = df["Target"].map(normalize_gene_name)
    df["TF_target_weight"] = pd.to_numeric(df["TF_target_weight"], errors="coerce")
    df["TF_target_weight"] = df["TF_target_weight"].where(
        np.isfinite(df["TF_target_weight"]),
        np.nan,
    )
    df["gene_peak_strength"] = safe_neglog10_pval(df["Pval"])
    df = df.merge(gene_risk[["SYMBOL", "gene_risk"]], left_on="Target", right_on="SYMBOL", how="left")
    df = df.merge(peak_signal, on="Regions", how="left")
    df["target_GRS"] = df["gene_risk"] * df["peak_snp_signal"] * df["gene_peak_strength"]
    df["weighted_target_GRS"] = df["TF_target_weight"] * df["target_GRS"]

    finite_mask = (
        np.isfinite(df["target_GRS"])
        & np.isfinite(df["weighted_target_GRS"])
        & np.isfinite(df["TF_target_weight"])
        & np.isfinite(df["gene_peak_strength"])
        & np.isfinite(df["gene_risk"])
        & np.isfinite(df["peak_snp_signal"])
    )
    df = df[finite_mask].copy()
    if df.empty:
        return df

    return (
        df.sort_values("weighted_target_GRS", ascending=False)
        .groupby(["TF", "Target"], as_index=False)
        .first()
    )


def weighted_eregulon_grs(target_scores):
    """Aggregate target scores into one weighted eRegulon GRS."""
    scores = target_scores["target_GRS"].to_numpy(dtype=float)
    weights = target_scores["TF_target_weight"].to_numpy(dtype=float)
    ok = np.isfinite(scores) & np.isfinite(weights)
    scores = scores[ok]
    weights = weights[ok]
    if scores.size == 0 or np.sum(weights**2) <= 0:
        return np.nan
    return float(np.sum(weights * scores) / np.sqrt(np.sum(weights**2)))


def rank_weighted_eregulons(
    tested_tfs,
    cluster_tf_targets,
    cluster_grn,
    target_scores,
    buffer=0,
    output_path=None,
):
    """Rank TF regulons by weighted eRegulon GRS."""
    rows = []
    for idx, tf in enumerate(tested_tfs, start=1):
        mod = target_scores[target_scores["TF"] == tf].copy()
        n_scored = int(mod["Target"].nunique())
        raw_targets = int(cluster_tf_targets.loc[cluster_tf_targets["TF"] == tf, "Target"].nunique())
        fgot_targets = int(cluster_grn.loc[cluster_grn["TF"] == tf, "Target"].nunique())
        weights = mod["TF_target_weight"].astype(float) if n_scored > 0 else pd.Series(dtype=float)
        weight_l2_norm = float(np.sqrt(np.sum(np.square(weights)))) if n_scored > 0 else np.nan
        score = weighted_eregulon_grs(mod) if n_scored > 0 else np.nan

        rows.append(
            {
                "RegulonID": f"Regulon_{idx}",
                "RegulonName": tf,
                "weighted_eRegulon_GRS": score,
                "n_targets_raw_cluster": raw_targets,
                "n_targets_with_cluster_FGOT": fgot_targets,
                "n_scored_targets": n_scored,
                "sum_weighted_target_GRS": float(mod["weighted_target_GRS"].sum())
                if n_scored > 0
                else np.nan,
                "weight_l2_norm": weight_l2_norm,
                "mean_TF_target_weight": float(mod["TF_target_weight"].mean())
                if n_scored > 0
                else np.nan,
                "max_TF_target_weight": float(mod["TF_target_weight"].max())
                if n_scored > 0
                else np.nan,
                "buffer": buffer,
                "score_definition": (
                    "sum(TF_target_weight * target_GRS) / "
                    "sqrt(sum(TF_target_weight^2))"
                ),
            }
        )

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        ["weighted_eRegulon_GRS", "n_scored_targets"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    ranking["weighted_GRS_rank"] = np.arange(1, len(ranking) + 1)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(output_path, index=False)

    return ranking


def validate_weighted_eregulon_outputs(
    tested_tfs,
    cluster_tf_targets,
    cluster_targets_by_tf,
    cluster_grn,
    cluster_grn_targets_by_tf,
    target_scores,
    ranking,
    min_targets=5,
    ranking_csv=None,
    target_scores_csv=None,
):
    """Validate weighted eRegulon intermediate and output tables."""
    if len(tested_tfs) != cluster_tf_targets["TF"].nunique():
        raise AssertionError("tested TF count mismatch")
    if not (cluster_tf_targets.groupby("TF")["Target"].nunique() > min_targets).all():
        raise AssertionError("A TF with too few cluster targets entered the ranking.")

    valid_subset = all(
        cluster_grn_targets_by_tf.get(tf, set()).issubset(targets)
        for tf, targets in cluster_targets_by_tf.items()
    )
    if not valid_subset:
        raise AssertionError("Scored targets include targets outside the raw cluster TF-target input.")

    if "has_cluster_gene_peak" in cluster_grn.columns:
        if not cluster_grn["has_cluster_gene_peak"].astype(bool).all():
            raise AssertionError("Weighted GRS contains non-cluster FGOT gene-peak evidence.")

    if not np.allclose(target_scores["gene_peak_strength"], target_scores["gene_peak_strength_expected"]):
        raise AssertionError("gene_peak_strength does not match exp(score_for_strength).")

    for col in ["TF_target_weight", "target_GRS", "weighted_target_GRS"]:
        if not np.isfinite(target_scores[col]).all():
            raise AssertionError(f"Non-finite values found in target_scores[{col!r}].")

    if ranking.empty:
        raise AssertionError("ranking table is empty")
    if ranking_csv is not None and not Path(ranking_csv).exists():
        raise FileNotFoundError(f"Missing ranking output: {ranking_csv}")
    if target_scores_csv is not None and not Path(target_scores_csv).exists():
        raise FileNotFoundError(f"Missing target score output: {target_scores_csv}")

    return True


def _display_regulon_name(name):
    name = str(name)
    return name.split("(")[0] + "(+)" if "(" in name else name + "(+)"


def plot_weighted_eregulon_lollipop(
    ranking,
    output_path=None,
    top_n=30,
    cluster_id=None,
    figsize=(3.6, 4.6),
):
    """Plot top weighted eRegulon GRS values as a lollipop chart."""
    import matplotlib.pyplot as plt

    top = (
        ranking.dropna(subset=["weighted_eRegulon_GRS"])
        .sort_values("weighted_eRegulon_GRS", ascending=False)
        .head(top_n)
        .sort_values("weighted_eRegulon_GRS", ascending=True)
        .copy()
    )
    top["DisplayName"] = top["RegulonName"].map(_display_regulon_name)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.6,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)
    x_values = top["weighted_eRegulon_GRS"]
    y_values = range(len(top))
    ax.hlines(y=y_values, xmin=0, xmax=x_values, color="#BDBDBD", linewidth=0.7, zorder=1)
    ax.scatter(
        x_values,
        y_values,
        s=18,
        color="#2B5C8A",
        edgecolor="black",
        linewidth=0.25,
        zorder=2,
    )
    ax.set_yticks(y_values)
    ax.set_yticklabels(top["DisplayName"])
    ax.set_xlabel("Weighted eRegulon GRS")
    ax.set_ylabel("")
    if cluster_id is not None:
        ax.set_title(f"Cluster {cluster_id}", fontsize=9, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=2.5, width=0.6, color="black")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_xlim(0, x_values.max() * 1.06)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, format=Path(output_path).suffix.lstrip(".") or None, bbox_inches="tight")
    plt.show()
    return fig, ax, top


def extract_top_tf_targets(target_scores, top_tfs, top_n=20):
    """Extract the highest-scoring targets for selected TFs."""
    tables = []
    for rank, tf in enumerate(top_tfs, start=1):
        sub = (
            target_scores[target_scores["TF"].astype(str).eq(tf)]
            .dropna(subset=["Target", "target_GRS"])
            .sort_values("weighted_target_GRS", ascending=False)
            .drop_duplicates("Target")
            .head(top_n)
            .copy()
        )
        sub.insert(0, "TF_weighted_GRS_rank", rank)
        tables.append(sub)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def plot_tf_target_network(
    target_scores,
    tf_name,
    grs_rank=1,
    output_path=None,
    top_n=20,
    vmax=3.0,
    figsize=(4.8, 4.8),
):
    """Plot one TF and its top target genes as a circular network."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    net = (
        target_scores[target_scores["TF"].astype(str).eq(tf_name)]
        .dropna(subset=["Target", "target_GRS"])
        .sort_values("weighted_target_GRS", ascending=False)
        .drop_duplicates("Target")
        .head(top_n)
        .copy()
    )
    if net.empty:
        raise ValueError(f"No scored targets found for {tf_name}.")

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n_targets = len(net)
    angles = np.pi / 2 - np.linspace(0, 2 * np.pi, n_targets, endpoint=False)
    target_x = np.cos(angles)
    target_y = np.sin(angles)
    tf_x, tf_y = 0.0, 0.0

    values = net["target_GRS"].astype(float).to_numpy()
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    target_colors = mpl.cm.Oranges(norm(values))

    fig, ax = plt.subplots(figsize=figsize)
    edge_weights = net["TF_target_weight"].astype(float).to_numpy()
    edge_max = max(np.nanmax(edge_weights), 1.0)
    for x_coord, y_coord, edge_weight in zip(target_x, target_y, edge_weights):
        ax.plot(
            [tf_x, x_coord],
            [tf_y, y_coord],
            color=mpl.cm.tab20(0),
            linewidth=5 * (edge_weight / edge_max),
            zorder=1,
        )

    ax.scatter(
        target_x,
        target_y,
        s=160,
        c=target_colors,
        marker="s",
        edgecolors="#333333",
        linewidths=0.45,
        zorder=3,
    )
    ax.scatter(
        [tf_x],
        [tf_y],
        s=360,
        c="#9E9AC8",
        marker="o",
        edgecolors="#222222",
        linewidths=0.65,
        zorder=4,
    )
    ax.text(
        tf_x,
        tf_y,
        tf_name,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="black",
        zorder=5,
    )

    label_radius = 1.17
    for gene, angle in zip(net["Target"], angles):
        x_coord = label_radius * np.cos(angle)
        y_coord = label_radius * np.sin(angle)
        ha = "left" if np.cos(angle) > 0.15 else "right" if np.cos(angle) < -0.15 else "center"
        va = "bottom" if np.sin(angle) > 0.15 else "top" if np.sin(angle) < -0.15 else "center"
        ax.text(x_coord, y_coord, gene, ha=ha, va=va, fontsize=6.5, color="#222222")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Oranges)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.025, ticks=[0, vmax / 2, vmax])
    cbar.set_label("Target GRS", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2.5, width=0.6)
    cbar.outline.set_linewidth(0.5)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", label="TF", markerfacecolor="#9E9AC8", markersize=8),
        Line2D([0], [0], marker="s", color="none", label="TG", markerfacecolor="#F4C27A", markersize=8),
    ]
    ax.legend(
        handles=legend_handles,
        title="type",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.02),
        fontsize=7,
        title_fontsize=8,
        handletextpad=0.5,
        borderaxespad=0,
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Weighted GRS rank {grs_rank}: {tf_name} eRegulon", fontsize=9, pad=8)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, format=Path(output_path).suffix.lstrip(".") or None, bbox_inches="tight")
    plt.show()
    return net.assign(TF_weighted_GRS_rank=grs_rank)


def _convert_gene_symbols_if_needed(adata, species, homolog_file):
    species = species.lower()
    if species in {"hg19", "human", "homo_sapiens", "hg38"}:
        print(f"Species is {species}; no gene-symbol conversion performed.")
        return adata

    if species not in {"mm10", "mm9", "mouse", "mus_musculus"}:
        print(f"Warning: unknown species '{species}'. No gene-symbol conversion performed.")
        return adata

    if homolog_file is None:
        raise ValueError(f"For species {species}, provide homolog_file for human mapping.")

    homologs = _read_homolog_table(homolog_file)
    mapping = dict(zip(homologs["Original_Symbol"], homologs["Human_Symbol"]))
    mapped_genes = [gene for gene in adata.var_names if gene in mapping]
    if len(mapped_genes) == 0:
        raise ValueError("No genes matched between adata and homolog_file.")

    print(f"Found {len(mapped_genes)} / {adata.n_vars} genes with human homologs.")
    adata = adata[:, mapped_genes].copy()
    adata.var["Original_Symbol"] = adata.var_names
    adata.var_names = [mapping[gene] for gene in adata.var_names]
    adata.var_names_make_unique()

    unique_mask = ~adata.var_names.str.contains(r"-\d+$")
    adata = adata[:, unique_mask].copy()
    print(f"Final gene count after mapping and deduplication: {adata.n_vars}")
    return adata


def _read_homolog_table(homolog_file):
    try:
        homologs = pd.read_csv(homolog_file, sep="\t", header=None)
        if homologs.shape[1] < 2:
            homologs = pd.read_csv(homolog_file, sep="\t")
    except Exception as exc:
        raise ValueError(f"Failed to read homolog file: {exc}") from exc

    homologs = homologs.iloc[:, :2].copy()
    homologs.columns = ["Original_Symbol", "Human_Symbol"]
    return homologs.dropna().drop_duplicates("Original_Symbol")


def _attach_unig_latent_embedding(adata, adata_C):
    if np.array_equal(adata.obs_names, adata_C.obs_names):
        adata_C_aligned = adata_C
    else:
        print("Warning: obs_names differ. Aligning expression and UniG embeddings.")
        common_cells = adata.obs_names.intersection(adata_C.obs_names)
        if len(common_cells) == 0:
            raise ValueError("No common cells between expression AnnData and adata_C.")
        if len(common_cells) < adata.n_obs:
            print(f"Subsetted to {len(common_cells)} common cells.")
        adata = adata[common_cells].copy()
        adata_C_aligned = adata_C[common_cells].copy()

    adata.obsm["latent_embedding"] = np.asarray(_to_dense(adata_C_aligned.X), dtype=np.float32)
    if "spatial" in adata_C_aligned.obsm:
        adata.obsm["spatial"] = adata_C_aligned.obsm["spatial"].copy()
    return adata


def _to_dense(matrix):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix)


def _tqdm_range(n, desc):
    try:
        from tqdm import tqdm

        return tqdm(range(n), desc=desc)
    except ImportError:
        return range(n)


def _format_cluster(value):
    return str(int(float(value)))


def _trait_name_from_cauchy_path(path, sample_name):
    name = path.name
    prefix = f"{sample_name}_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    return name.replace(".Cauchy.csv.gz", "")
