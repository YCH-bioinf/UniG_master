"""Bridge ConSpire TF-gene + FGOT peak-gene networks to scMORE grn format."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .liftover_peaks import ensure_peak_lift_mapping, lift_peak_colon_to_regions


def peak_colon_to_regions(peak: str) -> str:
    """chr15:74375198-74375698 -> chr15-74375198-74375698 (scMORE / snp2peak)."""
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
    """Map hg38 colon peaks to hg19 scMORE Regions string; return (regions, n_dropped)."""
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


def build_scmore_grn(
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
    Output Regions are lifted to hg19 for scMORE snp2peak.

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

    # Global union first, then filter TFs by total target count (scMORE createRegulon style).
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


def save_grn_by_celltype(
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
        grn_ct.to_csv(ct_dir / "grn_scmore.csv", index=False)
        (ct_dir / "grn_tf_names.txt").write_text("\n".join(tf_names) + "\n")
        manifest["celltypes"][str(ct)] = {
            "n_tf": len(tf_names),
            "n_edges": len(grn_ct),
            "grn_path": str(ct_dir / "grn_scmore.csv"),
        }

    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def save_grn_shared_structure(
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
    present TF is encoded as Pval=1.0, i.e. zero scMORE peak-gene strength after
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
        grn_ct.to_csv(ct_dir / "grn_scmore.csv", index=False)
        (ct_dir / "grn_tf_names.txt").write_text("\n".join(ct_tfs) + "\n")
        manifest["celltypes"][str(ct)] = {
            "n_tf": len(ct_tfs),
            "n_edges": len(grn_ct),
            "grn_path": str(ct_dir / "grn_scmore.csv"),
        }

    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def save_grn_outputs(
    grn: pd.DataFrame,
    tf_names: list[str],
    meta: dict,
    out_dir: str | Path,
    grn_long: pd.DataFrame | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grn_path = out_dir / "grn_scmore.csv"
    grn.to_csv(grn_path, index=False)
    (out_dir / "grn_tf_names.txt").write_text("\n".join(tf_names) + "\n")

    grn_mode = meta.get("grn_mode", "global")
    min_targets = int(meta.get("min_targets", 10))
    if grn_mode == "shared_structure" and grn_long is not None:
        manifest = save_grn_shared_structure(
            grn_long, grn, tf_names, out_dir, min_targets=min_targets
        )
        meta = {**meta, "shared_structure_manifest": manifest}
    elif grn_mode == "per_celltype" and grn_long is not None:
        manifest = save_grn_by_celltype(grn_long, out_dir, min_targets=min_targets)
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


def format_snp_info_for_scmore(
    sumstats_file: str | Path,
    bim_prefix: str | Path,
    out_path: str | Path,
    sample_size: int | None = None,
) -> Path:
    """
    Build scMORE snp_info: CHR, POS, ES, SE, LP, AF, SZ, SNP.

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
    """Run MAGMA gene-based analysis for scMORE inputs."""
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
