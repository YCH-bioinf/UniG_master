"""Lift peak coordinates from hg38 to hg19 for CRE / scMORE / gsMap pipelines."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PEAK_PATTERN = re.compile(r"^(?P<chrom>chr[^:]+):(?P<start>\d+)-(?P<end>\d+)$", re.IGNORECASE)

def _default_chain_path() -> Path:
    """Find the hg38->hg19 chain after moving downstream workflow folders."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "benchmark_help/data/pbmc/hg38ToHg19.over.chain",
        Path("/home/nas3/biod/yangchenghui/proj2/benchmark_help/data/pbmc/hg38ToHg19.over.chain"),
        Path("/home/nas3/biod/yangchenghui/benchmark_help/data/pbmc/hg38ToHg19.over.chain"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_CHAIN = _default_chain_path()


def parse_peak_colon(peak: str) -> tuple[str, int, int]:
    """Parse chr15:74375198-74375698 -> (chrom, start, end), 1-based closed interval."""
    peak = str(peak).strip()
    m = PEAK_PATTERN.match(peak)
    if not m:
        raise ValueError(f"Invalid peak format: {peak!r}")
    chrom = m.group("chrom")
    if not chrom.lower().startswith("chr"):
        chrom = f"chr{chrom}"
    return chrom, int(m.group("start")), int(m.group("end"))


def format_peak_colon(chrom: str, start: int, end: int) -> str:
    """Format (chrom, start, end) -> chr15:74375198-74375698."""
    chrom = str(chrom).strip()
    if not chrom.lower().startswith("chr"):
        chrom = f"chr{chrom}"
    return f"{chrom}:{start}-{end}"


def _lift_position(lo, chrom: str, pos_1based: int) -> tuple[str, int] | None:
    """Lift a single 1-based position; returns (chrom, pos_1based) or None."""
    # pyliftover uses 0-based coordinates
    lifted = lo.convert_coordinate(chrom, pos_1based - 1)
    if lifted is None or len(lifted) == 0:
        return None
    # Use first mapping if multiple
    new_chrom, new_pos_0, _strand, _score = lifted[0]
    return new_chrom, int(new_pos_0) + 1


def lift_interval(lo, chrom: str, start: int, end: int) -> tuple[str, int, int] | None:
    """Lift start and end; require same chromosome on hg19 and end > start."""
    start_lift = _lift_position(lo, chrom, start)
    end_lift = _lift_position(lo, chrom, end)
    if start_lift is None or end_lift is None:
        return None
    chrom_s, start_hg19 = start_lift
    chrom_e, end_hg19 = end_lift
    if chrom_s != chrom_e:
        return None
    if end_hg19 <= start_hg19:
        return None
    return chrom_s, start_hg19, end_hg19


def lift_peak_series(
    unique_peaks: pd.Series | list[str],
    chain_path: str | Path,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Lift unique peaks hg38 -> hg19.

    Returns (mapping_df, failed_peaks).
    mapping_df columns: peak_hg38, peak_hg19, status
    """
    try:
        from pyliftover import LiftOver
    except ImportError as e:
        raise ImportError("Install pyliftover: pip install pyliftover") from e

    chain_path = Path(chain_path)
    if not chain_path.exists():
        raise FileNotFoundError(f"Chain file not found: {chain_path}")

    lo = LiftOver(str(chain_path))
    peaks = pd.Series(unique_peaks).drop_duplicates().astype(str)

    rows: list[dict] = []
    failed: list[str] = []

    for peak_hg38 in peaks:
        try:
            chrom, start, end = parse_peak_colon(peak_hg38)
        except ValueError:
            failed.append(peak_hg38)
            rows.append({"peak_hg38": peak_hg38, "peak_hg19": "", "status": "parse_error"})
            continue

        lifted = lift_interval(lo, chrom, start, end)
        if lifted is None:
            failed.append(peak_hg38)
            rows.append({"peak_hg38": peak_hg38, "peak_hg19": "", "status": "lift_failed"})
        else:
            peak_hg19 = format_peak_colon(*lifted)
            rows.append({"peak_hg38": peak_hg38, "peak_hg19": peak_hg19, "status": "ok"})

    return pd.DataFrame(rows), failed


def load_peak_lift_mapping(mapping_path: str | Path) -> dict[str, str]:
    """Load peak_hg38 -> peak_hg19 mapping from TSV (status=ok only)."""
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Lift mapping not found: {mapping_path}")
    df = pd.read_csv(mapping_path, sep="\t")
    if "peak_hg38" not in df.columns or "peak_hg19" not in df.columns:
        raise ValueError(f"Mapping TSV must have peak_hg38 and peak_hg19 columns: {mapping_path}")
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    ok = df.dropna(subset=["peak_hg38", "peak_hg19"])
    ok = ok[ok["peak_hg19"].astype(str).str.len() > 0]
    return dict(zip(ok["peak_hg38"].astype(str), ok["peak_hg19"].astype(str)))


def lift_peak_colon_to_regions(peak_hg19_colon: str) -> str:
    """chr15:74375198-74375698 (hg19) -> chr15-74375198-74375698 for scMORE Regions."""
    peak = str(peak_hg19_colon).strip()
    if peak.startswith("chr"):
        chrom, rest = peak.split(":", 1)
    else:
        chrom, rest = peak.split(":", 1)
        chrom = f"chr{chrom}"
    start, end = rest.split("-", 1)
    return f"{chrom}-{start}-{end}"


def ensure_peak_lift_mapping(
    peaks_hg38: set[str] | list[str],
    mapping_path: str | Path,
    chain_path: str | Path = DEFAULT_CHAIN,
) -> dict[str, str]:
    """Return hg38->hg19 mapping; lift missing peaks and append to TSV if needed."""
    mapping_path = Path(mapping_path)
    existing: dict[str, str] = {}
    if mapping_path.exists():
        existing = load_peak_lift_mapping(mapping_path)

    peaks_hg38 = set(str(p) for p in peaks_hg38)
    missing = peaks_hg38 - set(existing.keys())
    if not missing:
        return existing

    mapping_df, _failed = lift_peak_series(sorted(missing), chain_path)
    new_ok = mapping_df[mapping_df["status"] == "ok"].set_index("peak_hg38")["peak_hg19"].to_dict()
    existing.update({str(k): str(v) for k, v in new_ok.items()})

    if mapping_path.exists():
        prev = pd.read_csv(mapping_path, sep="\t")
        combined = pd.concat([prev, mapping_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["peak_hg38"], keep="last")
    else:
        combined = mapping_df
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(mapping_path, sep="\t", index=False)
    return existing


def lift_gene_peak_csv(
    input_path: str | Path,
    output_path: str | Path,
    chain_path: str | Path = DEFAULT_CHAIN,
    mapping_path: str | Path | None = None,
    qc_path: str | Path | None = None,
) -> dict:
    """
    Read CRE CSV, lift peak column hg38->hg19, merge duplicates (max score).

    Returns QC statistics dict.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if mapping_path is None:
        mapping_path = out_dir / "GBM_peak_liftover_hg38_to_hg19.tsv"
    if qc_path is None:
        qc_path = out_dir / "GBM_peak_liftover_qc.json"

    df = pd.read_csv(input_path)
    if "peak" not in df.columns:
        raise ValueError(f"Missing 'peak' column in {input_path}")

    n_rows_in = len(df)
    n_unique_in = df["peak"].nunique()

    mapping_df, failed = lift_peak_series(df["peak"], chain_path)
    mapping_ok = mapping_df[mapping_df["status"] == "ok"].set_index("peak_hg38")["peak_hg19"]

    df = df.copy()
    df["peak_hg38"] = df["peak"]
    df["peak"] = df["peak_hg38"].map(mapping_ok)
    df = df.dropna(subset=["peak"])

    # Merge duplicate (gene, peak, celltype) after liftover
    group_cols = ["gene", "peak", "celltype"]
    for col in group_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in {input_path}")

    n_before_dedup = len(df)
    df = (
        df.groupby(group_cols, as_index=False)["score"]
        .max()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    out_cols = [c for c in ["gene", "peak", "score", "celltype"] if c in df.columns]
    df[out_cols].to_csv(output_path, index=False)
    mapping_df.to_csv(mapping_path, sep="\t", index=False)

    n_unique_ok = int((mapping_df["status"] == "ok").sum())
    n_unique_fail = int((mapping_df["status"] != "ok").sum())

    qc = {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "chain": str(Path(chain_path).resolve()),
        "mapping_table": str(Path(mapping_path).resolve()),
        "rows_input": n_rows_in,
        "rows_output": len(df),
        "rows_dropped_lift_fail": n_rows_in - n_before_dedup,
        "rows_merged_after_lift": n_before_dedup - len(df),
        "unique_peaks_input": n_unique_in,
        "unique_peaks_lifted_ok": n_unique_ok,
        "unique_peaks_lift_failed": n_unique_fail,
        "lift_success_rate": round(n_unique_ok / n_unique_in, 4) if n_unique_in else 0.0,
        "unique_peaks_output": int(df["peak"].nunique()),
        "peaks_per_celltype_output": df.groupby("celltype")["peak"].nunique().astype(int).to_dict(),
        "failed_peaks_sample": failed[:20],
    }
    Path(qc_path).write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Lift CRE peak coordinates hg38 -> hg19")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("regulatory_result/GBM_gene_CRE_Network_filtered.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("regulatory_result/GBM_gene_CRE_Network_filtered_hg19.csv"),
    )
    parser.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    parser.add_argument("--mapping", type=Path, default=None)
    parser.add_argument("--qc", type=Path, default=None)
    args = parser.parse_args()

    qc = lift_gene_peak_csv(
        args.input,
        args.output,
        chain_path=args.chain,
        mapping_path=args.mapping,
        qc_path=args.qc,
    )
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
