"""General preprocessing functions"""

from pathlib import Path
import multiprocessing as mp
import os
import re

from sklearn import preprocessing

from scipy.sparse import (
    issparse,
    csr_matrix,
)

import numpy as np
import pandas as pd
import squidpy as sq
import scanpy as sc


_GENOME_FASTA_REGISTRY = {}
_MOTIF_SCANNER = None
_N_MOTIFS = 0


def register_genome(genome, fasta_path):
    """Register a reference FASTA for later motif scans.

    The FASTA must be uncompressed and have a samtools-compatible ``.fai``
    index next to it.  Registration is process-local and makes subsequent
    calls as short as ``peak_motif_matrix(peaks, genome="mm10")``.
    """
    fasta_path = Path(fasta_path).expanduser().resolve()
    if not fasta_path.is_file():
        raise FileNotFoundError(f"Reference FASTA does not exist: {fasta_path}")
    fai_path = Path(f"{fasta_path}.fai")
    if not fai_path.is_file():
        raise FileNotFoundError(
            f"Missing FASTA index: {fai_path}. Run `samtools faidx {fasta_path}` first."
        )
    _GENOME_FASTA_REGISTRY[str(genome).lower()] = fasta_path


class _IndexedFasta:
    """Small dependency-free reader for an uncompressed indexed FASTA."""

    def __init__(self, fasta_path):
        self.path = Path(fasta_path)
        self.handle = self.path.open("rb")
        self.index = {}
        with Path(f"{self.path}.fai").open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    raise ValueError(f"Malformed FASTA index line: {line.rstrip()}")
                self.index[fields[0]] = tuple(map(int, fields[1:5]))

    def close(self):
        self.handle.close()

    def _resolve_chrom(self, chrom):
        candidates = [chrom]
        if chrom.startswith("chr"):
            candidates.append(chrom[3:])
        else:
            candidates.append(f"chr{chrom}")
        if chrom in {"chrM", "M"}:
            candidates.extend(["chrMT", "MT"])
        for candidate in candidates:
            if candidate in self.index:
                return candidate
        raise KeyError(
            f"Chromosome {chrom!r} is absent from {self.path}; "
            f"examples in FASTA: {list(self.index)[:5]}"
        )

    def fetch(self, chrom, start, end):
        chrom = self._resolve_chrom(chrom)
        length, offset, line_bases, line_width = self.index[chrom]
        if start < 1 or end < start or end > length:
            raise ValueError(
                f"Invalid 1-based interval {chrom}:{start}-{end} "
                f"for chromosome length {length}"
            )

        start0 = start - 1
        byte_offset = (
            offset
            + (start0 // line_bases) * line_width
            + (start0 % line_bases)
        )
        remaining = end - start + 1
        chunks = []
        self.handle.seek(byte_offset)
        while remaining:
            chunk = self.handle.readline().strip()
            if not chunk:
                raise EOFError(f"Unexpected end of FASTA while reading {chrom}:{start}-{end}")
            chunk = chunk[:remaining]
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks).decode("ascii").upper()


def _parse_peak_names(peaks):
    pattern = re.compile(r"^([^:\s]+)[:\-](\d+)-(\d+)$")
    parsed = []
    for peak in map(str, peaks):
        match = pattern.fullmatch(peak)
        if match is None:
            raise ValueError(
                f"Invalid peak {peak!r}; expected 'chr1:123-456' "
                "(or 'chr1-123-456')."
            )
        chrom, start, end = match.group(1), int(match.group(2)), int(match.group(3))
        if start < 1 or end < start:
            raise ValueError(f"Invalid 1-based peak interval: {peak!r}")
        parsed.append((chrom, start, end))
    return parsed


def _resolve_fasta(genome, fasta_path=None):
    genome_key = str(genome).lower()
    if fasta_path is not None:
        path = Path(fasta_path).expanduser().resolve()
    elif Path(str(genome)).expanduser().is_file():
        path = Path(str(genome)).expanduser().resolve()
    elif genome_key in _GENOME_FASTA_REGISTRY:
        path = _GENOME_FASTA_REGISTRY[genome_key]
    else:
        env_path = os.environ.get(f"UNIG_{genome_key.upper()}_FASTA")
        genome_dir = os.environ.get("UNIG_GENOME_DIR")
        candidates = []
        if env_path:
            candidates.append(Path(env_path).expanduser())
        if genome_dir:
            root = Path(genome_dir).expanduser()
            candidates.extend([root / f"{genome_key}.fa", root / genome_key / f"{genome_key}.fa"])
        path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"No FASTA registered for {genome!r}. Either pass `fasta_path=...`, "
                f"set UNIG_{genome_key.upper()}_FASTA, or call "
                f"register_genome({genome!r}, '/path/to/{genome_key}.fa') once."
            )

    if not path.is_file():
        raise FileNotFoundError(f"Reference FASTA does not exist: {path}")
    if not Path(f"{path}.fai").is_file():
        raise FileNotFoundError(
            f"Missing FASTA index: {path}.fai. Run `samtools faidx {path}` first."
        )
    return path


def _default_motif_path(genome, species=None):
    species_key = (species or "").lower().replace("_", " ")
    genome_key = str(genome).lower()
    genome_name = Path(str(genome)).name.lower()
    is_mouse = genome_key.startswith("mm") or genome_name.startswith("mm")
    is_human = genome_key.startswith("hg") or genome_name.startswith("hg")
    if species_key in {"mouse", "mus musculus", "10090"} or is_mouse:
        filename = "JASPAR2020_CORE_Mus_musculus.jaspar"
    elif species_key in {"human", "homo sapiens", "9606"} or is_human:
        filename = "JASPAR2020_CORE_Homo_sapiens.jaspar"
    else:
        raise ValueError(
            f"Cannot infer motif species from genome={genome!r}. "
            "Pass `species='Mus musculus'`/`'Homo sapiens'` or provide `motif_path`."
        )
    path = Path(__file__).with_name("data") / filename
    if not path.is_file():
        raise FileNotFoundError(f"Bundled motif database is missing: {path}")
    return path


def _read_jaspar_motifs(path):
    motifs = []
    motif_id = motif_name = None
    rows = {}

    def finish_motif():
        if motif_id is None:
            return
        if set(rows) != set("ACGT"):
            raise ValueError(f"Motif {motif_id} does not contain all A/C/G/T rows")
        widths = {len(rows[base]) for base in "ACGT"}
        if len(widths) != 1:
            raise ValueError(f"Motif {motif_id} has inconsistent PFM row widths")
        motifs.append((motif_id, motif_name, np.asarray([rows[b] for b in "ACGT"], dtype=float)))

    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish_motif()
                header = line[1:].split(None, 1)
                motif_id = header[0]
                motif_name = header[1] if len(header) == 2 else motif_id
                rows = {}
                continue
            base = line[0].upper()
            if base in "ACGT":
                rows[base] = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line[1:])]
        finish_motif()
    if not motifs:
        raise ValueError(f"No JASPAR motifs found in {path}")
    return motifs


def _motifmatchr_matrix(pfm, bg, pseudocount):
    """Reproduce TFBSTools::toPWM + motifmatchr::convert_pwm."""
    even = np.full(4, 0.25, dtype=float)
    probabilities = (pfm + even[:, None] * pseudocount) / (
        pfm.sum(axis=0, keepdims=True) + pseudocount
    )
    matrix = np.log2(probabilities / even[:, None])
    matrix -= (np.log2(even) - np.log2(bg))[:, None]
    return matrix.tolist()


def _init_motif_scanner(matrices, bg, thresholds, n_motifs, window):
    global _MOTIF_SCANNER, _N_MOTIFS
    import MOODS.scan

    scanner = MOODS.scan.Scanner(window)
    scanner.set_motifs(matrices, bg, thresholds)
    _MOTIF_SCANNER = scanner
    _N_MOTIFS = n_motifs


def _scan_one_peak(sequence):
    hits = _MOTIF_SCANNER.scan(sequence)
    return [len(hits[i]) + len(hits[i + _N_MOTIFS]) for i in range(_N_MOTIFS)]


def peak_motif_matrix(
    peaks,
    genome,
    fasta_path=None,
    species=None,
    motif_path=None,
    pvalue=5e-5,
    pseudocount=0.8,
    background="subject",
    window=7,
    n_jobs=1,
    binary=False,
    return_anndata=False,
):
    """Build a peak-by-motif matrix entirely from Python.

    This reproduces the original ``motifmatchr::matchMotifs(..., out="scores")``
    followed by ``motifCounts()`` workflow: matrix entries are numbers of motif
    hits across both strands, not continuous motif scores.

    Parameters
    ----------
    peaks
        Peak names such as ``chr1:12345-12789``. Coordinates are interpreted
        as 1-based and inclusive, matching the original R/GRanges script.
    genome
        Genome label (for example ``"mm10"`` or ``"hg38"``), or a FASTA path.
    fasta_path
        Optional uncompressed reference FASTA. It must have a neighboring
        ``.fai`` index. Omit after using :func:`register_genome`.
    species
        Motif species when it cannot be inferred from the genome label.
    motif_path
        Optional JASPAR-format PFM file. By default, the bundled species-specific
        JASPAR2020 CORE collection is used for mm*/hg* genomes.
    pvalue, pseudocount, background, window
        Motif scanning settings. Defaults match motifmatchr 1.28.0.
        ``background`` can be ``"subject"``, ``"even"``, or four A/C/G/T
        frequencies.
    n_jobs
        Number of worker processes used for motif scanning.
    binary
        Convert hit counts to presence/absence.
    return_anndata
        Return a sparse AnnData instead of a pandas DataFrame.
    """
    try:
        import MOODS.tools
    except ImportError as exc:
        raise ImportError(
            "peak_motif_matrix requires MOODS-python (`pip install MOODS-python`)."
        ) from exc

    peak_names = pd.Index(map(str, peaks), name="peak")
    if peak_names.empty:
        raise ValueError("`peaks` is empty")
    if peak_names.has_duplicates:
        duplicated = peak_names[peak_names.duplicated()].unique()[:5].tolist()
        raise ValueError(f"Peak names must be unique; duplicates include {duplicated}")
    intervals = _parse_peak_names(peak_names)
    fasta = _resolve_fasta(genome, fasta_path)

    reader = _IndexedFasta(fasta)
    try:
        sequences = [reader.fetch(*interval) for interval in intervals]
    finally:
        reader.close()

    if isinstance(background, str):
        if background == "even":
            bg = np.full(4, 0.25, dtype=float)
        elif background == "subject":
            base_counts = np.asarray(
                [sum(sequence.count(base) for sequence in sequences) for base in "ACGT"],
                dtype=float,
            )
            if np.any(base_counts == 0):
                raise ValueError("Cannot estimate background: at least one A/C/G/T count is zero")
            bg = base_counts / base_counts.sum()
        else:
            raise ValueError("`background` must be 'subject', 'even', or four frequencies")
    else:
        bg = np.asarray(background, dtype=float)
        if bg.shape != (4,) or np.any(bg <= 0):
            raise ValueError("Numeric `background` must contain four positive A/C/G/T values")
        bg = bg / bg.sum()

    motif_path = Path(motif_path) if motif_path is not None else _default_motif_path(genome, species)
    motifs = _read_jaspar_motifs(motif_path)
    forward = [_motifmatchr_matrix(pfm, bg, pseudocount) for _, _, pfm in motifs]
    reverse = [MOODS.tools.reverse_complement(matrix) for matrix in forward]
    thresholds = [MOODS.tools.threshold_from_p(matrix, bg.tolist(), pvalue) for matrix in forward]
    matrices = forward + reverse
    scanner_thresholds = thresholds + thresholds

    n_jobs = int(n_jobs)
    if n_jobs < 1:
        raise ValueError("`n_jobs` must be at least 1")
    if n_jobs == 1:
        _init_motif_scanner(matrices, bg.tolist(), scanner_thresholds, len(motifs), window)
        counts = np.asarray([_scan_one_peak(sequence) for sequence in sequences], dtype=np.int32)
    else:
        # ``fork`` also works when called from a Jupyter cell; fall back to
        # ``spawn`` on platforms where fork is unavailable.
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        context = mp.get_context(start_method)
        chunksize = max(1, len(sequences) // (n_jobs * 20))
        with context.Pool(
            n_jobs,
            initializer=_init_motif_scanner,
            initargs=(matrices, bg.tolist(), scanner_thresholds, len(motifs), window),
        ) as pool:
            counts = np.asarray(pool.map(_scan_one_peak, sequences, chunksize), dtype=np.int32)

    if binary:
        counts = (counts > 0).astype(np.int8)

    motif_ids = [motif_id for motif_id, _, _ in motifs]
    motif_names = [name.replace("::", "_") for _, name, _ in motifs]
    if len(set(motif_names)) != len(motif_names):
        raise ValueError("Motif names are not unique after replacing '::' with '_'")

    if return_anndata:
        from anndata import AnnData

        obs = pd.DataFrame(index=peak_names)
        var = pd.DataFrame({"matrix_id": motif_ids, "name": motif_names}, index=motif_names)
        adata = AnnData(X=csr_matrix(counts), obs=obs, var=var)
        adata.uns["motif_scan"] = {
            "genome": str(genome),
            "fasta": str(fasta),
            "motif_database": str(motif_path),
            "pvalue": float(pvalue),
            "pseudocount": float(pseudocount),
            "background": dict(zip("ACGT", map(float, bg))),
            "coordinates": "1-based-inclusive",
        }
        return adata
    return pd.DataFrame(counts, index=peak_names, columns=motif_names)


def spatially_variable_genes(
    adata,
    n_top: int = 2000,
    mode: str = "moran",            # "moran" or "geary"
    coord_type: str = "generic",   
    n_neigh: int = 6,               # used when coord_type != "grid"
):
    """
    Find spatially variable genes (SVGs) with Squidpy and write results to `adata.var`.

    Side effects on `adata.var`:
      - For Moran's I:   `moran_I`  (float)
      - For Geary's  C:  `geary_C`  (float)
      - Significance:    `spatial_qval` (if FDR available) or `spatial_pval`
      - Boolean flag:    `spatially_variable` (SVG membership)
      - Scanpy-style:    `spatially_variable` (same as SVG membership)
      - Rank column:     `spatially_variable_rank` (1..n_top for SVGs; NaN otherwise)

    Returns:
      top_genes : list[str]
      (optionally) ranked_df : pd.DataFrame, when `return_df=True`
    """
    # --- sanity checks ---
    if "spatial" not in adata.obsm:
        raise ValueError("Missing `adata.obsm['spatial']` with XY coordinates.")
    if mode not in {"moran", "geary"}:
        raise ValueError("`mode` must be 'moran' or 'geary'.")

    # --- build spatial graph (if needed) ---
    if "spatial_connectivities" not in adata.obsp:
        sq.gr.spatial_neighbors(
            adata,
            coord_type=coord_type,
            n_neighs =(None if coord_type == "grid" else n_neigh)
        )
        
    # --- compute spatial autocorrelation ---
    sq.gr.spatial_autocorr(
        adata,
        mode=mode,                   # "moran" or "geary"
    )

    # --- fetch result table from `adata.uns` ---
    key = "moranI" if mode == "moran" else "gearyC"
    res = adata.uns.get(key, None)
    if res is None:
        raise RuntimeError(f"Could not find `adata.uns['{key}']`. Was `spatial_autocorr` successful?")
    res = res.copy()

    # --- normalize column names across Squidpy versions ---
    stat_col = "I" if mode == "moran" else "C"
    fdr_cols = [c for c in res.columns if ("fdr" in c.lower() or "qval" in c.lower())]
    p_cols   = [c for c in res.columns if (("p" in c.lower()) and ("fdr" not in c.lower()))]

    # cast to numeric and clean
    for c in {stat_col, *fdr_cols, *p_cols}:
        if c in res.columns:
            res[c] = pd.to_numeric(res[c], errors="coerce")
    res = res.replace([np.inf, -np.inf], np.nan)
    if stat_col in res.columns:
        res = res.dropna(subset=[stat_col])

    # --- rank genes and pick top-N ---
    if fdr_cols:
        qcol = sorted(fdr_cols)[0]                     # e.g., "pval_emp_fdr_bh" or "pval_norm_fdr_bh"
        ranked = res.sort_values([qcol, stat_col], ascending=[True, False])
    elif p_cols:
        pcol = sorted(p_cols)[0]                       # e.g., "pval_emp" / "pval_norm" / "pval"
        ranked = res.sort_values([pcol, stat_col], ascending=[True, False])
    else:
        ranked = res.sort_values(stat_col, ascending=False)

    topk = min(int(n_top), ranked.shape[0])
    top_genes = ranked.head(topk).index.tolist()

    # --- write back to AnnData in a Scanpy-compatible way ---
    # statistic
    if mode == "moran":
        adata.var["moran_I"] = pd.Series(np.nan, index=adata.var_names)
        if stat_col in res.columns:
            adata.var.loc[res.index, "moran_I"] = res[stat_col].values
    else:
        adata.var["geary_C"] = pd.Series(np.nan, index=adata.var_names)
        if stat_col in res.columns:
            adata.var.loc[res.index, "geary_C"] = res[stat_col].values

    # q/p-values
    if fdr_cols:
        qcol = sorted(fdr_cols)[0]
        adata.var["spatial_qval"] = pd.Series(np.nan, index=adata.var_names)
        adata.var.loc[ranked.index, "spatial_qval"] = ranked[qcol].values
    elif p_cols:
        pcol = sorted(p_cols)[0]
        adata.var["spatial_pval"] = pd.Series(np.nan, index=adata.var_names)
        adata.var.loc[ranked.index, "spatial_pval"] = ranked[pcol].values

    # boolean flags
    adata.var["spatially_variable"] = adata.var_names.isin(top_genes)
  
    # provide a rank column like Scanpy often does
    hv_rank = pd.Series(np.nan, index=adata.var_names, dtype="float")
    hv_rank.loc[ranked.head(topk).index] = np.arange(1, topk + 1, dtype=float)
    adata.var["spatially_variable_rank"] = hv_rank

    # also keep a list in uns for convenience adata.uns["svg_top_genes"] = top_genes
    adata.uns["svg_top_genes"] = top_genes
    
    
    
def binarize(adata,
             threshold=1e-5):
    """Binarize an array.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    threshold: `float`, optional (default: 1e-5)
        Values below or equal to this are replaced by 0, above it by 1.

    Returns
    -------
    updates `adata` with the following fields.
    X: `numpy.ndarray` (`adata.X`)
        Store #observations × #var_genes binarized data matrix.
    """
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    adata.X = preprocessing.binarize(adata.X,
                                     threshold=threshold,
                                     copy=True)

def preprocess_rna(adata_rna, normalize=True, n_top_genes = 2000):
    adata = adata_rna.copy()
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=3)
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if 'spatial' in adata.obsm:
        spatially_variable_genes(adata,n_top=n_top_genes)
        adata.var['variable'] = adata.var['spatially_variable']
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        adata.var['variable'] = adata.var['highly_variable']
    return adata


def add_peak_columns(adata, inplace=True, col_chr='chr', col_start='start', col_end='end'):
    """Split chr-prefixed peak names into chromosome, start, and end columns.

    Accepts peak names formatted as ``chrX:START-END`` or ``chrX-START-END``.
    """
    peaks = adata.var.index.astype(str)

    m = peaks.to_series().str.extract(r'^(chr[0-9A-Za-z]+)[:\-](\d+)-(\d+)$')
    m.columns = [col_chr, col_start, col_end]

    bad = m.isna().any(axis=1)
    if bad.any():
        bad_examples = peaks[bad][:5]
        raise ValueError(
            "The following var.index values do not match the expected "
            "'chrX:START-END' or 'chrX-START-END' format, or do not start "
            "with 'chr', for example:\n"
            + "\n".join(map(str, bad_examples))
            + "\nPlease filter or clean these peak names before running."
        )

    m[col_start] = m[col_start].astype(int)
    m[col_end]   = m[col_end].astype(int)

    if inplace:
        adata.var[col_chr]   = m[col_chr].values
        adata.var[col_start] = m[col_start].values
        adata.var[col_end]   = m[col_end].values
        return adata
    else:
        m.index = adata.var.index
        return m
    
    
    
def preprocess_atac(adata_atac):
    adata = adata_atac.copy()
    sc.pp.filter_genes(adata, min_cells=3)

    peak_names = adata.var_names.astype(str)
    keep = peak_names.str.match(r"^chr[0-9A-Za-z]+[:\-]\d+-\d+$")
    adata = adata[:, keep].copy()

    peak_info = add_peak_columns(adata, inplace=False)
    adata.var_names = (
        peak_info['chr'].astype(str)
        + ":"
        + peak_info['start'].astype(str)
        + "-"
        + peak_info['end'].astype(str)
    )
    adata = add_peak_columns(adata)
    return adata


from tqdm import tqdm

def preprocess_gene_info(gene_info, scope = 250000):
    filtered_gene_info = []
    columns = ['id', 'chr', 'starts', 'ends', 'forward', 'backward', 'gene']
    print("Preprocessing gene_info:")
    for info in tqdm(gene_info.itertuples()):
        chr = info.chr
        starts = info.starts
        ends = int(info.ends)
        genes = info.genes
        gene_info_id = chr + '-' + str(starts) + '-' + str(ends) + '-' + genes
        forward = max(0, starts - scope)
        backward = starts + scope
        filtered_gene_info.append([gene_info_id, chr, starts, ends, forward, backward, genes])
    filtered_gene_info = pd.DataFrame(filtered_gene_info, columns=columns)
    filtered_gene_info = filtered_gene_info.drop_duplicates(subset=['id'])
    return filtered_gene_info

def gene_peaks_pairs_by_location(filtered_gene_info, hvg_genes, peaks_to_filter):
    gene_peaks = {}
    print("Search the genes-peaks correspondence based on gene_info and scope:")
    for info in tqdm(filtered_gene_info.itertuples()):
        if not info.gene in hvg_genes:
            continue
        id = info.id
        chr = info.chr
        starts = info.starts
        ends = info.ends
        forward = info.forward
        backward = info.backward
        gene = info.gene
        if not gene in gene_peaks:
            gene_peaks[gene] = set()
        for peak in peaks_to_filter:
            peak_chr, coordinates = peak.split(':')
            peak_start, peak_end = coordinates.split('-')
            if peak_chr == chr and int(peak_start) >= forward and int(peak_end) <= backward:
                gene_peaks[gene].add(peak)
    gene_peaks = {gene: peaks for gene, peaks in gene_peaks.items() if len(peaks) > 0}
    return gene_peaks

def select_genes_peaks_from_pairs(gene_peaks):
    filtered_peaks = set()
    filtered_genes = list(gene_peaks.keys())
    print("Search the filtered peaks:")
    for key in tqdm(gene_peaks.keys()):
        filtered_peaks.update(gene_peaks[key])
    filtered_peaks = list(filtered_peaks)
    print(f"There are {len(filtered_genes)} genes that can be identified from the TSS information.")
    print(f"There are {len(filtered_peaks)} peaks near the upstream and downstream of the above genes.")
    return filtered_genes, filtered_peaks

def select_gene_peaks_by_genes_location(gene_info, hvg_genes, peaks_to_filter, scope = 250000):
    filtered_gene_info = preprocess_gene_info(gene_info, scope)
    gene_peaks = gene_peaks_pairs_by_location(filtered_gene_info, hvg_genes, peaks_to_filter)
    filtered_genes, filtered_peaks = select_genes_peaks_from_pairs(gene_peaks)
    return filtered_genes, filtered_peaks, gene_peaks
