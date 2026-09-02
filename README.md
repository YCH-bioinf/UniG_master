# UniG

UniG is a Python toolkit for unified graph representation learning across multimodal spatial omics data. It builds heterogeneous biological graphs, trains PyTorch-BigGraph embeddings, and provides downstream analysis utilities for integration, imputation, single-cell mapping, gene regulatory network construction, trait association, and cell-cell communication.

## Project Layout

```text
UniG_master/
├── src/unig/                  # Python package
│   ├── preprocess.py           # Data preprocessing helpers
│   ├── pbg.py                  # Graph construction and PBG training
│   ├── utils.py                # Embedding and AnnData utilities
│   ├── plot.py                 # Plotting helpers
│   └── tasks/                  # Downstream task modules
│       ├── mapping.py          # Single-cell mapping
│       ├── grn.py              # GRN construction
│       ├── trait.py            # Trait association and MAGMA/GRS helpers
│       └── comm.py             # Cell-cell communication analysis
├── docs/                       # Analysis notebooks
├── src/gene_anno/              # Small gene annotation BED tables tracked in GitHub
├── genomes/                    # Local genome FASTA files, not for GitHub upload
└── setup.py
```

## Installation

Create a fresh conda environment named `env_unig` first:

```bash
conda create -n env_unig python=3.10 -y
conda activate env_unig
```

Install the compiled scientific dependencies with conda. This follows the same
general strategy as SIMBA: use conda for packages that often depend on compiled
libraries, then use pip for the editable UniG install.

```bash
conda install -c conda-forge -c bioconda \
  numpy pandas scipy scikit-learn anndata scanpy matplotlib seaborn tqdm \
  h5py pyarrow pyfaidx samtools bedtools -y
```

Install PyTorch according to your machine. For a CPU-only environment:

```bash
conda install -c pytorch pytorch torchvision torchaudio cpuonly -y
```

For GPU machines, install the PyTorch build matching your CUDA driver from the
official PyTorch instructions, then continue with UniG.

Install UniG in editable mode:

```bash
git clone git@github.com:YCH-bioinf/UniG_master.git
cd UniG_master
pip install -e .
```

The base installation includes the core package plus the dependencies needed by
the PBG and preprocessing modules.

Install optional dependencies for specific workflows only when needed:

```bash
pip install -e ".[mapping]"
pip install -e ".[grn]"
pip install -e ".[trait]"
pip install -e ".[comm]"
```

The `trait` extra includes MAGMA/GRS helper dependencies, including liftover
support. Install all declared optional dependencies, including advanced utility
extras:

```bash
pip install -e ".[all]"
```

## Notebooks

The `docs/` directory contains analysis notebooks for:

- integration and imputation
- single-cell mapping
- GRN construction
- trait association
- cell-cell communication

## Data


Large tutorial input datasets are deposited on
Zenodo (Zenodo DOI: https://doi.org/10.5281/zenodo.22243944), including the mouse embryonic brain, ISSAAC, human
heart, GBM data inputs needed to
reproduce the workflows.

Genome FASTA files are not included in the Zenodo upload. The tutorials expect
UCSC-style chromosome names and can use the UCSC `hg38` and `mm10` FASTA files:

```bash
cd UniG_master
mkdir -p genomes/hg38 genomes/mm10
wget -O genomes/hg38/hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz
gunzip -f genomes/hg38/hg38.fa.gz

wget -O genomes/mm10/mm10.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/latest/mm10.fa.gz
gunzip -f genomes/mm10/mm10.fa.gz
```

Optional FASTA index files can be rebuilt locally after download:

```bash
samtools faidx genomes/hg38/hg38.fa
samtools faidx genomes/mm10/mm10.fa
```

After downloading the Zenodo files and genome FASTA files, place or symlink them
to the paths expected by the tutorials, or update the path variables at the top
of each notebook.

## Support
If you have any questions, please feel free to contact us zhanglh@whu.edu.cn.
