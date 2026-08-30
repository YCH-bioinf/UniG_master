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
│       ├── trait.py            # Trait association analysis
│       ├── scmore.py           # scMORE/MAGMA helpers
│       └── comm.py             # Cell-cell communication analysis
├── docs/                       # Analysis notebooks
├── src/gene_anno/              # Small gene annotation BED tables tracked in GitHub
├── genomes/                    # Local genome FASTA files, not for GitHub upload
└── setup.py
```

## Installation

For development:

```bash
cd UniG_master
pip install -e .
```

The base installation includes the core package plus the dependencies needed by
the PBG and preprocessing modules.

Install optional dependencies for specific workflows:

```bash
pip install -e ".[mapping]"
pip install -e ".[grn]"
pip install -e ".[trait]"
pip install -e ".[comm]"
```

The `trait` extra includes the scMORE/liftover dependencies. Install all
declared optional dependencies, including advanced utility extras:

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

Small gene annotation tables are tracked directly in this GitHub repository
under `src/gene_anno/`.

Large reference files and tutorial input datasets are not stored in GitHub.
They are deposited on Zenodo:

- Zenodo DOI: TODO
- Zenodo download page: TODO

The Zenodo dataset is intended to include:

- `genomes/`: local `hg38` and `mm10` FASTA references and indexes used by
  preprocessing and GRN tutorials.
- Tutorial input datasets referenced by the notebooks in `docs/`, including
  the mouse embryonic brain, ISSAAC, human heart, GBM, MAGMA/gsMap reference,
  and GWAS summary-statistic inputs needed to reproduce the workflows.

See `docs/tutorial_data_manifest.tsv` for the file-level upload manifest.

After downloading the Zenodo files, place or symlink them to the paths expected
by the tutorials, or update the path variables at the top of each notebook.

## Support
If you have any questions, please feel free to contact us zhanglh@whu.edu.cn.
