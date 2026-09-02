from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).parent


def read_version():
    version_file = ROOT / "src" / "unig" / "version.py"
    namespace = {}
    exec(version_file.read_text(), namespace)
    return namespace["__version__"]


def read_readme():
    readme = ROOT / "README.md"
    return readme.read_text(encoding="utf-8") if readme.exists() else ""


core_requires = [
    "numpy>=1.23",
    "pandas>=1.5",
    "scipy>=1.9",
    "scikit-learn>=1.1",
    "anndata>=0.9",
    "scanpy>=1.9",
    "matplotlib>=3.6",
    "seaborn>=0.12",
    "tqdm>=4.64",
]

pbg_requires = [
    "attrs>=22.0",
    "torch>=1.13",
    "torchbiggraph",
    "POT>=0.9",
]

preprocess_requires = [
    "squidpy>=1.2",
    "MOODS-python",
]

utils_requires = [
    "faiss-cpu",
    "pybedtools",
    "rpy2",
]

mapping_requires = [
    "POT>=0.9",
]

grn_requires = [
    "pyfaidx>=0.7",
]

trait_requires = [
    "gsMap",
    "pyarrow>=10.0",
    "pyliftover>=0.4",
]

comm_requires = [
    "h5py>=3.7",
    "igraph>=0.10",
    "plotnine>=0.12",
    "arboreto",
    "ctxcore",
    "pyscenic",
    "louvain",
]


def unique_requires(*groups):
    return list(dict.fromkeys(dep for group in groups for dep in group))


base_requires = unique_requires(core_requires, pbg_requires, preprocess_requires)
trait_workflow_requires = trait_requires

extras_require = {
    "mapping": mapping_requires,
    "grn": grn_requires,
    "trait": trait_workflow_requires,
    "comm": comm_requires,
    "all": unique_requires(
        mapping_requires,
        grn_requires,
        trait_workflow_requires,
        comm_requires,
        utils_requires,
    ),
    # Backward-compatible aliases for the previous install commands.
    "basic": [],
    "pbg": [],
    "preprocess": [],
    "utils": utils_requires,
}


setup(
    name="unig",
    version=read_version(),
    description="Unified graph representation learning for multimodal spatial omics.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="UniG contributors",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    include_package_data=True,
    package_data={
        "unig": ["data/*.jaspar"],
    },
    python_requires=">=3.10",
    install_requires=base_requires,
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
