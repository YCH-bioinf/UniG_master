"""Runtime warning configuration for unig.

The filters here target noisy compatibility warnings emitted by third-party
libraries used during PBG training. They are intentionally narrow so project
warnings and real runtime errors still surface.
"""

import os
import warnings


def configure_runtime_warnings() -> None:
    """Silence known third-party warnings during unig runs."""
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/unig_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/unig_matplotlib")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/unig_numba")
    os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

    warnings.filterwarnings(
        "ignore",
        message=r".*TypedStorage is deprecated.*",
        category=UserWarning,
        module=r"torchbiggraph\.tensorlist",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The legacy Dask DataFrame implementation is deprecated.*",
        category=FutureWarning,
        module=r"dask\.dataframe",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"xarray_schema",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Importing read_hdf from `anndata` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Converting a tensor with requires_grad=True to a scalar.*",
        category=UserWarning,
    )

    try:
        import dask
    except Exception:
        pass
    else:
        dask.config.set({"dataframe.query-planning": True})

    try:
        import torch
    except Exception:
        pass
    else:
        sparse_checks = getattr(torch.sparse, "check_sparse_tensor_invariants", None)
        if sparse_checks is not None and hasattr(sparse_checks, "disable"):
            sparse_checks.disable()
