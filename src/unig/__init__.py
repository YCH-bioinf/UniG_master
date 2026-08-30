from ._warnings import configure_runtime_warnings

configure_runtime_warnings()

from importlib import import_module

from .version import __version__

_MODULE_ALIASES = {
    "pp": "preprocess",
    "tl": "utils",
    "pl": "plot",
    "mapping": "tasks.mapping",
    "grn": "tasks.grn",
    "trait": "tasks.trait",
    "comm": "tasks.comm",
    "scmore": "tasks.scmore",
}

_FUNCTION_ALIASES = {
    "gen_graph": ("pbg", "gen_graph"),
    "run_training_stages": ("pbg", "run_training_stages"),
}

__all__ = [
    "__version__",
    "settings",
    "pp",
    "tl",
    "pl",
    "mapping",
    "grn",
    "trait",
    "comm",
    "scmore",
    "gen_graph",
    "run_training_stages",
]


def __getattr__(name):
    if name == "settings":
        value = import_module(f"{__name__}.settings").settings
        globals()[name] = value
        return value

    if name in _MODULE_ALIASES:
        module = import_module(f"{__name__}.{_MODULE_ALIASES[name]}")
        globals()[name] = module
        return module

    if name in _FUNCTION_ALIASES:
        module_name, attr_name = _FUNCTION_ALIASES[name]
        value = getattr(import_module(f"{__name__}.{module_name}"), attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
