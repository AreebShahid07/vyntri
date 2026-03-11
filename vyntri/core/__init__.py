# Vyntri Core Package
from .config import Config
from .io import save_model, load_model


def __getattr__(name):
    if name == "VyntriPipeline":
        from .pipeline import VyntriPipeline
        return VyntriPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
