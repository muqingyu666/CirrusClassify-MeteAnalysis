"""High-level APIs for cirrus classification and analysis."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cirrusclassify")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
