"""Reusable model-package and transfer-learning utilities."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nidavellir-tools")
except PackageNotFoundError:  # Source checkout before installation.
    __version__ = "0.3.0"

__all__ = ["__version__"]
