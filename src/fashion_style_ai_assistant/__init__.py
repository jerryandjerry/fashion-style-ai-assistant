"""Fashion Style AI Assistant package."""

from importlib.metadata import version


__all__ = ["__version__"]

try:
    __version__ = version("fashion-style-ai-assistant")
except Exception:  # running from source without installed distribution metadata
    __version__ = "0.1.0"
