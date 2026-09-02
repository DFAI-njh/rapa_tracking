"""RoadTrack V3 and its display-only ROS2 visualization package."""

from pathlib import Path

__version__ = "0.1.0"
PACKAGE_ROOT = Path(__file__).resolve().parent


def package_path(*parts: str) -> Path:
    """Return an unpacked editable/wheel resource path."""
    return PACKAGE_ROOT.joinpath(*parts)


__all__ = ["PACKAGE_ROOT", "__version__", "package_path"]
