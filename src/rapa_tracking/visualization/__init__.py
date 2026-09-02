from pathlib import Path

from .renderer import SimpleTrackVisualizationRenderer

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"
TrackingVisualizationRenderer = SimpleTrackVisualizationRenderer

__all__ = [
    "DEFAULT_CONFIG_PATH", "SimpleTrackVisualizationRenderer",
    "TrackingVisualizationRenderer",
]
