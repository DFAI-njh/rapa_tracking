"""RoadTrack V3 public API with V2 compatibility aliases."""

from pathlib import Path

from .tracker import RoadTrackV3, RoadTrackV3Adapter, TrackOutput

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / 'configs' / 'bosch_roadtrack_v3.yaml'
from .geometry import (
    box_iou_diou_3d, box_iou_giou_3d, box_iou_giou_bev, odiou3d_cost,
    ro_gdiou_similarity,
)

# Compatibility names are package-local only; frozen roadtrack_v2 is untouched.
RoadTrackV2 = RoadTrackV3
RoadTrackV2Adapter = RoadTrackV3Adapter

__all__ = [
    'RoadTrackV2', 'RoadTrackV2Adapter', 'TrackOutput', 'box_iou_diou_3d',
    'box_iou_giou_3d', 'box_iou_giou_bev', 'DEFAULT_CONFIG_PATH',
    'odiou3d_cost',
    'ro_gdiou_similarity', 'RoadTrackV3',
    'RoadTrackV3Adapter',
]
