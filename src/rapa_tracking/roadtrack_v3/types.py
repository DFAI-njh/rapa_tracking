"""Public RoadTrack V3 value types."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass(frozen=True)
class TrackOutput:
    box: np.ndarray
    score: float
    label: int
    track_id: int
    state: str
    velocity: Optional[np.ndarray] = None

