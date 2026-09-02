"""Angle normalization helpers for pi-symmetric 3D boxes."""

import math

def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_residual(measured: float, predicted: float) -> float:
    """Return the closest residual for a pi-symmetric 3D bounding box."""
    residual = _wrap_angle(measured - predicted)
    if residual > math.pi / 2:
        residual -= math.pi
    elif residual < -math.pi / 2:
        residual += math.pi
    return residual


