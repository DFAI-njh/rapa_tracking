"""Robust scalar Doppler and observable Cartesian WLS estimates.

`median_vr` remains a scalar LOS measurement. It is never converted to a full
Cartesian velocity by multiplying with LOS. Cartesian velocity is returned
only when multiple Doppler equations form a well-conditioned WLS system.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RadarMotionEstimate:
    median_vr: Optional[float]
    radial_variance: Optional[float]
    los_xy: Optional[np.ndarray]
    cartesian_velocity: Optional[np.ndarray]
    cartesian_covariance: Optional[np.ndarray]
    source: str
    support: int
    mad: Optional[float]
    condition: Optional[float]
    spread: Optional[float]
    residual: Optional[float]
    quality: float


def column(points, name, index):
    if points.dtype.names and name in points.dtype.names:
        return np.asarray(points[name], dtype=np.float64)
    return np.asarray(points[:, index], dtype=np.float64) if points.ndim == 2 else None


def estimate_motion(points, config, center_xy):
    points = np.asarray(points)
    if len(points) < int(config.get('radial_min_support', 3)):
        return RadarMotionEstimate(None, None, None, None, None, 'position',
                                   len(points), None, None, None, None, 0.)
    x, y, vr = column(points, 'x', 0), column(points, 'y', 1), column(points, 'radial_velocity', 3)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(vr)
    x, y, vr = x[finite], y[finite], vr[finite]
    if not len(vr):
        return RadarMotionEstimate(None, None, None, None, None, 'position', 0,
                                   None, None, None, None, 0.)
    median = float(np.median(vr))
    mad = float(np.median(np.abs(vr - median)))
    sigma = max(1.4826 * mad, float(config.get('velocity_std_floor', .5)))
    keep = np.abs(vr - median) <= float(config.get('mad_scale', 3.5)) * sigma
    x, y, vr = x[keep], y[keep], vr[keep]
    median = float(np.median(vr)) if len(vr) else None
    center = np.asarray(center_xy, dtype=np.float64)
    los = center / max(float(np.linalg.norm(center)), 1e-6)
    variance = max(sigma ** 2, float(config.get('variance_floor', .25)))
    mode = str(config.get('mode', 'adaptive'))
    cartesian = covariance = condition = spread = residual = None
    source = 'radial' if len(vr) >= int(config.get('radial_min_support', 3)) else 'position'
    ranges = np.hypot(x, y)
    valid = ranges > 1e-3
    design = np.column_stack((x[valid] / ranges[valid], y[valid] / ranges[valid]))
    target = vr[valid]
    if mode != 'radial' and len(target) >= int(config.get('wls_min_support', 5)):
        angles = np.unwrap(np.arctan2(design[:, 1], design[:, 0]))
        spread = float(np.ptp(angles))
        singular = np.linalg.svd(design, compute_uv=False)
        condition = float(singular[0] / max(singular[-1], 1e-12))
        if (singular[-1] >= float(config.get('min_singular', .15))
                and condition <= float(config.get('max_condition', 30.))
                and spread >= float(config.get('min_spread', .08))):
            cartesian, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
            errors = target - design @ cartesian
            residual = float(np.sqrt(np.mean(errors ** 2)))
            if residual <= float(config.get('max_residual', 3.)):
                covariance = np.linalg.pinv(design.T @ design) * max(residual ** 2, variance)
                source = 'wls'
            else:
                cartesian = covariance = None
    if mode == 'wls' and source != 'wls':
        source = 'position'
        median = None
    quality = min(1., len(vr) / max(float(config.get('full_support', 12)), 1.)) / (1. + mad)
    return RadarMotionEstimate(median, variance, los, cartesian, covariance,
                               source, len(vr), mad, condition, spread, residual,
                               float(quality))
