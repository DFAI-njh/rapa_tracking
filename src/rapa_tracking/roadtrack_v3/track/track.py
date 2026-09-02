"""Track state, class hysteresis and timestamp lifecycle."""

import math
from typing import Dict, Optional

import numpy as np

from ..motion.cv_filter import _RadarBoxFilter
from ..utils.angle import _wrap_angle
from ..radar.velocity import RadarMotionEstimate

class _Track:
    def __init__(self, track_id: int, box: np.ndarray, score: float, label: int,
                 timestamp_ns: int, config: Dict):
        self.track_id = int(track_id)
        self.filter = _RadarBoxFilter(box, config['filters'])
        self.score = float(score)
        self.label = int(label)
        self.status = 'tentative'
        self.hits = 1
        self.consecutive_hits = 1
        self.misses = 0
        self.age = 1
        self.updated = True
        self.update_kind = 'high'
        self.last_observation = np.asarray(box, dtype=np.float64).copy()
        self.last_observation_ns = int(timestamp_ns)
        self.created_ns = int(timestamp_ns)
        self.confirmation_history = [True]
        self.pending_label: Optional[int] = None
        self.pending_label_hits = 0

    @property
    def box(self) -> np.ndarray:
        return self.filter.box

    def predict(self, dt: float) -> None:
        self.filter.predict(dt)
        self.age += 1
        self.updated = False
        self.update_kind = 'prediction'

    def update(self, box: np.ndarray, score: float, label: int,
               timestamp_ns: int, kind: str, config: Dict,
               radar_motion: Optional[RadarMotionEstimate] = None) -> None:
        previous_observation = self.last_observation.copy()
        previous_timestamp_ns = self.last_observation_ns
        misses_before_update = self.misses
        velocity_config = config['observation_velocity']
        measurement_noise_scale = (
            float(velocity_config.get('low_measurement_noise_scale', 4.0))
            if kind == 'low' else 1.0)
        low_semantics = str(config.get('low_score_update', 'full'))
        if kind != 'low' or low_semantics == 'full':
            self.filter.update(box, measurement_noise_scale)
        elif low_semantics == 'position_only':
            self.filter.update_position_only(box, measurement_noise_scale)
        observation_dt = (int(timestamp_ns) - previous_timestamp_ns) / 1e9
        if (velocity_config['enabled'] and not (
                kind == 'low' and low_semantics == 'none')
                and velocity_config['min_dt'] <= observation_dt <= velocity_config['max_dt']):
            measured_velocity = (
                np.asarray(box[:3], dtype=np.float64) - previous_observation[:3]) / observation_dt
            if kind == 'low':
                gain = (velocity_config['gain_low_after_miss']
                        if misses_before_update else velocity_config['gain_low'])
            else:
                gain = (velocity_config['gain_after_miss'] if misses_before_update
                        else velocity_config['gain'])
            self.filter.velocity = (
                (1.0 - gain) * self.filter.velocity + gain * measured_velocity)
        radar_config = config.get('radar', {})
        if (radar_motion is not None and radar_config.get('kf_update', False)
                and radar_motion.quality >= float(radar_config.get('min_quality', 0.))):
            if (radar_motion.source == 'wls'
                    and radar_motion.cartesian_velocity is not None):
                self.filter.update_cartesian_velocity(
                    radar_motion.cartesian_velocity,
                    radar_motion.cartesian_covariance)
            elif (radar_motion.median_vr is not None
                  and radar_motion.los_xy is not None):
                self.filter.update_radial_velocity(
                    radar_motion.median_vr, radar_motion.los_xy,
                    radar_motion.radial_variance)
        self.last_observation = np.asarray(box, dtype=np.float64).copy()
        self.last_observation_ns = int(timestamp_ns)
        self.updated = True
        self.update_kind = kind
        self.misses = 0
        self.hits += 1
        self.consecutive_hits += 1
        self.confirmation_history.append(True)
        self.confirmation_history = self.confirmation_history[
            -int(config['confirmation']['window']):]
        self.score += config['score_hit_gain'] * (float(score) - self.score)
        self.score = float(np.clip(self.score, 0.0, 1.0))
        self._update_label(int(label), int(config['class_update_min_hits']))

    def confirmation_ready(self, config: Dict) -> bool:
        confirmation = config['confirmation']
        if bool(confirmation.get('require_consecutive', False)):
            return self.consecutive_hits >= int(confirmation['hits'])
        return sum(self.confirmation_history) >= int(confirmation['hits'])

    def _update_label(self, label: int, required_hits: int) -> None:
        if label == self.label:
            self.pending_label = None
            self.pending_label_hits = 0
            return
        if self.pending_label == label:
            self.pending_label_hits += 1
        else:
            self.pending_label = label
            self.pending_label_hits = 1
        if self.pending_label_hits >= required_hits:
            self.label = label
            self.pending_label = None
            self.pending_label_hits = 0

    def mark_missed(self, timestamp_ns: int, config: Dict) -> None:
        self.updated = False
        self.update_kind = 'prediction'
        self.misses += 1
        self.consecutive_hits = 0
        self.confirmation_history.append(False)
        self.confirmation_history = self.confirmation_history[
            -int(config['confirmation']['window']):]
        self.score = float(np.clip(
            self.score * config['score_miss_decay'], 0.0, 1.0))
        seconds_since_observation = max(
            0.0, (int(timestamp_ns) - self.last_observation_ns) / 1e9)
        seconds_since_birth = max(
            0.0, (int(timestamp_ns) - self.created_ns) / 1e9)
        lifecycle = config['lifecycle']
        if self.status == 'tentative':
            if (seconds_since_birth > float(lifecycle['tentative_timeout_seconds'])
                    or self.misses > int(lifecycle['tentative_max_misses'])):
                self.status = 'dead'
        elif self.status == 'dormant':
            if seconds_since_observation > float(lifecycle['reid_max_lost_seconds']):
                self.status = 'dead'
        elif seconds_since_observation > float(lifecycle['visible_max_lost_seconds']):
            if (config['reidentification']['enabled']
                    and float(lifecycle['reid_max_lost_seconds'])
                    > float(lifecycle['visible_max_lost_seconds'])):
                self.status = 'dormant'
            else:
                self.status = 'dead'
        else:
            self.status = 'lost'

    def transform_frame(self, pose: np.ndarray) -> None:
        rotation = np.asarray(pose[:2, :2], dtype=np.float64)
        translation = np.asarray(pose[:2, 2], dtype=np.float64)
        yaw_delta = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        self.filter.transform_frame(pose)
        self.last_observation[:2] = rotation @ self.last_observation[:2] + translation
        self.last_observation[6] = _wrap_angle(
            float(self.last_observation[6]) + yaw_delta)

