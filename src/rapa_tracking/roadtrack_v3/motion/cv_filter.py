"""Decoupled CV, vertical, yaw and shape filter from frozen V2."""

import math
from typing import Dict

import numpy as np

from ..utils.angle import _wrap_angle, _yaw_residual
from .models import ca_matrices, cv_matrices

class _RadarBoxFilter:
    """Decoupled CV estimators for radar position, height, yaw and size.

    Position ``[x, y, vx, vy]`` is deliberately independent from the
    ``[z, vz]`` and ``[yaw, yaw_rate]`` filters. Length/width/height use an
    EMA. A noisy RAPA-R size or yaw observation therefore cannot inject a
    position/velocity correction through a single large box-state KF.
    """

    def __init__(self, box: np.ndarray, config: Dict):
        position = config['position']
        vertical = config['vertical']
        yaw = config['yaw']
        self.position_model = str(position.get('model', 'cv'))
        if self.position_model == 'ca':
            self.position_x = np.asarray([box[0], box[1], 0., 0., 0., 0.])
            diagonal = position.get('ca_p_diag', list(position['p_diag']) + [64., 64.])
        else:
            self.position_x = np.asarray([box[0], box[1], 0., 0.])
            diagonal = position['p_diag']
        self.position_P = np.diag(np.asarray(diagonal, dtype=np.float64))
        self.position_r = np.asarray(position['measurement_std'], dtype=np.float64) ** 2
        self.position_acceleration_std = float(position['acceleration_std'])
        self.position_jerk_std = float(position.get('jerk_std', 20.))
        self.max_acceleration = float(position.get('max_acceleration', 20.))
        self.vertical_x = np.asarray([box[2], 0.0], dtype=np.float64)
        self.vertical_P = np.diag(np.asarray(vertical['p_diag'], dtype=np.float64))
        self.vertical_r = float(vertical['measurement_std']) ** 2
        self.vertical_acceleration_std = float(vertical['acceleration_std'])
        self.yaw_x = np.asarray([box[6], 0.0], dtype=np.float64)
        self.yaw_P = np.diag(np.asarray(yaw['p_diag'], dtype=np.float64))
        self.yaw_r = float(yaw['measurement_std']) ** 2
        self.yaw_acceleration_std = float(yaw['acceleration_std'])
        self.size = np.maximum(np.asarray(box[3:6], dtype=np.float64), 1e-3)
        self.size_alpha = float(config['size']['smoothing_alpha'])

    @property
    def box(self) -> np.ndarray:
        return np.asarray([
            self.position_x[0], self.position_x[1], self.vertical_x[0],
            self.size[0], self.size[1], self.size[2],
            _wrap_angle(float(self.yaw_x[0])),
        ], dtype=np.float64)

    @property
    def velocity(self) -> np.ndarray:
        return np.asarray([
            self.position_x[2], self.position_x[3], self.vertical_x[1],
        ], dtype=np.float64)

    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float64)
        self.position_x[2:4] = value[:2]
        if len(value) >= 3:
            self.vertical_x[1] = value[2]

    @staticmethod
    def _cv_predict(state: np.ndarray, covariance: np.ndarray, dt: float,
                    acceleration_std: float):
        transition = np.asarray([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = float(acceleration_std) ** 2
        process = q * np.asarray([
            [.25 * dt ** 4, .5 * dt ** 3],
            [.5 * dt ** 3, dt ** 2],
        ], dtype=np.float64)
        return transition @ state, transition @ covariance @ transition.T + process

    @staticmethod
    def _linear_update(state: np.ndarray, covariance: np.ndarray,
                       measurement: np.ndarray, observation: np.ndarray,
                       measurement_covariance: np.ndarray):
        innovation = measurement - observation @ state
        innovation_covariance = (
            observation @ covariance @ observation.T + measurement_covariance)
        gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        updated = state + gain @ innovation
        identity_minus_kh = np.eye(len(state)) - gain @ observation
        updated_covariance = (
            identity_minus_kh @ covariance @ identity_minus_kh.T
            + gain @ measurement_covariance @ gain.T)
        return updated, (updated_covariance + updated_covariance.T) / 2

    def predict(self, dt: float) -> None:
        transition, process = (ca_matrices(dt, self.position_jerk_std)
                               if self.position_model == 'ca'
                               else cv_matrices(dt, self.position_acceleration_std))
        self.position_x = transition @ self.position_x
        self.position_P = transition @ self.position_P @ transition.T + process
        if self.position_model == 'ca':
            norm = float(np.linalg.norm(self.position_x[4:6]))
            if norm > self.max_acceleration:
                self.position_x[4:6] *= self.max_acceleration / norm
        self.vertical_x, self.vertical_P = self._cv_predict(
            self.vertical_x, self.vertical_P, dt, self.vertical_acceleration_std)
        self.yaw_x, self.yaw_P = self._cv_predict(
            self.yaw_x, self.yaw_P, dt, self.yaw_acceleration_std)
        self.yaw_x[0] = _wrap_angle(float(self.yaw_x[0]))

    def update(self, box: np.ndarray, measurement_noise_scale: float = 1.0) -> None:
        noise_scale = max(float(measurement_noise_scale), 1.0)
        position_h = np.zeros((2, len(self.position_x)))
        position_h[0, 0] = position_h[1, 1] = 1.
        self.position_x, self.position_P = self._linear_update(
            self.position_x, self.position_P, np.asarray(box[:2]), position_h,
            np.diag(self.position_r * noise_scale))
        scalar_h = np.asarray([[1., 0.]])
        self.vertical_x, self.vertical_P = self._linear_update(
            self.vertical_x, self.vertical_P, np.asarray([box[2]]), scalar_h,
            np.asarray([[self.vertical_r * noise_scale]]))
        yaw_measurement = self.yaw_x[0] + _yaw_residual(
            float(box[6]), float(self.yaw_x[0]))
        self.yaw_x, self.yaw_P = self._linear_update(
            self.yaw_x, self.yaw_P, np.asarray([yaw_measurement]), scalar_h,
            np.asarray([[self.yaw_r * noise_scale]]))
        self.yaw_x[0] = _wrap_angle(float(self.yaw_x[0]))
        alpha = self.size_alpha / noise_scale
        self.size = np.maximum(
            (1.0 - alpha) * self.size + alpha * np.asarray(box[3:6]), 1e-3)

    def update_position_only(self, box, measurement_noise_scale=1.0):
        scale = max(float(measurement_noise_scale), 1.)
        observation = np.zeros((2, len(self.position_x)))
        observation[0, 0] = observation[1, 1] = 1.
        self.position_x, self.position_P = self._linear_update(
            self.position_x, self.position_P, np.asarray(box[:2]), observation,
            np.diag(self.position_r * scale))
        scalar = np.asarray([[1., 0.]])
        self.vertical_x, self.vertical_P = self._linear_update(
            self.vertical_x, self.vertical_P, np.asarray([box[2]]), scalar,
            np.asarray([[self.vertical_r * scale]]))

    def update_radial_velocity(self, radial_velocity, los_xy, variance):
        observation = np.zeros((1, len(self.position_x)))
        observation[0, 2:4] = np.asarray(los_xy)
        self.position_x, self.position_P = self._linear_update(
            self.position_x, self.position_P, np.asarray([radial_velocity]),
            observation, np.asarray([[max(float(variance), 1e-4)]]))

    def update_cartesian_velocity(self, velocity, covariance):
        observation = np.zeros((2, len(self.position_x)))
        observation[0, 2] = observation[1, 3] = 1.
        self.position_x, self.position_P = self._linear_update(
            self.position_x, self.position_P, np.asarray(velocity), observation,
            np.asarray(covariance))

    def mahalanobis_position_squared(self, point: np.ndarray,
                                     dimensions: str = 'xyz') -> float:
        residual_xy = np.asarray(point[:2], dtype=np.float64) - self.position_x[:2]
        covariance_xy = self.position_P[:2, :2] + np.diag(self.position_r)
        try:
            distance = float(
                residual_xy @ np.linalg.solve(covariance_xy, residual_xy))
            if dimensions == 'xyz':
                residual_z = float(point[2]) - float(self.vertical_x[0])
                distance += residual_z ** 2 / max(
                    float(self.vertical_P[0, 0] + self.vertical_r), 1e-9)
            return distance
        except np.linalg.LinAlgError:
            return float('inf')

    def transform_frame(self, pose: np.ndarray) -> None:
        rotation = np.asarray(pose[:2, :2], dtype=np.float64)
        translation = np.asarray(pose[:2, 2], dtype=np.float64)
        self.position_x[:2] = rotation @ self.position_x[:2] + translation
        self.position_x[2:4] = rotation @ self.position_x[2:4]
        transform = np.zeros((len(self.position_x), len(self.position_x)))
        transform[:2, :2] = rotation
        transform[2:4, 2:4] = rotation
        if self.position_model == 'ca':
            self.position_x[4:6] = rotation @ self.position_x[4:6]
            transform[4:6, 4:6] = rotation
        self.position_P = transform @ self.position_P @ transform.T
        self.yaw_x[0] = _wrap_angle(
            float(self.yaw_x[0])
            + math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
