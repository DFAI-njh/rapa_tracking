"""RoadTrack V3: a RAPA-R-optimized radar-only online 3D tracker.

The public adapter intentionally matches the small interface used by the
Bosch ROS inference node while keeping the implementation independent from
SimpleTrack. Boxes use ``[x, y, z, length, width, height, yaw]`` throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from rapa_tracking.preprocess import class_agnostic_rotated_3d_nms
from .geometry.metrics import box_iou_giou_3d, box_iou_giou_bev
from .types import TrackOutput
from .track.track import _Track
from .radar import RadarMotionEstimate, assign_points_to_boxes, estimate_motion
from .utils.angle import _wrap_angle, _yaw_residual


_LARGE_COST = 1.0e6


def _deep_update(target: Dict, override: Dict) -> None:
    """Recursively apply a mode profile without sharing nested dictionaries."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


class RoadTrackV3:
    """Core tracker operating in one consistent Cartesian coordinate frame."""

    def __init__(self, config: Dict):
        self.config = config
        self.tracks: List[_Track] = []
        self.next_track_id = 1
        self.last_timestamp_ns: Optional[int] = None
        self.last_diagnostics = self._empty_diagnostics()
        self._radar_motions = []

    def _empty_diagnostics(self) -> Dict:
        return {
            'mode': self.config.get('mode', '3d_optimized'),
            'metric': self.config['association']['metric'],
            'high_matches': 0, 'tentative_matches': 0, 'low_matches': 0,
            'reid_matches': 0, 'births': 0, 'reactivations': 0,
            'reidentifications': 0, 'missed': 0,
            'match_details': [], 'rejected_pairs': {},
        }

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
        self.last_timestamp_ns = None
        self.last_diagnostics = self._empty_diagnostics()

    def transform_frame(self, pose: np.ndarray) -> None:
        for track in self.tracks:
            track.transform_frame(pose)

    def _association_cost(self, track: _Track, detection: np.ndarray,
                          timestamp_ns: int, gate: Dict,
                          detection_index: Optional[int] = None) -> float:
        track_box = track.box
        mode = self.config.get('mode', '3d_optimized')
        is_bev = mode == 'bev_optimized'
        center_distance = float(np.linalg.norm(detection[:2] - track_box[:2]))
        adaptive_center_gate = (
            float(gate['max_center_distance'])
            + min(track.misses, int(gate.get('miss_growth_limit', 3)))
            * float(gate.get('miss_distance_growth', 0.0)))
        if center_distance > adaptive_center_gate:
            return _LARGE_COST
        if (not is_bev and
                abs(float(detection[2] - track_box[2])) > float(gate['max_z_difference'])):
            return _LARGE_COST
        size_slice = slice(3, 5) if is_bev else slice(3, 6)
        ratios = (np.asarray(detection[size_slice], dtype=np.float64)
                  / np.maximum(track_box[size_slice], 1e-3))
        if (np.any(ratios < float(gate['min_size_ratio']))
                or np.any(ratios > float(gate['max_size_ratio']))):
            return _LARGE_COST
        mahalanobis = track.filter.mahalanobis_position_squared(
            detection[:3], self.config['association']['mahalanobis_dimensions'])
        mahalanobis_gate = float(gate['mahalanobis_threshold'])
        if not math.isfinite(mahalanobis) or mahalanobis > mahalanobis_gate:
            return _LARGE_COST
        radar_config = self.config.get('radar', {})
        if (radar_config.get('association_gate', False)
                and detection_index is not None
                and detection_index < len(self._radar_motions)):
            motion = self._radar_motions[detection_index]
            if (motion is not None and motion.median_vr is not None
                    and motion.quality >= float(radar_config.get('min_quality', 0.))):
                los = motion.los_xy
                predicted = float(los @ track.filter.velocity[:2])
                residual = float(motion.median_vr - predicted)
                velocity_covariance = track.filter.position_P[2:4, 2:4]
                variance = float(los @ velocity_covariance @ los
                                 + motion.radial_variance)
                normalized = residual ** 2 / max(variance, 1e-6)
                if normalized > float(radar_config.get(
                        'radial_gate_mahalanobis', 9.21)):
                    return _LARGE_COST
        velocity_consistency = self.config['association']['velocity_consistency']
        velocity_cost = 0.0
        observation_dt = (int(timestamp_ns) - track.last_observation_ns) / 1e9
        if (velocity_consistency['enabled']
                and track.hits >= int(velocity_consistency['min_track_hits'])
                and float(velocity_consistency['min_dt']) <= observation_dt
                <= float(velocity_consistency['max_dt'])):
            observed_velocity = (
                np.asarray(detection[:3], dtype=np.float64)
                - track.last_observation[:3]) / observation_dt
            velocity_dimensions = 2 if is_bev else 3
            velocity_residual = float(np.linalg.norm(
                observed_velocity[:velocity_dimensions]
                - track.filter.velocity[:velocity_dimensions]))
            adaptive_velocity_gate = (
                float(velocity_consistency['max_residual_mps'])
                + min(track.misses,
                      int(velocity_consistency['miss_growth_limit']))
                * float(velocity_consistency['miss_growth_mps']))
            if velocity_residual > adaptive_velocity_gate:
                return _LARGE_COST
            velocity_cost = float(np.clip(
                velocity_residual / adaptive_velocity_gate, 0.0, 1.0))
        shape_penalty = min(float(np.mean(np.abs(np.log(np.maximum(ratios, 1e-6))))), 1.0)
        velocity = track.filter.velocity[:2]
        speed = float(np.linalg.norm(velocity))
        displacement = np.asarray(detection[:2]) - track.last_observation[:2]
        displacement_norm = float(np.linalg.norm(displacement))
        if (speed >= self.config['association']['direction_min_speed']
                and displacement_norm >= self.config['association']['direction_min_displacement']
                and timestamp_ns > track.last_observation_ns):
            cosine = float(np.clip(
                np.dot(velocity, displacement) / (speed * displacement_norm), -1.0, 1.0))
            direction_gate = self.config['association']['direction_hard_gate']
            if (direction_gate['enabled']
                    and track.status != 'dormant'
                    and track.hits >= int(direction_gate['min_track_hits'])
                    and cosine < float(direction_gate['min_cosine'])):
                return _LARGE_COST

        association = self.config['association']
        metric = association['metric']
        _, giou_similarity = box_iou_giou_3d(track_box, detection)
        giou_cost = float(np.clip(
            (1.0 - giou_similarity) / 2.0, 0.0, 1.0))
        yaw_cost = float(np.clip(
            abs(_yaw_residual(float(detection[6]), float(track_box[6])))
            / (math.pi / 2.0), 0.0, 1.0))
        if metric == 'giou3d':
            cost = giou_cost
        elif metric == 'bev_hybrid':
            _, similarity = box_iou_giou_bev(track_box, detection)
            bev_giou_cost = float(np.clip((1. - similarity) / 2., 0., 1.))
            delta = np.asarray(detection[:2] - track_box[:2], dtype=np.float64)
            cosine, sine = math.cos(float(track_box[6])), math.sin(float(track_box[6]))
            longitudinal = abs(float(cosine * delta[0] + sine * delta[1]))
            lateral = abs(float(-sine * delta[0] + cosine * delta[1]))
            miss_count = min(track.misses, int(gate.get('miss_growth_limit', 3)))
            longitudinal_gate = (float(gate['max_longitudinal_distance'])
                                 + miss_count * float(gate.get(
                                     'miss_longitudinal_growth', 0.0)))
            lateral_gate = (float(gate['max_lateral_distance'])
                            + miss_count * float(gate.get(
                                'miss_lateral_growth', 0.0)))
            if longitudinal > longitudinal_gate or lateral > lateral_gate:
                return _LARGE_COST
            longitudinal_cost = float(np.clip(
                longitudinal / max(longitudinal_gate, 1e-6), 0., 1.))
            lateral_cost = float(np.clip(
                lateral / max(lateral_gate, 1e-6), 0., 1.))
            weights = association['bev_weights']
            cost = (
                weights['giou'] * bev_giou_cost
                + weights['longitudinal'] * longitudinal_cost
                + weights['lateral'] * lateral_cost
                + weights['velocity'] * velocity_cost
                + weights['yaw'] * yaw_cost
                + weights['size'] * shape_penalty)
        else:
            raise RuntimeError(f'unsupported internal association metric: {metric}')
        return float(cost) if cost <= float(gate['max_cost']) else _LARGE_COST

    def _associate(self, track_indices: Sequence[int], detection_indices: Sequence[int],
                   detections: np.ndarray, timestamp_ns: int,
                   gate: Dict,
                   detection_penalties: Optional[Dict[int, float]] = None,
                   stage: str = 'unknown',
                   ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        track_indices, detection_indices = list(track_indices), list(detection_indices)
        if not track_indices or not detection_indices:
            return [], track_indices, detection_indices
        cost = np.full((len(track_indices), len(detection_indices)),
                       _LARGE_COST, dtype=np.float64)
        for row, track_index in enumerate(track_indices):
            for column, detection_index in enumerate(detection_indices):
                candidate_cost = self._association_cost(
                    self.tracks[track_index], detections[detection_index],
                    timestamp_ns, gate, detection_index)
                if candidate_cost >= _LARGE_COST:
                    rejected = self.last_diagnostics['rejected_pairs']
                    rejected[stage] = int(rejected.get(stage, 0)) + 1
                    continue
                if detection_penalties:
                    candidate_cost += float(
                        detection_penalties.get(detection_index, 0.0))
                if candidate_cost <= float(gate['max_cost']):
                    cost[row, column] = candidate_cost
        rows, columns = linear_sum_assignment(cost)
        matches: List[Tuple[int, int]] = []
        matched_tracks, matched_detections = set(), set()
        for row, column in zip(rows, columns):
            if cost[row, column] >= _LARGE_COST:
                continue
            track_index, detection_index = track_indices[row], detection_indices[column]
            matches.append((track_index, detection_index))
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
        return (matches,
                [index for index in track_indices if index not in matched_tracks],
                [index for index in detection_indices if index not in matched_detections])

    def _associate_cascade(self, track_indices: Sequence[int],
                           detection_indices: Sequence[int], detections: np.ndarray,
                           timestamp_ns: int, gate: Dict,
                           detection_penalties: Optional[Dict[int, float]] = None,
                           stage: str = 'unknown'):
        remaining_detections = list(detection_indices)
        matches: List[Tuple[int, int]] = []
        unmatched_tracks: List[int] = []
        miss_ages = sorted(set(self.tracks[index].misses for index in track_indices))
        for miss_age in miss_ages:
            group = [index for index in track_indices
                     if self.tracks[index].misses == miss_age]
            group_matches, group_unmatched, remaining_detections = self._associate(
                group, remaining_detections, detections, timestamp_ns, gate,
                detection_penalties, stage)
            matches.extend(group_matches)
            unmatched_tracks.extend(group_unmatched)
        return matches, unmatched_tracks, remaining_detections

    def _record_match_details(self, stage: str, matches: Sequence[Tuple[int, int]],
                              detections: np.ndarray, scores: np.ndarray,
                              timestamp_ns: int, gate: Dict) -> None:
        if not bool(self.config.get('debug', {}).get('record_match_details', False)):
            return
        details = self.last_diagnostics['match_details']
        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            track_box = track.box
            detection = detections[detection_index]
            center_distance = float(np.linalg.norm(detection[:2] - track_box[:2]))
            _, giou = box_iou_giou_3d(track_box, detection)
            ratios = np.asarray(detection[3:6]) / np.maximum(track_box[3:6], 1e-3)
            size_difference = float(np.mean(
                np.abs(np.log(np.maximum(ratios, 1e-6)))))
            mahalanobis = track.filter.mahalanobis_position_squared(
                detection[:3], self.config['association']['mahalanobis_dimensions'])
            details.append({
                'track_id': int(track.track_id),
                'detection_index': int(detection_index),
                'stage': str(stage),
                'total_cost': float(self._association_cost(
                    track, detection, timestamp_ns, gate, detection_index)),
                'center_distance': center_distance,
                'giou3d': float(giou),
                'yaw_difference': abs(_yaw_residual(
                    float(detection[6]), float(track_box[6]))),
                'size_difference': size_difference,
                'mahalanobis': float(mahalanobis),
                'detection_score': float(scores[detection_index]),
                'track_state': str(track.status),
            })

    def _is_duplicate_birth(self, detection: np.ndarray) -> bool:
        duplicate = self.config['duplicate_birth']
        for track in self.tracks:
            if track.status in ('dead', 'dormant'):
                continue
            center_distance = float(np.linalg.norm(detection[:2] - track.box[:2]))
            if center_distance > duplicate['max_center_distance']:
                continue
            iou, _ = box_iou_giou_3d(track.box, detection)
            if iou >= duplicate['min_iou']:
                return True
        return False

    def update(self, detections: np.ndarray, scores: np.ndarray, labels: np.ndarray,
               timestamp_ns: int,
               radar_motions: Optional[Sequence[RadarMotionEstimate]] = None,
               ) -> List[_Track]:
        detections = np.asarray(detections, dtype=np.float64).reshape(-1, 7)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if radar_motions is None:
            radar_motions = [None] * len(detections)
        self._radar_motions = radar_motions
        timestamp_ns = int(timestamp_ns)
        self.last_diagnostics = self._empty_diagnostics()
        if self.last_timestamp_ns is not None and timestamp_ns <= self.last_timestamp_ns:
            self.reset()
        if self.last_timestamp_ns is None:
            dt = float(self.config['default_dt'])
        else:
            dt = np.clip((timestamp_ns - self.last_timestamp_ns) / 1e9,
                         self.config['min_predict_dt'], self.config['max_predict_dt'])
        self.last_timestamp_ns = timestamp_ns
        for track in self.tracks:
            track.predict(float(dt))

        low_threshold = float(self.config['low_score_threshold'])
        high_threshold = float(self.config['high_score_threshold'])
        high_detections = [index for index, score in enumerate(scores)
                           if score >= high_threshold]
        low_detections = [index for index, score in enumerate(scores)
                          if low_threshold <= score < high_threshold]
        established = [index for index, track in enumerate(self.tracks)
                       if track.status in ('confirmed', 'lost')]
        tentative = [index for index, track in enumerate(self.tracks)
                     if track.status == 'tentative']
        dormant = [index for index, track in enumerate(self.tracks)
                   if track.status == 'dormant']

        joint = self.config['association']['confidence_aware_joint']
        if joint['enabled']:
            high_set, low_set = set(high_detections), set(low_detections)
            joint_detections = high_detections + low_detections
            penalties = {
                index: float(joint['low_score_cost_penalty'])
                for index in low_detections
            }
            joint_matches, unmatched_established, remaining_joint = (
                self._associate_cascade(
                    established, joint_detections, detections, timestamp_ns,
                    self.config['association']['high'], penalties, 'established_joint'))
            high_matches = [match for match in joint_matches
                            if match[1] in high_set]
            low_matches = [match for match in joint_matches
                           if match[1] in low_set]
            remaining_high = [index for index in remaining_joint
                              if index in high_set]
            remaining_low = [index for index in remaining_joint
                             if index in low_set]
        else:
            high_matches, unmatched_established, remaining_high = (
                self._associate_cascade(
                    established, high_detections, detections, timestamp_ns,
                    self.config['association']['high'], stage='established_high'))
            low_matches = []
            remaining_low = low_detections
        tentative_matches, unmatched_tentative, remaining_high = self._associate(
            tentative, remaining_high, detections, timestamp_ns,
            self.config['association']['tentative'], stage='tentative_high')
        low_rescue_matches, unmatched_established, _ = self._associate_cascade(
            unmatched_established, remaining_low, detections, timestamp_ns,
            self.config['association']['low'], stage='established_low')
        low_matches.extend(low_rescue_matches)
        # Dormant identities compete globally. A recency cascade can let a
        # newer but spatially implausible dormant track steal a detection from
        # an older, much closer identity before the cost matrix is considered.
        reid_matches, unmatched_dormant, remaining_high = self._associate(
            dormant, remaining_high, detections, timestamp_ns,
            self.config['association']['reid'], stage='dormant_reid')

        self.last_diagnostics.update({
            'high_matches': len(high_matches),
            'tentative_matches': len(tentative_matches),
            'low_matches': len(low_matches),
            'reid_matches': len(reid_matches),
        })

        self._record_match_details(
            'established_high', high_matches, detections, scores, timestamp_ns,
            self.config['association']['high'])
        self._record_match_details(
            'tentative_high', tentative_matches, detections, scores, timestamp_ns,
            self.config['association']['tentative'])
        self._record_match_details(
            'established_low', low_matches, detections, scores, timestamp_ns,
            self.config['association']['low'])
        self._record_match_details(
            'dormant_reid', reid_matches, detections, scores, timestamp_ns,
            self.config['association']['reid'])

        matched_track_indices = set()
        for kind, matches in (('high', high_matches), ('high', tentative_matches),
                              ('low', low_matches), ('reid', reid_matches)):
            for track_index, detection_index in matches:
                track = self.tracks[track_index]
                was_lost = track.status == 'lost'
                was_dormant = track.status == 'dormant'
                track.update(detections[detection_index], scores[detection_index],
                             labels[detection_index], timestamp_ns, kind, self.config,
                             radar_motions[detection_index])
                if (track.status == 'tentative'
                        and track.confirmation_ready(self.config)):
                    track.status = 'confirmed'
                elif track.status == 'lost':
                    track.status = 'confirmed'
                elif track.status == 'dormant':
                    track.status = 'confirmed'
                if was_lost:
                    self.last_diagnostics['reactivations'] += 1
                if was_dormant:
                    self.last_diagnostics['reactivations'] += 1
                    self.last_diagnostics['reidentifications'] += 1
                matched_track_indices.add(track_index)

        for track_index in (unmatched_established + unmatched_tentative
                            + unmatched_dormant):
            if track_index not in matched_track_indices:
                self.tracks[track_index].mark_missed(timestamp_ns, self.config)
                self.last_diagnostics['missed'] += 1

        for detection_index in remaining_high:
            if scores[detection_index] < self.config['new_track_score_threshold']:
                continue
            if self._is_duplicate_birth(detections[detection_index]):
                continue
            track = _Track(self.next_track_id, detections[detection_index],
                           scores[detection_index], labels[detection_index],
                           timestamp_ns, self.config)
            if int(self.config['confirmation']['hits']) <= 1:
                track.status = 'confirmed'
            self.next_track_id += 1
            self.tracks.append(track)
            self.last_diagnostics['births'] += 1

        self.tracks = [track for track in self.tracks if track.status != 'dead']
        return list(self.tracks)


class RoadTrackV3Adapter:
    """Inference-node adapter with optional map-frame ego-motion compensation."""

    def __init__(self, config_path: str, input_score_threshold: Optional[float] = None,
                 output_mode: str = 'all', min_hits_to_birth: Optional[int] = None,
                 association_metric: Optional[str] = None,
                 input_nms_threshold: Optional[float] = None):
        with Path(config_path).open('r', encoding='utf-8') as stream:
            loaded = yaml.safe_load(stream) or {}
        config = loaded.get('roadtrack_v3', loaded.get('roadtrack_v2', loaded))
        profiles = config.pop('profiles', {})
        requested_mode = association_metric or config.get('mode', '3d_optimized')
        if requested_mode not in profiles:
            raise ValueError(
                f'Unsupported RoadTrack V3 mode {requested_mode!r}; '
                f'choose from {sorted(profiles)}')
        _deep_update(config, profiles[requested_mode])
        config['mode'] = requested_mode
        if input_score_threshold is not None:
            config['low_score_threshold'] = max(
                float(config['low_score_threshold']), float(input_score_threshold))
        if min_hits_to_birth is not None:
            config['min_hits_to_birth'] = int(min_hits_to_birth)
            config['confirmation']['hits'] = int(min_hits_to_birth)
            config['confirmation']['window'] = max(
                int(config['confirmation']['window']), int(min_hits_to_birth))
        if input_nms_threshold is not None:
            config['input_nms']['iou_threshold'] = float(input_nms_threshold)
        self._validate_config(config)
        self.config = config
        self.core = RoadTrackV3(config)
        self.output_mode = output_mode
        self.min_hits_to_birth = int(config['confirmation']['hits'])
        self.miss_score_decay = float(config['score_miss_decay'])
        self.hit_score_gain = float(config['score_hit_gain'])
        self.track_nms_threshold = None
        self.last_track_nms_suppressed = 0
        self.last_input_nms_suppressed = 0
        self.yaw_profiles = False
        self.association_metric = config['association']['metric']
        self.association_mode = config['mode']
        self.last_diagnostics = self.core.last_diagnostics
        self._world_enabled = False
        self._last_ego_pose: Optional[np.ndarray] = None

    @staticmethod
    def _validate_config(config: Dict) -> None:
        if not 0 <= config['low_score_threshold'] < config['high_score_threshold'] <= 1:
            raise ValueError('RoadTrack V3 requires 0 <= low_score_threshold < high_score_threshold <= 1')
        if config['new_track_score_threshold'] < config['high_score_threshold']:
            raise ValueError('new_track_score_threshold must be >= high_score_threshold')
        if int(config['min_hits_to_birth']) < 1:
            raise ValueError('min_hits_to_birth must be >= 1')
        if not 0.0 <= float(config['input_nms']['iou_threshold']) <= 1.0:
            raise ValueError('input_nms.iou_threshold must be within [0, 1]')
        reidentification = config['reidentification']
        if (reidentification['enabled']
                and int(reidentification['max_age_since_update'])
                <= int(config['max_age_since_update'])):
            raise ValueError(
                'reidentification.max_age_since_update must be greater than '
                'max_age_since_update when re-identification is enabled')
        if config['mode'] not in ('3d_optimized', 'bev_optimized'):
            raise ValueError('RoadTrack V3 exposes only 3d_optimized and bev_optimized')
        supported_metrics = {'giou3d', 'bev_hybrid'}
        metric = config['association']['metric']
        if metric not in supported_metrics:
            raise ValueError(
                f'Unsupported RoadTrack association metric {metric!r}; '
                f'choose from {sorted(supported_metrics)}')
        if config['association']['mahalanobis_dimensions'] not in ('xy', 'xyz'):
            raise ValueError('mahalanobis_dimensions must be xy or xyz')
        velocity_consistency = config['association']['velocity_consistency']
        if float(velocity_consistency['max_residual_mps']) <= 0:
            raise ValueError('velocity_consistency.max_residual_mps must be positive')
        if int(velocity_consistency['min_track_hits']) < 2:
            raise ValueError('velocity_consistency.min_track_hits must be at least 2')
        direction_gate = config['association']['direction_hard_gate']
        if not -1.0 <= float(direction_gate['min_cosine']) <= 1.0:
            raise ValueError('direction_hard_gate.min_cosine must be within [-1, 1]')
        joint = config['association']['confidence_aware_joint']
        if float(joint['low_score_cost_penalty']) < 0:
            raise ValueError(
                'confidence_aware_joint.low_score_cost_penalty must be non-negative')
        bev_weights = config['association']['bev_weights']
        if not math.isclose(sum(float(value) for value in bev_weights.values()), 1.0,
                            rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('RoadTrack V3 BEV association weights must sum to 1')
        confirmation = config['confirmation']
        if not 1 <= int(confirmation['hits']) <= int(confirmation['window']):
            raise ValueError('confirmation requires 1 <= hits <= window')
        lifecycle = config['lifecycle']
        if not (0 < float(lifecycle['tentative_timeout_seconds'])
                <= float(lifecycle['reid_max_lost_seconds'])):
            raise ValueError('invalid RoadTrack V3 lifecycle timeouts')
        if not (0 < float(lifecycle['visible_max_lost_seconds'])
                < float(lifecycle['reid_max_lost_seconds'])):
            raise ValueError('reid timeout must exceed visible timeout')

    def reset(self) -> None:
        self.core.reset()
        self._world_enabled = False
        self._last_ego_pose = None

    @staticmethod
    def _transform_boxes(boxes: np.ndarray, pose: np.ndarray) -> np.ndarray:
        output = np.asarray(boxes, dtype=np.float64).copy()
        if not len(output):
            return output.reshape(-1, 7)
        rotation = pose[:2, :2]
        output[:, :2] = output[:, :2] @ rotation.T + pose[:2, 2]
        yaw_delta = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        output[:, 6] = np.arctan2(
            np.sin(output[:, 6] + yaw_delta), np.cos(output[:, 6] + yaw_delta))
        return output

    @staticmethod
    def _inverse_transform_box(box: np.ndarray, pose: np.ndarray) -> np.ndarray:
        output = np.asarray(box, dtype=np.float64).copy()
        rotation = pose[:2, :2]
        output[:2] = rotation.T @ (output[:2] - pose[:2, 2])
        yaw_delta = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        output[6] = _wrap_angle(float(output[6]) - yaw_delta)
        return output

    def _input_nms(self, boxes: np.ndarray, scores: np.ndarray,
                   labels: np.ndarray):
        """Class-agnostic rotated-3D NMS over raw detector measurements.

        RAPA-R already performs per-class NMS. This second, configurable pass
        removes cross-class duplicates before they can create independent
        track state. It never reads or expects a detector-side identity.
        """
        config = self.config['input_nms']
        if not config['enabled'] or len(boxes) < 2:
            self.last_input_nms_suppressed = 0
            return boxes, scores, labels
        boxes, scores, labels, suppressed = class_agnostic_rotated_3d_nms(
            boxes, scores, labels, float(config['iou_threshold']),
            box_iou_giou_3d)
        self.last_input_nms_suppressed = suppressed
        return boxes, scores, labels

    def update(self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
               timestamp_ns: int, ego_pose: Optional[np.ndarray] = None,
               radar_points: Optional[np.ndarray] = None) -> List[TrackOutput]:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        boxes, scores, labels = self._input_nms(boxes, scores, labels)
        radar_motions = None
        radar_config = self.config.get('radar', {})
        if radar_points is not None and radar_config.get('enabled', False):
            groups = assign_points_to_boxes(
                np.asarray(radar_points), boxes,
                radar_config.get('box_margin', [0., 0., 0.]))
            radar_motions = [estimate_motion(
                np.asarray(radar_points)[indices], radar_config, boxes[index, :2])
                for index, indices in enumerate(groups)]
        if ego_pose is not None:
            ego_pose = np.asarray(ego_pose, dtype=np.float64).reshape(3, 3)
            if not self._world_enabled:
                # Rebase already-created ego-frame tracks without a visual jump
                # when the first interpolated navigation pose becomes available.
                self.core.transform_frame(ego_pose)
                self._world_enabled = True
            self._last_ego_pose = ego_pose
        active_pose = self._last_ego_pose if self._world_enabled else None
        tracking_boxes = (
            self._transform_boxes(boxes, active_pose)
            if active_pose is not None else boxes)
        tracks = self.core.update(
            tracking_boxes, scores, labels, timestamp_ns, radar_motions)
        self.last_diagnostics = dict(self.core.last_diagnostics)
        self.last_diagnostics['input_nms_suppressed'] = self.last_input_nms_suppressed
        outputs: List[TrackOutput] = []
        for track in tracks:
            # Dormant tracks retain only KF/ID memory. They are deliberately
            # hidden so the normal five-frame visualization lifetime remains.
            if track.status in ('dormant', 'dead'):
                continue
            if self.output_mode == 'observed' and not track.updated:
                continue
            box = track.box
            velocity = np.asarray(track.filter.velocity, dtype=np.float64).copy()
            if active_pose is not None:
                box = self._inverse_transform_box(box, active_pose)
                # The filter state uses the pose/world orientation in this
                # mode. Rotate its vector back into the published ego axes;
                # translation must never be applied to a velocity vector.
                velocity[:2] = active_pose[:2, :2].T @ velocity[:2]
            if track.status == 'tentative':
                state = f'birth_{track.consecutive_hits}'
            elif track.update_kind == 'reid':
                state = 'reidentified'
            elif track.updated:
                state = 'update_low' if track.update_kind == 'low' else 'update_high'
            else:
                state = f'prediction_{track.misses}'
            outputs.append(TrackOutput(
                box=np.asarray(box, dtype=np.float32), score=float(track.score),
                label=int(track.label), track_id=int(track.track_id), state=state,
                velocity=np.asarray(velocity, dtype=np.float32)))
        return outputs
