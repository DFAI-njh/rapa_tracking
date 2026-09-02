from pathlib import Path

import numpy as np
import yaml

from rapa_tracking.roadtrack_v3 import (
    RoadTrackV3, RoadTrackV3Adapter, box_iou_diou_3d, box_iou_giou_3d,
    odiou3d_cost,
)

RoadTrack = RoadTrackV3
RoadTrackAdapter = RoadTrackV3Adapter


CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'bosch_roadtrack_v3.yaml'


def box(x=0.0, y=0.0, length=4.5, label=0):
    del label
    return np.asarray([x, y, 0.0, length, 1.9, 1.6, 0.0], dtype=np.float32)


def update(adapter, timestamp, boxes, scores=None, labels=None, pose=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    count = len(boxes)
    scores = np.asarray(scores if scores is not None else [.9] * count, dtype=np.float32)
    labels = np.asarray(labels if labels is not None else [0] * count, dtype=np.int64)
    return adapter.update(boxes, scores, labels, int(timestamp * 1e9), pose)


def test_rotated_iou_identity_and_separation():
    iou, giou = box_iou_giou_3d(box(), box())
    assert abs(iou - 1.0) < 1e-6
    assert abs(giou - 1.0) < 1e-6
    iou, giou = box_iou_giou_3d(box(), box(x=20))
    assert iou == 0.0 and giou < 0.0


def test_nonoverlap_diou_orders_distance_and_differs_from_giou():
    near, far = box(x=6.0), box(x=12.0)
    _, near_giou = box_iou_giou_3d(box(), near)
    _, far_giou = box_iou_giou_3d(box(), far)
    near_iou, near_diou = box_iou_diou_3d(box(), near)
    far_iou, far_diou = box_iou_diou_3d(box(), far)
    assert near_iou == far_iou == 0.0
    assert near_diou > far_diou
    assert near_giou > far_giou
    assert not np.isclose(near_diou, near_giou)


def test_odiou_pi_equivalence_and_right_angle_penalty():
    square = box().copy()
    square[3] = square[4] = 4.0
    opposite, right_angle = square.copy(), square.copy()
    opposite[6] = np.pi
    right_angle[6] = np.pi / 2
    assert odiou3d_cost(square, opposite, yaw_weight=.2) < 1e-6
    assert odiou3d_cost(square, right_angle, yaw_weight=.2) > .19


def test_mahalanobis_hard_gate_rejects_uncertain_geometry_match():
    config = RoadTrackAdapter(str(CONFIG)).config
    tracker = RoadTrack(config)
    tracker.update(np.asarray([box()]), np.asarray([.9]), np.asarray([0]), 100_000_000)
    tracker.tracks[0].predict(.1)
    gate = dict(config['association']['tentative'])
    gate['max_center_distance'] = 100.0
    gate['mahalanobis_threshold'] = .01
    cost = tracker._association_cost(
        tracker.tracks[0], box(x=1.0), 200_000_000, gate)
    assert cost >= 1e6


def test_birth_requires_three_consecutive_high_hits():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    assert update(tracker, 0.1, [box()])[0].state == 'birth_1'
    assert update(tracker, 0.2, [box(x=.3)])[0].state == 'birth_2'
    outputs = update(tracker, 0.3, [box(x=.6)])
    assert len(outputs) == 1 and outputs[0].track_id == 1
    assert outputs[0].state == 'update_high'


def test_n_of_m_birth_tolerates_one_radar_dropout(tmp_path):
    loaded = yaml.safe_load(CONFIG.read_text())
    loaded['roadtrack_v3']['confirmation'].update({
        'hits': 3, 'window': 5, 'require_consecutive': False,
    })
    config_path = tmp_path / 'n_of_m.yaml'
    config_path.write_text(yaml.safe_dump(loaded))
    tracker = RoadTrackAdapter(str(config_path), output_mode='all')
    first = update(tracker, 0.1, [box()])
    original_id = first[0].track_id
    update(tracker, 0.2, [])
    outputs = update(tracker, 0.3, [box(x=.3)])
    assert len(outputs) == 1 and outputs[0].track_id == original_id
    assert outputs[0].state == 'birth_1'
    outputs = update(tracker, 0.4, [box(x=.6)])
    assert outputs[0].track_id == original_id
    assert outputs[0].state == 'update_high'


def test_low_score_preserves_confirmed_id_but_never_births():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    for frame in range(1, 4):
        outputs = update(tracker, frame / 10, [box(x=frame * .5)])
    track_id = outputs[0].track_id
    outputs = update(tracker, .4, [box(x=2.0)], scores=[.3])
    assert len(outputs) == 1 and outputs[0].track_id == track_id
    assert outputs[0].state == 'update_low'
    empty = RoadTrackAdapter(str(CONFIG), output_mode='all')
    assert update(empty, .1, [box()], scores=[.3]) == []


def test_established_track_prefers_better_low_candidate_over_high_neighbor():
    # This test isolates confidence-aware association. The production
    # class-agnostic input NMS is tested separately and would intentionally
    # remove one of these substantially overlapping synthetic boxes first.
    tracker = RoadTrackAdapter(
        str(CONFIG), output_mode='all', input_nms_threshold=1.0)
    for frame in range(1, 4):
        outputs = update(tracker, frame / 10, [box(x=.1 * frame)])
    original_id = outputs[0].track_id

    outputs = update(
        tracker, .4, [box(x=.5), box(x=3.5)], scores=[.3, .9])
    original = next(item for item in outputs if item.track_id == original_id)
    newborn = next(item for item in outputs if item.track_id != original_id)
    assert original.state == 'update_low'
    assert abs(float(original.box[0]) - .5) < 1.0
    assert newborn.state == 'birth_1'


def test_single_low_score_position_outlier_does_not_break_identity():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    for frame, x in enumerate([0.0, .2, .4], start=1):
        outputs = update(tracker, frame / 10, [box(x=x)])
    original_id = outputs[0].track_id

    noisy = update(tracker, .4, [box(x=3.4)], scores=[.3])
    assert noisy[0].track_id == original_id
    recovered = update(tracker, .5, [box(x=.8)], scores=[.9])
    assert any(item.track_id == original_id and item.state == 'update_high'
               for item in recovered)


def test_one_detection_updates_only_one_track():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    for frame in range(1, 4):
        update(tracker, frame / 10, [box(y=-2), box(y=2)])
    outputs = update(tracker, .4, [box(y=-1.6)], scores=[.3])
    updated = [item for item in outputs if item.state == 'update_low']
    predicted = [item for item in outputs if item.state.startswith('prediction_')]
    assert len(updated) == 1 and len(predicted) == 1


def test_input_nms_is_class_agnostic_and_keeps_higher_score_box():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    outputs = update(
        tracker, .1, [box(x=0.0), box(x=.05)],
        scores=[.7, .9], labels=[0, 1])
    assert len(outputs) == 1
    assert outputs[0].label == 1
    assert tracker.last_diagnostics['input_nms_suppressed'] == 1


def test_class_changes_after_ten_consecutive_matches_without_id_change():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    for frame in range(1, 4):
        outputs = update(tracker, frame / 10, [box(x=.1 * frame)], labels=[0])
    track_id = outputs[0].track_id
    for frame in range(4, 13):
        outputs = update(tracker, frame / 10, [box(x=.1 * frame)], labels=[1])
        assert outputs[0].label == 0 and outputs[0].track_id == track_id
    outputs = update(tracker, 1.3, [box(x=1.3)], labels=[1])
    assert outputs[0].label == 1 and outputs[0].track_id == track_id


def test_fast_motion_keeps_id_after_velocity_is_observed():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    identifiers = []
    # Five metres per frame is larger than the 4.5 m box length, so every
    # consecutive pair has zero 3D IoU.
    for frame, x in enumerate([0, 5, 10, 15, 20], start=1):
        outputs = update(tracker, frame / 10, [box(x=x)])
        identifiers.append(outputs[0].track_id)
    assert identifiers == [1, 1, 1, 1, 1]


def test_output_exposes_kf_velocity_in_published_box_axes():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    for frame, x in enumerate([0.0, 1.0, 2.0, 3.0], start=1):
        outputs = update(tracker, frame / 10, [box(x=x)])
    assert outputs[0].velocity.shape == (3,)
    assert np.all(np.isfinite(outputs[0].velocity))
    assert outputs[0].velocity[0] > 0.0


def test_implausible_lateral_velocity_jump_is_rejected():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    for frame, x in enumerate([0.0, 1.0, 2.0], start=1):
        outputs = update(tracker, frame / 10, [box(x=x)])
    original_id = outputs[0].track_id

    # A 6 m lateral jump in 0.1 s is not a plausible continuation even though
    # the deliberately broad geometric gate would otherwise accept it.
    outputs = update(tracker, .4, [box(x=3.0, y=9.0)])
    original = next(item for item in outputs if item.track_id == original_id)
    newborn = next(item for item in outputs if item.track_id != original_id)
    assert original.state == 'prediction_1'
    assert newborn.state == 'birth_1'


def test_hidden_dormant_track_reidentifies_without_extending_visible_ghost():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    for frame in range(1, 4):
        outputs = update(tracker, frame / 10, [box(x=.2 * frame)])
    original_id = outputs[0].track_id

    # Five predicted frames remain visible, then only hidden identity/KF state
    # is retained. This does not bring back the old long visual ghost.
    for frame in range(4, 10):
        outputs = update(tracker, frame / 10, [])
    assert outputs == []
    assert tracker.core.tracks[0].status == 'dormant'

    outputs = update(tracker, 1.0, [box(x=2.0)])
    assert len(outputs) == 1
    assert outputs[0].track_id == original_id
    assert outputs[0].state == 'reidentified'
    assert tracker.last_diagnostics['reidentifications'] == 1


def test_detection_after_reidentification_window_gets_new_id():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    for frame in range(1, 4):
        outputs = update(tracker, frame / 10, [box(x=.2 * frame)])
    original_id = outputs[0].track_id

    for frame in range(4, 35):
        update(tracker, frame / 10, [])
    outputs = update(tracker, 3.5, [box(x=7.0)])
    assert len(outputs) == 1
    assert outputs[0].track_id != original_id
    assert outputs[0].state == 'birth_1'


def test_dormant_reidentification_uses_best_global_cost_not_recency():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    for frame in range(1, 4):
        update(tracker, frame / 10, [box(x=0.0), box(x=8.0)])
    older_id = tracker.core.tracks[0].track_id
    tracker.core.tracks[0].status = 'dormant'
    tracker.core.tracks[0].misses = 20
    tracker.core.tracks[1].status = 'dormant'
    tracker.core.tracks[1].misses = 6

    outputs = update(tracker, .4, [box(x=.2)])
    assert len(outputs) == 1
    assert outputs[0].track_id == older_id
    assert outputs[0].state == 'reidentified'


def test_both_optimized_modes_are_runtime_switchable_and_reported():
    for mode, metric in (('3d_optimized', 'giou3d'),
                         ('bev_optimized', 'bev_hybrid')):
        tracker = RoadTrackAdapter(
            str(CONFIG), output_mode='all', association_metric=mode)
        update(tracker, .1, [box()])
        outputs = update(tracker, .2, [box(x=.2)])
        assert outputs[0].track_id == 1
        assert tracker.last_diagnostics['metric'] == metric
        assert tracker.last_diagnostics['mode'] == mode
        assert tracker.last_diagnostics['tentative_matches'] == 1


def test_ego_motion_keeps_stationary_world_object_stable():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    identifiers = []
    world_positions = []
    for frame, ego_x in enumerate([0.0, 1.0, 2.0, 3.0], start=1):
        pose = np.asarray([[1., 0., ego_x], [0., 1., 0.], [0., 0., 1.]])
        outputs = update(tracker, frame / 10, [box(x=20.0 - ego_x)], pose=pose)
        identifiers.append(outputs[0].track_id)
        world_positions.append(tracker.core.tracks[0].box[0])
    assert identifiers == [1, 1, 1, 1]
    assert max(world_positions) - min(world_positions) < .5


def test_size_filter_reduces_radar_dimension_jitter():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    raw_lengths = [4.0, 5.0, 3.8, 5.2, 4.1]
    filtered = []
    for frame, length in enumerate(raw_lengths, start=1):
        outputs = update(
            tracker, frame * .1, [box(x=.1 * frame, length=length)])
        filtered.append(float(outputs[0].box[3]))
    assert np.std(filtered) < np.std(raw_lengths)


def test_timestamp_lifecycle_is_independent_of_frame_count():
    tracker = RoadTrackAdapter(str(CONFIG), output_mode='all')
    for timestamp in (.10, .22, .31):
        outputs = update(tracker, timestamp, [box(x=timestamp)])
    original_id = outputs[0].track_id
    # A single long wall-clock gap exceeds the visible timeout even though it
    # is only one update call. Identity remains hidden for re-identification.
    assert update(tracker, 1.0, []) == []
    assert tracker.core.tracks[0].status == 'dormant'
    recovered = update(tracker, 1.1, [box(x=1.1)])
    assert recovered[0].track_id == original_id


def test_debug_match_detail_contains_radar_cost_components():
    config = RoadTrackAdapter(str(CONFIG)).config
    config['debug']['record_match_details'] = True
    tracker = RoadTrack(config)
    tracker.update(np.asarray([box()]), np.asarray([.9]), np.asarray([0]), 100_000_000)
    tracker.update(np.asarray([box(x=.2)]), np.asarray([.9]), np.asarray([0]), 200_000_000)
    detail = tracker.last_diagnostics['match_details'][0]
    assert {'track_id', 'detection_index', 'stage', 'total_cost',
            'center_distance', 'giou3d', 'yaw_difference', 'size_difference',
            'mahalanobis', 'detection_score', 'track_state'} <= set(detail)
