from pathlib import Path

import numpy as np
import yaml

from rapa_tracking.visualization.renderer import SimpleTrackVisualizationRenderer


CONFIG = Path(__file__).with_name('config.yaml')


def renderer():
    instance = SimpleTrackVisualizationRenderer.__new__(
        SimpleTrackVisualizationRenderer)
    instance.cfg = yaml.safe_load(CONFIG.read_text())['tracking_visualization']
    return instance


def box(x, y=0.0):
    return np.asarray([x, y, 0.0, 4.5, 1.9, 1.6, 0.0], np.float32)


def test_collision_risk_is_high_for_imminent_closing_motion():
    risk = renderer()._collision_risk_percent(box(20.0), [-10.0, 0.0, 0.0])
    assert 50.0 < risk < 100.0


def test_collision_risk_is_zero_for_far_separating_motion():
    risk = renderer()._collision_risk_percent(box(20.0), [10.0, 0.0, 0.0])
    assert risk == 0.0


def test_collision_risk_is_bounded_for_lateral_near_miss():
    risk = renderer()._collision_risk_percent(box(10.0, 8.0), [0.0, -5.0, 0.0])
    assert 0.0 <= risk < 50.0


def test_collision_risk_is_100_for_current_footprint_overlap():
    risk = renderer()._collision_risk_percent(box(1.0), [0.0, 0.0, 0.0])
    assert risk == 100.0


def test_original_smoothstep_configuration_is_restored():
    config = renderer().cfg
    assert config['interpolation_publish_hz'] == 30.0
    assert config['lane_damping_gain'] == 0.05
    assert 'position_smooth_time_sec' not in config
    assert 'lane_follow_time_constant_sec' not in config


def test_original_box_interpolation_keeps_shortest_yaw_path():
    start = box(0.0)
    start[6] = np.deg2rad(179.0)
    target = box(10.0)
    target[6] = np.deg2rad(-179.0)
    middle = SimpleTrackVisualizationRenderer._interpolate_box(
        start, target, 0.5)
    assert np.isclose(middle[0], 5.0)
    assert abs(abs(float(middle[6])) - np.pi) < np.deg2rad(0.1)
