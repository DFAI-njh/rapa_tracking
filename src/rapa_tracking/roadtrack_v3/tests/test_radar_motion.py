import numpy as np

from rapa_tracking.roadtrack_v3.motion.models import ca_matrices, cv_matrices
from rapa_tracking.roadtrack_v3.radar import assign_points_to_boxes, estimate_motion


DTYPE = np.dtype([('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                  ('radial_velocity', 'f8')])


def cloud(angles, velocity=(8., -2.), outlier=None):
    angles = np.asarray(angles)
    result = np.zeros(len(angles), dtype=DTYPE)
    result['x'] = 20 * np.cos(angles)
    result['y'] = 20 * np.sin(angles)
    result['radial_velocity'] = (
        velocity[0] * np.cos(angles) + velocity[1] * np.sin(angles))
    if outlier is not None:
        result['radial_velocity'][outlier] = 100.
    return result


def cfg():
    return {'radial_min_support': 3, 'wls_min_support': 5,
            'velocity_std_floor': .5, 'variance_floor': .25,
            'mad_scale': 3.5, 'min_singular': .1,
            'max_condition': 30., 'min_spread': .08,
            'max_residual': 1., 'full_support': 12}


def test_oriented_assignment_is_exclusive():
    points = np.zeros(3, dtype=DTYPE)
    points['x'] = [0., 2., 10.]
    boxes = np.asarray([[0., 0., 0., 4., 2., 2., 0.],
                        [2., 0., 0., 4., 2., 2., 0.]])
    groups = assign_points_to_boxes(points, boxes)
    assert groups[0].tolist() == [0]
    assert groups[1].tolist() == [1]


def test_median_radial_rejects_outlier_without_inventing_cartesian():
    estimate = estimate_motion(
        cloud(np.zeros(9), (7., 11.), outlier=4), cfg(), [20., 0.])
    assert abs(estimate.median_vr - 7.) < 1e-6
    assert estimate.source == 'radial'
    assert estimate.cartesian_velocity is None


def test_wls_recovers_velocity_with_diverse_los():
    estimate = estimate_motion(
        cloud(np.linspace(-.4, .4, 15)), cfg(), [20., 0.])
    assert estimate.source == 'wls'
    assert np.allclose(estimate.cartesian_velocity, [8., -2.], atol=1e-6)


def test_parallel_los_rejects_wls_and_keeps_scalar_radial():
    estimate = estimate_motion(cloud(np.zeros(8), (6., 9.)), cfg(), [20., 0.])
    assert estimate.source == 'radial'
    assert estimate.cartesian_velocity is None
    assert abs(estimate.median_vr - 6.) < 1e-6


def test_tangential_motion_does_not_become_zero_cartesian_measurement():
    estimate = estimate_motion(cloud(np.zeros(8), (0., 12.)), cfg(), [20., 0.])
    assert abs(estimate.median_vr) < 1e-9
    assert estimate.cartesian_velocity is None


def test_cv_ca_process_covariance_is_psd():
    for transition, process in (cv_matrices(.1, 2.), ca_matrices(.1, 2.)):
        assert transition.shape == process.shape
        assert np.min(np.linalg.eigvalsh(process)) >= -1e-10
