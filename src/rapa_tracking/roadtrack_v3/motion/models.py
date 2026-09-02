"""Planar CV and CA transition/process covariance builders."""

import numpy as np


def cv_matrices(dt, acceleration_std):
    transition = np.eye(4)
    transition[0, 2] = transition[1, 3] = dt
    q = float(acceleration_std) ** 2 * np.asarray([
        [.25 * dt ** 4, .5 * dt ** 3], [.5 * dt ** 3, dt ** 2]])
    process = np.zeros((4, 4))
    process[np.ix_([0, 2], [0, 2])] = q
    process[np.ix_([1, 3], [1, 3])] = q
    return transition, process


def ca_matrices(dt, jerk_std):
    transition = np.eye(6)
    transition[0, 2] = transition[1, 3] = dt
    transition[0, 4] = transition[1, 5] = .5 * dt ** 2
    transition[2, 4] = transition[3, 5] = dt
    q = float(jerk_std) ** 2 * np.asarray([
        [dt ** 6 / 36, dt ** 5 / 12, dt ** 4 / 6],
        [dt ** 5 / 12, dt ** 4 / 4, dt ** 3 / 2],
        [dt ** 4 / 6, dt ** 3 / 2, dt ** 2],
    ])
    process = np.zeros((6, 6))
    process[np.ix_([0, 2, 4], [0, 2, 4])] = q
    process[np.ix_([1, 3, 5], [1, 3, 5])] = q
    return transition, process
