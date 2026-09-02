"""Exclusive oriented 3D point-to-detection assignment."""

import numpy as np


def xyz(points):
    if points.dtype.names:
        return np.column_stack([points[name] for name in ('x', 'y', 'z')])
    return np.asarray(points[:, :3])


def assign_points_to_boxes(points, boxes, margin=(0., 0., 0.)):
    points, boxes = np.asarray(points), np.asarray(boxes).reshape(-1, 7)
    coordinates = xyz(points)
    costs = np.full((len(points), len(boxes)), np.inf)
    margin = np.asarray(margin)
    for index, box in enumerate(boxes):
        delta = coordinates - box[:3]
        c, s = np.cos(box[6]), np.sin(box[6])
        local = np.column_stack((c * delta[:, 0] + s * delta[:, 1],
                                 -s * delta[:, 0] + c * delta[:, 1], delta[:, 2]))
        half = np.maximum(box[3:6] / 2 + margin, 1e-3)
        inside = np.all(np.abs(local) <= half, axis=1)
        costs[inside, index] = np.linalg.norm(local[inside] / half, axis=1)
    if not len(boxes):
        return []
    winner = np.argmin(costs, axis=1)
    valid = np.isfinite(np.min(costs, axis=1))
    return [np.flatnonzero(valid & (winner == index)) for index in range(len(boxes))]
