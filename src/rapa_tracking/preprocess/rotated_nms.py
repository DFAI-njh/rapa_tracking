"""Tracker-input NMS independent of detector implementation and identity."""

from typing import Callable, Tuple

import numpy as np


def class_agnostic_rotated_3d_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        iou_threshold: float,
        iou_function: Callable[[np.ndarray, np.ndarray], Tuple[float, float]],
):
    """Keep highest-score boxes using class-agnostic rotated 3D IoU.

    This function is intentionally tracker-side. It consumes only detector
    measurements (box, score, label), never a detector or GT identity. Labels
    are returned unchanged for downstream class smoothing but do not partition
    NMS, so cross-class duplicates are removed.
    """
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if len(boxes) < 2:
        return boxes, scores, labels, 0

    order = np.argsort(-scores, kind='stable')
    kept = []
    for candidate in order:
        if all(iou_function(boxes[candidate], boxes[index])[0] <= iou_threshold
               for index in kept):
            kept.append(int(candidate))
    indices = np.asarray(kept, dtype=np.int64)
    suppressed = int(len(boxes) - len(indices))
    return boxes[indices], scores[indices], labels[indices], suppressed
