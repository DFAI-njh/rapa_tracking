"""Exact rotated box geometry used by RoadTrack V3 association."""

import math
from typing import Iterable, List, Tuple

import numpy as np

def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _bev_corners(box: np.ndarray) -> np.ndarray:
    half_l, half_w = max(float(box[3]), 1e-3) / 2, max(float(box[4]), 1e-3) / 2
    local = np.asarray([
        [-half_l, -half_w], [half_l, -half_w],
        [half_l, half_w], [-half_l, half_w],
    ], dtype=np.float64)
    c, s = math.cos(float(box[6])), math.sin(float(box[6]))
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return local @ rotation.T + np.asarray(box[:2], dtype=np.float64)


def _line_intersection(start: np.ndarray, end: np.ndarray,
                       clip_start: np.ndarray, clip_end: np.ndarray) -> np.ndarray:
    edge = clip_end - clip_start
    direction = end - start
    denominator = _cross2d(edge, direction)
    if abs(denominator) < 1e-12:
        return end.copy()
    amount = -_cross2d(edge, start - clip_start) / denominator
    return start + amount * direction


def _clip_convex_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = [point.copy() for point in subject]
    for index in range(len(clip)):
        clip_start = clip[index]
        clip_end = clip[(index + 1) % len(clip)]
        input_points, output = output, []
        if not input_points:
            break
        start = input_points[-1]
        for end in input_points:
            end_inside = _cross2d(clip_end - clip_start, end - clip_start) >= -1e-9
            start_inside = _cross2d(clip_end - clip_start, start - clip_start) >= -1e-9
            if end_inside:
                if not start_inside:
                    output.append(_line_intersection(start, end, clip_start, clip_end))
                output.append(end.copy())
            elif start_inside:
                output.append(_line_intersection(start, end, clip_start, clip_end))
            start = end
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return abs(float(np.dot(points[:, 0], np.roll(points[:, 1], -1))
                     - np.dot(points[:, 1], np.roll(points[:, 0], -1)))) / 2


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted(set((float(point[0]), float(point[1])) for point in points))
    if len(unique) <= 1:
        return np.asarray(unique, dtype=np.float64).reshape(-1, 2)

    def build(items: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
        hull: List[Tuple[float, float]] = []
        for item in items:
            while len(hull) >= 2:
                a = np.asarray(hull[-1]) - np.asarray(hull[-2])
                b = np.asarray(item) - np.asarray(hull[-1])
                if _cross2d(a, b) > 1e-12:
                    break
                hull.pop()
            hull.append(item)
        return hull

    lower = build(unique)
    upper = build(reversed(unique))
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _rotated_iou_3d_stats(first: np.ndarray, second: np.ndarray):
    """Return exact rotated intersection/union data shared by all metrics."""
    first_corners, second_corners = _bev_corners(first), _bev_corners(second)
    intersection_area = _polygon_area(_clip_convex_polygon(first_corners, second_corners))
    first_bottom = float(first[2]) - max(float(first[5]), 1e-3) / 2
    first_top = float(first[2]) + max(float(first[5]), 1e-3) / 2
    second_bottom = float(second[2]) - max(float(second[5]), 1e-3) / 2
    second_top = float(second[2]) + max(float(second[5]), 1e-3) / 2
    intersection_height = max(0.0, min(first_top, second_top)
                              - max(first_bottom, second_bottom))
    intersection = intersection_area * intersection_height
    first_volume = max(float(np.prod(np.maximum(first[3:6], 1e-3))), 1e-9)
    second_volume = max(float(np.prod(np.maximum(second[3:6], 1e-3))), 1e-9)
    union = max(first_volume + second_volume - intersection, 1e-9)
    iou = intersection / union
    return (float(iou), float(union), first_corners, second_corners,
            first_bottom, first_top, second_bottom, second_top)


def box_iou_giou_3d(first: np.ndarray, second: np.ndarray) -> Tuple[float, float]:
    """Compute rotated 3D IoU and convex-enclosure GIoU.

    This is the compatibility metric used by SimpleTrack
    (https://arxiv.org/abs/2111.09621). The implementation is clean-room and
    uses exact convex polygon clipping in BEV followed by vertical overlap.
    """
    (iou, union, first_corners, second_corners, first_bottom, first_top,
     second_bottom, second_top) = _rotated_iou_3d_stats(first, second)
    enclosure_area = _polygon_area(_convex_hull(
        np.concatenate((first_corners, second_corners), axis=0)))
    enclosure_height = max(first_top, second_top) - min(first_bottom, second_bottom)
    enclosure = max(enclosure_area * enclosure_height, union, 1e-9)
    giou = iou - (enclosure - union) / enclosure
    return float(iou), float(giou)


def box_iou_giou_bev(first, second):
    first_corners, second_corners = _bev_corners(first), _bev_corners(second)
    intersection = _polygon_area(_clip_convex_polygon(first_corners, second_corners))
    union = max(float(first[3] * first[4] + second[3] * second[4] - intersection), 1e-9)
    iou = intersection / union
    enclosure = max(_polygon_area(_convex_hull(np.concatenate(
        (first_corners, second_corners), axis=0))), union, 1e-9)
    return float(iou), float(iou - (enclosure - union) / enclosure)


def box_iou_diou_3d(first: np.ndarray, second: np.ndarray) -> Tuple[float, float]:
    """Compute rotated 3D IoU and 3D-DIoU similarity.

    ``DIoU3D = IoU3D - ||center1-center2||^2 / c^2``, where ``c`` is the
    diagonal of the axis-aligned enclosing cuboid over the vertices of both
    *oriented* boxes. The formula follows *3D Distance Intersection over Union
    for Multi-Object Tracking in Point Cloud* (Sensors 2023):
    https://www.mdpi.com/1424-8220/23/7/3390
    """
    (iou, _, first_corners, second_corners, first_bottom, first_top,
     second_bottom, second_top) = _rotated_iou_3d_stats(first, second)
    all_corners = np.concatenate((first_corners, second_corners), axis=0)
    xy_extent = np.max(all_corners, axis=0) - np.min(all_corners, axis=0)
    z_extent = max(first_top, second_top) - min(first_bottom, second_bottom)
    enclosing_diagonal_squared = max(
        float(np.dot(xy_extent, xy_extent) + z_extent ** 2), 1e-9)
    center_delta = np.asarray(first[:3], dtype=np.float64) - second[:3]
    distance_penalty = float(np.dot(center_delta, center_delta)) / enclosing_diagonal_squared
    return float(iou), float(iou - distance_penalty)


def odiou3d_cost(first: np.ndarray, second: np.ndarray,
                 yaw_weight: float = 0.20) -> float:
    """Return clean-room orientation-aware 3D-DIoU association loss.

    ``1 - IoU3D + d^2/c^2 + gamma*(1-|cos(delta_yaw)|)`` follows the ODIoU
    structure introduced for SE-SSD (CVPR 2021), adapted here as an online
    rotated-3D association cost. Absolute cosine makes yaw 0 and pi equivalent:
    https://openaccess.thecvf.com/content/CVPR2021/papers/
    Zheng_SE-SSD_Self-Ensembling_Single-Stage_Object_Detector_From_Point_Cloud_CVPR_2021_paper.pdf
    """
    iou, diou = box_iou_diou_3d(first, second)
    distance_penalty = iou - diou
    yaw_penalty = 1.0 - abs(math.cos(float(first[6] - second[6])))
    return float(1.0 - iou + distance_penalty + yaw_weight * yaw_penalty)


def ro_gdiou_similarity(first, second, enclosure_weight=1., center_weight=1.):
    iou, giou = box_iou_giou_3d(first, second)
    _, diou = box_iou_diou_3d(first, second)
    return float(iou - enclosure_weight * (iou - giou)
                 - center_weight * (iou - diou))

