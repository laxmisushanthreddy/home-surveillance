"""
Detection-to-track association utilities.

Implements:
  - IoU distance matrix computation
  - BoostTrack++ soft confidence weighting
  - Hungarian algorithm (via scipy.optimize.linear_sum_assignment)
  - Cascaded matching strategy
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(
    bboxes_a: np.ndarray,
    bboxes_b: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of xyxy boxes.

    Args:
        bboxes_a: (M, 4) array of xyxy boxes.
        bboxes_b: (N, 4) array of xyxy boxes.

    Returns:
        (M, N) IoU matrix.
    """
    # Intersection
    tl = np.maximum(bboxes_a[:, None, :2], bboxes_b[None, :, :2])
    br = np.minimum(bboxes_a[:, None, 2:], bboxes_b[None, :, 2:])
    wh = np.maximum(br - tl, 0)
    inter = wh[..., 0] * wh[..., 1]

    # Areas
    area_a = ((bboxes_a[:, 2] - bboxes_a[:, 0]) *
               (bboxes_a[:, 3] - bboxes_a[:, 1]))
    area_b = ((bboxes_b[:, 2] - bboxes_b[:, 0]) *
               (bboxes_b[:, 3] - bboxes_b[:, 1]))
    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


def iou_distance(
    tracks: list,
    detections: list,
) -> np.ndarray:
    """
    Compute IoU-based cost matrix (1 - IoU) for track-detection pairs.

    Args:
        tracks:     List of Track objects with .to_xyxy() method.
        detections: List of Detection objects with .xyxy property.

    Returns:
        (len(tracks), len(detections)) cost matrix.
    """
    if not tracks or not detections:
        return np.empty((len(tracks), len(detections)))

    track_boxes = np.array([t.to_xyxy() for t in tracks])
    det_boxes   = np.array([d.xyxy for d in detections])
    iou         = iou_batch(track_boxes, det_boxes)
    return 1.0 - iou


def boost_confidence_cost(
    cost_matrix: np.ndarray,
    detections: list,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Apply BoostTrack++ soft detection confidence weighting to cost matrix.

    Core innovation: instead of binary accept/reject at IoU threshold,
    weight costs by detection confidence. High-confidence detections
    produce lower cost (preferred matches) even at moderate IoU.

    cost_ij_boosted = cost_ij * (1 - alpha * confidence_j)

    Args:
        cost_matrix: (M, N) raw IoU cost matrix.
        detections:  List of N Detection objects with .confidence.
        alpha:       Confidence weighting strength. 0 = no boost.

    Returns:
        (M, N) boosted cost matrix.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    conf = np.array([d.confidence for d in detections])
    weight = 1.0 - alpha * conf          # shape (N,)
    return cost_matrix * weight[None, :] # broadcast over tracks


def linear_assignment(
    cost_matrix: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Run Hungarian algorithm and split results into matches / unmatched.

    Args:
        cost_matrix: (M, N) cost matrix. Lower = better match.
        threshold:   Matches with cost > threshold are rejected.

    Returns:
        Tuple of:
          - matches:            List of (track_idx, det_idx) pairs.
          - unmatched_tracks:   Track indices with no valid match.
          - unmatched_dets:     Detection indices with no valid match.
    """
    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_dets = [], [], []
    matched_rows, matched_cols = set(), set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > threshold:
            continue
        matches.append((r, c))
        matched_rows.add(r)
        matched_cols.add(c)

    unmatched_tracks = [i for i in range(cost_matrix.shape[0])
                        if i not in matched_rows]
    unmatched_dets   = [j for j in range(cost_matrix.shape[1])
                        if j not in matched_cols]

    return matches, unmatched_tracks, unmatched_dets
