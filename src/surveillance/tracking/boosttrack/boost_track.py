"""
BoostTrack++ core tracker orchestration.

Implements the full tracking loop:
  1. Predict all existing tracks forward (Kalman)
  2. Build cost matrix (IoU + confidence boost)
  3. First-pass association: confirmed tracks vs high-conf detections
  4. Second-pass association: remaining tracks vs low-conf detections
  5. Update matched tracks, mark missed, create new tentative tracks
  6. Delete expired tracks

Reference: Stanojevic et al., "BoostTrack++", 2024
"""

import numpy as np

from surveillance.tracking.boosttrack.track import Track, TrackState
from surveillance.tracking.boosttrack.matching import (
    boost_confidence_cost,
    iou_distance,
    linear_assignment,
)


class BoostTracker:
    """
    BoostTrack++ multi-object tracker.
    Designed for single-class tracking (persons).
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        conf_threshold_high: float = 0.5,
        conf_threshold_low: float = 0.1,
        boost_alpha: float = 0.4,
    ) -> None:
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._conf_high = conf_threshold_high
        self._conf_low = conf_threshold_low
        self._boost_alpha = boost_alpha

        self._tracks: list[Track] = []
        self.frame_count: int = 0

    def update(self, detections: list) -> list[Track]:
        """
        Run one tracking step for a new frame.

        Args:
            detections: List of Detection objects from Phase 1.

        Returns:
            List of active Track objects (Tentative + Confirmed + Lost).
        """
        self.frame_count += 1

        # ── Step 1: Predict all tracks forward ────────────────────────────
        for track in self._tracks:
            track.predict()

        # ── Step 2: Split detections by confidence ─────────────────────────
        high_dets = [d for d in detections if d.confidence >= self._conf_high]
        low_dets  = [d for d in detections
                     if self._conf_low <= d.confidence < self._conf_high]

        # All active (non-deleted) tracks for association
        active_tracks = [t for t in self._tracks
                         if t.state != TrackState.Deleted]

        # Track which tracks and detections have been matched
        matched_track_ids: set[int] = set()
        matched_high_det_indices: set[int] = set()

        # ── Step 3: First pass — all active tracks vs high-conf detections ─
        if active_tracks and high_dets:
            matches_a, _, unmatched_dets_a = self._associate(
                active_tracks, high_dets
            )
            for t_idx, d_idx in matches_a:
                det = high_dets[d_idx]
                active_tracks[t_idx].update(
                    self._det_to_measurement(det), det.confidence
                )
                matched_track_ids.add(active_tracks[t_idx].track_id)
                matched_high_det_indices.add(d_idx)
        else:
            unmatched_dets_a = list(range(len(high_dets)))

        # ── Step 4: Second pass — unmatched tracks vs low-conf detections ──
        unmatched_tracks = [t for t in active_tracks
                            if t.track_id not in matched_track_ids]

        if unmatched_tracks and low_dets:
            matches_b, _, _ = self._associate(unmatched_tracks, low_dets)
            for t_idx, d_idx in matches_b:
                det = low_dets[d_idx]
                unmatched_tracks[t_idx].update(
                    self._det_to_measurement(det), det.confidence
                )
                matched_track_ids.add(unmatched_tracks[t_idx].track_id)

        # ── Step 5: Mark truly unmatched tracks as missed ─────────────────
        for track in active_tracks:
            if track.track_id not in matched_track_ids:
                track.mark_missed()

        # ── Step 6: Create new tentative tracks for unmatched high-conf dets
        for d_idx in unmatched_dets_a:
            det = high_dets[d_idx]
            new_track = Track(
                measurement=self._det_to_measurement(det),
                confidence=det.confidence,
                min_hits=self._min_hits,
                max_age=self._max_age,
            )
            self._tracks.append(new_track)

        # ── Step 7: Remove deleted tracks ─────────────────────────────────
        self._tracks = [t for t in self._tracks if not t.is_deleted()]

        # Return all non-deleted tracks
        return list(self._tracks)

    def _associate(
        self,
        tracks: list[Track],
        detections: list,
    ) -> tuple[list, list, list]:
        """Associate tracks with detections using boosted IoU cost."""
        cost_matrix = iou_distance(tracks, detections)
        cost_matrix = boost_confidence_cost(
            cost_matrix, detections, alpha=self._boost_alpha
        )
        return linear_assignment(cost_matrix, threshold=self._iou_threshold)

    @staticmethod
    def _det_to_measurement(det) -> np.ndarray:
        """Convert Detection (xyxy) to Kalman measurement (cx, cy, ar, h)."""
        x1, y1, x2, y2 = det.xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h  = y2 - y1
        ar = (x2 - x1) / h if h > 0 else 1.0
        return np.array([cx, cy, ar, h])

    def reset(self) -> None:
        """Reset tracker state. Call between video sequences."""
        self._tracks = []
        self.frame_count = 0
        Track.reset_id_counter()

    @property
    def active_track_count(self) -> int:
        return len([t for t in self._tracks if t.is_confirmed()])
