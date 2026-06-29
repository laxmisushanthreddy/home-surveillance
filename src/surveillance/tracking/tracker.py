"""
MultiObjectTracker — BoostTrack++ adapter.

This is the only file in the codebase that knows BoostTrack++ exists.
All downstream phases call this interface exclusively.

Usage:
    from surveillance.tracking.tracker import MultiObjectTracker
    from surveillance.core.config import load_config

    cfg = load_config("tracking")
    tracker = MultiObjectTracker(cfg)

    for frame_idx, detections in enumerate(detection_stream):
        tracked_persons = tracker.update(detections, frame_idx)
        confirmed = [p for p in tracked_persons if p.is_confirmed]
        # Pass confirmed to Phase 3
"""

from surveillance.core.config import load_config
from surveillance.core.logger import get_logger
from surveillance.detection.schema import Detection
from surveillance.tracking.boosttrack.boost_track import BoostTracker
from surveillance.tracking.boosttrack.track import TrackState as _InternalState
from surveillance.tracking.schema import TrackedPerson, TrackState
from surveillance.utils.timer import StageTimer

logger = get_logger(__name__)

# Map internal TrackState integers to our public TrackState enum
_STATE_MAP = {
    _InternalState.Tentative: TrackState.Tentative,
    _InternalState.Confirmed: TrackState.Confirmed,
    _InternalState.Lost:      TrackState.Lost,
    _InternalState.Deleted:   TrackState.Deleted,
}


class MultiObjectTracker:
    """
    Multi-object tracker wrapping BoostTrack++.

    Adapter pattern: downstream phases never import from boosttrack/.
    If BoostTrack++ is replaced, only this file changes.

    Lifecycle:
        tracker = MultiObjectTracker(cfg)
        for frame_idx, detections in stream:
            tracked = tracker.update(detections, frame_idx)
    """

    def __init__(self, cfg=None) -> None:
        """
        Args:
            cfg: OmegaConf DictConfig from load_config("tracking").
                 Auto-loaded if None.
        """
        self._cfg = cfg if cfg is not None else load_config("tracking")
        tc = self._cfg.model.tracker

        self._tracker = BoostTracker(
            max_age=tc.max_age,
            min_hits=tc.min_hits,
            iou_threshold=tc.iou_threshold,
            conf_threshold_high=tc.conf_threshold_high,
            conf_threshold_low=tc.conf_threshold_low,
            boost_alpha=tc.boost_alpha,
        )

        self._min_track_age = self._cfg.pipeline.min_track_age
        self._frame_count = 0

        logger.info(
            "MultiObjectTracker initialized | max_age=%d | min_hits=%d",
            tc.max_age, tc.min_hits,
        )

    def update(
        self,
        detections: list[Detection],
        frame_idx: int = 0,
    ) -> list[TrackedPerson]:
        """
        Update tracker with new detections and return tracked persons.

        Args:
            detections: List[Detection] from PersonDetector.detect().
                        Can be empty (no persons this frame).
            frame_idx:  Current frame index for bookkeeping.

        Returns:
            List[TrackedPerson] — all active tracks.
            Filter by .is_confirmed for downstream face detection.
        """
        self._frame_count += 1

        with StageTimer("BoostTrack++") as timer:
            raw_tracks = self._tracker.update(detections)

        if self._frame_count % 100 == 0:
            logger.debug(
                "Frame %d | Tracking: %.1f ms | Active: %d confirmed",
                frame_idx,
                timer.elapsed_ms,
                self._tracker.active_track_count,
            )

        tracked_persons = []
        for track in raw_tracks:
            xyxy = track.to_xyxy()
            public_state = _STATE_MAP.get(track.state, TrackState.Lost)

            person = TrackedPerson(
                track_id=track.track_id,
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
                confidence=float(track.confidence),
                state=public_state,
                age=track.age,
                time_since_update=track.time_since_update,
                frame_idx=frame_idx,
            )

            if person.is_valid():
                tracked_persons.append(person)

        logger.debug(
            "Frame %d | %d tracks returned (%d confirmed)",
            frame_idx,
            len(tracked_persons),
            sum(1 for p in tracked_persons if p.is_confirmed),
        )

        return tracked_persons

    def reset(self) -> None:
        """Reset tracker. Call between video sequences."""
        self._tracker.reset()
        self._frame_count = 0
        logger.info("Tracker reset.")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def active_confirmed_count(self) -> int:
        return self._tracker.active_track_count
