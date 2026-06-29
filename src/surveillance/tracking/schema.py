"""
Data schemas for the tracking stage.

TrackedPerson is the data contract between Phase 2 (tracking)
and Phase 3 (face detection). Face detection only runs on
tracks with state == TrackState.CONFIRMED.
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class TrackState(Enum):
    """
    Lifecycle state of a track.

    Tentative: Newly created, not yet confirmed by sufficient hits.
               Do NOT run face detection on tentative tracks.

    Confirmed: Seen for min_hits consecutive frames.
               Safe to run face detection and recognition.

    Lost:      Detection missed for 1+ frames.
               Kalman prediction shown, no face detection.

    Deleted:   Missed for max_age frames. Removed from memory.
    """
    Tentative = auto()
    Confirmed = auto()
    Lost      = auto()
    Deleted   = auto()


@dataclass
class TrackedPerson:
    """
    A single tracked person with persistent identity.

    This is the output contract of Phase 2.
    Phase 3 consumes List[TrackedPerson] filtered to Confirmed state.

    Attributes:
        track_id:          Persistent integer ID, unique per session.
        x1, y1, x2, y2:   Kalman-filtered bounding box (xyxy, pixel coords).
        confidence:        Detection confidence of the matched detection.
                           0.0 if track is in Lost state (no matched detection).
        state:             Current lifecycle state.
        age:               Total frames since this track was first created.
        time_since_update: Frames since last matched detection.
                           0 = matched this frame. 1+ = Lost.
        frame_idx:         Frame index when this track was last updated.
    """
    track_id:          int
    x1:                float
    y1:                float
    x2:                float
    y2:                float
    confidence:        float
    state:             TrackState
    age:               int
    time_since_update: int
    frame_idx:         int = 0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def is_confirmed(self) -> bool:
        """True if this track is safe for face detection."""
        return self.state == TrackState.Confirmed

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.Lost

    @property
    def is_new(self) -> bool:
        """True if this track was just confirmed this frame."""
        return self.state == TrackState.Confirmed and self.age == self.time_since_update

    def is_valid(self) -> bool:
        """Check positive area and valid confidence."""
        return (
            self.x2 > self.x1
            and self.y2 > self.y1
            and 0.0 <= self.confidence <= 1.0
        )
