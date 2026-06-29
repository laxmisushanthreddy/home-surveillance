"""
Single track state machine.

Each Track represents one person being tracked.
Manages Kalman filter state and lifecycle transitions.
"""

import numpy as np
from surveillance.tracking.boosttrack.kalman_filter import KalmanFilter


class TrackState:
    Tentative = 1
    Confirmed = 2
    Lost      = 3
    Deleted   = 4


class Track:
    """
    Single tracked person.

    Attributes:
        track_id:          Unique persistent integer ID.
        state:             Current TrackState.
        hits:              Number of frames with matched detection.
        age:               Total frames since creation.
        time_since_update: Frames since last matched detection.
        confidence:        Confidence of last matched detection.
    """

    _id_counter: int = 0

    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset ID counter. Call between video sequences."""
        cls._id_counter = 0

    def __init__(
        self,
        measurement: np.ndarray,
        confidence: float,
        min_hits: int = 3,
        max_age: int = 30,
    ) -> None:
        """
        Initialize a new track from a first detection.

        Args:
            measurement: (cx, cy, aspect_ratio, height) array.
            confidence:  Detection confidence [0, 1].
            min_hits:    Frames needed to confirm a track.
            max_age:     Max frames without match before deletion.
        """
        self.track_id = Track._next_id()
        self._kf = KalmanFilter()
        self._mean, self._covariance = self._kf.initiate(measurement)

        self.state = TrackState.Tentative
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confidence = confidence
        self._min_hits = min_hits
        self._max_age = max_age

    def predict(self) -> None:
        """Advance Kalman filter by one time step."""
        self._mean, self._covariance = self._kf.predict(
            self._mean, self._covariance
        )
        self.age += 1
        self.time_since_update += 1

    def update(self, measurement: np.ndarray, confidence: float) -> None:
        """
        Update track with a matched detection.

        Args:
            measurement: (cx, cy, aspect_ratio, height).
            confidence:  Matched detection confidence.
        """
        self._mean, self._covariance = self._kf.update(
            self._mean, self._covariance, measurement
        )
        self.hits += 1
        self.time_since_update = 0
        self.confidence = confidence

        if self.state == TrackState.Tentative and self.hits >= self._min_hits:
            self.state = TrackState.Confirmed

    def mark_missed(self) -> None:
        """
        Called when no detection was matched this frame.
        Transitions Tentative → Deleted, Confirmed → Lost.
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
        else:
            self.state = TrackState.Lost

    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def to_xyxy(self) -> np.ndarray:
        """
        Convert Kalman state to xyxy bounding box.

        State is stored as (cx, cy, aspect_ratio, height).
        Convert: w = aspect_ratio * height
        """
        cx, cy, ar, h = self._mean[:4]
        w = ar * h
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    def gating_distance(self, measurements: np.ndarray) -> np.ndarray:
        """Mahalanobis distance to a set of measurements."""
        return self._kf.gating_distance(
            self._mean, self._covariance, measurements
        )
