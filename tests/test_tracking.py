"""Phase 2 tracking tests."""

import numpy as np
import pytest

from surveillance.tracking.schema import TrackedPerson, TrackState
from surveillance.detection.schema import Detection


# ── Schema tests ───────────────────────────────────────────────────────────────

class TestTrackedPersonSchema:

    def _make(self, **kwargs):
        defaults = dict(track_id=1, x1=10, y1=20, x2=100, y2=200,
                        confidence=0.8, state=TrackState.Confirmed,
                        age=5, time_since_update=0)
        defaults.update(kwargs)
        return TrackedPerson(**defaults)

    def test_xyxy(self):
        p = self._make(x1=10, y1=20, x2=100, y2=200)
        assert p.xyxy == (10, 20, 100, 200)

    def test_width_height(self):
        p = self._make(x1=0, y1=0, x2=80, y2=120)
        assert p.width == 80
        assert p.height == 120

    def test_center(self):
        p = self._make(x1=0, y1=0, x2=100, y2=80)
        assert p.center == (50.0, 40.0)

    def test_is_confirmed_true(self):
        p = self._make(state=TrackState.Confirmed)
        assert p.is_confirmed

    def test_is_confirmed_false_for_lost(self):
        p = self._make(state=TrackState.Lost)
        assert not p.is_confirmed

    def test_is_lost(self):
        p = self._make(state=TrackState.Lost)
        assert p.is_lost

    def test_is_valid_true(self):
        assert self._make().is_valid()

    def test_is_valid_zero_area(self):
        p = self._make(x1=50, y1=50, x2=50, y2=50)
        assert not p.is_valid()


# ── Kalman filter tests ────────────────────────────────────────────────────────

class TestKalmanFilter:

    def test_initiate_returns_correct_shapes(self):
        from surveillance.tracking.boosttrack.kalman_filter import KalmanFilter
        kf = KalmanFilter()
        meas = np.array([100.0, 200.0, 0.5, 150.0])
        mean, cov = kf.initiate(meas)
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)

    def test_predict_returns_correct_shapes(self):
        from surveillance.tracking.boosttrack.kalman_filter import KalmanFilter
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 150.0]))
        mean_p, cov_p = kf.predict(mean, cov)
        assert mean_p.shape == (8,)
        assert cov_p.shape == (8, 8)

    def test_update_returns_correct_shapes(self):
        from surveillance.tracking.boosttrack.kalman_filter import KalmanFilter
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 150.0]))
        mean_p, cov_p = kf.predict(mean, cov)
        mean_u, cov_u = kf.update(mean_p, cov_p, np.array([102.0, 201.0, 0.5, 150.0]))
        assert mean_u.shape == (8,)

    def test_kalman_reduces_uncertainty_after_update(self):
        from surveillance.tracking.boosttrack.kalman_filter import KalmanFilter
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 150.0]))
        mean_p, cov_p = kf.predict(mean, cov)
        _, cov_u = kf.update(mean_p, cov_p, np.array([100.0, 200.0, 0.5, 150.0]))
        # Trace of covariance should decrease after update
        assert np.trace(cov_u) < np.trace(cov_p)


# ── Matching tests ─────────────────────────────────────────────────────────────

class TestMatching:

    def test_iou_batch_perfect_overlap(self):
        from surveillance.tracking.boosttrack.matching import iou_batch
        box = np.array([[0, 0, 100, 100]], dtype=float)
        iou = iou_batch(box, box)
        assert iou[0, 0] == pytest.approx(1.0)

    def test_iou_batch_no_overlap(self):
        from surveillance.tracking.boosttrack.matching import iou_batch
        a = np.array([[0, 0, 50, 50]], dtype=float)
        b = np.array([[100, 100, 200, 200]], dtype=float)
        assert iou_batch(a, b)[0, 0] == pytest.approx(0.0)

    def test_linear_assignment_basic(self):
        from surveillance.tracking.boosttrack.matching import linear_assignment
        cost = np.array([[0.1, 0.9], [0.8, 0.2]])
        matches, unmatched_t, unmatched_d = linear_assignment(cost, threshold=0.5)
        assert (0, 0) in matches
        assert (1, 1) in matches
        assert unmatched_t == []
        assert unmatched_d == []

    def test_linear_assignment_above_threshold_rejected(self):
        from surveillance.tracking.boosttrack.matching import linear_assignment
        cost = np.array([[0.9, 0.95]])  # Both above threshold 0.5
        matches, unmatched_t, unmatched_d = linear_assignment(cost, threshold=0.5)
        assert matches == []
        assert 0 in unmatched_t


# ── Tracker integration tests ──────────────────────────────────────────────────

class TestMultiObjectTracker:

    @pytest.fixture
    def tracker(self):
        from surveillance.core.config import load_config
        from surveillance.tracking.tracker import MultiObjectTracker
        cfg = load_config("tracking")
        return MultiObjectTracker(cfg)

    def _make_detection(self, x1, y1, x2, y2, conf=0.9):
        return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)

    def test_empty_detections_returns_empty(self, tracker):
        result = tracker.update([], frame_idx=0)
        assert result == []

    def test_single_detection_creates_tentative_track(self, tracker):
        det = self._make_detection(100, 100, 200, 300)
        result = tracker.update([det], frame_idx=0)
        assert len(result) == 1
        assert result[0].state == TrackState.Tentative

    def test_track_confirmed_after_min_hits(self, tracker):
        det = self._make_detection(100, 100, 200, 300)
        # Feed same detection for min_hits=3 consecutive frames
        result = None
        for i in range(3):
            result = tracker.update([det], frame_idx=i)
        assert any(p.is_confirmed for p in result)

    def test_track_id_is_persistent(self, tracker):
        det = self._make_detection(100, 100, 200, 300)
        ids = []
        for i in range(5):
            result = tracker.update([det], frame_idx=i)
            if result:
                ids.append(result[0].track_id)
        # All frames should have the same track_id
        assert len(set(ids)) == 1

    def test_two_detections_get_different_ids(self, tracker):
        det1 = self._make_detection(10, 10, 80, 200)
        det2 = self._make_detection(400, 10, 480, 200)
        result = tracker.update([det1, det2], frame_idx=0)
        assert len(result) == 2
        assert result[0].track_id != result[1].track_id

    def test_frame_count_increments(self, tracker):
        tracker.update([], frame_idx=0)
        tracker.update([], frame_idx=1)
        assert tracker.frame_count == 2

    def test_reset_clears_tracks(self, tracker):
        det = self._make_detection(100, 100, 200, 300)
        for i in range(3):
            tracker.update([det], frame_idx=i)
        tracker.reset()
        result = tracker.update([], frame_idx=0)
        assert result == []

    def test_all_returned_persons_are_valid(self, tracker):
        det = self._make_detection(100, 100, 200, 300)
        for i in range(3):
            result = tracker.update([det], frame_idx=i)
        for p in result:
            assert p.is_valid()
