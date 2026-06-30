"""Phase 3 face detection tests."""

import numpy as np
import pytest

from surveillance.face.schema import FaceDetectionResult
from surveillance.face.alignment import (
    align_face, landmarks_to_frame_coords, ARCFACE_REFERENCE
)
from surveillance.face.quality import laplacian_variance, FaceQualityFilter
from surveillance.tracking.schema import TrackedPerson, TrackState


def _make_tracked_person(**kwargs):
    defaults = dict(track_id=1, x1=50, y1=50, x2=250, y2=450,
                    confidence=0.9, state=TrackState.Confirmed,
                    age=5, time_since_update=0)
    defaults.update(kwargs)
    return TrackedPerson(**defaults)


def _make_face_result(**kwargs):
    defaults = dict(
        track_id=1,
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        bbox_in_frame=(100.0, 80.0, 200.0, 180.0),
        confidence=0.92,
        quality_score=120.0,
        frame_idx=10,
    )
    defaults.update(kwargs)
    return FaceDetectionResult(**defaults)


# ── Schema tests ───────────────────────────────────────────────────────────────

class TestFaceDetectionResultSchema:

    def test_is_valid_true(self):
        r = _make_face_result()
        assert r.is_valid()

    def test_is_valid_wrong_shape(self):
        r = _make_face_result(aligned_face=np.zeros((64, 64, 3), dtype=np.uint8))
        assert not r.is_valid()

    def test_is_valid_wrong_dtype(self):
        r = _make_face_result(aligned_face=np.zeros((112, 112, 3), dtype=np.float32))
        assert not r.is_valid()

    def test_is_acceptable_quality_true(self):
        assert _make_face_result(quality_score=100.0).is_acceptable_quality

    def test_is_acceptable_quality_false(self):
        assert not _make_face_result(quality_score=30.0).is_acceptable_quality

    def test_face_width_height(self):
        r = _make_face_result(bbox_in_frame=(100.0, 80.0, 200.0, 180.0))
        assert r.face_width == 100.0
        assert r.face_height == 100.0

    def test_is_large_enough_true(self):
        r = _make_face_result(bbox_in_frame=(0.0, 0.0, 50.0, 50.0))
        assert r.is_large_enough

    def test_is_large_enough_false(self):
        r = _make_face_result(bbox_in_frame=(0.0, 0.0, 10.0, 10.0))
        assert not r.is_large_enough


# ── Alignment tests ────────────────────────────────────────────────────────────

class TestAlignment:

    def test_align_face_returns_correct_shape(self):
        # Create a synthetic image with known landmarks near ARCFACE_REFERENCE
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Use landmarks close to where they would realistically appear
        landmarks = np.array([
            [200, 150], [280, 150], [240, 190],
            [210, 230], [270, 230],
        ], dtype=np.float32)
        result = align_face(image, landmarks)
        assert result is not None
        assert result.shape == (112, 112, 3)
        assert result.dtype == np.uint8

    def test_landmarks_to_frame_coords(self):
        lm_crop = np.array([[10, 20], [30, 40]], dtype=np.float32)
        lm_frame = landmarks_to_frame_coords(lm_crop, crop_x1=100.0, crop_y1=50.0)
        np.testing.assert_array_almost_equal(lm_frame, [[110, 70], [130, 90]])

    def test_landmarks_to_frame_coords_zero_offset(self):
        lm = np.array([[10, 20]], dtype=np.float32)
        result = landmarks_to_frame_coords(lm, 0.0, 0.0)
        np.testing.assert_array_equal(result, lm)

    def test_arcface_reference_shape(self):
        assert ARCFACE_REFERENCE.shape == (5, 2)
        assert ARCFACE_REFERENCE.dtype == np.float32


# ── Quality tests ──────────────────────────────────────────────────────────────

class TestQuality:

    def test_laplacian_variance_sharp(self):
        # Sharp image with high contrast edges
        sharp = np.zeros((112, 112, 3), dtype=np.uint8)
        sharp[::4, :] = 255  # Strong horizontal lines
        score = laplacian_variance(sharp)
        assert score > 100

    def test_laplacian_variance_blurry(self):
        # Uniform image has zero variance
        blank = np.full((112, 112, 3), 128, dtype=np.uint8)
        assert laplacian_variance(blank) == 0.0

    def test_quality_filter_rejects_small_face(self):
        filt = FaceQualityFilter(min_face_size=20)
        face = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        lm = np.zeros((5, 2), dtype=np.float32)
        # Face too small
        _, passes = filt.assess(face, lm, (0.0, 0.0, 10.0, 10.0))
        assert not passes

    def test_quality_filter_accepts_valid_face(self):
        filt = FaceQualityFilter(min_blur_score=10.0, min_face_size=20)
        # Sharp enough face
        face = np.zeros((80, 60, 3), dtype=np.uint8)
        face[::3, :] = 255
        lm = np.array([
            [20, 25], [40, 25], [30, 35], [22, 45], [38, 45]
        ], dtype=np.float32)
        score, passes = filt.assess(face, lm, (0.0, 0.0, 60.0, 80.0))
        assert passes
        assert score > 0


# ── FaceDetector integration test (requires insightface) ──────────────────────

class TestFaceDetector:

    @pytest.fixture(scope="class")
    def face_detector(self):
        from surveillance.core.config import load_config
        from surveillance.face.detector import FaceDetector
        cfg = load_config("face_detection")
        return FaceDetector(cfg)

    def test_detector_initializes(self, face_detector):
        assert face_detector is not None

    def test_detect_empty_tracks_returns_empty(self, face_detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = face_detector.detect(frame, [], frame_idx=0)
        assert result == []

    def test_detect_none_frame_returns_empty(self, face_detector):
        result = face_detector.detect(None, [], frame_idx=0)
        assert result == []

    def test_detect_tentative_track_skipped(self, face_detector):
        from surveillance.tracking.schema import TrackedPerson, TrackState
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Tentative track — should be skipped
        tentative = _make_tracked_person(state=TrackState.Tentative)
        result = face_detector.detect(frame, [tentative], frame_idx=0)
        assert result == []

    def test_all_results_are_valid(self, face_detector):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        person = _make_tracked_person()
        results = face_detector.detect(frame, [person], frame_idx=0)
        for r in results:
            assert isinstance(r, FaceDetectionResult)
