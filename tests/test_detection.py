"""Phase 1 detection tests."""

import numpy as np
import pytest

from surveillance.detection.schema import Detection


class TestDetectionSchema:

    def test_xyxy_property(self):
        d = Detection(x1=10, y1=20, x2=100, y2=200, confidence=0.9)
        assert d.xyxy == (10, 20, 100, 200)

    def test_xywh_property(self):
        d = Detection(x1=10, y1=20, x2=110, y2=120, confidence=0.9)
        assert d.xywh == (10, 20, 100, 100)

    def test_width_height(self):
        d = Detection(x1=0, y1=0, x2=50, y2=80, confidence=0.8)
        assert d.width == 50
        assert d.height == 80

    def test_area(self):
        d = Detection(x1=0, y1=0, x2=10, y2=20, confidence=0.8)
        assert d.area == 200

    def test_center(self):
        d = Detection(x1=0, y1=0, x2=100, y2=80, confidence=0.8)
        assert d.center == (50.0, 40.0)

    def test_is_valid_true(self):
        assert Detection(x1=10, y1=10, x2=100, y2=100, confidence=0.75).is_valid()

    def test_is_valid_zero_area(self):
        assert not Detection(x1=50, y1=50, x2=50, y2=50, confidence=0.9).is_valid()

    def test_is_valid_inverted_box(self):
        assert not Detection(x1=100, y1=100, x2=10, y2=10, confidence=0.9).is_valid()

    def test_is_valid_bad_confidence(self):
        assert not Detection(x1=0, y1=0, x2=100, y2=100, confidence=1.5).is_valid()

    def test_default_class_id(self):
        assert Detection(x1=0, y1=0, x2=100, y2=100, confidence=0.8).class_id == 0


class TestPersonDetector:

    @pytest.fixture(scope="class")
    def detector(self):
        from surveillance.core.config import load_config, override_config
        from surveillance.detection.detector import PersonDetector
        cfg = load_config("detection")
        cfg = override_config(cfg, {
            "model.yolo.device": "cpu",
            "inference.half_precision": False,
        })
        return PersonDetector(cfg)

    def test_detector_initializes(self, detector):
        assert detector is not None
        assert detector.device == "cpu"

    def test_detect_returns_list(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert isinstance(detector.detect(frame), list)

    def test_detect_none_returns_empty(self, detector):
        assert detector.detect(None) == []

    def test_all_detections_are_valid(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for d in detector.detect(frame):
            assert isinstance(d, Detection)
            assert d.is_valid()

    def test_detections_sorted_by_confidence(self, detector):
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        if len(result) > 1:
            confs = [d.confidence for d in result]
            assert confs == sorted(confs, reverse=True)

    def test_frame_count_increments(self, detector):
        initial = detector.frame_count
        detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert detector.frame_count == initial + 1

    def test_detect_batch_length(self, detector):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        results = detector.detect_batch(frames)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, list)
