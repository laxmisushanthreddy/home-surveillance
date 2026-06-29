"""Tests for bounding box utilities."""

import numpy as np
import pytest

from surveillance.utils.bbox import (
    clip_boxes, compute_iou, crop_box,
    xywh_to_xyxy, xyxy_to_xywh,
)


def test_xyxy_to_xywh():
    boxes = np.array([[10, 20, 110, 120]], dtype=np.float32)
    np.testing.assert_array_equal(xyxy_to_xywh(boxes), [[10, 20, 100, 100]])


def test_xywh_to_xyxy():
    boxes = np.array([[10, 20, 100, 100]], dtype=np.float32)
    np.testing.assert_array_equal(xywh_to_xyxy(boxes), [[10, 20, 110, 120]])


def test_roundtrip(dummy_boxes):
    np.testing.assert_allclose(xywh_to_xyxy(xyxy_to_xywh(dummy_boxes)), dummy_boxes)


def test_clip_boxes():
    boxes = np.array([[-10, -5, 700, 500]], dtype=np.float32)
    c = clip_boxes(boxes, img_shape=(480, 640))
    assert c[0, 0] == 0.0 and c[0, 1] == 0.0
    assert c[0, 2] == 640.0 and c[0, 3] == 480.0


def test_iou_perfect():
    b = np.array([0, 0, 100, 100], dtype=np.float32)
    assert compute_iou(b, b) == pytest.approx(1.0)


def test_iou_no_overlap():
    b1 = np.array([0, 0, 50, 50], dtype=np.float32)
    b2 = np.array([100, 100, 200, 200], dtype=np.float32)
    assert compute_iou(b1, b2) == pytest.approx(0.0)


def test_crop_valid(dummy_bgr_image):
    box = np.array([100, 50, 300, 200], dtype=np.float32)
    crop = crop_box(dummy_bgr_image, box)
    assert crop.shape == (150, 200, 3)


def test_crop_with_padding(dummy_bgr_image):
    box = np.array([200, 100, 400, 300], dtype=np.float32)
    crop = crop_box(dummy_bgr_image, box, padding=0.1)
    assert crop.shape[0] > 200 and crop.shape[1] > 200
