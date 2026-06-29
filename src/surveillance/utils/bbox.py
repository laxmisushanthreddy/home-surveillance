"""
Bounding box utility functions.

Formats used throughout this project:
  xyxy    — (x1, y1, x2, y2) absolute pixel, top-left + bottom-right
  xywh    — (x, y, w, h) top-left corner + width + height
  cxcywh  — (cx, cy, w, h) center + width + height (YOLO internal format)

All functions accept NumPy arrays of shape (N, 4) for batch processing.
"""

import numpy as np
from numpy.typing import NDArray


def xyxy_to_xywh(boxes: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert (x1, y1, x2, y2) → (x, y, w, h)."""
    out = boxes.copy()
    out[..., 2] = boxes[..., 2] - boxes[..., 0]
    out[..., 3] = boxes[..., 3] - boxes[..., 1]
    return out


def xywh_to_xyxy(boxes: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert (x, y, w, h) → (x1, y1, x2, y2)."""
    out = boxes.copy()
    out[..., 2] = boxes[..., 0] + boxes[..., 2]
    out[..., 3] = boxes[..., 1] + boxes[..., 3]
    return out


def clip_boxes(
    boxes: NDArray[np.float32],
    img_shape: tuple[int, int],
) -> NDArray[np.float32]:
    """
    Clip xyxy boxes to image boundaries.

    Args:
        boxes:     Shape (N, 4) in xyxy format.
        img_shape: (height, width) of the image.
    """
    h, w = img_shape
    out = boxes.copy()
    out[..., 0] = np.clip(out[..., 0], 0, w)
    out[..., 1] = np.clip(out[..., 1], 0, h)
    out[..., 2] = np.clip(out[..., 2], 0, w)
    out[..., 3] = np.clip(out[..., 3], 0, h)
    return out


def compute_iou(
    box1: NDArray[np.float32],
    box2: NDArray[np.float32],
) -> float:
    """
    Compute IoU between two xyxy boxes.

    Args:
        box1: Shape (4,).
        box2: Shape (4,).

    Returns:
        IoU in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def crop_box(
    image: NDArray[np.uint8],
    box: NDArray[np.float32],
    padding: float = 0.0,
) -> NDArray[np.uint8]:
    """
    Crop a region from an image using an xyxy box.

    Args:
        image:   H×W×C uint8 image.
        box:     Shape (4,) in xyxy format.
        padding: Fractional padding around the box (0.1 = 10%).

    Raises:
        ValueError: If the resulting crop region is empty.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box.astype(np.float32)

    if padding > 0:
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * padding
        y1 -= bh * padding
        x2 += bw * padding
        y2 += bh * padding

    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Empty crop after clipping: [{x1},{y1},{x2},{y2}]")

    return image[y1:y2, x1:x2]
