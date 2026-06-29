"""
Frame annotation utilities.
All drawing functions return annotated copies — originals are never mutated.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from surveillance.core.constants import COLOR_MAP


def draw_bbox(
    image: NDArray[np.uint8],
    box: tuple[int, int, int, int],
    label: str = "",
    color: tuple[int, int, int] | None = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> NDArray[np.uint8]:
    """
    Draw a bounding box and optional label on an image copy.

    Args:
        image:      BGR uint8 image.
        box:        (x1, y1, x2, y2) pixel coordinates.
        label:      Text drawn above the box.
        color:      BGR color. Defaults to COLOR_MAP["person"].
        thickness:  Line thickness in pixels.
        font_scale: Font size scaling factor.

    Returns:
        Annotated copy.
    """
    img = image.copy()
    color = color or COLOR_MAP["person"]
    x1, y1, x2, y2 = box

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            img, label, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
        )

    return img


def draw_fps(
    image: NDArray[np.uint8],
    fps: float,
    position: tuple[int, int] = (10, 30),
) -> NDArray[np.uint8]:
    """Overlay FPS counter on the image. Returns annotated copy."""
    img = image.copy()
    cv2.putText(
        img, f"FPS: {fps:.1f}", position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    return img
