"""
Image I/O and basic processing utilities.

All functions use OpenCV as the backend.
BGR is native OpenCV format — convert to RGB only when handing off to PyTorch.
"""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from surveillance.core.exceptions import InvalidImageError
from surveillance.core.logger import get_logger

logger = get_logger(__name__)


def load_image(path: Path | str) -> NDArray[np.uint8]:
    """Load an image from disk. Returns BGR uint8."""
    path = Path(path)
    if not path.exists():
        raise InvalidImageError(f"Image file not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise InvalidImageError(f"cv2.imread returned None for: {path}")
    return img


def save_image(image: NDArray[np.uint8], path: Path | str) -> None:
    """Save a BGR uint8 image to disk. Creates parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise InvalidImageError(f"cv2.imwrite failed for: {path}")


def validate_image(image: NDArray) -> None:
    """
    Assert that an image array is valid.

    Raises:
        InvalidImageError: If None, wrong type, wrong dims, or wrong dtype.
    """
    if image is None:
        raise InvalidImageError("Image is None.")
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"Expected np.ndarray, got {type(image)}.")
    if image.ndim not in (2, 3):
        raise InvalidImageError(f"Expected 2D or 3D array, got shape {image.shape}.")
    if image.dtype != np.uint8:
        raise InvalidImageError(f"Expected uint8, got {image.dtype}.")


def bgr_to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert BGR → RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert RGB → BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def resize_with_aspect(
    image: NDArray[np.uint8],
    target_size: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> NDArray[np.uint8]:
    """
    Resize so the longer side equals target_size, preserving aspect ratio.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def letterbox(
    image: NDArray[np.uint8],
    target_size: tuple[int, int],
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[NDArray[np.uint8], float, tuple[int, int]]:
    """
    Resize with letterboxing (padding to preserve aspect ratio).
    This is the standard YOLO preprocessing step.

    Args:
        image:       Input BGR image.
        target_size: (width, height) of the output.
        pad_color:   BGR padding color.

    Returns:
        Tuple of (padded_image, scale_factor, (pad_x, pad_y)).
    """
    ih, iw = image.shape[:2]
    tw, th = target_size

    scale = min(tw / iw, th / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (tw - new_w) // 2
    pad_y = (th - new_h) // 2

    output = np.full((th, tw, 3), pad_color, dtype=np.uint8)
    output[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return output, scale, (pad_x, pad_y)
