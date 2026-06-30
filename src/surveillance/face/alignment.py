"""
5-point facial landmark alignment.

Computes an affine transformation from detected landmarks to the
ArcFace/AdaFace reference template and warps the face to 112x112.

This alignment is MANDATORY for AdaFace recognition. Unaligned faces
produce severely degraded embeddings.

Reference template: standard ArcFace alignment target used across
InsightFace, ArcFace, CosFace, AdaFace training pipelines.
"""

import cv2
import numpy as np
from numpy.typing import NDArray


# Standard ArcFace/AdaFace 112x112 reference landmarks
# Order: left_eye, right_eye, nose_tip, left_mouth, right_mouth
ARCFACE_REFERENCE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

ALIGNED_SIZE = (112, 112)


def estimate_affine(
    src_landmarks: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Estimate affine transformation matrix from detected landmarks
    to the ArcFace reference template.

    Uses cv2.estimateAffinePartial2D which finds the optimal
    similarity transform (rotation + scale + translation, no shear).
    This is preferable to full affine for faces because faces do not
    shear — only rotate, scale, and translate.

    Args:
        src_landmarks: (5, 2) detected landmarks in image coordinates.

    Returns:
        (2, 3) affine transformation matrix M such that:
        aligned = cv2.warpAffine(image, M, (112, 112))
    """
    M, _ = cv2.estimateAffinePartial2D(
        src_landmarks,
        ARCFACE_REFERENCE,
        method=cv2.LMEDS,
    )
    if M is None:
        # Fallback: use least-squares if LMEDS fails (degenerate configuration)
        M, _ = cv2.estimateAffinePartial2D(
            src_landmarks,
            ARCFACE_REFERENCE,
            method=cv2.RANSAC,
        )
    return M


def align_face(
    image: NDArray[np.uint8],
    landmarks: NDArray[np.float32],
) -> NDArray[np.uint8] | None:
    """
    Align a face to the ArcFace 112x112 template.

    Args:
        image:     Full BGR frame or person crop (H x W x 3, uint8).
        landmarks: (5, 2) facial landmarks in image coordinates.

    Returns:
        112x112x3 uint8 aligned face, or None if alignment fails.
    """
    M = estimate_affine(landmarks)
    if M is None:
        return None

    aligned = cv2.warpAffine(
        image,
        M,
        ALIGNED_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned


def landmarks_to_frame_coords(
    landmarks_in_crop: NDArray[np.float32],
    crop_x1: float,
    crop_y1: float,
) -> NDArray[np.float32]:
    """
    Convert landmark coordinates from crop space to full frame space.

    Args:
        landmarks_in_crop: (5, 2) landmarks relative to crop top-left.
        crop_x1, crop_y1:  Crop top-left corner in frame coordinates.

    Returns:
        (5, 2) landmarks in frame coordinates.
    """
    offset = np.array([crop_x1, crop_y1], dtype=np.float32)
    return landmarks_in_crop + offset
