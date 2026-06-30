"""
Face quality assessment.

Filters faces that are too blurry, too small, or at extreme angles
before passing them to mask classification and recognition.

We use simple, fast metrics — no neural quality model.
For research-grade quality assessment, FaceQnet or SER-FIQ could
be substituted here without changing the downstream interface.
"""

import cv2
import numpy as np
from numpy.typing import NDArray


def laplacian_variance(face: NDArray[np.uint8]) -> float:
    """
    Compute Laplacian variance as a blur/sharpness score.

    The Laplacian operator highlights edges. A sharp face has many
    strong edges (high variance). A blurry face has few/weak edges
    (low variance).

    Args:
        face: H x W x 3 uint8 BGR face crop (any size).

    Returns:
        Float sharpness score. Higher = sharper.
        Typical values:
          < 10:  Very blurry (motion blur, out of focus)
          10-50: Moderately blurry (may still work for recognition)
          > 50:  Acceptable sharpness
          > 200: Sharp
    """
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_pose_angle(landmarks: np.ndarray) -> float:
    """
    Estimate approximate yaw (left-right head rotation) from landmarks.

    Uses the horizontal distance between eyes relative to face width
    as a proxy for yaw angle. This is a fast approximation — not a
    true 3D pose estimator.

    Args:
        landmarks: (5, 2) array. landmarks[0] = left eye, landmarks[1] = right eye.

    Returns:
        Approximate yaw in degrees. 0 = frontal. >45 = profile.
    """
    left_eye  = landmarks[0]
    right_eye = landmarks[1]
    nose      = landmarks[2]

    eye_center = (left_eye + right_eye) / 2
    # Horizontal offset of nose from eye midpoint, normalized by eye distance
    eye_dist = np.linalg.norm(right_eye - left_eye)
    if eye_dist < 1e-6:
        return 90.0  # Degenerate — treat as profile

    horizontal_offset = abs(nose[0] - eye_center[0])
    yaw_approx = np.degrees(np.arctan2(horizontal_offset, eye_dist))
    return float(yaw_approx)


class FaceQualityFilter:
    """
    Stateless face quality filter.

    Applies blur, size, and pose checks to reject low-quality faces
    before they reach mask classification and recognition.
    """

    def __init__(
        self,
        min_blur_score: float = 50.0,
        min_face_size: int = 20,
        max_yaw_degrees: float = 45.0,
    ) -> None:
        """
        Args:
            min_blur_score:   Minimum Laplacian variance (sharpness threshold).
            min_face_size:    Minimum face width AND height in pixels.
            max_yaw_degrees:  Maximum estimated yaw before rejection.
        """
        self._min_blur = min_blur_score
        self._min_size = min_face_size
        self._max_yaw  = max_yaw_degrees

    def assess(
        self,
        face_crop: NDArray[np.uint8],
        landmarks: np.ndarray,
        face_bbox: tuple[float, float, float, float],
    ) -> tuple[float, bool]:
        """
        Assess quality of a detected face.

        Args:
            face_crop:  Aligned or raw face crop (any size, uint8 BGR).
            landmarks:  (5, 2) facial landmarks.
            face_bbox:  (x1, y1, x2, y2) face box in frame coordinates.

        Returns:
            Tuple of:
              - quality_score: Laplacian variance (blur score).
              - passes:        True if face passes all quality checks.
        """
        # ── Size check ────────────────────────────────────────────────────
        x1, y1, x2, y2 = face_bbox
        w, h = x2 - x1, y2 - y1
        if w < self._min_size or h < self._min_size:
            return 0.0, False

        # ── Blur check ────────────────────────────────────────────────────
        blur_score = laplacian_variance(face_crop)
        if blur_score < self._min_blur:
            return blur_score, False

        # ── Pose check ────────────────────────────────────────────────────
        yaw = estimate_pose_angle(landmarks)
        if yaw > self._max_yaw:
            return blur_score, False

        return blur_score, True
