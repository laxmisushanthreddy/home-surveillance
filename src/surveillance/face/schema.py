"""
Data schemas for the face detection stage.

FaceDetectionResult is the contract between Phase 3 (face detection)
and Phase 4 (mask classification). The aligned_face field is a
112x112 BGR image ready for EfficientNetV2-B0 and AdaFace.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class FaceDetectionResult:
    """
    A detected and aligned face associated with a tracked person.

    Attributes:
        track_id:      ID of the TrackedPerson this face belongs to.
        aligned_face:  112x112x3 BGR uint8 aligned face crop.
                       Ready for mask classification and recognition.
        landmarks:     (5, 2) float32 array of facial landmarks in
                       ORIGINAL frame coordinates (not crop coords).
                       Order: left_eye, right_eye, nose, left_mouth, right_mouth.
        bbox_in_frame: (x1, y1, x2, y2) face box in original frame coords.
        confidence:    SCRFD face detection confidence [0, 1].
        quality_score: Laplacian variance blur score. Higher = sharper.
                       Threshold: > 50 considered acceptable quality.
        frame_idx:     Frame index this result was produced from.
    """
    track_id:       int
    aligned_face:   np.ndarray           # shape (112, 112, 3), dtype uint8
    landmarks:      np.ndarray           # shape (5, 2), dtype float32
    bbox_in_frame:  tuple[float, float, float, float]
    confidence:     float
    quality_score:  float
    frame_idx:      int = 0

    @property
    def is_acceptable_quality(self) -> bool:
        """True if face is sharp enough for reliable recognition."""
        return self.quality_score > 50.0

    @property
    def face_width(self) -> float:
        x1, _, x2, _ = self.bbox_in_frame
        return x2 - x1

    @property
    def face_height(self) -> float:
        _, y1, _, y2 = self.bbox_in_frame
        return y2 - y1

    @property
    def is_large_enough(self) -> bool:
        """True if face is large enough for reliable landmark detection."""
        return self.face_width >= 20.0 and self.face_height >= 20.0

    def is_valid(self) -> bool:
        """Full validity check: shape, size, quality, confidence."""
        return (
            self.aligned_face is not None
            and self.aligned_face.shape == (112, 112, 3)
            and self.aligned_face.dtype == np.uint8
            and self.is_large_enough
            and 0.0 <= self.confidence <= 1.0
        )
