"""
FaceDetector — SCRFD-2.5GF adapter.

Wraps InsightFace's SCRFD model behind a clean interface.
Only this file knows InsightFace exists.

Design pattern: Adapter
  FaceDetector.detect(frame, tracked_persons) -> List[FaceDetectionResult]

Usage:
    from surveillance.face.detector import FaceDetector
    cfg = load_config("face_detection")
    detector = FaceDetector(cfg)
    results = detector.detect(frame, confirmed_tracks)
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from surveillance.core.config import load_config
from surveillance.core.exceptions import ModelLoadError, ModelInferenceError
from surveillance.core.logger import get_logger
from surveillance.face.alignment import align_face, landmarks_to_frame_coords
from surveillance.face.quality import FaceQualityFilter
from surveillance.face.schema import FaceDetectionResult
from surveillance.tracking.schema import TrackedPerson
from surveillance.utils.bbox import crop_box
from surveillance.utils.timer import StageTimer

logger = get_logger(__name__)


class FaceDetector:
    """
    SCRFD-2.5GF face detector with 5-point landmark alignment.

    Processes confirmed tracks only. For each track:
      1. Crop person ROI from frame (with padding)
      2. Run SCRFD on crop
      3. Filter by quality
      4. Align best face to 112x112
      5. Return FaceDetectionResult

    Lifecycle:
        detector = FaceDetector(cfg)   # Loads model once
        for frame, tracked in pipeline:
            results = detector.detect(frame, tracked)
    """

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg if cfg is not None else load_config("face_detection")
        self._model_cfg   = self._cfg.model.scrfd
        self._quality_cfg = self._cfg.quality
        self._pipe_cfg    = self._cfg.pipeline

        self._model = None
        self._quality_filter = FaceQualityFilter(
            min_blur_score=self._quality_cfg.min_blur_score,
            min_face_size=self._quality_cfg.min_face_size,
            max_yaw_degrees=self._quality_cfg.max_yaw_degrees,
        )

        self._load_model()

    def _load_model(self) -> None:
        """Load SCRFD via InsightFace. Auto-downloads weights if needed."""
        try:
            import insightface
            from insightface.app import FaceAnalysis

            logger.info(
                "Loading SCRFD via InsightFace model pack: %s",
                self._model_cfg.name,
            )

            # FaceAnalysis with det_only — we only need detection + landmarks
            # Recognition is handled by AdaFace in Phase 5
            self._app = FaceAnalysis(
                name=self._model_cfg.name,
                allowed_modules=["detection"],  # Skip recognition module
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            det_size = tuple(self._model_cfg.det_size)
            self._app.prepare(ctx_id=0, det_size=det_size)

            logger.info("SCRFD loaded. det_size=%s", det_size)

        except Exception as e:
            raise ModelLoadError(f"Failed to load SCRFD: {e}") from e

    def detect(
        self,
        frame: NDArray[np.uint8],
        tracked_persons: list[TrackedPerson],
        frame_idx: int = 0,
    ) -> list[FaceDetectionResult]:
        """
        Detect and align faces for all confirmed tracked persons.

        Args:
            frame:           Full BGR frame from camera.
            tracked_persons: All tracked persons (confirmed + lost).
                             Only confirmed tracks are processed.
            frame_idx:       Current frame index.

        Returns:
            List[FaceDetectionResult] — one per detected face.
            May be shorter than len(tracked_persons) if some tracks
            have no detectable face this frame.
        """
        confirmed = [p for p in tracked_persons if p.is_confirmed]
        if not confirmed or frame is None:
            return []

        results: list[FaceDetectionResult] = []
        padding = self._pipe_cfg.person_crop_padding

        with StageTimer("SCRFD face detection") as timer:
            for person in confirmed:
                face_result = self._process_track(
                    frame, person, padding, frame_idx
                )
                if face_result is not None:
                    results.append(face_result)

        logger.debug(
            "Frame %d | SCRFD: %.1f ms | %d/%d tracks with face",
            frame_idx,
            timer.elapsed_ms,
            len(results),
            len(confirmed),
        )

        return results

    def _process_track(
        self,
        frame: NDArray[np.uint8],
        person: TrackedPerson,
        padding: float,
        frame_idx: int,
    ) -> FaceDetectionResult | None:
        """
        Process a single confirmed track: crop → detect → quality → align.

        Returns FaceDetectionResult or None if no valid face found.
        """
        # ── Step 1: Crop person ROI with padding ──────────────────────────
        try:
            box = np.array([person.x1, person.y1, person.x2, person.y2],
                           dtype=np.float32)
            person_crop = crop_box(frame, box, padding=padding)
        except ValueError:
            logger.debug("Track %d: invalid crop box", person.track_id)
            return None

        crop_x1 = max(0, person.x1 - (person.x2 - person.x1) * padding)
        crop_y1 = max(0, person.y1 - (person.y2 - person.y1) * padding)

        # ── Step 2: Run SCRFD on person crop ──────────────────────────────
        try:
            faces = self._app.get(person_crop)
        except Exception as e:
            logger.debug("Track %d: SCRFD inference failed: %s",
                         person.track_id, e)
            return None

        if not faces:
            return None

        # ── Step 3: Select best face (highest confidence) ─────────────────
        select_by = self._pipe_cfg.select_by
        if select_by == "confidence":
            best_face = max(faces, key=lambda f: f.det_score)
        elif select_by == "size":
            best_face = max(faces,
                            key=lambda f: (f.bbox[2]-f.bbox[0]) *
                                          (f.bbox[3]-f.bbox[1]))
        else:
            best_face = faces[0]

        # ── Step 4: Convert to frame coordinates ──────────────────────────
        bbox_in_crop = best_face.bbox          # xyxy in crop space
        lm_in_crop   = best_face.kps           # (5, 2) in crop space

        bbox_in_frame = (
            float(bbox_in_crop[0] + crop_x1),
            float(bbox_in_crop[1] + crop_y1),
            float(bbox_in_crop[2] + crop_x1),
            float(bbox_in_crop[3] + crop_y1),
        )
        lm_in_frame = landmarks_to_frame_coords(lm_in_crop, crop_x1, crop_y1)

        # ── Step 5: Quality assessment (on raw crop before alignment) ──────
        # Extract face crop from person_crop for quality check
        x1c = max(0, int(bbox_in_crop[0]))
        y1c = max(0, int(bbox_in_crop[1]))
        x2c = min(person_crop.shape[1], int(bbox_in_crop[2]))
        y2c = min(person_crop.shape[0], int(bbox_in_crop[3]))
        raw_face_crop = person_crop[y1c:y2c, x1c:x2c]

        if raw_face_crop.size == 0:
            return None

        quality_score, passes = self._quality_filter.assess(
            raw_face_crop, lm_in_crop, bbox_in_frame
        )

        if not passes:
            logger.debug(
                "Track %d: face rejected (quality=%.1f)",
                person.track_id, quality_score,
            )
            return None

        # ── Step 6: Align face to 112x112 ────────────────────────────────
        aligned = align_face(frame, lm_in_frame.astype(np.float32))
        if aligned is None:
            logger.debug("Track %d: alignment failed", person.track_id)
            return None

        return FaceDetectionResult(
            track_id=person.track_id,
            aligned_face=aligned,
            landmarks=lm_in_frame,
            bbox_in_frame=bbox_in_frame,
            confidence=float(best_face.det_score),
            quality_score=quality_score,
            frame_idx=frame_idx,
        )
