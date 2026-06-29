"""
YOLOv11s Person Detector — Adapter pattern.

PersonDetector.detect(frame) -> List[Detection]
All downstream phases call this interface; none touch Ultralytics directly.
Swapping the underlying model = rewriting only this file.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray
from ultralytics import YOLO

from surveillance.core.config import load_config
from surveillance.core.constants import WEIGHTS_DIR
from surveillance.core.exceptions import ModelInferenceError, ModelLoadError
from surveillance.core.logger import get_logger
from surveillance.detection.schema import Detection
from surveillance.utils.timer import StageTimer
from surveillance.weights.manager import WeightManager

logger = get_logger(__name__)


class PersonDetector:
    """
    Real-time person detector using YOLOv11s.

    Lifecycle:
        detector = PersonDetector(cfg)      # Load model once
        for frame in video_stream:
            detections = detector.detect(frame)   # Fast per-frame call
    """

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg if cfg is not None else load_config("detection")
        self._model_cfg = self._cfg.model.yolo

        self._device = self._resolve_device()
        self._conf = self._model_cfg.confidence_threshold
        self._iou = self._model_cfg.iou_threshold
        self._input_size = self._model_cfg.input_size
        self._person_class_id = self._model_cfg.person_class_id
        self._max_det = self._model_cfg.max_detections
        self._half = self._cfg.inference.half_precision and self._device != "cpu"

        self._model: Optional[YOLO] = None
        self._frame_count: int = 0

        self._load_model()

    def _resolve_device(self) -> str:
        requested = self._model_cfg.device
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available. Falling back to CPU."
            )
            return "cpu"
        return requested

    def _load_model(self) -> None:
        weight_filename = self._model_cfg.weights
        weight_path = WEIGHTS_DIR / weight_filename

        if not weight_path.exists():
            logger.info("Weight not found locally. Trying WeightManager download.")
            try:
                manager = WeightManager()
                weight_path = manager.get("yolov11s")
            except Exception:
                logger.info(
                    "WeightManager unavailable. Ultralytics will auto-download."
                )
                weight_path = Path(weight_filename)

        try:
            logger.info("Loading YOLOv11s from: %s", weight_path)
            self._model = YOLO(str(weight_path))
            self._model.fuse()
            logger.info(
                "YOLOv11s ready. Device: %s | FP16: %s",
                self._device, self._half,
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLOv11s: {e}") from e

    def detect(
        self,
        frame: NDArray[np.uint8],
        frame_idx: int = 0,
    ) -> list[Detection]:
        """
        Run person detection on a single BGR frame.

        Args:
            frame:     H x W x 3 BGR uint8 numpy array from OpenCV.
            frame_idx: Frame counter for bookkeeping.

        Returns:
            List[Detection] sorted by confidence descending.
            Empty list if no persons detected or frame is invalid.
        """
        if frame is None or frame.size == 0:
            logger.warning("detect() received empty frame. Returning [].")
            return []

        self._frame_count += 1

        try:
            with StageTimer("YOLOv11s") as timer:
                results = self._model.predict(
                    source=frame,
                    conf=self._conf,
                    iou=self._iou,
                    imgsz=self._input_size,
                    classes=[self._person_class_id],
                    max_det=self._max_det,
                    device=self._device,
                    half=self._half,
                    verbose=self._cfg.inference.verbose,
                )

            if self._frame_count % 100 == 0:
                logger.debug(
                    "Frame %d | %.1f ms", self._frame_count, timer.elapsed_ms
                )

        except Exception as e:
            raise ModelInferenceError(f"YOLOv11s forward pass failed: {e}") from e

        return self._parse_results(results, frame_idx)

    def _parse_results(self, results, frame_idx: int) -> list[Detection]:
        detections: list[Detection] = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            if int(classes[i]) != self._person_class_id:
                continue
            det = Detection(
                x1=float(xyxy[i, 0]),
                y1=float(xyxy[i, 1]),
                x2=float(xyxy[i, 2]),
                y2=float(xyxy[i, 3]),
                confidence=float(confs[i]),
                class_id=int(classes[i]),
                frame_idx=frame_idx,
            )
            if det.is_valid():
                detections.append(det)

        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.debug("Frame %d | %d person(s) detected", frame_idx, len(detections))
        return detections

    def detect_batch(
        self,
        frames: list[NDArray[np.uint8]],
        start_frame_idx: int = 0,
    ) -> list[list[Detection]]:
        """Batch inference — more efficient than calling detect() in a loop."""
        if not frames:
            return []
        try:
            results = self._model.predict(
                source=frames,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._input_size,
                classes=[self._person_class_id],
                max_det=self._max_det,
                device=self._device,
                half=self._half,
                verbose=False,
            )
        except Exception as e:
            raise ModelInferenceError(f"Batch inference failed: {e}") from e

        return [
            self._parse_results([r], start_frame_idx + i)
            for i, r in enumerate(results)
        ]

    def warmup(self, n_iterations: int = 3) -> None:
        """
        Run dummy forward passes to warm up GPU CUDA kernels.
        Call once after initialization, before the real-time loop.
        """
        logger.info("Warming up YOLOv11s (%d passes)...", n_iterations)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(n_iterations):
            self._model.predict(
                source=dummy,
                imgsz=self._input_size,
                device=self._device,
                verbose=False,
            )
        logger.info("Warmup complete.")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def device(self) -> str:
        return self._device
