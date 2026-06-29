"""
Standalone person detection script — Phase 1 test.

Usage:
    python scripts/run_detection.py                       # webcam
    python scripts/run_detection.py --source video.mp4    # video file
    python scripts/run_detection.py --source video.mp4 --save
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surveillance.core.config import load_config
from surveillance.core.logger import configure_logging, get_logger
from surveillance.detection.detector import PersonDetector
from surveillance.utils.timer import FrameTimer
from surveillance.utils.visualization import draw_bbox, draw_fps

configure_logging()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: Person Detection")
    parser.add_argument("--source", default=0,
                        help="0=webcam or path to video file")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output to outputs/videos/")
    parser.add_argument("--conf", type=float, default=None,
                        help="Override confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config("detection")

    if args.conf is not None:
        from surveillance.core.config import override_config
        cfg = override_config(cfg, {"model.yolo.confidence_threshold": args.conf})

    detector = PersonDetector(cfg)
    detector.warmup()

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", args.source)
        sys.exit(1)

    writer = None
    if args.save:
        out_path = Path("outputs/videos/detection_output.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h)
        )
        logger.info("Saving output to: %s", out_path)

    timer = FrameTimer(window=30)
    frame_idx = 0
    logger.info("Running. Press q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream.")
                break

            detections = detector.detect(frame, frame_idx=frame_idx)

            annotated = frame.copy()
            for det in detections:
                annotated = draw_bbox(
                    annotated,
                    (int(det.x1), int(det.y1), int(det.x2), int(det.y2)),
                    label=f"person {det.confidence:.2f}",
                )

            timer.tick()
            annotated = draw_fps(annotated, timer.fps())

            if writer:
                writer.write(annotated)

            cv2.imshow("Phase 1 - Person Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info(
            "Done. %d frames processed. FPS: %.1f",
            frame_idx, timer.fps()
        )


if __name__ == "__main__":
    main()
