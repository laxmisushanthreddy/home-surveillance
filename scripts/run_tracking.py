"""
Standalone detection + tracking script — Phase 2 test.

Runs YOLOv11s + BoostTrack++ and annotates each frame with
track IDs and bounding boxes.

Usage:
    python scripts/run_tracking.py
    python scripts/run_tracking.py --source video.mp4 --save
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surveillance.core.config import load_config
from surveillance.core.logger import configure_logging, get_logger
from surveillance.core.constants import COLOR_MAP
from surveillance.detection.detector import PersonDetector
from surveillance.tracking.tracker import MultiObjectTracker
from surveillance.tracking.schema import TrackState
from surveillance.utils.timer import FrameTimer
from surveillance.utils.visualization import draw_fps

configure_logging()
logger = get_logger(__name__)


def draw_track(image, person, show_state: bool = True):
    """Draw tracked person box with ID and state."""
    img = image.copy()
    color = {
        TrackState.Confirmed: (0, 255, 0),
        TrackState.Tentative: (255, 165, 0),
        TrackState.Lost:      (0, 0, 255),
    }.get(person.state, (128, 128, 128))

    x1, y1, x2, y2 = int(person.x1), int(person.y1), int(person.x2), int(person.y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    state_str = person.state.name if show_state else ""
    label = f"ID:{person.track_id} {state_str} {person.confidence:.2f}"
    cv2.putText(img, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: Tracking")
    p.add_argument("--source", default=0)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    det_cfg = load_config("detection")
    trk_cfg = load_config("tracking")

    detector = PersonDetector(det_cfg)
    detector.warmup()
    tracker = MultiObjectTracker(trk_cfg)

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open: %s", args.source)
        sys.exit(1)

    writer = None
    if args.save:
        out = Path("outputs/videos/tracking_output.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps_src, (w, h))

    timer = FrameTimer(window=30)
    frame_idx = 0
    logger.info("Tracking running. Press q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame, frame_idx=frame_idx)
            tracked = tracker.update(detections, frame_idx=frame_idx)

            annotated = frame.copy()
            for person in tracked:
                annotated = draw_track(annotated, person)

            timer.tick()
            annotated = draw_fps(annotated, timer.fps())

            # Stats overlay
            n_confirmed = sum(1 for p in tracked if p.is_confirmed)
            cv2.putText(annotated, f"Tracks: {n_confirmed}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if writer:
                writer.write(annotated)

            cv2.imshow("Phase 2 - Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("Done. %d frames. FPS: %.1f", frame_idx, timer.fps())


if __name__ == "__main__":
    main()
