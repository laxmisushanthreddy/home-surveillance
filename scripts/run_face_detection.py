"""
Standalone detection + tracking + face detection script — Phase 3 test.

Usage:
    python scripts/run_face_detection.py
    python scripts/run_face_detection.py --source video.mp4 --save
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surveillance.core.config import load_config
from surveillance.core.logger import configure_logging, get_logger
from surveillance.detection.detector import PersonDetector
from surveillance.face.detector import FaceDetector
from surveillance.tracking.tracker import MultiObjectTracker
from surveillance.tracking.schema import TrackState
from surveillance.utils.timer import FrameTimer
from surveillance.utils.visualization import draw_fps

configure_logging()
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3: Face Detection")
    p.add_argument("--source", default=0)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    det_cfg  = load_config("detection")
    trk_cfg  = load_config("tracking")
    face_cfg = load_config("face_detection")

    detector      = PersonDetector(det_cfg)
    tracker       = MultiObjectTracker(trk_cfg)
    face_detector = FaceDetector(face_cfg)

    detector.warmup()

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open: %s", args.source)
        sys.exit(1)

    writer = None
    if args.save:
        out = Path("outputs/videos/face_detection_output.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                  cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))

    timer = FrameTimer(window=30)
    frame_idx = 0
    logger.info("Running Phase 3. Press q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame, frame_idx=frame_idx)
            tracked    = tracker.update(detections, frame_idx=frame_idx)
            faces      = face_detector.detect(frame, tracked, frame_idx=frame_idx)

            annotated = frame.copy()

            # Draw person tracks
            for person in tracked:
                color = (0, 255, 0) if person.is_confirmed else (255, 165, 0)
                x1,y1,x2,y2 = int(person.x1),int(person.y1),int(person.x2),int(person.y2)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                cv2.putText(annotated, f"ID:{person.track_id}",
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw face detections
            for face in faces:
                fx1,fy1,fx2,fy2 = [int(v) for v in face.bbox_in_frame]
                cv2.rectangle(annotated, (fx1,fy1), (fx2,fy2), (0,165,255), 2)
                # Draw landmarks
                for lm in face.landmarks:
                    cv2.circle(annotated, (int(lm[0]), int(lm[1])), 2, (0,0,255), -1)
                label = f"Q:{face.quality_score:.0f}"
                cv2.putText(annotated, label, (fx1, fy1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,165,255), 1)

                # Show aligned face thumbnail in corner
                if face.aligned_face is not None:
                    thumb = cv2.resize(face.aligned_face, (56, 56))
                    annotated[10:66, 10:66] = thumb

            timer.tick()
            annotated = draw_fps(annotated, timer.fps())
            cv2.putText(annotated, f"Faces: {len(faces)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

            if writer:
                writer.write(annotated)

            cv2.imshow("Phase 3 - Face Detection", annotated)
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
