"""
Diagnostic version of Phase 3 test — verbose logging + on-screen FPS.

Logs every frame to outputs/logs/diagnostic_run.log with:
  - detection count, track count, confirmed count, face count
  - per-stage timing

Usage:
    python scripts/diagnose_pipeline.py
"""

import sys
import time
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

configure_logging()
logger = get_logger(__name__)


def main():
    det_cfg  = load_config("detection")
    trk_cfg  = load_config("tracking")
    face_cfg = load_config("face_detection")

    detector      = PersonDetector(det_cfg)
    tracker       = MultiObjectTracker(trk_cfg)
    face_detector = FaceDetector(face_cfg)
    detector.warmup()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        sys.exit(1)

    timer = FrameTimer(window=30)
    frame_idx = 0

    diag_log = open("outputs/logs/diagnostic_run.log", "w")
    diag_log.write("frame,det_ms,trk_ms,face_ms,total_ms,n_det,n_trk,n_confirmed,n_face,track_ids\n")

    print("Diagnostic run started. Press q to quit.")
    print("Watch the terminal for live per-frame stats.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            detections = detector.detect(frame, frame_idx=frame_idx)
            t1 = time.perf_counter()

            tracked = tracker.update(detections, frame_idx=frame_idx)
            t2 = time.perf_counter()

            faces = face_detector.detect(frame, tracked, frame_idx=frame_idx)
            t3 = time.perf_counter()

            det_ms   = (t1 - t0) * 1000
            trk_ms   = (t2 - t1) * 1000
            face_ms  = (t3 - t2) * 1000
            total_ms = (t3 - t0) * 1000

            n_confirmed = sum(1 for p in tracked if p.is_confirmed)
            track_ids = ";".join(str(p.track_id) for p in tracked)

            diag_log.write(
                f"{frame_idx},{det_ms:.1f},{trk_ms:.1f},{face_ms:.1f},{total_ms:.1f},"
                f"{len(detections)},{len(tracked)},{n_confirmed},{len(faces)},{track_ids}\n"
            )
            diag_log.flush()

            # Print every 15 frames so terminal isn't flooded
            if frame_idx % 15 == 0:
                print(f"Frame {frame_idx:5d} | det={det_ms:5.1f}ms trk={trk_ms:5.1f}ms "
                      f"face={face_ms:5.1f}ms total={total_ms:6.1f}ms | "
                      f"detections={len(detections)} tracks={len(tracked)} "
                      f"confirmed={n_confirmed} faces={len(faces)} ids=[{track_ids}]")

            # Draw
            annotated = frame.copy()
            for person in tracked:
                color = (0,255,0) if person.is_confirmed else (0,165,255) if person.state==TrackState.Tentative else (0,0,255)
                x1,y1,x2,y2 = int(person.x1),int(person.y1),int(person.x2),int(person.y2)
                cv2.rectangle(annotated, (x1,y1),(x2,y2), color, 2)
                cv2.putText(annotated, f"ID:{person.track_id} {person.state.name}",
                            (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for face in faces:
                fx1,fy1,fx2,fy2 = [int(v) for v in face.bbox_in_frame]
                cv2.rectangle(annotated, (fx1,fy1),(fx2,fy2), (0,165,255), 2)
                for lm in face.landmarks:
                    cv2.circle(annotated, (int(lm[0]),int(lm[1])), 2, (0,0,255), -1)

            timer.tick()
            cv2.putText(annotated, f"FPS: {timer.fps():.1f}  Total: {total_ms:.0f}ms",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(annotated, f"Tracks:{len(tracked)} Confirmed:{sum(1 for p in tracked if p.is_confirmed)} Faces:{len(faces)}",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.imshow("Diagnostic", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        diag_log.close()
        print(f"\nDone. {frame_idx} frames. Log saved to outputs/logs/diagnostic_run.log")
        print(f"Final FPS: {timer.fps():.1f}")


if __name__ == "__main__":
    main()
