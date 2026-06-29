"""
Latency and FPS measurement utilities.

Usage:
    timer = FrameTimer()
    timer.tick()
    fps = timer.fps()

    with StageTimer("YOLOv11s") as t:
        result = detector.detect(frame)
    logger.info("Inference: %.1f ms", t.elapsed_ms)
"""

import time
from collections import deque
from dataclasses import dataclass, field


class FrameTimer:
    """
    Rolling-window FPS counter for real-time video pipelines.
    Maintains the last `window` frame timestamps.
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        """Record current timestamp. Call once per processed frame."""
        self._timestamps.append(time.perf_counter())

    def fps(self) -> float:
        """Compute FPS from the rolling window. Returns 0.0 if < 2 frames."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def latency_ms(self) -> float:
        """Most recent frame-to-frame latency in milliseconds."""
        if len(self._timestamps) < 2:
            return 0.0
        return (self._timestamps[-1] - self._timestamps[-2]) * 1000.0

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        self._timestamps.clear()


@dataclass
class StageTimer:
    """
    Context manager for measuring latency of a single pipeline stage.

    Usage:
        with StageTimer("Detection") as t:
            result = model.infer(frame)
        print(f"{t.stage_name}: {t.elapsed_ms:.1f} ms")
    """

    stage_name: str
    elapsed_ms: float = field(default=0.0, init=False)
    _start: float = field(default=0.0, init=False, repr=False)

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
