"""
Data schemas for the detection stage.

Detection is the data contract between Phase 1 (detection)
and Phase 2 (tracking). BoostTrack++ consumes List[Detection].
"""

from dataclasses import dataclass


@dataclass
class Detection:
    """
    A single person detection from YOLOv11s.

    Attributes:
        x1, y1, x2, y2: Bounding box in absolute pixel coordinates (xyxy).
        confidence:      Detection confidence score in [0, 1].
        class_id:        COCO class ID. Always 0 (person) after filtering.
        frame_idx:       Frame index this detection was produced from.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    frame_idx: int = 0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def is_valid(self) -> bool:
        """Check positive area and valid confidence before passing to tracker."""
        return (
            self.x2 > self.x1
            and self.y2 > self.y1
            and 0.0 <= self.confidence <= 1.0
        )
