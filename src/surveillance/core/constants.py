"""
Project-wide structural constants.

These never change across experiments.
Hyperparameters belong in configs/ — not here.
"""

from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────
# Resolves to repo root regardless of where the calling script lives.
# Path: constants.py → core/ → surveillance/ → src/ → PROJECT_ROOT
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

# ── Key Directories ───────────────────────────────────────────────────────────
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"

# ── Supported File Types ──────────────────────────────────────────────────────
SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
)
SUPPORTED_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm"}
)

# ── Annotation Colors (BGR for OpenCV) ───────────────────────────────────────
COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "person":   (0, 255, 0),
    "face":     (255, 165, 0),
    "mask":     (0, 0, 255),
    "no_mask":  (0, 255, 0),
    "track_id": (255, 255, 0),
    "alert":    (0, 0, 255),
}

# ── Face Processing ───────────────────────────────────────────────────────────
# Standard aligned face resolution used by ArcFace / AdaFace / InsightFace
ALIGNED_FACE_SIZE: tuple[int, int] = (112, 112)

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_DIM: int = 512  # AdaFace IR-101 produces 512-D L2-normalized vectors
