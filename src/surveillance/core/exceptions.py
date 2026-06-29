"""
Custom exception hierarchy for the surveillance system.

Each subsystem gets its own exception class so callers
can handle failure modes precisely rather than catching
the generic Exception.
"""


class SurveillanceError(Exception):
    """Base class for all project exceptions."""


# ── Configuration ─────────────────────────────────────────────────────────────
class ConfigError(SurveillanceError):
    """Raised when a config file is missing, malformed, or has invalid values."""

class MissingConfigKeyError(ConfigError):
    """Raised when a required config key is absent."""


# ── Model Weights ─────────────────────────────────────────────────────────────
class WeightError(SurveillanceError):
    """Base class for weight-related errors."""

class WeightNotFoundError(WeightError):
    """Raised when a weight file does not exist at the expected path."""

class WeightChecksumError(WeightError):
    """Raised when a downloaded weight file fails SHA-256 verification."""

class WeightDownloadError(WeightError):
    """Raised when a weight download fails."""


# ── Inference ─────────────────────────────────────────────────────────────────
class ModelError(SurveillanceError):
    """Base class for model-related errors."""

class ModelLoadError(ModelError):
    """Raised when a model cannot be loaded from weights."""

class ModelInferenceError(ModelError):
    """Raised when a model forward pass fails."""


# ── Video / Camera ────────────────────────────────────────────────────────────
class VideoError(SurveillanceError):
    """Base class for video/camera errors."""

class CameraOpenError(VideoError):
    """Raised when OpenCV cannot open a camera or video file."""

class FrameReadError(VideoError):
    """Raised when a frame cannot be read from the stream."""


# ── Image Processing ──────────────────────────────────────────────────────────
class ImageError(SurveillanceError):
    """Base class for image processing errors."""

class InvalidImageError(ImageError):
    """Raised when an image is None, empty, or has unexpected shape."""


# ── Alert System ──────────────────────────────────────────────────────────────
class AlertError(SurveillanceError):
    """Base class for alert system errors."""

class FirebaseError(AlertError):
    """Raised on Firebase connection or write failure."""

class TwilioError(AlertError):
    """Raised on Twilio SMS delivery failure."""
