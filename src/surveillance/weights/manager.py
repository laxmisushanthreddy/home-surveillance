"""
Model weight manager.

Handles: registry lookup → local existence check →
         download with progress bar → SHA-256 verification → path return.
"""

import hashlib
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

from surveillance.core.constants import WEIGHTS_DIR
from surveillance.core.exceptions import (
    WeightChecksumError,
    WeightDownloadError,
    WeightNotFoundError,
)
from surveillance.core.logger import get_logger

logger = get_logger(__name__)
REGISTRY_PATH: Path = WEIGHTS_DIR / "registry.yaml"


class WeightManager:
    """
    Registry-based model weight manager.

    registry.yaml maps model names to:
      url:      Download URL
      filename: Local filename in weights/
      sha256:   Expected hex digest ("" to skip verification)
    """

    def __init__(self) -> None:
        self._registry: dict = self._load_registry()

    def _load_registry(self) -> dict:
        if not REGISTRY_PATH.exists():
            logger.warning("Weight registry not found at %s.", REGISTRY_PATH)
            return {}
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get(self, model_name: str, force_download: bool = False) -> Path:
        """
        Return local path to a weight file, downloading if necessary.

        Args:
            model_name:     Key in registry.yaml.
            force_download: Re-download even if file exists locally.

        Raises:
            WeightNotFoundError:  model_name not in registry.
            WeightDownloadError:  Network or HTTP error during download.
            WeightChecksumError:  SHA-256 mismatch after download.
        """
        if model_name not in self._registry:
            raise WeightNotFoundError(
                f"'{model_name}' not in registry: {REGISTRY_PATH}"
            )

        entry = self._registry[model_name]
        local_path = WEIGHTS_DIR / entry["filename"]

        if local_path.exists() and not force_download:
            logger.debug("Weight exists locally: %s", local_path.name)
            if entry.get("sha256"):
                self._verify(local_path, entry["sha256"])
            return local_path

        logger.info("Downloading: %s → %s", model_name, local_path.name)
        self._download(entry["url"], local_path)

        if entry.get("sha256"):
            self._verify(local_path, entry["sha256"])

        return local_path

    def _download(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            raise WeightDownloadError(f"Failed to download {url}: {e}") from e

        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, ncols=80
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        logger.info("Downloaded: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)

    def _verify(self, path: Path, expected: str) -> None:
        logger.debug("Verifying checksum: %s", path.name)
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        actual = sha256.hexdigest()
        if actual != expected:
            raise WeightChecksumError(
                f"Checksum mismatch for {path.name}:\n"
                f"  Expected: {expected}\n  Got: {actual}"
            )
        logger.debug("Checksum OK: %s", path.name)

    def list_available(self) -> list[str]:
        """Return all model names in the registry."""
        return list(self._registry.keys())

    def is_downloaded(self, model_name: str) -> bool:
        """Check if a weight file exists locally without downloading."""
        if model_name not in self._registry:
            return False
        return (WEIGHTS_DIR / self._registry[model_name]["filename"]).exists()
