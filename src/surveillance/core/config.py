"""
Centralized configuration loader using OmegaConf.

Usage:
    from surveillance.core.config import load_config
    cfg = load_config("detection")
    print(cfg.model.yolo.confidence_threshold)
"""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from surveillance.core.constants import CONFIGS_DIR
from surveillance.core.exceptions import ConfigError, MissingConfigKeyError
from surveillance.core.logger import get_logger

logger = get_logger(__name__)


def load_config(phase: str | None = None) -> DictConfig:
    """
    Load project configuration.

    Always loads base.yaml first. If `phase` is provided, loads
    configs/<phase>.yaml and merges it on top, with phase values
    taking precedence over base values.

    Args:
        phase: Optional phase name (e.g. "detection", "tracking").

    Returns:
        Merged, read-only OmegaConf DictConfig.

    Raises:
        ConfigError: If any config file is missing or malformed.
    """
    base_path = CONFIGS_DIR / "base.yaml"
    if not base_path.exists():
        raise ConfigError(f"Base config not found: {base_path}")

    try:
        cfg = OmegaConf.load(base_path)
        logger.debug("Loaded base config: %s", base_path)
    except Exception as e:
        raise ConfigError(f"Failed to parse base config: {e}") from e

    if phase is not None:
        phase_path = CONFIGS_DIR / f"{phase}.yaml"
        if not phase_path.exists():
            raise ConfigError(f"Phase config not found: {phase_path}")
        try:
            phase_cfg = OmegaConf.load(phase_path)
            cfg = OmegaConf.merge(cfg, phase_cfg)
            logger.debug("Merged phase config: %s", phase_path)
        except Exception as e:
            raise ConfigError(f"Failed to parse phase config '{phase}': {e}") from e

    OmegaConf.set_readonly(cfg, True)
    return cfg


def get_value(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """
    Safely retrieve a nested key using dot notation.

    Args:
        cfg:     Config object.
        key:     Dot-delimited key, e.g. "model.yolo.confidence_threshold".
        default: Returned if key is absent. If None, raises on missing key.

    Raises:
        MissingConfigKeyError: If key is absent and default is None.
    """
    try:
        value = OmegaConf.select(cfg, key)
        if value is None and default is None:
            raise MissingConfigKeyError(f"Required config key '{key}' not found.")
        return value if value is not None else default
    except Exception as e:
        if default is not None:
            return default
        raise MissingConfigKeyError(f"Config key '{key}' missing: {e}") from e


def override_config(cfg: DictConfig, overrides: dict[str, Any]) -> DictConfig:
    """
    Return a new config with overrides applied. Original is not mutated.

    Used by training scripts to apply CLI argument overrides.
    """
    OmegaConf.set_readonly(cfg, False)
    merged = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    OmegaConf.set_readonly(merged, True)
    return merged
