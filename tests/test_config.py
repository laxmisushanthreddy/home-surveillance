"""Tests for the configuration system."""

import pytest
from omegaconf import DictConfig

from surveillance.core.config import get_value, load_config
from surveillance.core.exceptions import ConfigError, MissingConfigKeyError


def test_load_base_config_returns_dictconfig():
    assert isinstance(load_config(), DictConfig)


def test_base_config_has_required_top_level_keys(base_config):
    for key in ("project", "hardware", "paths", "logging", "video", "pipeline"):
        assert key in base_config, f"Missing top-level key: {key}"


def test_config_project_name(base_config):
    assert base_config.project.name == "home-surveillance"


def test_config_is_readonly(base_config):
    with pytest.raises(Exception):
        base_config.project.name = "tampered"


def test_load_nonexistent_phase_raises():
    with pytest.raises(ConfigError):
        load_config("nonexistent_xyz")


def test_get_value_existing_key(base_config):
    assert get_value(base_config, "project.name") == "home-surveillance"


def test_get_value_missing_key_with_default(base_config):
    assert get_value(base_config, "no.such.key", default="fallback") == "fallback"


def test_get_value_missing_key_no_default(base_config):
    with pytest.raises(MissingConfigKeyError):
        get_value(base_config, "no.such.key")
