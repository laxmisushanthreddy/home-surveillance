"""Shared pytest fixtures."""

import numpy as np
import pytest

from surveillance.core.config import load_config
from surveillance.core.logger import configure_logging


@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    configure_logging(enable_rich=False)


@pytest.fixture
def base_config():
    return load_config()


@pytest.fixture
def dummy_bgr_image() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_boxes() -> np.ndarray:
    return np.array([
        [10,  20,  100, 200],
        [50,  60,  150, 180],
        [200, 300, 400, 450],
        [0,   0,   640, 480],
    ], dtype=np.float32)
