"""
Session-scoped fixtures shared across all test modules.
"""

import pytest
from utils.config import load_config, get_verify_ssl


@pytest.fixture(scope="session")
def cfg():
    """Loaded config.yaml dict, available to every test."""
    return load_config()


@pytest.fixture(scope="session")
def verify_ssl(cfg):
    return get_verify_ssl(cfg)
