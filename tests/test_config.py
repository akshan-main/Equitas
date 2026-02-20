"""Test config loading and validation."""
from __future__ import annotations

import os
import tempfile

import pytest

from equitas.config import (
    ExperimentConfig,
    load_config,
    save_config,
    validate_config,
)


class TestConfig:
    def test_default_config_valid(self):
        config = ExperimentConfig()
        validate_config(config)

    def test_round_trip(self):
        config = ExperimentConfig(name="test_roundtrip", seed=123)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            path = f.name
        try:
            save_config(config, path)
            loaded = load_config(path)
            assert loaded.name == "test_roundtrip"
            assert loaded.seed == 123
        finally:
            os.unlink(path)

    def test_invalid_corruption_rate(self):
        config = ExperimentConfig()
        config.corruption.corruption_rate = 1.5
        with pytest.raises(AssertionError):
            validate_config(config)

    def test_invalid_adversary_type(self):
        config = ExperimentConfig()
        config.corruption.adversary_type = "nonexistent"
        with pytest.raises(AssertionError):
            validate_config(config)
