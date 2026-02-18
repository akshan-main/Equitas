"""Shared test fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from equitas.config import ExperimentConfig, LLMConfig
from equitas.worlds.governance import GovernanceWorld


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def governance_world():
    return GovernanceWorld(
        crisis_axes=["resource_scarcity", "external_threat", "inequality", "economic_instability"],
        policy_dims=["tax_merchants", "welfare_workers", "military_spend", "education_investment"],
        actions_per_crisis=3,
        class_ids=["guardian", "auxiliary", "producer"],
    )


@pytest.fixture
def default_config():
    return ExperimentConfig(
        name="test",
        environment="governance",
        seed=42,
        num_runs=1,
        llm=LLMConfig(model="gpt-4o"),
    )
