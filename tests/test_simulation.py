"""Integration test for simulation engine with mock LLM."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from equitas.config import AggregatorConfig, ExperimentConfig, LLMConfig, WorldConfig, CommitteeConfig, CorruptionConfig
from equitas.simulation.engine import run_hierarchical_simulation
from equitas.worlds.governance import GovernanceWorld


@pytest.fixture
def small_config():
    """Minimal config for fast testing."""
    return ExperimentConfig(
        name="test_sim",
        seed=42,
        llm=LLMConfig(model="test"),
        world=WorldConfig(num_rounds=3, actions_per_crisis=3),
        committee=CommitteeConfig(members_per_class=3, num_judges=3),
        corruption=CorruptionConfig(
            corruption_rate=0.3,
            adversary_type="selfish",
            corruption_target="members",
        ),
        aggregators=[
            AggregatorConfig(method="majority_vote"),
            AggregatorConfig(method="multiplicative_weights", eta=1.0),
        ],
    )


@pytest.fixture
def world():
    return GovernanceWorld(
        crisis_axes=["resource_scarcity", "external_threat", "inequality", "economic_instability"],
        policy_dims=["tax_merchants", "welfare_workers", "military_spend", "education_investment"],
        actions_per_crisis=3,
    )


@pytest.mark.asyncio
async def test_hierarchical_simulation_with_mock(small_config, world):
    """Test that the simulation runs end-to-end with a mocked LLM."""
    # Mock the LLM client to return deterministic responses
    mock_complete = AsyncMock(return_value="After careful analysis, Action A is best.\n\nACTION_ID: A")

    with patch("equitas.agents.llm_client.LLMClientWrapper.__init__", lambda self, config: None), \
         patch("equitas.agents.llm_client.LLMClientWrapper.complete", mock_complete):
        result = await run_hierarchical_simulation(small_config, world, seed=42)

    # Check structure
    assert len(result.rounds) == 3
    assert not result.aggregator_log.empty
    assert not result.agent_log.empty

    # Check aggregator log has expected columns
    expected_cols = {"round_id", "aggregator", "chosen_action_id", "city_utility", "fairness_jain"}
    assert expected_cols.issubset(set(result.aggregator_log.columns))

    # Check agent log
    assert "agent_id" in result.agent_log.columns
    assert "role" in result.agent_log.columns

    # Check weight history exists for MW
    assert "multiplicative_weights" in result.weight_history
    assert len(result.weight_history["multiplicative_weights"]) == 3

    # Check oracle is in results
    assert "oracle" in result.aggregator_log["aggregator"].values
