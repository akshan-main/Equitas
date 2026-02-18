"""Test all aggregation methods."""
from __future__ import annotations

import numpy as np
import pytest

from equitas.aggregators.base import weighted_vote
from equitas.aggregators.confidence_weighted import ConfidenceWeightedVoteAggregator
from equitas.aggregators.ema_trust import EMATrustAggregator
from equitas.aggregators.majority_vote import MajorityVoteAggregator
from equitas.aggregators.multiplicative_weights import MultiplicativeWeightsAggregator
from equitas.aggregators.best_single import OracleUpperBoundAggregator
from equitas.aggregators.random_dictator import RandomDictatorAggregator
from equitas.aggregators.supervisor_rerank import SupervisorRerankAggregator
from equitas.aggregators.trimmed_vote import TrimmedVoteAggregator
from equitas.aggregators.registry import create_aggregator
from equitas.config import AggregatorConfig


class TestWeightedVote:
    def test_majority(self):
        assert weighted_vote(["A", "A", "B"], np.array([1, 1, 1])) == "A"

    def test_weighted(self):
        # B has more weight
        assert weighted_vote(["A", "B"], np.array([0.3, 0.7])) == "B"

    def test_tie_alphabetical(self):
        assert weighted_vote(["B", "A"], np.array([1, 1])) == "A"


class TestMajorityVote:
    def test_basic(self):
        agg = MajorityVoteAggregator(3)
        result = agg.select(["A", "A", "B"], round_id=0)
        assert result == "A"

    def test_update_noop(self):
        agg = MajorityVoteAggregator(3)
        agg.update([0.1, 0.5, 0.9], round_id=0)
        assert np.allclose(agg.get_weights(), [1 / 3, 1 / 3, 1 / 3])


class TestMW:
    def test_downweights_bad_agent(self):
        agg = MultiplicativeWeightsAggregator(3, eta=1.0)
        # Agent 2 has high loss repeatedly
        for _ in range(5):
            agg.update([0.1, 0.1, 0.9], round_id=0)
        weights = agg.get_weights()
        assert weights[2] < weights[0]
        assert weights[2] < weights[1]

    def test_reset(self):
        agg = MultiplicativeWeightsAggregator(3, eta=1.0)
        agg.update([0.1, 0.5, 0.9], round_id=0)
        agg.reset()
        assert np.allclose(agg.get_weights(), [1 / 3, 1 / 3, 1 / 3])


class TestEMATrust:
    def test_converges_to_good_agent(self):
        agg = EMATrustAggregator(3, ema_alpha=0.5)
        for _ in range(10):
            agg.update([0.0, 0.5, 1.0], round_id=0)
        weights = agg.get_weights()
        assert weights[0] > weights[1] > weights[2]


class TestTrimmedVote:
    def test_trims_worst(self):
        agg = TrimmedVoteAggregator(5, trim_fraction=0.2)
        # Agent 4 has worst cumulative loss
        for _ in range(5):
            agg.update([0.1, 0.1, 0.1, 0.1, 0.9], round_id=0)
        weights = agg.get_weights()
        # Agent 4 should be trimmed (weight = 0)
        assert weights[4] == 0.0


class TestOracleUpperBound:
    def test_retrospective(self):
        agg = OracleUpperBoundAggregator(3)
        agg.select(["A", "B", "C"], round_id=0)
        agg.update([0.5, 0.1, 0.9], round_id=0)
        agg.select(["B", "A", "C"], round_id=1)
        agg.update([0.5, 0.2, 0.8], round_id=1)
        retro = agg.retrospective_decisions()
        # Agent 1 has lowest cumulative loss (0.3)
        assert retro == ["B", "A"]


class TestConfidenceWeighted:
    def test_downweights_bad_agent(self):
        agg = ConfidenceWeightedVoteAggregator(3)
        # Agent 2 has high loss repeatedly
        for _ in range(10):
            agg.update([0.1, 0.1, 0.9], round_id=0)
        weights = agg.get_weights()
        # Agent 2 should have lower weight (more cumulative loss)
        assert weights[2] < weights[0]
        assert weights[2] < weights[1]

    def test_harmonic_decay_is_forgiving(self):
        """Harmonic decay is more forgiving than MW — weights never reach 0."""
        agg = ConfidenceWeightedVoteAggregator(3)
        for _ in range(50):
            agg.update([0.0, 0.0, 1.0], round_id=0)
        weights = agg.get_weights()
        # Even with 50 rounds of max loss, weight should still be positive
        assert weights[2] > 0.0

    def test_reset(self):
        agg = ConfidenceWeightedVoteAggregator(3)
        agg.update([0.1, 0.5, 0.9], round_id=0)
        agg.reset()
        assert np.allclose(agg.get_weights(), [1 / 3, 1 / 3, 1 / 3])

    def test_selects_good_agent_action(self):
        agg = ConfidenceWeightedVoteAggregator(3)
        for _ in range(10):
            agg.update([0.0, 0.5, 1.0], round_id=0)
        # Agent 0 should dominate voting
        result = agg.select(["A", "B", "C"], round_id=0)
        assert result == "A"


class TestRandomDictator:
    def test_returns_valid_action(self):
        agg = RandomDictatorAggregator(3, seed=42)
        result = agg.select(["A", "B", "C"], round_id=0)
        assert result in ["A", "B", "C"]

    def test_uniform_distribution(self):
        """Over many rounds, random dictator should pick each agent roughly equally."""
        agg = RandomDictatorAggregator(3, seed=0)
        counts = {"A": 0, "B": 0, "C": 0}
        for i in range(3000):
            result = agg.select(["A", "B", "C"], round_id=i)
            counts[result] += 1
        # Each should be roughly 1000 ± 100
        for action, count in counts.items():
            assert 800 < count < 1200, f"{action}: {count}"

    def test_update_noop(self):
        agg = RandomDictatorAggregator(3, seed=0)
        agg.update([0.1, 0.5, 0.9], round_id=0)
        # Weights are always uniform
        assert np.allclose(agg.get_weights(), [1 / 3, 1 / 3, 1 / 3])

    def test_mechanism_meta(self):
        agg = RandomDictatorAggregator(3)
        assert agg.mechanism_meta.is_online is False
        assert agg.mechanism_meta.is_oracle is False


class TestSupervisorRerank:
    def test_follows_best_agent(self):
        agg = SupervisorRerankAggregator(3)
        # Build history: agent 0 is best
        for _ in range(5):
            agg.update([0.1, 0.5, 0.9], round_id=0)
        # Should follow agent 0
        result = agg.select(["A", "B", "C"], round_id=5)
        assert result == "A"

    def test_all_in_on_leader(self):
        """Weights should be 1.0 for leader, 0.0 for others."""
        agg = SupervisorRerankAggregator(3)
        agg.update([0.1, 0.5, 0.9], round_id=0)
        weights = agg.get_weights()
        assert weights[0] == 1.0
        assert weights[1] == 0.0
        assert weights[2] == 0.0

    def test_leader_switch(self):
        """If agent 1 becomes better, supervisor switches."""
        agg = SupervisorRerankAggregator(3)
        # Agent 0 starts best
        agg.update([0.1, 0.5, 0.9], round_id=0)
        assert agg.select(["A", "B", "C"], round_id=1) == "A"
        # Many rounds where agent 1 is best
        for _ in range(20):
            agg.update([0.9, 0.0, 0.9], round_id=0)
        # Now agent 1 should be leader
        assert agg.select(["A", "B", "C"], round_id=21) == "B"

    def test_reset(self):
        agg = SupervisorRerankAggregator(3)
        agg.update([0.1, 0.5, 0.9], round_id=0)
        agg.reset()
        assert np.allclose(agg.get_weights(), [1 / 3, 1 / 3, 1 / 3])

    def test_mechanism_meta(self):
        agg = SupervisorRerankAggregator(3)
        assert agg.mechanism_meta.is_online is True


class TestRegistry:
    def test_create_majority(self):
        cfg = AggregatorConfig(method="majority_vote")
        agg = create_aggregator(cfg, 5)
        assert agg.name == "majority_vote"

    def test_create_mw(self):
        cfg = AggregatorConfig(method="multiplicative_weights", eta=2.0)
        agg = create_aggregator(cfg, 5)
        assert agg.name == "multiplicative_weights"

    def test_create_confidence_weighted(self):
        cfg = AggregatorConfig(method="confidence_weighted")
        agg = create_aggregator(cfg, 5)
        assert agg.name == "confidence_weighted"

    def test_create_random_dictator(self):
        cfg = AggregatorConfig(method="random_dictator")
        agg = create_aggregator(cfg, 5)
        assert agg.name == "random_dictator"

    def test_create_supervisor_rerank(self):
        cfg = AggregatorConfig(method="supervisor_rerank")
        agg = create_aggregator(cfg, 5)
        assert agg.name == "supervisor_rerank"
