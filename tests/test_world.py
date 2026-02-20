"""Test governance world model."""
from __future__ import annotations

import numpy as np
import pytest

from equitas.worlds.governance import GovernanceWorld


class TestGovernanceWorld:
    def test_generate_crises(self, governance_world, rng):
        crises = governance_world.generate_rounds(5, rng)
        assert len(crises) == 5
        for c in crises:
            assert len(c.actions) == 3
            assert len(c.axes) == 4

    def test_evaluate_action(self, governance_world, rng):
        crises = governance_world.generate_rounds(1, rng)
        crisis = crises[0]
        outcome = governance_world.evaluate_action(crisis, crisis.actions[0].id)
        assert 0.0 <= outcome.city_utility <= 1.0
        assert 0.0 <= outcome.fairness_jain <= 1.0
        assert outcome.unfairness >= 0.0
        assert outcome.worst_group_utility >= 0.0

    def test_best_action(self, governance_world, rng):
        crises = governance_world.generate_rounds(1, rng)
        crisis = crises[0]
        best_id, best_outcome = governance_world.best_action(crisis)
        assert best_id in [a.id for a in crisis.actions]
        # Best should have highest city utility
        for action in crisis.actions:
            other = governance_world.evaluate_action(crisis, action.id)
            assert best_outcome.city_utility >= other.city_utility - 1e-10

    def test_class_utilities_range(self, governance_world, rng):
        crises = governance_world.generate_rounds(10, rng)
        for crisis in crises:
            for action in crisis.actions:
                utils = governance_world.compute_class_utilities(crisis, action)
                for val in utils.values():
                    assert 0.0 < val < 1.0  # sigmoid output

    def test_format_prompt(self, governance_world, rng):
        crises = governance_world.generate_rounds(1, rng)
        text = governance_world.format_round_for_prompt(crises[0])
        assert "CRISIS" in text
        assert "Action A" in text
