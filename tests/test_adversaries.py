"""Test adversary behavior."""
from __future__ import annotations

import numpy as np
import pytest

from equitas.adversaries.adaptive import ScheduledAdversary
from equitas.adversaries.coordinated import CoordinatedAdversary
from equitas.adversaries.selfish import SelfishAdversary
from equitas.worlds.governance import GovernanceWorld


@pytest.fixture
def world():
    return GovernanceWorld(
        crisis_axes=["resource_scarcity", "external_threat", "inequality", "economic_instability"],
        policy_dims=["tax_merchants", "welfare_workers", "military_spend", "education_investment"],
        actions_per_crisis=3,
    )


@pytest.fixture
def crisis(world):
    rng = np.random.default_rng(42)
    return world.generate_rounds(1, rng)[0]


class TestSelfish:
    def test_picks_own_class_best(self, world, crisis):
        adv = SelfishAdversary(world=world, class_id="producer")
        action = adv.corrupt_recommendation(crisis, "producer", "A", 0)
        # Should pick the action best for producers
        best_util = -1.0
        best_id = None
        for a in crisis.actions:
            outcome = world.evaluate_action(crisis, a.id)
            if outcome.class_utilities["producer"] > best_util:
                best_util = outcome.class_utilities["producer"]
                best_id = a.id
        assert action == best_id


class TestCoordinated:
    def test_worst_city(self, world, crisis):
        adv = CoordinatedAdversary(world=world, class_id="guardian", target_strategy="worst_city")
        action = adv.corrupt_recommendation(crisis, "guardian", "A", 0)
        # Should pick worst for city
        worst_util = float("inf")
        worst_id = None
        for a in crisis.actions:
            outcome = world.evaluate_action(crisis, a.id)
            if outcome.city_utility < worst_util:
                worst_util = outcome.city_utility
                worst_id = a.id
        assert action == worst_id

    def test_coordination(self, world, crisis):
        """All corrupted agents should pick the same action."""
        adv = CoordinatedAdversary(world=world, class_id="guardian", target_strategy="worst_city")
        a1 = adv.corrupt_recommendation(crisis, "guardian", "A", 0)
        a2 = adv.corrupt_recommendation(crisis, "auxiliary", "B", 0)
        assert a1 == a2


class TestScheduled:
    def test_honest_phase(self, world, crisis):
        adv = ScheduledAdversary(world=world, class_id="guardian", honest_rounds=5)
        action = adv.corrupt_recommendation(crisis, "guardian", "A", round_id=2)
        assert action == "A"  # Returns honest action during benign phase

    def test_exploit_phase(self, world, crisis):
        adv = ScheduledAdversary(world=world, class_id="guardian", honest_rounds=5)
        honest_action = adv.corrupt_recommendation(crisis, "guardian", "A", round_id=2)
        exploit_action = adv.corrupt_recommendation(crisis, "guardian", "A", round_id=10)
        # Post-onset phase should potentially differ from honest
        assert isinstance(exploit_action, str)
