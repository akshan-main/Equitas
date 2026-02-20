"""Type 3: Scheduled (delayed-onset) adversary — Huber-compliant contamination schedule."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..types import Crisis
from ..worlds.base import BaseWorld
from .base import BaseAdversary


class ScheduledAdversary(BaseAdversary):
    """
    Delayed-onset contamination (Huber-compliant).

    Phase 1: benign outputs for first honest_rounds rounds.
    Phase 2: switches to a fixed contaminating distribution Q
             (worst_city or best_own_class strategy).

    This is a non-stationary contamination schedule but does NOT
    react to the learner's state — it is purely round-based.
    """

    def __init__(
        self,
        world: BaseWorld,
        class_id: str,
        honest_rounds: int = 10,
        exploit_strategy: str = "worst_city",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.world = world
        self.own_class_id = class_id
        self.honest_rounds = honest_rounds
        self.exploit_strategy = exploit_strategy
        self.rng = rng or np.random.default_rng(0)
        self._round_targets: Dict[int, str] = {}

    def corrupt_recommendation(
        self,
        crisis: Crisis,
        class_id: str,
        honest_action_id: str,
        round_id: int,
        context: Optional[Dict] = None,
    ) -> str:
        # Trust-building phase: act honestly
        if round_id < self.honest_rounds:
            return honest_action_id

        # Exploitation phase: coordinate on worst action
        if crisis.id in self._round_targets:
            return self._round_targets[crisis.id]

        if self.exploit_strategy == "worst_city":
            target = self._worst_city_action(crisis)
        elif self.exploit_strategy == "best_own_class":
            target = self._best_class_action(crisis)
        else:
            target = self.rng.choice([a.id for a in crisis.actions])

        self._round_targets[crisis.id] = target
        return target

    def _worst_city_action(self, crisis: Crisis) -> str:
        worst_id = crisis.actions[0].id
        worst_util = float("inf")
        for action in crisis.actions:
            outcome = self.world.evaluate_action(crisis, action.id)
            if outcome.city_utility < worst_util:
                worst_util = outcome.city_utility
                worst_id = action.id
        return worst_id

    def _best_class_action(self, crisis: Crisis) -> str:
        best_id = crisis.actions[0].id
        best_util = -float("inf")
        for action in crisis.actions:
            outcome = self.world.evaluate_action(crisis, action.id)
            util = outcome.class_utilities.get(self.own_class_id, 0.0)
            if util > best_util:
                best_util = util
                best_id = action.id
        return best_id

    def reset(self) -> None:
        self._round_targets = {}
