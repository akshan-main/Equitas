"""Type 2: Coordinated adversary — all corrupted agents push same action."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..types import Crisis
from ..worlds.base import BaseWorld
from .base import BaseAdversary


class CoordinatedAdversary(BaseAdversary):
    """
    All corrupted agents coordinate on the same wrong proposal.
    Strategies:
    - "worst_city": pick action minimizing city utility
    - "best_own_class": all push action maximizing target class utility
    - "random": all pick the same random action (deterministic per round)
    """

    def __init__(
        self,
        world: BaseWorld,
        class_id: str,
        target_strategy: str = "worst_city",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.world = world
        self.own_class_id = class_id
        self.target_strategy = target_strategy
        self.rng = rng or np.random.default_rng(0)
        # Cache per-round target to ensure coordination
        self._round_targets: Dict[int, str] = {}

    def corrupt_recommendation(
        self,
        crisis: Crisis,
        class_id: str,
        honest_action_id: str,
        round_id: int,
        context: Optional[Dict] = None,
    ) -> str:
        # Use cached target if already computed for this round (coordination)
        cache_key = (crisis.id, round_id)
        if crisis.id in self._round_targets:
            return self._round_targets[crisis.id]

        if self.target_strategy == "worst_city":
            target = self._worst_city_action(crisis)
        elif self.target_strategy == "best_own_class":
            target = self._best_class_action(crisis)
        else:  # random
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
