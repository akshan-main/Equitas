"""Type 1: Selfish bias adversary — maximizes own class utility."""
from __future__ import annotations

from typing import Dict, Optional

from ..types import Crisis
from ..worlds.base import BaseWorld
from .base import BaseAdversary


class SelfishAdversary(BaseAdversary):
    """Picks action maximizing own class utility, ignoring city welfare."""

    def __init__(self, world: BaseWorld, class_id: str) -> None:
        self.world = world
        self.own_class_id = class_id

    def corrupt_recommendation(
        self,
        crisis: Crisis,
        class_id: str,
        honest_action_id: str,
        round_id: int,
        context: Optional[Dict] = None,
    ) -> str:
        best_id = crisis.actions[0].id
        best_util = -float("inf")
        for action in crisis.actions:
            outcome = self.world.evaluate_action(crisis, action.id)
            util = outcome.class_utilities.get(self.own_class_id, 0.0)
            if util > best_util:
                best_util = util
                best_id = action.id
        return best_id
