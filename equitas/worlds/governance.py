"""Plato-inspired synthetic governance world."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from ..metrics.fairness import jain_fairness_index
from ..types import ActionSpec, Crisis, Outcome
from .base import BaseWorld

# How each class weighs crisis axes (negative = bad for that class)
CLASS_AXIS_WEIGHTS: Dict[str, Dict[str, float]] = {
    "guardian": {
        "resource_scarcity": -0.3,
        "external_threat": -0.4,
        "inequality": -0.9,
        "economic_instability": -0.2,
    },
    "auxiliary": {
        "resource_scarcity": -0.4,
        "external_threat": -1.0,
        "inequality": -0.2,
        "economic_instability": -0.3,
    },
    "producer": {
        "resource_scarcity": -1.0,
        "external_threat": -0.2,
        "inequality": -0.3,
        "economic_instability": -0.8,
    },
}

# How each class weighs policy dimensions (positive = beneficial)
CLASS_POLICY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "guardian": {
        "tax_merchants": 0.2,
        "welfare_workers": 0.5,
        "military_spend": 0.3,
        "education_investment": 1.0,
    },
    "auxiliary": {
        "tax_merchants": 0.1,
        "welfare_workers": 0.2,
        "military_spend": 1.0,
        "education_investment": 0.3,
    },
    "producer": {
        "tax_merchants": -0.7,
        "welfare_workers": 0.8,
        "military_spend": -0.3,
        "education_investment": 0.4,
    },
}

ACTION_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


class GovernanceWorld(BaseWorld):

    def __init__(
        self,
        crisis_axes: List[str],
        policy_dims: List[str],
        actions_per_crisis: int = 3,
        class_ids: List[str] | None = None,
    ) -> None:
        self.crisis_axes = crisis_axes
        self.policy_dims = policy_dims
        self.actions_per_crisis = actions_per_crisis
        self.class_ids = class_ids or ["guardian", "auxiliary", "producer"]

    def generate_rounds(self, num_rounds: int, rng: np.random.Generator) -> List[Crisis]:
        crises = []
        for i in range(num_rounds):
            axes = {ax: float(rng.uniform(0, 1)) for ax in self.crisis_axes}
            actions = []
            for j in range(self.actions_per_crisis):
                policy = {dim: float(rng.uniform(0, 1)) for dim in self.policy_dims}
                label = ACTION_LABELS[j] if j < len(ACTION_LABELS) else f"X{j}"
                desc = ", ".join(f"{d}={v:.2f}" for d, v in policy.items())
                actions.append(ActionSpec(id=label, description=desc, policy=policy))
            axes_desc = ", ".join(f"{a}={v:.2f}" for a, v in axes.items())
            crises.append(Crisis(
                id=i,
                axes=axes,
                description=f"Crisis {i}: {axes_desc}",
                actions=actions,
            ))
        return crises

    def compute_class_utilities(
        self, crisis: Crisis, action: ActionSpec,
    ) -> Dict[str, float]:
        utils = {}
        for cls_id in self.class_ids:
            ax_w = CLASS_AXIS_WEIGHTS.get(cls_id, {})
            pol_w = CLASS_POLICY_WEIGHTS.get(cls_id, {})
            score = 0.0
            for ax, val in crisis.axes.items():
                score += ax_w.get(ax, 0.0) * val
            for dim, val in action.policy.items():
                score += pol_w.get(dim, 0.0) * val
            utils[cls_id] = self._sigmoid(score)
        return utils

    def evaluate_action(self, crisis: Crisis, action_id: str) -> Outcome:
        action = self._find_action(crisis, action_id)
        class_utils = self.compute_class_utilities(crisis, action)
        vals = list(class_utils.values())
        cu = float(np.mean(vals))
        jain = jain_fairness_index(vals)
        return Outcome(
            crisis_id=crisis.id,
            action_id=action_id,
            class_utilities=class_utils,
            city_utility=cu,
            unfairness=max(vals) - min(vals),
            fairness_jain=jain,
            worst_group_utility=min(vals),
        )

    def best_action(self, crisis: Crisis) -> Tuple[str, Outcome]:
        best_outcome = None
        for action in crisis.actions:
            outcome = self.evaluate_action(crisis, action.id)
            if best_outcome is None or outcome.city_utility > best_outcome.city_utility:
                best_outcome = outcome
        assert best_outcome is not None
        return best_outcome.action_id, best_outcome

    def get_action_ids(self, crisis: Crisis) -> List[str]:
        return [a.id for a in crisis.actions]

    def format_round_for_prompt(self, crisis: Crisis) -> str:
        lines = [f"CRISIS: {crisis.description}", "", "Available actions:"]
        for a in crisis.actions:
            lines.append(f"  Action {a.id}: {a.description}")
        return "\n".join(lines)

    def _find_action(self, crisis: Crisis, action_id: str) -> ActionSpec:
        for a in crisis.actions:
            if a.id == action_id:
                return a
        # Fallback to first action
        return crisis.actions[0]

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))
