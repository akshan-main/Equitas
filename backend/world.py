"""
World model for Plato-style city crises and policy actions.

- Generates random crisis scenarios.
- Defines utilities for each social class.
- Evaluates actions in terms of city utility and fairness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import numpy as np

from .config import CLASS_IDS, CRISIS_AXES, POLICY_DIMS, ACTIONS_PER_CRISIS
from .robust_stats import fairness_index_jain


@dataclass
class ActionSpec:
    """A policy action the city can take in a given crisis."""
    id: str             # e.g. "A", "B", "C"
    description: str
    policy: Dict[str, float]  # values in [0, 1] per POLICY_DIMS


@dataclass
class Crisis:
    """A crisis scenario facing the city."""
    id: int
    axes: Dict[str, float]        # values in [0, 1] per CRISIS_AXES
    description: str
    actions: List[ActionSpec]


@dataclass
class Outcome:
    """Outcome metrics for a particular (crisis, action)."""
    crisis_id: int
    action_id: str
    class_utilities: Dict[str, float]
    city_utility: float
    unfairness: float  # max(class_u) - min(class_u)
    fairness_jain: float  # Jain's fairness index in [0, 1]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# Preference weights for each class over crisis axes and policy dims.
# Signs encode who likes what (e.g., merchants dislike high taxes on merchants).
CLASS_AXIS_WEIGHTS: Dict[str, Dict[str, float]] = {
    "philosopher": {
        "resource_scarcity": -0.2,
        "external_threat": -0.3,
        "inequality": -0.8,
        "economic_instability": -0.5,
    },
    "warrior": {
        "resource_scarcity": -0.3,
        "external_threat": -1.0,
        "inequality": -0.2,
        "economic_instability": -0.3,
    },
    "merchant": {
        "resource_scarcity": -0.6,
        "external_threat": -0.4,
        "inequality": -0.1,
        "economic_instability": -1.0,
    },
    "worker": {
        "resource_scarcity": -1.0,
        "external_threat": -0.2,
        "inequality": -0.8,
        "economic_instability": -0.6,
    },
}

CLASS_POLICY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "philosopher": {
        "tax_merchants": 0.1,
        "welfare_workers": 0.5,
        "military_spend": 0.1,
        "education_investment": 1.0,
    },
    "warrior": {
        "tax_merchants": 0.1,
        "welfare_workers": 0.2,
        "military_spend": 1.0,
        "education_investment": 0.2,
    },
    "merchant": {
        "tax_merchants": -1.0,
        "welfare_workers": -0.1,
        "military_spend": 0.2,
        "education_investment": 0.4,
    },
    "worker": {
        "tax_merchants": 0.3,
        "welfare_workers": 1.0,
        "military_spend": 0.0,
        "education_investment": 0.6,
    },
}


def generate_random_crises(
    num_crises: int,
    rng: np.random.Generator,
    actions_per_crisis: int = ACTIONS_PER_CRISIS,
) -> List[Crisis]:
    """
    Generate synthetic crises with random axes and random policy actions.
    """

    crises: List[Crisis] = []

    for cid in range(num_crises):
        axes = {axis: float(rng.uniform(0.0, 1.0)) for axis in CRISIS_AXES}

        desc = (
            "The city faces a crisis.\n"
            f"- Resource scarcity: {axes['resource_scarcity']:.2f}\n"
            f"- External threat: {axes['external_threat']:.2f}\n"
            f"- Inequality: {axes['inequality']:.2f}\n"
            f"- Economic instability: {axes['economic_instability']:.2f}\n"
            "You must choose one policy response."
        )

        actions: List[ActionSpec] = []
        for a_idx in range(actions_per_crisis):
            label_chr = chr(ord("A") + a_idx)
            policy = {dim: float(rng.uniform(0.0, 1.0)) for dim in POLICY_DIMS}
            action_desc = (
                f"Action {label_chr}: "
                f"tax_merchants={policy['tax_merchants']:.2f}, "
                f"welfare_workers={policy['welfare_workers']:.2f}, "
                f"military_spend={policy['military_spend']:.2f}, "
                f"education_investment={policy['education_investment']:.2f}"
            )
            actions.append(
                ActionSpec(
                    id=label_chr,
                    description=action_desc,
                    policy=policy,
                )
            )

        crises.append(
            Crisis(
                id=cid,
                axes=axes,
                description=desc,
                actions=actions,
            )
        )

    return crises


def compute_class_utilities(
    crisis: Crisis,
    action: ActionSpec,
) -> Dict[str, float]:
    """
    Map each class -> utility in [0, 1] given a crisis and action.

    Utility = sigmoid( w_axis·axes + w_policy·policy )
    """

    class_utils: Dict[str, float] = {}
    for c in CLASS_IDS:
        axis_w = CLASS_AXIS_WEIGHTS[c]
        policy_w = CLASS_POLICY_WEIGHTS[c]

        score_axes = sum(axis_w[a] * crisis.axes[a] for a in CRISIS_AXES)
        score_policy = sum(policy_w[d] * action.policy[d] for d in POLICY_DIMS)
        score = score_axes + score_policy

        class_utils[c] = _sigmoid(score)

    return class_utils


def summarize_outcome(
    crisis: Crisis,
    action: ActionSpec,
) -> Outcome:
    """
    Compute full Outcome for a given (crisis, action).
    """

    class_utils = compute_class_utilities(crisis, action)
    city_utility = float(sum(class_utils.values()) / len(class_utils))
    unfairness = float(max(class_utils.values()) - min(class_utils.values()))
    fairness_j = fairness_index_jain(list(class_utils.values()))

    return Outcome(
        crisis_id=crisis.id,
        action_id=action.id,
        class_utilities=class_utils,
        city_utility=city_utility,
        unfairness=unfairness,
        fairness_jain=fairness_j,
    )


def best_action_for_city(crisis: Crisis) -> Tuple[ActionSpec, Outcome]:
    """
    Compute action that maximizes city_utility.
    """
    best_action = None
    best_outcome = None

    for a in crisis.actions:
        o = summarize_outcome(crisis, a)
        if best_outcome is None or o.city_utility > best_outcome.city_utility:
            best_action = a
            best_outcome = o

    assert best_action is not None
    assert best_outcome is not None
    return best_action, best_outcome
