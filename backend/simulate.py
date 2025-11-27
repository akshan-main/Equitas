"""
Core simulation loop:

- Generate crises.
- Create class advisors (some corrupted).
- Run two aggregators in parallel:
    (1) Equal-weight majority vote baseline.
    (2) Multiplicative Weights aggregator.

For each crisis:
- each advisor chooses an action (via LLM).
- each aggregator selects a final action.
- world computes city utility and fairness.
- MW aggregator updates advisor weights based on losses
  (city regret + fairness penalty).

Outputs:
- aggregator_log: per-crisis metrics for each aggregator.
- advisor_log: per-advisor per-crisis losses and corruption status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import asyncio
import numpy as np
import pandas as pd

from .world import (
    Crisis,
    generate_random_crises,
    summarize_outcome,
    best_action_for_city,
)
from .advisors import create_class_advisors
from .mw import MultiplicativeWeightsAggregator, MWConfig, weighted_vote


@dataclass
class SimulationConfig:
    num_crises: int = 20
    epsilon_corruption: float = 0.0
    seed: int = 0
    eta: float = 1.0   # MW learning rate
    alpha: float = 1.0 # weight on city regret in loss
    beta: float = 0.5  # weight on fairness penalty in loss


@dataclass
class SimulationResult:
    config: SimulationConfig
    aggregator_log: pd.DataFrame
    advisor_log: pd.DataFrame


async def run_simulation(config: SimulationConfig) -> SimulationResult:
    rng = np.random.default_rng(config.seed)

    crises: List[Crisis] = generate_random_crises(
        num_crises=config.num_crises,
        rng=rng,
    )

    advisors = create_class_advisors(
        epsilon_corruption=config.epsilon_corruption,
        rng=rng,
    )

    num_advisors = len(advisors)
    mw = MultiplicativeWeightsAggregator(
        num_advisors=num_advisors,
        config=MWConfig(
            eta=config.eta,
            alpha=config.alpha,
            beta=config.beta,
        ),
    )

    agg_rows: List[Dict[str, Any]] = []
    adv_rows: List[Dict[str, Any]] = []

    for crisis in crises:
        # 1) LLM advisors recommend actions (in parallel)
        recommended_actions: List[str] = await _gather_recommendations(advisors, crisis)

        # 2) Equal-weight baseline aggregator
        equal_weights = np.ones(num_advisors, dtype=float) / num_advisors
        equal_action_id = weighted_vote(recommended_actions, equal_weights)

        # 3) Multiplicative Weights aggregator
        mw_action_id = mw.select_action(recommended_actions)

        # 4) Evaluate outcomes
        action_map = {a.id: a for a in crisis.actions}

        equal_outcome = summarize_outcome(crisis, action_map[equal_action_id])
        mw_outcome = summarize_outcome(crisis, action_map[mw_action_id])

        # Best possible city-utility action (for regret)
        best_action, best_outcome = best_action_for_city(crisis)
        best_city_utility = best_outcome.city_utility

        # 5) Compute per-advisor losses (city regret + fairness penalty)
        losses: List[float] = []
        for idx, advisor in enumerate(advisors):
            a_id = recommended_actions[idx]
            adv_outcome = summarize_outcome(crisis, action_map[a_id])

            city_regret = max(0.0, best_city_utility - adv_outcome.city_utility)
            # Fairness penalty based on Jain's index: 0 when perfectly fair, up to 1 when very unfair.
            fairness_penalty = 1.0 - adv_outcome.fairness_jain

            loss = config.alpha * city_regret + config.beta * fairness_penalty
            losses.append(loss)

            adv_rows.append(
                {
                    "crisis_id": crisis.id,
                    "advisor_index": idx,
                    "class_id": advisor.class_id,
                    "corrupted": advisor.corrupted,
                    "recommended_action_id": a_id,
                    "city_regret": city_regret,
                    "fairness_penalty": fairness_penalty,
                    "loss": loss,
                    "epsilon": config.epsilon_corruption,
                    "best_city_utility": best_city_utility,
                    "adv_city_utility": adv_outcome.city_utility,
                    "adv_unfairness_gap": adv_outcome.unfairness,
                    "adv_fairness_jain": adv_outcome.fairness_jain,
                }
            )

        # 6) Update MW weights
        mw.update(losses)

        # 7) Log aggregator outcomes
        for name, outcome, action_id in [
            ("equal", equal_outcome, equal_action_id),
            ("mw", mw_outcome, mw_action_id),
        ]:
            agg_rows.append(
                {
                    "crisis_id": crisis.id,
                    "aggregator": name,
                    "chosen_action_id": action_id,
                    "city_utility": outcome.city_utility,
                    "unfairness_gap": outcome.unfairness,
                    "fairness_jain": outcome.fairness_jain,
                    "epsilon": config.epsilon_corruption,
                }
            )

    agg_df = pd.DataFrame(agg_rows)
    adv_df = pd.DataFrame(adv_rows)

    return SimulationResult(
        config=config,
        aggregator_log=agg_df,
        advisor_log=adv_df,
    )


async def _gather_recommendations(advisors, crisis: Crisis) -> List[str]:
    """
    Helper to query all advisors in parallel.
    """
    tasks = [advisor.recommend_action(crisis) for advisor in advisors]
    return await asyncio.gather(*tasks)
