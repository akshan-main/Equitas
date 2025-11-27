"""
Core simulation loop with two LLM panels:

- Advisors (class-based LLM agents): recommend actions from their perspective.
- Judges (class-based LLM agents): judge which action should be taken.

We run four LLM-based aggregators:
    (1) adv_equal   : equal-weight majority vote over advisors
    (2) adv_mw      : multiplicative weights over advisors
    (3) judge_equal : equal-weight majority vote over judges
    (4) judge_mw    : multiplicative weights over judges

Plus an oracle baseline:
    (5) oracle      : best action according to the world model

Loss for MW updates combines:
- regret in city utility (vs oracle best action)
- fairness penalty (1 - Jain fairness index)

Outputs:
- aggregator_log: per-crisis metrics for each aggregator.
- agent_log     : per-agent per-crisis losses for both panels.
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
from .judges import create_class_judges
from .mw import MultiplicativeWeightsAggregator, MWConfig, weighted_vote


@dataclass
class SimulationConfig:
    # Number of crises per run
    num_crises: int = 20

    # Fraction of corrupted advisors and judges
    advisor_epsilon: float = 0.25
    judge_epsilon: float = 0.0

    # Seed for RNG (crises, corruption assignment, etc.)
    seed: int = 0

    # MW learning rates
    eta_advisors: float = 1.0
    eta_judges: float = 1.0

    # Loss combination weights
    alpha: float = 1.0  # city regret
    beta: float = 0.5   # fairness penalty


@dataclass
class SimulationResult:
    config: SimulationConfig
    aggregator_log: pd.DataFrame
    agent_log: pd.DataFrame


async def run_simulation(config: SimulationConfig) -> SimulationResult:
    rng = np.random.default_rng(config.seed)

    crises: List[Crisis] = generate_random_crises(
        num_crises=config.num_crises,
        rng=rng,
    )

    # LLM advisors and judges (two panels)
    advisors = create_class_advisors(
        epsilon_corruption=config.advisor_epsilon,
        rng=rng,
    )
    judges = create_class_judges(
        epsilon_corruption=config.judge_epsilon,
        rng=rng,
    )

    num_advisors = len(advisors)
    num_judges = len(judges)

    adv_mw = MultiplicativeWeightsAggregator(
        num_advisors,
        MWConfig(
            eta=config.eta_advisors,
            alpha=config.alpha,
            beta=config.beta,
        ),
    )
    judge_mw = MultiplicativeWeightsAggregator(
        num_judges,
        MWConfig(
            eta=config.eta_judges,
            alpha=config.alpha,
            beta=config.beta,
        ),
    )

    agg_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []

    for crisis in crises:
        # Map action id -> spec for this crisis
        action_map = {a.id: a for a in crisis.actions}

        # --- 1) LLM advisors recommend actions (parallel) ---
        adv_recs: List[str] = await _gather_advisor_recs(advisors, crisis)

        # --- 2) LLM judges choose actions (parallel) ---
        judge_recs: List[str] = await _gather_judge_choices(judges, crisis)

        # --- 3) Oracle best action according to world model ---
        best_action, best_outcome = best_action_for_city(crisis)
        best_city_utility = best_outcome.city_utility

        # --- 4) Aggregators over advisors ---
        equal_adv_weights = np.ones(num_advisors, dtype=float) / num_advisors
        adv_equal_action_id = weighted_vote(adv_recs, equal_adv_weights)
        adv_mw_action_id = adv_mw.select_action(adv_recs)

        # --- 5) Aggregators over judges ---
        equal_judge_weights = np.ones(num_judges, dtype=float) / num_judges
        judge_equal_action_id = weighted_vote(judge_recs, equal_judge_weights)
        judge_mw_action_id = judge_mw.select_action(judge_recs)

        # --- 6) Log aggregator outcomes ---
        for name, act_id in [
            ("adv_equal", adv_equal_action_id),
            ("adv_mw", adv_mw_action_id),
            ("judge_equal", judge_equal_action_id),
            ("judge_mw", judge_mw_action_id),
            ("oracle", best_action.id),
        ]:
            outcome = summarize_outcome(crisis, action_map[act_id])
            agg_rows.append(
                {
                    "crisis_id": crisis.id,
                    "aggregator": name,
                    "chosen_action_id": act_id,
                    "city_utility": outcome.city_utility,
                    "unfairness_gap": outcome.unfairness,
                    "fairness_jain": outcome.fairness_jain,
                    "advisor_epsilon": config.advisor_epsilon,
                    "judge_epsilon": config.judge_epsilon,
                }
            )

        # --- 7) Per-advisor losses + MW update ---
        adv_losses: List[float] = []
        for idx, advisor in enumerate(advisors):
            a_id = adv_recs[idx]
            a_out = summarize_outcome(crisis, action_map[a_id])

            city_regret = max(0.0, best_city_utility - a_out.city_utility)
            fairness_penalty = 1.0 - a_out.fairness_jain
            loss = config.alpha * city_regret + config.beta * fairness_penalty

            adv_losses.append(loss)

            agent_rows.append(
                {
                    "crisis_id": crisis.id,
                    "panel": "advisor",
                    "agent_index": idx,
                    "class_id": advisor.class_id,
                    "corrupted": advisor.corrupted,
                    "recommended_action_id": a_id,
                    "city_regret": city_regret,
                    "fairness_penalty": fairness_penalty,
                    "loss": loss,
                    "advisor_epsilon": config.advisor_epsilon,
                    "judge_epsilon": config.judge_epsilon,
                    "adv_city_utility": a_out.city_utility,
                    "adv_unfairness_gap": a_out.unfairness,
                    "adv_fairness_jain": a_out.fairness_jain,
                    "best_city_utility": best_city_utility,
                }
            )

        adv_mw.update(adv_losses)

        # --- 8) Per-judge losses + MW update ---
        judge_losses: List[float] = []
        for idx, judge in enumerate(judges):
            j_id = judge_recs[idx]
            j_out = summarize_outcome(crisis, action_map[j_id])

            city_regret = max(0.0, best_city_utility - j_out.city_utility)
            fairness_penalty = 1.0 - j_out.fairness_jain
            loss = config.alpha * city_regret + config.beta * fairness_penalty

            judge_losses.append(loss)

            agent_rows.append(
                {
                    "crisis_id": crisis.id,
                    "panel": "judge",
                    "agent_index": idx,
                    "class_id": judge.class_id,
                    "corrupted": judge.corrupted,
                    "recommended_action_id": j_id,
                    "city_regret": city_regret,
                    "fairness_penalty": fairness_penalty,
                    "loss": loss,
                    "advisor_epsilon": config.advisor_epsilon,
                    "judge_epsilon": config.judge_epsilon,
                    "adv_city_utility": j_out.city_utility,
                    "adv_unfairness_gap": j_out.unfairness,
                    "adv_fairness_jain": j_out.fairness_jain,
                    "best_city_utility": best_city_utility,
                }
            )

        judge_mw.update(judge_losses)

    agg_df = pd.DataFrame(agg_rows)
    agent_df = pd.DataFrame(agent_rows)

    return SimulationResult(
        config=config,
        aggregator_log=agg_df,
        agent_log=agent_df,
    )


async def _gather_advisor_recs(advisors, crisis: Crisis) -> List[str]:
    tasks = [adv.recommend_action(crisis) for adv in advisors]
    return await asyncio.gather(*tasks)


async def _gather_judge_choices(judges, crisis: Crisis) -> List[str]:
    tasks = [j.choose_action(crisis) for j in judges]
    return await asyncio.gather(*tasks)
