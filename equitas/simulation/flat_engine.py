"""
Flat (non-hierarchical) simulation for comparison.
All agents vote directly — single-level aggregation, no leaders/judges/governor.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..adversaries.corruption_schedule import CorruptionSchedule
from ..agents.llm_client import LLMClientWrapper
from ..agents.member import MemberAgent, create_class_members
from ..aggregators.base import BaseAggregator
from ..aggregators.best_single import OracleUpperBoundAggregator  # noqa: F401
from ..aggregators.registry import create_aggregator
from ..config import AggregatorConfig, ExperimentConfig
from ..metrics.welfare import combined_loss
from ..simulation.engine import _make_adversary_factory
from ..types import (
    AdversaryType,
    AgentRecommendation,
    Outcome,
    RoundResult,
    SimulationResult,
)
from ..worlds.base import BaseWorld


async def run_flat_simulation(
    config: ExperimentConfig,
    world: BaseWorld,
    seed: int,
) -> SimulationResult:
    """
    Flat architecture: all N*K members vote directly.
    Single-level aggregation. No leaders, judges, or governor.
    """
    rng = np.random.default_rng(seed)
    llm_client = LLMClientWrapper(config.llm)

    adversary_factory = _make_adversary_factory(
        config.corruption.adversary_type, world, llm_client, config, rng,
    )

    # Create all members (flat pool)
    all_members: List[MemberAgent] = []
    for cls_id in config.committee.class_ids:
        members = create_class_members(
            class_id=cls_id,
            num_members=config.committee.members_per_class,
            corruption_rate=config.corruption.corruption_rate,
            adversary_type=AdversaryType(config.corruption.adversary_type),
            llm_client=llm_client,
            world=world,
            rng=rng,
            adversary_factory=adversary_factory,
        )
        all_members.extend(members)

    total_agents = len(all_members)

    # Create aggregators
    aggregators: Dict[str, BaseAggregator] = {}
    for agg_cfg in config.aggregators:
        aggregators[agg_cfg.method] = create_aggregator(agg_cfg, total_agents)

    # Corruption schedule
    corrupted_ids = [m.agent_id for m in all_members if m.corrupted]
    schedule = CorruptionSchedule(
        onset_round=config.corruption.corruption_onset_round or 0,
        corrupted_agent_ids=corrupted_ids,
    )

    # Generate rounds
    crises = world.generate_rounds(config.world.num_rounds, rng)

    round_results: List[RoundResult] = []
    agg_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []
    weight_history: Dict[str, List[np.ndarray]] = {
        name: [] for name in aggregators
    }

    for round_id, crisis in enumerate(crises):
        schedule.apply_to_agents(all_members, round_id)

        # All members recommend in parallel
        tasks = [m.act(crisis, round_id) for m in all_members]
        recs: List[AgentRecommendation] = await asyncio.gather(*tasks)

        # Oracle
        oracle_id, oracle_outcome = world.best_action(crisis)

        # Each aggregator selects
        action_ids = [r.action_id for r in recs]
        agg_decisions: Dict[str, str] = {}
        agg_outcomes: Dict[str, Outcome] = {}

        for agg_name, agg in aggregators.items():
            chosen = agg.select(action_ids, round_id)
            outcome = world.evaluate_action(crisis, chosen)
            agg_decisions[agg_name] = chosen
            agg_outcomes[agg_name] = outcome

        # Compute losses and update
        for agg_name, agg in aggregators.items():
            agg_cfg = next(
                (c for c in config.aggregators if c.method == agg_name),
                AggregatorConfig(),
            )
            losses = []
            for rec in recs:
                m_outcome = world.evaluate_action(crisis, rec.action_id)
                loss = combined_loss(
                    m_outcome.city_utility,
                    oracle_outcome.city_utility,
                    m_outcome.fairness_jain,
                    alpha=agg_cfg.alpha,
                    beta=agg_cfg.beta,
                )
                losses.append(loss)
            agg.update(losses, round_id)

        # Log
        for agg_name, outcome in agg_outcomes.items():
            agg_rows.append({
                "round_id": round_id,
                "aggregator": agg_name,
                "chosen_action_id": agg_decisions[agg_name],
                "city_utility": outcome.city_utility,
                "unfairness_gap": outcome.unfairness,
                "fairness_jain": outcome.fairness_jain,
                "worst_group_utility": outcome.worst_group_utility,
                "oracle_city_utility": oracle_outcome.city_utility,
                "regret": oracle_outcome.city_utility - outcome.city_utility,
            })
        agg_rows.append({
            "round_id": round_id,
            "aggregator": "oracle",
            "chosen_action_id": oracle_id,
            "city_utility": oracle_outcome.city_utility,
            "unfairness_gap": oracle_outcome.unfairness,
            "fairness_jain": oracle_outcome.fairness_jain,
            "worst_group_utility": oracle_outcome.worst_group_utility,
            "oracle_city_utility": oracle_outcome.city_utility,
            "regret": 0.0,
        })

        for rec in recs:
            agent_rows.append({
                "round_id": round_id,
                "agent_id": rec.agent_id,
                "class_id": rec.class_id,
                "role": "member",
                "corrupted": rec.corrupted,
                "adversary_type": rec.adversary_type.value,
                "recommended_action_id": rec.action_id,
                "oracle_action_id": oracle_id,
            })

        for agg_name, agg in aggregators.items():
            weight_history[agg_name].append(agg.get_weights().copy())

        # Store round result (simplified for flat)
        round_results.append(RoundResult(
            round_id=round_id,
            crisis=crisis,
            oracle_action_id=oracle_id,
            oracle_outcome=oracle_outcome,
            class_member_recs={},
            class_leader_proposals={},
            judge_evaluations=[],
            aggregator_decisions=agg_decisions,
            aggregator_outcomes=agg_outcomes,
        ))

        print(f"  Round {round_id + 1}/{len(crises)}", end="\r")

    # Post-hoc best_single
    for agg_name, agg in aggregators.items():
        if isinstance(agg, OracleUpperBoundAggregator):
            retro = agg.retrospective_decisions()
            bs_rows = [r for r in agg_rows if r["aggregator"] == "oracle_upper_bound"]
            for i, row in enumerate(bs_rows):
                if i < len(retro):
                    crisis = crises[i]
                    outcome = world.evaluate_action(crisis, retro[i])
                    row["chosen_action_id"] = retro[i]
                    row["city_utility"] = outcome.city_utility
                    row["unfairness_gap"] = outcome.unfairness
                    row["fairness_jain"] = outcome.fairness_jain
                    row["worst_group_utility"] = outcome.worst_group_utility
                    row["regret"] = round_results[i].oracle_outcome.city_utility - outcome.city_utility

    print()

    return SimulationResult(
        config=config,
        rounds=round_results,
        aggregator_log=pd.DataFrame(agg_rows),
        agent_log=pd.DataFrame(agent_rows),
        weight_history=weight_history,
    )
