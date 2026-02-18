"""
Record/replay for Full-Hierarchy mode.

Recording format stores:
  - member_recs: per class (shared across aggregators)
  - leader_calls: keyed by "cls_id:action_id" (deduped)
  - judge_calls: keyed by proposal_set_key (deduped)

Replay re-runs aggregation at both levels from cached LLM outputs.
Zero API calls — lets you add/change aggregators for free.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..aggregators.base import BaseAggregator
from ..aggregators.best_single import OracleUpperBoundAggregator
from ..aggregators.registry import create_aggregator
from ..config import AggregatorConfig, ExperimentConfig
from ..metrics.welfare import combined_loss
from ..types import AdversaryType, AgentRecommendation, SimulationResult
from ..worlds.base import BaseWorld

# Reuse serialization helpers from Governor-Only replay
from .replay import (
    agent_rec_from_dict,
    rec_from_agent_rec,
    save_recording,
    load_recording,
)


def _proposal_set_key(choices: Dict[str, str]) -> str:
    return "|".join(f"{cls}:{act}" for cls, act in sorted(choices.items()))


async def replay_fh_simulation(
    config: ExperimentConfig,
    world: BaseWorld,
    recording_path: str,
) -> SimulationResult:
    """
    Replay Full-Hierarchy recordings. Each aggregator independently
    picks intra-class winners, then looks up cached leader and judge
    outputs from the recording.
    """
    records = load_recording(recording_path)
    crises = world.generate_rounds(len(records), np.random.default_rng(config.seed))

    # Detect format
    if not records or records[0].get("format") != "full_hierarchy":
        raise ValueError(
            f"Recording at {recording_path} is not full_hierarchy format. "
            "Use replay.replay_simulation for Governor-Only recordings."
        )

    # Create per-aggregator intra-class aggregators
    per_agg_intra: Dict[str, Dict[str, BaseAggregator]] = {}
    for agg_cfg in config.aggregators:
        per_agg_intra[agg_cfg.method] = {}
        for cls_id in config.committee.class_ids:
            per_agg_intra[agg_cfg.method][cls_id] = create_aggregator(
                agg_cfg, num_agents=config.committee.members_per_class,
            )

    # Governor aggregators
    # Determine num_judges from first record
    first_judge_calls = records[0].get("judge_calls", {})
    num_judges = 0
    if first_judge_calls:
        first_pset = next(iter(first_judge_calls.values()))
        num_judges = len(first_pset)

    governor_aggs: Dict[str, BaseAggregator] = {}
    for agg_cfg in config.aggregators:
        governor_aggs[agg_cfg.method] = create_aggregator(
            agg_cfg, num_agents=num_judges,
        )

    agg_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []
    weight_history: Dict[str, List[np.ndarray]] = {
        name: [] for name in governor_aggs
    }

    for i, rec in enumerate(records):
        crisis = crises[i] if i < len(crises) else crises[-1]
        oracle_action_id = rec["oracle_action_id"]
        oracle_outcome = world.evaluate_action(crisis, oracle_action_id)

        # Reconstruct member recs
        member_recs_by_class: Dict[str, List[AgentRecommendation]] = {}
        for cls_id, members in rec["member_recs"].items():
            member_recs_by_class[cls_id] = [agent_rec_from_dict(m) for m in members]

        # Leader call cache
        leader_cache: Dict[str, AgentRecommendation] = {}
        for cache_key, prop_dict in rec.get("leader_calls", {}).items():
            leader_cache[cache_key] = agent_rec_from_dict(prop_dict)

        # Judge call cache
        judge_cache: Dict[str, List[AgentRecommendation]] = {}
        for pset_key, judges in rec.get("judge_calls", {}).items():
            judge_cache[pset_key] = [agent_rec_from_dict(j) for j in judges]

        # Per-aggregator pipeline
        for agg_name, cls_aggs in per_agg_intra.items():
            agg_cfg = next(
                (c for c in config.aggregators if c.method == agg_name),
                AggregatorConfig(),
            )
            alpha = agg_cfg.alpha if agg_cfg else 1.0
            beta = agg_cfg.beta if agg_cfg else 0.5

            # Intra-class selection
            choices: Dict[str, str] = {}
            for cls_id, agg in cls_aggs.items():
                action_ids = [r.action_id for r in member_recs_by_class[cls_id]]
                chosen = agg.select(action_ids, i)
                choices[cls_id] = chosen

            # Look up proposal set
            pset_key = _proposal_set_key(choices)

            if pset_key not in judge_cache:
                # This aggregator's selection wasn't seen during recording —
                # skip (shouldn't happen if recording is complete)
                print(f"  WARNING: proposal set {pset_key} not in recording, "
                      f"skipping {agg_name} round {i}")
                continue

            judge_recs = judge_cache[pset_key]
            judge_action_ids = [j.action_id for j in judge_recs]

            # Governor
            chosen = governor_aggs[agg_name].select(judge_action_ids, i)
            outcome = world.evaluate_action(crisis, chosen)

            agg_rows.append({
                "round_id": i,
                "aggregator": agg_name,
                "chosen_action_id": chosen,
                "city_utility": outcome.city_utility,
                "unfairness_gap": outcome.unfairness,
                "fairness_jain": outcome.fairness_jain,
                "worst_group_utility": outcome.worst_group_utility,
                "oracle_city_utility": oracle_outcome.city_utility,
                "regret": oracle_outcome.city_utility - outcome.city_utility,
            })

            # Update governor
            judge_losses = []
            for jrec in judge_recs:
                j_out = world.evaluate_action(crisis, jrec.action_id)
                loss = combined_loss(
                    j_out.city_utility, oracle_outcome.city_utility,
                    j_out.fairness_jain, alpha=alpha, beta=beta,
                )
                judge_losses.append(loss)
            governor_aggs[agg_name].update(judge_losses, i)

            # Update intra-class
            for cls_id, agg in cls_aggs.items():
                member_losses = []
                for mrec in member_recs_by_class[cls_id]:
                    m_out = world.evaluate_action(crisis, mrec.action_id)
                    loss = combined_loss(
                        m_out.city_utility, oracle_outcome.city_utility,
                        m_out.fairness_jain, alpha=alpha, beta=beta,
                    )
                    member_losses.append(loss)
                agg.update(member_losses, i)

        # Oracle row
        agg_rows.append({
            "round_id": i,
            "aggregator": "oracle",
            "chosen_action_id": oracle_action_id,
            "city_utility": oracle_outcome.city_utility,
            "unfairness_gap": oracle_outcome.unfairness,
            "fairness_jain": oracle_outcome.fairness_jain,
            "worst_group_utility": oracle_outcome.worst_group_utility,
            "oracle_city_utility": oracle_outcome.city_utility,
            "regret": 0.0,
        })

        # Weight history
        for agg_name, agg in governor_aggs.items():
            weight_history[agg_name].append(agg.get_weights().copy())

    # Post-hoc oracle_upper_bound
    for agg_name, agg in governor_aggs.items():
        if isinstance(agg, OracleUpperBoundAggregator):
            retro = agg.retrospective_decisions()
            ub_rows = [r for r in agg_rows if r["aggregator"] == "oracle_upper_bound"]
            for j, row in enumerate(ub_rows):
                if j < len(retro):
                    crisis = crises[j] if j < len(crises) else crises[-1]
                    outcome = world.evaluate_action(crisis, retro[j])
                    row["chosen_action_id"] = retro[j]
                    row["city_utility"] = outcome.city_utility
                    row["unfairness_gap"] = outcome.unfairness
                    row["fairness_jain"] = outcome.fairness_jain
                    row["worst_group_utility"] = outcome.worst_group_utility

    return SimulationResult(
        config=config,
        rounds=[],
        aggregator_log=pd.DataFrame(agg_rows),
        agent_log=pd.DataFrame(agent_rows),
        weight_history=weight_history,
    )
