"""
Full-Hierarchy simulation engine.

Each configured aggregator independently controls BOTH levels:
  Level 1 (intra-class): picks its own winner from member votes
  Level 2 (inter-class): picks final action from its own judges

LLM calls are deduplicated: if two aggregators pick the same intra-class
winner, they share the leader LLM call and resulting judge pipeline.

Governor-Only engine (engine.py) uses a fixed TrimmedVote at Level 1.
This engine gives each aggregator its own intra-class instance.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..agents.governor import Governor
from ..agents.judge import JudgeAgent, create_judges
from ..agents.leader import LeaderAgent
from ..agents.llm_client import LLMClientWrapper
from ..agents.member import MemberAgent, create_class_members
from ..aggregators.base import BaseAggregator
from ..aggregators.best_single import OracleUpperBoundAggregator
from ..aggregators.majority_vote import MajorityVoteAggregator
from ..aggregators.registry import create_aggregator
from ..config import AggregatorConfig, ExperimentConfig
from ..metrics.welfare import combined_loss
from ..types import (
    AdversaryType,
    AgentRecommendation,
    Crisis,
    Outcome,
    RoundResult,
    SimulationResult,
)
from ..worlds.base import BaseWorld

# Reuse adversary factory and helpers from Governor-Only engine
from .engine import (
    _collect_all_agents,
    _get_corrupted_ids,
    _make_adversary_factory,
)
from ..adversaries.corruption_schedule import CorruptionSchedule


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FHCommittee:
    """Full-Hierarchy committee: per-aggregator intra-class aggregators."""
    members: Dict[str, List[MemberAgent]]
    leaders: Dict[str, LeaderAgent]
    judges: List[JudgeAgent]
    governor_aggregators: Dict[str, Governor]
    # agg_name -> cls_id -> BaseAggregator  (one per aggregator per class)
    per_agg_intra: Dict[str, Dict[str, BaseAggregator]]
    _aggregator_configs: Dict[str, AggregatorConfig]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _build_fh_committee(
    config: ExperimentConfig,
    world: BaseWorld,
    llm_client: LLMClientWrapper,
    rng: np.random.Generator,
) -> FHCommittee:
    class_ids = config.committee.class_ids
    adv_type = config.corruption.adversary_type
    corruption_target = config.corruption.corruption_target

    adversary_factory = _make_adversary_factory(
        adv_type, world, llm_client, config, rng,
    )

    member_rate = config.corruption.corruption_rate if corruption_target in ("members", "both") else 0.0
    judge_rate = config.corruption.corruption_rate if corruption_target in ("judges", "both") else 0.0

    # Per-aggregator intra-class aggregators
    per_agg_intra: Dict[str, Dict[str, BaseAggregator]] = {}
    agg_configs: Dict[str, AggregatorConfig] = {}
    for agg_cfg in config.aggregators:
        per_agg_intra[agg_cfg.method] = {}
        agg_configs[agg_cfg.method] = agg_cfg
        for cls_id in class_ids:
            per_agg_intra[agg_cfg.method][cls_id] = create_aggregator(
                agg_cfg, num_agents=config.committee.members_per_class,
            )

    # Members
    realization = config.corruption.corruption_realization
    all_members: Dict[str, List[MemberAgent]] = {}
    for cls_id in class_ids:
        all_members[cls_id] = create_class_members(
            class_id=cls_id,
            num_members=config.committee.members_per_class,
            corruption_rate=member_rate,
            adversary_type=AdversaryType(adv_type),
            llm_client=llm_client,
            world=world,
            rng=rng,
            adversary_factory=adversary_factory,
            corruption_realization=realization,
        )

    # Leaders (internal aggregator unused — chosen_action passed explicitly)
    leaders: Dict[str, LeaderAgent] = {}
    for cls_id in class_ids:
        leaders[cls_id] = LeaderAgent(
            agent_id=f"{cls_id}_leader",
            class_id=cls_id,
            intra_class_aggregator=MajorityVoteAggregator(config.committee.members_per_class),
            llm_client=llm_client,
            world=world,
        )

    # Judges
    judges = create_judges(
        num_judges=config.committee.num_judges,
        class_ids=class_ids,
        corruption_rate=judge_rate,
        adversary_type=AdversaryType(adv_type),
        llm_client=llm_client,
        world=world,
        rng=rng,
        sees_rationales=True,
        adversary_factory=adversary_factory,
    )

    # Governors
    governor_aggs: Dict[str, Governor] = {}
    for agg_cfg in config.aggregators:
        agg = create_aggregator(agg_cfg, num_agents=config.committee.num_judges)
        governor_aggs[agg_cfg.method] = Governor(aggregator=agg)

    return FHCommittee(
        members=all_members,
        leaders=leaders,
        judges=judges,
        governor_aggregators=governor_aggs,
        per_agg_intra=per_agg_intra,
        _aggregator_configs=agg_configs,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _proposal_set_key(choices: Dict[str, str]) -> str:
    """Deterministic key for a set of per-class action choices."""
    return "|".join(f"{cls}:{act}" for cls, act in sorted(choices.items()))


def _fh_collect_all_agents(committee: FHCommittee) -> list:
    agents: list = []
    for cls_members in committee.members.values():
        agents.extend(cls_members)
    agents.extend(committee.judges)
    return agents


def _fh_get_corrupted_ids(committee: FHCommittee) -> List[str]:
    ids = []
    for cls_members in committee.members.values():
        for m in cls_members:
            if m.corrupted:
                ids.append(m.agent_id)
    for j in committee.judges:
        if j.corrupted:
            ids.append(j.agent_id)
    return ids


# ---------------------------------------------------------------------------
# Execute one round
# ---------------------------------------------------------------------------

async def _execute_fh_round(
    round_id: int,
    crisis: Crisis,
    committee: FHCommittee,
    world: BaseWorld,
    config: ExperimentConfig,
) -> RoundResult:
    """Execute one round of full-hierarchy governance.

    1. All members vote (shared LLM calls)
    2. Each aggregator independently picks intra-class winners
    3. Deduplicate: run leader LLM for unique (class, action) pairs
    4. Deduplicate: run judge LLM for unique proposal sets
    5. Each governor selects from its own judges
    6. Update weights at both levels per aggregator
    """
    oracle_action_id, oracle_outcome = world.best_action(crisis)

    # --- Members vote (shared) ---
    class_member_recs: Dict[str, List[AgentRecommendation]] = {}
    all_member_tasks = []
    class_order = config.committee.class_ids
    class_sizes = []
    for cls_id in class_order:
        members = committee.members[cls_id]
        class_sizes.append(len(members))
        for m in members:
            all_member_tasks.append(m.act(crisis, round_id))

    all_member_results = await asyncio.gather(*all_member_tasks)
    idx = 0
    for cls_id, size in zip(class_order, class_sizes):
        class_member_recs[cls_id] = list(all_member_results[idx:idx + size])
        idx += size

    # --- Per-aggregator intra-class selection ---
    agg_intra_choices: Dict[str, Dict[str, str]] = {}
    for agg_name, cls_aggs in committee.per_agg_intra.items():
        agg_intra_choices[agg_name] = {}
        for cls_id, agg in cls_aggs.items():
            action_ids = [rec.action_id for rec in class_member_recs[cls_id]]
            chosen = agg.select(action_ids, round_id)
            agg_intra_choices[agg_name][cls_id] = chosen

    # --- Deduplicate leader LLM calls ---
    unique_leader_pairs: set = set()
    for choices in agg_intra_choices.values():
        for cls_id, action_id in choices.items():
            unique_leader_pairs.add((cls_id, action_id))

    leader_tasks = []
    leader_keys: List[Tuple[str, str]] = []
    for cls_id, action_id in sorted(unique_leader_pairs):
        leader = committee.leaders[cls_id]
        member_recs = class_member_recs[cls_id]
        task = leader.act(crisis, member_recs, round_id, chosen_action=action_id)
        leader_tasks.append(task)
        leader_keys.append((cls_id, action_id))

    leader_results = await asyncio.gather(*leader_tasks)
    leader_cache: Dict[Tuple[str, str], AgentRecommendation] = {}
    for key, result in zip(leader_keys, leader_results):
        leader_cache[key] = result

    # --- Per-aggregator proposal sets ---
    agg_pset_keys: Dict[str, str] = {}
    unique_psets: Dict[str, Dict[str, AgentRecommendation]] = {}

    for agg_name, choices in agg_intra_choices.items():
        pkey = _proposal_set_key(choices)
        agg_pset_keys[agg_name] = pkey
        if pkey not in unique_psets:
            proposals = {}
            for cls_id, action_id in choices.items():
                proposals[cls_id] = leader_cache[(cls_id, action_id)]
            unique_psets[pkey] = proposals

    # --- Deduplicate judge LLM calls ---
    all_judge_tasks = []
    all_judge_meta: List[Tuple[str, int]] = []
    for pkey, proposals in sorted(unique_psets.items()):
        for j_idx, judge in enumerate(committee.judges):
            all_judge_tasks.append(judge.act(crisis, proposals, round_id))
            all_judge_meta.append((pkey, j_idx))

    all_judge_results = await asyncio.gather(*all_judge_tasks)
    judge_cache: Dict[str, List[AgentRecommendation]] = {
        pkey: [] for pkey in unique_psets
    }
    for (pkey, _), result in zip(all_judge_meta, all_judge_results):
        judge_cache[pkey].append(result)

    # --- Governor decisions ---
    aggregator_decisions: Dict[str, str] = {}
    aggregator_outcomes: Dict[str, Outcome] = {}

    for agg_name, governor in committee.governor_aggregators.items():
        pkey = agg_pset_keys[agg_name]
        judge_recs = judge_cache[pkey]
        chosen = governor.decide(judge_recs, round_id)
        outcome = world.evaluate_action(crisis, chosen)
        aggregator_decisions[agg_name] = chosen
        aggregator_outcomes[agg_name] = outcome

    # --- Update weights ---
    for agg_name, governor in committee.governor_aggregators.items():
        pkey = agg_pset_keys[agg_name]
        judge_recs = judge_cache[pkey]
        agg_cfg = committee._aggregator_configs.get(agg_name)
        alpha = agg_cfg.alpha if agg_cfg else 1.0
        beta = agg_cfg.beta if agg_cfg else 0.5

        judge_losses = []
        for jrec in judge_recs:
            j_outcome = world.evaluate_action(crisis, jrec.action_id)
            loss = combined_loss(
                j_outcome.city_utility, oracle_outcome.city_utility,
                j_outcome.fairness_jain, alpha=alpha, beta=beta,
            )
            judge_losses.append(loss)
        governor.update(judge_losses, round_id)

    for agg_name, cls_aggs in committee.per_agg_intra.items():
        agg_cfg = committee._aggregator_configs.get(agg_name)
        alpha = agg_cfg.alpha if agg_cfg else 1.0
        beta = agg_cfg.beta if agg_cfg else 0.5

        for cls_id, agg in cls_aggs.items():
            member_losses = []
            for rec in class_member_recs[cls_id]:
                m_outcome = world.evaluate_action(crisis, rec.action_id)
                loss = combined_loss(
                    m_outcome.city_utility, oracle_outcome.city_utility,
                    m_outcome.fairness_jain, alpha=alpha, beta=beta,
                )
                member_losses.append(loss)
            agg.update(member_losses, round_id)

    # --- Weight snapshots ---
    mw_snap = {}
    for agg_name, gov in committee.governor_aggregators.items():
        mw_snap[agg_name] = gov.get_weights().copy()

    # Reference data for backward-compat RoundResult fields
    first_agg = list(agg_intra_choices.keys())[0] if agg_intra_choices else None
    ref_proposals: Dict[str, AgentRecommendation] = {}
    ref_judges: List[AgentRecommendation] = []
    if first_agg:
        for cls_id, action_id in agg_intra_choices[first_agg].items():
            ref_proposals[cls_id] = leader_cache[(cls_id, action_id)]
        ref_judges = judge_cache[agg_pset_keys[first_agg]]

    # Serialize leader cache for recording
    leader_cache_ser: Dict[str, AgentRecommendation] = {}
    for (cls_id, action_id), prop in leader_cache.items():
        leader_cache_ser[f"{cls_id}:{action_id}"] = prop

    return RoundResult(
        round_id=round_id,
        crisis=crisis,
        oracle_action_id=oracle_action_id,
        oracle_outcome=oracle_outcome,
        class_member_recs=class_member_recs,
        class_leader_proposals=ref_proposals,
        judge_evaluations=ref_judges,
        aggregator_decisions=aggregator_decisions,
        aggregator_outcomes=aggregator_outcomes,
        mw_weights_snapshot=mw_snap,
        leader_call_cache=leader_cache_ser,
        judge_call_cache=judge_cache,
        agg_proposal_keys=agg_pset_keys,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_fh_round(
    rr: RoundResult,
    config: ExperimentConfig,
    agg_rows: List[Dict[str, Any]],
    agent_rows: List[Dict[str, Any]],
) -> None:
    # Aggregator log
    for agg_name, outcome in rr.aggregator_outcomes.items():
        agg_rows.append({
            "round_id": rr.round_id,
            "aggregator": agg_name,
            "chosen_action_id": rr.aggregator_decisions[agg_name],
            "city_utility": outcome.city_utility,
            "unfairness_gap": outcome.unfairness,
            "fairness_jain": outcome.fairness_jain,
            "worst_group_utility": outcome.worst_group_utility,
            "oracle_city_utility": rr.oracle_outcome.city_utility,
            "regret": rr.oracle_outcome.city_utility - outcome.city_utility,
        })

    # Oracle
    agg_rows.append({
        "round_id": rr.round_id,
        "aggregator": "oracle",
        "chosen_action_id": rr.oracle_action_id,
        "city_utility": rr.oracle_outcome.city_utility,
        "unfairness_gap": rr.oracle_outcome.unfairness,
        "fairness_jain": rr.oracle_outcome.fairness_jain,
        "worst_group_utility": rr.oracle_outcome.worst_group_utility,
        "oracle_city_utility": rr.oracle_outcome.city_utility,
        "regret": 0.0,
    })

    # Members (shared)
    for cls_id, recs in rr.class_member_recs.items():
        for rec in recs:
            agent_rows.append({
                "round_id": rr.round_id,
                "agent_id": rec.agent_id,
                "class_id": rec.class_id,
                "role": "member",
                "corrupted": rec.corrupted,
                "adversary_type": rec.adversary_type.value,
                "recommended_action_id": rec.action_id,
                "oracle_action_id": rr.oracle_action_id,
            })

    # Judges (per unique proposal set)
    if rr.judge_call_cache and rr.agg_proposal_keys:
        logged_psets: set = set()
        for agg_name, pkey in rr.agg_proposal_keys.items():
            if pkey in logged_psets:
                continue
            logged_psets.add(pkey)
            for jrec in rr.judge_call_cache[pkey]:
                agent_rows.append({
                    "round_id": rr.round_id,
                    "agent_id": jrec.agent_id,
                    "class_id": jrec.class_id,
                    "role": "judge",
                    "corrupted": jrec.corrupted,
                    "adversary_type": jrec.adversary_type.value,
                    "recommended_action_id": jrec.action_id,
                    "oracle_action_id": rr.oracle_action_id,
                    "proposal_set": pkey,
                })


# ---------------------------------------------------------------------------
# Post-hoc oracle
# ---------------------------------------------------------------------------

def _finalize_fh_best_single(
    committee: FHCommittee,
    rounds: List[RoundResult],
    agg_rows: List[Dict[str, Any]],
    world: BaseWorld,
) -> None:
    gov = committee.governor_aggregators.get("oracle_upper_bound")
    if gov is None:
        return
    agg = gov.aggregator
    if not isinstance(agg, OracleUpperBoundAggregator):
        return

    retro = agg.retrospective_decisions()
    ub_rows = [r for r in agg_rows if r["aggregator"] == "oracle_upper_bound"]
    for i, row in enumerate(ub_rows):
        if i < len(retro):
            crisis = rounds[i].crisis
            outcome = world.evaluate_action(crisis, retro[i])
            row["chosen_action_id"] = retro[i]
            row["city_utility"] = outcome.city_utility
            row["unfairness_gap"] = outcome.unfairness
            row["fairness_jain"] = outcome.fairness_jain
            row["worst_group_utility"] = outcome.worst_group_utility
            row["regret"] = rounds[i].oracle_outcome.city_utility - outcome.city_utility


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_fh_simulation(
    config: ExperimentConfig,
    world: BaseWorld,
    seed: int,
    recording_path: Optional[str] = None,
) -> SimulationResult:
    """
    Full-Hierarchy simulation. Each aggregator independently controls
    intra-class selection AND inter-class judge aggregation.
    """
    from .replay_fh import rec_from_agent_rec, save_recording

    rng = np.random.default_rng(seed)
    llm_client = LLMClientWrapper(config.llm)

    committee = _build_fh_committee(config, world, llm_client, rng)
    crises = world.generate_rounds(config.world.num_rounds, rng)

    corrupted_ids = _fh_get_corrupted_ids(committee)
    schedule = CorruptionSchedule(
        onset_round=config.corruption.corruption_onset_round or 0,
        corrupted_agent_ids=corrupted_ids,
    )

    round_results: List[RoundResult] = []
    agg_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []
    weight_history: Dict[str, List[np.ndarray]] = {
        agg_name: [] for agg_name in committee.governor_aggregators
    }
    recording_data: List[Dict[str, Any]] = []

    for round_id, crisis in enumerate(crises):
        all_agents = _fh_collect_all_agents(committee)
        schedule.apply_to_agents(all_agents, round_id)

        print(f"  Round {round_id + 1}/{len(crises)}", end="\r")
        rr = await _execute_fh_round(round_id, crisis, committee, world, config)
        round_results.append(rr)

        _log_fh_round(rr, config, agg_rows, agent_rows)

        # Record (full-hierarchy format)
        if recording_path is not None:
            m_ser: Dict[str, list] = {}
            for cls_id, recs in rr.class_member_recs.items():
                m_ser[cls_id] = [rec_from_agent_rec(r) for r in recs]

            l_cache_ser: Dict[str, dict] = {}
            if rr.leader_call_cache:
                for cache_key, prop in rr.leader_call_cache.items():
                    l_cache_ser[cache_key] = rec_from_agent_rec(prop)

            j_cache_ser: Dict[str, list] = {}
            if rr.judge_call_cache:
                for pset_key, judge_recs in rr.judge_call_cache.items():
                    j_cache_ser[pset_key] = [rec_from_agent_rec(j) for j in judge_recs]

            recording_data.append({
                "round_id": round_id,
                "crisis_id": crisis.id,
                "oracle_action_id": rr.oracle_action_id,
                "member_recs": m_ser,
                "leader_calls": l_cache_ser,
                "judge_calls": j_cache_ser,
                "format": "full_hierarchy",
            })

        for agg_name, gov in committee.governor_aggregators.items():
            weight_history[agg_name].append(gov.get_weights().copy())

    _finalize_fh_best_single(committee, round_results, agg_rows, world)

    if recording_path is not None and recording_data:
        save_recording(recording_data, recording_path)
        print(f"\n  Recording saved to {recording_path}")

    print()

    return SimulationResult(
        config=config,
        rounds=round_results,
        aggregator_log=pd.DataFrame(agg_rows),
        agent_log=pd.DataFrame(agent_rows),
        weight_history=weight_history,
    )
