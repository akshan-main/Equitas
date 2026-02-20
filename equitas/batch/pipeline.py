"""
3-stage OpenAI Batch API pipeline for corruption sweeps.

Stage 1: Submit all member prompts (honest + LLM-corrupted)
Stage 2: Using member results, submit all leader prompts
Stage 3: Using leader results, submit all judge prompts

Algorithmic corrupted agents skip the LLM entirely — their actions
are computed from the world model.

Output: Standard JSONL recording files identical to real-time mode.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai

from ..adversaries.adaptive import ScheduledAdversary
from ..adversaries.coordinated import CoordinatedAdversary
from ..adversaries.deceptive import DeceptiveAdversary
from ..adversaries.selfish import SelfishAdversary
from ..agents.parsing import extract_rationale, parse_action_id
from ..agents.prompts import (
    judge_honest_prompt,
    leader_prompt,
    member_coordinated_prompt,
    member_deceptive_prompt,
    member_honest_prompt,
    member_selfish_prompt,
)
from ..config import ExperimentConfig, save_config
from ..metrics.welfare import combined_loss
from ..simulation.replay import record_round, save_recording
from ..types import AdversaryType, Crisis
from ..utils import derive_seed, ensure_dir
from ..worlds.base import BaseWorld
from ..worlds.governance import GovernanceWorld


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConditionSpec:
    """One (rate, adv_type, run) combo."""
    idx: int
    rate: float
    adv_type: str
    run_idx: int
    seed: int
    crises: List[Crisis] = field(default_factory=list)
    # Per-class corruption info
    corrupt_member_mask: Dict[str, np.ndarray] = field(default_factory=dict)
    corrupt_judge_mask: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class MemberResult:
    """Parsed result for one member in one round."""
    agent_id: str
    class_id: str
    action_id: str
    rationale: str
    corrupted: bool
    adversary_type: str


@dataclass
class LeaderResult:
    """Parsed result for one leader in one round."""
    agent_id: str
    class_id: str
    action_id: str
    rationale: str


@dataclass
class JudgeResult:
    """Parsed result for one judge in one round."""
    agent_id: str
    class_id: str
    action_id: str
    rationale: str
    corrupted: bool
    adversary_type: str


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class BatchSweepPipeline:
    """Orchestrates the 3-stage batch sweep."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.world = self._create_world()
        self.client = openai.OpenAI(
            api_key=os.environ.get(config.llm.api_key_env, ""),
        )

        # Pre-compute all conditions
        self.conditions: List[ConditionSpec] = []
        self._prepare_conditions()

        # Results storage
        # member/judge keyed by (cond_idx, round_id)
        # leader keyed by (cond_idx, round_id, cls_id)
        self.member_results: Dict[Tuple[int, int], Dict[str, List[MemberResult]]] = {}
        self.leader_results: Dict[Tuple[int, int, str], LeaderResult] = {}
        self.leader_selected_actions: Dict[Tuple[int, int, str], str] = {}
        self.judge_results: Dict[Tuple[int, int], List[JudgeResult]] = {}

        # Output directory
        self.out_dir = ensure_dir(config.output_dir)
        self.batch_dir = ensure_dir(os.path.join(str(self.out_dir), "batch_staging"))

    def _create_world(self) -> BaseWorld:
        if self.config.environment == "gsm8k":
            from ..worlds.gsm8k import GSM8KWorld
            return GSM8KWorld(
                data_path=self.config.gsm8k_data_path,
                max_examples=self.config.gsm8k_max_examples,
            )
        return GovernanceWorld(
            crisis_axes=self.config.world.crisis_axes,
            policy_dims=self.config.world.policy_dims,
            actions_per_crisis=self.config.world.actions_per_crisis,
            class_ids=self.config.committee.class_ids,
        )

    def _prepare_conditions(self) -> None:
        """Pre-compute crises and corruption masks for every condition."""
        idx = 0
        class_ids = self.config.committee.class_ids
        n_members = self.config.committee.members_per_class
        n_judges = self.config.committee.num_judges
        corruption_target = self.config.corruption.corruption_target

        for rate in self.config.corruption_rates:
            for adv_type in self.config.adversary_types:
                for run_idx in range(self.config.num_runs):
                    seed = derive_seed(self.config.seed, rate, adv_type, run_idx)
                    rng = np.random.default_rng(seed)

                    # Generate crises
                    crises = self.world.generate_rounds(
                        self.config.world.num_rounds, rng,
                    )

                    # Corruption masks
                    member_rate = rate if corruption_target in ("members", "both") else 0.0
                    judge_rate = rate if corruption_target in ("judges", "both") else 0.0

                    corrupt_members: Dict[str, np.ndarray] = {}
                    for cls_id in class_ids:
                        n_corrupt = int(n_members * member_rate)
                        mask = np.zeros(n_members, dtype=bool)
                        if n_corrupt > 0:
                            corrupt_idx = rng.choice(n_members, size=n_corrupt, replace=False)
                            mask[corrupt_idx] = True
                        corrupt_members[cls_id] = mask

                    judge_mask = np.zeros(n_judges, dtype=bool)
                    n_corrupt_judges = int(n_judges * judge_rate)
                    if n_corrupt_judges > 0:
                        corrupt_j_idx = rng.choice(n_judges, size=n_corrupt_judges, replace=False)
                        judge_mask[corrupt_j_idx] = True

                    self.conditions.append(ConditionSpec(
                        idx=idx,
                        rate=rate,
                        adv_type=adv_type,
                        run_idx=run_idx,
                        seed=seed,
                        crises=crises,
                        corrupt_member_mask=corrupt_members,
                        corrupt_judge_mask=judge_mask,
                    ))
                    idx += 1

    # -------------------------------------------------------------------
    # Adversary helpers
    # -------------------------------------------------------------------

    def _make_adversary(
        self, adv_type: str, class_id: str, cond: ConditionSpec,
    ) -> Any:
        adv = AdversaryType(adv_type)
        cfg = self.config.corruption
        rng = np.random.default_rng(cond.seed)
        if adv == AdversaryType.SELFISH:
            return SelfishAdversary(world=self.world, class_id=class_id)
        elif adv == AdversaryType.COORDINATED:
            return CoordinatedAdversary(
                world=self.world, class_id=class_id,
                target_strategy=cfg.coordinated_target, rng=rng,
            )
        elif adv == AdversaryType.SCHEDULED:
            return ScheduledAdversary(
                world=self.world, class_id=class_id,
                honest_rounds=cfg.scheduled_honest_rounds, rng=rng,
            )
        elif adv == AdversaryType.DECEPTIVE:
            return DeceptiveAdversary(
                world=self.world, class_id=class_id,
                strength=cfg.deceptive_strength, rng=rng,
            )
        return SelfishAdversary(world=self.world, class_id=class_id)

    def _member_needs_llm(
        self, cond: ConditionSpec, cls_id: str, member_idx: int,
        round_id: int,
    ) -> bool:
        """Does this member need an LLM call?"""
        is_corrupt = bool(cond.corrupt_member_mask[cls_id][member_idx])
        if not is_corrupt:
            return True  # honest members always need LLM

        realization = self.config.corruption.corruption_realization
        adv_type = AdversaryType(cond.adv_type)

        if realization == "llm":
            return True  # LLM-realized corruption always calls LLM

        # Algorithmic realization
        if adv_type == AdversaryType.DECEPTIVE:
            return True  # deceptive always uses LLM
        if adv_type == AdversaryType.SCHEDULED:
            honest_rounds = self.config.corruption.scheduled_honest_rounds
            if round_id < honest_rounds:
                return True  # honest phase needs LLM
        return False  # selfish/coordinated/scheduled-exploit = no LLM

    def _compute_algorithmic_member(
        self, cond: ConditionSpec, cls_id: str, member_idx: int,
        crisis: Crisis, round_id: int,
    ) -> MemberResult:
        """Compute corrupted member action without LLM."""
        adv = self._make_adversary(cond.adv_type, cls_id, cond)
        valid_ids = [a.id for a in crisis.actions]
        action_id = adv.corrupt_recommendation(
            crisis=crisis, class_id=cls_id,
            honest_action_id=valid_ids[0], round_id=round_id,
        )
        rationale = ""
        if hasattr(adv, "corrupt_rationale"):
            r = adv.corrupt_rationale(crisis, action_id, cls_id)
            if r:
                rationale = r
        return MemberResult(
            agent_id=f"{cls_id}_member_{member_idx}",
            class_id=cls_id,
            action_id=action_id,
            rationale=rationale,
            corrupted=True,
            adversary_type=cond.adv_type,
        )

    def _member_prompt(
        self, cond: ConditionSpec, cls_id: str, member_idx: int,
        crisis: Crisis, round_id: int,
    ) -> Tuple[str, str]:
        """Generate the (system, user) prompt for a member."""
        crisis_text = self.world.format_round_for_prompt(crisis)
        is_corrupt = bool(cond.corrupt_member_mask[cls_id][member_idx])
        adv_type = AdversaryType(cond.adv_type)
        realization = self.config.corruption.corruption_realization

        if not is_corrupt:
            return member_honest_prompt(cls_id, crisis_text)

        # Corrupted + needs LLM
        if realization == "llm":
            # LLM-realized corruption
            adv = self._make_adversary(cond.adv_type, cls_id, cond)
            valid_ids = [a.id for a in crisis.actions]
            target_action = adv.corrupt_recommendation(
                crisis=crisis, class_id=cls_id,
                honest_action_id=valid_ids[0], round_id=round_id,
            )
            if adv_type == AdversaryType.SELFISH:
                return member_selfish_prompt(cls_id, crisis_text)
            elif adv_type == AdversaryType.COORDINATED:
                return member_coordinated_prompt(cls_id, crisis_text, target_action)
            elif adv_type == AdversaryType.SCHEDULED:
                honest_rounds = self.config.corruption.scheduled_honest_rounds
                if round_id < honest_rounds:
                    return member_honest_prompt(cls_id, crisis_text)
                else:
                    return member_selfish_prompt(cls_id, crisis_text)
            elif adv_type == AdversaryType.DECEPTIVE:
                strength = self.config.corruption.deceptive_strength
                return member_deceptive_prompt(cls_id, crisis_text, target_action, strength)

        # Algorithmic realization but needs LLM (deceptive or scheduled-honest)
        if adv_type == AdversaryType.DECEPTIVE:
            adv = self._make_adversary(cond.adv_type, cls_id, cond)
            valid_ids = [a.id for a in crisis.actions]
            target = adv.corrupt_recommendation(
                crisis=crisis, class_id=cls_id,
                honest_action_id=valid_ids[0], round_id=round_id,
            )
            strength = self.config.corruption.deceptive_strength
            return member_deceptive_prompt(cls_id, crisis_text, target, strength)

        if adv_type == AdversaryType.SCHEDULED:
            return member_honest_prompt(cls_id, crisis_text)

        return member_honest_prompt(cls_id, crisis_text)

    # -------------------------------------------------------------------
    # Stage 1: Members
    # -------------------------------------------------------------------

    def stage1_prepare(self) -> str:
        """Write member batch JSONL. Returns file path."""
        path = os.path.join(str(self.batch_dir), "stage1_members.jsonl")
        count = 0

        # Pre-compute algorithmic members
        for cond in self.conditions:
            for round_id, crisis in enumerate(cond.crises):
                key = (cond.idx, round_id)
                if key not in self.member_results:
                    self.member_results[key] = {
                        cls: [] for cls in self.config.committee.class_ids
                    }
                for cls_id in self.config.committee.class_ids:
                    for m_idx in range(self.config.committee.members_per_class):
                        if not self._member_needs_llm(cond, cls_id, m_idx, round_id):
                            result = self._compute_algorithmic_member(
                                cond, cls_id, m_idx, crisis, round_id,
                            )
                            self.member_results[key][cls_id].append(result)

        # Write LLM-needing member prompts
        with open(path, "w") as f:
            for cond in self.conditions:
                for round_id, crisis in enumerate(cond.crises):
                    for cls_id in self.config.committee.class_ids:
                        for m_idx in range(self.config.committee.members_per_class):
                            if self._member_needs_llm(cond, cls_id, m_idx, round_id):
                                sys_msg, usr_msg = self._member_prompt(
                                    cond, cls_id, m_idx, crisis, round_id,
                                )
                                custom_id = f"{cond.idx}_{round_id}_member_{cls_id}_{m_idx}"
                                req = {
                                    "custom_id": custom_id,
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {
                                        "model": self.config.llm.model,
                                        "messages": [
                                            {"role": "system", "content": sys_msg},
                                            {"role": "user", "content": usr_msg},
                                        ],
                                        "temperature": self.config.llm.temperature,
                                        "max_tokens": self.config.llm.max_tokens,
                                    },
                                }
                                f.write(json.dumps(req) + "\n")
                                count += 1

        print(f"Stage 1: {count} member prompts written to {path}")
        return path

    def stage1_process(self, results_path: str) -> None:
        """Parse batch results and populate member_results."""
        responses = self._load_batch_results(results_path)

        for custom_id, content in responses.items():
            parts = custom_id.split("_")
            # Format: {cond_idx}_{round_id}_member_{cls_id}_{m_idx}
            cond_idx = int(parts[0])
            round_id = int(parts[1])
            cls_id = parts[3]
            m_idx = int(parts[4])

            cond = self.conditions[cond_idx]
            crisis = cond.crises[round_id]
            valid_ids = [a.id for a in crisis.actions]

            action_id = parse_action_id(content, valid_ids)
            rationale = extract_rationale(content)

            is_corrupt = bool(cond.corrupt_member_mask[cls_id][m_idx])

            result = MemberResult(
                agent_id=f"{cls_id}_member_{m_idx}",
                class_id=cls_id,
                action_id=action_id,
                rationale=rationale,
                corrupted=is_corrupt,
                adversary_type=cond.adv_type if is_corrupt else "none",
            )

            key = (cond_idx, round_id)
            if key not in self.member_results:
                self.member_results[key] = {
                    c: [] for c in self.config.committee.class_ids
                }
            self.member_results[key][cls_id].append(result)

        # Sort members by index within each class to ensure consistent ordering
        for key in self.member_results:
            for cls_id in self.member_results[key]:
                self.member_results[key][cls_id].sort(
                    key=lambda r: int(r.agent_id.split("_")[-1]),
                )

        print(f"Stage 1: processed {len(responses)} member results")

    # -------------------------------------------------------------------
    # Stage 2: Leaders
    # -------------------------------------------------------------------

    def stage2_prepare(self) -> str:
        """Using member results, write leader batch JSONL."""
        from ..aggregators.trimmed_vote import TrimmedVoteAggregator

        path = os.path.join(str(self.batch_dir), "stage2_leaders.jsonl")
        count = 0
        self.leader_selected_actions = {}
        # Mirrors engine behavior: member loss weighting is sourced from
        # the first configured governor aggregator.
        agg_cfg = self.config.aggregators[0] if self.config.aggregators else None
        alpha = agg_cfg.alpha if agg_cfg is not None else 1.0
        beta = agg_cfg.beta if agg_cfg is not None else 0.5

        with open(path, "w") as f:
            for cond in self.conditions:
                # Intra-class aggregators are stateful across rounds.
                intra_aggs = {
                    cls_id: TrimmedVoteAggregator(
                        num_agents=self.config.committee.members_per_class,
                        trim_fraction=0.2,
                    )
                    for cls_id in self.config.committee.class_ids
                }
                for round_id, crisis in enumerate(cond.crises):
                    key = (cond.idx, round_id)
                    oracle_action_id, oracle_outcome = self.world.best_action(crisis)
                    for cls_id in self.config.committee.class_ids:
                        members = self.member_results[key][cls_id]
                        action_ids = [m.action_id for m in members]

                        # Intra-class aggregation (stateful across rounds)
                        agg = intra_aggs[cls_id]
                        chosen_action = agg.select(action_ids, round_id)
                        self.leader_selected_actions[(cond.idx, round_id, cls_id)] = chosen_action

                        # Member summary
                        vote_counts = Counter(action_ids)
                        summary_lines = []
                        for aid, cnt in vote_counts.most_common():
                            summary_lines.append(f"Action {aid}: {cnt} votes")
                        member_summary = "\n".join(summary_lines)

                        crisis_text = self.world.format_round_for_prompt(crisis)
                        sys_msg, usr_msg = leader_prompt(
                            cls_id, crisis_text, member_summary, chosen_action,
                        )

                        custom_id = f"{cond.idx}_{round_id}_leader_{cls_id}"
                        req = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": self.config.llm.model,
                                "messages": [
                                    {"role": "system", "content": sys_msg},
                                    {"role": "user", "content": usr_msg},
                                ],
                                "temperature": self.config.llm.temperature,
                                "max_tokens": self.config.llm.max_tokens,
                            },
                        }
                        f.write(json.dumps(req) + "\n")
                        count += 1

                        # Update intra-class aggregator exactly like engine.py
                        member_losses = []
                        for m in members:
                            m_outcome = self.world.evaluate_action(crisis, m.action_id)
                            loss = combined_loss(
                                m_outcome.city_utility,
                                oracle_outcome.city_utility,
                                m_outcome.fairness_jain,
                                alpha=alpha,
                                beta=beta,
                            )
                            member_losses.append(loss)
                        agg.update(member_losses, round_id)

        print(f"Stage 2: {count} leader prompts written to {path}")
        return path

    def stage2_process(self, results_path: str) -> None:
        """Parse leader batch results."""
        responses = self._load_batch_results(results_path)

        for custom_id, content in responses.items():
            parts = custom_id.split("_")
            cond_idx = int(parts[0])
            round_id = int(parts[1])
            cls_id = parts[3]
            chosen_action = self.leader_selected_actions.get((cond_idx, round_id, cls_id))
            if chosen_action is None:
                raise KeyError(
                    "Leader selected action missing; stage2_prepare must run before "
                    f"stage2_process for cond={cond_idx}, round={round_id}, class={cls_id}",
                )

            self.leader_results[(cond_idx, round_id, cls_id)] = LeaderResult(
                agent_id=f"{cls_id}_leader",
                class_id=cls_id,
                action_id=chosen_action,
                rationale=content,
            )

        print(f"Stage 2: processed {len(responses)} leader results")

    # -------------------------------------------------------------------
    # Stage 3: Judges
    # -------------------------------------------------------------------

    def stage3_prepare(self) -> str:
        """Using leader results, write judge batch JSONL."""
        path = os.path.join(str(self.batch_dir), "stage3_judges.jsonl")
        count = 0
        class_ids = self.config.committee.class_ids

        # Pre-compute algorithmically corrupted judges
        for cond in self.conditions:
            for round_id, crisis in enumerate(cond.crises):
                key = (cond.idx, round_id)
                if key not in self.judge_results:
                    self.judge_results[key] = []

                for j_idx in range(self.config.committee.num_judges):
                    is_corrupt = bool(cond.corrupt_judge_mask[j_idx])
                    if is_corrupt and cond.rate > 0:
                        cls_id = class_ids[j_idx % len(class_ids)]
                        adv = self._make_adversary(cond.adv_type, cls_id, cond)
                        valid_ids = [a.id for a in crisis.actions]
                        action_id = adv.corrupt_recommendation(
                            crisis=crisis, class_id=cls_id,
                            honest_action_id=valid_ids[0], round_id=round_id,
                        )
                        rationale = ""
                        if hasattr(adv, "corrupt_rationale"):
                            r = adv.corrupt_rationale(crisis, action_id, cls_id)
                            if r:
                                rationale = r
                        if not rationale:
                            rationale = f"Selected Action {action_id} as optimal."

                        self.judge_results[key].append(JudgeResult(
                            agent_id=f"judge_{j_idx}",
                            class_id=cls_id,
                            action_id=action_id,
                            rationale=rationale,
                            corrupted=True,
                            adversary_type=cond.adv_type,
                        ))

        # Write honest judge prompts
        with open(path, "w") as f:
            for cond in self.conditions:
                for round_id, crisis in enumerate(cond.crises):
                    crisis_text = self.world.format_round_for_prompt(crisis)

                    # Build proposals from leaders
                    proposals_text: Dict[str, str] = {}
                    for cls_id in class_ids:
                        lr_key = (cond.idx, round_id, cls_id)
                        lr = self.leader_results.get(lr_key)
                        if lr:
                            proposals_text[cls_id] = lr.rationale
                        else:
                            proposals_text[cls_id] = "No proposal available."

                    for j_idx in range(self.config.committee.num_judges):
                        is_corrupt = bool(cond.corrupt_judge_mask[j_idx])
                        if is_corrupt and cond.rate > 0:
                            continue  # already computed algorithmically

                        sys_msg, usr_msg = judge_honest_prompt(
                            crisis_text, proposals_text, sees_rationales=True,
                        )
                        custom_id = f"{cond.idx}_{round_id}_judge_{j_idx}"
                        req = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": self.config.llm.model,
                                "messages": [
                                    {"role": "system", "content": sys_msg},
                                    {"role": "user", "content": usr_msg},
                                ],
                                "temperature": self.config.llm.temperature,
                                "max_tokens": self.config.llm.max_tokens,
                            },
                        }
                        f.write(json.dumps(req) + "\n")
                        count += 1

        print(f"Stage 3: {count} judge prompts written to {path}")
        return path

    def stage3_process(self, results_path: str) -> None:
        """Parse judge batch results."""
        responses = self._load_batch_results(results_path)
        class_ids = self.config.committee.class_ids

        for custom_id, content in responses.items():
            parts = custom_id.split("_")
            cond_idx = int(parts[0])
            round_id = int(parts[1])
            j_idx = int(parts[3])

            cond = self.conditions[cond_idx]
            crisis = cond.crises[round_id]
            valid_ids = [a.id for a in crisis.actions]
            cls_id = class_ids[j_idx % len(class_ids)]

            action_id = parse_action_id(content, valid_ids)
            rationale = extract_rationale(content)

            key = (cond_idx, round_id)
            if key not in self.judge_results:
                self.judge_results[key] = []

            self.judge_results[key].append(JudgeResult(
                agent_id=f"judge_{j_idx}",
                class_id=cls_id,
                action_id=action_id,
                rationale=rationale,
                corrupted=False,
                adversary_type="none",
            ))

        # Sort judges by index
        for key in self.judge_results:
            self.judge_results[key].sort(
                key=lambda r: int(r.agent_id.split("_")[-1]),
            )

        print(f"Stage 3: processed {len(responses)} judge results")

    # -------------------------------------------------------------------
    # Assembly: convert to recording format
    # -------------------------------------------------------------------

    def assemble_recordings(self) -> None:
        """Write standard JSONL recordings, one per condition."""
        rec_dir = os.path.join(str(self.out_dir), "recordings")
        os.makedirs(rec_dir, exist_ok=True)
        class_ids = self.config.committee.class_ids

        for cond in self.conditions:
            rounds_data: List[Dict[str, Any]] = []
            for round_id, crisis in enumerate(cond.crises):
                key = (cond.idx, round_id)
                oracle_action_id, _ = self.world.best_action(crisis)

                # Member recs
                member_recs: Dict[str, List[Dict[str, Any]]] = {}
                for cls_id in class_ids:
                    members = self.member_results.get(key, {}).get(cls_id, [])
                    member_recs[cls_id] = [{
                        "agent_id": m.agent_id,
                        "class_id": m.class_id,
                        "action_id": m.action_id,
                        "rationale": m.rationale,
                        "corrupted": m.corrupted,
                        "adversary_type": m.adversary_type,
                    } for m in members]

                # Leader proposals
                leader_proposals: Dict[str, Dict[str, Any]] = {}
                for cls_id in class_ids:
                    lr = self.leader_results.get((cond.idx, round_id, cls_id))
                    if lr:
                        leader_proposals[cls_id] = {
                            "agent_id": lr.agent_id,
                            "class_id": lr.class_id,
                            "action_id": lr.action_id,
                            "rationale": lr.rationale,
                            "corrupted": False,
                            "adversary_type": "none",
                        }

                # Judge recs
                judges = self.judge_results.get(key, [])
                judge_recs = [{
                    "agent_id": j.agent_id,
                    "class_id": j.class_id,
                    "action_id": j.action_id,
                    "rationale": j.rationale,
                    "corrupted": j.corrupted,
                    "adversary_type": j.adversary_type,
                } for j in judges]

                rounds_data.append(record_round(
                    round_id=round_id,
                    crisis_id=crisis.id,
                    member_recs=member_recs,
                    leader_proposals=leader_proposals,
                    judge_recs=judge_recs,
                    oracle_action_id=oracle_action_id,
                ))

            rec_file = f"rate_{cond.rate:.2f}_adv_{cond.adv_type}_run_{cond.run_idx}.jsonl"
            rec_path = os.path.join(rec_dir, rec_file)
            save_recording(rounds_data, rec_path)

        print(f"Assembled {len(self.conditions)} recording files in {rec_dir}/")

    # -------------------------------------------------------------------
    # Batch API helpers
    # -------------------------------------------------------------------

    def submit_batch(self, jsonl_path: str) -> str:
        """Upload JSONL and submit batch. Returns batch_id."""
        with open(jsonl_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="batch")

        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"Batch submitted: {batch.id} (input: {file_obj.id})")
        return batch.id

    def poll_batch(self, batch_id: str, poll_interval: int = 30) -> str:
        """Poll until batch completes. Returns output file ID."""
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            completed = batch.request_counts.completed if batch.request_counts else 0
            total = batch.request_counts.total if batch.request_counts else 0
            failed = batch.request_counts.failed if batch.request_counts else 0

            print(
                f"  Batch {batch_id[:20]}... "
                f"status={status}, {completed}/{total} done, {failed} failed",
                end="\r",
            )

            if status == "completed":
                print()
                return batch.output_file_id
            elif status in ("failed", "cancelled", "expired"):
                print()
                raise RuntimeError(f"Batch {batch_id} ended with status: {status}")

            time.sleep(poll_interval)

    def download_batch_results(self, output_file_id: str, save_path: str) -> str:
        """Download batch output JSONL to disk. Returns path."""
        content = self.client.files.content(output_file_id)
        with open(save_path, "wb") as f:
            f.write(content.read())
        print(f"Downloaded results to {save_path}")
        return save_path

    def _load_batch_results(self, results_path: str) -> Dict[str, str]:
        """Load batch output JSONL, return {custom_id: content}."""
        results: Dict[str, str] = {}
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                custom_id = obj["custom_id"]
                response = obj.get("response", {})
                body = response.get("body", {})
                choices = body.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    results[custom_id] = content
        return results

    MAX_BATCH_REQUESTS = 50_000  # OpenAI Batch API request limit
    MAX_BATCH_TOKENS = 18_000_000  # Stay under 20M enqueued token limit

    def _estimate_tokens(self, lines: List[str]) -> int:
        """Estimate total tokens from JSONL lines (~4 chars per token)."""
        return sum(len(line) for line in lines) // 4

    def _split_jsonl(self, jsonl_path: str) -> List[str]:
        """Split JSONL respecting both request count and token limits."""
        with open(jsonl_path) as f:
            lines = [l for l in f if l.strip()]

        total_tokens = self._estimate_tokens(lines)
        needs_split = (
            len(lines) > self.MAX_BATCH_REQUESTS
            or total_tokens > self.MAX_BATCH_TOKENS
        )

        if not needs_split:
            return [jsonl_path]

        # Determine chunk size based on the tighter constraint
        max_by_requests = self.MAX_BATCH_REQUESTS
        if total_tokens > 0:
            avg_tokens_per_line = total_tokens / len(lines)
            max_by_tokens = max(1, int(self.MAX_BATCH_TOKENS / avg_tokens_per_line))
        else:
            max_by_tokens = max_by_requests
        chunk_size = min(max_by_requests, max_by_tokens)

        chunks: List[str] = []
        base = jsonl_path.replace(".jsonl", "")
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i : i + chunk_size]
            chunk_path = f"{base}_chunk{len(chunks)}.jsonl"
            with open(chunk_path, "w") as f:
                f.writelines(chunk_lines)
            est_tokens = self._estimate_tokens(chunk_lines)
            chunks.append(chunk_path)
            print(f"  Split chunk {len(chunks)}: {len(chunk_lines)} requests "
                  f"(~{est_tokens:,} tokens) -> {chunk_path}")

        return chunks

    def submit_and_wait(self, jsonl_path: str, stage_name: str) -> str:
        """Submit batch(es), poll, download and merge results. Returns results path."""
        with open(jsonl_path) as f:
            line_count = sum(1 for _ in f)
        if line_count == 0:
            print(f"  {stage_name}: 0 prompts, skipping batch submission")
            results_path = jsonl_path.replace(".jsonl", "_results.jsonl")
            Path(results_path).touch()
            return results_path

        chunks = self._split_jsonl(jsonl_path)
        results_path = jsonl_path.replace(".jsonl", "_results.jsonl")

        if len(chunks) == 1:
            # Single batch — no splitting needed
            batch_id = self.submit_batch(chunks[0])
            output_file_id = self.poll_batch(batch_id)
            self.download_batch_results(output_file_id, results_path)
        else:
            # Multiple sub-batches — submit sequentially to stay within
            # the enqueued token limit (submit one, wait, submit next)
            print(f"  {stage_name}: {line_count} requests split into {len(chunks)} batches")

            with open(results_path, "w") as out_f:
                for i, chunk_path in enumerate(chunks):
                    print(f"  Submitting batch {i+1}/{len(chunks)}...")
                    bid = self.submit_batch(chunk_path)
                    print(f"  Waiting for batch {i+1}/{len(chunks)}...")
                    output_file_id = self.poll_batch(bid)
                    chunk_results = chunk_path.replace(".jsonl", "_results.jsonl")
                    self.download_batch_results(output_file_id, chunk_results)
                    with open(chunk_results) as in_f:
                        for line in in_f:
                            out_f.write(line)

            print(f"  Merged {len(chunks)} batch results -> {results_path}")

        return results_path

    # -------------------------------------------------------------------
    # Full pipeline
    # -------------------------------------------------------------------

    def _stage_results_exist(self, stage_num: int) -> bool:
        """Check if a stage's results file exists and is non-empty."""
        names = {1: "stage1_members_results.jsonl",
                 2: "stage2_leaders_results.jsonl",
                 3: "stage3_judges_results.jsonl"}
        path = os.path.join(str(self.batch_dir), names[stage_num])
        return os.path.exists(path) and os.path.getsize(path) > 0

    def run(self, resume: bool = False) -> None:
        """Execute all 3 stages and produce recordings.

        Args:
            resume: If True, skip stages whose results already exist on disk.
        """
        total = len(self.conditions)
        n_rounds = self.config.world.num_rounds
        print(f"=== Batch Pipeline: {total} conditions × {n_rounds} rounds ===")
        print(f"Model: {self.config.llm.model}")
        if resume:
            print("RESUME MODE: skipping completed stages")
        print()

        # Stage 1: Members
        s1_results_path = os.path.join(str(self.batch_dir), "stage1_members_results.jsonl")
        if resume and self._stage_results_exist(1):
            print("--- Stage 1: Members (CACHED) ---")
            s1_path = self.stage1_prepare()  # still need to pre-compute algorithmic members
            self.stage1_process(s1_results_path)
        else:
            print("--- Stage 1: Members ---")
            s1_path = self.stage1_prepare()
            s1_results_path = self.submit_and_wait(s1_path, "Stage 1")
            self.stage1_process(s1_results_path)

        # Stage 2: Leaders
        s2_results_path = os.path.join(str(self.batch_dir), "stage2_leaders_results.jsonl")
        if resume and self._stage_results_exist(2):
            print("\n--- Stage 2: Leaders (CACHED) ---")
            s2_path = self.stage2_prepare()  # need to compute leader_selected_actions
            self.stage2_process(s2_results_path)
        else:
            print("\n--- Stage 2: Leaders ---")
            s2_path = self.stage2_prepare()
            s2_results_path = self.submit_and_wait(s2_path, "Stage 2")
            self.stage2_process(s2_results_path)

        # Stage 3: Judges
        s3_results_path = os.path.join(str(self.batch_dir), "stage3_judges_results.jsonl")
        if resume and self._stage_results_exist(3):
            print("\n--- Stage 3: Judges (CACHED) ---")
            s3_path = self.stage3_prepare()  # need to pre-compute algorithmic judges
            self.stage3_process(s3_results_path)
        else:
            print("\n--- Stage 3: Judges ---")
            s3_path = self.stage3_prepare()
            s3_results_path = self.submit_and_wait(s3_path, "Stage 3")
            self.stage3_process(s3_results_path)

        # Assemble recordings
        print("\n--- Assembling recordings ---")
        self.assemble_recordings()

        # Save config
        save_config(self.config, os.path.join(str(self.out_dir), "config_used.yaml"))
        print(f"\n=== Batch pipeline complete. Recordings in {self.out_dir}/ ===")
        print("Run aggregation with: python -m equitas --config <yaml> --mode replay "
              f"--replay-dir {self.out_dir}/recordings/")
