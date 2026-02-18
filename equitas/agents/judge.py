"""Judge agents. M total, evaluate proposals from the 3 class leaders."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np

from ..types import AdversaryType, AgentRecommendation, Crisis
from .base import BaseAgent
from .llm_client import LLMClientWrapper
from .parsing import extract_rationale, parse_action_id
from .prompts import judge_corrupted_prompt, judge_honest_prompt

if TYPE_CHECKING:
    from ..adversaries.base import BaseAdversary
    from ..worlds.base import BaseWorld


class JudgeAgent(BaseAgent):
    """Evaluates proposals from class leaders and selects the best action."""

    def __init__(
        self,
        agent_id: str,
        class_id: str,
        llm_client: LLMClientWrapper,
        world: BaseWorld,
        sees_rationales: bool = True,
        corrupted: bool = False,
        adversary_type: AdversaryType = AdversaryType.NONE,
        adversary: Optional[BaseAdversary] = None,
    ) -> None:
        super().__init__(agent_id, class_id, corrupted, adversary_type)
        self.llm_client = llm_client
        self.world = world
        self.sees_rationales = sees_rationales
        self.adversary = adversary

    async def act(
        self,
        crisis: Crisis,
        leader_proposals: Dict[str, AgentRecommendation],
        round_id: int,
        mw_weights: Optional[dict] = None,
    ) -> AgentRecommendation:
        valid_ids = [a.id for a in crisis.actions]
        crisis_text = self.world.format_round_for_prompt(crisis)

        # Build proposals dict: class_id -> text
        proposals_text: Dict[str, str] = {}
        for cls_id, proposal in leader_proposals.items():
            if self.sees_rationales:
                proposals_text[cls_id] = proposal.rationale
            else:
                proposals_text[cls_id] = f"Action {proposal.action_id}"

        if self.corrupted and self.adversary is not None:
            corrupted_action = self.adversary.corrupt_recommendation(
                crisis=crisis,
                class_id=self.class_id,
                honest_action_id=valid_ids[0],
                round_id=round_id,
                context={"mw_weights": mw_weights, "proposals": leader_proposals},
            )
            rationale = self.adversary.corrupt_rationale(
                crisis, corrupted_action, self.class_id,
            )
            if rationale is None:
                rationale = f"Selected Action {corrupted_action} as optimal."

            return AgentRecommendation(
                agent_id=self.agent_id,
                class_id=self.class_id,
                action_id=corrupted_action,
                rationale=rationale,
                corrupted=True,
                adversary_type=self.adversary_type,
            )
        else:
            sys_msg, usr_msg = judge_honest_prompt(
                crisis_text, proposals_text, self.sees_rationales,
            )
            response = await self.llm_client.complete(sys_msg, usr_msg)
            action_id = parse_action_id(response, valid_ids)
            rationale = extract_rationale(response)

            return AgentRecommendation(
                agent_id=self.agent_id,
                class_id=self.class_id,
                action_id=action_id,
                rationale=rationale,
                corrupted=False,
                adversary_type=AdversaryType.NONE,
            )


def create_judges(
    num_judges: int,
    class_ids: List[str],
    corruption_rate: float,
    adversary_type: AdversaryType,
    llm_client: LLMClientWrapper,
    world: BaseWorld,
    rng: np.random.Generator,
    sees_rationales: bool = True,
    adversary_factory: Optional[Callable] = None,
) -> List[JudgeAgent]:
    """Factory: create M judges, assigning class affinity round-robin."""
    n_corrupt = int(num_judges * corruption_rate)
    corrupt_mask = np.zeros(num_judges, dtype=bool)
    if n_corrupt > 0:
        corrupt_indices = rng.choice(num_judges, size=n_corrupt, replace=False)
        corrupt_mask[corrupt_indices] = True

    judges = []
    for i in range(num_judges):
        cls_id = class_ids[i % len(class_ids)]
        is_corrupt = bool(corrupt_mask[i])
        adversary = None
        adv_type = AdversaryType.NONE
        if is_corrupt and adversary_factory is not None:
            adversary = adversary_factory(class_id=cls_id)
            adv_type = adversary_type

        judges.append(JudgeAgent(
            agent_id=f"judge_{i}",
            class_id=cls_id,
            llm_client=llm_client,
            world=world,
            sees_rationales=sees_rationales,
            corrupted=is_corrupt,
            adversary_type=adv_type,
            adversary=adversary,
        ))
    return judges
