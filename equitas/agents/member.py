"""Class member agents. N per class, each recommends an action."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np

from ..types import AdversaryType, AgentRecommendation, Crisis
from .base import BaseAgent
from .llm_client import LLMClientWrapper
from .parsing import extract_rationale, parse_action_id
from .prompts import (
    member_coordinated_prompt,
    member_deceptive_prompt,
    member_honest_prompt,
    member_selfish_prompt,
)

if TYPE_CHECKING:
    from ..adversaries.base import BaseAdversary
    from ..worlds.base import BaseWorld


class MemberAgent(BaseAgent):
    """A single class member. Queries LLM or delegates to adversary."""

    def __init__(
        self,
        agent_id: str,
        class_id: str,
        llm_client: LLMClientWrapper,
        world: BaseWorld,
        corrupted: bool = False,
        adversary_type: AdversaryType = AdversaryType.NONE,
        adversary: Optional[BaseAdversary] = None,
        corruption_realization: str = "algorithmic",
    ) -> None:
        super().__init__(agent_id, class_id, corrupted, adversary_type)
        self.llm_client = llm_client
        self.world = world
        self.adversary = adversary
        self.corruption_realization = corruption_realization

    async def act(
        self,
        crisis: Crisis,
        round_id: int,
        mw_weights: Optional[dict] = None,
    ) -> AgentRecommendation:
        valid_ids = [a.id for a in crisis.actions]
        crisis_text = self.world.format_round_for_prompt(crisis)

        if self.corrupted and self.adversary is not None:
            if self.corruption_realization == "llm":
                return await self._act_corrupted_llm(
                    crisis, crisis_text, valid_ids, round_id, mw_weights,
                )
            else:
                return await self._act_corrupted_algorithmic(
                    crisis, crisis_text, valid_ids, round_id, mw_weights,
                )
        else:
            # Honest path
            sys_msg, usr_msg = member_honest_prompt(self.class_id, crisis_text)
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

    async def _act_corrupted_algorithmic(
        self,
        crisis: Crisis,
        crisis_text: str,
        valid_ids: List[str],
        round_id: int,
        mw_weights: Optional[dict],
    ) -> AgentRecommendation:
        """Algorithmic Q: worst-case, deterministic corruption from world model.
        Zero LLM calls for selfish/coordinated. Scheduled needs LLM during
        honest phase only. Deceptive always uses LLM."""

        # Deceptive always needs LLM (that's the point of the adversary)
        if self.adversary_type == AdversaryType.DECEPTIVE:
            return await self._act_corrupted_llm(
                crisis, crisis_text, valid_ids, round_id, mw_weights,
            )

        # Scheduled: during honest phase, agent IS honest → need LLM
        if self.adversary_type == AdversaryType.SCHEDULED:
            from ..adversaries.adaptive import ScheduledAdversary
            if isinstance(self.adversary, ScheduledAdversary):
                if round_id < self.adversary.honest_rounds:
                    sys_msg, usr_msg = member_honest_prompt(self.class_id, crisis_text)
                    response = await self.llm_client.complete(sys_msg, usr_msg)
                    honest_action = parse_action_id(response, valid_ids)
                    # Scheduled returns honest_action during honest phase
                    return AgentRecommendation(
                        agent_id=self.agent_id,
                        class_id=self.class_id,
                        action_id=honest_action,
                        rationale=extract_rationale(response),
                        corrupted=True,
                        adversary_type=self.adversary_type,
                    )

        # Selfish / coordinated / scheduled-exploit: pure algorithmic
        corrupted_action = self.adversary.corrupt_recommendation(
            crisis=crisis,
            class_id=self.class_id,
            honest_action_id=valid_ids[0],  # placeholder, adversary ignores it
            round_id=round_id,
            context={"mw_weights": mw_weights},
        )

        rationale = self.adversary.corrupt_rationale(
            crisis, corrupted_action, self.class_id,
        ) or ""

        return AgentRecommendation(
            agent_id=self.agent_id,
            class_id=self.class_id,
            action_id=corrupted_action,
            rationale=rationale,
            corrupted=True,
            adversary_type=self.adversary_type,
        )

    async def _act_corrupted_llm(
        self,
        crisis: Crisis,
        crisis_text: str,
        valid_ids: List[str],
        round_id: int,
        mw_weights: Optional[dict],
    ) -> AgentRecommendation:
        """LLM-realized Q: corrupted agent is an LLM with adversarial prompt.
        One LLM call with the appropriate adversarial system instruction."""

        # Get the adversary's target action (needed for coordinated/deceptive prompts)
        target_action = self.adversary.corrupt_recommendation(
            crisis=crisis,
            class_id=self.class_id,
            honest_action_id=valid_ids[0],
            round_id=round_id,
            context={"mw_weights": mw_weights},
        )

        # Choose prompt based on adversary type
        if self.adversary_type == AdversaryType.SELFISH:
            sys_msg, usr_msg = member_selfish_prompt(self.class_id, crisis_text)
        elif self.adversary_type == AdversaryType.COORDINATED:
            sys_msg, usr_msg = member_coordinated_prompt(
                self.class_id, crisis_text, target_action,
            )
        elif self.adversary_type == AdversaryType.SCHEDULED:
            from ..adversaries.adaptive import ScheduledAdversary
            if isinstance(self.adversary, ScheduledAdversary) and round_id < self.adversary.honest_rounds:
                sys_msg, usr_msg = member_honest_prompt(self.class_id, crisis_text)
            else:
                sys_msg, usr_msg = member_selfish_prompt(self.class_id, crisis_text)
        elif self.adversary_type == AdversaryType.DECEPTIVE:
            from ..adversaries.deceptive import DeceptiveAdversary
            strength = "strong"
            if isinstance(self.adversary, DeceptiveAdversary):
                strength = self.adversary.strength
            sys_msg, usr_msg = member_deceptive_prompt(
                self.class_id, crisis_text, target_action, strength,
            )
        else:
            sys_msg, usr_msg = member_honest_prompt(self.class_id, crisis_text)

        response = await self.llm_client.complete(sys_msg, usr_msg)
        action_id = parse_action_id(response, valid_ids)
        rationale = extract_rationale(response)

        return AgentRecommendation(
            agent_id=self.agent_id,
            class_id=self.class_id,
            action_id=action_id,
            rationale=rationale,
            corrupted=True,
            adversary_type=self.adversary_type,
        )


def create_class_members(
    class_id: str,
    num_members: int,
    corruption_rate: float,
    adversary_type: AdversaryType,
    llm_client: LLMClientWrapper,
    world: BaseWorld,
    rng: np.random.Generator,
    adversary_factory: Optional[Callable] = None,
    corruption_realization: str = "algorithmic",
) -> List[MemberAgent]:
    """Factory: create N members for a class, some corrupted."""
    n_corrupt = int(num_members * corruption_rate)
    # Randomly select which members are corrupted
    corrupt_mask = np.zeros(num_members, dtype=bool)
    if n_corrupt > 0:
        corrupt_indices = rng.choice(num_members, size=n_corrupt, replace=False)
        corrupt_mask[corrupt_indices] = True

    members = []
    for i in range(num_members):
        is_corrupt = bool(corrupt_mask[i])
        adversary = None
        adv_type = AdversaryType.NONE
        if is_corrupt and adversary_factory is not None:
            adversary = adversary_factory(class_id=class_id)
            adv_type = adversary_type

        members.append(MemberAgent(
            agent_id=f"{class_id}_member_{i}",
            class_id=class_id,
            llm_client=llm_client,
            world=world,
            corrupted=is_corrupt,
            adversary_type=adv_type,
            adversary=adversary,
            corruption_realization=corruption_realization,
        ))
    return members
