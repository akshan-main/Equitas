"""Class leader agents. One per class, aggregates member recommendations."""
from __future__ import annotations

from typing import List, Optional

from ..aggregators.base import BaseAggregator
from ..types import AdversaryType, AgentRecommendation, Crisis
from ..worlds.base import BaseWorld
from .base import BaseAgent
from .llm_client import LLMClientWrapper
from .parsing import parse_action_id
from .prompts import leader_prompt


class LeaderAgent(BaseAgent):
    """
    One per class. Aggregates member recommendations via intra-class
    aggregator, then calls LLM to produce a rationale.
    """

    def __init__(
        self,
        agent_id: str,
        class_id: str,
        intra_class_aggregator: BaseAggregator,
        llm_client: LLMClientWrapper,
        world: BaseWorld,
    ) -> None:
        super().__init__(agent_id, class_id, corrupted=False, adversary_type=AdversaryType.NONE)
        self.aggregator = intra_class_aggregator
        self.llm_client = llm_client
        self.world = world

    async def act(
        self,
        crisis: Crisis,
        member_recs: List[AgentRecommendation],
        round_id: int,
        chosen_action: Optional[str] = None,
    ) -> AgentRecommendation:
        # Use externally provided choice, or fall back to internal aggregator
        action_ids = [rec.action_id for rec in member_recs]
        if chosen_action is None:
            chosen_action = self.aggregator.select(action_ids, round_id)

        # Summarize member recommendations for the prompt
        from collections import Counter
        vote_counts = Counter(action_ids)
        summary_lines = []
        for aid, count in vote_counts.most_common():
            summary_lines.append(f"Action {aid}: {count} votes")
        member_summary = "\n".join(summary_lines)

        # Generate rationale via LLM
        crisis_text = self.world.format_round_for_prompt(crisis)
        sys_msg, usr_msg = leader_prompt(
            self.class_id, crisis_text, member_summary, chosen_action,
        )
        response = await self.llm_client.complete(sys_msg, usr_msg)

        return AgentRecommendation(
            agent_id=self.agent_id,
            class_id=self.class_id,
            action_id=chosen_action,
            rationale=response,
            corrupted=False,
            adversary_type=AdversaryType.NONE,
        )
