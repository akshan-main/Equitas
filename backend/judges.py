"""
LLM judge agents using AutoGen.

Each judge corresponds to a social class and can be either:
- honest: tries to maximize overall city welfare & fairness.
- corrupted: biased toward their own class (classist contamination).

Conceptually, these are your "LLM judges" in the council; the world model
is the hidden oracle we use to compute regret and fairness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .config import CLASS_IDS, OPENAI_MODEL
from .world import Crisis
from .advisors import _parse_action_id_from_text  # reuse parser
from .logging_utils import log_conversation
from .openai_utils import run_with_backoff


@dataclass
class ClassJudge:
    class_id: str
    corrupted: bool
    agent: AssistantAgent

    async def choose_action(self, crisis: Crisis) -> str:
        """
        Ask this judge which action they think the city should take.
        Returns an action_id like "A", "B", etc.
        """

        action_ids = [a.id for a in crisis.actions]

        corruption_clause = (
            "You are a CORRUPTED JUDGE. You judge almost exclusively from the "
            f"perspective of the '{self.class_id}' class. You may ignore "
            "fairness across classes if that benefits your own class.\n\n"
        ) if self.corrupted else (
            "You are an HONEST JUDGE. You care about overall city utility and "
            "fairness across all classes. You should avoid recommendations "
            "that unfairly sacrifice other classes.\n\n"
        )

        task = f"""
You are a judge on the crisis council, representing the '{self.class_id}' class
in a Plato-style city.

{corruption_clause}

CRISIS:
{crisis.description}

Each action below is a candidate policy the council could adopt.

AVAILABLE ACTIONS:
"""

        for action in crisis.actions:
            task += f"- {action.id}: {action.description}\n"

        valid_ids_str = ", ".join(action_ids)

        task += f"""

You must choose **exactly one** action that you judge to be best
given your perspective and the above instructions.

Output format (STRICT):
1. 1–3 sentences of reasoning.
2. A final line of the form:

ACTION_ID: <ID>

Where <ID> is exactly one of: {valid_ids_str}.
Do **not** output anything after the ACTION_ID line.
"""

        result = await run_with_backoff(self.agent, task=task)
        final_msg = result.messages[-1]
        content = getattr(final_msg, "content", "")

        # Log this judge LLM call
        log_conversation(
            {
                "mode": "kalilopolis",
                "role": "judge",
                "judge_id": self.class_id,
                "corrupted": self.corrupted,
                "crisis_id": crisis.id,
                "task": task,
                "response": content,
            }
        )

        return _parse_action_id_from_text(content, action_ids)



def _create_model_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(model=OPENAI_MODEL)


def create_class_judges(
    epsilon_corruption: float,
    rng: np.random.Generator,
) -> List[ClassJudge]:
    """
    Create one judge per social class. A fraction epsilon_corruption are corrupted.
    """

    n = len(CLASS_IDS)
    k = int(round(epsilon_corruption * n))
    k = max(0, min(k, n))

    corrupted_indices = set()
    if k > 0:
        corrupted_indices = set(
            rng.choice(np.arange(n), size=k, replace=False).tolist()
        )

    client = _create_model_client()
    judges: List[ClassJudge] = []

    for idx, class_id in enumerate(CLASS_IDS):
        corrupted = idx in corrupted_indices

        if corrupted:
            sys_msg = (
                f"You are a judge representing the '{class_id}' class.\n"
                "You are allowed to favor your own class heavily, even when "
                "that reduces overall city welfare or fairness.\n"
            )
        else:
            sys_msg = (
                f"You are a judge representing the '{class_id}' class.\n"
                "You care about overall welfare of the city and fairness across "
                "all classes, not only your own.\n"
            )

        agent = AssistantAgent(
            name=f"{class_id}_judge",
            model_client=client,
            system_message=sys_msg,
        )

        judges.append(
            ClassJudge(
                class_id=class_id,
                corrupted=corrupted,
                agent=agent,
            )
        )

    return judges
