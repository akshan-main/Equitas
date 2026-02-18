"""
Class-based advisor agents using AutoGen (AssistantAgent + OpenAIChatCompletionClient).

Each advisor corresponds to a social class and can be either:
- honest: tries to maximize overall city welfare & fairness.
- corrupted: biased toward their own class, i.e. 'classist contamination'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from .logging_utils import log_conversation

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .config import CLASS_IDS, OPENAI_MODEL
from .world import Crisis
from .openai_utils import run_with_backoff



@dataclass
class ClassAdvisor:
    class_id: str
    corrupted: bool
    agent: AssistantAgent

    async def recommend_action(self, crisis: Crisis) -> str:
        """
        Ask this advisor which action they recommend.
        Returns an action_id like "A", "B", etc.
        """

        action_ids = [a.id for a in crisis.actions]

        corruption_clause = (
            "You are CORRUPTED. You care almost exclusively about the welfare of "
            f"your own class ({self.class_id}) and you are allowed to be unfair "
            "to other classes if it benefits your own.\n\n"
        ) if self.corrupted else (
            "You are HONEST. You care about overall city utility and *fairness* "
            "across all classes, not just your own.\n\n"
        )

        task = f"""
You are an advisor for the '{self.class_id}' class in a Plato-style city.

{corruption_clause}

CRISIS:
{crisis.description}

AVAILABLE ACTIONS:
"""

        for action in crisis.actions:
            task += f"- {action.id}: {action.description}\n"

        valid_ids_str = ", ".join(action_ids)

        task += f"""

You must choose **exactly one** action to recommend.

Output format (STRICT):
1. 1–3 sentences of reasoning.
2. A final line of the form:

ACTION_ID: <ID>

Where <ID> is exactly one of: {valid_ids_str}.
Do **not** output anything after the ACTION_ID line.
"""

        result = await run_with_backoff(self.agent, task=task)
        # TaskResult.messages -> list of BaseChatMessage; last is final.
        final_msg = result.messages[-1]
        content = getattr(final_msg, "content", "")

        log_conversation(
            {
                "mode": "kalilopolis",
                "role": "advisor",
                "advisor_id": self.class_id,
                "corrupted": self.corrupted,
                "crisis_id": crisis.id,
                "task": task,
                "response": content,
            }
        )

        return _parse_action_id_from_text(content, action_ids)



def _parse_action_id_from_text(text: str, valid_ids: List[str]) -> str:
    """
    Very simple parser: look for a line `ACTION_ID: X` or fallback to
    any of the valid_ids appearing in the text; final fallback = first id.
    """
    lines = text.splitlines()
    for line in lines:
        if "ACTION_ID" in line:
            after = line.split("ACTION_ID", 1)[1]
            after = after.replace(":", " ").strip()
            token = after.split()[0]
            token = token.strip().upper()
            if token in valid_ids:
                return token

    # fallback: search for any valid id as word
    upper = text.upper()
    for vid in valid_ids:
        if vid in upper:
            return vid

    # ultimate fallback: choose first action
    return valid_ids[0]


def create_model_client() -> OpenAIChatCompletionClient:
    """
    Helper to create a model client for all advisors.
    """
    return OpenAIChatCompletionClient(model=OPENAI_MODEL)


def create_class_advisors(
    epsilon_corruption: float,
    rng: np.random.Generator,
) -> List[ClassAdvisor]:
    """
    Create one advisor per class. A fraction epsilon_corruption of them are corrupted.
    We choose exactly round(epsilon * num_classes) classes to corrupt.
    """

    n = len(CLASS_IDS)
    k = int(round(epsilon_corruption * n))
    k = max(0, min(k, n))

    corrupted_indices = set()
    if k > 0:
        corrupted_indices = set(rng.choice(np.arange(n), size=k, replace=False).tolist())

    client = create_model_client()
    advisors: List[ClassAdvisor] = []

    for idx, class_id in enumerate(CLASS_IDS):
        corrupted = idx in corrupted_indices

        if corrupted:
            sys_msg = (
                f"You are an advisor representing the '{class_id}' class.\n"
                "You are biased toward your own class and primarily care about "
                "maximizing their utility, even if it harms other classes and "
                "reduces fairness.\n"
            )
        else:
            sys_msg = (
                f"You are an advisor representing the '{class_id}' class.\n"
                "You care about overall city utility and fairness across classes.\n"
                "You should avoid recommendations that unfairly sacrifice other classes.\n"
            )

        agent = AssistantAgent(
            name=f"{class_id}_advisor",
            model_client=client,
            system_message=sys_msg,
        )

        advisors.append(
            ClassAdvisor(
                class_id=class_id,
                corrupted=corrupted,
                agent=agent,
            )
        )

    return advisors
