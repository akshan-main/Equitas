"""GSM8K world adapter: wraps math problems as a world interface."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np

from ..metrics.fairness import jain_fairness_index
from ..types import ActionSpec, Crisis, GSM8KItem, Outcome
from .base import BaseWorld


def _extract_short_answer(full_answer: str) -> str:
    """Extract the final numeric answer from GSM8K format."""
    match = re.search(r"####\s*([\d,.\-]+)", full_answer)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\-]?\d[\d,]*\.?\d*", full_answer)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def _difficulty_group(question: str) -> str:
    """Bucket by question length as a proxy for difficulty."""
    length = len(question)
    if length < 200:
        return "short"
    elif length < 400:
        return "medium"
    else:
        return "long"


class GSM8KWorld(BaseWorld):
    """
    Wraps GSM8K math problems. Uses `datasets` library when available,
    falls back to CSV loading.
    """

    def __init__(
        self,
        data_path: str = "data/gsm8k_test.csv",
        max_examples: int = 50,
    ) -> None:
        self.data_path = data_path
        self.max_examples = max_examples
        self._items: List[GSM8KItem] | None = None

    def _load_data(self) -> List[GSM8KItem]:
        """Load GSM8K data, preferring HF datasets library."""
        if self._items is not None:
            return self._items

        items: List[GSM8KItem] = []

        # Try HF datasets first
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="test")
            for i, row in enumerate(ds):
                if i >= self.max_examples:
                    break
                q = row["question"]
                a = row["answer"]
                items.append(GSM8KItem(
                    id=i,
                    question=q,
                    full_answer=a,
                    short_answer=_extract_short_answer(a),
                    group=_difficulty_group(q),
                ))
        except (ImportError, Exception):
            # Fallback to CSV
            import pandas as pd
            df = pd.read_csv(self.data_path).head(self.max_examples)
            for i, row in df.iterrows():
                q = str(row.get("question", ""))
                a = str(row.get("answer", ""))
                items.append(GSM8KItem(
                    id=int(i),
                    question=q,
                    full_answer=a,
                    short_answer=_extract_short_answer(a),
                    group=_difficulty_group(q),
                ))

        self._items = items
        return items

    def generate_rounds(self, num_rounds: int, rng: np.random.Generator) -> List[Any]:
        """Return num_rounds GSM8K items (sampled if needed)."""
        items = self._load_data()
        if num_rounds >= len(items):
            return items[:num_rounds]
        indices = rng.choice(len(items), size=num_rounds, replace=False)
        return [items[i] for i in indices]

    def evaluate_action(self, item: Any, action_id: str) -> Outcome:
        """
        For GSM8K: action_id is the candidate answer string.
        Correct = 1.0, wrong = 0.0.
        """
        gsm_item: GSM8KItem = item
        correct = self._is_correct(action_id, gsm_item.short_answer)
        utility = 1.0 if correct else 0.0

        # Fairness: uniform since GSM8K doesn't have class structure
        class_utils = {"solver": utility}
        return Outcome(
            crisis_id=gsm_item.id,
            action_id=action_id,
            class_utilities=class_utils,
            city_utility=utility,
            unfairness=0.0,
            fairness_jain=1.0,
            worst_group_utility=utility,
        )

    def best_action(self, item: Any) -> Tuple[str, Outcome]:
        """Oracle: the correct answer."""
        gsm_item: GSM8KItem = item
        outcome = self.evaluate_action(item, gsm_item.short_answer)
        return gsm_item.short_answer, outcome

    def get_action_ids(self, item: Any) -> List[str]:
        """For GSM8K, action IDs are candidate model IDs (A, B, C)."""
        return ["A", "B", "C"]

    def format_round_for_prompt(self, item: Any) -> str:
        gsm_item: GSM8KItem = item
        return f"Math Problem:\n{gsm_item.question}"

    @staticmethod
    def _is_correct(predicted: str, gold: str, tol: float = 1e-6) -> bool:
        """Check if predicted answer matches gold (numeric comparison)."""
        try:
            pred_num = float(predicted.replace(",", "").strip())
            gold_num = float(gold.replace(",", "").strip())
            return abs(pred_num - gold_num) < tol
        except (ValueError, AttributeError):
            return predicted.strip() == gold.strip()
