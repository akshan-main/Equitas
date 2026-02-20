"""
Global configuration for the Plato-style MW + robust LLM governance project.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI model & key for AutoGen
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to your .env file or environment."
    )

# Ensure AutoGen / OpenAI client sees it
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Plato-style social classes
CLASS_IDS = ["guardian", "auxiliary", "producer"]

# Axes describing crises in the city
CRISIS_AXES = [
    "resource_scarcity",
    "external_threat",
    "inequality",
    "economic_instability",
]

# Policy levers that actions can tune
POLICY_DIMS = [
    "tax_merchants",
    "welfare_workers",
    "military_spend",
    "education_investment",
]

# How many actions per crisis scenario
ACTIONS_PER_CRISIS = 3
