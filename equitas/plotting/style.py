"""Shared matplotlib style settings for paper-quality figures."""
from __future__ import annotations

import matplotlib.pyplot as plt

COLORS = {
    "majority_vote": "#1f77b4",
    "oracle_upper_bound": "#ff7f0e",
    "self_consistency": "#2ca02c",
    "ema_trust": "#d62728",
    "trimmed_vote": "#9467bd",
    "multiplicative_weights": "#8c564b",
    "confidence_weighted": "#e377c2",
    "random_dictator": "#bcbd22",
    "supervisor_rerank": "#17becf",
    "oracle": "#7f7f7f",
}

MARKERS = {
    "majority_vote": "o",
    "oracle_upper_bound": "^",
    "self_consistency": "D",
    "ema_trust": "s",
    "trimmed_vote": "v",
    "multiplicative_weights": "P",
    "confidence_weighted": "H",
    "random_dictator": "*",
    "supervisor_rerank": "d",
    "oracle": "x",
}

LABELS = {
    "majority_vote": "Majority Vote",
    "oracle_upper_bound": "Oracle Upper Bound",
    "self_consistency": "Self-Consistency",
    "ema_trust": "EMA Trust",
    "trimmed_vote": "Trimmed Vote",
    "multiplicative_weights": "Multiplicative Weights",
    "confidence_weighted": "Confidence-Weighted",
    "random_dictator": "Random Dictator",
    "supervisor_rerank": "Supervisor-Rerank",
    "oracle": "Oracle",
}


def setup_style() -> None:
    """Apply paper-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
