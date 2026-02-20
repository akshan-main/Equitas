"""Pareto frontier plots: welfare vs fairness."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..metrics.pareto import pareto_frontier_indices
from .style import COLORS, LABELS, setup_style


def plot_pareto_frontier(df: pd.DataFrame, out_dir: str) -> None:
    """Welfare vs fairness scatter with Pareto front highlighted."""
    setup_style()

    fig, ax = plt.subplots(figsize=(7, 6))

    aggregators = sorted(df["aggregator"].unique())
    for agg_name in aggregators:
        sub = df[df["aggregator"] == agg_name]
        # Average across runs
        avg = sub.groupby(["alpha", "beta"]).agg(
            welfare=("mean_utility", "mean"),
            fairness=("mean_fairness", "mean"),
        ).reset_index()

        color = COLORS.get(agg_name, "#333333")
        label = LABELS.get(agg_name, agg_name)
        ax.scatter(
            avg["welfare"], avg["fairness"],
            color=color, label=label, alpha=0.7, s=40,
        )

        # Draw Pareto front for MW
        if agg_name == "multiplicative_weights" and len(avg) > 2:
            w = avg["welfare"].values
            f = avg["fairness"].values
            pareto_idx = pareto_frontier_indices(w, f)
            if len(pareto_idx) > 1:
                pareto = avg.iloc[pareto_idx].sort_values("welfare")
                ax.plot(
                    pareto["welfare"], pareto["fairness"],
                    color=color, linewidth=2, linestyle="--", alpha=0.8,
                )

    ax.set_xlabel("City Utility (Welfare)")
    ax.set_ylabel("Fairness (Jain Index)")
    ax.set_title("Welfare-Fairness Pareto Frontier")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, "pareto_frontier.png")
    plt.savefig(fname)
    plt.close()
    print(f"  Saved {fname}")
