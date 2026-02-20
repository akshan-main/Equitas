"""Committee size scaling plots."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from .style import COLORS, LABELS, MARKERS, setup_style


def plot_scaling(df: pd.DataFrame, out_dir: str) -> None:
    """Robustness vs committee size N."""
    setup_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    aggregators = sorted(df["aggregator"].unique())

    for agg_name in aggregators:
        sub = df[df["aggregator"] == agg_name]
        avg = sub.groupby("members_per_class").agg(
            utility=("mean_utility", "mean"),
            fairness=("mean_fairness", "mean"),
        ).reset_index()

        color = COLORS.get(agg_name, "#333333")
        marker = MARKERS.get(agg_name, "o")
        label = LABELS.get(agg_name, agg_name)

        ax1.plot(
            avg["members_per_class"], avg["utility"],
            marker=marker, color=color, label=label, linewidth=2,
        )
        ax2.plot(
            avg["members_per_class"], avg["fairness"],
            marker=marker, color=color, label=label, linewidth=2,
        )

    ax1.set_xlabel("Members per Class (N)")
    ax1.set_ylabel("City Utility")
    ax1.set_title("Utility vs Committee Size")

    ax2.set_xlabel("Members per Class (N)")
    ax2.set_ylabel("Fairness (Jain)")
    ax2.set_title("Fairness vs Committee Size")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    fname = os.path.join(out_dir, "scaling_robustness.png")
    plt.savefig(fname)
    plt.close()
    print(f"  Saved {fname}")
