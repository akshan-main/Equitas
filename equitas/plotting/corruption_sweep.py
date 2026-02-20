"""Plots for corruption rate sweep experiments."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from .style import COLORS, LABELS, MARKERS, setup_style


def plot_corruption_sweep(summary: pd.DataFrame, out_dir: str) -> None:
    """
    Generate plots: one panel per adversary type.
    X-axis: corruption_rate, Y-axis: metric, lines per aggregator.
    """
    setup_style()

    adv_types = sorted(summary["adversary_type"].unique())
    aggregators = sorted(summary["aggregator"].unique())

    # Utility plot
    _plot_metric(
        summary, adv_types, aggregators,
        metric="trimmed_mean_utility",
        ci_lo="ci_low_utility", ci_hi="ci_high_utility",
        ylabel="City Utility (trimmed mean)",
        title_prefix="City Utility vs Corruption",
        filename=os.path.join(out_dir, "sweep_utility_vs_corruption.png"),
    )

    # Fairness plot
    _plot_metric(
        summary, adv_types, aggregators,
        metric="trimmed_mean_fairness",
        ci_lo="ci_low_fairness", ci_hi="ci_high_fairness",
        ylabel="Fairness (Jain, trimmed mean)",
        title_prefix="Fairness vs Corruption",
        filename=os.path.join(out_dir, "sweep_fairness_vs_corruption.png"),
    )

    # Worst-group utility
    if "mean_worst_group" in summary.columns:
        _plot_metric(
            summary, adv_types, aggregators,
            metric="mean_worst_group",
            ylabel="Worst-Group Utility",
            title_prefix="Worst-Group Utility vs Corruption",
            filename=os.path.join(out_dir, "sweep_worst_group_vs_corruption.png"),
        )


def _plot_metric(
    summary: pd.DataFrame,
    adv_types: list,
    aggregators: list,
    metric: str,
    ylabel: str,
    title_prefix: str,
    filename: str,
    ci_lo: str | None = None,
    ci_hi: str | None = None,
) -> None:
    n_panels = len(adv_types)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, adv in zip(axes, adv_types):
        sub = summary[summary["adversary_type"] == adv]
        for agg_name in aggregators:
            agg_data = sub[sub["aggregator"] == agg_name].sort_values("corruption_rate")
            if agg_data.empty:
                continue
            x = agg_data["corruption_rate"].values
            y = agg_data[metric].values
            color = COLORS.get(agg_name, "#333333")
            marker = MARKERS.get(agg_name, "o")
            label = LABELS.get(agg_name, agg_name)

            ax.plot(x, y, marker=marker, color=color, label=label, linewidth=2, markersize=6)

            if ci_lo and ci_hi and ci_lo in agg_data.columns:
                lo = agg_data[ci_lo].values
                hi = agg_data[ci_hi].values
                ax.fill_between(x, lo, hi, alpha=0.15, color=color)

        ax.set_xlabel("Corruption Rate")
        ax.set_title(f"{adv.capitalize()}")
        if ax == axes[0]:
            ax.set_ylabel(ylabel)

    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.suptitle(title_prefix, y=1.02)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  Saved {filename}")
