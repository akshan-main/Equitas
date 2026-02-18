"""Winner bands plot: which mechanism dominates at each epsilon."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import COLORS, LABELS, setup_style


def plot_winner_bands(
    summary: pd.DataFrame,
    out_dir: str,
    metric: str = "trimmed_mean_utility",
    oracle_name: str = "oracle_upper_bound",
) -> None:
    """Generate a winner-bands figure: stacked bar showing which
    mechanism leads at each (adversary_type, epsilon) condition.

    Also generates a margin plot showing how far ahead the winner is.
    """
    setup_style()
    non_oracle = summary[summary["aggregator"] != oracle_name]
    adv_types = sorted(non_oracle["adversary_type"].unique())
    aggregators = sorted(non_oracle["aggregator"].unique())

    # --- Figure 1: Winner bands (stacked categorical) ---
    n_panels = len(adv_types)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, adv in zip(axes, adv_types):
        sub = non_oracle[non_oracle["adversary_type"] == adv]
        rates = sorted(sub["corruption_rate"].unique())

        winners = []
        values = []
        margins = []
        for rate in rates:
            rate_data = sub[sub["corruption_rate"] == rate].sort_values(
                metric, ascending=False)
            best = rate_data.iloc[0]
            winners.append(best["aggregator"])
            values.append(best[metric])
            if len(rate_data) >= 2:
                margins.append(best[metric] - rate_data.iloc[1][metric])
            else:
                margins.append(0.0)

        colors = [COLORS.get(w, "#333333") for w in winners]
        bars = ax.bar(
            [str(r) for r in rates], values, color=colors, edgecolor="white",
            linewidth=0.5,
        )

        # Annotate winner name
        for bar, w, m in zip(bars, winners, margins):
            label = LABELS.get(w, w).split("(")[0].strip()
            if len(label) > 10:
                label = label[:9] + "."
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                label,
                ha="center", va="bottom", fontsize=7, rotation=30,
            )

        ax.set_xlabel("Corruption Rate (ε)")
        ax.set_title(adv.capitalize())
        if ax == axes[0]:
            ax.set_ylabel(f"Best {metric.split('_')[-1].capitalize()}")

    fig.suptitle("Winner Bands: Best Mechanism per Condition", y=1.05)
    plt.tight_layout()
    path = os.path.join(out_dir, "regime_winner_bands.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")

    # --- Figure 2: Pairwise margin (how far ahead is winner) ---
    fig2, axes2 = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes2 = [axes2]

    for ax, adv in zip(axes2, adv_types):
        sub = non_oracle[non_oracle["adversary_type"] == adv]
        rates = sorted(sub["corruption_rate"].unique())

        # For each pair of key aggregators, plot margin vs epsilon
        for agg_name in aggregators:
            agg_data = sub[sub["aggregator"] == agg_name].sort_values("corruption_rate")
            if agg_data.empty:
                continue
            x = agg_data["corruption_rate"].values
            y = agg_data[metric].values
            color = COLORS.get(agg_name, "#333333")
            label = LABELS.get(agg_name, agg_name)
            ax.plot(x, y, color=color, label=label, linewidth=2, marker="o",
                    markersize=5)

        # Shade winner region
        for i, rate in enumerate(rates):
            rate_data = sub[sub["corruption_rate"] == rate].sort_values(
                metric, ascending=False)
            if rate_data.empty:
                continue
            best_agg = rate_data.iloc[0]["aggregator"]
            best_color = COLORS.get(best_agg, "#333333")
            if i < len(rates) - 1:
                next_rate = rates[i + 1]
            else:
                next_rate = rate + (rates[1] - rates[0]) if len(rates) > 1 else rate + 0.25
            ax.axvspan(rate - 0.01, (rate + next_rate) / 2, alpha=0.08,
                       color=best_color)

        ax.set_xlabel("Corruption Rate (ε)")
        ax.set_title(adv.capitalize())
        if ax == axes2[0]:
            ax.set_ylabel("Metric Value")

    axes2[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig2.suptitle("Mechanism Performance with Winner Regions", y=1.02)
    plt.tight_layout()
    path2 = os.path.join(out_dir, "regime_performance_bands.png")
    plt.savefig(path2)
    plt.close()
    print(f"  Saved {path2}")
