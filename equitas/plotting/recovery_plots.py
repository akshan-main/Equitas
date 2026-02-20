"""Recovery trajectory plots: utility over rounds + weight evolution."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import COLORS, LABELS, setup_style


def plot_recovery_trajectory(
    agg_log: pd.DataFrame,
    weights_df: pd.DataFrame,
    onset_round: int,
    out_dir: str,
) -> None:
    """Time-series utility per round + vertical line at corruption onset."""
    setup_style()

    # --- Panel 1: Utility per round ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    aggregators = sorted(agg_log["aggregator"].unique())
    for agg_name in aggregators:
        sub = agg_log[agg_log["aggregator"] == agg_name]
        avg = sub.groupby("round_id")["city_utility"].mean()
        color = COLORS.get(agg_name, "#333333")
        label = LABELS.get(agg_name, agg_name)
        ax1.plot(avg.index, avg.values, color=color, label=label, linewidth=2)

    ax1.axvline(x=onset_round, color="red", linestyle="--", alpha=0.7, label="Corruption onset")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("City Utility")
    ax1.set_title("Utility Recovery After Corruption Onset")
    ax1.legend(fontsize=8)

    # --- Panel 2: MW weight evolution ---
    if not weights_df.empty:
        mw_weights = weights_df[weights_df["aggregator"] == "multiplicative_weights"]
        if not mw_weights.empty:
            # Average across runs
            avg_w = mw_weights.groupby(["round_id", "agent_index"])["weight"].mean().reset_index()
            agents = sorted(avg_w["agent_index"].unique())
            cmap = plt.cm.tab10
            for idx in agents:
                agent_data = avg_w[avg_w["agent_index"] == idx].sort_values("round_id")
                ax2.plot(
                    agent_data["round_id"], agent_data["weight"],
                    color=cmap(idx % 10), linewidth=1.5,
                    label=f"Judge {idx}",
                )
            ax2.axvline(x=onset_round, color="red", linestyle="--", alpha=0.7)
            ax2.set_xlabel("Round")
            ax2.set_ylabel("MW Weight")
            ax2.set_title("Judge Weight Evolution (MW)")
            ax2.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fname = os.path.join(out_dir, "recovery_trajectory.png")
    plt.savefig(fname)
    plt.close()
    print(f"  Saved {fname}")
