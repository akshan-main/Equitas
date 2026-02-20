"""
Generate GO-vs-FH comparison tables across all 5 experiment types.
Outputs CSVs + LaTeX-ready tables to tables/ directory.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

OUT = "tables"
os.makedirs(OUT, exist_ok=True)

# ── Paths ──
SWEEP_GO = "outputs/run2/governance_sweep_batch/sweep_summary.csv"
SWEEP_FH = "outputs/run2/governance_sweep_fh_batch/sweep_summary.csv"
RECOV_GO = "outputs/run2/governance_recovery/recovery_aggregator_log.csv"
RECOV_FH = "outputs/run2/governance_recovery_fh/recovery_fh_aggregator_log.csv"
SCALE_GO = "outputs/run2/governance_scaling/scaling_results.csv"
SCALE_FH = "outputs/run2/governance_scaling_fh/scaling_fh_results.csv"
HVF_GO = "outputs/run2/governance_hier_vs_flat/hierarchical_vs_flat.csv"
HVF_FH = "outputs/run2/governance_hier_vs_flat_fh/fh_hierarchical_vs_flat.csv"
PARETO_GO = "outputs/run2/governance_pareto/pareto_results.csv"
PARETO_FH = "outputs/run2/governance_pareto_fh/pareto_fh_results.csv"


def fmt(x: float, decimals: int = 4) -> str:
    return f"{x:.{decimals}f}"


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert DataFrame to a LaTeX booktabs table."""
    ncols = len(df.columns)
    col_fmt = "l" + "r" * (ncols - 1)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
    ]
    # Header
    header = " & ".join(str(c) for c in df.columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    # Rows
    for _, row in df.iterrows():
        row_str = " & ".join(str(v) for v in row.values) + r" \\"
        lines.append(row_str)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =====================================================================
# Table 1: Sweep — GO vs FH aggregator performance (averaged over ε, adv)
# =====================================================================
def table_sweep_aggregator():
    go = pd.read_csv(SWEEP_GO)
    fh = pd.read_csv(SWEEP_FH)

    # Exclude oracle_upper_bound from GO (FH doesn't have it in the same sense)
    # Actually, keep it — both have it
    go_agg = (
        go.groupby("aggregator")
        .agg(
            go_utility=("mean_utility", "mean"),
            go_fairness=("mean_fairness", "mean"),
            go_worst_group=("mean_worst_group", "mean"),
        )
        .reset_index()
    )
    fh_agg = (
        fh.groupby("aggregator")
        .agg(
            fh_utility=("mean_utility", "mean"),
            fh_fairness=("mean_fairness", "mean"),
            fh_worst_group=("mean_worst_group", "mean"),
        )
        .reset_index()
    )
    merged = go_agg.merge(fh_agg, on="aggregator", how="outer")
    merged["delta_utility"] = merged["fh_utility"] - merged["go_utility"]
    merged["delta_fairness"] = merged["fh_fairness"] - merged["go_fairness"]
    merged["delta_worst_group"] = merged["fh_worst_group"] - merged["go_worst_group"]

    # Format
    display = pd.DataFrame()
    display["Aggregator"] = merged["aggregator"]
    display["GO Utility"] = merged["go_utility"].map(lambda x: fmt(x))
    display["FH Utility"] = merged["fh_utility"].map(lambda x: fmt(x))
    display["\u0394 Utility"] = merged["delta_utility"].map(lambda x: f"{x:+.4f}")
    display["GO Fairness"] = merged["go_fairness"].map(lambda x: fmt(x))
    display["FH Fairness"] = merged["fh_fairness"].map(lambda x: fmt(x))
    display["\u0394 Fairness"] = merged["delta_fairness"].map(lambda x: f"{x:+.4f}")
    display["GO Worst-Grp"] = merged["go_worst_group"].map(lambda x: fmt(x))
    display["FH Worst-Grp"] = merged["fh_worst_group"].map(lambda x: fmt(x))
    display["\u0394 Worst-Grp"] = merged["delta_worst_group"].map(lambda x: f"{x:+.4f}")

    display.to_csv(os.path.join(OUT, "table1_sweep_aggregator_go_vs_fh.csv"), index=False)

    tex = to_latex(
        display,
        caption="Sweep: GO vs FH aggregator performance (averaged over $\\varepsilon$ and adversary type).",
        label="tab:sweep-go-vs-fh",
    )
    with open(os.path.join(OUT, "table1_sweep_aggregator_go_vs_fh.tex"), "w") as f:
        f.write(tex)

    print("Table 1: Sweep aggregator GO vs FH")
    print(display.to_string(index=False))
    print()
    return merged


# =====================================================================
# Table 2: Sweep — GO vs FH by corruption rate (averaged over adv, agg)
# =====================================================================
def table_sweep_by_corruption():
    go = pd.read_csv(SWEEP_GO)
    fh = pd.read_csv(SWEEP_FH)

    go_cr = (
        go.groupby("corruption_rate")
        .agg(go_utility=("mean_utility", "mean"), go_fairness=("mean_fairness", "mean"),
             go_worst_group=("mean_worst_group", "mean"))
        .reset_index()
    )
    fh_cr = (
        fh.groupby("corruption_rate")
        .agg(fh_utility=("mean_utility", "mean"), fh_fairness=("mean_fairness", "mean"),
             fh_worst_group=("mean_worst_group", "mean"))
        .reset_index()
    )
    merged = go_cr.merge(fh_cr, on="corruption_rate")
    merged["delta_utility"] = merged["fh_utility"] - merged["go_utility"]
    merged["delta_fairness"] = merged["fh_fairness"] - merged["go_fairness"]

    display = pd.DataFrame()
    display["Corruption Rate"] = merged["corruption_rate"]
    display["GO Utility"] = merged["go_utility"].map(lambda x: fmt(x))
    display["FH Utility"] = merged["fh_utility"].map(lambda x: fmt(x))
    display["\u0394 Utility"] = merged["delta_utility"].map(lambda x: f"{x:+.4f}")
    display["GO Fairness"] = merged["go_fairness"].map(lambda x: fmt(x))
    display["FH Fairness"] = merged["fh_fairness"].map(lambda x: fmt(x))
    display["\u0394 Fairness"] = merged["delta_fairness"].map(lambda x: f"{x:+.4f}")

    display.to_csv(os.path.join(OUT, "table2_sweep_corruption_go_vs_fh.csv"), index=False)

    tex = to_latex(
        display,
        caption="Sweep: GO vs FH by corruption rate (averaged over adversary types and aggregators).",
        label="tab:sweep-corruption-go-vs-fh",
    )
    with open(os.path.join(OUT, "table2_sweep_corruption_go_vs_fh.tex"), "w") as f:
        f.write(tex)

    print("Table 2: Sweep by corruption rate GO vs FH")
    print(display.to_string(index=False))
    print()


# =====================================================================
# Table 3: Sweep — GO vs FH by adversary type (averaged over ε, agg)
# =====================================================================
def table_sweep_by_adversary():
    go = pd.read_csv(SWEEP_GO)
    fh = pd.read_csv(SWEEP_FH)

    go_adv = (
        go.groupby("adversary_type")
        .agg(go_utility=("mean_utility", "mean"), go_fairness=("mean_fairness", "mean"),
             go_worst_group=("mean_worst_group", "mean"))
        .reset_index()
    )
    fh_adv = (
        fh.groupby("adversary_type")
        .agg(fh_utility=("mean_utility", "mean"), fh_fairness=("mean_fairness", "mean"),
             fh_worst_group=("mean_worst_group", "mean"))
        .reset_index()
    )
    merged = go_adv.merge(fh_adv, on="adversary_type")
    merged["delta_utility"] = merged["fh_utility"] - merged["go_utility"]
    merged["delta_fairness"] = merged["fh_fairness"] - merged["go_fairness"]

    display = pd.DataFrame()
    display["Adversary"] = merged["adversary_type"]
    display["GO Utility"] = merged["go_utility"].map(lambda x: fmt(x))
    display["FH Utility"] = merged["fh_utility"].map(lambda x: fmt(x))
    display["\u0394 Utility"] = merged["delta_utility"].map(lambda x: f"{x:+.4f}")
    display["GO Fairness"] = merged["go_fairness"].map(lambda x: fmt(x))
    display["FH Fairness"] = merged["fh_fairness"].map(lambda x: fmt(x))
    display["\u0394 Fairness"] = merged["delta_fairness"].map(lambda x: f"{x:+.4f}")
    display["GO Worst-Grp"] = merged["go_worst_group"].map(lambda x: fmt(x))
    display["FH Worst-Grp"] = merged["fh_worst_group"].map(lambda x: fmt(x))

    display.to_csv(os.path.join(OUT, "table3_sweep_adversary_go_vs_fh.csv"), index=False)

    tex = to_latex(
        display,
        caption="Sweep: GO vs FH by adversary type (averaged over $\\varepsilon$ and aggregators).",
        label="tab:sweep-adversary-go-vs-fh",
    )
    with open(os.path.join(OUT, "table3_sweep_adversary_go_vs_fh.tex"), "w") as f:
        f.write(tex)

    print("Table 3: Sweep by adversary type GO vs FH")
    print(display.to_string(index=False))
    print()


# =====================================================================
# Table 4: Recovery — GO vs FH pre/post corruption onset
# =====================================================================
def table_recovery():
    go = pd.read_csv(RECOV_GO)
    fh = pd.read_csv(RECOV_FH)

    onset = go["onset_round"].iloc[0]

    def split_phases(df, label):
        pre = df[df["round_id"] < onset].groupby("aggregator").agg(
            pre_utility=("city_utility", "mean"),
            pre_fairness=("fairness_jain", "mean"),
        ).reset_index()
        post = df[df["round_id"] >= onset].groupby("aggregator").agg(
            post_utility=("city_utility", "mean"),
            post_fairness=("fairness_jain", "mean"),
        ).reset_index()
        merged = pre.merge(post, on="aggregator")
        merged["utility_drop"] = merged["post_utility"] - merged["pre_utility"]
        merged["fairness_drop"] = merged["post_fairness"] - merged["pre_fairness"]
        return merged

    go_phases = split_phases(go, "GO")
    fh_phases = split_phases(fh, "FH")

    combined = go_phases.merge(fh_phases, on="aggregator", suffixes=("_go", "_fh"))

    display = pd.DataFrame()
    display["Aggregator"] = combined["aggregator"]
    display["GO Pre-U"] = combined["pre_utility_go"].map(lambda x: fmt(x))
    display["GO Post-U"] = combined["post_utility_go"].map(lambda x: fmt(x))
    display["GO Drop"] = combined["utility_drop_go"].map(lambda x: f"{x:+.4f}")
    display["FH Pre-U"] = combined["pre_utility_fh"].map(lambda x: fmt(x))
    display["FH Post-U"] = combined["post_utility_fh"].map(lambda x: fmt(x))
    display["FH Drop"] = combined["utility_drop_fh"].map(lambda x: f"{x:+.4f}")
    display["GO \u0394Fair"] = combined["fairness_drop_go"].map(lambda x: f"{x:+.4f}")
    display["FH \u0394Fair"] = combined["fairness_drop_fh"].map(lambda x: f"{x:+.4f}")

    display.to_csv(os.path.join(OUT, "table4_recovery_go_vs_fh.csv"), index=False)

    tex = to_latex(
        display,
        caption=f"Recovery: GO vs FH utility and fairness before/after corruption onset (round {onset}).",
        label="tab:recovery-go-vs-fh",
    )
    with open(os.path.join(OUT, "table4_recovery_go_vs_fh.tex"), "w") as f:
        f.write(tex)

    print(f"Table 4: Recovery GO vs FH (onset round {onset})")
    print(display.to_string(index=False))
    print()


# =====================================================================
# Table 5: Scaling — GO vs FH by committee size
# =====================================================================
def table_scaling():
    go = pd.read_csv(SCALE_GO)
    fh = pd.read_csv(SCALE_FH)

    go_s = (
        go.groupby(["members_per_class", "aggregator"])
        .agg(go_utility=("mean_utility", "mean"), go_fairness=("mean_fairness", "mean"))
        .reset_index()
    )
    fh_s = (
        fh.groupby(["members_per_class", "aggregator"])
        .agg(fh_utility=("mean_utility", "mean"), fh_fairness=("mean_fairness", "mean"))
        .reset_index()
    )
    merged = go_s.merge(fh_s, on=["members_per_class", "aggregator"], how="outer")

    # Pivot: rows = aggregator, columns = committee size, values = utility
    # Simpler: average across aggregators per size
    go_size = go.groupby("members_per_class").agg(
        go_utility=("mean_utility", "mean"), go_fairness=("mean_fairness", "mean"),
        go_worst_group=("mean_worst_group", "mean"),
    ).reset_index()
    fh_size = fh.groupby("members_per_class").agg(
        fh_utility=("mean_utility", "mean"), fh_fairness=("mean_fairness", "mean"),
        fh_worst_group=("mean_worst_group", "mean"),
    ).reset_index()
    sizes = go_size.merge(fh_size, on="members_per_class")
    sizes["delta_utility"] = sizes["fh_utility"] - sizes["go_utility"]
    sizes["delta_fairness"] = sizes["fh_fairness"] - sizes["go_fairness"]

    display = pd.DataFrame()
    display["N (per class)"] = sizes["members_per_class"]
    display["GO Utility"] = sizes["go_utility"].map(lambda x: fmt(x))
    display["FH Utility"] = sizes["fh_utility"].map(lambda x: fmt(x))
    display["\u0394 Utility"] = sizes["delta_utility"].map(lambda x: f"{x:+.4f}")
    display["GO Fairness"] = sizes["go_fairness"].map(lambda x: fmt(x))
    display["FH Fairness"] = sizes["fh_fairness"].map(lambda x: fmt(x))
    display["\u0394 Fairness"] = sizes["delta_fairness"].map(lambda x: f"{x:+.4f}")
    display["GO Worst-Grp"] = sizes["go_worst_group"].map(lambda x: fmt(x))
    display["FH Worst-Grp"] = sizes["fh_worst_group"].map(lambda x: fmt(x))

    display.to_csv(os.path.join(OUT, "table5_scaling_go_vs_fh.csv"), index=False)

    tex = to_latex(
        display,
        caption="Scaling: GO vs FH by committee size (averaged over aggregators and runs).",
        label="tab:scaling-go-vs-fh",
    )
    with open(os.path.join(OUT, "table5_scaling_go_vs_fh.tex"), "w") as f:
        f.write(tex)

    print("Table 5: Scaling GO vs FH")
    print(display.to_string(index=False))
    print()

    # Also: per-aggregator breakdown at each size (detailed table)
    detail = merged.copy()
    detail["delta_utility"] = detail["fh_utility"] - detail["go_utility"]
    detail_display = pd.DataFrame()
    detail_display["N"] = detail["members_per_class"]
    detail_display["Aggregator"] = detail["aggregator"]
    detail_display["GO Utility"] = detail["go_utility"].map(lambda x: fmt(x) if pd.notna(x) else "—")
    detail_display["FH Utility"] = detail["fh_utility"].map(lambda x: fmt(x) if pd.notna(x) else "—")
    detail_display["\u0394 Utility"] = detail["delta_utility"].map(
        lambda x: f"{x:+.4f}" if pd.notna(x) else "—"
    )
    detail_display = detail_display.sort_values(["N", "Aggregator"])
    detail_display.to_csv(os.path.join(OUT, "table5b_scaling_detail_go_vs_fh.csv"), index=False)


# =====================================================================
# Table 6: Hierarchical vs Flat — GO vs FH architecture comparison
# =====================================================================
def table_hier_vs_flat():
    go = pd.read_csv(HVF_GO)
    fh = pd.read_csv(HVF_FH)

    # GO: architectures are "hierarchical" and "flat"
    # FH: architectures are "fh_hierarchical" and "fh_flat"

    def arch_summary(df, hier_label, flat_label):
        hier = df[df["architecture"] == hier_label].groupby("corruption_rate").agg(
            hier_utility=("mean_utility", "mean"),
            hier_fairness=("mean_fairness", "mean"),
            hier_worst_group=("mean_worst_group", "mean"),
        ).reset_index()
        flat = df[df["architecture"] == flat_label].groupby("corruption_rate").agg(
            flat_utility=("mean_utility", "mean"),
            flat_fairness=("mean_fairness", "mean"),
            flat_worst_group=("mean_worst_group", "mean"),
        ).reset_index()
        return hier.merge(flat, on="corruption_rate")

    go_arch = arch_summary(go, "hierarchical", "flat")
    fh_arch = arch_summary(fh, "fh_hierarchical", "flat")

    merged = go_arch.merge(fh_arch, on="corruption_rate", suffixes=("_go", "_fh"))

    display = pd.DataFrame()
    display["Rate"] = merged["corruption_rate"]
    display["GO Hier-U"] = merged["hier_utility_go"].map(lambda x: fmt(x))
    display["GO Flat-U"] = merged["flat_utility_go"].map(lambda x: fmt(x))
    display["GO Gap"] = (merged["hier_utility_go"] - merged["flat_utility_go"]).map(
        lambda x: f"{x:+.4f}"
    )
    display["FH Hier-U"] = merged["hier_utility_fh"].map(lambda x: fmt(x))
    display["FH Flat-U"] = merged["flat_utility_fh"].map(lambda x: fmt(x))
    display["FH Gap"] = (merged["hier_utility_fh"] - merged["flat_utility_fh"]).map(
        lambda x: f"{x:+.4f}"
    )
    display["GO Hier-Fair"] = merged["hier_fairness_go"].map(lambda x: fmt(x))
    display["FH Hier-Fair"] = merged["hier_fairness_fh"].map(lambda x: fmt(x))

    display.to_csv(os.path.join(OUT, "table6_hier_vs_flat_go_vs_fh.csv"), index=False)

    tex = to_latex(
        display,
        caption="Hierarchical vs Flat: GO vs FH architecture comparison by corruption rate.",
        label="tab:hier-flat-go-vs-fh",
    )
    with open(os.path.join(OUT, "table6_hier_vs_flat_go_vs_fh.tex"), "w") as f:
        f.write(tex)

    print("Table 6: Hierarchical vs Flat GO vs FH")
    print(display.to_string(index=False))
    print()


# =====================================================================
# Table 7: Pareto — GO vs FH frontier comparison
# =====================================================================
def table_pareto():
    go = pd.read_csv(PARETO_GO)
    fh = pd.read_csv(PARETO_FH)

    # Average over runs, keep alpha/beta/aggregator
    go_p = (
        go.groupby(["alpha", "beta", "aggregator"])
        .agg(go_utility=("mean_utility", "mean"), go_fairness=("mean_fairness", "mean"),
             go_worst_group=("mean_worst_group", "mean"))
        .reset_index()
    )
    fh_p = (
        fh.groupby(["alpha", "beta", "aggregator"])
        .agg(fh_utility=("mean_utility", "mean"), fh_fairness=("mean_fairness", "mean"),
             fh_worst_group=("mean_worst_group", "mean"))
        .reset_index()
    )

    # MW only (the parameterized aggregator)
    go_mw = go_p[go_p["aggregator"] == "multiplicative_weights"].copy()
    fh_mw = fh_p[fh_p["aggregator"] == "multiplicative_weights"].copy()
    merged = go_mw.merge(fh_mw, on=["alpha", "beta"], suffixes=("_go", "_fh"))

    merged["delta_utility"] = merged["fh_utility"] - merged["go_utility"]
    merged["delta_fairness"] = merged["fh_fairness"] - merged["go_fairness"]

    display = pd.DataFrame()
    display["alpha"] = merged["alpha"]
    display["beta"] = merged["beta"]
    display["GO Utility"] = merged["go_utility"].map(lambda x: fmt(x))
    display["FH Utility"] = merged["fh_utility"].map(lambda x: fmt(x))
    display["\u0394 Utility"] = merged["delta_utility"].map(lambda x: f"{x:+.4f}")
    display["GO Fairness"] = merged["go_fairness"].map(lambda x: fmt(x))
    display["FH Fairness"] = merged["fh_fairness"].map(lambda x: fmt(x))
    display["\u0394 Fairness"] = merged["delta_fairness"].map(lambda x: f"{x:+.4f}")

    display.to_csv(os.path.join(OUT, "table7_pareto_mw_go_vs_fh.csv"), index=False)

    tex = to_latex(
        display,
        caption="Pareto MW: GO vs FH utility--fairness across $(\\alpha, \\beta)$ grid.",
        label="tab:pareto-go-vs-fh",
    )
    with open(os.path.join(OUT, "table7_pareto_mw_go_vs_fh.tex"), "w") as f:
        f.write(tex)

    print("Table 7: Pareto MW GO vs FH")
    print(display.to_string(index=False))
    print()

    # Summary: average GO vs FH across entire Pareto grid
    summary = pd.DataFrame([{
        "Mode": "GO",
        "Mean Utility": fmt(go_mw["go_utility"].mean()),
        "Mean Fairness": fmt(go_mw["go_fairness"].mean()),
        "Mean Worst-Grp": fmt(go_mw["go_worst_group"].mean()),
    }, {
        "Mode": "FH",
        "Mean Utility": fmt(fh_mw["fh_utility"].mean()),
        "Mean Fairness": fmt(fh_mw["fh_fairness"].mean()),
        "Mean Worst-Grp": fmt(fh_mw["fh_worst_group"].mean()),
    }])
    summary.to_csv(os.path.join(OUT, "table7b_pareto_summary_go_vs_fh.csv"), index=False)


# =====================================================================
# Table 8: Grand summary — one row per experiment type
# =====================================================================
def table_grand_summary():
    rows = []

    # Sweep
    go_sw = pd.read_csv(SWEEP_GO)
    fh_sw = pd.read_csv(SWEEP_FH)
    rows.append({
        "Experiment": "Sweep",
        "GO Utility": go_sw["mean_utility"].mean(),
        "FH Utility": fh_sw["mean_utility"].mean(),
        "GO Fairness": go_sw["mean_fairness"].mean(),
        "FH Fairness": fh_sw["mean_fairness"].mean(),
        "GO Worst-Grp": go_sw["mean_worst_group"].mean(),
        "FH Worst-Grp": fh_sw["mean_worst_group"].mean(),
    })

    # Recovery
    go_r = pd.read_csv(RECOV_GO)
    fh_r = pd.read_csv(RECOV_FH)
    rows.append({
        "Experiment": "Recovery",
        "GO Utility": go_r["city_utility"].mean(),
        "FH Utility": fh_r["city_utility"].mean(),
        "GO Fairness": go_r["fairness_jain"].mean(),
        "FH Fairness": fh_r["fairness_jain"].mean(),
        "GO Worst-Grp": go_r["worst_group_utility"].mean(),
        "FH Worst-Grp": fh_r["worst_group_utility"].mean(),
    })

    # Scaling
    go_sc = pd.read_csv(SCALE_GO)
    fh_sc = pd.read_csv(SCALE_FH)
    rows.append({
        "Experiment": "Scaling",
        "GO Utility": go_sc["mean_utility"].mean(),
        "FH Utility": fh_sc["mean_utility"].mean(),
        "GO Fairness": go_sc["mean_fairness"].mean(),
        "FH Fairness": fh_sc["mean_fairness"].mean(),
        "GO Worst-Grp": go_sc["mean_worst_group"].mean(),
        "FH Worst-Grp": fh_sc["mean_worst_group"].mean(),
    })

    # Hier vs Flat (hierarchical arm only)
    go_hf = pd.read_csv(HVF_GO)
    fh_hf = pd.read_csv(HVF_FH)
    go_hier = go_hf[go_hf["architecture"] == "hierarchical"]
    fh_hier = fh_hf[fh_hf["architecture"] == "fh_hierarchical"]
    rows.append({
        "Experiment": "Hier vs Flat",
        "GO Utility": go_hier["mean_utility"].mean(),
        "FH Utility": fh_hier["mean_utility"].mean(),
        "GO Fairness": go_hier["mean_fairness"].mean(),
        "FH Fairness": fh_hier["mean_fairness"].mean(),
        "GO Worst-Grp": go_hier["mean_worst_group"].mean(),
        "FH Worst-Grp": fh_hier["mean_worst_group"].mean(),
    })

    # Pareto (MW only)
    go_p = pd.read_csv(PARETO_GO)
    fh_p = pd.read_csv(PARETO_FH)
    go_mw = go_p[go_p["aggregator"] == "multiplicative_weights"]
    fh_mw = fh_p[fh_p["aggregator"] == "multiplicative_weights"]
    rows.append({
        "Experiment": "Pareto (MW)",
        "GO Utility": go_mw["mean_utility"].mean(),
        "FH Utility": fh_mw["mean_utility"].mean(),
        "GO Fairness": go_mw["mean_fairness"].mean(),
        "FH Fairness": fh_mw["mean_fairness"].mean(),
        "GO Worst-Grp": go_mw["mean_worst_group"].mean(),
        "FH Worst-Grp": fh_mw["mean_worst_group"].mean(),
    })

    df = pd.DataFrame(rows)
    df["Delta Utility"] = df["FH Utility"] - df["GO Utility"]
    df["Delta Fairness"] = df["FH Fairness"] - df["GO Fairness"]

    display = pd.DataFrame()
    display["Experiment"] = df["Experiment"]
    display["GO Util"] = df["GO Utility"].map(lambda x: fmt(x))
    display["FH Util"] = df["FH Utility"].map(lambda x: fmt(x))
    display["\u0394 Util"] = df["Delta Utility"].map(lambda x: f"{x:+.4f}")
    display["GO Fair"] = df["GO Fairness"].map(lambda x: fmt(x))
    display["FH Fair"] = df["FH Fairness"].map(lambda x: fmt(x))
    display["\u0394 Fair"] = df["Delta Fairness"].map(lambda x: f"{x:+.4f}")
    display["GO WG"] = df["GO Worst-Grp"].map(lambda x: fmt(x))
    display["FH WG"] = df["FH Worst-Grp"].map(lambda x: fmt(x))

    display.to_csv(os.path.join(OUT, "table8_grand_summary_go_vs_fh.csv"), index=False)

    tex = to_latex(
        display,
        caption="Grand summary: GO vs FH across all five experiment types.",
        label="tab:grand-summary",
    )
    with open(os.path.join(OUT, "table8_grand_summary_go_vs_fh.tex"), "w") as f:
        f.write(tex)

    print("Table 8: Grand Summary GO vs FH")
    print(display.to_string(index=False))
    print()


# =====================================================================
# Table 9: Sweep — Best aggregator per regime (GO vs FH)
# =====================================================================
def table_regime_winners():
    go_det = pd.read_csv("outputs/run2/governance_sweep_batch/regime_map_detailed.csv")
    fh_det = pd.read_csv("outputs/run2/governance_sweep_fh_batch/regime_map_detailed.csv")

    # Filter to welfare metric only
    go_w = go_det[go_det["metric"] == "welfare"][["adversary_type", "corruption_rate", "best_aggregator", "margin"]].copy()
    fh_w = fh_det[fh_det["metric"] == "welfare"][["adversary_type", "corruption_rate", "best_aggregator", "margin"]].copy()

    merged = go_w.merge(fh_w, on=["adversary_type", "corruption_rate"], suffixes=("_go", "_fh"))

    display = pd.DataFrame()
    display["Adversary"] = merged["adversary_type"]
    display["Rate"] = merged["corruption_rate"]
    display["GO Best"] = merged["best_aggregator_go"]
    display["GO Margin"] = merged["margin_go"].map(lambda x: fmt(x))
    display["FH Best"] = merged["best_aggregator_fh"]
    display["FH Margin"] = merged["margin_fh"].map(lambda x: fmt(x))
    display["Same?"] = (merged["best_aggregator_go"] == merged["best_aggregator_fh"]).map(
        lambda x: "Yes" if x else "No"
    )

    display.to_csv(os.path.join(OUT, "table9_regime_winners_go_vs_fh.csv"), index=False)

    tex = to_latex(
        display,
        caption="Regime winners: best aggregator per (adversary, $\\varepsilon$) — GO vs FH.",
        label="tab:regime-winners",
    )
    with open(os.path.join(OUT, "table9_regime_winners_go_vs_fh.tex"), "w") as f:
        f.write(tex)

    print("Table 9: Regime Winners GO vs FH")
    print(display.to_string(index=False))
    print()


# =====================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("GENERATING GO vs FH COMPARISON TABLES")
    print("=" * 80)
    print()

    table_sweep_aggregator()
    table_sweep_by_corruption()
    table_sweep_by_adversary()
    table_recovery()
    table_scaling()
    table_hier_vs_flat()
    table_pareto()
    table_grand_summary()
    table_regime_winners()

    print("=" * 80)
    n_csv = len([f for f in os.listdir(OUT) if f.endswith(".csv")])
    n_tex = len([f for f in os.listdir(OUT) if f.endswith(".tex")])
    print(f"All tables saved to {OUT}/  ({n_csv} CSV + {n_tex} LaTeX files)")
    print("=" * 80)
