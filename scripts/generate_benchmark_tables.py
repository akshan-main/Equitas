"""
Generate Equitas benchmark tables (FH-primary, GO diagnostic).

Main tables (FH benchmark):
  B1  Aggregator leaderboard (headline numbers)
  B2  Performance by corruption rate
  B3  Performance by adversary type
  B4  Regime winners (best aggregator per regime)
  B5  Recovery: pre/post corruption onset
  B6  Scaling: robustness by committee size
  B7  Hierarchical vs Flat architecture
  B8  Pareto welfare-fairness frontier (MW)

Diagnostic (appendix):
  D1  GO vs FH gap analysis (why FH matters)

Output: tables/benchmark/  (CSV + LaTeX)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

OUT = "tables/benchmark"
os.makedirs(OUT, exist_ok=True)

# ── Paths ──
FH_SWEEP = "outputs/run2/governance_sweep_fh_batch/sweep_summary.csv"
FH_REGIME = "outputs/run2/governance_sweep_fh_batch/regime_map_detailed.csv"
FH_AGG_LOG = "outputs/run2/governance_sweep_fh_batch/sweep_aggregator_log.csv"
FH_RECOV = "outputs/run2/governance_recovery_fh/recovery_fh_aggregator_log.csv"
FH_RECOV_W = "outputs/run2/governance_recovery_fh/recovery_fh_weight_history.csv"
FH_SCALE = "outputs/run2/governance_scaling_fh/scaling_fh_results.csv"
FH_HVF = "outputs/run2/governance_hier_vs_flat_fh/fh_hierarchical_vs_flat.csv"
FH_PARETO = "outputs/run2/governance_pareto_fh/pareto_fh_results.csv"
GO_SWEEP = "outputs/run2/governance_sweep_batch/sweep_summary.csv"
GO_RECOV = "outputs/run2/governance_recovery/recovery_aggregator_log.csv"
GO_SCALE = "outputs/run2/governance_scaling/scaling_results.csv"
GO_HVF = "outputs/run2/governance_hier_vs_flat/hierarchical_vs_flat.csv"
GO_PARETO = "outputs/run2/governance_pareto/pareto_results.csv"

AGG_ORDER = [
    "oracle", "multiplicative_weights", "supervisor_rerank",
    "confidence_weighted", "ema_trust", "trimmed_vote",
    "majority_vote", "self_consistency", "oracle_upper_bound",
    "random_dictator",
]

AGG_SHORT = {
    "oracle": "Oracle",
    "multiplicative_weights": "MW",
    "supervisor_rerank": "Supervisor",
    "confidence_weighted": "Conf-Wt",
    "ema_trust": "EMA",
    "trimmed_vote": "Trimmed",
    "majority_vote": "Majority",
    "self_consistency": "Self-Cons",
    "oracle_upper_bound": "OracleUB",
    "random_dictator": "Random",
}


def fmt(x, d=4):
    if pd.isna(x):
        return "—"
    return f"{x:.{d}f}"


def sfmt(x, d=4):
    """Signed format."""
    if pd.isna(x):
        return "—"
    return f"{x:+.{d}f}"


def sort_agg(df, col="aggregator"):
    """Sort aggregator rows in canonical order."""
    order_map = {a: i for i, a in enumerate(AGG_ORDER)}
    df = df.copy()
    df["_sort"] = df[col].map(order_map).fillna(99)
    df = df.sort_values("_sort").drop(columns="_sort")
    return df


def to_latex(df, caption, label):
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
        " & ".join(str(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def save(df, name, caption, label):
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    tex = to_latex(df, caption, label)
    with open(os.path.join(OUT, f"{name}.tex"), "w") as f:
        f.write(tex)


# =====================================================================
# B1: Aggregator Leaderboard (FH, averaged over all conditions)
# =====================================================================
def table_b1():
    df = pd.read_csv(FH_SWEEP)
    agg = df.groupby("aggregator").agg(
        utility=("trimmed_mean_utility", "mean"),
        ci_lo=("ci_low_utility", "mean"),
        ci_hi=("ci_high_utility", "mean"),
        fairness=("trimmed_mean_fairness", "mean"),
        worst_group=("mean_worst_group", "mean"),
        regret=("mean_regret", "mean"),
    ).reset_index()
    agg = sort_agg(agg)

    # Rank by utility
    agg["rank"] = agg["utility"].rank(ascending=False).astype(int)

    out = pd.DataFrame()
    out["Rank"] = agg["rank"]
    out["Aggregator"] = agg["aggregator"].map(AGG_SHORT)
    out["Utility"] = agg["utility"].map(lambda x: fmt(x))
    out["90% CI"] = agg.apply(lambda r: f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]", axis=1)
    out["Fairness"] = agg["fairness"].map(lambda x: fmt(x))
    out["Worst-Grp"] = agg["worst_group"].map(lambda x: fmt(x))
    out["Regret"] = agg["regret"].map(lambda x: fmt(x))
    out = out.sort_values("Rank")

    save(out, "B1_aggregator_leaderboard",
         "Equitas FH benchmark: aggregator leaderboard (all conditions averaged, trimmed means).",
         "tab:fh-leaderboard")
    print("B1: Aggregator Leaderboard")
    print(out.to_string(index=False))
    print()


# =====================================================================
# B2: Performance by corruption rate (FH)
# =====================================================================
def table_b2():
    df = pd.read_csv(FH_SWEEP)

    # Pivot: rows = aggregator, columns = corruption rate
    pivot = df.pivot_table(
        index="aggregator", columns="corruption_rate",
        values="trimmed_mean_utility", aggfunc="mean",
    )
    pivot = pivot.reset_index()
    pivot = sort_agg(pivot)

    # Robustness = utility at 0.75 / utility at 0.00
    pivot["robustness"] = pivot[0.75] / pivot[0.0]

    out = pd.DataFrame()
    out["Aggregator"] = pivot["aggregator"].map(AGG_SHORT)
    out["ε=0.00"] = pivot[0.0].map(lambda x: fmt(x))
    out["ε=0.25"] = pivot[0.25].map(lambda x: fmt(x))
    out["ε=0.50"] = pivot[0.5].map(lambda x: fmt(x))
    out["ε=0.75"] = pivot[0.75].map(lambda x: fmt(x))
    out["Robustness"] = pivot["robustness"].map(lambda x: fmt(x, 3))

    save(out, "B2_utility_by_corruption",
         "FH utility by corruption rate $\\varepsilon$ (averaged over adversary types).",
         "tab:fh-corruption")
    print("B2: Utility by Corruption Rate")
    print(out.to_string(index=False))
    print()


# =====================================================================
# B3: Performance by adversary type (FH)
# =====================================================================
def table_b3():
    df = pd.read_csv(FH_SWEEP)

    pivot = df.pivot_table(
        index="aggregator", columns="adversary_type",
        values="trimmed_mean_utility", aggfunc="mean",
    )
    pivot = pivot.reset_index()
    pivot = sort_agg(pivot)

    out = pd.DataFrame()
    out["Aggregator"] = pivot["aggregator"].map(AGG_SHORT)
    for adv in ["selfish", "coordinated", "scheduled", "deceptive"]:
        out[adv.capitalize()] = pivot[adv].map(lambda x: fmt(x))

    save(out, "B3_utility_by_adversary",
         "FH utility by adversary type (averaged over corruption rates).",
         "tab:fh-adversary")
    print("B3: Utility by Adversary Type")
    print(out.to_string(index=False))
    print()


# =====================================================================
# B4: Regime winners (FH) — best aggregator per (adversary, rate, metric)
# =====================================================================
def table_b4():
    det = pd.read_csv(FH_REGIME)

    # Welfare metric only for main table
    w = det[det["metric"] == "welfare"].copy()
    w = w.sort_values(["adversary_type", "corruption_rate"])

    out = pd.DataFrame()
    out["Adversary"] = w["adversary_type"]
    out["ε"] = w["corruption_rate"]
    out["Best"] = w["best_aggregator"].map(lambda x: AGG_SHORT.get(x, x))
    out["Utility"] = w["value"].map(lambda x: fmt(x))
    out["Runner-up"] = w["runner_up"].map(lambda x: AGG_SHORT.get(x, x))
    out["Margin"] = w["margin"].map(lambda x: fmt(x))

    save(out, "B4_regime_winners_welfare",
         "FH regime winners: best aggregator by welfare (utility).",
         "tab:fh-regime-welfare")
    print("B4: Regime Winners (Welfare)")
    print(out.to_string(index=False))
    print()

    # Also fairness + worst-group
    for metric_name in ["fairness", "worst_group"]:
        m = det[det["metric"] == metric_name].copy()
        m = m.sort_values(["adversary_type", "corruption_rate"])
        mout = pd.DataFrame()
        mout["Adversary"] = m["adversary_type"]
        mout["ε"] = m["corruption_rate"]
        mout["Best"] = m["best_aggregator"].map(lambda x: AGG_SHORT.get(x, x))
        mout["Value"] = m["value"].map(lambda x: fmt(x))
        mout["Runner-up"] = m["runner_up"].map(lambda x: AGG_SHORT.get(x, x))
        mout["Margin"] = m["margin"].map(lambda x: fmt(x))
        save(mout, f"B4b_regime_winners_{metric_name}",
             f"FH regime winners: best aggregator by {metric_name.replace('_', ' ')}.",
             f"tab:fh-regime-{metric_name}")


# =====================================================================
# B5: Recovery from corruption onset (FH)
# =====================================================================
def table_b5():
    df = pd.read_csv(FH_RECOV)
    onset = int(df["onset_round"].iloc[0])
    total_rounds = df["round_id"].max() + 1

    pre = df[df["round_id"] < onset]
    post = df[df["round_id"] >= onset]

    pre_agg = pre.groupby("aggregator").agg(
        pre_util=("city_utility", "mean"),
        pre_fair=("fairness_jain", "mean"),
    ).reset_index()
    post_agg = post.groupby("aggregator").agg(
        post_util=("city_utility", "mean"),
        post_fair=("fairness_jain", "mean"),
    ).reset_index()

    merged = pre_agg.merge(post_agg, on="aggregator")
    merged["util_drop"] = merged["post_util"] - merged["pre_util"]
    merged["fair_drop"] = merged["post_fair"] - merged["pre_fair"]
    merged = sort_agg(merged)

    # Recovery speed: how many rounds post-onset until utility stabilizes
    # (we use last 5 rounds as "recovered" baseline)
    last5 = df[df["round_id"] >= total_rounds - 5]
    late_agg = last5.groupby("aggregator").agg(late_util=("city_utility", "mean")).reset_index()
    merged = merged.merge(late_agg, on="aggregator")
    merged["recovery_gap"] = merged["late_util"] - merged["pre_util"]

    out = pd.DataFrame()
    out["Aggregator"] = merged["aggregator"].map(AGG_SHORT)
    out["Pre-onset U"] = merged["pre_util"].map(lambda x: fmt(x))
    out["Post-onset U"] = merged["post_util"].map(lambda x: fmt(x))
    out["U Drop"] = merged["util_drop"].map(lambda x: sfmt(x))
    out["Late U"] = merged["late_util"].map(lambda x: fmt(x))
    out["Recovery Gap"] = merged["recovery_gap"].map(lambda x: sfmt(x))
    out["Fair Drop"] = merged["fair_drop"].map(lambda x: sfmt(x))

    save(out, "B5_recovery",
         f"FH recovery: utility before/after corruption onset (round {onset}/{total_rounds}).",
         "tab:fh-recovery")
    print(f"B5: Recovery (onset round {onset})")
    print(out.to_string(index=False))
    print()


# =====================================================================
# B6: Scaling by committee size (FH)
# =====================================================================
def table_b6():
    df = pd.read_csv(FH_SCALE)

    # Per-aggregator × size
    agg_size = df.groupby(["members_per_class", "aggregator"]).agg(
        utility=("mean_utility", "mean"),
        fairness=("mean_fairness", "mean"),
        worst_group=("mean_worst_group", "mean"),
    ).reset_index()

    # Pivot: rows = aggregator, columns = N
    pivot = agg_size.pivot_table(
        index="aggregator", columns="members_per_class",
        values="utility", aggfunc="mean",
    ).reset_index()
    pivot = sort_agg(pivot)

    sizes = sorted(df["members_per_class"].unique())
    out = pd.DataFrame()
    out["Aggregator"] = pivot["aggregator"].map(AGG_SHORT)
    for n in sizes:
        out[f"N={n}"] = pivot[n].map(lambda x: fmt(x))

    # Scaling slope: (utility at max N) - (utility at min N)
    pivot["slope"] = pivot[sizes[-1]] - pivot[sizes[0]]
    out["Δ(N_max−N_min)"] = pivot["slope"].map(lambda x: sfmt(x))

    save(out, "B6_scaling",
         "FH utility by committee size $N$ (members per class), $\\varepsilon=0.5$.",
         "tab:fh-scaling")
    print("B6: Scaling")
    print(out.to_string(index=False))
    print()


# =====================================================================
# B7: Hierarchical vs Flat (FH)
# =====================================================================
def table_b7():
    df = pd.read_csv(FH_HVF)

    hier = df[df["architecture"] == "fh_hierarchical"]
    flat = df[df["architecture"] == "flat"]

    h = hier.groupby(["corruption_rate", "aggregator"]).agg(
        h_util=("mean_utility", "mean"),
        h_fair=("mean_fairness", "mean"),
        h_wg=("mean_worst_group", "mean"),
    ).reset_index()
    f = flat.groupby(["corruption_rate", "aggregator"]).agg(
        f_util=("mean_utility", "mean"),
        f_fair=("mean_fairness", "mean"),
        f_wg=("mean_worst_group", "mean"),
    ).reset_index()

    # Average over aggregators per corruption rate
    h_cr = h.groupby("corruption_rate").mean(numeric_only=True).reset_index()
    f_cr = f.groupby("corruption_rate").mean(numeric_only=True).reset_index()
    m = h_cr.merge(f_cr, on="corruption_rate")
    m["util_gap"] = m["h_util"] - m["f_util"]
    m["fair_gap"] = m["h_fair"] - m["f_fair"]

    out = pd.DataFrame()
    out["ε"] = m["corruption_rate"]
    out["Hier Utility"] = m["h_util"].map(lambda x: fmt(x))
    out["Flat Utility"] = m["f_util"].map(lambda x: fmt(x))
    out["Δ Utility"] = m["util_gap"].map(lambda x: sfmt(x))
    out["Hier Fairness"] = m["h_fair"].map(lambda x: fmt(x))
    out["Flat Fairness"] = m["f_fair"].map(lambda x: fmt(x))
    out["Δ Fairness"] = m["fair_gap"].map(lambda x: sfmt(x))

    save(out, "B7_hier_vs_flat",
         "FH hierarchy vs flat architecture by corruption rate.",
         "tab:fh-hier-flat")
    print("B7: Hierarchical vs Flat")
    print(out.to_string(index=False))
    print()

    # Per-aggregator breakdown at high corruption
    h75 = h[h["corruption_rate"] == 0.75][["aggregator", "h_util", "h_fair"]].copy()
    f75 = f[f["corruption_rate"] == 0.75][["aggregator", "f_util", "f_fair"]].copy()
    high = h75.merge(f75, on="aggregator")
    high["gap"] = high["h_util"] - high["f_util"]
    high = sort_agg(high)
    detail = pd.DataFrame()
    detail["Aggregator"] = high["aggregator"].map(AGG_SHORT)
    detail["Hier U"] = high["h_util"].map(lambda x: fmt(x))
    detail["Flat U"] = high["f_util"].map(lambda x: fmt(x))
    detail["Gap"] = high["gap"].map(lambda x: sfmt(x))
    save(detail, "B7b_hier_vs_flat_detail_075",
         "FH hierarchy vs flat per aggregator at $\\varepsilon=0.75$.",
         "tab:fh-hier-flat-detail")


# =====================================================================
# B8: Pareto welfare-fairness frontier (FH, MW only)
# =====================================================================
def table_b8():
    df = pd.read_csv(FH_PARETO)
    mw = df[df["aggregator"] == "multiplicative_weights"]

    p = mw.groupby(["alpha", "beta"]).agg(
        utility=("mean_utility", "mean"),
        fairness=("mean_fairness", "mean"),
        worst_group=("mean_worst_group", "mean"),
    ).reset_index()

    out = pd.DataFrame()
    out["α"] = p["alpha"]
    out["β"] = p["beta"]
    out["Utility"] = p["utility"].map(lambda x: fmt(x))
    out["Fairness"] = p["fairness"].map(lambda x: fmt(x))
    out["Worst-Grp"] = p["worst_group"].map(lambda x: fmt(x))

    save(out, "B8_pareto_mw",
         "FH Pareto frontier: MW utility--fairness over $(\\alpha, \\beta)$ grid.",
         "tab:fh-pareto")
    print("B8: Pareto (MW)")
    print(out.to_string(index=False))
    print()

    # Pareto-optimal points
    pts = p[["alpha", "beta", "utility", "fairness"]].copy()
    pts = pts.sort_values("utility", ascending=False)
    # Simple Pareto filter: keep if no other point dominates on BOTH utility AND fairness
    optimal = []
    for _, row in pts.iterrows():
        dominated = False
        for _, other in pts.iterrows():
            if other["utility"] > row["utility"] and other["fairness"] > row["fairness"]:
                dominated = True
                break
        if not dominated:
            optimal.append(row)
    opt_df = pd.DataFrame(optimal)
    opt_out = pd.DataFrame()
    opt_out["α"] = opt_df["alpha"]
    opt_out["β"] = opt_df["beta"]
    opt_out["Utility"] = opt_df["utility"].map(lambda x: fmt(x))
    opt_out["Fairness"] = opt_df["fairness"].map(lambda x: fmt(x))
    save(opt_out, "B8b_pareto_frontier_points",
         "FH Pareto-optimal $(\\alpha, \\beta)$ configurations for MW.",
         "tab:fh-pareto-frontier")
    print(f"  Pareto-optimal points: {len(opt_out)}")
    print()


# =====================================================================
# D1: GO vs FH gap analysis (diagnostic / appendix)
# =====================================================================
def table_d1():
    go_sw = pd.read_csv(GO_SWEEP)
    fh_sw = pd.read_csv(FH_SWEEP)

    go_agg = go_sw.groupby("aggregator").agg(
        go_u=("trimmed_mean_utility", "mean"),
        go_f=("trimmed_mean_fairness", "mean"),
    ).reset_index()
    fh_agg = fh_sw.groupby("aggregator").agg(
        fh_u=("trimmed_mean_utility", "mean"),
        fh_f=("trimmed_mean_fairness", "mean"),
    ).reset_index()
    m = go_agg.merge(fh_agg, on="aggregator")
    m["Δu"] = m["fh_u"] - m["go_u"]
    m["Δf"] = m["fh_f"] - m["go_f"]
    m = sort_agg(m)

    out = pd.DataFrame()
    out["Aggregator"] = m["aggregator"].map(AGG_SHORT)
    out["GO Utility"] = m["go_u"].map(lambda x: fmt(x))
    out["FH Utility"] = m["fh_u"].map(lambda x: fmt(x))
    out["Δ Utility"] = m["Δu"].map(lambda x: sfmt(x))
    out["GO Fairness"] = m["go_f"].map(lambda x: fmt(x))
    out["FH Fairness"] = m["fh_f"].map(lambda x: fmt(x))
    out["Δ Fairness"] = m["Δf"].map(lambda x: sfmt(x))

    save(out, "D1_go_vs_fh_gap",
         "Diagnostic: GO vs FH gap per aggregator (sweep, all conditions averaged). "
         "GO can overstate robustness by hiding intra-class differences.",
         "tab:go-vs-fh-gap")
    print("D1: GO vs FH Gap (Diagnostic)")
    print(out.to_string(index=False))
    print()

    # High-corruption breakdown (where gap matters most)
    go75 = go_sw[go_sw["corruption_rate"] == 0.75].groupby("aggregator").agg(
        go_u=("trimmed_mean_utility", "mean")).reset_index()
    fh75 = fh_sw[fh_sw["corruption_rate"] == 0.75].groupby("aggregator").agg(
        fh_u=("trimmed_mean_utility", "mean")).reset_index()
    m75 = go75.merge(fh75, on="aggregator")
    m75["gap"] = m75["fh_u"] - m75["go_u"]
    m75 = sort_agg(m75)

    out75 = pd.DataFrame()
    out75["Aggregator"] = m75["aggregator"].map(AGG_SHORT)
    out75["GO (ε=0.75)"] = m75["go_u"].map(lambda x: fmt(x))
    out75["FH (ε=0.75)"] = m75["fh_u"].map(lambda x: fmt(x))
    out75["Gap"] = m75["gap"].map(lambda x: sfmt(x))
    save(out75, "D1b_go_vs_fh_high_corruption",
         "GO vs FH gap at high corruption ($\\varepsilon=0.75$). "
         "FH reveals method differences hidden by GO.",
         "tab:go-vs-fh-high")
    print("D1b: GO vs FH at ε=0.75")
    print(out75.to_string(index=False))
    print()


# =====================================================================
# D2: GO vs FH grand summary (one row per experiment)
# =====================================================================
def table_d2():
    rows = []

    go_sw = pd.read_csv(GO_SWEEP)
    fh_sw = pd.read_csv(FH_SWEEP)
    rows.append({"Experiment": "Sweep",
                 "GO Util": go_sw["trimmed_mean_utility"].mean(),
                 "FH Util": fh_sw["trimmed_mean_utility"].mean(),
                 "GO Fair": go_sw["trimmed_mean_fairness"].mean(),
                 "FH Fair": fh_sw["trimmed_mean_fairness"].mean()})

    go_r = pd.read_csv(GO_RECOV)
    fh_r = pd.read_csv(FH_RECOV)
    rows.append({"Experiment": "Recovery",
                 "GO Util": go_r["city_utility"].mean(),
                 "FH Util": fh_r["city_utility"].mean(),
                 "GO Fair": go_r["fairness_jain"].mean(),
                 "FH Fair": fh_r["fairness_jain"].mean()})

    go_sc = pd.read_csv(GO_SCALE)
    fh_sc = pd.read_csv(FH_SCALE)
    rows.append({"Experiment": "Scaling",
                 "GO Util": go_sc["mean_utility"].mean(),
                 "FH Util": fh_sc["mean_utility"].mean(),
                 "GO Fair": go_sc["mean_fairness"].mean(),
                 "FH Fair": fh_sc["mean_fairness"].mean()})

    go_hf = pd.read_csv(GO_HVF)
    fh_hf = pd.read_csv(FH_HVF)
    go_h = go_hf[go_hf["architecture"] == "hierarchical"]
    fh_h = fh_hf[fh_hf["architecture"] == "fh_hierarchical"]
    rows.append({"Experiment": "Hier vs Flat",
                 "GO Util": go_h["mean_utility"].mean(),
                 "FH Util": fh_h["mean_utility"].mean(),
                 "GO Fair": go_h["mean_fairness"].mean(),
                 "FH Fair": fh_h["mean_fairness"].mean()})

    go_p = pd.read_csv(GO_PARETO)
    fh_p = pd.read_csv(FH_PARETO)
    go_mw = go_p[go_p["aggregator"] == "multiplicative_weights"]
    fh_mw = fh_p[fh_p["aggregator"] == "multiplicative_weights"]
    rows.append({"Experiment": "Pareto (MW)",
                 "GO Util": go_mw["mean_utility"].mean(),
                 "FH Util": fh_mw["mean_utility"].mean(),
                 "GO Fair": go_mw["mean_fairness"].mean(),
                 "FH Fair": fh_mw["mean_fairness"].mean()})

    df = pd.DataFrame(rows)
    df["Δ Util"] = df["FH Util"] - df["GO Util"]
    df["Δ Fair"] = df["FH Fair"] - df["GO Fair"]

    out = pd.DataFrame()
    out["Experiment"] = df["Experiment"]
    out["GO Util"] = df["GO Util"].map(lambda x: fmt(x))
    out["FH Util"] = df["FH Util"].map(lambda x: fmt(x))
    out["Δ Util"] = df["Δ Util"].map(lambda x: sfmt(x))
    out["GO Fair"] = df["GO Fair"].map(lambda x: fmt(x))
    out["FH Fair"] = df["FH Fair"].map(lambda x: fmt(x))
    out["Δ Fair"] = df["Δ Fair"].map(lambda x: sfmt(x))

    save(out, "D2_go_vs_fh_grand_summary",
         "GO vs FH grand summary across all experiments. "
         "GO overstates convergence; FH reveals true aggregator separation.",
         "tab:go-vs-fh-grand")
    print("D2: GO vs FH Grand Summary")
    print(out.to_string(index=False))
    print()


# =====================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("EQUITAS FH BENCHMARK TABLES")
    print("=" * 80)
    print()

    print("── MAIN BENCHMARK TABLES (FH) ──\n")
    table_b1()
    table_b2()
    table_b3()
    table_b4()
    table_b5()
    table_b6()
    table_b7()
    table_b8()

    print("── DIAGNOSTIC TABLES (GO vs FH) ──\n")
    table_d1()
    table_d2()

    print("=" * 80)
    n_csv = len([f for f in os.listdir(OUT) if f.endswith(".csv")])
    n_tex = len([f for f in os.listdir(OUT) if f.endswith(".tex")])
    print(f"All tables saved to {OUT}/  ({n_csv} CSV + {n_tex} LaTeX)")
    print("=" * 80)
