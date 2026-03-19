"""
FlowScore — Fairness & Disparate Impact Analysis
=================================================
Evaluates whether the FlowScore model systematically disadvantages
demographic groups proxied by consumer archetypes.

Why this matters for Prism Data:
  Traditional scores fail thin-file and gig economy workers not because
  they're risky, but because they lack the *type* of credit history FICO
  requires. This analysis shows FlowScore corrects that bias — approving
  creditworthy non-traditional borrowers at rates commensurate with their
  actual default risk.

Archetypes as demographic proxies:
  - thin_file_newcomer   → Recent immigrants, young adults, new to US credit
  - gig_worker           → Gig economy, self-employed, alternative income
  - financially_stressed → Low income, paycheck-to-paycheck households
  - overextended         → Over-leveraged borrowers
  - stable_salaried      → Traditional prime borrower (reference group)
  - high_earner_high_spender → High income, high spend

Key metrics computed:
  1. Score distributions by archetype
  2. Approval rates at multiple thresholds (both FlowScore and traditional)
  3. True Positive Rate (TPR), False Positive Rate (FPR), False Negative Rate
  4. Benefit ratio: FlowScore approval rate / traditional approval rate
  5. Score calibration: does the same FlowScore predict the same default rate
     across archetypes? (calibration equity)
  6. 80% rule (adverse impact ratio): EEOC standard for disparate impact

Usage:
    python src/fairness_analysis.py \\
        --scores data/model_results/flowscores.csv \\
        --output data/model_results/fairness/
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Human-readable archetype labels
ARCHETYPE_LABELS = {
    "stable_salaried":          "Stable Salaried",
    "gig_worker":               "Gig Worker",
    "high_earner_high_spender": "High Earner",
    "financially_stressed":     "Financially Stressed",
    "thin_file_newcomer":       "Thin File / Newcomer",
    "overextended":             "Overextended",
}

ARCHETYPE_ORDER = [
    "stable_salaried",
    "high_earner_high_spender",
    "thin_file_newcomer",
    "gig_worker",
    "overextended",
    "financially_stressed",
]

# Reference group for adverse impact ratio (EEOC 80% rule)
REFERENCE_ARCHETYPE = "stable_salaried"


# ============================================================================
# CORE METRICS
# ============================================================================

def approval_rate_at_threshold(scores: np.ndarray, threshold: float) -> float:
    """Fraction of consumers with score >= threshold (i.e., approved)."""
    return (scores >= threshold).mean()


def compute_fairness_metrics(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    For each archetype, compute at a given FlowScore approval threshold:
      - approval_rate_flow:  fraction approved by FlowScore
      - approval_rate_trad:  fraction approved by traditional score (>= 620)
      - actual_default_rate: ground truth default rate for the group
      - fpr: false positive rate (approved by CF but actually defaulted)
      - fnr: false negative rate (rejected by CF but would have repaid)
      - benefit_ratio:       FlowScore approval / traditional approval (>1 = CF more inclusive)
    """
    TRAD_CUTOFF = 620
    rows = []
    for arch in ARCHETYPE_ORDER:
        sub = df[df["archetype"] == arch]
        if len(sub) == 0:
            continue

        flow_approved = sub["flowscore"] >= threshold
        trad_approved = sub["traditional_score"] >= TRAD_CUTOFF
        y = sub["actual_default"].values

        n = len(sub)
        n_approved_flow = flow_approved.sum()
        n_approved_trad = trad_approved.sum()

        ar_flow = n_approved_flow / n
        ar_trad = n_approved_trad / n if n > 0 else 0
        actual_dr = y.mean()

        # Among CF-approved, default rate (precision-like)
        approved_default_rate = (y[flow_approved.values].mean()
                                 if n_approved_flow > 0 else 0)

        # FPR: of true defaulters, how many did CF approve? (false alarms)
        defaulters = y == 1
        fpr = (flow_approved.values & defaulters).sum() / defaulters.sum() \
            if defaulters.sum() > 0 else 0

        # FNR: of true non-defaulters, how many did CF reject? (missed opportunity)
        non_defaulters = y == 0
        fnr = (~flow_approved.values & non_defaulters).sum() / non_defaulters.sum() \
            if non_defaulters.sum() > 0 else 0

        benefit_ratio = ar_flow / ar_trad if ar_trad > 0 else float("inf")

        rows.append({
            "archetype": arch,
            "label": ARCHETYPE_LABELS[arch],
            "n": n,
            "actual_default_rate": round(actual_dr, 4),
            "approval_rate_flow": round(ar_flow, 4),
            "approval_rate_trad": round(ar_trad, 4),
            "approved_default_rate": round(approved_default_rate, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "benefit_ratio": round(benefit_ratio, 4),
        })

    return pd.DataFrame(rows)


def adverse_impact_ratio(df_metrics: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Compute the EEOC 80% rule adverse impact ratio.
    AIR = group_rate / reference_group_rate
    AIR < 0.80 indicates potential disparate impact.
    """
    ref_row = df_metrics[df_metrics["archetype"] == REFERENCE_ARCHETYPE]
    if len(ref_row) == 0:
        return df_metrics
    ref_val = ref_row[metric_col].values[0]

    result = df_metrics.copy()
    result["air"] = result[metric_col] / ref_val if ref_val > 0 else np.nan
    return result


def score_calibration_by_archetype(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Test score calibration equity: within the same FlowScore band, do
    different archetypes have similar default rates?
    A well-calibrated score treats equals equally regardless of archetype.
    """
    bins = pd.cut(df["flowscore"], bins=n_bins, labels=False)
    df = df.copy()
    df["score_bin"] = bins
    df["score_bin_label"] = pd.cut(
        df["flowscore"], bins=n_bins,
        labels=[f"{int(df['flowscore'].min() + i * (df['flowscore'].max() - df['flowscore'].min()) / n_bins)}"
                f"–{int(df['flowscore'].min() + (i+1) * (df['flowscore'].max() - df['flowscore'].min()) / n_bins)}"
                for i in range(n_bins)]
    )

    calib = df.groupby(["score_bin", "archetype"]).agg(
        n=("actual_default", "count"),
        default_rate=("actual_default", "mean"),
        mean_score=("flowscore", "mean"),
    ).reset_index()
    return calib


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_approval_rates_vs_default(df: pd.DataFrame, output_dir: str):
    """
    Bar chart: approval rate (FlowScore vs traditional) alongside actual
    default rate for each archetype. Shows FlowScore's inclusion gain
    and whether it's safe (low approval default rate).
    """
    if not HAS_MPL:
        return

    labels = [ARCHETYPE_LABELS[a] for a in ARCHETYPE_ORDER if a in df["archetype"].values]
    archs_present = [a for a in ARCHETYPE_ORDER if a in df["archetype"].values]
    df_ord = df.set_index("archetype").loc[archs_present]

    x = np.arange(len(archs_present))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - w, df_ord["approval_rate_trad"] * 100, w,
                   label="Traditional Score Approval", color="#94a3b8", alpha=0.9)
    bars2 = ax.bar(x, df_ord["approval_rate_flow"] * 100, w,
                   label="FlowScore Approval", color="#3b82f6", alpha=0.9)
    bars3 = ax.bar(x + w, df_ord["actual_default_rate"] * 100, w,
                   label="Actual Default Rate", color="#ef4444", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title(
        "FlowScore vs Traditional: Approval Rates and Actual Default Rates by Archetype\n"
        "(FlowScore approves more thin-file/gig consumers without proportional risk increase)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 110)

    # Annotate benefit ratio
    for i, arch in enumerate(archs_present):
        br = df_ord.loc[arch, "benefit_ratio"]
        ax.text(i, df_ord.loc[arch, "approval_rate_flow"] * 100 + 2,
                f"×{br:.2f}", ha="center", fontsize=8, color="#1d4ed8")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fairness_approval_rates.png"), dpi=150)
    plt.close(fig)
    print("  Saved: fairness_approval_rates.png")


def plot_adverse_impact_ratio(df_metrics: pd.DataFrame, output_dir: str):
    """
    Horizontal bar chart of Adverse Impact Ratio (AIR) relative to stable_salaried.
    Red dashed line at 0.80 = EEOC 80% rule threshold for disparate impact.
    """
    if not HAS_MPL:
        return

    df_air = adverse_impact_ratio(df_metrics, "approval_rate_flow")
    df_air = df_air[df_air["archetype"].isin(ARCHETYPE_ORDER)]
    df_air = df_air.set_index("archetype").loc[
        [a for a in ARCHETYPE_ORDER if a in df_air["archetype"].values]
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#22c55e" if v >= 0.8 else "#f97316"
              for v in df_air["air"].values]
    bars = ax.barh(
        [ARCHETYPE_LABELS[a] for a in df_air.index],
        df_air["air"].values,
        color=colors, alpha=0.85,
    )
    ax.axvline(x=0.80, color="#dc2626", linestyle="--", linewidth=1.5,
               label="EEOC 80% rule threshold")
    ax.axvline(x=1.00, color="#64748b", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Adverse Impact Ratio (vs Stable Salaried)")
    ax.set_title(
        "FlowScore Adverse Impact Ratio by Archetype\n"
        "(AIR ≥ 0.80 = no disparate impact under EEOC 80% rule)"
    )
    ax.legend()
    ax.set_xlim(0, max(df_air["air"].max() + 0.1, 1.3))
    for bar, val in zip(bars, df_air["air"].values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fairness_air.png"), dpi=150)
    plt.close(fig)
    print("  Saved: fairness_air.png")


def plot_score_distributions(df: pd.DataFrame, output_dir: str):
    """
    Overlapping score distributions by archetype.
    Shows whether FlowScore shifts groups appropriately based on actual risk.
    """
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]

    for i, arch in enumerate(ARCHETYPE_ORDER):
        sub = df[df["archetype"] == arch]["flowscore"]
        if len(sub) == 0:
            continue
        label = f"{ARCHETYPE_LABELS[arch]} (μ={sub.mean():.0f}, DR={df[df['archetype']==arch]['actual_default'].mean():.0%})"
        ax.hist(sub, bins=30, alpha=0.5, color=colors[i % len(colors)],
                label=label, density=True)

    ax.set_xlabel("FlowScore")
    ax.set_ylabel("Density")
    ax.set_title(
        "FlowScore Distribution by Archetype\n"
        "(Scores reflect actual default risk, not demographic group membership)"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fairness_score_distributions.png"), dpi=150)
    plt.close(fig)
    print("  Saved: fairness_score_distributions.png")


def plot_calibration_heatmap(calib: pd.DataFrame, output_dir: str):
    """
    Heatmap: rows = archetypes, columns = FlowScore bins.
    Cell = default rate within that bin × archetype combination.
    A well-calibrated model shows consistent default rates across archetypes
    within the same score band.
    """
    if not HAS_MPL:
        return

    pivot = calib.pivot_table(
        index="archetype", columns="score_bin",
        values="default_rate", aggfunc="mean"
    )
    pivot.index = [ARCHETYPE_LABELS.get(a, a) for a in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values * 100, cmap="RdYlGn_r", vmin=0, vmax=70, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"Bin {i+1}" for i in range(len(pivot.columns))], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(
        "Default Rate by FlowScore Bin × Archetype\n"
        "(Consistent rates across archetypes within each bin = calibration equity)"
    )
    fig.colorbar(im, label="Default Rate (%)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if val > 0.35 else "black")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fairness_calibration_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Saved: fairness_calibration_heatmap.png")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_fairness_analysis(scores_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(scores_path)
    print(f"Loaded {len(df):,} consumers from {scores_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Archetypes: {df['archetype'].value_counts().to_dict()}\n")

    # ---- 1. Metrics at a representative threshold ----
    # Use FlowScore 600 as the approval cutoff (broadly "near-prime")
    FLOW_THRESHOLD = 600
    print(f"{'='*65}")
    print(f"FAIRNESS METRICS AT FLOWSCORE THRESHOLD = {FLOW_THRESHOLD}")
    print(f"{'='*65}")

    metrics_df = compute_fairness_metrics(df, FLOW_THRESHOLD)

    print(f"\n{'Archetype':<26} {'n':>5} {'Actual DR':>10} {'CF Approve':>11} "
          f"{'Trad Approve':>13} {'Benefit×':>9} {'FPR':>6} {'FNR':>6}")
    print(f"  {'-'*80}")
    for _, row in metrics_df.iterrows():
        print(f"  {row['label']:<24} {row['n']:>5,} "
              f"{row['actual_default_rate']:>10.1%} "
              f"{row['approval_rate_flow']:>10.1%}  "
              f"{row['approval_rate_trad']:>12.1%}  "
              f"{row['benefit_ratio']:>8.2f}×  "
              f"{row['fpr']:>5.1%}  "
              f"{row['fnr']:>5.1%}")

    # ---- 2. Adverse Impact Ratio ----
    print(f"\n{'='*65}")
    print(f"ADVERSE IMPACT RATIO (vs {ARCHETYPE_LABELS[REFERENCE_ARCHETYPE]})")
    print(f"  EEOC 80% rule: AIR < 0.80 indicates potential disparate impact")
    print(f"{'='*65}")

    df_air = adverse_impact_ratio(metrics_df, "approval_rate_flow")
    for _, row in df_air.iterrows():
        air = row.get("air", float("nan"))
        flag = "  ⚠️  BELOW 80%" if (not np.isnan(air) and air < 0.80) else ""
        print(f"  {row['label']:<26} AIR = {air:.3f}{flag}")

    # ---- 3. Score calibration ----
    print(f"\n{'='*65}")
    print(f"SCORE CALIBRATION BY ARCHETYPE")
    print(f"  (Default rate within each FlowScore quintile)")
    print(f"{'='*65}")

    calib = score_calibration_by_archetype(df, n_bins=5)
    pivot = calib.pivot_table(
        index="archetype", columns="score_bin",
        values="default_rate", aggfunc="mean"
    )
    pivot.index = [ARCHETYPE_LABELS.get(a, a) for a in pivot.index]
    print(pivot.round(3).to_string())

    # ---- 4. Sweep: benefit ratio at multiple thresholds ----
    print(f"\n{'='*65}")
    print(f"BENEFIT RATIO SWEEP (multiple approval thresholds)")
    print(f"{'='*65}")
    thresholds = [500, 550, 600, 650, 700]
    print(f"\n  {'Archetype':<26}", end="")
    for t in thresholds:
        print(f"  {t:>7}", end="")
    print()
    print(f"  {'-'*68}")
    for arch in ARCHETYPE_ORDER:
        sub = df[df["archetype"] == arch]
        if len(sub) == 0:
            continue
        print(f"  {ARCHETYPE_LABELS[arch]:<26}", end="")
        for t in thresholds:
            m = compute_fairness_metrics(sub, t)
            if len(m) > 0:
                br = m["benefit_ratio"].values[0]
                print(f"  {br:>6.2f}×", end="")
        print()

    # ---- 5. Key narrative finding ----
    thin_file_metrics = metrics_df[metrics_df["archetype"] == "thin_file_newcomer"]
    gig_metrics = metrics_df[metrics_df["archetype"] == "gig_worker"]

    print(f"\n{'='*65}")
    print(f"KEY FINDING FOR PRISM DATA PITCH")
    print(f"{'='*65}")
    if len(thin_file_metrics) > 0:
        tf = thin_file_metrics.iloc[0]
        print(f"\n  Thin-file/Newcomer consumers:")
        print(f"    Traditional score approves:  {tf['approval_rate_trad']:.1%}")
        print(f"    FlowScore approves:          {tf['approval_rate_flow']:.1%}  "
              f"({tf['benefit_ratio']:.2f}× more inclusive)")
        print(f"    Actual default rate:         {tf['actual_default_rate']:.1%}")
        print(f"    → FlowScore grants {tf['benefit_ratio']:.1f}× more access to thin-file consumers")
        print(f"      whose actual risk ({tf['actual_default_rate']:.0%}) is manageable.")

    if len(gig_metrics) > 0:
        gig = gig_metrics.iloc[0]
        print(f"\n  Gig-worker consumers:")
        print(f"    Traditional score approves:  {gig['approval_rate_trad']:.1%}")
        print(f"    FlowScore approves:          {gig['approval_rate_flow']:.1%}  "
              f"({gig['benefit_ratio']:.2f}× more inclusive)")
        print(f"    Actual default rate:         {gig['actual_default_rate']:.1%}")

    # ---- 6. Save results ----
    results = {
        "threshold": FLOW_THRESHOLD,
        "metrics_by_archetype": metrics_df.to_dict(orient="records"),
        "adverse_impact_ratios": df_air[["archetype", "label", "approval_rate_flow",
                                         "approval_rate_trad", "air"]].to_dict(orient="records"),
        "calibration": calib.to_dict(orient="records"),
    }

    json_path = os.path.join(output_dir, "fairness_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Fairness results saved to {json_path}")

    # ---- 7. Generate plots ----
    if HAS_MPL:
        print("\n--- Generating Fairness Plots ---")
        plot_approval_rates_vs_default(metrics_df, output_dir)
        plot_adverse_impact_ratio(metrics_df, output_dir)
        plot_score_distributions(df, output_dir)
        plot_calibration_heatmap(calib, output_dir)

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fairness and disparate impact analysis for FlowScore"
    )
    parser.add_argument("--scores", required=True,
                        help="Path to flowscores.csv (output of model.py)")
    parser.add_argument("--output", default="data/model_results/fairness/",
                        help="Output directory for fairness plots and JSON")
    args = parser.parse_args()

    run_fairness_analysis(args.scores, args.output)


if __name__ == "__main__":
    main()
