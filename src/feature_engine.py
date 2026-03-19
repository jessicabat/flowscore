"""
FlowScore — Feature Engineering Pipeline
==========================================
Transforms categorized transaction data into numeric features for credit risk modeling.
Usage:
    python src/feature_engine.py --input data/flowscore_dataset.json \
                                 --output data/features.csv

    # Or import in a notebook:
    from feature_engine import extract_features
    df = extract_features(consumers)
"""

import argparse
import json
import os
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================================
# CATEGORY GROUPS (used for feature computation)
# ============================================================================

INCOME_CATEGORIES = {"payroll", "gig_income", "government_benefits"}
HOUSING_CATEGORIES = {"rent", "mortgage"}
ESSENTIAL_CATEGORIES = {"rent", "mortgage", "utilities", "groceries", "insurance", "healthcare"}
DISCRETIONARY_CATEGORIES = {"dining", "food_delivery", "shopping", "travel", "subscription"}
OBLIGATION_CATEGORIES = {"rent", "mortgage", "loan_payment", "bnpl", "insurance"}
RED_FLAG_CATEGORIES = {"gambling", "fee", "payday_loan_deposit", "payday_loan_repayment"}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _get_month_key(date_str: str) -> str:
    """Return 'YYYY-MM' from a date string."""
    return date_str[:7]


def _linear_slope(values: List[float]) -> float:
    """Compute the slope of a simple linear regression over a sequence.
    Values are assumed to be evenly spaced (e.g., monthly)."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.array(values, dtype=float)
    # slope = cov(x,y) / var(x)
    x_mean = x.mean()
    y_mean = y.mean()
    var_x = ((x - x_mean) ** 2).sum()
    if var_x == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / var_x)


def _shannon_entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy of a category distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def _coefficient_of_variation(values: List[float]) -> float:
    """CV = std / mean. Higher = more volatile."""
    if len(values) < 2:
        return 0.0
    mean = np.mean(values)
    if abs(mean) < 1e-6:
        return 0.0
    return float(np.std(values) / abs(mean))


# ============================================================================
# MONTHLY AGGREGATION
# ============================================================================

def _aggregate_monthly(transactions: List[Dict]) -> Dict[str, Dict]:
    """
    Group transactions by month and compute monthly aggregates.
    Returns dict keyed by 'YYYY-MM' with:
      - total_inflow, total_outflow, net_cashflow
      - category_spend: {category: total_amount}
      - n_transactions
      - income_deposits: list of (amount, merchant, date) for income txns
    """
    months = defaultdict(lambda: {
        "total_inflow": 0.0,
        "total_outflow": 0.0,
        "category_spend": defaultdict(float),
        "category_count": defaultdict(int),
        "n_transactions": 0,
        "income_deposits": [],
        "dates": [],
    })

    for txn in transactions:
        mk = _get_month_key(txn["date"])
        m = months[mk]
        amt = txn["amount"]
        cat = txn["category"]

        m["n_transactions"] += 1
        m["dates"].append(txn["date"])

        if amt > 0:
            m["total_inflow"] += amt
            if cat in INCOME_CATEGORIES:
                m["income_deposits"].append({
                    "amount": amt,
                    "merchant": txn["merchant"],
                    "date": txn["date"],
                    "category": cat,
                })
        else:
            m["total_outflow"] += abs(amt)
            m["category_spend"][cat] += abs(amt)
            m["category_count"][cat] += 1

    # Compute net cash flow per month
    for mk, m in months.items():
        m["net_cashflow"] = m["total_inflow"] - m["total_outflow"]

    return dict(months)


# ============================================================================
# FEATURE EXTRACTION — ONE CONSUMER
# ============================================================================

def extract_consumer_features(consumer: Dict) -> Dict[str, float]:
    """
    Extract all features for a single consumer.
    Returns a flat dict of feature_name: value.
    """
    txns = consumer["transactions"]
    if not txns:
        return {}

    monthly = _aggregate_monthly(txns)
    sorted_months = sorted(monthly.keys())
    n_months = len(sorted_months)

    features = {}

    monthly_incomes = [monthly[m]["total_inflow"] for m in sorted_months]

    features["monthly_income_mean"] = float(np.mean(monthly_incomes)) if monthly_incomes else 0
    features["monthly_income_std"] = float(np.std(monthly_incomes)) if len(monthly_incomes) > 1 else 0
    features["income_cv"] = _coefficient_of_variation(monthly_incomes)

    # Income regularity: how consistent is the time between income deposits?
    all_income_dates = []
    income_sources = set()
    has_gig = False
    has_govt = False
    for m in sorted_months:
        for dep in monthly[m]["income_deposits"]:
            all_income_dates.append(_parse_date(dep["date"]))
            income_sources.add(dep["merchant"])
            if dep["category"] == "gig_income":
                has_gig = True
            if dep["category"] == "government_benefits":
                has_govt = True

    all_income_dates.sort()
    if len(all_income_dates) >= 2:
        gaps = [(all_income_dates[i+1] - all_income_dates[i]).days
                for i in range(len(all_income_dates) - 1)]
        features["income_regularity"] = float(np.std(gaps)) if gaps else 0
    else:
        features["income_regularity"] = 99.0  # very irregular / no data

    features["income_source_count"] = len(income_sources)
    features["has_gig_income"] = 1.0 if has_gig else 0.0
    features["has_government_benefits"] = 1.0 if has_govt else 0.0

    # Pay frequency score: detect biweekly (14-day gaps) vs monthly (30-day) vs irregular
    if len(all_income_dates) >= 3:
        median_gap = float(np.median(gaps))
        if 12 <= median_gap <= 16:
            features["payroll_frequency_score"] = 5.0  # biweekly, very regular
        elif 27 <= median_gap <= 33:
            features["payroll_frequency_score"] = 4.0  # monthly
        elif 6 <= median_gap <= 8:
            features["payroll_frequency_score"] = 4.0  # weekly
        else:
            features["payroll_frequency_score"] = 2.0  # irregular
    else:
        features["payroll_frequency_score"] = 1.0  # insufficient data

    features["income_trend_slope"] = _linear_slope(monthly_incomes)

    # Income-to-expense ratio
    total_income = sum(monthly_incomes)
    monthly_expenses = [monthly[m]["total_outflow"] for m in sorted_months]
    total_expenses = sum(monthly_expenses)
    features["income_to_expense_ratio"] = (
        total_income / total_expenses if total_expenses > 0 else 0
    )


    features["monthly_spend_mean"] = float(np.mean(monthly_expenses)) if monthly_expenses else 0
    features["monthly_spend_std"] = float(np.std(monthly_expenses)) if len(monthly_expenses) > 1 else 0

    # Discretionary vs essential ratio
    total_by_group = {"discretionary": 0, "essential": 0, "all": 0}
    category_totals = defaultdict(float)
    category_counts = defaultdict(int)
    for m in sorted_months:
        for cat, amt in monthly[m]["category_spend"].items():
            category_totals[cat] += amt
            total_by_group["all"] += amt
            if cat in DISCRETIONARY_CATEGORIES:
                total_by_group["discretionary"] += amt
            if cat in ESSENTIAL_CATEGORIES:
                total_by_group["essential"] += amt
        for cat, cnt in monthly[m]["category_count"].items():
            category_counts[cat] += cnt

    features["discretionary_ratio"] = (
        total_by_group["discretionary"] / total_by_group["all"]
        if total_by_group["all"] > 0 else 0
    )
    features["essential_ratio"] = (
        total_by_group["essential"] / total_by_group["all"]
        if total_by_group["all"] > 0 else 0
    )

    # Subscription analysis
    sub_merchants = set()
    sub_total = 0.0
    for txn in txns:
        if txn["category"] == "subscription" and txn["amount"] < 0:
            sub_merchants.add(txn["merchant"])
            sub_total += abs(txn["amount"])
    features["subscription_count"] = len(sub_merchants)
    features["subscription_total_monthly"] = sub_total / max(n_months, 1)

    # Transaction size stats
    all_amounts = [abs(txn["amount"]) for txn in txns if txn["amount"] < 0]
    if all_amounts:
        features["avg_transaction_size"] = float(np.mean(all_amounts))
        features["median_transaction_size"] = float(np.median(all_amounts))
    else:
        features["avg_transaction_size"] = 0
        features["median_transaction_size"] = 0

    features["spending_trend_slope"] = _linear_slope(monthly_expenses)

    # Category diversity (Shannon entropy)
    features["category_diversity"] = _shannon_entropy(
        {cat: int(cnt) for cat, cnt in category_counts.items()}
    )

    # Dining out ratio: how much food spend is restaurants vs groceries?
    dining_total = category_totals.get("dining", 0) + category_totals.get("food_delivery", 0)
    grocery_total = category_totals.get("groceries", 0)
    food_total = dining_total + grocery_total
    features["dining_out_ratio"] = dining_total / food_total if food_total > 0 else 0

    # Weekend spend ratio
    weekend_spend = 0.0
    weekday_spend = 0.0
    for txn in txns:
        if txn["amount"] < 0:
            day_of_week = _parse_date(txn["date"]).weekday()
            if day_of_week >= 5:  # Saturday=5, Sunday=6
                weekend_spend += abs(txn["amount"])
            else:
                weekday_spend += abs(txn["amount"])
    total_spend = weekend_spend + weekday_spend
    features["weekend_spend_ratio"] = weekend_spend / total_spend if total_spend > 0 else 0

    monthly_net = [monthly[m]["net_cashflow"] for m in sorted_months]
    features["net_monthly_cashflow_mean"] = float(np.mean(monthly_net)) if monthly_net else 0
    features["net_monthly_cashflow_std"] = float(np.std(monthly_net)) if len(monthly_net) > 1 else 0
    features["savings_rate"] = (
        features["net_monthly_cashflow_mean"] / features["monthly_income_mean"]
        if features["monthly_income_mean"] > 0 else 0
    )
    features["months_negative_cashflow"] = sum(1 for nf in monthly_net if nf < 0)
    features["min_monthly_cashflow"] = min(monthly_net) if monthly_net else 0
    features["cashflow_trend_slope"] = _linear_slope(monthly_net)

    # Estimated balance trend: cumulative sum of monthly net cash flow
    cumulative = np.cumsum(monthly_net)
    features["estimated_balance_trend"] = _linear_slope(list(cumulative)) if len(cumulative) > 1 else 0


    # Detect housing payment
    housing_amounts = []
    housing_dates = []
    for txn in txns:
        if txn["category"] in HOUSING_CATEGORIES and txn["amount"] < 0:
            housing_amounts.append(abs(txn["amount"]))
            housing_dates.append(_parse_date(txn["date"]))

    if housing_amounts:
        features["housing_payment_amount"] = float(np.median(housing_amounts))
        # Consistency: std dev of day-of-month for housing payments
        days_of_month = [d.day for d in housing_dates]
        features["housing_payment_consistency"] = float(np.std(days_of_month)) if len(days_of_month) > 1 else 0
    else:
        features["housing_payment_amount"] = 0
        features["housing_payment_consistency"] = 99.0  # no housing payment detected

    # Loan payments
    loan_merchants = set()
    loan_total = 0.0
    for txn in txns:
        if txn["category"] == "loan_payment" and txn["amount"] < 0:
            loan_merchants.add(txn["merchant"])
            loan_total += abs(txn["amount"])
    features["loan_payment_count"] = len(loan_merchants)

    # BNPL
    bnpl_merchants = set()
    bnpl_total = 0.0
    for txn in txns:
        if txn["category"] == "bnpl" and txn["amount"] < 0:
            bnpl_merchants.add(txn["merchant"])
            bnpl_total += abs(txn["amount"])
    features["bnpl_active"] = 1.0 if bnpl_merchants else 0.0
    features["bnpl_payment_count"] = len(bnpl_merchants)

    # Total monthly obligations
    monthly_obligations = (
        features["housing_payment_amount"] +
        loan_total / max(n_months, 1) +
        bnpl_total / max(n_months, 1) +
        category_totals.get("insurance", 0) / max(n_months, 1)
    )
    features["total_monthly_obligations"] = monthly_obligations

    # DTI proxy: obligations / income
    features["obligation_to_income_ratio"] = (
        monthly_obligations / features["monthly_income_mean"]
        if features["monthly_income_mean"] > 0 else 0
    )

    # Loan stacking: 3+ concurrent loan payments
    features["loan_stacking_flag"] = 1.0 if features["loan_payment_count"] >= 3 else 0.0

    overdraft_count = 0
    gambling_spend = 0.0
    gambling_count = 0
    payday_count = 0
    late_fee_count = 0
    fee_total = 0.0

    for txn in txns:
        cat = txn["category"]
        if cat == "fee":
            fee_total += abs(txn["amount"])
            if "OVERDRAFT" in txn["merchant"].upper() or "NSF" in txn["merchant"].upper():
                overdraft_count += 1
            if "LATE" in txn["merchant"].upper():
                late_fee_count += 1
        if cat == "gambling" and txn["amount"] < 0:
            gambling_spend += abs(txn["amount"])
            gambling_count += 1
        if cat in ("payday_loan_deposit", "payday_loan_repayment"):
            payday_count += 1

    features["overdraft_count"] = overdraft_count
    features["overdraft_frequency"] = overdraft_count / max(n_months, 1)
    features["gambling_total_spend"] = gambling_spend
    features["gambling_frequency"] = gambling_count / max(n_months, 1)
    features["payday_loan_flag"] = 1.0 if payday_count > 0 else 0.0
    features["payday_loan_count"] = payday_count
    features["late_fee_count"] = late_fee_count
    features["fee_total"] = fee_total

    features["_n_months"] = n_months
    features["_n_transactions"] = len(txns)
    features["_consumer_id"] = consumer["consumer_id"]
    features["_archetype"] = consumer.get("archetype", "unknown")
    features["_default_12m"] = consumer.get("default_12m", -1)
    features["_traditional_score"] = consumer.get("traditional_score", -1)

    return features


# ============================================================================
# BATCH EXTRACTION
# ============================================================================

def extract_features(consumers: List[Dict]) -> pd.DataFrame:
    """
    Extract features for a list of consumers.
    Returns a DataFrame with one row per consumer, features as columns.
    """
    records = []
    for i, consumer in enumerate(consumers):
        features = extract_consumer_features(consumer)
        records.append(features)
        if (i + 1) % 500 == 0:
            print(f"  Extracted features for {i+1}/{len(consumers)} consumers...")

    df = pd.DataFrame(records)

    # Separate metadata columns (prefixed with _)
    meta_cols = [c for c in df.columns if c.startswith("_")]
    feat_cols = [c for c in df.columns if not c.startswith("_")]

    print(f"\nExtracted {len(feat_cols)} features for {len(df)} consumers.")
    print(f"Metadata columns: {meta_cols}")

    return df


# ============================================================================
# FEATURE SUMMARY
# ============================================================================

def print_feature_summary(df: pd.DataFrame):
    """Print summary statistics for all features."""
    feat_cols = [c for c in df.columns if not c.startswith("_")]

    print(f"\n{'='*70}")
    print(f"FEATURE SUMMARY ({len(feat_cols)} features, {len(df)} consumers)")
    print(f"{'='*70}")

    # Group features by prefix
    groups = {
        "Income": [c for c in feat_cols if any(c.startswith(p) for p in
                   ["monthly_income", "income_", "has_gig", "has_gov", "payroll"])],
        "Spending": [c for c in feat_cols if any(c.startswith(p) for p in
                     ["monthly_spend", "discretionary", "essential", "subscription",
                      "avg_trans", "median_trans", "spending_trend", "category_div",
                      "dining_out", "weekend"])],
        "Balance & Savings": [c for c in feat_cols if any(c.startswith(p) for p in
                              ["net_monthly", "savings_rate", "months_neg", "min_monthly",
                               "cashflow_trend", "estimated_bal"])],
        "Obligations": [c for c in feat_cols if any(c.startswith(p) for p in
                        ["housing", "loan_payment", "bnpl", "total_monthly_oblig",
                         "obligation_to", "loan_stacking"])],
        "Red Flags": [c for c in feat_cols if any(c.startswith(p) for p in
                      ["overdraft", "gambling", "payday", "late_fee", "fee_total"])],
    }

    for group_name, cols in groups.items():
        if not cols:
            continue
        print(f"\n--- {group_name} ---")
        for col in cols:
            vals = df[col]
            print(f"  {col:40s}  mean={vals.mean():>10.2f}  std={vals.std():>10.2f}  "
                  f"min={vals.min():>10.2f}  max={vals.max():>10.2f}")

    # Default rate comparison by key features
    if "_default_12m" in df.columns:
        print(f"\n--- Feature Means by Default Status ---")
        defaults = df[df["_default_12m"] == 1]
        non_defaults = df[df["_default_12m"] == 0]

        key_features = [
            "income_cv", "savings_rate", "obligation_to_income_ratio",
            "overdraft_frequency", "gambling_total_spend", "payday_loan_flag",
            "months_negative_cashflow", "loan_stacking_flag", "income_trend_slope",
            "discretionary_ratio", "bnpl_active",
        ]
        print(f"  {'Feature':<35s}  {'No Default':>12s}  {'Default':>12s}  {'Diff':>10s}")
        print(f"  {'-'*72}")
        for feat in key_features:
            if feat in df.columns:
                nd_mean = non_defaults[feat].mean()
                d_mean = defaults[feat].mean()
                diff = d_mean - nd_mean
                print(f"  {feat:<35s}  {nd_mean:>12.4f}  {d_mean:>12.4f}  {diff:>+10.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract features from FlowScore dataset")
    parser.add_argument("--input", required=True, help="Path to dataset JSON")
    parser.add_argument("--output", required=True, help="Path to save features CSV")
    parser.add_argument("--n_consumers", type=int, default=None,
                        help="Limit number of consumers (default: all)")
    args = parser.parse_args()

    print(f"Loading dataset from {args.input}...")
    with open(args.input) as f:
        consumers = json.load(f)
    print(f"Loaded {len(consumers)} consumers.")

    if args.n_consumers:
        consumers = consumers[:args.n_consumers]
        print(f"Using first {len(consumers)} consumers.")

    print("Extracting features...")
    df = extract_features(consumers)

    print_feature_summary(df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved features to {args.output}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()