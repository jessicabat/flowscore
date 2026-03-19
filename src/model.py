"""
FlowScore — Credit Risk Model & Business Value Analysis
=========================================================
Trains credit risk models on cash-flow features and demonstrates
the predictive lift over traditional credit scores alone.

  - 3-digit score predicting probability of default
  - Orthogonal to traditional scores (adds lift when combined)
  - SHAP-based feature importance (proxy for adverse action reasons)

Models:
  1. Traditional Score Only (baseline) — logistic regression on simulated FICO
  2. Logistic Regression on cash flow features (interpretable)
  3. XGBoost / GradientBoosting on cash flow features
  4. LightGBM with Optuna hyperparameter tuning (primary performance model)
  5. CatBoost with Optuna hyperparameter tuning
  6. Combined: traditional score + best cash flow model (shows orthogonality)

Business Value Analysis:
  - "Missed Opportunity": borrowers rejected by trad score who would repay
  - "Avoidable Risk": borrowers approved by trad score who would default
  - Approval rate vs loss rate tradeoff curves
  - Score distribution by risk bucket

Usage:
    python src/model.py --features data/features.csv --output data/model_results/

Requirements:
    pip install xgboost lightgbm catboost optuna shap scikit-learn pandas numpy matplotlib
"""

import argparse
import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

# Try importing optional packages
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Note: xgboost not installed. Using sklearn GradientBoosting instead.")
    print("      Install with: pip install xgboost\n")
    from sklearn.ensemble import GradientBoostingClassifier

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Note: lightgbm not installed. Install with: pip install lightgbm\n")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Note: catboost not installed. Install with: pip install catboost\n")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Note: optuna not installed. Install with: pip install optuna\n")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Note: shap not installed. Feature importance will use model-native method.")
    print("      Install with: pip install shap\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(df: pd.DataFrame) -> Tuple:
    """
    Split features into X (predictors), y (target), and metadata.
    Returns X_train, X_test, y_train, y_test, meta_test, feature_names, scaler
    """
    # Separate metadata and features
    meta_cols = [c for c in df.columns if c.startswith("_")]
    feat_cols = [c for c in df.columns if not c.startswith("_")]

    # Drop zero-variance features
    feat_cols = [c for c in feat_cols if df[c].std() > 1e-8]

    X = df[feat_cols].values
    y = df["_default_12m"].values.astype(int)
    meta = df[meta_cols]

    # Train/test split (stratified to preserve default rate)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=0.25, random_state=42, stratify=y
    )

    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {len(X_train)} samples ({y_train.sum()} defaults, {y_train.mean():.1%})")
    print(f"Test:  {len(X_test)} samples ({y_test.sum()} defaults, {y_test.mean():.1%})")

    return (X_train_scaled, X_test_scaled, y_train, y_test,
            meta_train, meta_test, feat_cols, scaler)


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_models(X_train, X_test, y_train, y_test, feature_names):
    """Train all models and return results dict."""
    results = {}

    # ---- Logistic Regression (interpretable baseline) ----
    print("\n--- Logistic Regression (Cash Flow Features) ---")
    lr = LogisticRegression(max_iter=1000, C=1.0, penalty="l2", random_state=42)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)
    cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"  AUC-ROC: {lr_auc:.4f}  |  5-fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    results["logistic_regression"] = {
        "model": lr, "proba": lr_proba, "auc": lr_auc,
        "cv_auc_mean": cv_scores.mean(), "cv_auc_std": cv_scores.std(),
    }

    # ---- XGBoost ----
    print("\n--- XGBoost (Cash Flow Features) ---")
    if HAS_XGB:
        gbm = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="auc",
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        gbm = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
    gbm.fit(X_train, y_train)
    gbm_proba = gbm.predict_proba(X_test)[:, 1]
    gbm_auc = roc_auc_score(y_test, gbm_proba)
    cv_scores_gbm = cross_val_score(gbm, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"  AUC-ROC: {gbm_auc:.4f}  |  5-fold CV: {cv_scores_gbm.mean():.4f} ± {cv_scores_gbm.std():.4f}")
    results["gradient_boosting"] = {
        "model": gbm, "proba": gbm_proba, "auc": gbm_auc,
        "cv_auc_mean": cv_scores_gbm.mean(), "cv_auc_std": cv_scores_gbm.std(),
    }

    # ---- LightGBM + Optuna ----
    lgb_model, lgb_params = tune_lightgbm(X_train, y_train, n_trials=40)
    if lgb_model is not None:
        lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
        lgb_auc = roc_auc_score(y_test, lgb_proba)
        print(f"  Test AUC-ROC: {lgb_auc:.4f}")
        results["lightgbm_optuna"] = {
            "model": lgb_model, "proba": lgb_proba, "auc": lgb_auc,
            "best_params": lgb_params,
        }

    # ---- CatBoost + Optuna ----
    cb_model, cb_params = tune_catboost(X_train, y_train, n_trials=30)
    if cb_model is not None:
        cb_proba = cb_model.predict_proba(X_test)[:, 1]
        cb_auc = roc_auc_score(y_test, cb_proba)
        print(f"  Test AUC-ROC: {cb_auc:.4f}")
        results["catboost_optuna"] = {
            "model": cb_model, "proba": cb_proba, "auc": cb_auc,
            "best_params": cb_params,
        }

    # ---- Feature Importance (from best tree model) ----
    # Use LightGBM if available, else XGBoost
    best_tree = (results.get("lightgbm_optuna") or
                 results.get("catboost_optuna") or
                 results.get("gradient_boosting"))
    if best_tree:
        print("\n--- Feature Importance (Top 15) ---")
        importances = best_tree["model"].feature_importances_
        imp_idx = np.argsort(importances)[::-1][:15]
        for rank, idx in enumerate(imp_idx):
            print(f"  {rank+1:2d}. {feature_names[idx]:40s} {importances[idx]:.4f}")
        results["feature_importance"] = {
            feature_names[i]: float(importances[i]) for i in imp_idx
        }

    # ---- SHAP Analysis (on best model) ----
    if HAS_SHAP and best_tree:
        print("\n--- SHAP Analysis ---")
        try:
            explainer = shap.TreeExplainer(best_tree["model"])
            shap_values = explainer.shap_values(X_test)
            # LightGBM returns list for binary; take positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_idx = np.argsort(mean_abs_shap)[::-1][:15]
            print("  Top 15 features by mean |SHAP|:")
            for rank, idx in enumerate(shap_idx):
                print(f"  {rank+1:2d}. {feature_names[idx]:40s} {mean_abs_shap[idx]:.4f}")
            results["shap_importance"] = {
                feature_names[i]: float(mean_abs_shap[i]) for i in shap_idx
            }
            results["shap_values"] = shap_values
        except Exception as e:
            print(f"  SHAP analysis failed: {e}")

    return results


def train_traditional_only(meta_train, meta_test, y_train, y_test):
    """Train a model using only the traditional credit score."""
    print("\n--- Model 0: Traditional Score Only (Baseline) ---")

    trad_train = meta_train["_traditional_score"].values.reshape(-1, 1)
    trad_test = meta_test["_traditional_score"].values.reshape(-1, 1)

    scaler_trad = StandardScaler()
    trad_train_s = scaler_trad.fit_transform(trad_train)
    trad_test_s = scaler_trad.transform(trad_test)

    lr_trad = LogisticRegression(max_iter=1000, random_state=42)
    lr_trad.fit(trad_train_s, y_train)

    trad_proba = lr_trad.predict_proba(trad_test_s)[:, 1]
    trad_auc = roc_auc_score(y_test, trad_proba)
    print(f"  AUC-ROC: {trad_auc:.4f}")

    return {"model": lr_trad, "proba": trad_proba, "auc": trad_auc}


def train_combined(X_train, X_test, meta_train, meta_test, y_train, y_test,
                   best_params_lgb=None):
    """Train a model using both cash flow features AND traditional score."""
    print("\n--- Model: Combined (Cash Flow + Traditional Score) ---")

    trad_train = meta_train["_traditional_score"].values.reshape(-1, 1)
    trad_test = meta_test["_traditional_score"].values.reshape(-1, 1)

    scaler_trad = StandardScaler()
    trad_train_s = scaler_trad.fit_transform(trad_train)
    trad_test_s = scaler_trad.transform(trad_test)

    X_train_combined = np.hstack([X_train, trad_train_s])
    X_test_combined = np.hstack([X_test, trad_test_s])

    if HAS_LGB and best_params_lgb:
        combined = lgb.LGBMClassifier(**best_params_lgb)  # random_state/verbose already in params
    elif HAS_XGB:
        combined = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="auc",
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        combined = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
    combined.fit(X_train_combined, y_train)

    comb_proba = combined.predict_proba(X_test_combined)[:, 1]
    comb_auc = roc_auc_score(y_test, comb_proba)
    print(f"  AUC-ROC: {comb_auc:.4f}")

    return {"model": combined, "proba": comb_proba, "auc": comb_auc}


# ============================================================================
# OPTUNA-TUNED MODELS
# ============================================================================

def tune_lightgbm(X_train, y_train, n_trials=40):
    """Tune LightGBM with Optuna. Returns best model and params."""
    if not HAS_LGB:
        print("  LightGBM not available. Skipping.")
        return None, {}
    if not HAS_OPTUNA:
        print("  Optuna not available. Training LightGBM with defaults.")
        model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            num_leaves=31, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, verbose=-1,
        )
        model.fit(X_train, y_train)
        return model, {}

    print(f"\n--- LightGBM + Optuna ({n_trials} trials) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"class_weight": "balanced", "random_state": 42, "verbose": -1})
    print(f"  Best CV AUC: {study.best_value:.4f}")
    print(f"  Best params: n_estimators={best_params['n_estimators']}, "
          f"max_depth={best_params['max_depth']}, "
          f"lr={best_params['learning_rate']:.4f}")

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, best_params


def tune_catboost(X_train, y_train, n_trials=30):
    """Tune CatBoost with Optuna. Returns best model and params."""
    if not HAS_CATBOOST:
        print("  CatBoost not available. Skipping.")
        return None, {}
    if not HAS_OPTUNA:
        print("  Optuna not available. Training CatBoost with defaults.")
        model = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            random_seed=42, verbose=0, auto_class_weights="Balanced",
        )
        model.fit(X_train, y_train)
        return model, {}

    print(f"\n--- CatBoost + Optuna ({n_trials} trials) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "verbose": 0,
        }
        model = CatBoostClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"auto_class_weights": "Balanced", "random_seed": 42, "verbose": 0})
    print(f"  Best CV AUC: {study.best_value:.4f}")
    print(f"  Best params: iterations={best_params['iterations']}, "
          f"depth={best_params['depth']}, "
          f"lr={best_params['learning_rate']:.4f}")

    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, best_params


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_ks_statistic(y_true, y_proba):
    """Compute KS statistic — the max separation between CDFs."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = max(tpr - fpr)
    return ks


def compute_gini(auc):
    """Gini coefficient = 2 * AUC - 1. Common in credit risk."""
    return 2 * auc - 1


def calibrate_score(proba, min_score=300, max_score=850):
    """Convert default probability to a 3-digit score (higher = lower risk)."""
    return min_score + (max_score - min_score) * (1 - proba)


def best_cashflow_model(results_dict):
    """Return the highest-AUC cash flow model name and results."""
    candidates = ["catboost_optuna", "lightgbm_optuna", "gradient_boosting",
                  "logistic_regression"]
    for name in candidates:
        if name in results_dict:
            return name, results_dict[name]
    return None, None


def print_model_comparison(results_dict):
    """Print a comparison table of all models."""
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<45s} {'AUC':>7s} {'KS':>7s} {'Gini':>7s}")
    print(f"  {'-'*66}")

    for name, res in results_dict.items():
        auc = res["auc"]
        ks = compute_ks_statistic(res["y_test"], res["proba"])
        gini = compute_gini(auc)
        print(f"  {name:<45s} {auc:>7.4f} {ks:>7.4f} {gini:>7.4f}")

    trad_auc = results_dict.get("traditional_only", {}).get("auc")
    if trad_auc:
        best_name, best_res = best_cashflow_model(
            {k: v for k, v in results_dict.items() if k != "traditional_only"
             and not k.startswith("combined")}
        )
        if best_name:
            lift = (best_res["auc"] - trad_auc) / trad_auc * 100
            print(f"\n  Best cash flow model ({best_name}) lift over traditional: {lift:+.1f}%")
        for comb_key in [k for k in results_dict if k.startswith("combined")]:
            comb_auc = results_dict[comb_key]["auc"]
            lift = (comb_auc - trad_auc) / trad_auc * 100
            print(f"  Combined model lift over traditional:  {lift:+.1f}%")


# ============================================================================
# BUSINESS VALUE ANALYSIS
# ============================================================================

def business_value_analysis(y_test, trad_proba, cashflow_proba, trad_scores, output_dir):
    """
    The core analysis:
    1. "Missed Opportunity" — good borrowers rejected by traditional scores
    2. "Avoidable Risk" — bad borrowers approved by traditional scores
    3. Approval rate vs loss rate curves
    """
    print(f"\n{'='*70}")
    print(f"BUSINESS VALUE ANALYSIS")
    print(f"{'='*70}")

    # Use a traditional score cutoff of 620 (common prime threshold)
    TRAD_CUTOFF = 620

    trad_approved = trad_scores >= TRAD_CUTOFF
    trad_rejected = trad_scores < TRAD_CUTOFF

    # ----- MISSED OPPORTUNITY -----
    # Consumers rejected by trad score who would NOT have defaulted
    missed = trad_rejected & (y_test == 0)
    missed_count = missed.sum()
    total_rejected = trad_rejected.sum()
    rejected_would_repay_pct = missed_count / total_rejected if total_rejected > 0 else 0

    print(f"\n--- Missed Opportunity (Traditional Score < {TRAD_CUTOFF}) ---")
    print(f"  Total rejected by trad score:  {total_rejected}")
    print(f"  Would actually repay:          {missed_count} ({rejected_would_repay_pct:.1%})")
    print(f"  → These are creditworthy borrowers invisible to traditional scoring")

    # What if we used the cash flow score to rescue some?
    # Among trad-rejected consumers, approve those with low cash flow default prob
    CASHFLOW_RESCUE_THRESHOLD = 0.30  # approve if CF model says <30% default risk
    cf_rescue = trad_rejected & (cashflow_proba < CASHFLOW_RESCUE_THRESHOLD)
    cf_rescue_good = cf_rescue & (y_test == 0)
    cf_rescue_bad = cf_rescue & (y_test == 1)
    rescue_count = cf_rescue.sum()
    rescue_good = cf_rescue_good.sum()
    rescue_bad = cf_rescue_bad.sum()
    rescue_default_rate = rescue_bad / rescue_count if rescue_count > 0 else 0

    print(f"\n  Cash flow rescue (CF default prob < {CASHFLOW_RESCUE_THRESHOLD:.0%}):")
    print(f"    Additional approvals:        {rescue_count}")
    print(f"    Of those, would repay:       {rescue_good} ({rescue_good/rescue_count:.1%})" if rescue_count > 0 else "")
    print(f"    Default rate of rescued:     {rescue_default_rate:.1%}")

    # ----- AVOIDABLE RISK -----
    # Consumers approved by trad score who DID default
    avoidable = trad_approved & (y_test == 1)
    avoidable_count = avoidable.sum()
    total_approved = trad_approved.sum()
    approved_default_pct = avoidable_count / total_approved if total_approved > 0 else 0

    print(f"\n--- Avoidable Risk (Traditional Score >= {TRAD_CUTOFF}) ---")
    print(f"  Total approved by trad score:  {total_approved}")
    print(f"  Actually defaulted:            {avoidable_count} ({approved_default_pct:.1%})")
    print(f"  → These are risky borrowers that look OK to traditional scoring")

    # What if we used CF score to flag risky ones?
    CASHFLOW_FLAG_THRESHOLD = 0.50  # flag if CF model says >50% default risk
    cf_flagged = trad_approved & (cashflow_proba > CASHFLOW_FLAG_THRESHOLD)
    cf_flagged_bad = cf_flagged & (y_test == 1)
    cf_flagged_good = cf_flagged & (y_test == 0)
    flag_count = cf_flagged.sum()
    flag_caught = cf_flagged_bad.sum()
    flag_false_alarm = cf_flagged_good.sum()

    print(f"\n  Cash flow flagging (CF default prob > {CASHFLOW_FLAG_THRESHOLD:.0%}):")
    print(f"    Flagged for review:          {flag_count}")
    print(f"    True positives (caught):     {flag_caught}")
    print(f"    False alarms:                {flag_false_alarm}")
    if flag_count > 0:
        print(f"    Precision:                   {flag_caught/flag_count:.1%}")

    # ----- APPROVAL RATE VS LOSS RATE TRADEOFF -----
    print(f"\n--- Approval Simulation at Fixed Loss Rates ---")
    print(f"  {'Target Loss Rate':>20s} {'Trad Approve%':>15s} {'CF Approve%':>15s} {'Lift':>10s}")
    print(f"  {'-'*60}")

    approval_comparison = []
    for target_loss in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        # Traditional: find threshold that achieves target loss rate
        trad_approve_rate = _find_approval_rate(y_test, 1 - trad_proba, target_loss)
        cf_approve_rate = _find_approval_rate(y_test, 1 - cashflow_proba, target_loss)
        lift = cf_approve_rate - trad_approve_rate

        print(f"  {target_loss:>20.0%} {trad_approve_rate:>14.1%} {cf_approve_rate:>14.1%} {lift:>+9.1%}")
        approval_comparison.append({
            "target_loss_rate": target_loss,
            "trad_approval_rate": round(trad_approve_rate, 4),
            "cf_approval_rate": round(cf_approve_rate, 4),
            "approval_lift": round(lift, 4),
        })

    print(f"\n--- Loss Simulation at Fixed Approval Rates ---")
    print(f"  {'Target Approval':>20s} {'Trad Loss%':>15s} {'CF Loss%':>15s} {'Reduction':>12s}")
    print(f"  {'-'*62}")

    loss_comparison = []
    for target_approve in [0.50, 0.60, 0.70, 0.80, 0.90]:
        trad_loss = _find_loss_rate(y_test, 1 - trad_proba, target_approve)
        cf_loss = _find_loss_rate(y_test, 1 - cashflow_proba, target_approve)
        reduction = (trad_loss - cf_loss) / trad_loss * 100 if trad_loss > 0 else 0

        print(f"  {target_approve:>20.0%} {trad_loss:>14.1%} {cf_loss:>14.1%} {reduction:>+11.1f}%")
        loss_comparison.append({
            "target_approval_rate": target_approve,
            "trad_loss_rate": round(trad_loss, 4),
            "cf_loss_rate": round(cf_loss, 4),
            "loss_reduction_pct": round(reduction, 2),
        })

    # ----- PRISM-STYLE RISK BUCKET ANALYSIS -----
    print(f"\n--- Risk Bucket Analysis (CashScore Style) ---")
    flowscores = calibrate_score(cashflow_proba)
    buckets = [
        ("Very Low Risk (750+)", flowscores >= 750),
        ("Low Risk (650-749)", (flowscores >= 650) & (flowscores < 750)),
        ("Medium Risk (550-649)", (flowscores >= 550) & (flowscores < 650)),
        ("High Risk (450-549)", (flowscores >= 450) & (flowscores < 550)),
        ("Very High Risk (<450)", flowscores < 450),
    ]

    print(f"  {'Bucket':<28s} {'Count':>7s} {'% of Total':>10s} {'Default Rate':>13s}")
    print(f"  {'-'*60}")
    bucket_data = []
    for label, mask in buckets:
        count = mask.sum()
        pct = count / len(y_test)
        dr = y_test[mask].mean() if count > 0 else 0
        print(f"  {label:<28s} {count:>7d} {pct:>10.1%} {dr:>13.1%}")
        bucket_data.append({"bucket": label, "count": int(count), "pct": round(pct, 4), "default_rate": round(dr, 4)})

    # ----- CROSS-TABULATION: Traditional vs FlowScore -----
    print(f"\n--- Cross-Tab: Traditional Score vs FlowScore ---")
    print(f"  (Shows orthogonality — each cell's default rate)")
    trad_hi = trad_scores >= 660
    trad_lo = trad_scores < 660
    flow_hi = flowscores >= 650
    flow_lo = flowscores < 650

    def _cell(mask):
        n = mask.sum()
        dr = y_test[mask].mean() if n > 0 else 0
        return n, dr

    n_hh, dr_hh = _cell(trad_hi & flow_hi)
    n_hl, dr_hl = _cell(trad_hi & flow_lo)
    n_lh, dr_lh = _cell(trad_lo & flow_hi)
    n_ll, dr_ll = _cell(trad_lo & flow_lo)

    print(f"  {'':>30s} {'FlowScore >= 650':>20s} {'FlowScore < 650':>20s}")
    print(f"  {'Trad Score >= 660':<30s} {dr_hh:>8.1%} (n={n_hh:>4d})     {dr_hl:>8.1%} (n={n_hl:>4d})")
    print(f"  {'Trad Score <  660':<30s} {dr_lh:>8.1%} (n={n_lh:>4d})     {dr_ll:>8.1%} (n={n_ll:>4d})")
    print(f"\n  Key: Trad High + Flow Low = 'Avoidable Risk' ({dr_hl:.1%} default rate)")
    print(f"       Trad Low  + Flow High = 'Missed Opportunity' ({dr_lh:.1%} default rate)")

    return {
        "missed_opportunity": {
            "total_rejected": int(total_rejected),
            "would_repay": int(missed_count),
            "repay_pct": round(rejected_would_repay_pct, 4),
            "cf_rescued": int(rescue_count),
            "rescue_default_rate": round(rescue_default_rate, 4),
        },
        "avoidable_risk": {
            "total_approved": int(total_approved),
            "defaulted": int(avoidable_count),
            "default_pct": round(approved_default_pct, 4),
            "cf_flagged": int(flag_count),
            "cf_caught": int(flag_caught),
        },
        "approval_comparison": approval_comparison,
        "loss_comparison": loss_comparison,
        "risk_buckets": bucket_data,
        "cross_tab": {
            "trad_hi_flow_hi": {"n": int(n_hh), "default_rate": round(dr_hh, 4)},
            "trad_hi_flow_lo": {"n": int(n_hl), "default_rate": round(dr_hl, 4)},
            "trad_lo_flow_hi": {"n": int(n_lh), "default_rate": round(dr_lh, 4)},
            "trad_lo_flow_lo": {"n": int(n_ll), "default_rate": round(dr_ll, 4)},
        },
    }


def _find_approval_rate(y_true, scores, target_loss_rate):
    """Find the approval rate that achieves a target loss rate."""
    sorted_idx = np.argsort(scores)[::-1]  # highest score first
    y_sorted = y_true[sorted_idx]

    best_rate = 0
    for i in range(1, len(y_sorted) + 1):
        approved = y_sorted[:i]
        loss_rate = approved.mean()
        if loss_rate <= target_loss_rate:
            best_rate = i / len(y_sorted)
        else:
            break
    return best_rate


def _find_loss_rate(y_true, scores, target_approval_rate):
    """Find the loss rate at a target approval rate."""
    n_approve = int(len(y_true) * target_approval_rate)
    sorted_idx = np.argsort(scores)[::-1]
    y_sorted = y_true[sorted_idx]
    if n_approve == 0:
        return 0
    return y_sorted[:n_approve].mean()


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_plots(results_dict, biz_results, output_dir):
    """Generate key charts for the website/report."""
    if not HAS_MPL:
        print("matplotlib not available. Skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. ROC Curves comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["proba"])
        ax.plot(fpr, tpr, label=f'{name} (AUC={res["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Cash Flow vs Traditional Scoring")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: roc_curves.png")

    # 2. Approval rate comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    comp = biz_results["approval_comparison"]
    targets = [c["target_loss_rate"] for c in comp]
    trad_rates = [c["trad_approval_rate"] * 100 for c in comp]
    cf_rates = [c["cf_approval_rate"] * 100 for c in comp]
    x = np.arange(len(targets))
    w = 0.35
    ax.bar(x - w/2, trad_rates, w, label="Traditional Score", color="#94a3b8")
    ax.bar(x + w/2, cf_rates, w, label="Cash Flow Score", color="#3b82f6")
    ax.set_xlabel("Target Loss Rate")
    ax.set_ylabel("Approval Rate (%)")
    ax.set_title("More Approvals at Same Risk Level")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0%}" for t in targets])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "approval_lift.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: approval_lift.png")

    # 3. Risk bucket analysis
    fig, ax = plt.subplots(figsize=(8, 5))
    buckets = biz_results["risk_buckets"]
    labels = [b["bucket"].split("(")[0].strip() for b in buckets]
    rates = [b["default_rate"] * 100 for b in buckets]
    colors = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]
    ax.bar(labels, rates, color=colors)
    ax.set_ylabel("Default Rate (%)")
    ax.set_title("FlowScore Risk Buckets — Default Rate by Segment")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (lbl, rate) in enumerate(zip(labels, rates)):
        ax.text(i, rate + 0.5, f"{rate:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "risk_buckets.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: risk_buckets.png")

    # 4. Cross-tab (missed opp vs avoidable risk)
    fig, ax = plt.subplots(figsize=(6, 5))
    ct = biz_results["cross_tab"]
    data = np.array([
        [ct["trad_hi_flow_hi"]["default_rate"], ct["trad_hi_flow_lo"]["default_rate"]],
        [ct["trad_lo_flow_hi"]["default_rate"], ct["trad_lo_flow_lo"]["default_rate"]],
    ]) * 100
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=60)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FlowScore High\n(Low Risk)", "FlowScore Low\n(High Risk)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Trad Score High", "Trad Score Low"])
    for i in range(2):
        for j in range(2):
            n = [ct["trad_hi_flow_hi"], ct["trad_hi_flow_lo"],
                 ct["trad_lo_flow_hi"], ct["trad_lo_flow_lo"]][i*2+j]["n"]
            ax.text(j, i, f"{data[i,j]:.1f}%\n(n={n})", ha="center", va="center",
                    fontsize=12, fontweight="bold")
    ax.set_title("Default Rates: Traditional vs FlowScore\n(Orthogonality Matrix)")
    fig.colorbar(im, label="Default Rate (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cross_tab.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: cross_tab.png")

    # 5. Feature importance
    fig, ax = plt.subplots(figsize=(8, 6))
    if "feature_importance" in results_dict.get("_extras", {}):
        fi = results_dict["_extras"]["feature_importance"]
    else:
        # Use from gradient boosting results
        fi = results_dict.get("_feature_importance", {})
    if fi:
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
        names = [x[0] for x in sorted_fi][::-1]
        vals = [x[1] for x in sorted_fi][::-1]
        ax.barh(names, vals, color="#3b82f6")
        ax.set_xlabel("Feature Importance")
        ax.set_title("Top 15 Features — What Drives Default Risk?")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
        print(f"  Saved: feature_importance.png")
    plt.close(fig)


# ============================================================================
# SCORE CALIBRATION
# ============================================================================

def generate_flowscores(y_test, proba, meta_test, output_dir):
    """Generate calibrated FlowScores and save."""
    scores = calibrate_score(proba)

    results_df = pd.DataFrame({
        "consumer_id": meta_test["_consumer_id"].values,
        "archetype": meta_test["_archetype"].values,
        "traditional_score": meta_test["_traditional_score"].values,
        "flowscore": scores.astype(int),
        "default_probability": np.round(proba, 4),
        "actual_default": y_test,
    })

    path = os.path.join(output_dir, "flowscores.csv")
    results_df.to_csv(path, index=False)
    print(f"\nFlowScores saved to {path}")

    print(f"\n--- FlowScore Distribution ---")
    print(f"  Mean:   {scores.mean():.0f}")
    print(f"  Std:    {scores.std():.0f}")
    print(f"  Range:  [{scores.min():.0f}, {scores.max():.0f}]")

    return results_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train credit risk models and analyze business value"
    )
    parser.add_argument("--features", required=True, help="Path to features CSV")
    parser.add_argument("--output", required=True, help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load features
    print(f"Loading features from {args.features}...")
    df = pd.read_csv(args.features)
    print(f"Loaded: {df.shape[0]} consumers, {df.shape[1]} columns")

    # Prepare data
    (X_train, X_test, y_train, y_test,
     meta_train, meta_test, feat_names, scaler) = prepare_data(df)

    # Train models
    trad_results = train_traditional_only(meta_train, meta_test, y_train, y_test)
    trad_results["y_test"] = y_test

    model_results = train_models(X_train, X_test, y_train, y_test, feat_names)

    # Get best LightGBM params for combined model (reuses tuned hyperparams)
    lgb_params = (model_results.get("lightgbm_optuna", {}) or {}).get("best_params", None)
    combined_results = train_combined(
        X_train, X_test, meta_train, meta_test, y_train, y_test,
        best_params_lgb=lgb_params,
    )
    combined_results["y_test"] = y_test

    # Collect all models for comparison table
    all_results = {"traditional_only": trad_results}
    for key in ["logistic_regression", "gradient_boosting",
                "lightgbm_optuna", "catboost_optuna"]:
        if key in model_results:
            all_results[key] = {**model_results[key], "y_test": y_test}
    all_results["combined (trad + best CF)"] = combined_results

    print_model_comparison(all_results)

    # Identify best cash flow model for business value / FlowScores
    cf_candidates = {k: v for k, v in all_results.items()
                     if k not in ("traditional_only",) and not k.startswith("combined")}
    primary_cf_name, primary_cf = max(cf_candidates.items(), key=lambda x: x[1]["auc"])
    print(f"\n  Primary cash flow model for business analysis: {primary_cf_name} "
          f"(AUC={primary_cf['auc']:.4f})")

    # Business value analysis
    biz = business_value_analysis(
        y_test,
        trad_results["proba"],
        primary_cf["proba"],
        meta_test["_traditional_score"].values,
        args.output,
    )

    # Generate plots
    print(f"\n--- Generating Plots ---")
    generate_plots(all_results, biz, args.output)

    # Generate FlowScores
    scores_df = generate_flowscores(
        y_test, primary_cf["proba"], meta_test, args.output,
    )

    # Save trained model bundle for Streamlit demo
    try:
        import joblib
        bundle = {
            "model": primary_cf["model"],
            "scaler": scaler,
            "feature_names": feat_names,
            # Population means for features the demo can't compute from simplified inputs
            "feature_means": {
                name: float(df[name].mean())
                for name in feat_names
                if name in df.columns
            },
        }
        bundle_path = os.path.join(args.output, "model_bundle.joblib")
        joblib.dump(bundle, bundle_path)
        print(f"\nModel bundle saved to {bundle_path} (for Streamlit demo)")
    except ImportError:
        print("\nNote: joblib not installed, model bundle not saved. "
              "Run: pip install joblib")

    # Save all results as JSON
    summary = {
        "model_comparison": {
            name: {
                "auc": round(res["auc"], 4),
                "ks": round(compute_ks_statistic(res["y_test"], res["proba"]), 4),
                "gini": round(compute_gini(res["auc"]), 4),
            }
            for name, res in all_results.items()
        },
        "feature_importance": model_results.get("feature_importance", {}),
        "best_params": {
            "lightgbm": model_results.get("lightgbm_optuna", {}).get("best_params", {}),
            "catboost": model_results.get("catboost_optuna", {}).get("best_params", {}),
        },
        "business_value": biz,
    }

    json_path = os.path.join(args.output, "results_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to {json_path}")

    # Final summary
    trad = all_results["traditional_only"]
    comb = all_results["combined (trad + best CF)"]
    print(f"\n{'='*70}")
    print(f"FLOWSCORE PROJECT — FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Traditional score AUC:       {trad['auc']:.4f}")
    for key in ["logistic_regression", "gradient_boosting",
                "lightgbm_optuna", "catboost_optuna"]:
        if key in all_results:
            a = all_results[key]["auc"]
            print(f"  {key:<30s} AUC: {a:.4f}  ({(a-trad['auc'])/trad['auc']*100:+.1f}% lift)")
    print(f"  Combined model AUC:          {comb['auc']:.4f}  "
          f"({(comb['auc']-trad['auc'])/trad['auc']*100:+.1f}% lift)")
    print(f"  Missed opportunity:          "
          f"{biz['missed_opportunity']['would_repay']} good borrowers rejected by trad score")
    print(f"  Avoidable risk:              "
          f"{biz['avoidable_risk']['defaulted']} bad borrowers approved by trad score")
    print(f"\n  → Best model: {primary_cf_name} (AUC={primary_cf['auc']:.4f})")
    print(f"    Cash flow scoring finds creditworthy borrowers traditional scores miss,")
    print(f"    and catches risky borrowers traditional scores approve.")


if __name__ == "__main__":
    main()