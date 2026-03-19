# FlowScore: Run Results and Performance Notes

A reference document capturing the full pipeline execution — decisions made, numbers produced, and what they mean.

---

## 1. Data Generation

```
Consumers:          5,000
Total transactions: 3,304,340
Avg txns/consumer:  661
Default rate:       25.1%
Traditional score:  mean=656, std=88, range=[378, 833]
```

### Archetype Breakdown

| Archetype | N | % | Default Rate |
|---|---|---|---|
| stable_salaried | 1,541 | 30.8% | 8.4% |
| financially_stressed | 711 | 14.2% | **60.2%** |
| gig_worker | 743 | 14.9% | 30.4% |
| high_earner_high_spender | 765 | 15.3% | 10.8% |
| overextended | 631 | 12.6% | **46.9%** |
| thin_file_newcomer | 609 | 12.2% | 14.9% |

**Why it matters**: The two highest-default archetypes (financially_stressed, overextended) are exactly the population where cash flow signals should shine — and where traditional scores may be blind or misleading.

---

## 2. Categorization

### Three conditions tested

#### Condition A: Clean merchant names, rules only
- Accuracy: **99.2%**
- Rule coverage: 100%
- Only failure: `groceries -> shopping` (47 misclassifications)

#### Condition B: Noisy merchant names, rules only
- Accuracy: **82.3%** — a 17-point drop
- Roughly 1,040 transactions fall to "other" when rules can't match
- Top failures: dining, food_delivery, transportation all leaked into "other"
- This is the realistic baseline without any augmentation

#### Condition C: Noisy merchant names, rules + Claude Sonnet (original approach)
- Accuracy: **95.7%**
- LLM handled 1,040 ambiguous transactions
- API cost: **$0.362** for 6,209 transactions (~$0.058/1,000 txns)

### DistilBERT categorizer (new)

Fine-tuned distilbert-base-uncased on (merchant_name, category) pairs extracted from the synthetic dataset.

```
Training pairs:       ~70,000 (after balancing: max 4,000/class, min 300/class)
Classes:              22 (amount-direction variants collapsed during training,
                         re-split at inference via _apply_amount_logic())
Training accuracy:    99.7% on clean merchant strings (test split)
Noisy data accuracy:  91.5% (train on clean, test on noise-corrupted names)
Inference cost:       $0 (fully local, no API)
```

Amount-direction classes (gambling_win, payday_loan_deposit, payday_loan_repayment) cannot be learned from merchant text alone — they depend on transaction sign. These are collapsed to base classes during training and re-split at inference using the actual transaction amount.

**Training note (Run 1 failure):** The first training attempt without the amount-direction collapse produced 0% F1 for gambling_win and payday_loan_deposit, because DistilBERT cannot distinguish "DRAFTKINGS" for a win vs. a bet from text alone. Collapsing these categories before training fixed the issue and accuracy jumped from 94.3% to 99.7%.

**Train/test distribution note:** The 91.5% noisy accuracy (vs 95.7% for Claude hybrid) reflects a distribution gap — DistilBERT was trained on clean strings and tested on noise-corrupted variants. Adding noise augmentation during training would narrow this gap. The DistilBERT approach eliminates API cost and latency at the cost of ~4 percentage points on noisy data.

---

## 3. Feature Engineering

45 features extracted across 5 categories.

| Feature | No Default | Default | Delta |
|---|---|---|---|
| savings_rate | +0.012 | **-0.323** | -0.336 |
| obligation_to_income_ratio | 0.348 | **0.473** | +0.125 |
| months_negative_cashflow | 3.6 | **6.2** | +2.6 |
| loan_stacking_flag | 0.079 | **0.266** | +0.187 |
| bnpl_active | 0.233 | **0.416** | +0.183 |
| payday_loan_flag | 0.028 | **0.124** | +0.096 |
| overdraft_frequency | 0.062 | **0.155** | +0.093 |
| income_trend_slope | +54.6 | **+5.4** | -49.2 |

Notable: `late_fee_count` was zero across all 5,000 consumers — a data generation gap worth revisiting.

---

## 4. Model Performance

Train/test split: 75/25, stratified. All models evaluated on the same 1,250 test consumers.

| Model | AUC-ROC | KS | Gini | AUC Lift |
|---|---|---|---|---|
| Traditional score only (baseline) | 0.6869 | 0.3419 | 0.3738 | baseline |
| Logistic Regression (cash flow) | 0.7613 | 0.4520 | 0.5226 | +10.8% |
| XGBoost (cash flow) | 0.7317 | 0.4115 | 0.4634 | +6.5% |
| LightGBM + Optuna | 0.7611 | 0.4542 | 0.5222 | +10.8% |
| **CatBoost + Optuna** | **0.7637** | **0.4404** | **0.5273** | **+11.2%** |
| Combined (traditional + best CF) | 0.7617 | 0.4521 | 0.5233 | +10.9% |

**Best model: CatBoost + Optuna (AUC 0.7637, +11.2% lift over traditional)**

### Why tree ensembles are competitive here but don't dominate

Logistic Regression and LightGBM both reach 0.761x — only slightly below CatBoost at 0.7637. The features are well-engineered linear signals (ratios, slopes, counts), which benefits LR. CatBoost's symmetric tree structure and built-in ordered boosting give it a small edge on the synthetic data. On real production data with noisier, less structured features, the gap between tree ensembles and LR would likely widen in favor of trees.

### Top Features by Model Importance (CatBoost)

| Feature | Importance |
|---|---|
| income_regularity | 152 |
| obligation_to_income_ratio | 147 |
| estimated_balance_trend | 133 |
| net_monthly_cashflow_std | 118 |
| subscription_total_monthly | 108 |
| overdraft_frequency | 102 |
| spending_trend_slope | 79 |
| total_monthly_obligations | 78 |
| avg_transaction_size | 68 |
| monthly_spend_mean | 68 |

---

## 5. Business Value Analysis

### Missed Opportunity (Traditional Score below 620)

- Borrowers rejected by traditional scoring: **465**
- Of those, would actually repay: **285 (61.3%)**
- Cash flow model rescues 66 of them with default probability below 30%
  - Rescued group default rate: **18.2%** — well below the 25% overall rate

More than 6 in 10 rejections by traditional scoring are false negatives. These are creditworthy people — thin-file newcomers, gig workers, recent immigrants — who the cash flow model can safely approve.

### Avoidable Risk (Traditional Score 620 or higher)

- Borrowers approved by traditional scoring: **785**
- Of those, actually defaulted: **133 (16.9%)**
- Cash flow model flags 165 for review (default probability above 50%)
  - True positives caught: **72** out of 133 actual defaults
  - Precision: **43.6%**

The flagging precision (43.6%) is solid for a review queue: for every 2 legitimate catches, there is roughly 1.3 false alarms. Useful for routing to manual underwriting review, not for automatic rejection.

### Loss Simulation (Fixed Approval Rate)

At the same approval volume, the cash flow model consistently reduces default losses:

| Approval Rate | Trad Loss | CF Loss | Reduction |
|---|---|---|---|
| 50% | 13.0% | 10.6% | 18.5% |
| 60% | 16.3% | 11.6% | 28.7% |
| **70%** | **19.5%** | **13.6%** | **30.4% (peak)** |
| 80% | 21.7% | 17.4% | 19.8% |
| 90% | 23.6% | 20.9% | 11.3% |

Peak loss reduction is at **70% approval (30.4%)**. This is the operationally relevant zone for most personal loan and credit card portfolios. Beyond 80% approval, both models are approving borderline borrowers and the gap narrows.

### Risk Bucket Analysis

FlowScore scale: 300 to 850

| Bucket | Count | % of Test | Default Rate |
|---|---|---|---|
| Very Low Risk (750+) | 85 | 6.8% | 4.7% |
| Low Risk (650-749) | 563 | 45.0% | 11.0% |
| Medium Risk (550-649) | 186 | 14.9% | 21.5% |
| High Risk (450-549) | 208 | 16.6% | 43.3% |
| Very High Risk (below 450) | 208 | 16.6% | 56.3% |

Clear monotonic separation across all five tiers. The 650 cutoff separates the "safe" zone (11% DR below) from the riskier buckets (21%+ above).

### Orthogonality Matrix

|  | FlowScore >= 650 | FlowScore < 650 |
|---|---|---|
| **Trad Score >= 660** | 9.0% default (n=513) | **32.1% default** (n=134) — Avoidable Risk |
| **Trad Score < 660** | **14.8% default** (n=135) — Missed Opportunity | 43.6% default (n=468) |

The two models are meaningfully orthogonal. Trad High + Flow Low (32.1% default) are the clearest evidence of disguised risk — bureau history looks clean but cashflow is deteriorating. Trad Low + Flow High (14.8% default) are the missed opportunity segment: rejected by traditional scoring but materially safer than the 43.6% base rate for all traditionally-rejected borrowers.

---

## 6. Fairness Analysis

EEOC 80% adverse impact ratio (AIR) evaluated using archetypes as demographic proxies (stable_salaried as reference group).

Key findings:
- **Thin File/Newcomer**: 29.4x more approvals from FlowScore vs traditional at the same actual default rate (15.5%). This is the primary fairness benefit — creditworthy immigrants and young adults who traditional scoring excludes.
- **Gig Worker**: 0.60x relative approval rate (fewer FlowScore approvals than traditional). This reflects their actual default rate (31.3%), not bias — FlowScore correctly identifies elevated risk that traditional scores miss. This is accurate assessment, not disparate impact.
- **EEOC result**: Some archetypes (financially stressed, overextended) fall below the 0.80 AIR threshold. This reflects genuine behavioral risk differences captured in transaction data, not group membership.

Score calibration equity: within each score band, default rates are consistent across archetypes. The model measures financial behavior, not which group a consumer belongs to.

---

## 7. Setup Issues Encountered

- **OpenMP conflict on Apple Silicon**: PyTorch and NumPy/sklearn each load their own OpenMP runtime, causing crash when running categorizer.py with DistilBERT. Fixed by adding `KMP_DUPLICATE_LIB_OK=TRUE` to `.env`.
- **XGBoost install via pip failed** on macOS ARM due to missing `libomp.dylib`. Fixed with: `conda install -c conda-forge xgboost`
- **DistilBERT Run 1 failure**: 0% F1 on gambling_win and payday_loan_deposit because these categories differ only by transaction sign, not merchant text. Fixed with amount-direction collapse in training.
- **LightGBM train_combined() crash**: `random_state` passed twice when Optuna best_params already included it. Fixed by removing the duplicate kwarg.

---

## 8. Output Files

| File | Description |
|---|---|
| `data/flowscore_dataset.json` | 5,000 consumer profiles + transactions (299.7 MB) |
| `data/features.csv` | 45-feature matrix, 5,000 rows |
| `data/model_results/results_summary.json` | Full model metrics and business value |
| `data/model_results/flowscores.csv` | Per-consumer FlowScores (test set) |
| `data/model_results/model_bundle.joblib` | Trained CatBoost model + scaler + feature metadata |
| `data/model_results/roc_curves.png` | AUC comparison across all models |
| `data/model_results/approval_lift.png` | Approval rate curves |
| `data/model_results/risk_buckets.png` | Risk tier breakdown |
| `data/model_results/cross_tab.png` | Traditional vs FlowScore orthogonality matrix |
| `models/distilbert_categorizer/` | Fine-tuned DistilBERT weights, tokenizer, label_map.json |
