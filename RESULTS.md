# FlowScore — Run Results & Performance Notes

A reference document capturing the full pipeline execution: decisions made, numbers produced, and what they mean.

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

## 2. Categorization — Three Conditions Tested

The categorizer was tested across three conditions to isolate the value of noise handling and LLM fallback.

### Condition A: Clean merchant names, rules-only
- Accuracy: **99.2%**
- Rule coverage: 100%
- Only failure: `groceries → shopping` (47 misclassifications)

### Condition B: Noisy merchant names, rules-only
- Accuracy: **82.3%** — a 17-point drop
- ~1,040 transactions fell to "other" (couldn't be matched)
- Top failures: dining, food_delivery, transportation all leaked into "other"
- **This is the realistic baseline without LLM augmentation**

### Condition C: Noisy merchant names, hybrid (rules + Claude)
- Accuracy: **95.7%** — recovers 13+ points over rules-only
- LLM handled 1,040 ambiguous transactions
- API cost: **$0.362** for 6,209 transactions (~$0.058/1,000 txns)
- Remaining failures: mostly `groceries → shopping` boundary cases

### Decision: Why hybrid?
Rules are fast and free — they handle ~83% of real transactions correctly. The LLM fallback is invoked only for the ~17% that rules can't confidently classify, making it cost-effective while recovering most of the accuracy gap.

---

## 3. Feature Engineering

45 features extracted across 5 categories. Key distributional signals:

| Feature | No Default | Default | Δ |
|---|---|---|---|
| savings_rate | +0.012 | **-0.323** | -0.336 |
| obligation_to_income_ratio | 0.348 | **0.473** | +0.125 |
| months_negative_cashflow | 3.6 | **6.2** | +2.6 |
| loan_stacking_flag | 0.079 | **0.266** | +0.187 |
| bnpl_active | 0.233 | **0.416** | +0.183 |
| payday_loan_flag | 0.028 | **0.124** | +0.096 |
| overdraft_frequency | 0.062 | **0.155** | +0.093 |
| income_trend_slope | +54.6 | **+5.4** | -49.2 |

Notable: `late_fee_count` was zero across all 5,000 consumers — a data generation gap worth revisiting if extending this project.

---

## 4. Model Performance

Train/test split: 75/25, stratified. All models evaluated on the same 1,250 test consumers.

| Model | AUC-ROC | KS | Gini |
|---|---|---|---|
| Traditional score only (baseline) | 0.6869 | 0.3419 | 0.3738 |
| Logistic Regression (cash flow) | **0.7613** | 0.4520 | 0.5226 |
| Gradient Boosting / XGBoost (cash flow) | 0.7317 | 0.4115 | 0.4634 |
| Combined (cash flow + trad score) | 0.7360 | 0.4105 | 0.4720 |

**Cash flow AUC lift over traditional: +6.5% (gradient boosting) / +10.8% (logistic regression)**

### Why logistic regression beats XGBoost here
The LR model outperforms XGBoost on this dataset — likely because the features are well-engineered linear signals (ratios, counts, trends) and the dataset (5K consumers) isn't large enough for tree ensembles to shine. LR also benefits from being regularized (L2) and having standardized inputs.

### Top Features — XGBoost Importance vs. SHAP

| Rank | XGBoost Importance | SHAP (mean |SHAP|) |
|---|---|---|
| 1 | has_gig_income (19.2%) | obligation_to_income_ratio (0.361) |
| 2 | obligation_to_income_ratio (6.9%) | estimated_balance_trend (0.283) |
| 3 | estimated_balance_trend (4.6%) | income_regularity (0.233) |
| 4 | net_monthly_cashflow_mean (3.7%) | income_to_expense_ratio (0.181) |
| 5 | income_to_expense_ratio (2.8%) | min_monthly_cashflow (0.151) |

XGBoost importance flags `has_gig_income` at #1 — a binary flag with high split utility. SHAP gives a more nuanced picture: obligation ratios and balance trends are the true workhorses for individual predictions.

---

## 5. Business Value Analysis

### Missed Opportunity (Traditional Score < 620)

- Borrowers rejected by traditional scoring: **465**
- Of those, would actually repay: **285 (61.3%)**
- Cash flow model rescues 218 of them with default prob < 30%
  - Of rescued: **160 repay (73.4%)**, 58 would default (26.6%)

**Interpretation**: More than 6 in 10 rejections by traditional scoring are false negatives. These are creditworthy people — likely thin-file newcomers or gig workers — who the cash flow model can safely approve.

### Avoidable Risk (Traditional Score ≥ 620)

- Borrowers approved by traditional scoring: **785**
- Of those, actually defaulted: **133 (16.9%)**
- Cash flow model flags 50 for review (default prob > 50%)
  - True positives caught: **21**; False alarms: 29
  - Precision: **42.0%**

**Interpretation**: The flagging precision (42%) is modest — for every 2 legitimate catches, there's ~1.4 false alarms. This is acceptable for a review queue but not for automatic rejection.

### Approval Simulation (Fixed Loss Rate)

At a 20% target loss rate, traditional scoring approves **0%** of applicants (too conservative), while the cash flow model approves **83.8%**.

| Target Loss Rate | Trad Approve | CF Approve | Lift |
|---|---|---|---|
| 5–15% | 0.0% | 0.3% | +0.3% |
| **20%** | **0.0%** | **83.8%** | **+83.8%** |
| 25% | 0.0% | 99.9% | +99.9% |

### Loss Simulation (Fixed Approval Rate)

At the same approval rates, the cash flow model consistently reduces loss:

| Approval Rate | Trad Loss | CF Loss | Reduction |
|---|---|---|---|
| 50% | 13.0% | 11.0% | 14.8% |
| **60%** | **16.3%** | **12.8%** | **21.3%** |
| 70% | 19.5% | 15.8% | 19.3% |
| 80% | 21.7% | 18.9% | 12.9% |
| 90% | 23.6% | 21.6% | 8.3% |

Peak loss reduction is at **60% approval rate (~21% reduction)**. Beyond 80% approval, both models are approving borderline borrowers and the gap narrows.

### Risk Bucket Analysis (FlowScore Style)

FlowScore range: 300–850, mean=724, std=133

| Bucket | Count | % of Test | Default Rate |
|---|---|---|---|
| Very Low Risk (750+) | 744 | 59.5% | 12.6% |
| Low Risk (650–749) | 176 | 14.1% | 35.8% |
| Medium Risk (550–649) | 157 | 12.6% | 38.2% |
| High Risk (450–549) | 98 | 7.8% | 55.1% |
| Very High Risk (<450) | 75 | 6.0% | 56.0% |

Note: The jump from Very Low to Low Risk is large (12.6% → 35.8%). The 650 cutoff is a meaningful underwriting threshold.

### Orthogonality Matrix (Cross-Tab)

|  | FlowScore ≥ 650 | FlowScore < 650 |
|---|---|---|
| **Trad Score ≥ 660** | 11.6% default (n=605) | **45.2% default** (n=42) — Avoidable Risk |
| **Trad Score < 660** | **27.6% default** (n=315) — Missed Opportunity | 47.6% default (n=288) |

The two models are meaningfully orthogonal: each captures default risk the other misses. Trad High + Flow Low consumers (45.2% default) are the clearest evidence of disguised risk.

---

## 6. Setup Issues Encountered

- **XGBoost install via pip failed** on macOS ARM due to missing `libomp.dylib` (OpenMP runtime). Fixed with: `conda install -c conda-forge xgboost`
- **Categorizer path error**: script was run from wrong directory. Fixed by running from project root with `data/flowscore_dataset.json` path.

---

## 7. Output Files

| File | Description |
|---|---|
| `data/flowscore_dataset.json` | 5,000 consumer profiles + transactions (299.7 MB) |
| `data/features.csv` | 45-feature matrix, 5,000 rows |
| `data/model_results/results_summary.json` | Full model metrics |
| `data/model_results/flowscores.csv` | Per-consumer FlowScores |
| `data/model_results/roc_curves.png` | AUC comparison across 4 models |
| `data/model_results/approval_lift.png` | Approval rate curves |
| `data/model_results/risk_buckets.png` | Risk tier breakdown |
| `data/model_results/cross_tab.png` | Traditional vs FlowScore orthogonality |
