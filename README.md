# FlowScore

**Cash flow credit scoring from bank transaction data.**

FlowScore is an end-to-end ML pipeline that predicts credit default risk using deposit account transactions — no credit bureau history required.

**[View the project](https://jessicabat.github.io/flowscore/)**

---

## Why This Exists

Traditional credit scores (FICO, VantageScore) rely entirely on credit bureau history. This excludes roughly 76 million U.S. consumers with thin or no credit files. Cash flow underwriting uses bank transaction data — income patterns, spending behavior, savings habits — to assess creditworthiness independently.

FlowScore demonstrates this approach end-to-end: raw transactions to categorization to behavioral features to credit risk prediction.

## Results

| Model | AUC-ROC | KS Stat | Gini | AUC Lift |
|-------|---------|---------|------|----------|
| Traditional score only (baseline) | 0.6869 | 0.3419 | 0.3738 | baseline |
| Logistic Regression (cash flow) | 0.7613 | 0.4520 | 0.5226 | +10.8% |
| XGBoost (cash flow) | 0.7317 | 0.4115 | 0.4634 | +6.5% |
| LightGBM + Optuna | 0.7611 | 0.4542 | 0.5222 | +10.8% |
| **CatBoost + Optuna** | **0.7637** | **0.4404** | **0.5273** | **+11.2%** |
| Combined (traditional + best CF) | 0.7617 | 0.4521 | 0.5233 | +10.9% |

At a 70% approval rate, cash flow scoring reduces default losses by **30.4%** compared to traditional scoring alone (peak reduction).

### The Orthogonality Matrix

Cash flow scores capture risk that traditional scores cannot see:

|  | FlowScore High | FlowScore Low |
|--|----------------|---------------|
| **Trad Score High** | 9.0% default (n=513) | **32.1% default** (n=134) — Avoidable Risk |
| **Trad Score Low** | **14.8% default** (n=135) — Missed Opportunity | 43.6% default (n=468) |

135 creditworthy borrowers were rejected by the traditional score. 134 risky borrowers were approved. FlowScore catches both.

## Pipeline

```
Raw Transactions -> Categorization -> Feature Engineering -> Credit Risk Model
                   (Rules + DistilBERT)  (45 features)     (CatBoost + Optuna)
```

### Module 1: Transaction Categorization

Hybrid approach: a rule-based keyword engine handles roughly 83% of transactions. A fine-tuned DistilBERT model (distilbert-base-uncased, fine-tuned on merchant name to category pairs) handles the remaining ambiguous cases — no API calls required.

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Rules only (clean data) | 99.2% | Perfect-world baseline |
| Rules only (noisy data) | 82.3% | Realistic production baseline |
| DistilBERT (clean data) | 99.7% | After fine-tuning on 70K pairs |
| DistilBERT (noisy data) | 91.5% | Train-clean, test-noisy distribution gap |
| Hybrid rules + Claude Sonnet (noisy) | 95.7% | Original approach, requires API |

25 categories including payroll, gig income, rent, BNPL, gambling, payday loans, and fees.

### Module 2: Feature Engineering

45 features across five groups:

- **Income** (10): monthly income, stability, regularity, source count, gig/benefits flags, trend, pay frequency
- **Spending** (12): monthly spend, discretionary vs essential ratio, subscription count, transaction sizes, category diversity
- **Balance and Savings** (7): net cash flow, savings rate, months with negative flow, balance trend
- **Obligations** (8): housing payment, loan count, BNPL activity, DTI proxy, loan stacking flag
- **Red Flags** (8): overdraft frequency, gambling spend, payday loan flag, total fees

Top features by model importance: `income_regularity`, `obligation_to_income_ratio`, `estimated_balance_trend`, `net_monthly_cashflow_std`, `subscription_total_monthly`.

### Module 3: Credit Risk Model

Binary classification (default / no-default within 12 months). CatBoost and LightGBM are tuned with Optuna (30 and 40 trials respectively, 5-fold cross-validation). Score calibrated to a 300 to 850 range matching the intuition of traditional credit scores.

Business value analysis includes approval rate simulations, loss rate comparisons, risk bucket breakdowns, and an orthogonality matrix.

### Module 4: Fairness Analysis

EEOC 80% adverse impact ratio (AIR) analysis across archetypes used as demographic proxies. Score calibration equity analysis confirms consistent default rates within score bands across groups. Thin File/Newcomer consumers receive substantially more approvals from FlowScore than from traditional scoring.

## Repo Structure

```
flowscore/
├── data/
│   ├── generate_synthetic_data.py     # Consumer + transaction generator
│   ├── flowscore_dataset.json         # 5,000 consumers, 3.3M transactions
│   ├── features.csv                   # Extracted feature matrix
│   └── model_results/
│       ├── roc_curves.png
│       ├── risk_buckets.png
│       ├── cross_tab.png
│       ├── approval_lift.png
│       ├── flowscores.csv             # Per-consumer scores
│       ├── model_bundle.joblib        # Trained model + scaler + feature metadata
│       └── results_summary.json      # Full metrics
├── models/
│   └── distilbert_categorizer/        # Fine-tuned DistilBERT weights + tokenizer
├── src/
│   ├── categorizer.py                 # Hybrid rules + DistilBERT categorizer
│   ├── train_distilbert.py            # DistilBERT fine-tuning script
│   ├── noise_generator.py             # Realistic merchant string corruption
│   ├── feature_engine.py              # 45-feature extraction pipeline
│   ├── model.py                       # Credit risk models + business value
│   └── fairness_analysis.py           # EEOC disparate impact analysis
├── app.py                             # Streamlit interactive demo
├── index.html                         # Project site (GitHub Pages)
└── requirements.txt
```

## Quick Start

```bash
# 1. Generate synthetic data
python data/generate_synthetic_data.py --n_consumers 5000 --seed 42

# 2. Extract features
python src/feature_engine.py --input data/flowscore_dataset.json --output data/features.csv

# 3. Train models and run business value analysis
python src/model.py --features data/features.csv --output data/model_results/

# 4. (Optional) Fine-tune DistilBERT categorizer
python src/train_distilbert.py \
    --dataset data/flowscore_dataset.json \
    --output models/distilbert_categorizer/ \
    --epochs 3

# 5. (Optional) Run categorization with DistilBERT
python src/categorizer.py --input data/flowscore_dataset.json \
    --output data/results_distilbert.json \
    --distilbert models/distilbert_categorizer/ \
    --noise medium --n_consumers 10

# 6. Run the interactive demo
streamlit run app.py
```

## Requirements

**Demo only** (`requirements.txt`):
```
streamlit
pandas
numpy
scikit-learn
catboost
joblib
```

**Full pipeline** (training, categorizer, fairness analysis):
```
numpy pandas scikit-learn xgboost lightgbm catboost optuna shap matplotlib
transformers torch accelerate joblib python-dotenv anthropic
```

## Data

All data is synthetic. 5,000 consumer profiles across six behavioral archetypes (stable salaried, gig worker, high earner, financially stressed, thin-file newcomer, overextended), each with 6 to 12 months of transaction history. Default labels are generated from transaction-derived behavioral signals. This is a methodology demonstration, not a production model.

## Further Reading

FinRegLab's [independent research](https://finreglab.org/research/fact-sheet-cash-flow-data-in-underwriting-credit/) confirmed that cash flow variables are as predictive as traditional credit scores and provide meaningful lift when combined with them.

## Author

**Jessica** — Data Science, UC San Diego (ML/AI concentration). Previously built automated valuation models for loan collateral at AND Global, a fintech operating in markets where alternative credit scoring is essential. This project extends that experience to the U.S. open banking ecosystem.
