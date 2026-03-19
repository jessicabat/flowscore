# FlowScore

**Cash flow credit scoring from bank transaction data.**

FlowScore is an end-to-end ML pipeline that predicts credit default risk using deposit account transactions — no credit bureau history required.

**[View the project →](https://jessicabat.github.io/flowscore/)**

---

## Why This Exists

Traditional credit scores (FICO, VantageScore) rely entirely on credit bureau history. This excludes ~76 million U.S. consumers with thin or no credit files. Cash flow underwriting uses bank transaction data — income patterns, spending behavior, savings habits — to assess creditworthiness independently.

FlowScore demonstrates this approach end-to-end: raw transactions → categorization → behavioral features → credit risk prediction.

## Results

| Model | AUC-ROC | KS Stat | Lift vs Traditional |
|-------|---------|---------|---------------------|
| Traditional score only | 0.687 | 0.342 | — |
| Logistic Regression (cash flow) | 0.761 | 0.452 | +10.8% |
| XGBoost (cash flow) | 0.732 | 0.412 | +6.5% |
| Combined (trad + cash flow) | 0.736 | 0.411 | +7.1% |

At a 60% approval rate, cash flow scoring reduces default losses by **21.3%** compared to traditional scoring alone.

### The Orthogonality Matrix

Cash flow scores capture risk that traditional scores cannot see:

|  | FlowScore High | FlowScore Low |
|--|----------------|---------------|
| **Trad Score High** | 11.6% default | **45.2% default** (Avoidable Risk) |
| **Trad Score Low** | **27.6% default** (Missed Opportunity) | 47.6% default |

285 creditworthy borrowers were rejected by the traditional score. 133 risky borrowers were approved. FlowScore catches both.

## Pipeline

```
Raw Transactions → Categorization → Feature Engineering → Credit Risk Model
                   (Rules + LLM)    (45 features)         (LogReg + XGBoost)
```

### Module 1: Transaction Categorization

Hybrid approach: rule-based keyword matching handles ~83% of transactions; Claude Sonnet handles the remaining ambiguous cases via API.

| Approach | Accuracy | Cost |
|----------|----------|------|
| Rules only (clean data) | 99.2% | $0 |
| Rules only (noisy data) | 82.3% | $0 |
| Hybrid rules + Sonnet (noisy data) | **95.7%** | $0.36 / 10 consumers |

25 categories including payroll, gig income, rent, BNPL, gambling, payday loans, and fees.

### Module 2: Feature Engineering

45 features across five groups:

- **Income** (10): monthly income, stability, regularity, source count, gig/benefits flags, trend, pay frequency
- **Spending** (12): monthly spend, discretionary vs essential ratio, subscription count, transaction sizes, category diversity
- **Balance & Savings** (7): net cash flow, savings rate, months with negative flow, balance trend
- **Obligations** (8): housing payment, loan count, BNPL activity, DTI proxy, loan stacking flag
- **Red Flags** (8): overdraft frequency, gambling spend, payday loan flag, total fees

Top SHAP features: `obligation_to_income_ratio`, `estimated_balance_trend`, `income_regularity`, `income_to_expense_ratio`, `min_monthly_cashflow`.

### Module 3: Credit Risk Model

Binary classification (default / no-default within 12 months) using Logistic Regression and XGBoost. Score calibrated to a 300–850 range (higher = lower risk), matching the intuition of traditional credit scores.

Business value analysis includes approval rate simulations, loss rate comparisons, risk bucket breakdowns, and an orthogonality matrix.

## Repo Structure

```
flowscore/
├── data/
│   ├── generate_synthetic_data.py     # Consumer + transaction generator
│   ├── flowscore_dataset.json         # 5,000 consumers, 3.3M transactions
│   ├── features.csv                   # Extracted feature matrix
│   └── model_results/
│       ├── roc_curves.png             # Model comparison
│       ├── risk_buckets.png           # FlowScore risk segments
│       ├── cross_tab.png              # Orthogonality matrix
│       ├── approval_lift.png          # Approval rate comparison
│       ├── flowscores.csv             # Per-consumer scores
│       └── results_summary.json       # Full metrics
├── src/
│   ├── categorizer.py                 # Hybrid rules + LLM categorizer
│   ├── noise_generator.py             # Realistic merchant string corruption
│   ├── feature_engine.py              # 45-feature extraction pipeline
│   └── model.py                       # Credit risk models + business value
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

# 4. (Optional) Run transaction categorization with LLM
export ANTHROPIC_API_KEY="sk-ant-..."
python src/categorizer.py --input data/flowscore_dataset.json \
  --output data/categorized.json --n_consumers 10 --noise medium
```

## Requirements

```
numpy
pandas
scikit-learn
xgboost
shap
matplotlib
anthropic          # only for LLM categorization
python-dotenv      # only for LLM categorization (.env API key)
```

## Data

All data is synthetic. 5,000 consumer profiles across six behavioral archetypes (stable salaried, gig worker, high earner, financially stressed, thin-file newcomer, overextended), each with 6–12 months of transaction history. Default labels are generated from transaction-derived behavioral signals. This is a methodology demonstration, not a production model.

## Further Reading

FinRegLab's [independent research](https://finreglab.org/research/fact-sheet-cash-flow-data-in-underwriting-credit/) confirmed that cash flow variables are as predictive as traditional credit scores and provide meaningful lift when combined with them.

## Author

**Jessica** — Data Science, UC San Diego (ML/AI concentration). Previously built automated valuation models for loan collateral at AND Global, a fintech operating in markets where alternative credit scoring is essential. This project extends that experience to the U.S. open banking ecosystem.
