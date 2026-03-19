# FlowScore — End-to-End Pipeline Walkthrough

This document traces two real consumers from the synthetic dataset through every stage of the pipeline, showing exactly what happens at each step.

---

## The Two Consumers

| | Consumer A | Consumer B |
|--|-----------|-----------|
| **ID** | c_04556 | c_01700 |
| **Archetype** | Thin-file newcomer | Overextended |
| **Traditional score** | 494 | 662 |
| **FlowScore** | **846** | **407** |
| **Default probability** | 0.7% | 80.5% |
| **Actually defaulted?** | No | Yes |

**Consumer A** would be rejected by any traditional lender (score 494 is deep subprime). But FlowScore sees a responsible borrower and assigns 846 — near the top of the range. This is the **"Missed Opportunity"** that Prism Data's sales pitch is built on.

**Consumer B** would be approved by most traditional lenders (score 662 is near-prime). But FlowScore sees someone drowning in obligations and assigns 407 — high risk. This is the **"Avoidable Risk"** that cash flow scoring catches.

---

## Stage 1: Raw Transactions

This is what the pipeline receives. A chronological list of deposits and withdrawals with merchant names and amounts.

### Consumer A — Thin-File Newcomer ($2,648/mo income, 4 months history)

```
DATE          AMOUNT     MERCHANT                    
2024-09-02    -$34.73    DOORDASH ORDER              
2024-09-02    -$67.70    WALMART                     
2024-09-03    -$36.35    POSTMATES                   
2024-09-04    -$19.95    GRUBHUB                     
2024-09-04    -$65.50    EXXONMOBIL                  
2024-09-05    -$45.11    IHOP                        
2024-09-05   -$152.10    PAYPAL TRANSFER             
2024-09-06    -$17.28    TACO BELL                   
2024-09-08    -$35.36    DUNKIN DONUTS               
2024-09-08    -$19.93    SHELL                       
2024-09-11    -$53.27    DOORDASH ORDER
...
2024-09-24  +$1,312.83   STARK INDUSTRIES PAY         ← paycheck
...
2024-10-08  +$1,323.46   STARK INDUSTRIES PAY         ← paycheck
...
(181 transactions total over 4 months)
```

Short history, modest income, single employer. A traditional score penalizes the thin file. But what does the behavior say?

### Consumer B — Overextended ($4,761/mo income, 11 months history)

```
DATE          AMOUNT     MERCHANT                    
2024-02-05    -$54.66    STOP AND SHOP               
2024-02-05    -$53.23    SUBWAY                      
2024-02-07    -$58.34    INSTACART DELIVERY          
2024-02-07    -$65.22    SHELL                       
2024-02-08    -$54.26    BUFFALO WILD WINGS          
2024-02-08    -$57.32    POSTMATES                   
2024-02-09    -$34.75    DOORDASH ORDER              
2024-02-09    -$52.59    GRUBHUB                     
2024-02-10    -$37.16    POPEYES                     
2024-02-11  +$2,403.24   ACME CORP PAYROLL            ← paycheck
2024-02-11   -$114.54    PUBLIX                      
...
(1,023 transactions total over 11 months)
```

Higher income, longer history, decent traditional score. But look at the volume of spending.

---

## Stage 2: Transaction Categorization

Each transaction gets classified into one of 24 categories.

### Clean merchant → Category (rules handle this)

```
STARK INDUSTRIES PAY     →  payroll
DOORDASH ORDER           →  food_delivery
SHELL                    →  transportation
IHOP                     →  dining
WALMART                  →  shopping
DUNKIN DONUTS            →  dining
```

### Noisy merchant → Rules fail → LLM handles it

In real bank data, merchant strings are corrupted. Here's what the noise layer does:

```
Original:   DOORDASH ORDER
Noisy:      ONLINE PMT DOORDASH O         ← prefix added + truncated
Rule says:  None                           ← no rule matches
→ Sent to Claude Sonnet API
→ Sonnet returns: "food_delivery"          ✓ correct

Original:   STOP AND SHOP
Noisy:      ONLINE PMT STOP AND S         ← prefix added + truncated  
Rule says:  None                           ← "STOP AND SHOP" keyword is broken
→ Sent to Claude Sonnet API
→ Sonnet returns: "groceries"              ✓ correct

Original:   WALMART
Noisy:      WALMART                        ← no corruption this time
Rule says:  shopping                       ✓ matched by rules (free, instant)
```

**Result:** Rules catch ~84% at zero cost. Sonnet catches the remaining ~16% at $0.36 per 10 consumers. Combined accuracy: 95.7%.

---

## Stage 3: Feature Engineering

Categorized transactions are aggregated into 45 numeric features per consumer.

### Side-by-Side Feature Comparison

| Feature | Consumer A (Thin-File) | Consumer B (Overextended) | Signal |
|---------|----------------------|--------------------------|--------|
| **savings_rate** | -0.19 | **-0.32** | B spends 32% more than income |
| **obligation_to_income_ratio** | 0.29 | **0.51** | B's obligations eat half of income |
| **overdraft_frequency** | 0.00/mo | **0.18/mo** | B has recurring overdrafts |
| **months_negative_cashflow** | 3 of 4 | **9 of 11** | B bleeds cash almost every month |
| **loan_payment_count** | 0 | **4** | B has 4 concurrent loan payments |
| **loan_stacking_flag** | 0 | **1** | B is stacking loans (3+ lenders) |
| **bnpl_active** | 0 | **1** | B is using buy-now-pay-later |
| **fee_total** | $0 | **$65** | B is accumulating bank fees |
| **income_to_expense_ratio** | 0.84 | **0.76** | Both spend more than they earn, but B is worse |
| **housing_payment_amount** | $809 | **$1,348** | B has higher housing cost |
| **estimated_balance_trend** | -938 | **-2,099** | B's balance is declining fast |
| **min_monthly_cashflow** | -$1,909 | **-$3,022** | B's worst month is much worse |
| **gambling_total_spend** | $0 | $0 | Neither gambles |
| **payday_loan_flag** | 0 | 0 | Neither uses payday lenders |

Consumer A has a short history and slightly negative savings, but zero red flags. No overdrafts, no loans, no BNPL, no fees. A traditional score sees "thin file" and panics. FlowScore sees "responsible newcomer."

Consumer B earns more and has a longer history, but is drowning: 4 concurrent loans, stacking debt, overdrafting regularly, negative cash flow 9 out of 11 months. A traditional score sees "near-prime with history." FlowScore sees "overextended and deteriorating."

---

## Stage 4: SHAP Feature Importance

SHAP values explain which features drive each consumer's score. Here are the top features from the model and how they apply:

```
Global Top Features (by mean |SHAP|):
  1. obligation_to_income_ratio    0.361    ← Consumer B's 0.51 is very high
  2. estimated_balance_trend       0.283    ← Consumer B's -2,099 is alarming
  3. income_regularity             0.233    ← Both similar (~0.7)
  4. income_to_expense_ratio       0.181    ← Consumer B's 0.76 is dangerous
  5. min_monthly_cashflow          0.151    ← Consumer B's -3,022 is extreme
  6. subscription_total_monthly    0.148
  7. overdraft_frequency           0.145    ← Consumer B has 0.18/month
```

**For Consumer A**, the top features are neutral or positive: low obligation ratio (0.29), zero overdrafts, zero fees. SHAP pushes the prediction toward "no default."

**For Consumer B**, the top features are all red: high obligation ratio (0.51), rapidly declining balance, high overdraft frequency. SHAP pushes the prediction strongly toward "default."

---

## Stage 5: Credit Risk Score (FlowScore)

The model outputs a default probability, which is calibrated to a 300-850 score.

```
Score = 300 + 550 × (1 - default_probability)
```

### Consumer A — The Missed Opportunity

```
Default probability:  0.7%     (very low risk)
FlowScore:           846       (near-maximum)
Traditional score:   494       (deep subprime — AUTO REJECT)

Traditional decision: REJECTED
FlowScore decision:   APPROVED ✓

Actual outcome:       DID NOT DEFAULT ✓
```

A lender using only the traditional score would reject this person. They would lose a good customer. FlowScore sees the behavioral evidence and approves them.

### Consumer B — The Avoidable Risk

```
Default probability:  80.5%    (very high risk)
FlowScore:           407       (high risk bucket)
Traditional score:   662       (near-prime — LIKELY APPROVED)

Traditional decision: APPROVED
FlowScore decision:   FLAGGED FOR REVIEW ⚠️

Actual outcome:       DEFAULTED ✗
```

A lender using only the traditional score would approve this person. They would absorb the loss. FlowScore sees the cash flow deterioration and flags them.

---

## The Full Picture

```
                        Traditional Score
                    Low (<620)     High (≥620)
                 ┌──────────────┬──────────────┐
FlowScore High   │  27.6%       │  11.6%       │
(Low Risk)       │  n=315       │  n=605       │
                 │  ★ CONSUMER A│              │
                 ├──────────────┼──────────────┤
FlowScore Low    │  47.6%       │  45.2%       │
(High Risk)      │  n=288       │  n=42        │
                 │              │  ★ CONSUMER B│
                 └──────────────┴──────────────┘

Consumer A sits in the top-left: low traditional score, high FlowScore.
→ Traditional scoring misses them. Cash flow scoring finds them.

Consumer B sits in the bottom-right: high traditional score, low FlowScore.
→ Traditional scoring approves them. Cash flow scoring catches them.

This is why cash flow scores are orthogonal to traditional scores.
They measure different things. Together, they're more powerful than either alone.
```