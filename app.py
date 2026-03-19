"""
FlowScore: Interactive Demo (Streamlit / Streamlit Cloud)
=========================================================
Deploy: connect your GitHub repo at share.streamlit.io and select app.py.

Run locally:
    pip install streamlit pandas numpy scikit-learn catboost joblib
    streamlit run app.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="FlowScore: Cash Flow Credit Scoring",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject minimal CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }
  h1, h2, h3 { letter-spacing: -0.02em; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    font-weight: 600; font-size: 14px;
    padding: 8px 18px; border-radius: 6px 6px 0 0;
  }
  .metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 20px; height: 100%;
  }
  .score-number { font-size: 52px; font-weight: 800; line-height: 1; margin-bottom: 4px; }
  .score-bucket { font-size: 14px; font-weight: 600; color: #374151; margin-bottom: 12px; }
  .score-bar-bg {
    background: #e2e8f0; border-radius: 999px;
    height: 5px; margin-bottom: 12px;
  }
  .score-bar { height: 5px; border-radius: 999px; }
  .score-meta { color: #6b7280; font-size: 13px; }
  .label-tag {
    font-size: 10px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 8px;
  }
  .reason-card {
    border-radius: 0 8px 8px 0; padding: 12px 14px; margin-bottom: 8px;
  }
  .reason-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 2px;
  }
  .reason-title { font-size: 13px; font-weight: 600; color: #1e293b; margin-bottom: 1px; }
  .reason-desc { font-size: 12px; color: #64748b; line-height: 1.5; }
  .info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid #f1f5f9; font-size: 13px;
  }
  .info-label { color: #64748b; }
  .info-value { font-weight: 600; color: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ── Data loading (cached) ─────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "model_results")

@st.cache_resource
def load_scores():
    path = os.path.join(DATA_DIR, "flowscores.csv")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_resource
def load_results():
    path = os.path.join(DATA_DIR, "results_summary.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_resource
def load_model_bundle():
    path = os.path.join(DATA_DIR, "model_bundle.joblib")
    try:
        import joblib
        bundle = joblib.load(path)
        return bundle
    except Exception:
        return None

scores_df    = load_scores()
results      = load_results()
MODEL_BUNDLE = load_model_bundle()

# ── Constants ─────────────────────────────────────────────────────────────────
ARCHETYPE_LABELS = {
    "stable_salaried":          "Stable Salaried",
    "gig_worker":               "Gig Worker",
    "high_earner_high_spender": "High Earner / High Spender",
    "financially_stressed":     "Financially Stressed",
    "thin_file_newcomer":       "Thin File / Newcomer",
    "overextended":             "Overextended",
}

RISK_BUCKETS = [
    (750, 850, "Very Low Risk",  "#16a34a", "#f0fdf4"),
    (650, 749, "Low Risk",       "#4d7c0f", "#f7fee7"),
    (550, 649, "Medium Risk",    "#b45309", "#fffbeb"),
    (450, 549, "High Risk",      "#c2410c", "#fff7ed"),
    (300, 449, "Very High Risk", "#b91c1c", "#fef2f2"),
]

QUADRANT_INFO = {
    (True, True):   ("Both models agree: low risk",    "#f0fdf4",
                     "Traditional and FlowScore both indicate low credit risk."),
    (True, False):  ("Avoidable risk",                 "#fefce8",
                     "Traditional approved, but FlowScore detects elevated cashflow risk."),
    (False, True):  ("Missed opportunity",             "#eff6ff",
                     "Traditional rejected, but FlowScore identifies creditworthy cashflow behavior."),
    (False, False): ("Both models agree: high risk",   "#fef2f2",
                     "Traditional and FlowScore both flag this consumer as high risk."),
}

REASON_DESCRIPTIONS = {
    "obligation_to_income_ratio":   ("Debt-to-income ratio",     "high",  "Obligations consume a large share of monthly income"),
    "months_negative_cashflow":     ("Negative cashflow months", "high",  "Spending exceeded income in multiple months"),
    "overdraft_count":              ("Overdraft events",         "high",  "Account overdrafts indicate recurring liquidity gaps"),
    "overdraft_frequency":          ("Overdraft frequency",      "high",  "Frequent overdrafts suggest the balance regularly runs short"),
    "gambling_total_spend":         ("Gambling activity",        "high",  "Regular gambling spend increases financial uncertainty"),
    "gambling_frequency":           ("Gambling frequency",       "high",  "Recurring gambling transactions detected"),
    "payday_loan_flag":             ("Payday loan usage",        "high",  "Payday loan use signals acute short-term liquidity stress"),
    "payday_loan_count":            ("Payday loan count",        "high",  "Multiple payday loans suggest reliance on high-cost credit"),
    "bnpl_active":                  ("BNPL obligations",         "high",  "Buy-now-pay-later adds to monthly obligation burden"),
    "loan_stacking_flag":           ("Loan stacking",            "high",  "Multiple concurrent debt obligations detected"),
    "income_cv":                    ("Income volatility",        "high",  "High income variability makes repayment harder to sustain"),
    "net_monthly_cashflow_mean":    ("Net monthly cashflow",     "low",   "Strong positive cashflow supports consistent debt repayment"),
    "income_to_expense_ratio":      ("Income-to-expense ratio",  "low",   "A higher ratio reflects strong spending discipline"),
    "savings_rate":                 ("Savings rate",             "low",   "Positive savings buffer reduces the likelihood of default"),
    "estimated_balance_trend":      ("Balance trend",            "low",   "A rising balance indicates improving financial health"),
    "cashflow_trend_slope":         ("Cashflow trend",           "low",   "An improving cashflow trend is a reliable forward indicator"),
    "monthly_income_mean":          ("Monthly income level",     "low",   "Higher income provides greater capacity for debt repayment"),
    "income_regularity":            ("Income regularity",        "low",   "Consistent, predictable deposits support creditworthiness"),
    "min_monthly_cashflow":         ("Worst-month cashflow",     "low",   "A strong floor on monthly cashflow reduces tail risk"),
}

ARCHETYPE_DESC = {
    "stable_salaried":          "Regular salary, stable expenses, predictable cashflow",
    "gig_worker":               "Variable income from freelance or gig platforms",
    "high_earner_high_spender": "High income with elevated lifestyle spending",
    "financially_stressed":     "Income constraints and high obligation burden",
    "thin_file_newcomer":       "Limited credit history, new to credit",
    "overextended":             "Multiple debt obligations, elevated debt-to-income",
}

EXAMPLE_PROFILES = {
    "Stable Salaried":           dict(income=4500, spending=1200, housing=1400, loans=300,  regularity=0.90, overdrafts=0,  payday=False, gambling=0,   bnpl=False, savings=0.12,  trend=80),
    "Financially Stressed":      dict(income=2800, spending=1400, housing=1100, loans=150,  regularity=0.60, overdrafts=4,  payday=False, gambling=50,  bnpl=True,  savings=-0.02, trend=-30),
    "High Earner / High Spender":dict(income=9500, spending=3500, housing=2800, loans=800,  regularity=0.95, overdrafts=0,  payday=False, gambling=0,   bnpl=False, savings=0.18,  trend=220),
    "High Risk / Overextended":  dict(income=2200, spending=1600, housing=1000, loans=200,  regularity=0.40, overdrafts=9,  payday=True,  gambling=120, bnpl=True,  savings=-0.08, trend=-80),
    "Thin File / Newcomer":      dict(income=3500, spending=900,  housing=1100, loans=0,    regularity=0.85, overdrafts=1,  payday=False, gambling=0,   bnpl=False, savings=0.15,  trend=60),
}


# ── Helper functions ──────────────────────────────────────────────────────────

def get_risk_bucket(score: int):
    for lo, hi, label, color, bg in RISK_BUCKETS:
        if lo <= score <= hi:
            return label, color, bg
    return "Unknown", "#64748b", "#f8fafc"


def build_feature_vector(income, spending, housing, loans, regularity, overdrafts,
                         payday, gambling, bnpl, savings, trend, feature_names, feature_means):
    income = max(income, 1.0)
    spending = max(spending, 0.0)
    housing = max(housing, 0.0)
    loans = max(loans, 0.0)
    obligations = housing + loans
    net_cf = income - spending - obligations
    income_cv = max(0.0, 1.0 - regularity) * 0.35
    income_regularity_days = (1.0 - regularity) * 22.0

    computed = {
        "monthly_income_mean":        income,
        "monthly_income_std":         income * income_cv,
        "income_cv":                  income_cv,
        "income_regularity":          income_regularity_days,
        "income_to_expense_ratio":    income / max(spending + obligations, 1.0),
        "monthly_spend_mean":         spending,
        "monthly_spend_std":          spending * 0.15,
        "net_monthly_cashflow_mean":  net_cf,
        "savings_rate":               savings,
        "months_negative_cashflow":   max(0, min(12, int(max(0, -net_cf) / max(income * 0.05, 1)))),
        "min_monthly_cashflow":       net_cf - abs(net_cf) * 0.5,
        "estimated_balance_trend":    trend,
        "cashflow_trend_slope":       trend / max(income, 1.0),
        "housing_payment_amount":     housing,
        "total_monthly_obligations":  obligations,
        "obligation_to_income_ratio": obligations / income,
        "overdraft_count":            float(overdrafts),
        "overdraft_frequency":        float(overdrafts) / 12.0,
        "gambling_total_spend":       gambling * 12,
        "gambling_frequency":         1.0 if gambling > 0 else 0.0,
        "payday_loan_flag":           1.0 if payday else 0.0,
        "payday_loan_count":          1.0 if payday else 0.0,
        "bnpl_active":                1.0 if bnpl else 0.0,
        "loan_stacking_flag":         1.0 if (loans > 0 and housing > 0 and bnpl) else 0.0,
    }
    vector = [computed.get(f, feature_means.get(f, 0.0)) for f in feature_names]
    return np.array(vector, dtype=np.float64)


def get_reason_codes(fv, feature_names, feature_means, feature_importances, n=3):
    scores = []
    for i, feat in enumerate(feature_names):
        if feat not in REASON_DESCRIPTIONS:
            continue
        val = float(fv[i])
        mean = feature_means.get(feat, 0.0)
        importance = feature_importances.get(feat, 0.0)
        _, direction, _ = REASON_DESCRIPTIONS[feat]
        deviation = val - mean
        impact = deviation * importance if direction == "high" else -deviation * importance
        scores.append((feat, impact))
    scores.sort(key=lambda x: abs(x[1]), reverse=True)
    codes = []
    for feat, impact in scores[:n]:
        title, _, desc = REASON_DESCRIPTIONS[feat]
        sentiment = "risk" if impact > 0 else "strength"
        codes.append((sentiment, title, desc))
    return codes


# ── UI components (return HTML strings for st.markdown) ──────────────────────

def render_score_card(score: int, default_prob: float, label: str = "FLOWSCORE") -> str:
    bucket, color, bg = get_risk_bucket(score)
    pct = int((score - 300) / 550 * 100)
    return f"""
    <div style="background:{bg}; border:2px solid {color}30; border-radius:12px;
                padding:24px; text-align:center; box-shadow:0 1px 6px rgba(0,0,0,0.06);">
      <div class="label-tag" style="color:{color};">{label}</div>
      <div class="score-number" style="color:{color};">{score}</div>
      <div class="score-bucket">{bucket}</div>
      <div class="score-bar-bg">
        <div class="score-bar" style="background:{color}; width:{pct}%;"></div>
      </div>
      <div class="score-meta">
        Default probability: <strong style="color:{color};">{default_prob:.1%}</strong>
        &nbsp;&middot;&nbsp; Scale 300 to 850
      </div>
    </div>"""


def render_trad_card(score: int) -> str:
    bucket, color, bg = get_risk_bucket(score)
    approved = score >= 620
    status_color = "#15803d" if approved else "#b91c1c"
    status_text  = "Approved (cutoff 620)" if approved else "Rejected (below 620)"
    return f"""
    <div style="background:{bg}; border:1px solid {color}25; border-radius:12px;
                padding:20px; box-shadow:0 1px 4px rgba(0,0,0,0.05);">
      <div class="label-tag" style="color:#64748b;">Traditional Score</div>
      <div style="font-size:38px; font-weight:800; color:{color}; line-height:1; margin-bottom:4px;">{score}</div>
      <div style="font-size:13px; color:#374151; margin-bottom:10px;">{bucket}</div>
      <div style="font-size:12px; font-weight:600; color:{status_color};
                  background:{status_color}15; border-radius:6px; padding:4px 10px;
                  display:inline-block;">{status_text}</div>
    </div>"""


def render_quadrant_card(flow_score: int, trad_score: int, actual_default: bool) -> str:
    flow_high = flow_score >= 650
    trad_high = trad_score >= 620
    title, bg, desc = QUADRANT_INFO[(trad_high, flow_high)]
    outcome_color = "#b91c1c" if actual_default else "#15803d"
    outcome_text  = "Defaulted" if actual_default else "Repaid on time"
    return f"""
    <div style="background:{bg}; border-radius:12px; padding:20px; height:100%;
                box-shadow:0 1px 4px rgba(0,0,0,0.05);">
      <div class="label-tag" style="color:#64748b;">Model Agreement</div>
      <div style="font-size:15px; font-weight:700; color:#1e293b; margin-bottom:8px;">{title}</div>
      <div style="font-size:13px; color:#475569; margin-bottom:12px; line-height:1.5;">{desc}</div>
      <div style="font-size:12px; font-weight:600; color:{outcome_color};">
        Actual outcome: {outcome_text}
      </div>
    </div>"""


def render_reason_codes(codes: list) -> str:
    if not codes:
        return "<p style='color:#64748b; font-size:13px;'>No reason codes available.</p>"
    html = '<div style="margin-top:4px;"><div class="label-tag" style="color:#64748b; margin-bottom:10px;">Top Reason Codes</div>'
    for sentiment, title, desc in codes:
        is_risk = sentiment == "risk"
        bg      = "#fef2f2" if is_risk else "#f0fdf4"
        border  = "#fca5a5" if is_risk else "#86efac"
        label   = "Risk factor" if is_risk else "Strength"
        lc      = "#991b1b" if is_risk else "#166534"
        html += f"""
        <div class="reason-card" style="background:{bg}; border-left:3px solid {border};">
          <div class="reason-label" style="color:{lc};">{label}</div>
          <div class="reason-title">{title}</div>
          <div class="reason-desc">{desc}</div>
        </div>"""
    return html + "</div>"


def render_summary_card(income, spending, housing, loans, regularity, overdrafts,
                        savings, trend, payday, bnpl, gambling) -> str:
    obligations = housing + loans
    net_cf  = income - spending - obligations
    dti     = obligations / max(income, 1)
    cf_col  = "#15803d" if net_cf >= 0 else "#b91c1c"
    dti_col = "#15803d" if dti < 0.35 else ("#b45309" if dti < 0.50 else "#b91c1c")
    flags = []
    if payday:       flags.append('<span style="color:#92400e;">Payday loan</span>')
    if bnpl:         flags.append('<span style="color:#92400e;">BNPL active</span>')
    if gambling > 0: flags.append(f'<span style="color:#92400e;">Gambling ${gambling:,.0f}/mo</span>')
    if overdrafts > 0: flags.append(f'<span style="color:#92400e;">{overdrafts} overdraft(s)</span>')
    flags_html = ", ".join(flags) if flags else '<span style="color:#15803d;">None</span>'

    def row(label, value):
        return f'<div class="info-row"><span class="info-label">{label}</span><span class="info-value">{value}</span></div>'

    return f"""
    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:20px;">
      <div class="label-tag" style="color:#64748b; margin-bottom:12px;">Input Summary</div>
      {row("Monthly income", f"${income:,.0f}")}
      {row("Spending + obligations", f"${spending + obligations:,.0f}")}
      {row("Debt-to-income ratio", f'<span style="color:{dti_col};">{dti:.0%}</span>')}
      {row("Net monthly cashflow", f'<span style="color:{cf_col};">${net_cf:+,.0f}</span>')}
      {row("Savings rate", f"{savings:.0%}")}
      {row("Income regularity", f"{regularity:.0%}")}
      <div class="info-row" style="border-bottom:none;">
        <span class="info-label">Risk flags</span>
        <span style="font-size:12px; text-align:right;">{flags_html}</span>
      </div>
    </div>"""


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:20px 0 16px; border-bottom:1px solid #e2e8f0; margin-bottom:24px;">
  <h1 style="font-size:28px; font-weight:800; color:#1e293b; margin:0 0 8px; letter-spacing:-0.5px;">
    FlowScore
  </h1>
  <p style="color:#475569; margin:0 0 4px; font-size:15px;">
    Cash flow credit scoring built from bank transaction behavior
  </p>
  <p style="color:#94a3b8; font-size:12px; margin:0 0 14px;">
    Transactions &rarr; DistilBERT categorizer &rarr; Feature engineering &rarr; CatBoost + Optuna
  </p>
  <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap;">
    <span style="background:#eff6ff; color:#1d4ed8; font-size:11px; font-weight:600;
                 padding:4px 12px; border-radius:999px;">CatBoost AUC 0.764</span>
    <span style="background:#f0fdf4; color:#15803d; font-size:11px; font-weight:600;
                 padding:4px 12px; border-radius:999px;">61.3% of rejections are creditworthy</span>
    <span style="background:#fffbeb; color:#92400e; font-size:11px; font-weight:600;
                 padding:4px 12px; border-radius:999px;">30.4% loss reduction at 70% approval</span>
    <span style="background:#faf5ff; color:#6b21a8; font-size:11px; font-weight:600;
                 padding:4px 12px; border-radius:999px;">DistilBERT 99.7% accuracy</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Browse Test Consumers", "Score a Consumer", "Model Performance"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Browse
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        "Explore **1,250 pre-scored consumers** from the held-out test set. "
        "Select an archetype and use the slider to step through individual consumers."
    )

    if scores_df.empty:
        st.warning("No pre-computed scores found. Run `python src/model.py` first.")
    else:
        col_a, col_b = st.columns([2, 3])
        with col_a:
            arch_label = st.selectbox(
                "Consumer Archetype",
                list(ARCHETYPE_LABELS.values()),
                index=0,
            )
        arch_key = next(k for k, v in ARCHETYPE_LABELS.items() if v == arch_label)
        subset = scores_df[scores_df["archetype"] == arch_key].reset_index(drop=True)

        with col_b:
            idx = st.slider(
                f"Consumer index (0 to {max(len(subset)-1, 0)})",
                min_value=0, max_value=max(len(subset)-1, 0), value=0, step=1,
            )

        if len(subset) == 0:
            st.warning(f"No consumers found for archetype: {arch_label}")
        else:
            row = subset.iloc[idx]
            flow_score   = int(row["flowscore"])
            trad_score   = int(row["traditional_score"])
            default_prob = float(row["default_probability"])
            actual_def   = bool(row["actual_default"])
            consumer_id  = row.get("consumer_id", idx)

            # Row 1: FlowScore | Traditional Score
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(render_score_card(flow_score, default_prob), unsafe_allow_html=True)
            with c2:
                st.markdown(render_trad_card(trad_score), unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # Row 2: Model Agreement | Consumer Info
            c3, c4 = st.columns([3, 2])
            with c3:
                st.markdown(render_quadrant_card(flow_score, trad_score, actual_def), unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
                            padding:20px; height:100%;">
                  <div class="label-tag" style="color:#64748b; margin-bottom:10px;">Consumer Info</div>
                  <div style="font-size:14px; font-weight:700; color:#1e293b; margin-bottom:8px;">
                    Consumer {consumer_id}
                  </div>
                  <div class="label-tag" style="color:#6366f1; margin-bottom:4px;">Archetype</div>
                  <div style="font-size:14px; font-weight:600; color:#1e293b; margin-bottom:4px;">{arch_label}</div>
                  <div style="font-size:12px; color:#64748b; line-height:1.5; margin-bottom:12px;">
                    {ARCHETYPE_DESC.get(arch_key, "")}
                  </div>
                  <div style="font-size:12px; color:#94a3b8;">
                    Consumer {idx+1} of {len(subset)} in this archetype
                  </div>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Score a Custom Consumer
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        "Enter a financial profile to get a live FlowScore "
        "from the trained CatBoost model, along with the top contributing factors."
    )

    if MODEL_BUNDLE is None:
        st.warning("Model not loaded. Run `python src/model.py` first.")
    else:
        # Preset loader (outside form so it updates inputs before form submission)
        preset_name = st.selectbox(
            "Load a preset profile (optional)",
            ["-- choose --"] + list(EXAMPLE_PROFILES.keys()),
            index=0,
        )
        preset = EXAMPLE_PROFILES.get(preset_name, {})

        def pv(key, fallback):
            return preset.get(key, fallback)

        with st.form("score_form"):
            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("**Income and Cashflow**")
                income     = st.number_input("Monthly income ($)",                value=float(pv("income",     4500)), min_value=0.0, step=100.0)
                spending   = st.number_input("Monthly discretionary spending ($)", value=float(pv("spending",   1200)), min_value=0.0, step=100.0)
                regularity = st.slider("Income regularity (0 = erratic, 1 = perfectly regular)",
                                       0.0, 1.0, float(pv("regularity", 0.85)), step=0.05)
                savings    = st.slider("Savings rate (-50% to +50% of income)",
                                       -0.5, 0.5, float(pv("savings",    0.10)), step=0.02, format="%.0f%%")
                trend      = st.number_input("Monthly balance change ($ per month)", value=float(pv("trend", 50.0)), step=10.0)

            with col_r:
                st.markdown("**Obligations and Risk Flags**")
                housing    = st.number_input("Monthly housing payment ($)",  value=float(pv("housing",  1400)), min_value=0.0, step=50.0)
                loans      = st.number_input("Monthly loan payments ($)",    value=float(pv("loans",    300)),  min_value=0.0, step=50.0)
                overdrafts = st.slider("Overdraft events (last 12 months)",  0, 20, int(pv("overdrafts", 0)))
                gambling   = st.number_input("Monthly gambling spend ($)",   value=float(pv("gambling", 0)),    min_value=0.0, step=10.0)
                payday     = st.checkbox("Has active payday loan",           value=bool(pv("payday", False)))
                bnpl       = st.checkbox("Has active BNPL obligations",      value=bool(pv("bnpl",   False)))

            submitted = st.form_submit_button("Calculate FlowScore", use_container_width=True, type="primary")

        if submitted:
            feature_names = MODEL_BUNDLE["feature_names"]
            feature_means = MODEL_BUNDLE["feature_means"]
            model         = MODEL_BUNDLE["model"]
            scaler        = MODEL_BUNDLE["scaler"]

            fv    = build_feature_vector(income, spending, housing, loans, regularity,
                                         overdrafts, payday, gambling, bnpl, savings, trend,
                                         feature_names, feature_means)
            fv_df = pd.DataFrame([fv], columns=feature_names)

            try:
                fv_scaled = scaler.transform(fv_df) if scaler else fv_df.values
            except Exception:
                fv_scaled = fv_df.values

            try:
                default_prob = float(model.predict_proba(fv_scaled)[0][1])
            except Exception:
                default_prob = float(model.predict_proba(fv_df)[0][1])

            flow_score = max(300, min(850, int(300 + 550 * (1 - default_prob))))

            try:
                fi_array = model.feature_importances_
                feature_importances = dict(zip(feature_names, fi_array))
            except Exception:
                feature_importances = {f: 1.0 for f in feature_names}

            codes = get_reason_codes(fv, feature_names, feature_means, feature_importances)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            r1, r2 = st.columns([3, 2])
            with r1:
                st.markdown(render_score_card(flow_score, default_prob), unsafe_allow_html=True)
            with r2:
                st.markdown(render_summary_card(income, spending, housing, loans, regularity,
                                                overdrafts, savings, trend, payday, bnpl, gambling),
                            unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(render_reason_codes(codes), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        "Performance summary across all models trained on the "
        "5,000-consumer synthetic dataset with 45 engineered features."
    )

    if not results:
        st.warning("No model results found. Run `python src/model.py` first.")
    else:
        mc = results.get("model_comparison", {})
        model_display = {
            "traditional_only":          "Traditional Score (baseline)",
            "logistic_regression":       "Logistic Regression",
            "gradient_boosting":         "XGBoost",
            "lightgbm_optuna":           "LightGBM + Optuna",
            "catboost_optuna":           "CatBoost + Optuna",
            "combined (trad + best CF)": "Combined: Traditional + Best CF",
        }

        rows = []
        for key, display in model_display.items():
            if key in mc:
                m = mc[key]
                rows.append({
                    "Model": display,
                    "AUC-ROC": round(m.get("auc", 0), 4),
                    "KS Stat": round(m.get("ks", 0), 4),
                    "Gini":    round(m.get("gini", 0), 4),
                })

        df_perf = pd.DataFrame(rows)
        best_idx = df_perf["AUC-ROC"].idxmax()

        st.dataframe(
            df_perf.style.apply(
                lambda row: ["background-color: #f0fdf4; font-weight: bold" if row.name == best_idx
                             else "" for _ in row],
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

        bv = results.get("business_value", {})
        if bv:
            st.markdown("#### Business Value")
            lc = bv.get("loss_comparison", [])
            if lc:
                peak = max(lc, key=lambda x: x.get("loss_reduction_pct", 0))
                c1, c2, c3 = st.columns(3)
                c1.metric("Peak Loss Reduction",    f"{peak['loss_reduction_pct']:.1f}%",
                          f"at {int(peak['target_approval_rate']*100)}% approval rate")
                mo = bv.get("missed_opportunity", {})
                c2.metric("Creditworthy Rejections", f"{mo.get('repay_pct', 0):.1%}",
                          f"{mo.get('would_repay', 0)} of {mo.get('total_rejected', 0)} rejected consumers")
                av = bv.get("avoidable_risk", {})
                c3.metric("Avoidable Risk Caught",   f"{av.get('cf_caught', 0)} defaults",
                          f"{av.get('cf_flagged', 0)} total flagged")

        fi = results.get("feature_importance", {})
        if fi:
            st.markdown("#### Top Features by Importance")
            sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
            max_fi = sorted_fi[0][1]
            fi_df = pd.DataFrame(sorted_fi, columns=["Feature", "Importance"])
            fi_df["Relative"] = fi_df["Importance"] / max_fi
            st.dataframe(
                fi_df[["Feature", "Importance"]].style.bar(
                    subset=["Importance"], color="#3b82f620"
                ),
                use_container_width=True,
                hide_index=True,
            )

        roc_path = os.path.join(DATA_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curves: All Models", use_container_width=True)
