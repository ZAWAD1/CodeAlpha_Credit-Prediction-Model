import streamlit as st
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(
    page_title="Credit Risk Analyser",
    page_icon="💳",
    layout="centered"
)

# CSS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .main { background: #0f1117; }

  .title-block {
    text-align: center;
    padding: 2rem 0 1rem;
  }
  .title-block h1 {
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #f0f0f0;
    margin: 0;
  }
  .title-block p {
    color: #888;
    font-size: 0.9rem;
    margin-top: 0.4rem;
  }

  .section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #555;
    margin: 1.6rem 0 0.6rem;
  }

  /* Result cards */
  .result-safe {
    background: linear-gradient(135deg, #0d2b1a, #0a1f13);
    border: 1px solid #1a5c30;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
  }
  .result-risk {
    background: linear-gradient(135deg, #2b0d0d, #1f0a0a);
    border: 1px solid #5c1a1a;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
  }
  .result-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
  }
  .result-verdict {
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: -0.02em;
  }
  .result-prob {
    font-size: 0.9rem;
    margin-top: 0.4rem;
    opacity: 0.7;
  }

  /* Probability bar */
  .prob-bar-wrap {
    background: #1a1a2e;
    border-radius: 8px;
    height: 10px;
    margin: 0.8rem 0;
    overflow: hidden;
  }
  .prob-bar-fill-safe { background: #22c55e; height: 100%; border-radius: 8px; transition: width 0.6s ease; }
  .prob-bar-fill-risk { background: #ef4444; height: 100%; border-radius: 8px; transition: width 0.6s ease; }

  .stSlider > div > div { background: #1e1e2e !important; }
  .stNumberInput input { font-family: 'DM Mono', monospace; }

  .factor-box {
    background: #16161e;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #aaa;
  }
  .factor-box span { color: #f0f0f0; font-weight: 500; }

  hr { border-color: #2a2a3a; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# Model Loading
@st.cache_resource
def load_model():
    model_path = 'Model/credit_scoring_model.pkl'
    if not os.path.exists(model_path):
        model_path = 'credit_scoring_model.pkl'
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Header
st.markdown("""
<div class="title-block">
  <h1>💳 Credit Risk Analyser</h1>
  <p>Enter an applicant's financial details to assess default risk</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Form
st.markdown('<div class="section-label">Personal info</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35,
                          help="Applicant's age in years")
with col2:
    dependents = st.number_input("Number of dependents",
                                  min_value=0, max_value=20, value=1,
                                  help="People financially dependent on this person")

st.markdown('<div class="section-label">Income & Debt</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    monthly_income = st.number_input("Monthly income ($)",
                                      min_value=0, max_value=100000,
                                      value=5000, step=100,
                                      help="Self-reported monthly income in USD")
with col4:
    debt_ratio = st.slider("Debt ratio",
                            min_value=0.0, max_value=1.0, value=0.3, step=0.01,
                            help="Monthly debt payments ÷ monthly income. 0.3 = 30% of income goes to debt")

col5, col6 = st.columns(2)
with col5:
    credit_utilization = st.slider("Credit card utilization",
                                    min_value=0.0, max_value=1.5, value=0.3, step=0.01,
                                    help="How much of credit card limits are being used. 1.0 = fully maxed out")
with col6:
    open_loans = st.number_input("Total open loans & credit lines",
                                  min_value=0, max_value=50, value=5,
                                  help="Total number of open credit lines and loans")

real_estate_loans = st.number_input("Number of real estate loans / mortgages",
                                     min_value=0, max_value=20, value=1,
                                     help="Mortgages and home equity lines of credit")

st.markdown('<div class="section-label">Payment history  <span style="color:#555;font-size:0.65rem;font-weight:400;text-transform:none">(most important factor)</span></div>', unsafe_allow_html=True)

col7, col8, col9 = st.columns(3)
with col7:
    late_30_59 = st.number_input("Times 30–59 days late",
                                   min_value=0, max_value=20, value=0,
                                   help="How many times payments were 30 to 59 days overdue")
with col8:
    late_60_89 = st.number_input("Times 60–89 days late",
                                   min_value=0, max_value=20, value=0,
                                   help="How many times payments were 60 to 89 days overdue")
with col9:
    late_90 = st.number_input("Times 90+ days late",
                                min_value=0, max_value=20, value=0,
                                help="How many times payments were 90 or more days overdue (most serious)")


total_late  = late_30_59 + late_60_89 + late_90
debt_amount = debt_ratio * monthly_income

# Prediction
st.divider()
if st.button("Analyse credit risk →", use_container_width=True, type="primary"):

    input_data = pd.DataFrame([{
        'RevolvingUtilizationOfUnsecuredLines': credit_utilization,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': open_loans,
        'NumberOfTimes90DaysLate': late_90,
        'NumberRealEstateLoansOrLines': real_estate_loans,
        'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
        'NumberOfDependents': dependents,
        'total_late': total_late,
        'debt_amount': debt_amount
    }])

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    pct         = f"{probability * 100:.2f}"

    if prediction == 0:
        st.markdown(f"""
        <div class="result-safe">
          <div class="result-label" style="color:#4ade80">Low Risk</div>
          <div class="result-verdict" style="color:#4ade80">SAFE TO LEND</div>
          <div class="result-prob">{pct}% probability of default</div>
        </div>
        <div class="prob-bar-wrap">
          <div class="prob-bar-fill-safe" style="width:{pct}%"></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-risk">
          <div class="result-label" style="color:#f87171">High Risk</div>
          <div class="result-verdict" style="color:#f87171">LIKELY TO DEFAULT</div>
          <div class="result-prob">{pct}% probability of default</div>
        </div>
        <div class="prob-bar-wrap">
          <div class="prob-bar-fill-risk" style="width:{pct}%"></div>
        </div>
        """, unsafe_allow_html=True)

    
    # Features of prediction
    st.markdown('<div class="section-label">Key factors used</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f'<div class="factor-box">Total late payments <span>({total_late})</span> — strongest signal</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="factor-box">Credit utilization <span>({credit_utilization:.0%})</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="factor-box">Monthly income <span>(${monthly_income:,})</span></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown(f'<div class="factor-box">Debt ratio <span>({debt_ratio:.0%} of income)</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="factor-box">Estimated monthly debt <span>(${debt_amount:,.0f})</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="factor-box">Age <span>({age} years)</span></div>', unsafe_allow_html=True)
