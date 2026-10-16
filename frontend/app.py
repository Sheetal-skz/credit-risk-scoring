"""
Fintech Credit Risk Scoring & Explainable AI Platform.
Interactive Web Dashboard built with Streamlit, Plotly, and FastAPI integration.
Features:
- Portfolio EDA & Demographics
- Real-Time Loan Underwriting & Scoring Engine
- SHAP Feature Attribution & Adverse Action Reason Codes (ECOA / FCRA)
- Champion (WOE Scorecard) vs Challenger (XGBoost) Benchmark (AUC / KS / Gini)
- Batch Portfolio Underwriting & Risk Distribution
- Regulatory Compliance & Basel II Methodology Documentation
"""

import os
import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup Root Directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from frontend.styles import get_custom_css
from src.woe_binning import WOETransformer
from src.explainability import CreditExplainabilityEngine, REGULATORY_REASON_CODES


# ==============================================================================
# Page Setup & Styling
# ==============================================================================
st.set_page_config(
    page_title="Fintech Credit Risk Scoring & Explainability",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(get_custom_css(), unsafe_allow_html=True)


# ==============================================================================
# Data & Artifact Caching
# ==============================================================================
@st.cache_resource
def load_all_artifacts():
    """Load pre-trained models, scorecard tables, and metadata."""
    models_dir = ROOT_DIR / "models"
    data_dir = ROOT_DIR / "data"

    woe_model: WOETransformer = joblib.load(models_dir / "woe_scorecard.pkl")
    xgb_model = joblib.load(models_dir / "xgboost_model.pkl")
    cat_encoders = joblib.load(models_dir / "cat_encoders.pkl")
    explainer_engine = CreditExplainabilityEngine(artifacts_dir=str(models_dir))

    with open(models_dir / "model_metrics.json", "r") as f:
        metrics = json.load(f)

    scorecard_df = pd.read_csv(models_dir / "scorecard_table.csv")
    iv_summary_df = pd.read_csv(models_dir / "iv_summary.csv")
    portfolio_df = pd.read_csv(data_dir / "loan_data.csv")

    return {
        'woe_model': woe_model,
        'xgb_model': xgb_model,
        'cat_encoders': cat_encoders,
        'explainer_engine': explainer_engine,
        'metrics': metrics,
        'scorecard_df': scorecard_df,
        'iv_summary_df': iv_summary_df,
        'portfolio_df': portfolio_df
    }

try:
    artifacts = load_all_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure `python src/pipeline.py` has been executed.")
    st.stop()


# Helper function to convert PD to credit score (300 to 850)
def pd_to_credit_score(pd_val: float) -> int:
    eps = 1e-6
    clamped_pd = np.clip(pd_val, eps, 1.0 - eps)
    odds = (1.0 - clamped_pd) / clamped_pd
    factor = 20.0 / np.log(2.0)
    offset = 600.0 - (factor * np.log(50.0))
    score = offset + factor * np.log(odds)
    return int(np.clip(score, 300, 850))

def assign_risk_rating(pd_value: float) -> str:
    if pd_value < 0.05: return "AAA (Prime)"
    elif pd_value < 0.10: return "AA (High Quality)"
    elif pd_value < 0.18: return "A (Upper Medium)"
    elif pd_value < 0.28: return "BBB (Investment Grade)"
    elif pd_value < 0.40: return "BB (Vulnerable)"
    elif pd_value < 0.55: return "B (Speculative)"
    elif pd_value < 0.75: return "CCC (Substantial Risk)"
    else: return "D (Imminent Default)"

def make_decision(pd_value: float, threshold: float = 0.20) -> str:
    if pd_value <= threshold:
        return "APPROVED"
    elif pd_value <= threshold + 0.12:
        return "MANUAL_REVIEW"
    else:
        return "DECLINED"


# ==============================================================================
# Header & Navigation
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Credit Risk Scoring & Explainable AI Platform</div>
    <div class="hero-subtitle">
        Basel II/III Compliant Credit Scorecard, XGBoost Risk Engine, and SHAP Explainability for Regulatory Underwriting.
    </div>
    <div class="hero-badge-strip">
        <span class="hero-badge">🏛️ Basel II / IRB Framework</span>
        <span class="hero-badge">⚖️ ECOA & FCRA Compliant</span>
        <span class="hero-badge">📊 WOE / IV Scorecard</span>
        <span class="hero-badge">🚀 XGBoost Challenger</span>
        <span class="hero-badge">🔍 SHAP Explainability</span>
        <span class="hero-badge">⚡ FastAPI + Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
    st.title("Risk Underwriting")
    
    navigation = st.radio(
        "Navigation Module",
        [
            "📊 Executive Overview & EDA",
            "💳 Loan Origination & Scoring",
            "🔍 Explainability & Adverse Action",
            "⚖️ Champion vs Challenger",
            "📁 Batch Scoring & Portfolio",
            "📚 Regulatory & Methodology"
        ],
        index=1
    )

    st.markdown("---")
    st.subheader("⚙️ Global Underwriting Rules")
    global_threshold = st.slider(
        "Approval Cutoff PD Threshold",
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.01,
        help="Applicants with default probability below this threshold are automatically approved."
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #94A3B8;">
        <strong>API Service:</strong> <code>http://127.0.0.1:8000</code><br>
        <strong>Documentation:</strong> <code>/docs</code><br>
        <strong>Status:</strong> <span style="color:#10B981;">● Online</span>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# Page 1: Executive Overview & Portfolio EDA
# ==============================================================================
if navigation == "📊 Executive Overview & EDA":
    st.header("📊 Portfolio Analytics & Exploratory Data Analysis")
    st.markdown("Overview of the active Lending Club credit portfolio, delinquency trends, and feature discrimination.")

    df = artifacts['portfolio_df']
    metrics = artifacts['metrics']

    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Portfolio Size</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-delta delta-neutral">Active Loans Monitored</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        def_rate = df['loan_status'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Portfolio Default Rate</div>
            <div class="metric-value">{def_rate:.1%}</div>
            <div class="metric-delta delta-caution">Observed Bad Loans</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_score = df['fico_score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average FICO Score</div>
            <div class="metric-value">{avg_score:.0f}</div>
            <div class="metric-delta delta-positive">Prime Portfolio Median</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_loan = df['loan_amnt'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Loan Amount</div>
            <div class="metric-value">${avg_loan:,.0f}</div>
            <div class="metric-delta delta-neutral">Unsecured Personal Loans</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Default Rate by Loan Grade (A - G)")
        grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        grade_stats = df.groupby('grade')['loan_status'].agg(['count', 'mean']).reindex(grade_order).reset_index()
        grade_stats.columns = ['grade', 'total_count', 'default_rate']

        fig_grade = px.bar(
            grade_stats,
            x='grade',
            y='default_rate',
            text=[f"{r:.1%}" for r in grade_stats['default_rate']],
            color='default_rate',
            color_continuous_scale='Reds',
            labels={'grade': 'Lending Grade', 'default_rate': 'Default Rate'}
        )
        fig_grade.update_layout(template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig_grade, use_container_width=True)

    with c2:
        st.subheader("Information Value (IV) Feature Ranking")
        iv_df = artifacts['iv_summary_df'].head(10).sort_values(by='Information_Value', ascending=True)
        fig_iv = px.bar(
            iv_df,
            x='Information_Value',
            y='Feature',
            orientation='h',
            color='Information_Value',
            color_continuous_scale='Blues',
            text=[f"{iv:.3f}" for iv in iv_df['Information_Value']],
            labels={'Information_Value': 'Information Value (IV)', 'Feature': 'Risk Feature'}
        )
        fig_iv.add_vline(x=0.3, line_dash="dash", line_color="orange", annotation_text="Strong (>0.30)")
        fig_iv.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_iv, use_container_width=True)

    # Charts Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("FICO Score vs Debt-to-Income (DTI)")
        sample_subset = df.sample(min(1500, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample_subset,
            x='fico_score',
            y='dti',
            color=sample_subset['loan_status'].map({0: 'Good (Paid)', 1: 'Bad (Default)'}),
            color_discrete_map={'Good (Paid)': '#10B981', 'Bad (Default)': '#EF4444'},
            opacity=0.6,
            labels={'fico_score': 'Bureau FICO Score', 'dti': 'Debt-to-Income Ratio (%)'}
        )
        fig_scatter.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c4:
        st.subheader("Loan Purpose Distribution & Default Risk")
        purpose_stats = df.groupby('purpose')['loan_status'].agg(['count', 'mean']).reset_index()
        purpose_stats.columns = ['purpose', 'count', 'default_rate']
        fig_purpose = px.bar(
            purpose_stats.sort_values(by='count', ascending=True),
            x='count',
            y='purpose',
            orientation='h',
            color='default_rate',
            color_continuous_scale='Viridis',
            text=[f"{r:.1%}" for r in purpose_stats.sort_values(by='count', ascending=True)['default_rate']],
            labels={'count': 'Number of Loans', 'purpose': 'Loan Purpose'}
        )
        fig_purpose.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_purpose, use_container_width=True)


# ==============================================================================
# Page 2: Real-time Loan Origination & Underwriting
# ==============================================================================
elif navigation == "💳 Loan Origination & Scoring":
    st.header("💳 Real-Time Loan Application Underwriting")
    st.markdown("Enter applicant financial details or load realistic borrower personas for instant credit risk decisioning.")

    # Persona Presets
    persona_col1, persona_col2, persona_col3, persona_col4 = st.columns(4)
    with persona_col1:
        load_prime = st.button("🌟 Preset: Prime Borrower (Low Risk)")
    with persona_col2:
        load_near_prime = st.button("⚖️ Preset: Near-Prime (Moderate Risk)")
    with persona_col3:
        load_subprime = st.button("⚠️ Preset: Subprime (High Risk)")
    with persona_col4:
        load_random = st.button("🎲 Random Applicant from Test Data")

    # Handle preset state
    if "form_data" not in st.session_state or load_prime:
        st.session_state.form_data = {
            'loan_amnt': 15000.0, 'term': '36 months', 'int_rate': 8.5, 'installment': 473.5,
            'grade': 'A', 'emp_length': '10+ years', 'home_ownership': 'MORTGAGE', 'annual_inc': 120000.0,
            'verification_status': 'Verified', 'purpose': 'debt_consolidation', 'dti': 12.5,
            'delinq_2yrs': 0, 'fico_score': 770.0, 'inq_last_6mths': 0, 'open_acc': 14,
            'pub_rec': 0, 'revol_bal': 12000.0, 'revol_util': 25.0, 'total_acc': 32,
            'mort_acc': 2, 'pub_rec_bankruptcies': 0
        }
    if load_near_prime:
        st.session_state.form_data = {
            'loan_amnt': 18000.0, 'term': '36 months', 'int_rate': 14.5, 'installment': 619.0,
            'grade': 'C', 'emp_length': '4 years', 'home_ownership': 'RENT', 'annual_inc': 62000.0,
            'verification_status': 'Source Verified', 'purpose': 'credit_card', 'dti': 22.0,
            'delinq_2yrs': 0, 'fico_score': 675.0, 'inq_last_6mths': 1, 'open_acc': 10,
            'pub_rec': 0, 'revol_bal': 16500.0, 'revol_util': 58.0, 'total_acc': 20,
            'mort_acc': 0, 'pub_rec_bankruptcies': 0
        }
    if load_subprime:
        st.session_state.form_data = {
            'loan_amnt': 25000.0, 'term': '60 months', 'int_rate': 24.5, 'installment': 728.0,
            'grade': 'F', 'emp_length': '< 1 year', 'home_ownership': 'RENT', 'annual_inc': 38000.0,
            'verification_status': 'Not Verified', 'purpose': 'small_business', 'dti': 36.5,
            'delinq_2yrs': 2, 'fico_score': 615.0, 'inq_last_6mths': 4, 'open_acc': 6,
            'pub_rec': 1, 'revol_bal': 22000.0, 'revol_util': 88.5, 'total_acc': 14,
            'mort_acc': 0, 'pub_rec_bankruptcies': 1
        }
    if load_random:
        rand_row = artifacts['portfolio_df'].sample(1).iloc[0].drop('loan_status').to_dict()
        st.session_state.form_data = rand_row

    d = st.session_state.form_data

    # Input Form
    with st.form("applicant_form"):
        st.subheader("1. Applicant Financials & Credit Bureau Profile")
        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown("**Loan Characteristics**")
            loan_amnt = st.number_input("Loan Amount ($)", 1000.0, 50000.0, float(d['loan_amnt']), step=1000.0)
            term = st.selectbox("Term", ['36 months', '60 months'], index=0 if d['term'] == '36 months' else 1)
            int_rate = st.number_input("Interest Rate (%)", 5.0, 35.0, float(d['int_rate']), step=0.25)
            installment = st.number_input("Monthly Installment ($)", 10.0, 2000.0, float(d['installment']), step=10.0)
            purpose = st.selectbox(
                "Purpose",
                ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'small_business', 'medical', 'other'],
                index=['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'small_business', 'medical', 'other'].index(d['purpose']) if d['purpose'] in ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'small_business', 'medical', 'other'] else 0
            )

        with colB:
            st.markdown("**Employment & Demographics**")
            annual_inc = st.number_input("Annual Gross Income ($)", 10000.0, 500000.0, float(d['annual_inc']), step=5000.0)
            home_ownership = st.selectbox(
                "Home Ownership",
                ['MORTGAGE', 'RENT', 'OWN', 'OTHER'],
                index=['MORTGAGE', 'RENT', 'OWN', 'OTHER'].index(d['home_ownership']) if d['home_ownership'] in ['MORTGAGE', 'RENT', 'OWN', 'OTHER'] else 0
            )
            emp_opts = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
            emp_length = st.selectbox("Employment Length", emp_opts, index=emp_opts.index(d['emp_length']) if d['emp_length'] in emp_opts else 5)
            verification_status = st.selectbox(
                "Verification Status",
                ['Verified', 'Source Verified', 'Not Verified'],
                index=['Verified', 'Source Verified', 'Not Verified'].index(d['verification_status']) if d['verification_status'] in ['Verified', 'Source Verified', 'Not Verified'] else 0
            )
            grade = st.selectbox("Assigned Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=['A', 'B', 'C', 'D', 'E', 'F', 'G'].index(d['grade']) if d['grade'] in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] else 1)

        with colC:
            st.markdown("**Credit Bureau & Risk History**")
            fico_score = st.slider("FICO Score", 500.0, 850.0, float(d['fico_score']), step=1.0)
            dti = st.slider("Debt-to-Income DTI (%)", 0.0, 60.0, float(d['dti']), step=0.5)
            revol_util = st.slider("Revolving Line Utilization (%)", 0.0, 110.0, float(d['revol_util']), step=0.5)
            revol_bal = st.number_input("Revolving Balance ($)", 0.0, 150000.0, float(d['revol_bal']), step=1000.0)
            inq_last_6mths = st.number_input("Inquiries in Past 6 Months", 0, 15, int(d['inq_last_6mths']))
            delinq_2yrs = st.number_input("Delinquencies (Past 2 Yrs)", 0, 10, int(d['delinq_2yrs']))
            open_acc = st.number_input("Open Accounts", 1, 40, int(d['open_acc']))
            total_acc = st.number_input("Total Credit Lines", 1, 80, int(d['total_acc']))
            mort_acc = st.number_input("Mortgage Accounts", 0, 10, int(d['mort_acc']))
            pub_rec = st.number_input("Public Records", 0, 10, int(d['pub_rec']))
            pub_rec_bankruptcies = st.number_input("Bankruptcies", 0, 5, int(d['pub_rec_bankruptcies']))

        st.markdown("---")
        m_col1, m_col2 = st.columns([2, 1])
        with m_col1:
            selected_model = st.radio(
                "Select Scoring Engine",
                ["🚀 XGBoost Classifier (Challenger)", "📊 WOE/IV Logistic Scorecard (Champion)"],
                horizontal=True
            )
        with m_col2:
            submit_score = st.form_submit_button("⚡ Underwrite Loan Application", use_container_width=True)

    # Scoring Execution
    current_applicant = {
        'loan_amnt': loan_amnt, 'term': term, 'int_rate': int_rate, 'installment': installment,
        'grade': grade, 'emp_length': emp_length, 'home_ownership': home_ownership,
        'annual_inc': annual_inc, 'verification_status': verification_status, 'purpose': purpose,
        'dti': dti, 'delinq_2yrs': delinq_2yrs, 'fico_score': fico_score,
        'inq_last_6mths': inq_last_6mths, 'open_acc': open_acc, 'pub_rec': pub_rec,
        'revol_bal': revol_bal, 'revol_util': revol_util, 'total_acc': total_acc,
        'mort_acc': mort_acc, 'pub_rec_bankruptcies': pub_rec_bankruptcies
    }
    st.session_state.current_applicant = current_applicant

    # Compute Predictions
    if "XGBoost" in selected_model:
        xgb_clf = artifacts['xgb_model']
        encoders = artifacts['cat_encoders']
        df_single = pd.DataFrame([current_applicant])
        for col, enc in encoders.items():
            if col in df_single.columns:
                df_single[col] = enc.transform(df_single[[col]])
        pd_val = float(xgb_clf.predict_proba(df_single)[0, 1])
        credit_score = pd_to_credit_score(pd_val)
        points_breakdown = None
    else:
        woe_model: WOETransformer = artifacts['woe_model']
        credit_score, pd_val, points_breakdown = woe_model.score_applicant(current_applicant)

    risk_rating = assign_risk_rating(pd_val)
    decision = make_decision(pd_val, threshold=global_threshold)

    # Save to session for explainability tab
    st.session_state.scored_result = {
        'credit_score': credit_score,
        'pd_val': pd_val,
        'risk_rating': risk_rating,
        'decision': decision,
        'model_used': selected_model,
        'applicant': current_applicant,
        'points_breakdown': points_breakdown
    }

    st.markdown("---")
    st.subheader("🎯 Underwriting Decision & Credit Scorecard Result")

    res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

    with res_col1:
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            title={'text': "Calculated Credit Score", 'font': {'size': 20, 'color': '#F8FAFC'}},
            gauge={
                'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                'bar': {'color': "#38BDF8", 'thickness': 0.25},
                'bgcolor': "#1E293B",
                'steps': [
                    {'range': [300, 580], 'color': 'rgba(239, 68, 68, 0.4)'},
                    {'range': [580, 670], 'color': 'rgba(245, 158, 11, 0.4)'},
                    {'range': [670, 740], 'color': 'rgba(59, 130, 246, 0.4)'},
                    {'range': [740, 850], 'color': 'rgba(16, 185, 129, 0.4)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': credit_score
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Probability of Default (PD)</div>
            <div class="metric-value">{pd_val:.2%}</div>
            <div class="metric-delta delta-caution">Policy Cutoff: {global_threshold:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Assigned Risk Rating</div>
            <div class="metric-value" style="font-size: 1.5rem;">{risk_rating}</div>
            <div class="metric-delta delta-neutral">Internal Rating Scale</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col3:
        st.markdown("**Underwriting Decision**")
        if decision == "APPROVED":
            st.markdown("""
            <div class="decision-badge-approved">✅ APPROVED</div>
            <p style="margin-top: 10px; color: #94A3B8; font-size: 0.9rem;">
                Applicant default probability is safely below the risk appetite threshold. Automated credit line extension cleared.
            </p>
            """, unsafe_allow_html=True)
        elif decision == "MANUAL_REVIEW":
            st.markdown("""
            <div class="decision-badge-review">⚠️ MANUAL REVIEW</div>
            <p style="margin-top: 10px; color: #94A3B8; font-size: 0.9rem;">
                Borderline risk profile within discretionary tolerance. Refer application to Senior Credit Underwriter for manual asset verification.
            </p>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="decision-badge-declined">❌ DECLINED</div>
            <p style="margin-top: 10px; color: #94A3B8; font-size: 0.9rem;">
                Application declined due to elevated credit risk. Adverse Action Notice required under ECOA / Regulation B.
            </p>
            """, unsafe_allow_html=True)

    # Scorecard Breakdown if WOE
    if points_breakdown:
        st.subheader("📋 Traditional Scorecard Points Attribution")
        st.markdown("Exact points contributed to total score by each risk attribute under the Points-to-Double-Odds (PDO) formula:")
        pts_df = pd.DataFrame(list(points_breakdown.items()), columns=['Feature', 'Points'])
        pts_df = pts_df.sort_values(by='Points', ascending=False)
        fig_pts = px.bar(
            pts_df,
            x='Points',
            y='Feature',
            orientation='h',
            color='Points',
            color_continuous_scale='Tealgrn',
            text='Points'
        )
        fig_pts.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_pts, use_container_width=True)


# ==============================================================================
# Page 3: Explainability & Regulatory Adverse Action (ECOA / FCRA)
# ==============================================================================
elif navigation == "🔍 Explainability & Adverse Action":
    st.header("🔍 Explainable AI & Regulatory Adverse Action Notices")
    st.markdown("""
    Under the **Equal Credit Opportunity Act (ECOA / Regulation B)** and the **Fair Credit Reporting Act (FCRA)**,
    lenders are legally mandated to disclose the principal reasons when credit is denied or unfavorable terms are granted.
    SHAP (Shapley Additive exPlanations) resolves the black-box dilemma of gradient boosting by providing mathematically exact attributions.
    """)

    applicant = st.session_state.get('current_applicant', None)
    if not applicant:
        applicant = artifacts['portfolio_df'].iloc[0].drop('loan_status').to_dict()

    explainer_engine: CreditExplainabilityEngine = artifacts['explainer_engine']
    explanation = explainer_engine.explain_applicant(applicant, top_n_reasons=4)

    # Top drivers summary
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.subheader("🚨 Top Risk Drivers (Pushed Score Lower / Increased PD)")
        for item in explanation['top_risk_drivers']:
            shap_v = item['shap_value']
            color = "#EF4444" if shap_v > 0 else "#10B981"
            st.markdown(f"""
            <div style="background: #1E293B; border-left: 4px solid {color}; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                <strong>{item['feature']}</strong>: <code>{item['feature_value']}</code><br>
                <span style="font-size: 0.85rem; color: #94A3B8;">SHAP Impact: <span style="color: {color}; font-weight: 700;">{shap_v:+.4f}</span> ({item['risk_impact']})</span>
            </div>
            """, unsafe_allow_html=True)

    with col_exp2:
        st.subheader("🛡️ Top Protective Drivers (Improved Score / Decreased PD)")
        for item in explanation['top_protective_drivers']:
            shap_v = item['shap_value']
            color = "#10B981" if shap_v < 0 else "#EF4444"
            st.markdown(f"""
            <div style="background: #1E293B; border-left: 4px solid {color}; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                <strong>{item['feature']}</strong>: <code>{item['feature_value']}</code><br>
                <span style="font-size: 0.85rem; color: #94A3B8;">SHAP Impact: <span style="color: {color}; font-weight: 700;">{shap_v:+.4f}</span> ({item['risk_impact']})</span>
            </div>
            """, unsafe_allow_html=True)

    # Interactive SHAP Waterfall Plot
    st.markdown("---")
    st.subheader("📊 Individual Feature Attribution (SHAP Waterfall)")
    contributions_df = pd.DataFrame(explanation['all_contributions']).sort_values(by='shap_value', ascending=True)

    fig_shap = px.bar(
        contributions_df,
        x='shap_value',
        y='feature',
        orientation='h',
        color='risk_impact',
        color_discrete_map={'Increases Risk': '#EF4444', 'Decreases Risk': '#10B981'},
        text=[f"{v:+.3f}" for v in contributions_df['shap_value']],
        labels={'shap_value': 'SHAP Contribution to Log-Odds of Default', 'feature': 'Applicant Attribute'}
    )
    fig_shap.add_vline(x=0, line_color="#94A3B8", line_width=1.5)
    fig_shap.update_layout(template="plotly_dark", height=500, legend_title="Risk Impact")
    st.plotly_chart(fig_shap, use_container_width=True)

    # Regulatory Adverse Action Section
    st.markdown("---")
    st.subheader("📑 ECOA / FCRA Regulatory Adverse Action Statement")
    st.markdown("Official regulatory rejection reasons mapped from SHAP attributions:")

    reasons = explanation['adverse_action_reasons']
    for idx, r in enumerate(reasons, 1):
        st.markdown(f"""
        **Reason {idx}: [{r['code']}] {r['title']}**
        - *Regulatory Description:* {r['description']}
        - *Applicant Reported Value:* `{r['applicant_value']}`
        - *Relative Adverse Contribution:* `{r['shap_impact']:+.4f}`
        """)

    # Downloadable Notice Generator
    st.markdown("### Generate Formal Adverse Action Letter")
    applicant_name = st.text_input("Applicant Full Name", "Morgan Vance")
    applicant_id = st.text_input("Application Tracking ID", "APP-2026-99042")

    stored_result = st.session_state.get('scored_result', {})
    score_display = stored_result.get('credit_score', 580)
    decision_display = stored_result.get('decision', 'DECLINED')

    notice_doc = explainer_engine.generate_adverse_action_notice(
        applicant_name=applicant_name,
        applicant_id=applicant_id,
        applicant_dict=applicant,
        credit_score=score_display,
        decision=decision_display
    )

    st.markdown(f'<div class="notice-box">{notice_doc}</div>', unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Official Adverse Action Notice (.txt)",
        data=notice_doc,
        file_name=f"Adverse_Action_Notice_{applicant_id}.txt",
        mime="text/plain"
    )


# ==============================================================================
# Page 4: Champion vs Challenger Benchmark (WOE vs XGBoost)
# ==============================================================================
elif navigation == "⚖️ Champion vs Challenger":
    st.header("⚖️ Model Risk Governance: Champion vs Challenger")
    st.markdown("""
    Under Basel Committee on Banking Supervision (BCBS) guidelines for model risk management (SR 11-7),
    institutions compare the **Champion model** (traditional, regulatory-favored WOE Scorecard) against the
    **Challenger model** (modern gradient boosted XGBoost).
    """)

    metrics = artifacts['metrics']
    champion = metrics['champion_woe']
    challenger = metrics['challenger_xgboost']

    # KPI Comparison Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ROC AUC Score</div>
            <div class="metric-value">{challenger['auc']}</div>
            <div class="metric-delta delta-positive">Challenger vs {champion['auc']} Champion</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Gini Coefficient (2*AUC-1)</div>
            <div class="metric-value">{challenger['gini']}</div>
            <div class="metric-delta delta-positive">Challenger vs {champion['gini']} Champion</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Kolmogorov-Smirnov (KS)</div>
            <div class="metric-value">{challenger['ks_statistic']}</div>
            <div class="metric-delta delta-positive">Max Population Separation</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Brier Loss (Calibration)</div>
            <div class="metric-value">{challenger['brier_score']}</div>
            <div class="metric-delta delta-neutral">Lower is Better</div>
        </div>
        """, unsafe_allow_html=True)

    # Detailed Comparison Table
    st.markdown("### Model Governance Comparison Matrix")
    comp_df = pd.DataFrame([
        {"Metric": "ROC-AUC", "Champion (WOE Scorecard)": champion['auc'], "Challenger (XGBoost)": challenger['auc'], "Delta": round(challenger['auc'] - champion['auc'], 4)},
        {"Metric": "Gini Coefficient", "Champion (WOE Scorecard)": champion['gini'], "Challenger (XGBoost)": challenger['gini'], "Delta": round(challenger['gini'] - champion['gini'], 4)},
        {"Metric": "KS Statistic", "Champion (WOE Scorecard)": champion['ks_statistic'], "Challenger (XGBoost)": challenger['ks_statistic'], "Delta": round(challenger['ks_statistic'] - champion['ks_statistic'], 4)},
        {"Metric": "Brier Score", "Champion (WOE Scorecard)": champion['brier_score'], "Challenger (XGBoost)": challenger['brier_score'], "Delta": round(challenger['brier_score'] - champion['brier_score'], 4)},
        {"Metric": "Log Loss", "Champion (WOE Scorecard)": champion['log_loss'], "Challenger (XGBoost)": challenger['log_loss'], "Delta": round(challenger['log_loss'] - champion['log_loss'], 4)},
        {"Metric": "Explainability Standard", "Champion (WOE Scorecard)": "Direct Scorecard Points", "Challenger (XGBoost)": "SHAP TreeExplainer", "Delta": "Equal Post-hoc"}
    ])
    st.table(comp_df)

    # Curves Row
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.subheader("ROC Curves Comparison")
        fig_roc = go.Figure()
        # Champion ROC
        fig_roc.add_trace(go.Scatter(
            x=champion['roc_curve']['fpr'],
            y=champion['roc_curve']['tpr'],
            mode='lines',
            name=f"Champion WOE (AUC = {champion['auc']})",
            line=dict(color='#F59E0B', width=2.5, dash='dash')
        ))
        # Challenger ROC
        fig_roc.add_trace(go.Scatter(
            x=challenger['roc_curve']['fpr'],
            y=challenger['roc_curve']['tpr'],
            mode='lines',
            name=f"Challenger XGBoost (AUC = {challenger['auc']})",
            line=dict(color='#38BDF8', width=3)
        ))
        # Diagonal random line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Baseline (AUC = 0.50)',
            line=dict(color='#64748B', dash='dot')
        ))
        fig_roc.update_layout(
            template="plotly_dark",
            height=380,
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
            legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with r_col2:
        st.subheader("Kolmogorov-Smirnov (KS) Separation Curve")
        ks_data = challenger['ks_curve']
        fig_ks = go.Figure()
        fig_ks.add_trace(go.Scatter(
            x=ks_data['cutoffs'],
            y=ks_data['cum_goods'],
            mode='lines',
            name="Cumulative Non-Defaults (Goods)",
            line=dict(color='#10B981', width=2.5)
        ))
        fig_ks.add_trace(go.Scatter(
            x=ks_data['cutoffs'],
            y=ks_data['cum_bads'],
            mode='lines',
            name="Cumulative Defaults (Bads)",
            line=dict(color='#EF4444', width=2.5)
        ))
        fig_ks.update_layout(
            template="plotly_dark",
            height=380,
            xaxis_title="Score Cutoff / Probability Threshold",
            yaxis_title="Cumulative Population Distribution",
            legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95)
        )
        st.plotly_chart(fig_ks, use_container_width=True)

    # Cutoff Optimization Simulator
    st.markdown("---")
    st.subheader("🎛️ Underwriting Cutoff Strategy & Profit/Loss Optimization")
    st.markdown("Simulate the financial trade-off of adjusting the risk threshold:")

    sim_thresh = st.slider("Simulate Decision Cutoff Threshold (PD)", 0.05, 0.50, 0.20, step=0.01)
    
    # Financial assumptions
    avg_loan_principal = 15000
    net_interest_margin = 0.08  # 8% profit on non-defaults
    loss_given_default = 0.65   # 65% loss on defaults

    test_df = pd.read_csv(ROOT_DIR / "data" / "test.csv")
    y_true = test_df['loan_status'].values
    
    # Use cached XGB predictions
    xgb_clf = artifacts['xgb_model']
    encs = artifacts['cat_encoders']
    X_test_xgb = test_df.drop(columns=['loan_status']).copy()
    for c, enc in encs.items():
        if c in X_test_xgb.columns:
            X_test_xgb[c] = enc.transform(X_test_xgb[[c]])
    y_probs = xgb_clf.predict_proba(X_test_xgb)[:, 1]

    # Decisions at sim_thresh
    approved_mask = (y_probs <= sim_thresh)
    approval_rate = np.mean(approved_mask)
    approved_bads = np.sum((y_probs <= sim_thresh) & (y_true == 1))
    approved_goods = np.sum((y_probs <= sim_thresh) & (y_true == 0))
    portfolio_bad_rate = approved_bads / max(1, (approved_bads + approved_goods))

    est_revenue = approved_goods * avg_loan_principal * net_interest_margin
    est_credit_loss = approved_bads * avg_loan_principal * loss_given_default
    net_economic_profit = est_revenue - est_credit_loss

    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
    with sim_c1:
        st.metric("Approval Rate", f"{approval_rate:.1%}")
    with sim_c2:
        st.metric("Portfolio Default Rate", f"{portfolio_bad_rate:.2%}")
    with sim_c3:
        st.metric("Expected Gross Margin", f"${est_revenue:,.0f}")
    with sim_c4:
        st.metric("Net Economic Profit", f"${net_economic_profit:,.0f}", delta=f"${net_economic_profit:,.0f}")


# ==============================================================================
# Page 5: Batch Scoring & Portfolio Underwriting
# ==============================================================================
elif navigation == "📁 Batch Scoring & Portfolio":
    st.header("📁 Batch Portfolio Underwriting & Bulk Scoring")
    st.markdown("Upload a portfolio file (CSV) to underwrite thousands of applications simultaneously.")

    # Option to use sample batch
    use_sample = st.checkbox("Load pre-configured sample applicant batch (50 records)", value=True)

    batch_df = None
    if use_sample:
        sample_path = ROOT_DIR / "data" / "sample_batch_applicants.csv"
        if sample_path.exists():
            batch_df = pd.read_csv(sample_path)
            st.info("Loaded `data/sample_batch_applicants.csv` containing 50 loan applications.")
    else:
        uploaded_file = st.file_uploader("Upload Applicants CSV", type=['csv'])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)

    if batch_df is not None:
        st.subheader("Data Preview (First 5 records)")
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Underwriting Engine", use_container_width=True):
            with st.spinner("Scoring portfolio applications..."):
                input_df = batch_df.drop(columns=['loan_status'], errors='ignore')
                xgb_clf = artifacts['xgb_model']
                encoders = artifacts['cat_encoders']

                df_enc = input_df.copy()
                for c, enc in encoders.items():
                    if c in df_enc.columns:
                        df_enc[c] = enc.transform(df_enc[[c]])

                batch_pds = xgb_clf.predict_proba(df_enc)[:, 1]
                batch_scores = [pd_to_credit_score(p) for p in batch_pds]
                batch_ratings = [assign_risk_rating(p) for p in batch_pds]
                batch_decisions = [make_decision(p, threshold=global_threshold) for p in batch_pds]

                scored_df = batch_df.copy()
                scored_df['Credit_Score'] = batch_scores
                scored_df['PD'] = np.round(batch_pds, 4)
                scored_df['Risk_Rating'] = batch_ratings
                scored_df['Decision'] = batch_decisions

                st.session_state.scored_batch = scored_df

        if "scored_batch" in st.session_state:
            res_df = st.session_state.scored_batch
            st.success("Batch scoring completed successfully!")

            b_c1, b_c2, b_c3, b_c4 = st.columns(4)
            with b_c1:
                app_rate = (res_df['Decision'] == 'APPROVED').mean()
                st.metric("Approval Rate", f"{app_rate:.1%}")
            with b_c2:
                dec_rate = (res_df['Decision'] == 'DECLINED').mean()
                st.metric("Decline Rate", f"{dec_rate:.1%}")
            with b_c3:
                rev_rate = (res_df['Decision'] == 'MANUAL_REVIEW').mean()
                st.metric("Manual Review Rate", f"{rev_rate:.1%}")
            with b_c4:
                avg_sc = res_df['Credit_Score'].mean()
                st.metric("Portfolio Mean Score", f"{avg_sc:.1f}")

            # Risk Rating Distribution
            fig_batch = px.histogram(
                res_df,
                x='Credit_Score',
                color='Decision',
                nbins=25,
                color_discrete_map={'APPROVED': '#10B981', 'MANUAL_REVIEW': '#F59E0B', 'DECLINED': '#EF4444'},
                title="Portfolio Credit Score Distribution by Underwriting Decision"
            )
            fig_batch.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_batch, use_container_width=True)

            st.dataframe(res_df[['loan_amnt', 'annual_inc', 'fico_score', 'dti', 'Credit_Score', 'PD', 'Risk_Rating', 'Decision']].head(20), use_container_width=True)

            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Enriched Scored Portfolio (CSV)",
                data=csv_data,
                file_name="Underwritten_Loan_Portfolio.csv",
                mime="text/csv"
            )


# ==============================================================================
# Page 6: Regulatory & Methodology Hub
# ==============================================================================
elif navigation == "📚 Regulatory & Methodology":
    st.header("📚 Fintech Regulatory & Quantitative Methodology Hub")
    st.markdown("""
    Credit underwriting is one of the most strictly regulated applications of Artificial Intelligence in the financial sector.
    This guide details the mathematical foundations of the Scorecard and compliance requirements.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 WOE & IV Theory",
        "🏛️ Basel II IRB Calibration",
        "⚖️ ECOA / FCRA Governance",
        "🔍 TreeExplainer SHAP Math"
    ])

    with tab1:
        st.subheader("Weight of Evidence (WOE) & Information Value (IV)")
        st.markdown(r"""
        Weight of Evidence measures the predictive strength of a specific attribute bin in separating Good ($Y=0$) from Bad ($Y=1$) borrowers:
        $$\text{WOE}_i = \ln \left( \frac{\text{Distr Good}_i}{\text{Distr Bad}_i} \right) = \ln \left( \frac{G_i / N_G}{B_i / N_B} \right)$$
        
        Information Value (IV) aggregates the total discriminative power across all bins of a variable:
        $$\text{IV} = \sum_{i=1}^{k} (\text{Distr Good}_i - \text{Distr Bad}_i) \times \text{WOE}_i$$

        **Basel / Credit Bureau IV Evaluation Benchmark:**
        - $\text{IV} < 0.02$: **Unpredictive** — drop variable.
        - $0.02 \le \text{IV} < 0.10$: **Weak predictor** — secondary screening.
        - $0.10 \le \text{IV} < 0.30$: **Medium predictor** — typical financial feature.
        - $0.30 \le \text{IV} < 0.50$: **Strong predictor** — prime credit factor.
        - $\text{IV} \ge 0.50$: **Suspicious** — check for target leakage or collinearity.
        """)

        st.subheader("Active Model Information Value Summary Table")
        st.dataframe(artifacts['iv_summary_df'], use_container_width=True)

    with tab2:
        st.subheader("Basel II Internal Ratings-Based (IRB) Scaling")
        st.markdown(r"""
        To translate statistical log-odds into standard consumer credit scores (e.g. 300 to 850), lenders apply the **Points to Double Odds (PDO)** transformation:
        $$\text{Score} = \text{Offset} + \text{Factor} \cdot \ln(\text{Odds})$$
        where:
        $$\text{Factor} = \frac{\text{PDO}}{\ln(2)}$$
        $$\text{Offset} = \text{Base Points} - \text{Factor} \cdot \ln(\text{Base Odds})$$

        In our production system:
        - $\text{Base Points} = 600$
        - $\text{Base Odds} = 50:1$
        - $\text{PDO} = 20 \text{ points}$ (a 20-point increase cuts default risk by half)
        """)

    with tab3:
        st.subheader("Fair Lending, ECOA & Model Governance")
        st.markdown("""
        **Regulatory Frameworks Applied:**
        1. **Equal Credit Opportunity Act (ECOA) / Regulation B (12 C.F.R. § 1002.9):**
           - Creditors must state specific, principal reasons when taking adverse action.
           - Discouraged generic statements like "insufficient score" — reason codes must be actionable.
        2. **Fair Credit Reporting Act (FCRA - 15 U.S.C. § 1681m):**
           - Mandates disclosure of credit bureau contact and right to obtain free consumer report within 60 days.
        3. **Federal Reserve SR 11-7 / OCC 2011-12 (Model Risk Management):**
           - Mandates independent conceptual soundness validation, benchmarking champion against challenger algorithms, and ongoing outcomes monitoring.
        """)

    with tab4:
        st.subheader("SHAP (SHapley Additive exPlanations)")
        st.markdown(r"""
        For any loan applicant $x$, the model prediction $f(x)$ is decomposed as an additive sum of feature attributions $\phi_j$:
        $$f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j(x)$$
        where $\phi_0 = \mathbb{E}[f(x)]$ is the base expected value of the portfolio.
        
        **Properties satisfying legal credit explainability:**
        - **Local Accuracy:** Sum of SHAP values matches model output exactly.
        - **Missingness:** Features with no impact receive zero attribution.
        - **Consistency:** If a feature increases risk across models, its attribution does not decrease.
        """)
