"""
Fintech Credit Risk Scoring & Explainable AI Platform.
Interactive Executive Dashboard with Modern Light SaaS Aesthetic (Matching Reference Design).
Features:
- Executive Overview & Portfolio EDA
- Real-Time Loan Origination & Underwriting Engine
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
    page_title="Credit Risk Scoring & Explainability Dashboard",
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
# Top Navigation Bar
# ==============================================================================
st.markdown("""
<div class="top-navbar">
    <div class="top-navbar-left">
        <div class="app-logo">✦</div>
        <div class="app-title-text">Dashboard</div>
    </div>
    <div class="top-navbar-right">
        <div class="search-pill">
            <span>🔍</span> Search loan, applicant, analytics...
        </div>
        <div class="icon-circle-btn">💬</div>
        <div class="icon-circle-btn">🔔</div>
        <div class="user-avatar-pill">SV</div>
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:20px;">
        <div style="width:34px; height:34px; border-radius:10px; background:#0F172A; display:flex; align-items:center; justify-content:center; color:#38BDF8; font-weight:800;">✦</div>
        <div style="font-weight:700; font-size:1.1rem; color:#0F172A;">Fintech Underwriting</div>
    </div>
    """, unsafe_allow_html=True)

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
        index=0
    )

    st.markdown("---")
    st.markdown("<h4 style='font-size:0.9rem; color:#64748B; text-transform:uppercase;'>Global Policy Cutoff</h4>", unsafe_allow_html=True)
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
    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:12px; font-size: 0.8rem; color: #475569;">
        <strong>API Service:</strong> <code>http://127.0.0.1:8000</code><br>
        <strong>Documentation:</strong> <code>/docs</code><br>
        <strong>Engine:</strong> <span style="color:#059669; font-weight:600;">● Online (WOE + XGBoost)</span>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# Page 1: Executive Overview & Portfolio EDA (Exact Reference Dashboard Look)
# ==============================================================================
if navigation == "📊 Executive Overview & EDA":
    df = artifacts['portfolio_df']
    metrics = artifacts['metrics']

    # Hero Portfolio Header
    total_exposure = df['loan_amnt'].sum()
    st.markdown(f"""
    <div class="hero-section">
        <div>
            <div class="hero-label">Total Portfolio Exposure</div>
            <div class="hero-metric-group">
                <div class="hero-big-number">${total_exposure:,.2f}</div>
                <div class="hero-pill-badge">7.9% ↑</div>
            </div>
            <div class="hero-subtext">Prev. ${(total_exposure * 0.925):,.2f} &nbsp;•&nbsp; Active Seasoned Loans: {len(df):,}</div>
        </div>
        <div class="hero-action-pills">
            <div class="date-range-badge">
                <span>📅</span> 10 Oct 2026 - 16 Oct 2026 <span>▾</span>
            </div>
            <div class="quick-action-icon">🔍</div>
            <div class="quick-action-icon">📥</div>
            <div class="quick-action-icon">📤</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 KPI Mini-Cards in Top Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        def_rate = df['loan_status'].mean()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon-row">
                <div class="kpi-icon-disc kpi-icon-cyan">📊</div>
                <div class="kpi-arrow-link">›</div>
            </div>
            <div>
                <div class="kpi-title">Default Rate (PD)</div>
                <div class="kpi-number">{def_rate:.1%}</div>
            </div>
            <div class="kpi-footer-row">
                <span class="kpi-pill kpi-pill-green">+0.8% ↑</span>
                <span style="font-size:0.75rem; color:#94A3B8;">vs Benchmark</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        approved_vol = total_exposure * (1.0 - def_rate)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon-row">
                <div class="kpi-icon-disc kpi-icon-teal">🛡️</div>
                <div class="kpi-arrow-link">›</div>
            </div>
            <div>
                <div class="kpi-title">Performing Capital</div>
                <div class="kpi-number">${approved_vol:,.0f}</div>
            </div>
            <div class="kpi-footer-row">
                <span class="kpi-pill kpi-pill-green">+7.9% ↑</span>
                <span style="font-size:0.75rem; color:#94A3B8;">MoM Growth</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        avg_score = df['fico_score'].mean()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon-row">
                <div class="kpi-icon-disc kpi-icon-pink">💳</div>
                <div class="kpi-arrow-link">›</div>
            </div>
            <div>
                <div class="kpi-title">Average FICO Score</div>
                <div class="kpi-number">{avg_score:.0f}</div>
            </div>
            <div class="kpi-footer-row">
                <span class="kpi-pill kpi-pill-red">-2.1% ↓</span>
                <span style="font-size:0.75rem; color:#94A3B8;">Prime Median</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        auc_val = metrics['challenger_xgboost']['auc']
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon-row">
                <div class="kpi-icon-disc kpi-icon-blue">🚀</div>
                <div class="kpi-arrow-link">›</div>
            </div>
            <div>
                <div class="kpi-title">Challenger Model AUC</div>
                <div class="kpi-number">{auc_val:.3f}</div>
            </div>
            <div class="kpi-footer-row">
                <span class="kpi-pill kpi-pill-green">+8% ↑</span>
                <span style="font-size:0.75rem; color:#94A3B8;">KS: 57.7%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Middle Row: 2 Line Charts + 1 Vertical Bar Chart
    m_col1, m_col2, m_col3 = st.columns([1.2, 1.2, 1.0])

    # Generate smooth trend points
    dates = pd.date_range(start="2026-07-10", end="2026-08-10", periods=25)
    np.random.seed(42)
    revenue_trend = np.linspace(25000, 140000, 25) + np.random.normal(0, 5000, 25)
    expense_trend = np.linspace(15000, 95000, 25) + np.random.normal(0, 4000, 25)

    with m_col1:
        st.markdown("""
        <div class="white-card">
            <div class="card-header-row">
                <div class="card-header-title">Origination & Score Trajectory</div>
                <div class="card-header-more">›</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig_line1 = go.Figure()
        fig_line1.add_trace(go.Scatter(
            x=dates,
            y=revenue_trend,
            mode='lines',
            line=dict(color='#0D9488', width=2.5, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(13, 148, 136, 0.08)',
            name='Originations ($)'
        ))
        # Marker for current focal point
        fig_line1.add_trace(go.Scatter(
            x=[dates[12]],
            y=[revenue_trend[12]],
            mode='markers+text',
            marker=dict(color='#0F172A', size=8, line=dict(color='#FFFFFF', width=2)),
            text=[f"${revenue_trend[12]:,.0f}"],
            textposition="top center",
            showlegend=False
        ))
        fig_line1.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, color='#94A3B8'),
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9', color='#94A3B8'),
            showlegend=False
        )
        st.plotly_chart(fig_line1, use_container_width=True)

    with m_col2:
        st.markdown("""
        <div class="white-card">
            <div class="card-header-row">
                <div class="card-header-title">Risk Spread & Delinquency Trajectory</div>
                <div class="card-header-more">›</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig_line2 = go.Figure()
        fig_line2.add_trace(go.Scatter(
            x=dates,
            y=expense_trend,
            mode='lines',
            line=dict(color='#F97316', width=2.5, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(249, 115, 22, 0.08)',
            name='Expected Credit Loss'
        ))
        fig_line2.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, color='#94A3B8'),
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9', color='#94A3B8'),
            showlegend=False
        )
        st.plotly_chart(fig_line2, use_container_width=True)

    with m_col3:
        st.markdown("""
        <div class="white-card">
            <div class="card-header-row">
                <div class="card-header-title">Loan Inflow by Day</div>
                <div class="card-header-more">›</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sample_days = ['Jul 10', 'Jul 11', 'Jul 12', 'Jul 13', 'Jul 14', 'Jul 15', 'Jul 16']
        inflows = [68, 42, 62, 92, 54, 34, 58]
        fig_bar = go.Figure(go.Bar(
            x=sample_days,
            y=inflows,
            marker_color='#7DD3FC',
            marker_line_width=0,
            width=0.45
        ))
        fig_bar.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, color='#94A3B8'),
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9', color='#94A3B8'),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Bottom Row: 2 Donut Charts + 1 List Card (Exact Reference Look)
    b_col1, b_col2, b_col3 = st.columns([1.2, 1.2, 1.0])

    with b_col1:
        st.markdown("""
        <div class="white-card">
            <div class="card-header-row">
                <div class="card-header-title">Portfolio by Credit Grade</div>
                <div class="card-header-more">›</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        d1_col_legend, d1_col_chart = st.columns([1, 1.2])
        with d1_col_legend:
            st.markdown("""
            <div style="padding-top: 15px;">
                <div class="legend-dot-item">
                    <div class="legend-dot-left"><div class="legend-dot" style="background:#0284C7;"></div> Grade A & B (Prime)</div>
                    <div class="legend-amount">48.9%</div>
                </div>
                <div class="legend-dot-item">
                    <div class="legend-dot-left"><div class="legend-dot" style="background:#0F172A;"></div> Grade C & D (Near-Prime)</div>
                    <div class="legend-amount">31.9%</div>
                </div>
                <div class="legend-dot-item">
                    <div class="legend-dot-left"><div class="legend-dot" style="background:#94A3B8;"></div> Grade E, F, G (Subprime)</div>
                    <div class="legend-amount">19.2%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with d1_col_chart:
            fig_donut1 = go.Figure(data=[go.Pie(
                labels=['Prime (A/B)', 'Near-Prime (C/D)', 'Subprime (E/F/G)'],
                values=[48.9, 31.9, 19.2],
                hole=0.72,
                marker=dict(colors=['#0284C7', '#0F172A', '#94A3B8']),
                textinfo='none'
            )])
            fig_donut1.add_annotation(
                text="<b>$782,523</b><br><span style='font-size:10px; color:#94A3B8;'>Total Volume</span>",
                x=0.5, y=0.5, font_size=16, font_color="#0F172A", showarrow=False
            )
            fig_donut1.update_layout(
                height=180,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig_donut1, use_container_width=True)

    with b_col2:
        st.markdown("""
        <div class="white-card">
            <div class="card-header-row">
                <div class="card-header-title">Capital Allocation by Purpose</div>
                <div class="card-header-more">›</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        d2_col_legend, d2_col_chart = st.columns([1, 1.2])
        with d2_col_legend:
            st.markdown("""
            <div style="padding-top: 15px;">
                <div class="legend-dot-item">
                    <div class="legend-dot-left"><div class="legend-dot" style="background:#38BDF8;"></div> Debt Consolidation</div>
                    <div class="legend-amount">55%</div>
                </div>
                <div class="legend-dot-item">
                    <div class="legend-dot-left"><div class="legend-dot" style="background:#0F172A;"></div> Credit Card Refinance</div>
                    <div class="legend-amount">25%</div>
                </div>
                <div class="legend-dot-item">
                    <div class="legend-dot-left"><div class="legend-dot" style="background:#CBD5E1;"></div> Home & Business</div>
                    <div class="legend-amount">20%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with d2_col_chart:
            fig_donut2 = go.Figure(data=[go.Pie(
                labels=['Debt Consolidation', 'Credit Card', 'Home & Business'],
                values=[55, 25, 20],
                hole=0.72,
                marker=dict(colors=['#38BDF8', '#0F172A', '#CBD5E1']),
                textinfo='none'
            )])
            fig_donut2.add_annotation(
                text="<b>$508,639</b><br><span style='font-size:10px; color:#94A3B8;'>Allocated</span>",
                x=0.5, y=0.5, font_size=16, font_color="#0F172A", showarrow=False
            )
            fig_donut2.update_layout(
                height=180,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig_donut2, use_container_width=True)

    with b_col3:
        st.markdown("""
        <div class="white-card">
            <div class="card-header-row">
                <div class="card-header-title">Recent Underwriting Decisions</div>
                <div class="card-header-more">›</div>
            </div>
            <div>
                <div class="list-item-row">
                    <div class="list-item-left">
                        <div class="list-item-icon">📄</div>
                        <div>
                            <div class="list-item-title">Application #001</div>
                            <div class="list-item-subtitle">FICO 770 • Verified</div>
                        </div>
                    </div>
                    <span class="list-item-badge badge-design">Approved</span>
                </div>
                <div class="list-item-row">
                    <div class="list-item-left">
                        <div class="list-item-icon">📄</div>
                        <div>
                            <div class="list-item-title">Application #002</div>
                            <div class="list-item-subtitle">FICO 675 • Source Ver.</div>
                        </div>
                    </div>
                    <span class="list-item-badge badge-marketing">Review</span>
                </div>
                <div class="list-item-row">
                    <div class="list-item-left">
                        <div class="list-item-icon">📄</div>
                        <div>
                            <div class="list-item-title">Application #003</div>
                            <div class="list-item-subtitle">FICO 615 • High DTI</div>
                        </div>
                    </div>
                    <span class="list-item-badge badge-danger">Declined</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# Page 2: Real-time Loan Origination & Underwriting
# ==============================================================================
elif navigation == "💳 Loan Origination & Scoring":
    st.markdown("""
    <div class="white-card">
        <div class="card-header-title" style="font-size:1.4rem;">💳 Real-Time Loan Application Underwriting</div>
        <div style="font-size:0.88rem; color:#64748B; margin-top:4px;">
            Input applicant financials or select an instant borrower persona for real-time scoring and decisioning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Persona Presets
    persona_col1, persona_col2, persona_col3, persona_col4 = st.columns(4)
    with persona_col1:
        load_prime = st.button("🌟 Preset: Prime Borrower", use_container_width=True)
    with persona_col2:
        load_near_prime = st.button("⚖️ Preset: Near-Prime", use_container_width=True)
    with persona_col3:
        load_subprime = st.button("⚠️ Preset: Subprime", use_container_width=True)
    with persona_col4:
        load_random = st.button("🎲 Random Applicant", use_container_width=True)

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

    # Input Form in Clean Card
    with st.form("applicant_form"):
        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown("<h4 style='color:#0F172A; font-size:1rem; margin-bottom:12px;'>Loan Details</h4>", unsafe_allow_html=True)
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
            st.markdown("<h4 style='color:#0F172A; font-size:1rem; margin-bottom:12px;'>Employment & Income</h4>", unsafe_allow_html=True)
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
            grade = st.selectbox("Lending Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=['A', 'B', 'C', 'D', 'E', 'F', 'G'].index(d['grade']) if d['grade'] in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] else 1)

        with colC:
            st.markdown("<h4 style='color:#0F172A; font-size:1rem; margin-bottom:12px;'>Bureau Risk Factors</h4>", unsafe_allow_html=True)
            fico_score = st.slider("FICO Score", 500.0, 850.0, float(d['fico_score']), step=1.0)
            dti = st.slider("Debt-to-Income DTI (%)", 0.0, 60.0, float(d['dti']), step=0.5)
            revol_util = st.slider("Revolving Line Utilization (%)", 0.0, 110.0, float(d['revol_util']), step=0.5)
            revol_bal = st.number_input("Revolving Balance ($)", 0.0, 150000.0, float(d['revol_bal']), step=1000.0)
            inq_last_6mths = st.number_input("Inquiries (Past 6 Mos)", 0, 15, int(d['inq_last_6mths']))
            delinq_2yrs = st.number_input("Delinquencies (Past 2 Yrs)", 0, 10, int(d['delinq_2yrs']))
            open_acc = st.number_input("Open Accounts", 1, 40, int(d['open_acc']))
            total_acc = st.number_input("Total Credit Lines", 1, 80, int(d['total_acc']))
            mort_acc = st.number_input("Mortgage Accounts", 0, 10, int(d['mort_acc']))
            pub_rec = st.number_input("Public Records", 0, 10, int(d['pub_rec']))
            pub_rec_bankruptcies = st.number_input("Bankruptcies", 0, 5, int(d['pub_rec_bankruptcies']))

        st.markdown("<hr style='border:none; border-top:1px solid #EEF2F6; margin: 15px 0;'>", unsafe_allow_html=True)
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

    st.session_state.scored_result = {
        'credit_score': credit_score,
        'pd_val': pd_val,
        'risk_rating': risk_rating,
        'decision': decision,
        'model_used': selected_model,
        'applicant': current_applicant,
        'points_breakdown': points_breakdown
    }

    # Results Display in White Card
    st.markdown("""
    <div class="white-card">
        <div class="card-header-title" style="font-size:1.2rem; margin-bottom:16px;">🎯 Underwriting Decision & Credit Scorecard Result</div>
    </div>
    """, unsafe_allow_html=True)

    res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

    with res_col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            title={'text': "FICO-Scale Credit Score", 'font': {'size': 18, 'color': '#0F172A'}},
            number={'font': {'size': 44, 'color': '#0F172A'}},
            gauge={
                'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                'bar': {'color': "#0284C7", 'thickness': 0.22},
                'bgcolor': "#FFFFFF",
                'steps': [
                    {'range': [300, 580], 'color': 'rgba(239, 68, 68, 0.25)'},
                    {'range': [580, 670], 'color': 'rgba(245, 158, 11, 0.25)'},
                    {'range': [670, 740], 'color': 'rgba(56, 189, 248, 0.25)'},
                    {'range': [740, 850], 'color': 'rgba(16, 185, 129, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': "#0F172A", 'width': 3},
                    'thickness': 0.75,
                    'value': credit_score
                }
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.markdown(f"""
        <div class="white-card" style="padding:16px;">
            <div style="font-size:0.8rem; color:#64748B; font-weight:600; text-transform:uppercase;">Probability of Default (PD)</div>
            <div style="font-size:2rem; font-weight:800; color:#0F172A; margin:4px 0;">{pd_val:.2%}</div>
            <div style="font-size:0.8rem; color:#D97706; font-weight:600;">Policy Cutoff: {global_threshold:.1%}</div>
        </div>
        <div class="white-card" style="padding:16px;">
            <div style="font-size:0.8rem; color:#64748B; font-weight:600; text-transform:uppercase;">Assigned Risk Rating</div>
            <div style="font-size:1.6rem; font-weight:800; color:#0F172A; margin:4px 0;">{risk_rating}</div>
            <div style="font-size:0.8rem; color:#64748B;">Internal Rating Scale</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col3:
        st.markdown("<div class='white-card' style='padding:20px; height:100%;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.85rem; color:#64748B; font-weight:600; margin-bottom:8px;'>Automated Credit Decision</div>", unsafe_allow_html=True)
        if decision == "APPROVED":
            st.markdown("""
            <div class="decision-badge-approved">✅ APPROVED</div>
            <p style="margin-top: 12px; color: #475569; font-size: 0.88rem; line-height: 1.5;">
                Applicant risk metrics satisfy risk appetite parameters. Automatic origination and loan servicing approved.
            </p>
            """, unsafe_allow_html=True)
        elif decision == "MANUAL_REVIEW":
            st.markdown("""
            <div class="decision-badge-review">⚠️ MANUAL REVIEW</div>
            <p style="margin-top: 12px; color: #475569; font-size: 0.88rem; line-height: 1.5;">
                Marginal risk tier. Forward file to Senior Credit Underwriter for second-tier debt service capacity review.
            </p>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="decision-badge-declined">❌ DECLINED</div>
            <p style="margin-top: 12px; color: #475569; font-size: 0.88rem; line-height: 1.5;">
                Probability of default exceeds maximum portfolio cutoff. Regulatory Adverse Action Notice generated.
            </p>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# Page 3: Explainability & Adverse Action (ECOA / FCRA)
# ==============================================================================
elif navigation == "🔍 Explainability & Adverse Action":
    st.markdown("""
    <div class="white-card">
        <div class="card-header-title" style="font-size:1.4rem;">🔍 Explainable AI & Regulatory Adverse Action Center</div>
        <div style="font-size:0.88rem; color:#64748B; margin-top:4px;">
            Compliant with Equal Credit Opportunity Act (ECOA / Regulation B) and Fair Credit Reporting Act (FCRA).
        </div>
    </div>
    """, unsafe_allow_html=True)

    applicant = st.session_state.get('current_applicant', None)
    if not applicant:
        applicant = artifacts['portfolio_df'].iloc[0].drop('loan_status').to_dict()

    explainer_engine: CreditExplainabilityEngine = artifacts['explainer_engine']
    explanation = explainer_engine.explain_applicant(applicant, top_n_reasons=4)

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.markdown("<div class='white-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header-title' style='color:#DC2626;'>🚨 Top Risk Drivers (Adverse Factors)</div>", unsafe_allow_html=True)
        for item in explanation['top_risk_drivers']:
            shap_v = item['shap_value']
            color = "#DC2626" if shap_v > 0 else "#059669"
            st.markdown(f"""
            <div style="background:#FEF2F2; border-left:4px solid {color}; padding:10px 14px; margin-bottom:8px; border-radius:8px;">
                <strong style="color:#0F172A;">{item['feature']}</strong>: <code>{item['feature_value']}</code><br>
                <span style="font-size:0.82rem; color:#64748B;">SHAP Attribution: <span style="color:{color}; font-weight:700;">{shap_v:+.4f}</span></span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_exp2:
        st.markdown("<div class='white-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header-title' style='color:#059669;'>🛡️ Top Protective Drivers (Positive Factors)</div>", unsafe_allow_html=True)
        for item in explanation['top_protective_drivers']:
            shap_v = item['shap_value']
            color = "#059669" if shap_v < 0 else "#DC2626"
            st.markdown(f"""
            <div style="background:#ECFDF5; border-left:4px solid {color}; padding:10px 14px; margin-bottom:8px; border-radius:8px;">
                <strong style="color:#0F172A;">{item['feature']}</strong>: <code>{item['feature_value']}</code><br>
                <span style="font-size:0.82rem; color:#64748B;">SHAP Attribution: <span style="color:{color}; font-weight:700;">{shap_v:+.4f}</span></span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # SHAP Waterfall in Clean White Card
    st.markdown("<div class='white-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header-title'>📊 Feature Contribution Breakdown (SHAP Waterfall)</div>", unsafe_allow_html=True)
    contributions_df = pd.DataFrame(explanation['all_contributions']).sort_values(by='shap_value', ascending=True)

    fig_shap = px.bar(
        contributions_df,
        x='shap_value',
        y='feature',
        orientation='h',
        color='risk_impact',
        color_discrete_map={'Increases Risk': '#F87171', 'Decreases Risk': '#34D399'},
        text=[f"{v:+.3f}" for v in contributions_df['shap_value']],
        labels={'shap_value': 'SHAP Contribution to Log-Odds', 'feature': 'Applicant Feature'}
    )
    fig_shap.add_vline(x=0, line_color="#94A3B8", line_width=1.5)
    fig_shap.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9', color='#64748B'),
        yaxis=dict(color='#0F172A'),
        legend_title="Risk Impact"
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Formal Legal Notice Generator
    st.markdown("<div class='white-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header-title'>📑 Official ECOA / FCRA Regulatory Denial Notice</div>", unsafe_allow_html=True)
    reasons = explanation['adverse_action_reasons']
    for idx, r in enumerate(reasons, 1):
        st.markdown(f"""
        **Reason {idx}: [{r['code']}] {r['title']}**  
        *Regulatory Detail:* {r['description']} (Reported: `{r['applicant_value']}`)
        """)

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
        label="📥 Download Formal Adverse Action Notice (.txt)",
        data=notice_doc,
        file_name=f"Adverse_Action_Notice_{applicant_id}.txt",
        mime="text/plain"
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# Page 4: Champion vs Challenger Benchmark (WOE vs XGBoost)
# ==============================================================================
elif navigation == "⚖️ Champion vs Challenger":
    st.markdown("""
    <div class="white-card">
        <div class="card-header-title" style="font-size:1.4rem;">⚖️ Model Risk Governance: Champion vs Challenger</div>
        <div style="font-size:0.88rem; color:#64748B; margin-top:4px;">
            Benchmarking Traditional WOE Scorecard (Champion) vs Gradient Boosted XGBoost (Challenger) under SR 11-7 guidelines.
        </div>
    </div>
    """, unsafe_allow_html=True)

    metrics = artifacts['metrics']
    champion = metrics['champion_woe']
    challenger = metrics['challenger_xgboost']

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">ROC AUC Score</div>
            <div class="kpi-number">{challenger['auc']}</div>
            <div style="font-size:0.8rem; color:#059669; font-weight:600;">Challenger vs {champion['auc']} Champion</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Gini (2*AUC-1)</div>
            <div class="kpi-number">{challenger['gini']}</div>
            <div style="font-size:0.8rem; color:#059669; font-weight:600;">Challenger vs {champion['gini']} Champion</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Kolmogorov-Smirnov (KS)</div>
            <div class="kpi-number">{challenger['ks_statistic']}</div>
            <div style="font-size:0.8rem; color:#0284C7; font-weight:600;">Max Separation: 57.7%</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Brier Loss Calibration</div>
            <div class="kpi-number">{challenger['brier_score']}</div>
            <div style="font-size:0.8rem; color:#64748B;">Lower is Better</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("<div class='white-card'><div class='card-header-title'>ROC Curves Comparison</div>", unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=champion['roc_curve']['fpr'],
            y=champion['roc_curve']['tpr'],
            mode='lines',
            name=f"Champion WOE (AUC = {champion['auc']})",
            line=dict(color='#F59E0B', width=2.5, dash='dash')
        ))
        fig_roc.add_trace(go.Scatter(
            x=challenger['roc_curve']['fpr'],
            y=challenger['roc_curve']['tpr'],
            mode='lines',
            name=f"Challenger XGBoost (AUC = {challenger['auc']})",
            line=dict(color='#0284C7', width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Baseline (AUC = 0.50)',
            line=dict(color='#94A3B8', dash='dot')
        ))
        fig_roc.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#F1F5F9', color='#64748B', title="False Positive Rate"),
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9', color='#64748B', title="True Positive Rate"),
            legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r_col2:
        st.markdown("<div class='white-card'><div class='card-header-title'>Kolmogorov-Smirnov (KS) Curve</div>", unsafe_allow_html=True)
        ks_data = challenger['ks_curve']
        fig_ks = go.Figure()
        fig_ks.add_trace(go.Scatter(
            x=ks_data['cutoffs'],
            y=ks_data['cum_goods'],
            mode='lines',
            name="Cumulative Non-Defaults (Goods)",
            line=dict(color='#059669', width=2.5)
        ))
        fig_ks.add_trace(go.Scatter(
            x=ks_data['cutoffs'],
            y=ks_data['cum_bads'],
            mode='lines',
            name="Cumulative Defaults (Bads)",
            line=dict(color='#DC2626', width=2.5)
        ))
        fig_ks.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#F1F5F9', color='#64748B', title="Probability Threshold"),
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9', color='#64748B', title="Cumulative Rate"),
            legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95)
        )
        st.plotly_chart(fig_ks, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# Page 5: Batch Scoring & Portfolio Underwriting
# ==============================================================================
elif navigation == "📁 Batch Scoring & Portfolio":
    st.markdown("""
    <div class="white-card">
        <div class="card-header-title" style="font-size:1.4rem;">📁 Batch Underwriting & Bulk Portfolio Processing</div>
        <div style="font-size:0.88rem; color:#64748B; margin-top:4px;">
            Upload loan applications in bulk to generate automated credit scores, PDs, and regulatory risk ratings.
        </div>
    </div>
    """, unsafe_allow_html=True)

    use_sample = st.checkbox("Load pre-configured sample applicant batch (50 records)", value=True)

    batch_df = None
    if use_sample:
        sample_path = ROOT_DIR / "data" / "sample_batch_applicants.csv"
        if sample_path.exists():
            batch_df = pd.read_csv(sample_path)
            st.info("Loaded `data/sample_batch_applicants.csv` (50 applications).")
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)

    if batch_df is not None:
        st.markdown("<div class='white-card'>", unsafe_allow_html=True)
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Underwriting Engine", use_container_width=True):
            with st.spinner("Scoring applications..."):
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
            st.success("Batch Underwriting Complete!")

            b_c1, b_c2, b_c3, b_c4 = st.columns(4)
            with b_c1:
                st.metric("Approval Rate", f"{(res_df['Decision'] == 'APPROVED').mean():.1%}")
            with b_c2:
                st.metric("Decline Rate", f"{(res_df['Decision'] == 'DECLINED').mean():.1%}")
            with b_c3:
                st.metric("Review Rate", f"{(res_df['Decision'] == 'MANUAL_REVIEW').mean():.1%}")
            with b_c4:
                st.metric("Mean Score", f"{res_df['Credit_Score'].mean():.1f}")

            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Portfolio (CSV)",
                data=csv_data,
                file_name="Scored_Portfolio_Applications.csv",
                mime="text/csv"
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# Page 6: Regulatory & Methodology Hub
# ==============================================================================
elif navigation == "📚 Regulatory & Methodology":
    st.markdown("""
    <div class="white-card">
        <div class="card-header-title" style="font-size:1.4rem;">📚 Fintech Regulatory & Quantitative Methodology Hub</div>
        <div style="font-size:0.88rem; color:#64748B; margin-top:4px;">
            Mathematical foundations of Weight of Evidence, Basel II IRB Scaling, and ECOA / FCRA Governance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 WOE & IV Theory", "🏛️ Basel II Scaling", "⚖️ Fair Lending Governance"])
    with tab1:
        st.markdown(r"""
        **Weight of Evidence (WOE):**
        $$\text{WOE}_i = \ln \left( \frac{\text{Distr Good}_i}{\text{Distr Bad}_i} \right)$$
        **Information Value (IV):**
        $$\text{IV} = \sum_{i=1}^{k} (\text{Distr Good}_i - \text{Distr Bad}_i) \times \text{WOE}_i$$
        """)
        st.dataframe(artifacts['iv_summary_df'], use_container_width=True)
    with tab2:
        st.markdown(r"""
        **Points-to-Double-Odds Formula:**
        $$\text{Score} = \text{Offset} + \text{Factor} \cdot \ln(\text{Odds})$$
        Where Base Points = 600 at 50:1 odds, PDO = 20.
        """)
    with tab3:
        st.markdown("""
        - **ECOA / Regulation B (12 C.F.R. § 1002.9):** Requires actionable reasons for credit denial.
        - **FCRA (15 U.S.C. § 1681m):** Guarantees right to free credit report dispute.
        - **SR 11-7:** Mandates Champion vs Challenger model validation.
        """)
