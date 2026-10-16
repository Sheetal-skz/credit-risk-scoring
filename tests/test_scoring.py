"""
Unit and Integration Tests for Credit Risk Scoring Engine.
Tests:
- WOE Transformer & Scorecard points scaling
- XGBoost Challenger probability calibration
- SHAP Explainability & Adverse Action Reason Code generator
- Basel metrics calculation (AUC, KS, Gini)
"""

import sys
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import joblib

from src.woe_binning import WOETransformer
from src.model_training import calculate_credit_metrics
from src.explainability import CreditExplainabilityEngine, REGULATORY_REASON_CODES


@pytest.fixture
def sample_applicant():
    return {
        'loan_amnt': 15000.0,
        'term': '36 months',
        'int_rate': 11.5,
        'installment': 494.8,
        'grade': 'B',
        'emp_length': '5 years',
        'home_ownership': 'MORTGAGE',
        'annual_inc': 85000.0,
        'verification_status': 'Verified',
        'purpose': 'debt_consolidation',
        'dti': 15.2,
        'delinq_2yrs': 0,
        'fico_score': 720.0,
        'inq_last_6mths': 0,
        'open_acc': 12,
        'pub_rec': 0,
        'revol_bal': 11000.0,
        'revol_util': 38.5,
        'total_acc': 25,
        'mort_acc': 2,
        'pub_rec_bankruptcies': 0
    }


def test_woe_scorecard_scaling(sample_applicant):
    woe_model: WOETransformer = joblib.load("models/woe_scorecard.pkl")
    score, pd_val, points_breakdown = woe_model.score_applicant(sample_applicant)

    # Credit score must lie within bureau range
    assert 300 <= score <= 850, f"Score {score} out of valid bureau range [300, 850]"
    # Probability of default must be valid probability
    assert 0.0 < pd_val < 1.0, f"PD {pd_val} is not a valid probability"
    # Points breakdown must cover features
    assert len(points_breakdown) > 0
    assert sum(points_breakdown.values()) > 0


def test_xgboost_and_shap(sample_applicant):
    explainer_engine = CreditExplainabilityEngine(artifacts_dir="models")
    explanation = explainer_engine.explain_applicant(sample_applicant, top_n_reasons=4)

    assert 'base_value' in explanation
    assert len(explanation['top_risk_drivers']) == 5
    assert len(explanation['top_protective_drivers']) == 5
    assert len(explanation['adverse_action_reasons']) <= 4

    for reason in explanation['adverse_action_reasons']:
        assert reason['code'] in [v['code'] for v in REGULATORY_REASON_CODES.values()]


def test_credit_metrics_calculation():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Well-calibrated probabilities
    y_prob = np.array([0.05, 0.12, 0.18, 0.22, 0.75, 0.82, 0.89, 0.95])

    metrics = calculate_credit_metrics(y_true, y_prob)

    assert metrics['auc'] == 1.0
    assert metrics['gini'] == 1.0
    assert metrics['ks_statistic'] == 1.0
    assert 0.0 <= metrics['brier_score'] <= 1.0


def test_adverse_action_notice_text(sample_applicant):
    explainer_engine = CreditExplainabilityEngine(artifacts_dir="models")
    notice = explainer_engine.generate_adverse_action_notice(
        applicant_name="Test Applicant",
        applicant_id="APP-TEST-001",
        applicant_dict=sample_applicant,
        credit_score=620,
        decision="DECLINED"
    )

    assert "REGULATION B" in notice
    assert "Equal Credit Opportunity Act" in notice
    assert "Fair Credit Reporting Act" in notice
    assert "Test Applicant" in notice
    assert "APP-TEST-001" in notice
