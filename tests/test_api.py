"""
API Endpoint Integration Tests using FastAPI TestClient.
"""

import sys
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_payload():
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


def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_api_metrics(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "champion_woe" in data
    assert "challenger_xgboost" in data
    assert data["challenger_xgboost"]["auc"] > 0.80


def test_api_score_xgboost(client, sample_payload):
    res = client.post("/score", json={"applicant": sample_payload, "model_type": "xgboost"})
    assert res.status_code == 200
    body = res.json()
    assert 300 <= body["credit_score"] <= 850
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["decision"] in ["APPROVED", "MANUAL_REVIEW", "DECLINED"]
    assert "top_risk_drivers" in body


def test_api_score_woe(client, sample_payload):
    res = client.post("/score", json={"applicant": sample_payload, "model_type": "woe_scorecard"})
    assert res.status_code == 200
    body = res.json()
    assert 300 <= body["credit_score"] <= 850
    assert body["scorecard_points_breakdown"] is not None


def test_api_adverse_action_notice(client, sample_payload):
    res = client.post("/adverse-action/notice", json={
        "applicant_name": "Sarah Connor",
        "applicant_id": "APP-2026-X01",
        "applicant": sample_payload
    })
    assert res.status_code == 200
    assert "STATEMENT OF CREDIT DENIAL" in res.text
    assert "Sarah Connor" in res.text


def test_api_scorecard_lookup(client):
    res = client.get("/scorecard")
    assert res.status_code == 200
    data = res.json()
    assert len(data["scorecard"]) > 0
    assert len(data["iv_summary"]) > 0
