"""
Production FastAPI Application for Credit Risk Scoring & Explainable AI.
Provides RESTful endpoints for individual scoring, batch underwriting,
regulatory Adverse Action generation, and model benchmarking.
"""

import sys
import io
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.schemas import (
    ScoreRequest,
    ScoreResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    AdverseActionNoticeRequest,
    ApplicantProfile
)
from src.woe_binning import WOETransformer
from src.explainability import CreditExplainabilityEngine


from contextlib import asynccontextmanager

# Global model state
MODELS: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pre-trained models and metadata on service startup."""
    artifacts_dir = ROOT_DIR / "models"
    data_dir = ROOT_DIR / "data"

    try:
        MODELS['woe_model'] = joblib.load(artifacts_dir / "woe_scorecard.pkl")
        MODELS['xgb_model'] = joblib.load(artifacts_dir / "xgboost_model.pkl")
        MODELS['cat_encoders'] = joblib.load(artifacts_dir / "cat_encoders.pkl")
        MODELS['explainer_engine'] = CreditExplainabilityEngine(artifacts_dir=str(artifacts_dir))

        with open(artifacts_dir / "model_metrics.json", "r") as f:
            MODELS['metrics'] = json.load(f)

        MODELS['scorecard_df'] = pd.read_csv(artifacts_dir / "scorecard_table.csv")
        MODELS['iv_summary_df'] = pd.read_csv(artifacts_dir / "iv_summary.csv")

        if (data_dir / "loan_data.csv").exists():
            MODELS['portfolio_df'] = pd.read_csv(data_dir / "loan_data.csv")

        print("All credit scoring models and artifacts loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to load some artifacts: {e}")

    yield

    MODELS.clear()


app = FastAPI(
    title="Fintech Credit Risk Scoring API",
    description="Regulatory-grade Credit Risk Underwriting, WOE Scorecard, XGBoost, and SHAP Explainability Service.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def assign_risk_rating(pd_value: float) -> str:
    """Assign credit risk rating grade based on Probability of Default (PD)."""
    if pd_value < 0.05:
        return "AAA (Prime)"
    elif pd_value < 0.10:
        return "AA (High Quality)"
    elif pd_value < 0.18:
        return "A (Upper Medium)"
    elif pd_value < 0.28:
        return "BBB (Investment Grade)"
    elif pd_value < 0.40:
        return "BB (Vulnerable / Speculative)"
    elif pd_value < 0.55:
        return "B (Highly Speculative)"
    elif pd_value < 0.75:
        return "CCC (Substantial Default Risk)"
    else:
        return "D (Imminent Default)"


def make_decision(pd_value: float, threshold: float = 0.20) -> str:
    """Determine automated credit decision based on risk threshold."""
    if pd_value <= threshold:
        return "APPROVED"
    elif pd_value <= threshold + 0.12:
        return "MANUAL_REVIEW"
    else:
        return "DECLINED"


def pd_to_credit_score(pd_val: float) -> int:
    """Map Probability of Default to FICO-scale Credit Score (300 to 850)."""
    # Inverse log-odds mapping scaled to 300 - 850
    # Odds = (1 - pd) / pd
    eps = 1e-6
    clamped_pd = np.clip(pd_val, eps, 1.0 - eps)
    odds = (1.0 - clamped_pd) / clamped_pd
    # Standard base 600 at 50:1 odds, PDO 20
    factor = 20.0 / np.log(2.0)
    offset = 600.0 - (factor * np.log(50.0))
    score = offset + factor * np.log(odds)
    return int(np.clip(score, 300, 850))


@app.get("/", tags=["System"])
def root():
    return {
        "service": "Fintech Credit Risk Scoring API",
        "status": "online",
        "documentation": "/docs",
        "version": "1.0.0",
        "models_available": ["woe_scorecard (Champion)", "xgboost (Challenger)"]
    }


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "artifacts_loaded": list(MODELS.keys()),
        "model_engine": "XGBoost + WOE/IV Logistic Regression"
    }


@app.post("/score", response_model=ScoreResponse, tags=["Scoring Engine"])
def score_single_applicant(request: ScoreRequest):
    """
    Score a single loan application.
    Supports either Champion ('woe_scorecard') or Challenger ('xgboost') model.
    Returns credit score, PD %, decision, scorecard points breakdown, and SHAP explanations.
    """
    app_dict = request.applicant.model_dump()
    model_type = request.model_type.lower()
    threshold = request.decision_threshold

    if model_type not in ["xgboost", "woe_scorecard"]:
        raise HTTPException(status_code=400, detail="Invalid model_type. Choose 'xgboost' or 'woe_scorecard'")

    if 'woe_model' not in MODELS or 'xgb_model' not in MODELS:
        raise HTTPException(status_code=500, detail="Models not loaded. Please run pipeline first.")

    # 1. Scoring with WOE Scorecard
    if model_type == "woe_scorecard":
        woe_model: WOETransformer = MODELS['woe_model']
        score, pd_val, points_breakdown = woe_model.score_applicant(app_dict)
        rating = assign_risk_rating(pd_val)
        decision = make_decision(pd_val, threshold=threshold)

        return ScoreResponse(
            model_used="woe_scorecard (Champion)",
            credit_score=score,
            probability_of_default=round(pd_val, 4),
            risk_rating=rating,
            decision=decision,
            scorecard_points_breakdown=points_breakdown,
            shap_base_value=None,
            top_risk_drivers=None,
            top_protective_drivers=None,
            adverse_action_reasons=None
        )

    # 2. Scoring with XGBoost Challenger + SHAP
    else:
        xgb_model = MODELS['xgb_model']
        encoders = MODELS['cat_encoders']
        explainer_engine: CreditExplainabilityEngine = MODELS['explainer_engine']

        df = pd.DataFrame([app_dict])
        df_encoded = df.copy()
        for col, enc in encoders.items():
            if col in df_encoded.columns:
                df_encoded[col] = enc.transform(df_encoded[[col]])

        pd_val = float(xgb_model.predict_proba(df_encoded)[0, 1])
        score = pd_to_credit_score(pd_val)
        rating = assign_risk_rating(pd_val)
        decision = make_decision(pd_val, threshold=threshold)

        # Compute SHAP attributions
        explanation = explainer_engine.explain_applicant(app_dict, top_n_reasons=4)

        return ScoreResponse(
            model_used="xgboost (Challenger)",
            credit_score=score,
            probability_of_default=round(pd_val, 4),
            risk_rating=rating,
            decision=decision,
            scorecard_points_breakdown=None,
            shap_base_value=explanation['base_value'],
            top_risk_drivers=explanation['top_risk_drivers'],
            top_protective_drivers=explanation['top_protective_drivers'],
            adverse_action_reasons=explanation['adverse_action_reasons']
        )


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["Scoring Engine"])
def score_batch_applicants(request: BatchScoreRequest):
    """
    Score a batch of loan applicants and return portfolio level aggregates.
    """
    results: List[ScoreResponse] = []
    scores = []
    pds = []
    approved = 0
    declined = 0
    review = 0

    for applicant in request.applicants:
        single_req = ScoreRequest(
            applicant=applicant,
            model_type=request.model_type,
            decision_threshold=request.decision_threshold
        )
        res = score_single_applicant(single_req)
        results.append(res)
        scores.append(res.credit_score)
        pds.append(res.probability_of_default)

        if res.decision == "APPROVED":
            approved += 1
        elif res.decision == "DECLINED":
            declined += 1
        else:
            review += 1

    total = len(request.applicants)
    return BatchScoreResponse(
        total_applicants=total,
        approved_count=approved,
        declined_count=declined,
        review_count=review,
        portfolio_avg_score=round(float(np.mean(scores)), 1) if scores else 0.0,
        portfolio_avg_pd=round(float(np.mean(pds)), 4) if pds else 0.0,
        results=results
    )


@app.post("/score/batch-file", tags=["Scoring Engine"])
async def score_batch_csv(
    file: UploadFile = File(...),
    model_type: str = Query("xgboost", enum=["xgboost", "woe_scorecard"]),
    threshold: float = Query(0.20, ge=0.01, le=0.99)
):
    """
    Upload a CSV file of applicants, score all records, and return enriched dataframe.
    """
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    # Remove loan_status if present in test data
    input_df = df.drop(columns=['loan_status'], errors='ignore')

    scored_records = []
    for idx, row in input_df.iterrows():
        app_dict = row.to_dict()
        try:
            profile = ApplicantProfile(**app_dict)
            single_req = ScoreRequest(
                applicant=profile,
                model_type=model_type,
                decision_threshold=threshold
            )
            res = score_single_applicant(single_req)
            scored_records.append({
                'Credit_Score': res.credit_score,
                'PD': res.probability_of_default,
                'Risk_Rating': res.risk_rating,
                'Decision': res.decision
            })
        except Exception as e:
            scored_records.append({
                'Credit_Score': None,
                'PD': None,
                'Risk_Rating': 'ERROR',
                'Decision': f'Validation Error: {str(e)}'
            })

    result_df = pd.concat([input_df, pd.DataFrame(scored_records)], axis=1)
    return {
        "total_records": len(result_df),
        "approval_rate": round(float((result_df['Decision'] == 'APPROVED').mean()), 4),
        "decline_rate": round(float((result_df['Decision'] == 'DECLINED').mean()), 4),
        "review_rate": round(float((result_df['Decision'] == 'MANUAL_REVIEW').mean()), 4),
        "average_credit_score": round(float(result_df['Credit_Score'].dropna().mean()), 1),
        "preview": result_df.head(20).to_dict(orient="records")
    }


@app.post("/adverse-action/notice", response_class=PlainTextResponse, tags=["Regulatory Compliance"])
def generate_adverse_action_document(request: AdverseActionNoticeRequest):
    """
    Generate an official ECOA / FCRA compliant Adverse Action Notice text document.
    """
    if 'explainer_engine' not in MODELS:
        raise HTTPException(status_code=500, detail="Explainability engine not initialized.")

    engine: CreditExplainabilityEngine = MODELS['explainer_engine']
    app_dict = request.applicant.model_dump()

    # Score with XGBoost to get decision & score
    xgb_model = MODELS['xgb_model']
    encoders = MODELS['cat_encoders']
    df = pd.DataFrame([app_dict])
    df_encoded = df.copy()
    for col, enc in encoders.items():
        if col in df_encoded.columns:
            df_encoded[col] = enc.transform(df_encoded[[col]])

    pd_val = float(xgb_model.predict_proba(df_encoded)[0, 1])
    score = pd_to_credit_score(pd_val)
    decision = make_decision(pd_val, threshold=0.20)

    notice_text = engine.generate_adverse_action_notice(
        applicant_name=request.applicant_name,
        applicant_id=request.applicant_id,
        applicant_dict=app_dict,
        credit_score=score,
        decision=decision
    )
    return notice_text


@app.get("/metrics", tags=["Model Analytics"])
def get_model_metrics():
    """
    Retrieve Champion vs Challenger model validation metrics:
    ROC AUC, Gini coefficient, KS statistic, Brier Score, and curves.
    """
    if 'metrics' not in MODELS:
        raise HTTPException(status_code=500, detail="Metrics not loaded.")
    return MODELS['metrics']


@app.get("/scorecard", tags=["Scorecard"])
def get_scorecard_table():
    """
    Retrieve full WOE Scorecard lookup table and Information Value (IV) summary.
    """
    if 'scorecard_df' not in MODELS or 'iv_summary_df' not in MODELS:
        raise HTTPException(status_code=500, detail="Scorecard data not loaded.")

    return {
        "scorecard": MODELS['scorecard_df'].to_dict(orient="records"),
        "iv_summary": MODELS['iv_summary_df'].to_dict(orient="records")
    }


@app.get("/eda/summary", tags=["EDA & Portfolio Insights"])
def get_eda_summary():
    """
    Retrieve portfolio-level exploratory data analysis statistics.
    """
    if 'portfolio_df' not in MODELS:
        raise HTTPException(status_code=500, detail="Portfolio data not loaded.")

    df = MODELS['portfolio_df']
    default_rate = float(df['loan_status'].mean())
    avg_loan_amnt = float(df['loan_amnt'].mean())
    avg_fico = float(df['fico_score'].mean())
    avg_dti = float(df['dti'].mean())

    grade_stats = df.groupby('grade')['loan_status'].agg(['count', 'mean']).reset_index()
    grade_stats.columns = ['grade', 'total_loans', 'default_rate']

    purpose_stats = df.groupby('purpose')['loan_status'].agg(['count', 'mean']).reset_index()
    purpose_stats.columns = ['purpose', 'total_loans', 'default_rate']

    home_stats = df.groupby('home_ownership')['loan_status'].agg(['count', 'mean']).reset_index()
    home_stats.columns = ['home_ownership', 'total_loans', 'default_rate']

    return {
        "portfolio_summary": {
            "total_loans": len(df),
            "portfolio_default_rate": round(default_rate, 4),
            "average_loan_amount": round(avg_loan_amnt, 2),
            "average_fico_score": round(avg_fico, 1),
            "average_dti": round(avg_dti, 2)
        },
        "default_by_grade": grade_stats.to_dict(orient="records"),
        "default_by_purpose": purpose_stats.to_dict(orient="records"),
        "default_by_home_ownership": home_stats.to_dict(orient="records")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
