"""
Pydantic Schemas for FastAPI Credit Risk Scoring Service.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ApplicantProfile(BaseModel):
    loan_amnt: float = Field(..., ge=1000, le=100000, description="Requested loan amount in USD", json_schema_extra={"example": 15000.0})
    term: str = Field(..., description="Loan tenure", json_schema_extra={"example": "36 months"})
    int_rate: float = Field(..., ge=2.0, le=40.0, description="Interest rate percentage", json_schema_extra={"example": 12.5})
    installment: float = Field(..., ge=10.0, le=5000.0, description="Expected monthly installment in USD", json_schema_extra={"example": 501.8})
    grade: str = Field(..., description="Lending grade (A to G)", json_schema_extra={"example": "B"})
    emp_length: str = Field(..., description="Employment duration", json_schema_extra={"example": "5 years"})
    home_ownership: str = Field(..., description="Housing tenure (RENT, MORTGAGE, OWN, OTHER)", json_schema_extra={"example": "MORTGAGE"})
    annual_inc: float = Field(..., ge=10000, le=2000000, description="Gross annual income in USD", json_schema_extra={"example": 75000.0})
    verification_status: str = Field(..., description="Income verification state", json_schema_extra={"example": "Verified"})
    purpose: str = Field(..., description="Loan purpose category", json_schema_extra={"example": "debt_consolidation"})
    dti: float = Field(..., ge=0.0, le=100.0, description="Debt-to-income ratio percentage", json_schema_extra={"example": 16.5})
    delinq_2yrs: int = Field(..., ge=0, le=30, description="30+ days delinquencies in past 2 years", json_schema_extra={"example": 0})
    fico_score: float = Field(..., ge=300, le=850, description="Credit Bureau FICO score", json_schema_extra={"example": 710.0})
    inq_last_6mths: int = Field(..., ge=0, le=30, description="Inquiries in past 6 months", json_schema_extra={"example": 1})
    open_acc: int = Field(..., ge=0, le=100, description="Open credit lines", json_schema_extra={"example": 11})
    pub_rec: int = Field(..., ge=0, le=50, description="Derogatory public records count", json_schema_extra={"example": 0})
    revol_bal: float = Field(..., ge=0, description="Total revolving balance in USD", json_schema_extra={"example": 14200.0})
    revol_util: float = Field(..., ge=0.0, le=200.0, description="Revolving utilization rate percentage", json_schema_extra={"example": 48.2})
    total_acc: int = Field(..., ge=0, le=150, description="Total credit lines on record", json_schema_extra={"example": 24})
    mort_acc: int = Field(..., ge=0, le=30, description="Mortgage accounts count", json_schema_extra={"example": 2})
    pub_rec_bankruptcies: int = Field(..., ge=0, le=20, description="Bankruptcies on record", json_schema_extra={"example": 0})


class ScoreRequest(BaseModel):
    applicant: ApplicantProfile
    model_type: str = Field("xgboost", description="Model engine to use: 'xgboost' or 'woe_scorecard'")
    decision_threshold: float = Field(0.20, ge=0.01, le=0.99, description="Probability cutoff for loan approval")


class AdverseReasonItem(BaseModel):
    code: str
    title: str
    description: str
    feature: str
    applicant_value: Any
    shap_impact: float


class ScoreResponse(BaseModel):
    model_used: str
    credit_score: int
    probability_of_default: float
    risk_rating: str
    decision: str
    scorecard_points_breakdown: Optional[Dict[str, int]] = None
    shap_base_value: Optional[float] = None
    top_risk_drivers: Optional[List[Dict[str, Any]]] = None
    top_protective_drivers: Optional[List[Dict[str, Any]]] = None
    adverse_action_reasons: Optional[List[AdverseReasonItem]] = None


class BatchScoreRequest(BaseModel):
    applicants: List[ApplicantProfile]
    model_type: str = "xgboost"
    decision_threshold: float = 0.20


class BatchScoreResponse(BaseModel):
    total_applicants: int
    approved_count: int
    declined_count: int
    review_count: int
    portfolio_avg_score: float
    portfolio_avg_pd: float
    results: List[ScoreResponse]


class AdverseActionNoticeRequest(BaseModel):
    applicant_name: str = "John Doe"
    applicant_id: str = "APP-78901"
    applicant: ApplicantProfile
