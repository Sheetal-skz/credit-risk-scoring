"""
Explainability and Regulatory Compliance Engine.
Generates SHAP (TreeExplainer) explanations for XGBoost predictions and translates
feature attributions into official ECOA / Regulation B / FCRA Adverse Action Reason Codes.
"""

import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import shap


# Regulatory Reason Code Mappings (ECOA / FCRA standard library)
REGULATORY_REASON_CODES = {
    'fico_score': {
        'code': 'R01',
        'title': 'Low Credit Bureau Risk Score',
        'description': 'The applicant credit score is lower than established risk appetite guidelines.'
    },
    'dti': {
        'code': 'R02',
        'title': 'Excessive Debt-to-Income (DTI) Ratio',
        'description': 'Monthly debt obligations are too high relative to verified gross monthly income.'
    },
    'revol_util': {
        'code': 'R03',
        'title': 'High Revolving Credit Utilization',
        'description': 'Proportion of balances relative to total credit limits on revolving accounts is too high.'
    },
    'inq_last_6mths': {
        'code': 'R04',
        'title': 'Recent Credit Seeking Behavior',
        'description': 'Number of recent inquiries for new credit lines in the past 6 months is excessive.'
    },
    'delinq_2yrs': {
        'code': 'R05',
        'title': 'Past Delinquencies on Credit Accounts',
        'description': 'History of past-due payments or delinquencies within the preceding 24 months.'
    },
    'pub_rec_bankruptcies': {
        'code': 'R06',
        'title': 'Derogatory Public Record or Bankruptcy',
        'description': 'Public records indicate bankruptcy filing or derogatory judgment history.'
    },
    'annual_inc': {
        'code': 'R07',
        'title': 'Insufficient Annual Income',
        'description': 'Total verified gross annual income does not sufficiently cover requested credit facility.'
    },
    'int_rate': {
        'code': 'R08',
        'title': 'High Interest Rate Risk Tier',
        'description': 'Risk-based pricing matrix assigns applicant to an unserviceable rate tier.'
    },
    'loan_amnt': {
        'code': 'R09',
        'title': 'Excessive Loan Amount Requested',
        'description': 'Requested loan principal exceeds exposure limits given borrower asset profile.'
    },
    'emp_length': {
        'code': 'R10',
        'title': 'Short Employment Tenure',
        'description': 'Current verified length of continuous employment history does not meet threshold.'
    },
    'mort_acc': {
        'code': 'R11',
        'title': 'Lack of Established Real Estate Collateral History',
        'description': 'Applicant profile lacks mature mortgage line seasoning.'
    },
    'open_acc': {
        'code': 'R12',
        'title': 'Insufficient Active Credit Lines',
        'description': 'Limited number of actively reporting open credit accounts.'
    },
    'revol_bal': {
        'code': 'R13',
        'title': 'Elevated Revolving Debt Outstanding',
        'description': 'Total outstanding revolving balance is elevated relative to peers.'
    },
    'term': {
        'code': 'R14',
        'title': 'Extended Loan Maturity Exposure',
        'description': 'Requested 60-month duration presents heightened macroeconomic default risk.'
    }
}


class CreditExplainabilityEngine:
    """
    Computes SHAP explanations and generates regulatory Adverse Action notices.
    """

    def __init__(self, artifacts_dir: str = "models"):
        self.artifacts_dir = Path(artifacts_dir)
        self.xgb_model = joblib.load(self.artifacts_dir / "xgboost_model.pkl")
        self.encoders = joblib.load(self.artifacts_dir / "cat_encoders.pkl")
        self.explainer = shap.TreeExplainer(self.xgb_model)
        if hasattr(self.explainer, 'expected_value'):
            ev = self.explainer.expected_value
            if isinstance(ev, (np.ndarray, list)):
                self.expected_value = float(np.ravel(ev)[0])
            else:
                self.expected_value = float(ev)
        else:
            self.expected_value = 0.0

    def prepare_input(self, applicant_df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables identically to training pipeline."""
        df_encoded = applicant_df.copy()
        for col, enc in self.encoders.items():
            if col in df_encoded.columns:
                df_encoded[col] = enc.transform(df_encoded[[col]])
        return df_encoded

    def explain_applicant(
        self,
        applicant_dict: Dict[str, Any],
        top_n_reasons: int = 4
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for a single applicant and generate regulatory Adverse Action details.
        Positive SHAP values increase default probability (risk factor).
        Negative SHAP values decrease default probability (protective factor).
        """
        df = pd.DataFrame([applicant_dict])
        df_encoded = self.prepare_input(df)

        raw_shap = self.explainer.shap_values(df_encoded)
        if isinstance(raw_shap, list):
            shap_values = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
        else:
            shap_values = raw_shap
        
        shap_values = np.ravel(shap_values)
        feature_names = df_encoded.columns.tolist()

        contributions = []
        for feat, val, shap_val in zip(feature_names, df.iloc[0].values, shap_values):
            clean_val = val.item() if hasattr(val, 'item') else val
            if isinstance(clean_val, (np.integer, int)):
                clean_val = int(clean_val)
            elif isinstance(clean_val, (np.floating, float)):
                clean_val = float(clean_val)

            contributions.append({
                'feature': feat,
                'feature_value': clean_val,
                'shap_value': round(float(shap_val), 4),
                # Positive SHAP = pushes towards default (higher risk)
                'risk_impact': 'Increases Risk' if shap_val > 0 else 'Decreases Risk'
            })

        # Sort features by risk contribution (highest SHAP values first)
        risk_sorted = sorted(contributions, key=lambda x: x['shap_value'], reverse=True)
        protective_sorted = sorted(contributions, key=lambda x: x['shap_value'])

        # Top adverse reasons for regulatory notice (factors that pushed score worst)
        adverse_reasons = []
        for item in risk_sorted:
            feat = item['feature']
            if item['shap_value'] > 0 and feat in REGULATORY_REASON_CODES:
                code_meta = REGULATORY_REASON_CODES[feat]
                adverse_reasons.append({
                    'code': code_meta['code'],
                    'title': code_meta['title'],
                    'description': code_meta['description'],
                    'feature': feat,
                    'applicant_value': item['feature_value'],
                    'shap_impact': item['shap_value']
                })
            if len(adverse_reasons) >= top_n_reasons:
                break

        # If fewer than top_n reasons have positive SHAP, fill with highest relative contributors
        if len(adverse_reasons) < top_n_reasons:
            for item in risk_sorted:
                feat = item['feature']
                if feat in REGULATORY_REASON_CODES and not any(r['feature'] == feat for r in adverse_reasons):
                    code_meta = REGULATORY_REASON_CODES[feat]
                    adverse_reasons.append({
                        'code': code_meta['code'],
                        'title': code_meta['title'],
                        'description': code_meta['description'],
                        'feature': feat,
                        'applicant_value': item['feature_value'],
                        'shap_impact': item['shap_value']
                    })
                if len(adverse_reasons) >= top_n_reasons:
                    break

        return {
            'base_value': round(self.expected_value, 4),
            'top_risk_drivers': risk_sorted[:5],
            'top_protective_drivers': protective_sorted[:5],
            'all_contributions': contributions,
            'adverse_action_reasons': adverse_reasons
        }

    def generate_adverse_action_notice(
        self,
        applicant_name: str,
        applicant_id: str,
        applicant_dict: Dict[str, Any],
        credit_score: int,
        decision: str = "DECLINED"
    ) -> str:
        """
        Generates a legally structured ECOA / FCRA Adverse Action Notice.
        """
        explanation = self.explain_applicant(applicant_dict, top_n_reasons=4)
        reasons = explanation['adverse_action_reasons']

        reasons_text = ""
        for i, r in enumerate(reasons, 1):
            reasons_text += f"  {i}. [{r['code']}] {r['title']}\n"
            reasons_text += f"     Detail: {r['description']} (Reported: {r['applicant_value']})\n\n"

        notice = f"""================================================================================
                    STATEMENT OF CREDIT DENIAL, TERMINATION, OR CHANGE
                          (REGULATION B - 12 C.F.R. PART 1002)
================================================================================

Date: {pd.Timestamp.now().strftime('%B %d, %Y')}
Applicant ID: {applicant_id}
Applicant Name: {applicant_name}

DECISION: {decision}
Credit Risk Score: {credit_score} / 850
Score Evaluation Methodology: Hybrid Scorecard & Gradient Boosted Risk Engine

--------------------------------------------------------------------------------
PART I - PRINCIPAL REASONS FOR CREDIT DENIAL / ADVERSE ACTION
--------------------------------------------------------------------------------
Pursuant to Section 701(a)(2) of the Equal Credit Opportunity Act (ECOA) and
Section 615(a) of the Fair Credit Reporting Act (FCRA), our credit decision was
based in whole or in part on information provided in your application and credit
bureau report. The primary factors that adversely affected your risk score were:

{reasons_text}
--------------------------------------------------------------------------------
PART II - DISCLOSURE OF CREDIT SCORING & CONSUMER REPORTING
--------------------------------------------------------------------------------
Credit Reporting Agency:
  Apex Credit Bureau Services
  Consumer Inquiries Division, P.O. Box 90210
  Toll-Free Phone: 1-800-555-0199
  Website: https://www.apexcreditbureau.example

Under the Fair Credit Reporting Act, you have the right to obtain a free copy
of your consumer credit report within 60 days of receiving this notice. You also
have the right to dispute the accuracy or completeness of any information in
the report.

--------------------------------------------------------------------------------
PART III - EQUAL CREDIT OPPORTUNITY ACT (ECOA) NOTICE
--------------------------------------------------------------------------------
The Federal Equal Credit Opportunity Act prohibits creditors from discriminating
against credit applicants on the basis of race, color, religion, national origin,
sex, marital status, age (provided the applicant has the capacity to contract);
because all or part of the applicant's income derives from any public assistance
program; or because the applicant has in good faith exercised any right under the
Consumer Credit Protection Act.

Federal agency administering compliance:
  Consumer Financial Protection Bureau (CFPB)
  1700 G Street NW, Washington, DC 20552
================================================================================
"""
        return notice
