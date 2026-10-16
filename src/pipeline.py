"""
End-to-End Orchestration Pipeline for Credit Risk Scoring.
Generates data, fits WOE scorecard, trains XGBoost challenger,
computes validation metrics, and exports all production artifacts.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generator import save_dataset_splits
from src.model_training import train_models
from src.explainability import CreditExplainabilityEngine


def run_full_pipeline(n_samples: int = 15000):
    print("=" * 70)
    print("       STARTING CREDIT RISK SCORING PRODUCTION PIPELINE        ")
    print("=" * 70)

    # 1. Generate & Split Data
    print("\n[Step 1/3] Generating Lending Club portfolio dataset...")
    train_df, test_df = save_dataset_splits(
        output_dir="data",
        n_samples=n_samples,
        test_size=0.2,
        random_state=42
    )

    # 2. Train Models (WOE Champion vs XGBoost Challenger)
    print("\n[Step 2/3] Fitting WOE Scorecard and XGBoost Challenger...")
    metrics = train_models(train_df, test_df, artifacts_dir="models")

    # 3. Test Explainability & Adverse Action Engine
    print("\n[Step 3/3] Initializing SHAP Explainer & Regulatory Engine...")
    engine = CreditExplainabilityEngine(artifacts_dir="models")
    
    # Test on a sample applicant
    sample_applicant = test_df.iloc[0].drop('loan_status').to_dict()
    explanation = engine.explain_applicant(sample_applicant)
    print(f"  Sample applicant base SHAP value: {explanation['base_value']}")
    print(f"  Top risk drivers: {[d['feature'] for d in explanation['top_risk_drivers'][:3]]}")

    print("\n" + "=" * 70)
    print("Pipeline completed successfully! All artifacts saved to models/ and data/")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
