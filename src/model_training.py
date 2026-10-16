"""
Model training module for Credit Risk Scoring.
Trains:
1. Champion: Traditional Logistic Regression Scorecard on WOE features
2. Challenger: Modern XGBoost Classifier with calibrated default probabilities
Computes comprehensive credit risk metrics: ROC-AUC, Gini, KS statistic, Brier Score.
"""

import json
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss
)
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

from src.woe_binning import WOETransformer


def calculate_credit_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.20
) -> Dict[str, Any]:
    """
    Compute full suite of credit risk validation metrics.
    Includes Basel II/III metrics: AUC, Gini (2*AUC - 1), and KS (Kolmogorov-Smirnov).
    """
    auc = float(roc_auc_score(y_true, y_prob))
    gini = float(2 * auc - 1)
    brier = float(brier_score_loss(y_true, y_prob))
    lloss = float(log_loss(y_true, y_prob))

    # Calculate KS (Kolmogorov-Smirnov) statistic
    good_probs = y_prob[y_true == 0]
    bad_probs = y_prob[y_true == 1]
    ks_stat, p_val = ks_2samp(bad_probs, good_probs)
    ks_stat = float(ks_stat)

    # ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Classification at specified threshold
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # KS curve preparation
    num_thresholds = 50
    cutoffs = np.linspace(0.01, 0.99, num_thresholds)
    cum_goods = []
    cum_bads = []

    for c in cutoffs:
        cum_goods.append(float(np.mean(good_probs <= c)))
        cum_bads.append(float(np.mean(bad_probs <= c)))

    ks_curve_data = {
        'cutoffs': cutoffs.tolist(),
        'cum_goods': cum_goods,
        'cum_bads': cum_bads,
        'ks_stat': ks_stat
    }

    return {
        'auc': round(auc, 4),
        'gini': round(gini, 4),
        'ks_statistic': round(ks_stat, 4),
        'brier_score': round(brier, 4),
        'log_loss': round(lloss, 4),
        'threshold': threshold,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'confusion_matrix': cm,
        'roc_curve': {
            'fpr': [round(x, 4) for x in fpr[::max(1, len(fpr)//100)]],
            'tpr': [round(x, 4) for x in tpr[::max(1, len(tpr)//100)]]
        },
        'ks_curve': ks_curve_data
    }


def train_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts_dir: str = "models"
) -> Dict[str, Any]:
    """
    Train WOE Scorecard (Champion) and XGBoost (Challenger), save all artifacts.
    """
    out_path = Path(artifacts_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in train_df.columns if c != 'loan_status']
    target_col = 'loan_status'

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    # ==========================================
    # 1. Champion Model: WOE Logistic Scorecard
    # ==========================================
    print("Training Champion Model: WOE/IV Logistic Scorecard...")
    woe_model = WOETransformer(n_bins=5, base_points=600.0, base_odds=50.0, pdo=20.0)
    scorecard_df = woe_model.fit_scorecard(X_train, y_train)

    # Predictions on test set
    X_test_woe = woe_model.transform(X_test)
    woe_test_probs = woe_model.lr_model.predict_proba(X_test_woe)[:, 1]

    woe_metrics = calculate_credit_metrics(y_test.values, woe_test_probs)
    print(f"  WOE Scorecard -> AUC: {woe_metrics['auc']} | KS: {woe_metrics['ks_statistic']} | Gini: {woe_metrics['gini']}")

    # ==========================================
    # 2. Challenger Model: XGBoost Classifier
    # ==========================================
    print("Training Challenger Model: XGBoost...")
    # Prepare features for XGBoost (label encoding categoricals)
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    encoders = {}
    X_train_xgb = X_train.copy()
    X_test_xgb = X_test.copy()

    for col in cat_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_xgb[col] = enc.fit_transform(X_train[[col]])
        X_test_xgb[col] = enc.transform(X_test[[col]])
        encoders[col] = enc

    # Class imbalance handling
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_clf = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=1.0,  # kept at 1.0 to preserve calibrated probability calibration
        random_state=42,
        eval_metric="logloss"
    )
    xgb_clf.fit(X_train_xgb, y_train)

    xgb_test_probs = xgb_clf.predict_proba(X_test_xgb)[:, 1]
    xgb_metrics = calculate_credit_metrics(y_test.values, xgb_test_probs)
    print(f"  XGBoost       -> AUC: {xgb_metrics['auc']} | KS: {xgb_metrics['ks_statistic']} | Gini: {xgb_metrics['gini']}")

    # Feature importances
    xgb_importance = dict(zip(feature_cols, [round(float(x), 4) for x in xgb_clf.feature_importances_]))
    xgb_importance = dict(sorted(xgb_importance.items(), key=lambda item: item[1], reverse=True))

    # ==========================================
    # 3. Export Artifacts
    # ==========================================
    joblib.dump(woe_model, out_path / "woe_scorecard.pkl")
    joblib.dump(xgb_clf, out_path / "xgboost_model.pkl")
    joblib.dump(encoders, out_path / "cat_encoders.pkl")
    scorecard_df.to_csv(out_path / "scorecard_table.csv", index=False)
    woe_model.iv_summary.to_csv(out_path / "iv_summary.csv", index=False)

    summary_metrics = {
        'champion_woe': woe_metrics,
        'challenger_xgboost': xgb_metrics,
        'xgb_feature_importance': xgb_importance,
        'feature_names': feature_cols,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'default_rate_test': round(float(y_test.mean()), 4)
    }

    with open(out_path / "model_metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"Artifacts successfully saved to {artifacts_dir}/")
    return summary_metrics


if __name__ == "__main__":
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    train_models(train_df, test_df)
