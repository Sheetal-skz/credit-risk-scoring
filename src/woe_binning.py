"""
Weight of Evidence (WOE) and Information Value (IV) Scorecard Engine.
Provides standard Basel II / credit bureau binning, WOE transformation,
IV calculation, and points-to-double-odds (PDO) scorecard generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression


class WOETransformer:
    """
    Transforms continuous and categorical features into Weight of Evidence (WOE) values
    and generates an industry-standard points-to-double-odds Credit Scorecard.
    """

    def __init__(
        self,
        n_bins: int = 5,
        base_points: float = 600.0,
        base_odds: float = 50.0,
        pdo: float = 20.0
    ):
        self.n_bins = n_bins
        self.base_points = base_points
        self.base_odds = base_odds
        self.pdo = pdo
        
        # Scaling parameters
        self.factor = pdo / np.log(2.0)
        self.offset = base_points - (self.factor * np.log(base_odds))

        self.binning_rules: Dict[str, Dict[str, Any]] = {}
        self.iv_summary: pd.DataFrame = pd.DataFrame()
        self.scorecard_table: pd.DataFrame = pd.DataFrame()
        self.lr_model: Optional[LogisticRegression] = None
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        """
        Compute optimal bins, WOE values, and IV for all features.
        y must be binary: 0 = Good (non-default), 1 = Bad (default).
        """
        self.feature_names = list(X.columns)
        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()

        if total_good == 0 or total_bad == 0:
            raise ValueError("Target variable y must contain both Good (0) and Bad (1) classes.")

        iv_records = []
        rules = {}

        for col in self.feature_names:
            series = X[col]
            is_numeric = pd.api.types.is_numeric_dtype(series)

            if is_numeric:
                # Quantile / uniform binning
                try:
                    binned, bin_edges = pd.qcut(series, q=self.n_bins, retbins=True, duplicates='drop')
                except Exception:
                    binned, bin_edges = pd.cut(series, bins=self.n_bins, retbins=True)
                
                bin_labels = [f"[{round(bin_edges[i], 2)}, {round(bin_edges[i+1], 2)}]" for i in range(len(bin_edges)-1)]
            else:
                binned = series.astype(str)
                bin_edges = None
                bin_labels = list(series.unique())

            temp_df = pd.DataFrame({
                'bin': binned,
                'target': y
            })

            grouped = temp_df.groupby('bin', observed=False)['target'].agg(
                total='count',
                bad='sum'
            ).reset_index()

            grouped['good'] = grouped['total'] - grouped['bad']

            # Laplace smoothing to avoid division by zero or log(0)
            grouped['dist_good'] = (grouped['good'] + 0.5) / (total_good + 1.0)
            grouped['dist_bad'] = (grouped['bad'] + 0.5) / (total_bad + 1.0)

            # WOE: ln(dist_good / dist_bad) -> Positive WOE indicates lower risk (good)
            grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
            grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']

            col_iv = grouped['iv'].sum()

            iv_records.append({
                'Feature': col,
                'Information_Value': col_iv,
                'Predictive_Power': self._classify_iv(col_iv)
            })

            rules[col] = {
                'is_numeric': is_numeric,
                'bin_edges': bin_edges if is_numeric else None,
                'woe_map': dict(zip(grouped['bin'].astype(str), grouped['woe'])),
                'group_stats': grouped
            }

        self.binning_rules = rules
        self.iv_summary = pd.DataFrame(iv_records).sort_values(by='Information_Value', ascending=False).reset_index(drop=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Map features in X to their corresponding WOE values.
        """
        X_woe = pd.DataFrame(index=X.index)

        for col in self.feature_names:
            if col not in X.columns:
                continue

            rule = self.binning_rules[col]
            series = X[col]

            if rule['is_numeric']:
                edges = rule['bin_edges']
                # Clip values to edges
                binned = pd.cut(series, bins=edges, include_lowest=True)
                binned_str = binned.astype(str)
            else:
                binned_str = series.astype(str)

            woe_map = rule['woe_map']
            # Fallback to mean WOE (or 0.0) for unseen categories
            default_woe = 0.0
            X_woe[col] = binned_str.map(woe_map).fillna(default_woe)

        return X_woe

    def fit_scorecard(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit Logistic Regression on WOE transformed features and generate
        the explicit points-per-bin Credit Scorecard.
        """
        self.fit(X, y)
        X_woe = self.transform(X)

        self.lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.lr_model.fit(X_woe, y)

        intercept = self.lr_model.intercept_[0]
        coefficients = dict(zip(self.feature_names, self.lr_model.coef_[0]))

        scorecard_rows = []
        n_features = len(self.feature_names)

        # Distribute intercept and offset uniformly across features
        base_term = (self.offset / n_features) - (self.factor * (intercept / n_features))

        for col in self.feature_names:
            coef = coefficients[col]
            rule = self.binning_rules[col]
            woe_map = rule['woe_map']

            for bin_label, woe_val in woe_map.items():
                # In standard credit scoring: higher WOE = lower default risk = positive score impact
                # Logistic regression models P(Bad), so negative sign on coef * woe aligns points
                points = round(base_term - (self.factor * coef * woe_val))

                scorecard_rows.append({
                    'Feature': col,
                    'Bin': bin_label,
                    'WOE': round(woe_val, 4),
                    'Coefficient': round(coef, 4),
                    'Points': int(points)
                })

        self.scorecard_table = pd.DataFrame(scorecard_rows)
        return self.scorecard_table

    def score_applicant(self, applicant_dict: Dict[str, Any]) -> Tuple[int, float, Dict[str, int]]:
        """
        Score a single loan applicant using the scorecard table.
        Returns:
            total_score (300 - 850 range),
            probability of default (PD),
            points breakdown per feature
        """
        points_breakdown = {}
        total_score = 0

        for col in self.feature_names:
            val = applicant_dict.get(col, None)
            rule = self.binning_rules.get(col)
            if not rule or val is None:
                continue

            if rule['is_numeric']:
                edges = rule['bin_edges']
                bin_series = pd.cut([val], bins=edges, include_lowest=True)
                bin_label = str(bin_series[0])
            else:
                bin_label = str(val)

            # Lookup points
            sub = self.scorecard_table[
                (self.scorecard_table['Feature'] == col) &
                (self.scorecard_table['Bin'] == bin_label)
            ]
            if len(sub) > 0:
                pts = int(sub.iloc[0]['Points'])
            else:
                # Fallback to feature median points
                sub_all = self.scorecard_table[self.scorecard_table['Feature'] == col]
                pts = int(sub_all['Points'].median()) if len(sub_all) > 0 else 50

            points_breakdown[col] = pts
            total_score += pts

        # Bound score to realistic bureau scale (300 to 850)
        bounded_score = int(np.clip(total_score, 300, 850))

        # Convert score back to Probability of Default (PD) using inverse scorecard formula:
        # Score = Offset + Factor * ln(Odds) => Odds = exp((Score - Offset) / Factor)
        # PD = 1 / (1 + Odds)
        odds = np.exp((bounded_score - self.offset) / self.factor)
        pd_val = float(1.0 / (1.0 + odds))
        pd_val = np.clip(pd_val, 0.001, 0.999)

        return bounded_score, pd_val, points_breakdown

    @staticmethod
    def _classify_iv(iv: float) -> str:
        """Classify feature predictive power based on Information Value."""
        if iv < 0.02:
            return "Unpredictive (<0.02)"
        elif iv < 0.10:
            return "Weak (0.02-0.10)"
        elif iv < 0.30:
            return "Medium (0.10-0.30)"
        elif iv < 0.50:
            return "Strong (0.30-0.50)"
        else:
            return "Very Strong / Suspicious (>0.50)"
