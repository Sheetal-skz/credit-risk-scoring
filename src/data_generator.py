"""
Data generation module for Credit Risk Scoring.
Generates realistic Lending Club style loan portfolio datasets with authentic
demographic distributions, credit bureau variables, and financial correlation structures.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path


def generate_lending_club_data(
    n_samples: int = 15000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic loan portfolio data mirroring Lending Club characteristics.
    Features include borrower demographics, loan characteristics, and credit bureau records.
    """
    rng = np.random.default_rng(random_state)

    # 1. Borrower Demographics & Income
    # Annual income log-normally distributed, median ~$65,000
    log_inc = rng.normal(loc=11.0, scale=0.55, size=n_samples)
    annual_inc = np.round(np.clip(np.exp(log_inc), 18000, 320000), -2)

    home_ownership = rng.choice(
        ['MORTGAGE', 'RENT', 'OWN', 'OTHER'],
        size=n_samples,
        p=[0.49, 0.40, 0.10, 0.01]
    )

    emp_length_opts = [
        '< 1 year', '1 year', '2 years', '3 years', '4 years',
        '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
    ]
    emp_length_weights = [0.08, 0.06, 0.09, 0.08, 0.06, 0.07, 0.05, 0.05, 0.04, 0.03, 0.39]
    emp_length = rng.choice(emp_length_opts, size=n_samples, p=emp_length_weights)

    verification_status = rng.choice(
        ['Verified', 'Source Verified', 'Not Verified'],
        size=n_samples,
        p=[0.38, 0.35, 0.27]
    )

    # 2. Credit Bureau Profile (FICO, DTI, Inquiries, History)
    # FICO score: bell curve centered at 695, min 600, max 850
    fico_score = np.round(np.clip(rng.normal(loc=695, scale=45, size=n_samples), 600, 850))

    # Debt-to-income: correlated inversely with FICO and income
    base_dti = rng.normal(loc=18.5, scale=7.5, size=n_samples)
    dti = np.clip(base_dti + (700 - fico_score) * 0.08, 1.0, 42.0)
    dti = np.round(dti, 2)

    # Inquiries in last 6 months (0 to 5)
    inq_prob = np.clip(0.35 + (700 - fico_score) * 0.003, 0.05, 0.8)
    inq_last_6mths = rng.poisson(lam=inq_prob * 1.5, size=n_samples)
    inq_last_6mths = np.clip(inq_last_6mths, 0, 7)

    # Revolving line utilization (0% to 105%)
    base_util = rng.normal(loc=52.0, scale=22.0, size=n_samples)
    revol_util = np.clip(base_util + (700 - fico_score) * 0.25, 1.0, 99.5)
    revol_util = np.round(revol_util, 1)

    # Revolving balance ($500 to $80,000)
    revol_bal = np.round(np.clip(annual_inc * (dti / 100.0) * rng.uniform(0.3, 0.9, size=n_samples), 400, 95000), -1)

    # Delinquencies in last 2 years (mostly 0)
    delinq_prob = np.where(fico_score < 660, 0.35, np.where(fico_score < 720, 0.12, 0.03))
    delinq_2yrs = rng.binomial(n=4, p=delinq_prob, size=n_samples)

    # Open credit lines (2 to 32)
    open_acc = np.clip(rng.poisson(lam=11.5, size=n_samples), 2, 35)
    total_acc = open_acc + rng.poisson(lam=13.0, size=n_samples)

    # Mortgage accounts
    mort_prob = np.where(home_ownership == 'MORTGAGE', 0.65, 0.08)
    mort_acc = rng.binomial(n=5, p=mort_prob, size=n_samples)

    # Bankruptcies & Derogatory public records
    pub_rec_bankruptcies = np.where(
        fico_score < 660,
        rng.choice([0, 1, 2], size=n_samples, p=[0.78, 0.19, 0.03]),
        rng.choice([0, 1], size=n_samples, p=[0.97, 0.03])
    )
    pub_rec = pub_rec_bankruptcies + rng.binomial(n=1, p=0.04, size=n_samples)

    # 3. Loan Terms & Characteristics
    # Loan amount correlated with income
    base_loan = annual_inc * rng.uniform(0.12, 0.35, size=n_samples)
    loan_amnt = np.round(np.clip(base_loan, 1500, 40000), -2)

    term = rng.choice(['36 months', '60 months'], size=n_samples, p=[0.72, 0.28])

    # Interest rate depends on FICO, DTI, Term
    int_rate_base = 13.5 + (700 - fico_score) * 0.11 + (dti - 18) * 0.14
    int_rate_base += np.where(term == '60 months', 2.8, 0.0)
    int_rate = np.round(np.clip(int_rate_base + rng.normal(0, 1.2, size=n_samples), 5.32, 28.99), 2)

    # Installment formula: P * (r*(1+r)^n) / ((1+r)^n - 1)
    n_months = np.where(term == '36 months', 36, 60)
    monthly_r = (int_rate / 100.0) / 12.0
    installment = loan_amnt * (monthly_r * (1 + monthly_r)**n_months) / ((1 + monthly_r)**n_months - 1)
    installment = np.round(installment, 2)

    # Purpose of loan
    purpose_opts = [
        'debt_consolidation', 'credit_card', 'home_improvement',
        'major_purchase', 'small_business', 'medical', 'car', 'other'
    ]
    purpose = rng.choice(
        purpose_opts,
        size=n_samples,
        p=[0.55, 0.22, 0.07, 0.04, 0.03, 0.03, 0.02, 0.04]
    )

    # Lending Club Grade & Sub-grade based on interest rate
    def assign_grade(rate):
        if rate < 8.0: return 'A'
        elif rate < 11.5: return 'B'
        elif rate < 15.5: return 'C'
        elif rate < 19.5: return 'D'
        elif rate < 23.5: return 'E'
        elif rate < 26.5: return 'F'
        else: return 'G'

    grade = np.array([assign_grade(r) for r in int_rate])

    # 4. Target Variable: loan_status (0 = Good / Fully Paid, 1 = Bad / Default)
    # Log-odds of default based on financial credit fundamentals
    # Standard credit risk logit formula
    z = (
        -2.2
        - (fico_score - 700) * 0.022
        + (dti - 18) * 0.042
        + (revol_util - 50) * 0.016
        + (int_rate - 13) * 0.08
        + inq_last_6mths * 0.19
        + delinq_2yrs * 0.35
        + pub_rec_bankruptcies * 0.45
        + (loan_amnt / annual_inc - 0.20) * 1.5
        + np.where(term == '60 months', 0.25, 0.0)
        + np.where(home_ownership == 'RENT', 0.15, 0.0)
        + np.where(purpose == 'small_business', 0.38, 0.0)
    )

    prob_default = 1.0 / (1.0 + np.exp(-z))
    # Binary target
    loan_status = rng.binomial(n=1, p=prob_default)

    df = pd.DataFrame({
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'installment': installment,
        'grade': grade,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'annual_inc': annual_inc,
        'verification_status': verification_status,
        'purpose': purpose,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'fico_score': fico_score,
        'inq_last_6mths': inq_last_6mths,
        'open_acc': open_acc,
        'pub_rec': pub_rec,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'mort_acc': mort_acc,
        'pub_rec_bankruptcies': pub_rec_bankruptcies,
        'loan_status': loan_status
    })

    return df


def save_dataset_splits(
    output_dir: str = "data",
    n_samples: int = 15000,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and partition dataset into train and test sets."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = generate_lending_club_data(n_samples=n_samples, random_state=random_state)
    df.to_csv(out_path / "loan_data.csv", index=False)

    # Train / Test split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['loan_status']
    )

    train_df.to_csv(out_path / "train.csv", index=False)
    test_df.to_csv(out_path / "test.csv", index=False)
    
    # Also save a small sample of 50 records for quick batch scoring testing
    test_df.head(50).to_csv(out_path / "sample_batch_applicants.csv", index=False)

    print(f"Dataset successfully saved to {output_dir}:")
    print(f"  Total records: {len(df):,} | Default Rate: {df['loan_status'].mean():.2%}")
    print(f"  Train records: {len(train_df):,} | Test records: {len(test_df):,}")

    return train_df, test_df


if __name__ == "__main__":
    save_dataset_splits()
