"""
ChronosGuard: Enterprise-Scale Real-World Credit Data Ingestion & Temporal Splitter
Ingests 30,000 real-world records from UCI Credit Card Default Dataset (OpenML ID 42477),
maps features to ChronosGuard schema, assigns explicit calendar date ranges (Q1 2023 to Q1 2024),
and partitions into 5 robust chronological cohorts (6,000 samples each).
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from typing import Dict, Tuple

def generate_all_periods(save_dir: str = "data/raw", seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Ingests real-world credit data (UCI 30k benchmark), maps features to ChronosGuard schema,
    assigns chronological calendar dates, and partitions into 5 stable date-based cohorts.
    """
    os.makedirs(save_dir, exist_ok=True)
    custom_csv_path = os.environ.get("CHRONOSGUARD_DATASET_PATH", os.path.join(save_dir, "enterprise_credit.csv"))
    
    if os.path.exists(custom_csv_path):
        print(f"Loading enterprise-scale dataset from custom path: {custom_csv_path}...")
        df_raw = pd.read_csv(custom_csv_path)
    else:
        print("Fetching 30,000 real credit records from UCI Credit Card Default dataset (OpenML ID 42477)...")
        data = fetch_openml(data_id=42477, as_frame=True)
        df_raw = data.frame.copy()
        
    n_rows = len(df_raw)
    rng = np.random.default_rng(seed)
    
    expected_cols = {"credit_score", "annual_income", "debt_to_income", "loan_amount", "interest_rate", "employment_length", "revolving_utilization", "delinquencies_2yrs", "is_default"}
    if expected_cols.issubset(df_raw.columns):
        clean_df = df_raw[list(expected_cols)].copy()
    else:
        limit_bal = df_raw['x1'].astype(float)
        age = df_raw['x5'].astype(float)
        education = df_raw['x3'].astype(float)
        bill_amt1 = df_raw['x12'].astype(float)
        pay_0 = df_raw['x6'].astype(float)
        
        base_cs = 650 + (limit_bal / 5000.0) - (np.maximum(0, pay_0) * 20) - ((age - 30) * 0.5)
        credit_score = np.clip(base_cs + rng.normal(0, 15, size=n_rows), 320, 850).round(0)
        
        annual_income = np.clip(30000 + (limit_bal * 1.8) + (education * 5000) + rng.normal(0, 10000, size=n_rows), 18000, 250000).round(-2)
        
        dti_raw = np.abs(bill_amt1) / (annual_income / 12.0 + 1.0)
        debt_to_income = np.clip(dti_raw * 0.3 + 0.15 + rng.normal(0, 0.04, size=n_rows), 0.05, 0.75).round(3)
        
        loan_amount = np.clip(limit_bal * 0.25 + rng.normal(5000, 2000, size=n_rows), 1000, 40000).round(-2)
        
        base_ir = 0.10 + (np.maximum(0, pay_0) * 0.03) - ((credit_score - 650) / 1000.0)
        interest_rate = np.clip(base_ir + rng.normal(0, 0.015, size=n_rows), 0.05, 0.32).round(4)
        
        employment_length = np.clip((age - 21) * 0.4 + rng.integers(0, 3, size=n_rows), 0, 15).astype(int)
        
        util_raw = np.abs(bill_amt1) / (limit_bal + 1.0)
        revolving_utilization = np.clip(util_raw + rng.normal(0, 0.05, size=n_rows), 0.02, 0.98).round(3)
        
        delinq_raw = np.maximum(0, pay_0) + (df_raw['x7'].astype(float).clip(lower=0))
        delinquencies_2yrs = np.clip(delinq_raw + rng.poisson(0.1, size=n_rows), 0, 6).astype(int)
        
        is_default = df_raw['y'].astype(int).values
        if set(is_default) == {1, 2}:
            is_default = (is_default == 2).astype(int)
            
        clean_df = pd.DataFrame({
            "credit_score": credit_score,
            "annual_income": annual_income,
            "debt_to_income": debt_to_income,
            "loan_amount": loan_amount,
            "interest_rate": interest_rate,
            "employment_length": employment_length,
            "revolving_utilization": revolving_utilization,
            "delinquencies_2yrs": delinquencies_2yrs,
            "is_default": is_default
        })
        
    clean_df = clean_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    chunk_size = len(clean_df) // 5
    batches = {}
    
    # Define explicit date ranges and temporal drift characteristics for continuous timeline monitoring
    batch_specs = [
        ("T0_Reference", 0, "2023-01-01", "2023-03-31", 1.0, 0.0, 0.0, "Reference training baseline (Q1 2023)"),
        ("T1_Normal", 1, "2023-04-01", "2023-06-30", 1.0, 0.01, 0.0, "Production stream: Normal operation (Q2 2023)"),
        ("T2_Gradual_Drift", 2, "2023-07-01", "2023-09-30", 1.05, 0.08, 0.0, "Gradual covariate shift: Inflation & DTI expansion (Q3 2023)"),
        ("T3_Sudden_Shock", 3, "2023-10-01", "2023-12-31", 1.15, 0.20, 0.0, "Sudden covariate shock: Liquidity squeeze (Q4 2023)"),
        ("T4_Concept_Drift", 4, "2024-01-01", "2024-03-31", 1.10, 0.10, 0.25, "Concept drift: Structural risk shift (Q1 2024)")
    ]
    
    for idx, (name, pid, start_date, end_date, scale_fac, drift_fac, concept_fac, desc) in enumerate(batch_specs):
        start_idx = idx * chunk_size
        end_idx = (idx + 1) * chunk_size if idx < 4 else len(clean_df)
        batch_df = clean_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        
        # Generate random timestamps within the batch date range
        start_ts = pd.Timestamp(start_date).value // 10**9
        end_ts = pd.Timestamp(end_date).value // 10**9
        timestamps = rng.integers(start_ts, end_ts, size=len(batch_df))
        batch_df["transaction_date"] = pd.to_datetime(timestamps, unit='s').date
        
        if drift_fac > 0:
            batch_df["debt_to_income"] = np.clip(batch_df["debt_to_income"] * (1.0 + drift_fac), 0.05, 0.75).round(3)
            batch_df["interest_rate"] = np.clip(batch_df["interest_rate"] + (drift_fac * 0.05), 0.05, 0.32).round(4)
            batch_df["revolving_utilization"] = np.clip(batch_df["revolving_utilization"] * (1.0 + drift_fac), 0.02, 0.98).round(3)
            
        if concept_fac > 0:
            flip_mask = rng.random(len(batch_df)) < concept_fac
            batch_df.loc[flip_mask, "is_default"] = 1
            
        batch_df["batch_id"] = pid
        batch_df["batch_name"] = name
        batch_df.attrs["description"] = desc
        batch_df.attrs["date_range"] = f"{start_date} to {end_date}"
        
        batches[name] = batch_df
        file_path = os.path.join(save_dir, f"{name}.csv")
        batch_df.to_csv(file_path, index=False)
        print(f"Prepared Timeline Batch {name} ({start_date} to {end_date}): {len(batch_df)} rows, Default Rate: {batch_df['is_default'].mean():.2%}, saved to {file_path}")
        
    return batches

def generate_credit_batch(
    n_samples: int = 2500,
    batch_name: str = "Custom_Simulated_Stream",
    period_id: int = 99,
    covariate_drift_factor: float = 0.5,
    shock_factor: float = 0.0,
    concept_drift_factor: float = 0.0,
    seed: int = 42,
    **kwargs
) -> pd.DataFrame:
    covariate_drift_factor = kwargs.get("covariate_drift", covariate_drift_factor)
    concept_drift_factor = kwargs.get("concept_drift", concept_drift_factor)
    """
    Generate a synthetic credit batch for live drift simulation.
    Applies custom drift parameters to create realistic shifted distributions.
    
    Args:
        n_samples: Number of samples to generate
        batch_name: Name identifier for this batch
        period_id: Period ID for tracking
        covariate_drift_factor: Magnitude of covariate shift (0-2.0)
        shock_factor: Magnitude of macro shock / unemployment impact (0-2.0)
        concept_drift_factor: Probability of structural default shift (0-2.0)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with simulated credit data and applied drift
    """
    rng = np.random.default_rng(seed)
    
    # Generate synthetic data with realistic credit distributions
    credit_score = np.clip(
        650 + rng.normal(0, 80, size=n_samples) - (covariate_drift_factor * 30) - (shock_factor * 50),
        320, 850
    ).round(0)
    
    annual_income = np.clip(
        50000 + rng.normal(0, 20000, size=n_samples) - (shock_factor * 15000),
        15000, 250000
    ).round(-2)
    
    debt_to_income = np.clip(
        0.25 + (covariate_drift_factor * 0.15) + rng.normal(0, 0.08, size=n_samples) + (shock_factor * 0.1),
        0.05, 0.75
    ).round(3)
    
    loan_amount = np.clip(
        15000 + rng.normal(0, 8000, size=n_samples) + (covariate_drift_factor * 5000),
        1000, 40000
    ).round(-2)
    
    interest_rate = np.clip(
        0.12 + (covariate_drift_factor * 0.05) + (shock_factor * 0.03) + rng.normal(0, 0.02, size=n_samples),
        0.05, 0.32
    ).round(4)
    
    employment_length = np.clip(
        (rng.integers(0, 30, size=n_samples)) - (shock_factor * 3),
        0, 15
    ).astype(int)
    
    revolving_utilization = np.clip(
        0.35 + (covariate_drift_factor * 0.2) + rng.normal(0, 0.1, size=n_samples) + (shock_factor * 0.1),
        0.02, 0.98
    ).round(3)
    
    delinquencies_2yrs = np.clip(
        rng.poisson(0.5 + shock_factor * 1.5, size=n_samples) + (covariate_drift_factor * 0.5),
        0, 6
    ).astype(int)
    
    # Default probability depends on multiple factors
    base_default_prob = 0.15
    default_prob = np.clip(
        base_default_prob + (concept_drift_factor * 0.25) + (shock_factor * 0.15) + (covariate_drift_factor * 0.1),
        0.05, 0.60
    )
    is_default = (rng.random(n_samples) < default_prob).astype(int)
    
    # Create DataFrame
    sim_df = pd.DataFrame({
        "credit_score": credit_score,
        "annual_income": annual_income,
        "debt_to_income": debt_to_income,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "employment_length": employment_length,
        "revolving_utilization": revolving_utilization,
        "delinquencies_2yrs": delinquencies_2yrs,
        "is_default": is_default
    })
    
    # Add metadata attributes
    sim_df["batch_id"] = period_id
    sim_df["batch_name"] = batch_name
    sim_df.attrs["description"] = f"Simulated stream with covariate_drift={covariate_drift_factor}, shock={shock_factor}, concept_drift={concept_drift_factor}"
    sim_df.attrs["date_range"] = "Simulated"
    
    return sim_df

# Alias for backward compatibility
generate_stream_batch = generate_credit_batch
generate_simulated_stream_batch = generate_credit_batch

if __name__ == "__main__":
    generate_all_periods()

