from __future__ import annotations
import numpy as np
import pandas as pd
from fraudshield.feature_store.redis_windows import FEATURE_ORDER


def generate_fraud_dataset(
    n_samples: int = 8000,
    fraud_rate: float = 0.03,
    drift_shift: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates realistic Indian UPI/IMPS transaction records covering:
      - Legitimate baseline distribution
      - Account Takeover (ATO) bursts (high 5m velocity, rooted device, odd hour)
      - APP Coercion Scams (active call + screen share + first-time payee + high balance ratio)
      - Collusive Mule Layering (high beneficiary 1h fan-in + rapid pass-through + 2-hop risk)
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    legit_amt = np.exp(rng.normal(6.5 + drift_shift, 1.1, n_legit)).clip(50.0, 95_000.0)
    legit = pd.DataFrame({
        "timestamp_utc": np.sort(rng.uniform(1_700_000_000, 1_707_000_000, n_legit)),
        "amount_inr": legit_amt,
        "amount_log1p": np.log1p(legit_amt),
        "amount_to_balance_ratio": rng.beta(1.5, 10.0, n_legit),
        "tx_count_5m": rng.poisson(0.3, n_legit).astype(float),
        "tx_count_1h": rng.poisson(1.2, n_legit).astype(float),
        "tx_sum_24h_log1p": np.log1p(legit_amt * rng.uniform(1.0, 2.5, n_legit)),
        "von_mises_hour_dev": rng.beta(1.2, 6.0, n_legit) * 2.0,
        "payee_1h_unique_senders": rng.poisson(1.0, n_legit).astype(float),
        "payee_passthrough_ratio": rng.beta(1.0, 5.0, n_legit),
        "payee_2hop_mule_risk": rng.choice([0.0, 0.35], size=n_legit, p=[0.97, 0.03]),
        "is_first_time_payee": rng.binomial(1, 0.18, n_legit).astype(float),
        "is_active_phone_call": rng.binomial(1, 0.06, n_legit).astype(float),
        "is_screen_share_active": rng.binomial(1, 0.005, n_legit).astype(float),
        "is_rooted_or_emulator": rng.binomial(1, 0.01, n_legit).astype(float),
        "propensity_logged": np.ones(n_legit),
        "label": np.zeros(n_legit, dtype=int),
    })

    fraud_amt = np.exp(rng.normal(9.4 + drift_shift, 0.9, n_fraud)).clip(500.0, 100_000.0)
    fraud = pd.DataFrame({
        "timestamp_utc": np.sort(rng.uniform(1_700_000_000, 1_707_000_000, n_fraud)),
        "amount_inr": fraud_amt,
        "amount_log1p": np.log1p(fraud_amt),
        "amount_to_balance_ratio": rng.beta(5.0, 2.0, n_fraud),
        "tx_count_5m": rng.poisson(3.2, n_fraud).astype(float),
        "tx_count_1h": rng.poisson(6.5, n_fraud).astype(float),
        "tx_sum_24h_log1p": np.log1p(fraud_amt * rng.uniform(1.5, 4.0, n_fraud)),
        "von_mises_hour_dev": rng.beta(4.0, 2.0, n_fraud) * 2.0,
        "payee_1h_unique_senders": rng.poisson(9.0, n_fraud).astype(float),
        "payee_passthrough_ratio": rng.beta(6.0, 1.5, n_fraud),
        "payee_2hop_mule_risk": rng.choice([0.0, 0.35, 0.75], size=n_fraud, p=[0.35, 0.35, 0.30]),
        "is_first_time_payee": rng.binomial(1, 0.85, n_fraud).astype(float),
        "is_active_phone_call": rng.binomial(1, 0.45, n_fraud).astype(float),
        "is_screen_share_active": rng.binomial(1, 0.28, n_fraud).astype(float),
        "is_rooted_or_emulator": rng.binomial(1, 0.30, n_fraud).astype(float),
        "propensity_logged": rng.choice([1.0, 0.02], size=n_fraud, p=[0.95, 0.05]),
        "label": np.ones(n_fraud, dtype=int),
    })

    df = pd.concat([legit, fraud], ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
    assert all(col in df.columns for col in FEATURE_ORDER)
    return df
