from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import math
import time


class Action(str, Enum):
    """Discrete causal intervention space A (b_a = 2 bits)."""
    ALLOW = "ALLOW"                  # a_0: Approve payment immediately
    CHALLENGE_RBA = "CHALLENGE_RBA"  # a_1: RBI 2025/2026 Risk-Based Step-Up Auth
    COOLOFF_HOLD = "COOLOFF_HOLD"    # a_2: Targeted delay + scam warning (APP scams)
    BLOCK = "BLOCK"                  # a_3: Hard decline + Tier-2 alert


class TransactionEvent(BaseModel):
    """
    Total-Function Ingress Boundary over Universe U.
    Enforces constraint c_2: rejects non-finite/negative amounts & normalizes missing fields.
    """
    tx_id: str = Field(..., min_length=4, max_length=128)
    account_id: str = Field(..., min_length=4, max_length=64)
    payee_id: str = Field(..., min_length=4, max_length=64)
    amount_inr: float = Field(..., gt=0.0, le=1e8)
    timestamp_utc: float = Field(default_factory=time.time)
    channel: str = Field(default="UPI")  # UPI | IMPS | NEFT | CARD
    lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    lon: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    device_id: str = Field(default="UNKNOWN")
    customer_clv_inr: float = Field(default=25_000.0, ge=0.0, le=1e8)
    account_balance_inr: Optional[float] = Field(default=None, ge=0.0)

    # APP Scam ("Digital Arrest" / Impersonation) passive session telemetry
    is_active_phone_call: bool = False
    is_screen_share_active: bool = False
    is_rooted_or_emulator: bool = False
    is_first_time_payee: bool = False

    @field_validator("amount_inr", "customer_clv_inr", "timestamp_utc")
    @classmethod
    def reject_non_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Non-finite numerical field rejected at boundary.")
        return v

    @field_validator("channel")
    @classmethod
    def normalize_channel(cls, v: str) -> str:
        norm = v.strip().upper()
        return norm if norm in {"UPI", "IMPS", "NEFT", "CARD"} else "OTHER"


class DecisionRecord(BaseModel):
    """Immutable audit decision artifact satisfying RBI burden-of-proof (c_5)."""
    tx_id: str
    account_id: str = "UNKNOWN"
    payee_id: str = "UNKNOWN"
    amount_inr: float = 0.0
    action: Action
    calibrated_fraud_prob: float
    conformal_set: list[int] = Field(default_factory=lambda: [0])
    expected_cost_chosen: float
    propensity_logged: float         # pi_0(a_t | o_t) for IPS debiasing
    epistemic_branch: str            # KNOWN_SAP | SAMPLED_SASP | CONFORMAL_ABSTAIN | CIRCUIT_BREAKER
    reason_codes: list[str]
    feature_snapshot: list[float] = Field(default_factory=list)
    latency_ms: float
    model_version: str
    shadow_action: Optional[Action] = None
    decision_ts_utc: float = Field(default_factory=time.time)
