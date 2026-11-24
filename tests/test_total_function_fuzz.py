from __future__ import annotations
import math
from hypothesis import given, strategies as st
from pydantic import ValidationError
from fraudshield.core.schemas import TransactionEvent


@given(
    amount=st.floats(allow_nan=True, allow_infinity=True),
    clv=st.floats(allow_nan=True, allow_infinity=True),
    lat=st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True)),
    channel=st.text(min_size=0, max_size=20),
)
def test_total_function_boundary_rejects_invalid_and_never_crashes(amount, clv, lat, channel):
    """Constraint c_2 proof via Hypothesis fuzzing over U \ X_clean."""
    try:
        tx = TransactionEvent(
            tx_id="TX_FUZZ_0001",
            account_id="ACC_0001",
            payee_id="VPA_0001@upi",
            amount_inr=amount,
            customer_clv_inr=clv,
            lat=lat,
            channel=channel,
        )
        assert math.isfinite(tx.amount_inr) and 0.0 < tx.amount_inr <= 1e8
        assert math.isfinite(tx.customer_clv_inr) and 0.0 <= tx.customer_clv_inr <= 1e8
        assert tx.channel in {"UPI", "IMPS", "NEFT", "CARD", "OTHER"}
    except ValidationError:
        pass
