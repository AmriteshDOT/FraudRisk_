from __future__ import annotations
import asyncio
import time
import numpy as np
from fraudshield.api.main import InMemoryAsyncRedis
from fraudshield.core.schemas import TransactionEvent
from fraudshield.feature_store.redis_windows import OnlineStateCompressor
from fraudshield.inference.bmr_policy import BayesMinimumRiskPolicy
from fraudshield.known_engine.deterministic_rules import KnownGuardrailEngine


def test_hot_path_feature_and_bmr_latency_under_sla():
    """Verifies known-engine + state compression + BMR stays far under 50ms p99."""
    async def _bench():
        redis = InMemoryAsyncRedis()
        known = KnownGuardrailEngine(redis)
        comp = OnlineStateCompressor(redis)
        policy = BayesMinimumRiskPolicy(exploration_eps=0.0)

        lat_ms = []
        for i in range(150):
            tx = TransactionEvent(
                tx_id=f"TX_LAT_{i:04d}",
                account_id=f"ACC_{i % 10}",
                payee_id=f"VPA_{i % 20}@upi",
                amount_inr=12_500.0,
            )
            t0 = time.perf_counter()
            await known.evaluate(tx)
            _ = await comp.extract_features(tx)
            policy.select_action(tx, p_fraud=0.08, conformal_set=[0])
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
            await comp.record_transaction_async(tx)
        return float(np.percentile(lat_ms, 99))

    p99 = asyncio.run(_bench())
    assert p99 < 15.0, f"Feature+BMR p99={p99:.2f}ms leaves insufficient budget for ONNX"
