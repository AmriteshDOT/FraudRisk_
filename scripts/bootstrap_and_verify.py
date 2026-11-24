from __future__ import annotations
import asyncio
import time
import numpy as np
from fraudshield.training.synthetic_generator import generate_fraud_dataset
from fraudshield.training.oot_dual_trainer import train_dual_fraud_models
from fraudshield.training.drift_monitor import compute_psi, CanarySPRTKillSwitch
from fraudshield.inference.onnx_dual_model import DualOnnxCalibratedScorer
from fraudshield.inference.bmr_policy import BayesMinimumRiskPolicy
from fraudshield.core.schemas import TransactionEvent
from fraudshield.api.main import InMemoryAsyncRedis
from fraudshield.feature_store.redis_windows import OnlineStateCompressor
from fraudshield.known_engine.deterministic_rules import KnownGuardrailEngine


async def main() -> None:
    print("[1/4] Generating synthetic UPI/APP-scam/Mule-ring datasets...")
    df_del = generate_fraud_dataset(n_samples=6000, fraud_rate=0.035, seed=10)
    df_fb = generate_fraud_dataset(n_samples=1500, fraud_rate=0.18, seed=20)

    print("[2/4] Training Dal Pozzolo Dual Ensemble + Isotonic + Split Conformal + ONNX Export...")
    metrics = train_dual_fraud_models(df_fb, df_del, out_dir="artifacts")
    for k, v in metrics.items():
        print(f"      {k}: {v:.6f}")

    print("[3/4] Benchmarking Hot-Path Latency (200 sequential requests)...")
    redis = InMemoryAsyncRedis()
    known = KnownGuardrailEngine(redis)
    comp = OnlineStateCompressor(redis)
    scorer = DualOnnxCalibratedScorer("artifacts")
    policy = BayesMinimumRiskPolicy(seed=42)

    latencies_ms = []
    for i in range(200):
        tx = TransactionEvent(
            tx_id=f"TX_BENCH_{i:05d}",
            account_id=f"ACC_{i % 25:03d}",
            payee_id=f"VPA_{i % 40:03d}@upi",
            amount_inr=float(500.0 + (i * 450) % 75_000),
            is_first_time_payee=(i % 3 == 0),
            is_active_phone_call=(i % 11 == 0),
        )
        t0 = time.perf_counter()
        hit = await known.evaluate(tx)
        if hit is None:
            z = await comp.extract_features(tx)
            pred = scorer.predict(z)
            policy.select_action(tx, pred.p_cal, pred.conformal_set)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        await comp.record_transaction_async(tx)

    p50, p95, p99 = np.percentile(latencies_ms, [50, 95, 99])
    print(f"      Latency: p50={p50:.2f}ms | p95={p95:.2f}ms | p99={p99:.2f}ms (SLA <= 50ms)")
    assert p99 <= 50.0, f"SLA c_1 breached: p99={p99:.2f}ms"

    print("[4/4] Verifying Drift Monitor & Canary SPRT Kill-Switch...")
    df_shifted = generate_fraud_dataset(n_samples=2000, fraud_rate=0.035, drift_shift=0.9, seed=99)
    psi = compute_psi(df_del["amount_log1p"].values, df_shifted["amount_log1p"].values)
    print(f"      Injected Amount Shift PSI = {psi:.4f} (> 0.25 triggers retrain)")
    assert psi > 0.25

    sprt = CanarySPRTKillSwitch(p0=0.003, p1=0.015)
    status = "CONTINUE_MONITORING"
    for _ in range(10):
        status = sprt.update(is_false_positive=True)
    print(f"      SPRT Kill-Switch under FP burst: {status}")
    assert status == "ROLLBACK_CANARY"
    print("ALL END-TO-END VERIFICATIONS PASSED.")


if __name__ == "__main__":
    asyncio.run(main())
