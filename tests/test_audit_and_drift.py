from __future__ import annotations
import asyncio
import numpy as np
from fraudshield.core.audit import ChainedAuditEntry, CryptographicAuditLedger
from fraudshield.core.schemas import Action, DecisionRecord
from fraudshield.training.drift_monitor import CanarySPRTKillSwitch, compute_psi


def test_worm_hash_chain_detects_tampering():
    async def _run():
        ledger = CryptographicAuditLedger()
        entries = []
        for i in range(5):
            rec = DecisionRecord(
                tx_id=f"TX_AUDIT_{i:03d}",
                action=Action.ALLOW,
                calibrated_fraud_prob=0.01,
                expected_cost_chosen=1.2,
                propensity_logged=1.0,
                epistemic_branch="SAMPLED_SASP",
                reason_codes=["OK"],
                latency_ms=4.5,
                model_version="v1",
            )
            entries.append(await ledger.append(rec))

        ok, bad_seq = CryptographicAuditLedger.verify_chain(entries)
        assert ok is True and bad_seq is None

        tampered = list(entries)
        tampered[1] = ChainedAuditEntry(
            seq_id=entries[1].seq_id,
            prev_hash=entries[1].prev_hash,
            record_hash=entries[1].record_hash,
            payload_json=entries[1].payload_json.replace("ALLOW", "BLOCK"),
        )
        ok2, bad_seq2 = CryptographicAuditLedger.verify_chain(tampered)
        assert ok2 is False and bad_seq2 == 2

    asyncio.run(_run())


def test_psi_and_sprt_guards():
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 1.0, 5000)
    same = rng.normal(0.0, 1.0, 5000)
    drifted = rng.normal(1.2, 1.3, 5000)
    assert compute_psi(base, same) < 0.05
    assert compute_psi(base, drifted) > 0.25

    sprt = CanarySPRTKillSwitch(p0=0.003, p1=0.015)
    out = "CONTINUE_MONITORING"
    for _ in range(8):
        out = sprt.update(is_false_positive=True)
    assert out == "ROLLBACK_CANARY"
