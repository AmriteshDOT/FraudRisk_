from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import Optional
from fraudshield.core.schemas import DecisionRecord


@dataclass(frozen=True)
class ChainedAuditEntry:
    """
    Cryptographic WORM Audit Record satisfying RBI Burden-of-Proof (c_5):
      H_k = SHA256(H_{k-1} || canonical_json(D_k))
    Any retroactive mutation of D_j invalidates all hashes H_k for k >= j.
    """
    seq_id: int
    prev_hash: str
    record_hash: str
    payload_json: str


class CryptographicAuditLedger:
    """
    Append-only hash-chained ledger. In production, backs onto PostgreSQL WORM
    table with REVOKE UPDATE/DELETE privileges; supports in-memory mode for tests.
    """
    GENESIS_HASH = "0" * 64

    def __init__(self, pg_pool=None):
        self.pg_pool = pg_pool
        self._last_hash: str = self.GENESIS_HASH
        self._seq: int = 0
        self._memory_chain: list[ChainedAuditEntry] = []

    @staticmethod
    def _compute_hash(prev_hash: str, canonical_payload: str) -> str:
        digest = hashlib.sha256()
        digest.update(prev_hash.encode("utf-8"))
        digest.update(b"|")
        digest.update(canonical_payload.encode("utf-8"))
        return digest.hexdigest()

    async def append(self, record: DecisionRecord) -> ChainedAuditEntry:
        canonical = json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        self._seq += 1
        rec_hash = self._compute_hash(self._last_hash, canonical)
        entry = ChainedAuditEntry(
            seq_id=self._seq,
            prev_hash=self._last_hash,
            record_hash=rec_hash,
            payload_json=canonical,
        )
        self._last_hash = rec_hash

        if self.pg_pool is not None:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO fraud_audit_worm (seq_id, tx_id, prev_hash, record_hash, payload_json)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    entry.seq_id,
                    record.tx_id,
                    entry.prev_hash,
                    entry.record_hash,
                    entry.payload_json,
                )
        else:
            self._memory_chain.append(entry)
        return entry

    @classmethod
    def verify_chain(cls, entries: list[ChainedAuditEntry]) -> tuple[bool, Optional[int]]:
        """Mathematically verifies zero tampering across the entire audit chain."""
        expected_prev = cls.GENESIS_HASH
        for entry in entries:
            if entry.prev_hash != expected_prev:
                return False, entry.seq_id
            recomputed = cls._compute_hash(entry.prev_hash, entry.payload_json)
            if recomputed != entry.record_hash:
                return False, entry.seq_id
            expected_prev = entry.record_hash
        return True, None
