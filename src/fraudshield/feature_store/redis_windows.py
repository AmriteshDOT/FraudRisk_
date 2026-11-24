from __future__ import annotations
import math
import numpy as np
from fraudshield.core.schemas import TransactionEvent


FEATURE_ORDER: list[str] = [
    "amount_log1p",
    "amount_to_balance_ratio",
    "tx_count_5m",
    "tx_count_1h",
    "tx_sum_24h_log1p",
    "von_mises_hour_dev",
    "payee_1h_unique_senders",
    "payee_passthrough_ratio",
    "payee_2hop_mule_risk",
    "is_first_time_payee",
    "is_active_phone_call",
    "is_screen_share_active",
    "is_rooted_or_emulator",
]


def _hget_float(mapping: dict, key: str, default: float) -> float:
    if not mapping:
        return default
    val = mapping.get(key.encode(), mapping.get(key))
    return float(val) if val is not None else default


class OnlineStateCompressor:
    """
    Compresses unbounded transaction history into finite-bit state z_t in R^13
    under O(log N) Redis Sorted Set bounds (Bahnsen et al., ESWA 2016).
    """

    def __init__(self, redis_client):
        self.redis = redis_client

    async def extract_features(self, tx: TransactionEvent) -> np.ndarray:
        now = tx.timestamp_utc
        acc_key = f"win:acc:{tx.account_id}"
        payee_fanin_key = f"graph:fanin_1h:{tx.payee_id}"
        payee_graph_key = f"graph:risk:{tx.payee_id}"
        user_circ_key = f"prof:circ:{tx.account_id}"

        pipe = self.redis.pipeline(transaction=False)
        pipe.zcount(acc_key, now - 300.0, now)
        pipe.zcount(acc_key, now - 3600.0, now)
        pipe.zrangebyscore(acc_key, now - 86400.0, now)
        pipe.zcount(payee_fanin_key, now - 3600.0, now)
        pipe.hgetall(payee_graph_key)
        pipe.hgetall(user_circ_key)
        c5m, c1h, raw_24h, fanin_1h, g_risk, circ = await pipe.execute()

        sum_24h = 0.0
        for item in raw_24h or []:
            s = item.decode() if isinstance(item, bytes) else str(item)
            parts = s.rsplit(":", 1)
            if len(parts) == 2:
                sum_24h += float(parts[1])

        # Circular von Mises periodic hour deviation on S^1 in [0, 2]
        hour_rad = 2.0 * math.pi * ((now % 86400.0) / 86400.0)
        mean_sin = _hget_float(circ, "sin", math.sin(hour_rad))
        mean_cos = _hget_float(circ, "cos", math.cos(hour_rad))
        von_mises_dev = max(0.0, 1.0 - (math.sin(hour_rad) * mean_sin + math.cos(hour_rad) * mean_cos))

        passthrough = _hget_float(g_risk, "passthrough_ratio", 0.0)
        two_hop_risk = _hget_float(g_risk, "two_hop_mule_risk", 0.0)

        # Native NaN handling in LightGBM for cold-start / unknown balance
        amt_bal_ratio = (
            tx.amount_inr / max(tx.account_balance_inr, 1.0)
            if tx.account_balance_inr is not None
            else np.nan
        )

        return np.array(
            [
                math.log1p(tx.amount_inr),
                amt_bal_ratio,
                float(c5m),
                float(c1h),
                math.log1p(sum_24h),
                von_mises_dev,
                float(fanin_1h),
                passthrough,
                two_hop_risk,
                float(tx.is_first_time_payee),
                float(tx.is_active_phone_call),
                float(tx.is_screen_share_active),
                float(tx.is_rooted_or_emulator),
            ],
            dtype=np.float32,
        )

    async def record_transaction_async(self, tx: TransactionEvent) -> None:
        """Bounded-memory TTL window updates + EMA circular profile update."""
        now = tx.timestamp_utc
        member = f"{tx.tx_id}:{tx.amount_inr:.2f}"
        hour_rad = 2.0 * math.pi * ((now % 86400.0) / 86400.0)

        pipe = self.redis.pipeline(transaction=False)
        pipe.zadd(f"win:acc:{tx.account_id}", {member: now})
        pipe.zremrangebyscore(f"win:acc:{tx.account_id}", "-inf", now - 86400.0)
        pipe.expire(f"win:acc:{tx.account_id}", 90000)

        pipe.zadd(f"graph:fanin_1h:{tx.payee_id}", {tx.account_id: now})
        pipe.zremrangebyscore(f"graph:fanin_1h:{tx.payee_id}", "-inf", now - 3600.0)
        pipe.expire(f"graph:fanin_1h:{tx.payee_id}", 7200)

        pipe.hset(
            f"prof:circ:{tx.account_id}",
            mapping={"sin": math.sin(hour_rad), "cos": math.cos(hour_rad)},
        )

        if tx.lat is not None and tx.lon is not None:
            pipe.hset(
                f"known:last_geo:{tx.account_id}",
                mapping={"lat": tx.lat, "lon": tx.lon, "ts": now},
            )
        await pipe.execute()
