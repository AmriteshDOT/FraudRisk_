from __future__ import annotations
import math
from typing import Optional
from fraudshield.core.schemas import Action, TransactionEvent


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Exact great-circle distance on Earth sphere (R = 6371 km)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(min(1.0, a)))


class KnownGuardrailEngine:
    """
    Deductive (s, a, p, r, c) partition:
    Enforces hard physical, regulatory, and deterministic blocklist invariants
    in O(1) without wasting statistical capacity.
    """

    def __init__(
        self,
        redis_client,
        max_speed_kmh: float = 900.0,
        min_travel_km: float = 50.0,
        upi_hard_cap_inr: float = 100_000.0,
        app_coercion_hold_min_inr: float = 10_000.0,
    ):
        self.redis = redis_client
        self.max_speed_kmh = max_speed_kmh
        self.min_travel_km = min_travel_km
        self.upi_hard_cap_inr = upi_hard_cap_inr
        self.app_coercion_hold_min_inr = app_coercion_hold_min_inr

    async def evaluate(self, tx: TransactionEvent) -> Optional[tuple[Action, list[str]]]:
        pipe = self.redis.pipeline(transaction=False)
        pipe.sismember("known:blocked_payees", tx.payee_id)
        pipe.sismember("known:frozen_accounts", tx.account_id)
        pipe.hgetall(f"known:last_geo:{tx.account_id}")
        is_mule_payee, is_frozen, last_geo = await pipe.execute()

        if is_frozen:
            return Action.BLOCK, ["KNOWN_ACCOUNT_FROZEN"]
        if is_mule_payee:
            return Action.BLOCK, ["KNOWN_CONFIRMED_MULE_PAYEE"]

        # Hard NPCI UPI single-transaction statutory cap
        if tx.channel == "UPI" and tx.amount_inr > self.upi_hard_cap_inr:
            return Action.BLOCK, ["KNOWN_EXCEEDS_NPCI_UPI_CAP"]

        # Physical kinematic constraint: v = d / dt <= 900 km/h
        if tx.lat is not None and tx.lon is not None and last_geo:
            def _get(k: str) -> float:
                v = last_geo.get(k.encode(), last_geo.get(k))
                return float(v)

            prev_lat, prev_lon, prev_ts = _get("lat"), _get("lon"), _get("ts")
            dt_hours = max((tx.timestamp_utc - prev_ts) / 3600.0, 1e-4)
            dist_km = haversine_km(prev_lat, prev_lon, tx.lat, tx.lon)
            speed = dist_km / dt_hours
            if dist_km > self.min_travel_km and speed > self.max_speed_kmh:
                return Action.BLOCK, [f"KNOWN_IMPOSSIBLE_TRAVEL_{int(speed)}KMH"]

        # Deterministic high-precision APP scam coercion signature
        if (
            tx.is_screen_share_active
            and tx.is_active_phone_call
            and tx.is_first_time_payee
            and tx.amount_inr >= self.app_coercion_hold_min_inr
        ):
            return Action.COOLOFF_HOLD, ["KNOWN_APP_SCAM_SCREENSHARE_CALL_COERCION"]

        return None
