from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import os
import time
from typing import Any
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from fraudshield.core.config import FraudShieldConfig
from fraudshield.core.metrics import (
    CIRCUIT_BREAKER_TOTAL,
    DECISIONS_TOTAL,
    DUAL_MULTIPLIER_GAUGE,
    HOT_PATH_LATENCY_MS,
    SLA_BREACHES_TOTAL,
)
from fraudshield.core.schemas import Action, DecisionRecord, TransactionEvent
from fraudshield.feature_store.redis_windows import OnlineStateCompressor
from fraudshield.inference.bmr_policy import BayesMinimumRiskPolicy
from fraudshield.inference.onnx_dual_model import DualOnnxCalibratedScorer
from fraudshield.known_engine.deterministic_rules import KnownGuardrailEngine

MODEL_VERSION = "dual-lgbm-v2.4-onnx"


class InMemoryAsyncRedis:
    """Drop-in async Redis subset for zero-infra unit tests & local benchmarks."""

    def __init__(self):
        self.kv: dict[str, bytes] = {}
        self.sets: dict[str, set[str]] = {}
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    async def get(self, key: str):
        return self.kv.get(key)

    async def setex(self, key: str, ttl: int, val: str):
        self.kv[key] = val.encode() if isinstance(val, str) else val

    async def hset(self, key: str, mapping: dict):
        h = self.hashes.setdefault(key, {})
        for k, v in mapping.items():
            h[str(k).encode()] = str(v).encode()

    async def expire(self, key: str, ttl: int):
        return True

    def pipeline(self, transaction: bool = False):
        return _InMemoryPipeline(self)


class _InMemoryPipeline:
    def __init__(self, store: InMemoryAsyncRedis):
        self.store = store
        self.ops: list[tuple] = []

    def sismember(self, key: str, val: str):
        self.ops.append(("sismember", key, val))

    def hgetall(self, key: str):
        self.ops.append(("hgetall", key))

    def zcount(self, key: str, lo: float, hi: float):
        self.ops.append(("zcount", key, lo, hi))

    def zrangebyscore(self, key: str, lo: float, hi: float):
        self.ops.append(("zrangebyscore", key, lo, hi))

    def zadd(self, key: str, mapping: dict[str, float]):
        self.ops.append(("zadd", key, mapping))

    def zremrangebyscore(self, key: str, lo: Any, hi: float):
        self.ops.append(("zremrangebyscore", key, lo, hi))

    def expire(self, key: str, ttl: int):
        self.ops.append(("expire", key, ttl))

    def hset(self, key: str, mapping: dict):
        self.ops.append(("hset", key, mapping))

    async def execute(self):
        out = []
        for op in self.ops:
            cmd = op[0]
            if cmd == "sismember":
                out.append(op[2] in self.store.sets.get(op[1], set()))
            elif cmd == "hgetall":
                out.append(self.store.hashes.get(op[1], {}))
            elif cmd == "zcount":
                z = self.store.zsets.get(op[1], {})
                out.append(sum(1 for sc in z.values() if op[2] <= sc <= op[3]))
            elif cmd == "zrangebyscore":
                z = self.store.zsets.get(op[1], {})
                out.append([m.encode() for m, sc in z.items() if op[2] <= sc <= op[3]])
            elif cmd == "zadd":
                z = self.store.zsets.setdefault(op[1], {})
                z.update(op[2])
                out.append(len(op[2]))
            elif cmd == "zremrangebyscore":
                z = self.store.zsets.get(op[1], {})
                to_del = [m for m, sc in z.items() if sc <= op[3]]
                for m in to_del:
                    del z[m]
                out.append(len(to_del))
            elif cmd == "hset":
                await self.store.hset(op[1], op[2])
                out.append(True)
            else:
                out.append(True)
        return out


class NullAsyncKafkaProducer:
    """In-memory topic capture when Kafka is disabled in local tests."""

    def __init__(self):
        self.events: list[tuple[str, bytes]] = []

    async def send(self, topic: str, payload: bytes):
        self.events.append((topic, payload))


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = FraudShieldConfig(os.getenv("CONFIG_DIR", "configs"))
    artifact_dir = os.getenv("ARTIFACT_DIR", "artifacts")

    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(redis_url)
    else:
        redis_client = InMemoryAsyncRedis()

    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP")
    if kafka_bootstrap:
        from aiokafka import AIOKafkaProducer
        kafka_prod = AIOKafkaProducer(bootstrap_servers=kafka_bootstrap)
        await kafka_prod.start()
    else:
        kafka_prod = NullAsyncKafkaProducer()

    kp = cfg.constraints["known_physics"]
    sw = cfg.constraints["statistical_walls"]
    lat = cfg.constraints["latency"]

    state["cfg"] = cfg
    state["redis"] = redis_client
    state["kafka"] = kafka_prod
    state["sla_timeout_sec"] = float(lat["hot_path_timeout_ms"]) / 1000.0
    state["sla_p99_ms"] = float(lat["p99_ceiling_ms"])
    state["cb_high_val_inr"] = float(lat["circuit_breaker_high_val_inr"])
    state["known"] = KnownGuardrailEngine(
        redis_client,
        max_speed_kmh=kp["max_travel_speed_kmh"],
        min_travel_km=kp["min_impossible_travel_km"],
        upi_hard_cap_inr=kp["upi_hard_cap_inr"],
        app_coercion_hold_min_inr=kp["app_coercion_hold_min_inr"],
    )
    state["compressor"] = OnlineStateCompressor(redis_client)
    state["scorer"] = DualOnnxCalibratedScorer(
        artifact_dir=artifact_dir,
        weight_fb=cfg.costs["dual_ensemble"]["weight_feedback_model"],
    )
    state["policy"] = BayesMinimumRiskPolicy(
        cost_cfg=cfg.costs,
        exploration_eps=sw["ips_exploration_eps"],
        max_explore_amount_inr=sw["ips_max_explore_amount_inr"],
        max_explore_prob=sw["ips_max_prob_explore"],
    )
    DUAL_MULTIPLIER_GAUGE.set(state["policy"].mu_t)
    yield
    if hasattr(kafka_prod, "stop"):
        await kafka_prod.stop()
    if hasattr(redis_client, "aclose"):
        await redis_client.aclose()


app = FastAPI(title="Kotak FraudShield Hot-Path Engine", lifespan=lifespan)


@app.exception_handler(Exception)
async def total_function_safety_net(request: Request, exc: Exception) -> JSONResponse:
    """Constraint c_2: Total-function guarantee over entire input universe U."""
    CIRCUIT_BREAKER_TOTAL.labels(reason=type(exc).__name__).inc()
    fallback = DecisionRecord(
        tx_id="MALFORMED_OR_FAULT",
        action=Action.CHALLENGE_RBA,
        calibrated_fraud_prob=0.5,
        expected_cost_chosen=0.0,
        propensity_logged=1.0,
        epistemic_branch="CIRCUIT_BREAKER",
        reason_codes=[f"TOTAL_FN_SAFE_FALLBACK:{type(exc).__name__}"],
        latency_ms=0.0,
        model_version=MODEL_VERSION,
    )
    return JSONResponse(status_code=200, content=fallback.model_dump())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/score", response_model=DecisionRecord)
async def score_transaction(tx: TransactionEvent) -> DecisionRecord:
    t0 = time.perf_counter()
    redis = state["redis"]
    known: KnownGuardrailEngine = state["known"]
    compressor: OnlineStateCompressor = state["compressor"]
    scorer: DualOnnxCalibratedScorer = state["scorer"]
    policy: BayesMinimumRiskPolicy = state["policy"]
    kafka = state["kafka"]

    # 0. Exact-once Idempotency Guarantee under NPCI retry storms
    cached = await redis.get(f"idemp:{tx.tx_id}")
    if cached:
        return DecisionRecord.model_validate_json(cached)

    try:
        async with asyncio.timeout(state["sla_timeout_sec"]):
            # Step 1: Deductive Known Partition (s, a, p, r, c)
            known_hit = await known.evaluate(tx)
            if known_hit is not None:
                action, reasons = known_hit
                record = DecisionRecord(
                    tx_id=tx.tx_id,
                    account_id=tx.account_id,
                    payee_id=tx.payee_id,
                    amount_inr=tx.amount_inr,
                    action=action,
                    calibrated_fraud_prob=1.0 if action == Action.BLOCK else 0.85,
                    conformal_set=[1],
                    expected_cost_chosen=0.0,
                    propensity_logged=1.0,
                    epistemic_branch="KNOWN_SAP",
                    reason_codes=reasons,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    model_version=MODEL_VERSION,
                )
            else:
                # Step 2: Sampled Partition (s, a, s', r, c)
                z_t = await compressor.extract_features(tx)
                pred = scorer.predict(z_t)
                action, exp_cost, propensity, branch = policy.select_action(
                    tx, pred.p_cal, pred.conformal_set
                )
                record = DecisionRecord(
                    tx_id=tx.tx_id,
                    account_id=tx.account_id,
                    payee_id=tx.payee_id,
                    amount_inr=tx.amount_inr,
                    action=action,
                    calibrated_fraud_prob=pred.p_cal,
                    conformal_set=pred.conformal_set,
                    expected_cost_chosen=exp_cost,
                    propensity_logged=propensity,
                    epistemic_branch=branch,
                    reason_codes=["BMR_OPTIMAL_DECISION" if branch == "SAMPLED_SASP" else "CONFORMAL_OOD_UPGRADE"],
                    feature_snapshot=[float(x) if x == x else 0.0 for x in z_t],
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    model_version=MODEL_VERSION,
                )
    except (TimeoutError, Exception) as exc:
        CIRCUIT_BREAKER_TOTAL.labels(reason=type(exc).__name__).inc()
        safe_action = (
            Action.COOLOFF_HOLD
            if tx.amount_inr >= state["cb_high_val_inr"]
            else Action.CHALLENGE_RBA
        )
        record = DecisionRecord(
            tx_id=tx.tx_id,
            account_id=tx.account_id,
            payee_id=tx.payee_id,
            amount_inr=tx.amount_inr,
            action=safe_action,
            calibrated_fraud_prob=0.5,
            conformal_set=[0, 1],
            expected_cost_chosen=0.0,
            propensity_logged=1.0,
            epistemic_branch="CIRCUIT_BREAKER",
            reason_codes=[f"SLA_BREACH_FALLBACK:{type(exc).__name__}"],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            model_version=MODEL_VERSION,
        )

    HOT_PATH_LATENCY_MS.observe(record.latency_ms)
    DECISIONS_TOTAL.labels(action=record.action.value, epistemic_branch=record.epistemic_branch).inc()
    if record.latency_ms > state["sla_p99_ms"]:
        SLA_BREACHES_TOTAL.inc()

    asyncio.create_task(redis.setex(f"idemp:{tx.tx_id}", 300, record.model_dump_json()))
    asyncio.create_task(compressor.record_transaction_async(tx))
    asyncio.create_task(kafka.send("fraud.decisions.v1", record.model_dump_json().encode()))
    return record
