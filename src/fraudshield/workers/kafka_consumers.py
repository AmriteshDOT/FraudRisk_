from __future__ import annotations
import asyncio
import json
import logging
import os
import joblib
import redis.asyncio as aioredis
from aiokafka import AIOKafkaConsumer
from fraudshield.agentic_tier2.langgraph_investigator import build_tier2_investigator
from fraudshield.agentic_tier2.shap_explainer import ExactTreeShapExplainer
from fraudshield.core.audit import CryptographicAuditLedger
from fraudshield.core.schemas import Action, DecisionRecord
from fraudshield.feature_store.mule_graph import IncrementalMuleGraphEngine
from fraudshield.feature_store.redis_windows import FEATURE_ORDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudshield.workers")


async def run_cold_path_worker() -> None:
    """
    Decoupled Kafka consumer executing all non-latency-critical tasks:
      1. Cryptographic WORM audit ledger append (c_5)
      2. Incremental mule graph update -> Redis writeback
      3. TreeSHAP + LangGraph Tier-2 SAR drafting for COOLOFF_HOLD / BLOCK
    """
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    artifact_dir = os.getenv("ARTIFACT_DIR", "artifacts")

    redis_client = aioredis.from_url(redis_url)
    ledger = CryptographicAuditLedger(pg_pool=None)
    mule_engine = IncrementalMuleGraphEngine(redis_client=redis_client)

    lgbm_model = joblib.load(f"{artifact_dir}/model_del_lgbm.joblib")
    shap_exp = ExactTreeShapExplainer(lgbm_model, FEATURE_ORDER)
    investigator = build_tier2_investigator(shap_exp, mule_engine, llm_client=None)

    consumer = AIOKafkaConsumer(
        "fraud.decisions.v1",
        bootstrap_servers=kafka_bootstrap,
        group_id="fraudshield-cold-path",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    await consumer.start()
    logger.info("Cold-path worker listening on fraud.decisions.v1")
    try:
        async for msg in consumer:
            record = DecisionRecord.model_validate(msg.value)
            entry = await ledger.append(record)

            mule_engine.ingest_edge(
                record.account_id,
                record.payee_id,
                record.amount_inr,
                record.decision_ts_utc,
            )
            await mule_engine.sync_node_to_redis(record.payee_id)

            if record.action in {Action.COOLOFF_HOLD, Action.BLOCK} and record.feature_snapshot:
                out = investigator.invoke(
                    {
                        "tx_id": record.tx_id,
                        "account_id": record.account_id,
                        "payee_id": record.payee_id,
                        "amount_inr": record.amount_inr,
                        "calibrated_prob": record.calibrated_fraud_prob,
                        "feature_vector": record.feature_snapshot,
                        "shap_top_factors": [],
                        "mule_subgraph_summary": {},
                        "sar_draft": "",
                        "human_review_priority": "P1_STANDARD_REVIEW",
                    }
                )
                logger.info(
                    "Audit seq=%d | Priority=%s | TX=%s\n%s",
                    entry.seq_id,
                    out["human_review_priority"],
                    record.tx_id,
                    out["sar_draft"],
                )
    finally:
        await consumer.stop()
        await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(run_cold_path_worker())
