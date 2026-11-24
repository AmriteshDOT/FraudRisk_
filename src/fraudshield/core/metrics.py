from __future__ import annotations
from prometheus_client import Counter, Gauge, Histogram

HOT_PATH_LATENCY_MS = Histogram(
    "fraudshield_hot_path_latency_ms",
    "End-to-end synchronous scoring latency in milliseconds (SLA p99 <= 50ms)",
    buckets=(1, 2, 5, 10, 15, 20, 25, 30, 38, 50, 75, 100, 250),
)

DECISIONS_TOTAL = Counter(
    "fraudshield_decisions_total",
    "Total decisions partitioned by action and epistemic branch",
    ["action", "epistemic_branch"],
)

SLA_BREACHES_TOTAL = Counter(
    "fraudshield_sla_breaches_total",
    "Count of transactions breaching the 50ms latency ceiling c_1",
)

CIRCUIT_BREAKER_TOTAL = Counter(
    "fraudshield_circuit_breaker_total",
    "Count of fallback circuit-breaker activations",
    ["reason"],
)

DUAL_MULTIPLIER_GAUGE = Gauge(
    "fraudshield_analyst_dual_multiplier_mu",
    "Current online Lagrange dual multiplier mu_t enforcing analyst capacity c_4",
)

FEATURE_PSI_GAUGE = Gauge(
    "fraudshield_feature_psi",
    "Population Stability Index measuring covariate drift per feature",
    ["feature_name"],
)
