# Kotak FraudShield & Risk Engine — Bounded-Optimal Production Architecture

An end-to-end, production-grade real-time financial fraud, Authorized Push Payment (APP) scam, and collusive mule-network defense system engineered from first principles under the **5-Layer Irreducible Decision-Theoretic Framework**:

1. **Universe (Physics):** Controlled Markov Process $\mathcal{M}_{\text{univ}} = (\mathcal{S}, \mathcal{A}, P)$ partitioned into deductive Known Laws $(s, a, p)$ and unknown/adversarial Sampled Residuals $(s, a, s')$.
2. **Ultimate Human Goal $(\mathcal{J}, \mathcal{C})$:** Maximize net risk-adjusted banking value subject to:
   - $c_1$: Hot-path latency $\mathbb{P}(\Delta t > 50\text{ ms}) \le 0.01$ (p99 $\le 50\text{ ms}$)
   - $c_2$: Total-function soundness over entire input universe $\mathcal{U}$
   - $c_3$: False Positive Rate ceiling $\text{FPR}_{\text{block}} \le \alpha_{\max}$
   - $c_4$: Tier-2 human analyst queue capacity $\sum \mathbb{I}[a_t \in \{a_2, a_3\}] \le K_{\text{analysts}}$
   - $c_5$: Immutable cryptographic audit trail (RBI Burden-of-Proof compliance)
3. **Local Bridge $(r_t, c_t)$:** Example-Dependent Bayes Minimum Risk (Elkan 2001; Bahnsen et al. 2013/2016) + Online Dual Ascent multiplier $\mu_t$ + Split Conformal OOD abstention.
4. **BFSI Exogenous Ceiling:** $\pi^*_{\text{BFSI}}(s_t) = \arg\min_a \mathbb{E}_{P^*}[C(a, y; s_t)]$.
5. **Embedded CI Reality $f^* \in \mathcal{F}(M)$:** `FastAPI` + `Redis` sliding windows + `ONNX Runtime` dual-ensemble (Dal Pozzolo et al. 2018) + `Kafka` async cold path (`PostgreSQL` hash-chained WORM audit, incremental `NetworkX` mule graph, `TreeSHAP` + `LangGraph` FIU-IND SAR investigator).

## Quickstart (Zero-Infra Local Verification)

```bash
pip install -e ".[dev]"
python scripts/bootstrap_and_verify.py
pytest -v
```

## Full Distributed Stack (Docker Compose)

```bash
docker compose up --build
```
- Hot-Path API: `http://localhost:8000/v1/score`
- Prometheus Metrics: `http://localhost:8000/metrics`
- Healthcheck: `http://localhost:8000/health`
