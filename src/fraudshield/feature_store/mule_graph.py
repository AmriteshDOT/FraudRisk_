from __future__ import annotations
import networkx as nx


class IncrementalMuleGraphEngine:
    """
    Asynchronous Cold-Path Collusive Mule Topology Detector.
    Computes:
      1. Pass-through ratio: sum(out_amt) / max(sum(in_amt), 1)
      2. Fan-in concentration: unique_in_senders
      3. 2-hop proximity to confirmed mule accounts (decayed harmonic risk)
    Writes precomputed topological features back to Redis so the hot path
    stays strictly O(1) without live graph traversal.
    """

    def __init__(self, redis_client=None):
        self.graph = nx.DiGraph()
        self.confirmed_mules: set[str] = set()
        self.redis = redis_client

    def mark_confirmed_mule(self, account_id: str) -> None:
        self.confirmed_mules.add(account_id)

    def ingest_edge(self, src: str, dst: str, amount_inr: float, ts: float) -> dict[str, float]:
        if self.graph.has_edge(src, dst):
            self.graph[src][dst]["amount"] += amount_inr
            self.graph[src][dst]["count"] += 1
            self.graph[src][dst]["last_ts"] = ts
        else:
            self.graph.add_edge(src, dst, amount=amount_inr, count=1, last_ts=ts)

        return self.compute_node_risk(dst)

    def compute_node_risk(self, node_id: str) -> dict[str, float]:
        if node_id not in self.graph:
            return {"passthrough_ratio": 0.0, "two_hop_mule_risk": 0.0, "in_degree": 0.0}

        in_amt = sum(d.get("amount", 0.0) for _, _, d in self.graph.in_edges(node_id, data=True))
        out_amt = sum(d.get("amount", 0.0) for _, _, d in self.graph.out_edges(node_id, data=True))
        passthrough = min(out_amt / max(in_amt, 1.0), 2.0)

        # 2-hop undirected neighborhood proximity to confirmed mules
        undirected = self.graph.to_undirected(as_view=True)
        risk = 0.0
        if node_id in self.confirmed_mules:
            risk = 1.0
        else:
            lengths = nx.single_source_shortest_path_length(undirected, node_id, cutoff=2)
            for neighbor, dist in lengths.items():
                if neighbor in self.confirmed_mules and dist > 0:
                    risk = max(risk, 0.75 if dist == 1 else 0.35)

        return {
            "passthrough_ratio": round(passthrough, 4),
            "two_hop_mule_risk": round(risk, 4),
            "in_degree": float(self.graph.in_degree(node_id)),
        }

    async def sync_node_to_redis(self, node_id: str) -> dict[str, float]:
        metrics = self.compute_node_risk(node_id)
        if self.redis is not None:
            await self.redis.hset(
                f"graph:risk:{node_id}",
                mapping={
                    "passthrough_ratio": metrics["passthrough_ratio"],
                    "two_hop_mule_risk": metrics["two_hop_mule_risk"],
                },
            )
            await self.redis.expire(f"graph:risk:{node_id}", 86400)
        return metrics
