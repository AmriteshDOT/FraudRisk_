from __future__ import annotations
from typing import Optional, TypedDict
from langgraph.graph import END, StateGraph
from fraudshield.agentic_tier2.shap_explainer import ExactTreeShapExplainer
from fraudshield.feature_store.mule_graph import IncrementalMuleGraphEngine


class InvestigationState(TypedDict):
    tx_id: str
    account_id: str
    payee_id: str
    amount_inr: float
    calibrated_prob: float
    feature_vector: list[float]
    shap_top_factors: list[tuple[str, float]]
    mule_subgraph_summary: dict
    sar_draft: str
    human_review_priority: str


def _deterministic_sar_template(state: InvestigationState) -> str:
    """Zero-hallucination fallback template if external LLM is unavailable."""
    drivers = ", ".join(f"{k}={v:+.4f}" for k, v in state["shap_top_factors"])
    g = state["mule_subgraph_summary"]
    return (
        f"[FIU-IND SAR BRIEF | PRIORITY: {state['human_review_priority']}]\n"
        f"TX_ID: {state['tx_id']} | Account: {state['account_id']} -> Payee: {state['payee_id']}\n"
        f"Amount: INR {state['amount_inr']:.2f} | Calibrated P(Fraud): {state['calibrated_prob']:.4f}\n"
        f"Exact TreeSHAP Drivers: {drivers}\n"
        f"Mule Topology: in_degree={g.get('in_degree', 0)}, "
        f"passthrough_ratio={g.get('passthrough_ratio', 0.0)}, "
        f"2hop_risk={g.get('two_hop_mule_risk', 0.0)}\n"
        f"Action Recommended: Tier-2 analyst verification and lien review."
    )


def build_tier2_investigator(
    shap_explainer: ExactTreeShapExplainer,
    mule_engine: IncrementalMuleGraphEngine,
    llm_client: Optional[object] = None,
):
    """
    Asynchronous LangGraph Tier-2 Investigator resolving:
      - c_4 (analyst cognitive load / throughput)
      - c_5 (RBI/FIU-IND verifiable narrative + TreeSHAP attribution)
    """

    def compute_shap_node(state: InvestigationState) -> dict:
        drivers = shap_explainer.top_drivers(state["feature_vector"], top_k=5)
        return {"shap_top_factors": drivers}

    def inspect_mule_graph_node(state: InvestigationState) -> dict:
        summary = mule_engine.compute_node_risk(state["payee_id"])
        summary["is_hub"] = summary["in_degree"] >= 5 and summary["passthrough_ratio"] >= 0.80
        return {"mule_subgraph_summary": summary}

    def draft_fiu_sar_node(state: InvestigationState) -> dict:
        g = state["mule_subgraph_summary"]
        priority = (
            "P0_IMMEDIATE_FREEZE"
            if (g.get("is_hub") or state["amount_inr"] >= 100_000.0 or state["calibrated_prob"] >= 0.92)
            else "P1_STANDARD_REVIEW"
        )
        enriched = {**state, "human_review_priority": priority}

        if llm_client is not None:
            prompt = (
                "Draft a strictly factual FIU-IND Suspicious Activity Report (SAR) brief.\n"
                f"Evidence:\n{_deterministic_sar_template(enriched)}\n"
                "Rules: Cite ONLY the numerical evidence above. No speculation."
            )
            draft = llm_client.invoke(prompt).content
        else:
            draft = _deterministic_sar_template(enriched)

        return {"sar_draft": draft, "human_review_priority": priority}

    workflow = StateGraph(InvestigationState)
    workflow.add_node("compute_shap", compute_shap_node)
    workflow.add_node("inspect_mule_graph", inspect_mule_graph_node)
    workflow.add_node("draft_fiu_sar", draft_fiu_sar_node)

    workflow.set_entry_point("compute_shap")
    workflow.add_edge("compute_shap", "inspect_mule_graph")
    workflow.add_edge("inspect_mule_graph", "draft_fiu_sar")
    workflow.add_edge("draft_fiu_sar", END)
    return workflow.compile()
