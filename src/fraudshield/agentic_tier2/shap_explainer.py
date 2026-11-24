from __future__ import annotations
import numpy as np
import shap


class ExactTreeShapExplainer:
    """
    Exact TreeSHAP local attribution in O(T * L * D^2) (Lundberg & Lee, NeurIPS 2017).
    Runs strictly on the asynchronous cold path to preserve hot-path p99 <= 50ms.
    """

    def __init__(self, lgbm_model, feature_names: list[str]):
        self.explainer = shap.TreeExplainer(lgbm_model)
        self.feature_names = feature_names

    def top_drivers(self, feature_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        x = np.array([feature_vector], dtype=np.float32)
        sv = self.explainer.shap_values(x)
        vals = sv[1][0] if isinstance(sv, list) else sv[0]
        ranked = sorted(
            zip(self.feature_names, [round(float(v), 5) for v in vals]),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        return ranked[:top_k]
