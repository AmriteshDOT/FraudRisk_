from __future__ import annotations
import math
import numpy as np


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index (symmetric KL divergence):
      PSI = sum_b (q_b - p_b) * ln(q_b / p_b)
    Thresholds: < 0.10 stable | 0.10-0.25 warning | > 0.25 critical retrain trigger.
    """
    exp_clean = expected[np.isfinite(expected)]
    act_clean = actual[np.isfinite(actual)]
    if len(exp_clean) < 10 or len(act_clean) < 10:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(exp_clean, quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts, _ = np.histogram(exp_clean, bins=edges)
    act_counts, _ = np.histogram(act_clean, bins=edges)

    eps = 1e-6
    p = np.clip(exp_counts / max(exp_counts.sum(), 1), eps, 1.0)
    q = np.clip(act_counts / max(act_counts.sum(), 1), eps, 1.0)
    return float(np.sum((q - p) * np.log(q / p)))


class CanarySPRTKillSwitch:
    """
    Wald's Sequential Probability Ratio Test (SPRT, 1945) for safe canary rollout.
    Tests H0: FPR <= p0 (safe wall c_3) vs H1: FPR >= p1 (breach).
    Immediately triggers automatic rollback if log-likelihood ratio crosses upper bound B.
    """

    def __init__(self, p0: float = 0.003, p1: float = 0.008, alpha: float = 0.01, beta: float = 0.05):
        self.p0 = p0
        self.p1 = p1
        self.upper_bound = math.log((1.0 - beta) / alpha)
        self.lower_bound = math.log(beta / (1.0 - alpha))
        self.llr = 0.0
        self.n_obs = 0

    def update(self, is_false_positive: bool) -> str:
        """Returns: 'ROLLBACK_CANARY' | 'PROMOTE_SAFE' | 'CONTINUE_MONITORING'."""
        self.n_obs += 1
        if is_false_positive:
            self.llr += math.log(self.p1 / self.p0)
        else:
            self.llr += math.log((1.0 - self.p1) / (1.0 - self.p0))

        if self.llr >= self.upper_bound:
            return "ROLLBACK_CANARY"
        if self.llr <= self.lower_bound:
            return "PROMOTE_SAFE"
        return "CONTINUE_MONITORING"
