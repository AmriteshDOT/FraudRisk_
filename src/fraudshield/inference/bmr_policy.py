from __future__ import annotations
from typing import Any, Optional
import numpy as np
from fraudshield.core.schemas import Action, TransactionEvent


class BayesMinimumRiskPolicy:
    """
    Solves constrained Bayes Minimum Risk (Elkan 2001; Bahnsen et al. 2013/2016):
      a* = argmin_{a in A} [ p * C(a,1;x) + (1-p) * C(a,0;x) + mu_t * I(a in {HOLD, BLOCK}) ]
    with:
      - Online Projected Dual Ascent on mu_t to satisfy analyst capacity c_4
      - Controlled epsilon-propensity exploration for IPS Sample Selection Bias debiasing
      - Conformal uncertainty escalation when C(z) == [0, 1] on high-value txns
    """

    def __init__(
        self,
        cost_cfg: Optional[dict[str, Any]] = None,
        mu_analyst_dual: float = 0.0,
        exploration_eps: float = 0.02,
        max_explore_amount_inr: float = 2_000.0,
        max_explore_prob: float = 0.80,
        seed: Optional[int] = None,
    ):
        self.cfg = cost_cfg or {}
        self.mu_t = float(max(0.0, mu_analyst_dual))
        self.eps = float(exploration_eps)
        self.max_explore_amt = float(max_explore_amount_inr)
        self.max_explore_prob = float(max_explore_prob)
        self.rng = np.random.default_rng(seed)

    def _bank_liability_inr(self, tx: TransactionEvent) -> float:
        rep = float(self.cfg.get("rbi_liability", {}).get("reputational_overhead_inr", 250.0))
        return tx.amount_inr + rep

    def compute_expected_costs(self, tx: TransactionEvent, p_fraud: float) -> dict[Action, float]:
        iv = self.cfg.get("interventions", {})
        liab = self._bank_liability_inr(tx)
        clv = tx.customer_clv_inr

        app_coercion = tx.is_active_phone_call or tx.is_screen_share_active
        eta_rba = float(iv.get("rba_efficacy_app_scam", 0.25) if app_coercion else iv.get("rba_efficacy_ato", 0.85))
        eta_hold = float(iv.get("hold_efficacy_app_scam", 0.88) if app_coercion else iv.get("hold_efficacy_ato", 0.65))

        rba_cost = float(iv.get("rba_direct_cost_inr", 15.0))
        rba_drop = float(iv.get("rba_legit_dropoff_rate", 0.015))
        hold_cost = float(iv.get("hold_direct_cost_inr", 60.0))
        hold_churn = float(iv.get("hold_legit_churn_rate", 0.04))
        hold_ops = float(iv.get("hold_ops_cost_inr", 120.0))
        blk_insult = float(iv.get("block_insult_cost_inr", 150.0))
        blk_churn = float(iv.get("block_legit_churn_rate", 0.12))
        blk_ops = float(iv.get("block_ops_cost_inr", 120.0))

        cost_matrix = {
            Action.ALLOW: (0.0, liab),
            Action.CHALLENGE_RBA: (
                rba_cost + rba_drop * clv,
                (1.0 - eta_rba) * liab + rba_cost,
            ),
            Action.COOLOFF_HOLD: (
                hold_cost + hold_churn * clv + self.mu_t,
                (1.0 - eta_hold) * liab + hold_ops + self.mu_t,
            ),
            Action.BLOCK: (
                blk_insult + blk_churn * clv + self.mu_t,
                blk_ops + self.mu_t,
            ),
        }
        return {
            action: (1.0 - p_fraud) * c0 + p_fraud * c1
            for action, (c0, c1) in cost_matrix.items()
        }

    def select_action(
        self,
        tx: TransactionEvent,
        p_fraud: float,
        conformal_set: Optional[list[int]] = None,
    ) -> tuple[Action, float, float, str]:
        expected_risk = self.compute_expected_costs(tx, p_fraud)
        best_action = min(expected_risk, key=expected_risk.get)

        if (
            conformal_set == [0, 1]
            and best_action == Action.ALLOW
            and tx.amount_inr >= 25_000.0
        ):
            return (
                Action.CHALLENGE_RBA,
                expected_risk[Action.CHALLENGE_RBA],
                1.0,
                "CONFORMAL_ABSTAIN",
            )

        propensity = 1.0
        if (
            best_action == Action.BLOCK
            and tx.amount_inr <= self.max_explore_amt
            and p_fraud < self.max_explore_prob
            and self.eps > 0.0
        ):
            if self.rng.random() < self.eps:
                return (
                    Action.CHALLENGE_RBA,
                    expected_risk[Action.CHALLENGE_RBA],
                    self.eps,
                    "SAMPLED_SASP",
                )
            propensity = 1.0 - self.eps

        return best_action, expected_risk[best_action], propensity, "SAMPLED_SASP"

    def update_dual_multiplier(
        self,
        alerts_today: int,
        capacity_k: int,
        step_size: float = 25.0,
    ) -> float:
        """
        Projected Dual Ascent:
          mu_{t+1} = max(0, mu_t + eta * (alerts_t - K) / K)
        Enforces Tier-2 human analyst queue capacity c_4 without hand-tuned thresholds.
        """
        violation = (alerts_today - capacity_k) / max(capacity_k, 1)
        self.mu_t = max(0.0, self.mu_t + step_size * violation)
        return self.mu_t
