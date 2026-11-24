from __future__ import annotations
from fraudshield.core.schemas import Action, TransactionEvent
from fraudshield.inference.bmr_policy import BayesMinimumRiskPolicy
from fraudshield.inference.onnx_dual_model import elkan_prior_recalibrate


def test_bmr_monetary_monotonicity():
    """Higher monetary exposure must strictly tighten optimal intervention at fixed p."""
    policy = BayesMinimumRiskPolicy(exploration_eps=0.0)
    tx_small = TransactionEvent(tx_id="TX_SM_01", account_id="ACC_1", payee_id="PAY_1", amount_inr=100.0)
    tx_large = TransactionEvent(tx_id="TX_LG_01", account_id="ACC_1", payee_id="PAY_1", amount_inr=90_000.0)

    a_small, _, _, _ = policy.select_action(tx_small, p_fraud=0.15)
    a_large, _, _, _ = policy.select_action(tx_large, p_fraud=0.15)

    severity = {Action.ALLOW: 0, Action.CHALLENGE_RBA: 1, Action.COOLOFF_HOLD: 2, Action.BLOCK: 3}
    assert severity[a_large] > severity[a_small]


def test_app_scam_prefers_hold_over_rba():
    """When active call + screen share reduce RBA efficacy, BMR must prefer COOLOFF_HOLD."""
    policy = BayesMinimumRiskPolicy(exploration_eps=0.0)
    tx_app = TransactionEvent(
        tx_id="TX_APP_01",
        account_id="ACC_1",
        payee_id="PAY_1",
        amount_inr=45_000.0,
        is_active_phone_call=True,
        is_screen_share_active=True,
    )
    costs = policy.compute_expected_costs(tx_app, p_fraud=0.35)
    assert costs[Action.COOLOFF_HOLD] < costs[Action.CHALLENGE_RBA]


def test_dual_ascent_converges_upward_on_overload():
    policy = BayesMinimumRiskPolicy(mu_analyst_dual=0.0)
    mu1 = policy.update_dual_multiplier(alerts_today=750, capacity_k=500, step_size=25.0)
    assert mu1 > 0.0
    mu2 = policy.update_dual_multiplier(alerts_today=200, capacity_k=500, step_size=25.0)
    assert mu2 < mu1


def test_elkan_prior_recalibration_exactness():
    p_corr = elkan_prior_recalibrate(p_train=0.5, pi_true=0.01, pi_train=0.5)
    assert abs(p_corr - 0.0196078) < 1e-4
