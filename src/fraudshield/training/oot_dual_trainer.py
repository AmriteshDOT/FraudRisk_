from __future__ import annotations
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from fraudshield.feature_store.redis_windows import FEATURE_ORDER
from fraudshield.training.export_onnx import export_and_verify_lgbm_onnx


def train_dual_fraud_models(
    df_feedback: pd.DataFrame,
    df_delayed: pd.DataFrame,
    out_dir: str | Path = "artifacts",
    weight_fb: float = 0.45,
    conformal_alpha: float = 0.05,
) -> dict[str, float]:
    """
    Dal Pozzolo et al. (IEEE TNNLS 2018) Verification-Latency Dual Training:
      - f_FB: trained on recent immediate analyst feedback [T-7d, T]
      - f_DEL: trained on matured delayed ground truth [T-120d, T-30d]
      - IPS debiasing weights clipped at [1, 20] (Swaminathan & Joachims)
      - Strict Out-Of-Time (OOT) split for Isotonic Calibration + Split Conformal q_hat
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _fit_gbdt(df: pd.DataFrame, seed: int) -> lgb.LGBMClassifier:
        ips_w = np.clip(1.0 / np.clip(df["propensity_logged"].values, 0.05, 1.0), 1.0, 20.0)
        amt_w = np.log1p(df["amount_inr"].values)
        sample_w = ips_w * amt_w

        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        clf.fit(
            df[FEATURE_ORDER].values.astype(np.float32),
            df["label"].values.astype(int),
            sample_weight=sample_w,
        )
        return clf

    df_del = df_delayed.sort_values("timestamp_utc").reset_index(drop=True)
    n = len(df_del)
    i_cal = int(n * 0.70)
    i_test = int(n * 0.85)
    del_train = df_del.iloc[:i_cal]
    del_cal = df_del.iloc[i_cal:i_test]
    del_test = df_del.iloc[i_test:]

    model_fb = _fit_gbdt(df_feedback, seed=42)
    model_del = _fit_gbdt(del_train, seed=43)

    X_cal = del_cal[FEATURE_ORDER].values.astype(np.float32)
    y_cal = del_cal["label"].values.astype(int)
    raw_cal = weight_fb * model_fb.predict_proba(X_cal)[:, 1] + (1.0 - weight_fb) * model_del.predict_proba(X_cal)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal, y_cal)

    p_cal_hat = np.clip(iso.predict(raw_cal), 1e-6, 1.0 - 1e-6)
    nonconf = np.where(y_cal == 1, 1.0 - p_cal_hat, p_cal_hat)
    n_cal = len(nonconf)
    q_level = min(1.0, np.ceil((n_cal + 1) * (1.0 - conformal_alpha)) / n_cal)
    q_hat = float(np.quantile(nonconf, q_level, method="higher"))

    X_test = del_test[FEATURE_ORDER].values.astype(np.float32)
    y_test = del_test["label"].values.astype(int)
    raw_test = weight_fb * model_fb.predict_proba(X_test)[:, 1] + (1.0 - weight_fb) * model_del.predict_proba(X_test)[:, 1]
    p_test = np.clip(iso.predict(raw_test), 1e-6, 1.0 - 1e-6)

    pr_auc = float(average_precision_score(y_test, p_test))
    brier = float(brier_score_loss(y_test, p_test))

    parity_fb = export_and_verify_lgbm_onnx(model_fb, len(FEATURE_ORDER), out / "model_fb.onnx", X_test[:128])
    parity_del = export_and_verify_lgbm_onnx(model_del, len(FEATURE_ORDER), out / "model_del.onnx", X_test[:128])

    joblib.dump(model_del, out / "model_del_lgbm.joblib")
    joblib.dump(
        {
            "isotonic": iso,
            "conformal_q_hat": q_hat,
            "pi_true": float(y_cal.mean()),
            "pi_train": float(y_cal.mean()),
            "feature_order": FEATURE_ORDER,
        },
        out / "calibration_bundle.joblib",
    )

    return {
        "oot_pr_auc": pr_auc,
        "oot_brier": brier,
        "conformal_q_hat": q_hat,
        "onnx_parity_fb": parity_fb,
        "onnx_parity_del": parity_del,
    }
