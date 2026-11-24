from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import onnxruntime as ort


def elkan_prior_recalibrate(
    p_train: float,
    pi_true: float,
    pi_train: float,
) -> float:
    """
    Elkan (IJCAI 2001) exact Bayes prior shift correction:
      p_true = (pi_true / pi_train) * p_train / [ (1 - p_train) + (pi_true / pi_train) * p_train ]
    Restores calibrated posteriors when training used class weighting or downsampling.
    """
    p = float(np.clip(p_train, 1e-7, 1.0 - 1e-7))
    ratio = pi_true / max(pi_train, 1e-9)
    num = ratio * p
    den = (1.0 - p) + num
    return float(np.clip(num / max(den, 1e-12), 1e-6, 1.0 - 1e-6))


@dataclass(frozen=True)
class CalibratedPrediction:
    p_cal: float
    p_fb_raw: float
    p_del_raw: float
    conformal_set: list[int]  # Subset of {0, 1}; [0, 1] indicates high epistemic uncertainty


class DualOnnxCalibratedScorer:
    """
    Dal Pozzolo et al. (IEEE TNNLS 2018) Dual-Classifier + Isotonic Calibration
    + Split Conformal Prediction (Vovk et al., 2005) for OOD abstention.
    Executes in C++ via ONNX Runtime (< 3ms per transaction).
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        weight_fb: float = 0.45,
    ):
        base = Path(artifact_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess_fb = ort.InferenceSession(
            str(base / "model_fb.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.sess_del = ort.InferenceSession(
            str(base / "model_del.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.in_fb = self.sess_fb.get_inputs()[0].name
        self.in_del = self.sess_del.get_inputs()[0].name

        meta = joblib.load(base / "calibration_bundle.joblib")
        self.calibrator = meta["isotonic"]
        self.q_hat: float = float(meta["conformal_q_hat"])
        self.pi_true: float = float(meta.get("pi_true", 0.01))
        self.pi_train: float = float(meta.get("pi_train", 0.01))
        self.w_fb = weight_fb

    @staticmethod
    def _extract_pos_prob(ort_out) -> float:
        probs = ort_out[1][0]
        return float(probs[1])

    def predict(self, z: np.ndarray) -> CalibratedPrediction:
        x_in = z.reshape(1, -1).astype(np.float32)
        p_fb = self._extract_pos_prob(self.sess_fb.run(None, {self.in_fb: x_in}))
        p_del = self._extract_pos_prob(self.sess_del.run(None, {self.in_del: x_in}))

        raw_blend = self.w_fb * p_fb + (1.0 - self.w_fb) * p_del
        p_iso = float(self.calibrator.predict(np.array([raw_blend], dtype=np.float64))[0])
        p_cal = elkan_prior_recalibrate(p_iso, self.pi_true, self.pi_train)

        # Split Conformal Prediction set C(z) = { y in {0,1} : s(z, y) <= q_hat }
        # Non-conformity score s(z, y) = 1 - P(Y = y | z)
        c_set: list[int] = []
        if p_cal <= self.q_hat:          # s(z, 0) = p_cal <= q_hat
            c_set.append(0)
        if (1.0 - p_cal) <= self.q_hat:  # s(z, 1) = 1 - p_cal <= q_hat
            c_set.append(1)
        if not c_set:
            c_set = [0, 1]

        return CalibratedPrediction(
            p_cal=p_cal,
            p_fb_raw=p_fb,
            p_del_raw=p_del,
            conformal_set=c_set,
        )
