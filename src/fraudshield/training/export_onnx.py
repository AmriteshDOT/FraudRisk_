from __future__ import annotations
from pathlib import Path
import numpy as np
import onnxruntime as ort
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType


def export_and_verify_lgbm_onnx(
    model,
    n_features: int,
    output_path: str | Path,
    X_verify: np.ndarray,
    max_abs_tol: float = 1e-4,
) -> float:
    """
    Exports LightGBM booster to ONNX and proves numerical parity:
      max_i | P_lgbm(y=1 | x_i) - P_onnx(y=1 | x_i) | <= max_abs_tol
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=initial_type,
        target_opset=15,
    )
    onnxmltools.utils.save_model(onnx_model, str(out))

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    X_f32 = X_verify.astype(np.float32)
    lgbm_probs = model.predict_proba(X_f32)[:, 1]
    ort_out = sess.run(None, {in_name: X_f32})[1]
    onnx_probs = np.array([row[1] for row in ort_out], dtype=np.float64)

    max_diff = float(np.max(np.abs(lgbm_probs - onnx_probs)))
    if max_diff > max_abs_tol:
        raise AssertionError(f"ONNX parity violation: max_diff={max_diff:.6f} > {max_abs_tol}")
    return max_diff
