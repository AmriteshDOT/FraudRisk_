from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FraudShieldConfig:
    """Centralized loader for mathematical constraints C and cost matrix parameters."""

    def __init__(self, config_dir: str | Path = "configs"):
        base = Path(config_dir)
        self.constraints: dict[str, Any] = load_yaml(base / "constraints.yaml")
        self.costs: dict[str, Any] = load_yaml(base / "cost_matrix.yaml")
