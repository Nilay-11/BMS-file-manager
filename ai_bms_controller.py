from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import joblib
import pandas as pd


DRIVE_MODE_CODE = {"auto": 0.0, "charge": -1.0, "discharge": 1.0}


@dataclass
class AIBMSCurrentController:
    model_path: str
    model: object
    feature_cols: List[str]
    min_current_a: float
    max_current_a: float
    max_step_delta_a: float

    @classmethod
    def load(cls, model_path: str) -> "AIBMSCurrentController":
        artifact = joblib.load(model_path)
        if artifact.get("format_version") != "current_controller_v1":
            raise RuntimeError(
                f"Unsupported current controller format: {artifact.get('format_version')}"
            )
        return cls(
            model_path=model_path,
            model=artifact["model"],
            feature_cols=list(artifact["feature_cols"]),
            min_current_a=float(artifact["min_current_a"]),
            max_current_a=float(artifact["max_current_a"]),
            max_step_delta_a=float(artifact.get("max_step_delta_a", 16.0)),
        )

    def _clip(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def suggest_current(
        self,
        base_request_current_a: float,
        prev_current_a: float,
        avg_cell_v: float,
        min_cell_v: float,
        max_cell_v: float,
        max_temp_c: float,
        soc_ai_pct: float,
        soh_pct: float,
        anomaly_score_pct: float,
        converter_command_pct: float,
        latency_ms: float,
        low_power_mode: bool,
        drive_mode: str,
    ) -> float:
        features: Dict[str, float] = {
            "base_request_current_a": float(base_request_current_a),
            "prev_current_a": float(prev_current_a),
            "avg_cell_v": float(avg_cell_v),
            "min_cell_v": float(min_cell_v),
            "max_cell_v": float(max_cell_v),
            "max_temp_c": float(max_temp_c),
            "soc_ai_pct": float(soc_ai_pct),
            "soh_pct": float(soh_pct),
            "anomaly_score_pct": float(anomaly_score_pct),
            "converter_command_pct": float(converter_command_pct),
            "latency_ms": float(latency_ms),
            "low_power_mode": 1.0 if low_power_mode else 0.0,
            "drive_mode_code": float(DRIVE_MODE_CODE.get(drive_mode, 0.0)),
        }
        x = pd.DataFrame([features], columns=self.feature_cols)
        predicted_a = float(self.model.predict(x)[0])

        # Keep command transitions smooth to avoid oscillatory control.
        bounded_step = self._clip(
            predicted_a,
            prev_current_a - self.max_step_delta_a,
            prev_current_a + self.max_step_delta_a,
        )
        # Blend AI suggestion with base trajectory request so behavior remains intuitive.
        blended = 0.75 * bounded_step + 0.25 * float(base_request_current_a)
        return self._clip(blended, self.min_current_a, self.max_current_a)
