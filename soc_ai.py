from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


@dataclass
class RealWorldSocModel:
    model_path: str
    history: Deque[Dict[str, float]]
    artifact: Dict[str, object]
    feature_cols: List[str]
    base_features: List[str]
    lag_config: Dict[str, List[int]]
    roll_config: Dict[str, List[int]]
    max_lag: int

    @classmethod
    def load(cls, model_path: str) -> "RealWorldSocModel":
        artifact = joblib.load(model_path)
        if artifact.get("format_version") != "soc_model_v2":
            raise RuntimeError(f"Unsupported SOC model artifact format: {artifact.get('format_version')}")
        lag_config = artifact["lag_config"]
        max_lag = int(artifact.get("max_lag", max(max(v) for v in lag_config.values())))
        history = deque(maxlen=max(128, max_lag + 16))
        return cls(
            model_path=model_path,
            history=history,
            artifact=artifact,
            feature_cols=list(artifact["feature_cols"]),
            base_features=list(artifact["base_features"]),
            lag_config={k: list(v) for k, v in lag_config.items()},
            roll_config={k: list(v) for k, v in artifact["roll_config"].items()},
            max_lag=max_lag,
        )

    def _build_feature_row(self) -> Optional[Dict[str, float]]:
        if len(self.history) <= self.max_lag:
            return None

        row: Dict[str, float] = {}
        latest = self.history[-1]
        for col in self.base_features:
            row[col] = float(latest[col])

        hist_list = list(self.history)
        for col, lags in self.lag_config.items():
            for lag in lags:
                row[f"{col}_lag{lag}"] = float(hist_list[-1 - lag][col])

        for col, windows in self.roll_config.items():
            for window in windows:
                values = [float(x[col]) for x in hist_list[-window:]]
                row[f"{col}_roll{window}"] = float(np.mean(values))

        return row

    def predict_soc_pct(
        self,
        voltage_v: float,
        current_a: float,
        temp_c: float,
        power_kw: float,
        soh_pct: float,
        dt_s: float,
        latency_s: float,
        converter_command_pct: float = 0.0,
        anomaly_score_pct: float = 0.0,
    ) -> Optional[float]:
        latency_ms = float(latency_s * 1000.0)
        self.history.append(
            {
                "Voltage": float(voltage_v),
                "Temperature": float(temp_c),
                "Current": float(current_a),
                "Power": float(power_kw),
                "soh_pct": float(soh_pct),
                "latency_ms": latency_ms,
                "converter_command_pct": float(converter_command_pct),
                "anomaly_score_pct": float(anomaly_score_pct),
                "dt_s": float(dt_s),
            }
        )

        row = self._build_feature_row()
        if row is None:
            return None

        x = pd.DataFrame([row], columns=self.feature_cols)
        pred = self.artifact["model"].predict(x)
        return float(np.clip(pred[0], 0.0, 100.0))
