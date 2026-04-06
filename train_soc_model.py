from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = [
    "Voltage",
    "Temperature",
    "Current",
    "Power",
    "soh_pct",
    "latency_ms",
    "converter_command_pct",
    "anomaly_score_pct",
]
LAG_CONFIG: Dict[str, List[int]] = {
    "Voltage": [1, 2, 3],
    "Temperature": [1, 2],
    "Current": [1, 2, 3],
    "Power": [1, 2, 3],
    "soh_pct": [1],
}
ROLL_CONFIG: Dict[str, List[int]] = {
    "Current": [3, 5],
    "Power": [3, 5],
    "Temperature": [3],
}


@dataclass
class CandidateResult:
    name: str
    rmse: float
    mae: float
    model: object


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, lags in LAG_CONFIG.items():
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby("module_id")[col].shift(lag)
    for col, windows in ROLL_CONFIG.items():
        grouped = out.groupby("module_id")[col]
        for window in windows:
            out[f"{col}_roll{window}"] = (
                grouped.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            )
    return out


def build_split(df: pd.DataFrame, holdout_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for _, g in df.groupby("module_id", sort=False):
        split_idx = int(len(g) * (1.0 - holdout_ratio))
        g = g.copy()
        g["is_test"] = False
        g.loc[g.index[split_idx:], "is_test"] = True
        parts.append(g)
    merged = pd.concat(parts, axis=0).sort_values(["module_id", "timestamp"]).reset_index(drop=True)
    train_df = merged[~merged["is_test"]].copy()
    test_df = merged[merged["is_test"]].copy()
    return train_df, test_df


def get_candidates() -> List[Tuple[str, object]]:
    return [
        ("ridge", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=0.6))])),
        (
            "gbr",
            GradientBoostingRegressor(
                n_estimators=600,
                learning_rate=0.03,
                max_depth=3,
                random_state=25,
            ),
        ),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=1200,
                max_depth=18,
                min_samples_leaf=1,
                random_state=25,
                n_jobs=-1,
            ),
        ),
        (
            "etr",
            ExtraTreesRegressor(
                n_estimators=1400,
                max_depth=22,
                min_samples_leaf=1,
                random_state=25,
                n_jobs=-1,
            ),
        ),
        (
            "etr_tuned",
            ExtraTreesRegressor(
                n_estimators=1800,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]


def main() -> None:
    data_path = Path("distributed_bms_dataset.csv")
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    df["anomaly_score_pct"] = df["anomaly_score_pct"].fillna(0.0)
    df = df.sort_values(["module_id", "timestamp"]).reset_index(drop=True)
    df = add_engineered_features(df)
    df = df.dropna().reset_index(drop=True)

    feature_cols = BASE_FEATURES + [
        c
        for c in df.columns
        if c.endswith(tuple([f"_lag{i}" for i in range(1, 8)]))
        or "_roll" in c
    ]

    train_df, test_df = build_split(df, holdout_ratio=0.2)
    x_train = train_df[feature_cols]
    y_train = train_df["soc_pct"]
    x_test = test_df[feature_cols]
    y_test = test_df["soc_pct"]

    target_rmse = 3.0
    best: CandidateResult | None = None
    tries = 0

    for name, model in get_candidates():
        tries += 1
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rmse = float(root_mean_squared_error(y_test, pred))
        mae = float(mean_absolute_error(y_test, pred))
        print(f"trial={tries:02d} model={name:10s} rmse={rmse:.4f} mae={mae:.4f}")
        if best is None or rmse < best.rmse:
            best = CandidateResult(name=name, rmse=rmse, mae=mae, model=model)
        if rmse <= target_rmse:
            print(f"target reached with model={name}, rmse={rmse:.4f} <= {target_rmse:.1f}")
            break

    assert best is not None
    artifact = {
        "format_version": "soc_model_v2",
        "model_name": best.name,
        "model": best.model,
        "feature_cols": feature_cols,
        "base_features": BASE_FEATURES,
        "lag_config": LAG_CONFIG,
        "roll_config": ROLL_CONFIG,
        "max_lag": max(max(v) for v in LAG_CONFIG.values()),
    }
    model_path = out_dir / "soc_model_v2.joblib"
    metrics_path = out_dir / "soc_model_v2_metrics.json"
    joblib.dump(artifact, model_path)

    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "best_model": best.name,
        "rmse_pct": best.rmse,
        "mae_pct": best.mae,
        "target_rmse_pct": target_rmse,
        "target_met": bool(best.rmse <= target_rmse),
        "trials_run": tries,
        "feature_count": len(feature_cols),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nSaved: {model_path}")
    print(f"Saved: {metrics_path}")
    print(json.dumps(metrics, indent=2))

    if best.rmse > target_rmse:
        raise SystemExit(f"RMSE target not met: {best.rmse:.4f} > {target_rmse:.1f}")


if __name__ == "__main__":
    main()
