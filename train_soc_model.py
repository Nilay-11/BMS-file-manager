from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
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
    "Voltage": [1, 2, 3, 4],
    "Temperature": [1, 2, 3],
    "Current": [1, 2, 3, 4],
    "Power": [1, 2, 3, 4],
    "soh_pct": [1, 2],
    "latency_ms": [1, 2],
}
ROLL_CONFIG: Dict[str, List[int]] = {
    "Current": [3, 5, 7],
    "Power": [3, 5, 7],
    "Temperature": [3, 5],
    "Voltage": [3, 5],
}


@dataclass
class CandidateResult:
    name: str
    cv_rmse: float
    cv_mae: float
    model: object
    holdout_rmse: float
    holdout_mae: float


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, lags in LAG_CONFIG.items():
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby("module_id")[col].shift(lag)
    for col, windows in ROLL_CONFIG.items():
        grouped = out.groupby("module_id")[col]
        for window in windows:
            out[f"{col}_roll{window}"] = grouped.rolling(window, min_periods=1).mean().reset_index(
                level=0, drop=True
            )
    return out


def build_holdout_split(df: pd.DataFrame, holdout_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for _, g in df.groupby("module_id", sort=False):
        split_idx = int(len(g) * (1.0 - holdout_ratio))
        split_idx = max(1, min(len(g) - 1, split_idx))
        g = g.copy()
        g["is_test"] = False
        g.loc[g.index[split_idx:], "is_test"] = True
        parts.append(g)
    merged = pd.concat(parts, axis=0).sort_values(["module_id", "timestamp"]).reset_index(drop=True)
    train_df = merged[~merged["is_test"]].copy()
    test_df = merged[merged["is_test"]].copy()
    return train_df, test_df


def build_time_cv_folds(
    train_df: pd.DataFrame,
    n_folds: int = 3,
    val_ratio: float = 0.12,
    min_train_ratio: float = 0.45,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    modules = [(m, g.copy()) for m, g in train_df.groupby("module_id", sort=False)]
    if not modules:
        return []

    module_fold_points: Dict[str, List[int]] = {}
    module_val_len: Dict[str, int] = {}
    for module_id, g in modules:
        n = len(g)
        val_len = max(8, int(n * val_ratio))
        val_len = min(val_len, max(2, n // 3))
        min_train_end = max(10, int(n * min_train_ratio))
        max_train_end = n - val_len
        if max_train_end <= min_train_end:
            module_fold_points[module_id] = [max(1, n - val_len)]
            module_val_len[module_id] = val_len
            continue
        points = np.linspace(min_train_end, max_train_end, num=n_folds, dtype=int)
        module_fold_points[module_id] = sorted(set(int(x) for x in points))
        module_val_len[module_id] = val_len

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_folds):
        train_indices: List[int] = []
        val_indices: List[int] = []
        for module_id, g in modules:
            idx = g.index.to_numpy()
            points = module_fold_points[module_id]
            point = points[min(fold_idx, len(points) - 1)]
            val_len = module_val_len[module_id]
            val_start = max(1, min(point, len(idx) - 2))
            val_end = min(len(idx), val_start + val_len)
            if val_end <= val_start:
                continue
            train_indices.extend(idx[:val_start].tolist())
            val_indices.extend(idx[val_start:val_end].tolist())
        if train_indices and val_indices:
            folds.append((np.array(train_indices, dtype=int), np.array(val_indices, dtype=int)))
    return folds


def get_candidate_factories() -> List[Tuple[str, Callable[[], object]]]:
    return [
        ("ridge", lambda: Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=0.5))])),
        (
            "elastic_net",
            lambda: Pipeline(
                [("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.002, l1_ratio=0.08, max_iter=30000))]
            ),
        ),
        (
            "rf",
            lambda: RandomForestRegressor(
                n_estimators=1400,
                max_depth=24,
                min_samples_leaf=1,
                random_state=25,
                n_jobs=1,
            ),
        ),
        (
            "etr",
            lambda: ExtraTreesRegressor(
                n_estimators=2200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]


def run_cv(
    model_factory: Callable[[], object],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, float]:
    rmses: List[float] = []
    maes: List[float] = []
    for fold_id, (tr_idx, va_idx) in enumerate(folds, start=1):
        model = model_factory()
        model.fit(x_train.loc[tr_idx], y_train.loc[tr_idx])
        pred = model.predict(x_train.loc[va_idx])
        rmse = float(root_mean_squared_error(y_train.loc[va_idx], pred))
        mae = float(mean_absolute_error(y_train.loc[va_idx], pred))
        rmses.append(rmse)
        maes.append(mae)
        print(f"    fold={fold_id} rmse={rmse:.4f} mae={mae:.4f}")
    return float(np.mean(rmses)), float(np.mean(maes))


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
        if c.endswith(tuple([f"_lag{i}" for i in range(1, 10)])) or "_roll" in c
    ]

    train_df, test_df = build_holdout_split(df, holdout_ratio=0.2)
    x_train = train_df[feature_cols]
    y_train = train_df["soc_pct"]
    x_test = test_df[feature_cols]
    y_test = test_df["soc_pct"]

    cv_folds = build_time_cv_folds(train_df, n_folds=3)
    if not cv_folds:
        raise RuntimeError("Failed to build cross-validation folds from training data.")

    scored: List[Tuple[str, float, float, Callable[[], object]]] = []
    for name, factory in get_candidate_factories():
        print(f"\nTraining candidate: {name}")
        cv_rmse, cv_mae = run_cv(factory, x_train, y_train, cv_folds)
        print(f"  cv_result rmse={cv_rmse:.4f} mae={cv_mae:.4f}")
        scored.append((name, cv_rmse, cv_mae, factory))

    scored = sorted(scored, key=lambda x: x[1])
    top_k = min(3, len(scored))
    top = scored[:top_k]
    inv = np.array([1.0 / max(1e-6, x[1]) for x in top], dtype=float)
    weights = (inv / inv.sum()).tolist()

    members = []
    member_results: List[CandidateResult] = []
    for (name, cv_rmse, cv_mae, factory), weight in zip(top, weights):
        model = factory()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        holdout_rmse = float(root_mean_squared_error(y_test, pred))
        holdout_mae = float(mean_absolute_error(y_test, pred))
        members.append({"name": name, "weight": float(weight), "model": model})
        member_results.append(
            CandidateResult(
                name=name,
                cv_rmse=cv_rmse,
                cv_mae=cv_mae,
                model=model,
                holdout_rmse=holdout_rmse,
                holdout_mae=holdout_mae,
            )
        )
        print(
            f"member={name:12s} weight={weight:.4f} holdout_rmse={holdout_rmse:.4f} holdout_mae={holdout_mae:.4f}"
        )

    blend_pred = np.zeros(len(x_test), dtype=float)
    for info in members:
        blend_pred += float(info["weight"]) * info["model"].predict(x_test)

    blend_rmse = float(root_mean_squared_error(y_test, blend_pred))
    blend_mae = float(mean_absolute_error(y_test, blend_pred))

    artifact = {
        "format_version": "soc_model_v3",
        "model_family": "weighted_ensemble",
        "members": members,
        "feature_cols": feature_cols,
        "base_features": BASE_FEATURES,
        "lag_config": LAG_CONFIG,
        "roll_config": ROLL_CONFIG,
        "max_lag": max(max(v) for v in LAG_CONFIG.values()),
        "holdout_metrics": {"rmse_pct": blend_rmse, "mae_pct": blend_mae},
    }
    model_path = out_dir / "soc_model_v2.joblib"
    metrics_path = out_dir / "soc_model_v2_metrics.json"
    joblib.dump(artifact, model_path)

    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_cols)),
        "ensemble_members": [
            {
                "name": r.name,
                "weight": float(next(x["weight"] for x in members if x["name"] == r.name)),
                "cv_rmse_pct": r.cv_rmse,
                "cv_mae_pct": r.cv_mae,
                "holdout_rmse_pct": r.holdout_rmse,
                "holdout_mae_pct": r.holdout_mae,
            }
            for r in member_results
        ],
        "blend_rmse_pct": blend_rmse,
        "blend_mae_pct": blend_mae,
        "target_rmse_pct": 1.0,
        "target_met": bool(blend_rmse <= 1.0),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nEnsemble holdout RMSE: {blend_rmse:.4f}")
    print(f"Ensemble holdout MAE:  {blend_mae:.4f}")
    print(f"Saved: {model_path}")
    print(f"Saved: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
