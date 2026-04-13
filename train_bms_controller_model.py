from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from simulated_bms import BMSController, create_simulated_pack


FEATURE_COLS = [
    "base_request_current_a",
    "prev_current_a",
    "avg_cell_v",
    "min_cell_v",
    "max_cell_v",
    "max_temp_c",
    "soc_ai_pct",
    "soh_pct",
    "anomaly_score_pct",
    "converter_command_pct",
    "latency_ms",
    "low_power_mode",
    "drive_mode_code",
]
TARGET_COL = "target_current_a"


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def random_drive_mode(rng: random.Random) -> str:
    p = rng.random()
    if p < 0.24:
        return "charge"
    if p < 0.78:
        return "discharge"
    return "auto"


def generate_expert_dataset(
    episodes: int,
    steps_per_episode: int,
    dt_s: float,
    series_cells: int,
    parallel_groups: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: List[Dict[str, float]] = []

    for episode in range(episodes):
        pack = create_simulated_pack(series_cells, parallel_groups, rng)
        bms = BMSController()
        prev_current = 0.0
        drive_mode = "auto"
        low_power_mode = False

        for step in range(steps_per_episode):
            t_s = step * dt_s
            if step % 75 == 0:
                drive_mode = random_drive_mode(rng)
            if rng.random() < 0.015:
                low_power_mode = not low_power_mode
            bms.low_power_mode = low_power_mode

            ambient_c = 28.0 + 4.0 * math.sin((episode * steps_per_episode + step) / 700.0)
            avg_cell_v = sum(c.terminal_voltage(prev_current) for c in pack.cells) / len(pack.cells)
            min_cell_v = min(c.terminal_voltage(prev_current) for c in pack.cells)
            max_cell_v = max(c.terminal_voltage(prev_current) for c in pack.cells)
            max_temp_c = pack.max_temperature_c()
            true_soc_pct = pack.mean_soc() * 100.0
            soc_noise = rng.gauss(0.0, 1.1)
            soc_ai_pct = clip(true_soc_pct + soc_noise, 0.0, 100.0)
            soh_pct = (sum(c.soh for c in pack.cells) / len(pack.cells)) * 100.0
            converter_command_pct = clip(50.0 + rng.gauss(0.0, 20.0), 0.0, 100.0)
            anomaly_score_pct = clip(abs(rng.gauss(3.0, 2.5)), 0.0, 100.0)
            latency_ms = clip(80.0 + abs(rng.gauss(0.0, 40.0)), 5.0, 500.0)

            profile_mag = abs(24.0 + 40.0 * math.sin(t_s / 55.0) + rng.gauss(0.0, 10.0))
            if drive_mode == "charge":
                base_request = -profile_mag
            elif drive_mode == "discharge":
                base_request = profile_mag
            else:
                base_request = profile_mag if (step % 210) < 150 else -0.7 * profile_mag

            if rng.random() < 0.01:
                # Inject occasional stress points so the AI learns protective behavior.
                spike_idx = rng.randrange(len(pack.cells))
                pack.cells[spike_idx].temperature_c += rng.uniform(5.0, 11.0)
            if rng.random() < 0.008:
                weak_idx = rng.randrange(len(pack.cells))
                pack.cells[weak_idx].soc = max(0.04, pack.cells[weak_idx].soc - rng.uniform(0.03, 0.07))

            command = bms.command(base_request, pack, soc_estimate_pct=soc_ai_pct)
            target_current = command.current_a

            drive_mode_code = 0.0 if drive_mode == "auto" else (-1.0 if drive_mode == "charge" else 1.0)
            rows.append(
                {
                    "base_request_current_a": float(base_request),
                    "prev_current_a": float(prev_current),
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
                    "drive_mode_code": drive_mode_code,
                    TARGET_COL: float(target_current),
                }
            )

            pack.step(
                pack_current_a=target_current,
                ambient_c=ambient_c,
                dt_s=dt_s,
                balance_mask=command.balance_mask,
                cooling_on=command.cooling_on,
            )
            prev_current = target_current

    return pd.DataFrame(rows)


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    split_idx = max(1, min(len(df) - 1, split_idx))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def train_and_select(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[str, object, Dict[str, float]]:
    candidates = {
        "etr": ExtraTreesRegressor(
            n_estimators=1600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=1,
            random_state=42,
        ),
    }

    best_name = ""
    best_model = None
    best_metrics: Dict[str, float] = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rmse = float(root_mean_squared_error(y_test, pred))
        mae = float(mean_absolute_error(y_test, pred))
        print(f"candidate={name:8s} rmse={rmse:.4f}A mae={mae:.4f}A")
        if not best_metrics or rmse < best_metrics["rmse_a"]:
            best_name = name
            best_model = model
            best_metrics = {"rmse_a": rmse, "mae_a": mae}

    assert best_model is not None
    return best_name, best_model, best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AI current-controller model for the BMS from expert simulation."
    )
    parser.add_argument("--episodes", type=int, default=45, help="Number of simulation episodes.")
    parser.add_argument("--steps", type=int, default=420, help="Steps per episode.")
    parser.add_argument("--dt", type=float, default=1.0, help="Simulation step duration in seconds.")
    parser.add_argument("--series-cells", type=int, default=96)
    parser.add_argument("--parallel-groups", type=int, default=40)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--model-out", type=str, default="models/current_controller_v1.joblib")
    parser.add_argument("--metrics-out", type=str, default="models/current_controller_v1_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    df = generate_expert_dataset(
        episodes=args.episodes,
        steps_per_episode=args.steps,
        dt_s=args.dt,
        series_cells=args.series_cells,
        parallel_groups=args.parallel_groups,
        seed=args.seed,
    )
    train_df, test_df = split_dataset(df, train_ratio=0.8)
    x_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    x_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    model_name, model, metrics = train_and_select(x_train, y_train, x_test, y_test)
    pred = model.predict(x_test)

    deltas = np.abs(np.diff(train_df[TARGET_COL].to_numpy()))
    max_step_delta = float(np.percentile(deltas, 95)) if len(deltas) else 16.0

    out_model_path = Path(args.model_out)
    out_metrics_path = Path(args.metrics_out)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format_version": "current_controller_v1",
        "model_name": model_name,
        "model": model,
        "feature_cols": FEATURE_COLS,
        "min_current_a": float(df[TARGET_COL].min()),
        "max_current_a": float(df[TARGET_COL].max()),
        "max_step_delta_a": max(8.0, max_step_delta),
    }
    joblib.dump(artifact, out_model_path)

    metrics_json = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "feature_count": int(len(FEATURE_COLS)),
        "best_model": model_name,
        "rmse_a": float(metrics["rmse_a"]),
        "mae_a": float(metrics["mae_a"]),
        "mean_abs_tracking_error_a": float(np.mean(np.abs(pred - y_test.to_numpy()))),
        "max_step_delta_a": float(artifact["max_step_delta_a"]),
        "target_current_minmax_a": [float(artifact["min_current_a"]), float(artifact["max_current_a"])],
    }
    out_metrics_path.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    print(f"Saved: {out_model_path}")
    print(f"Saved: {out_metrics_path}")
    print(json.dumps(metrics_json, indent=2))


if __name__ == "__main__":
    main()
