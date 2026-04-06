from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

joblib = None
tf = None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class AISOCModel:
    model_path: str
    x_scaler_path: Optional[str]
    y_scaler_path: Optional[str]
    sequence_length: int = 20
    model: object = field(init=False)
    x_scaler: object = field(init=False, default=None)
    y_scaler: object = field(init=False, default=None)
    input_feature_count: int = field(init=False, default=3)
    history: List[np.ndarray] = field(default_factory=list, init=False)
    previous_soc_pct: Optional[float] = field(default=None, init=False)
    smoothing_alpha: float = 0.15

    def __post_init__(self) -> None:
        global tf, joblib
        if tf is None:
            try:
                import tensorflow as _tf

                tf = _tf
            except Exception as exc:
                raise RuntimeError(
                    "TensorFlow is unavailable, so AI BMS control cannot start. Install dependencies first."
                ) from exc

        if joblib is None:
            try:
                import joblib as _joblib

                joblib = _joblib
            except Exception:
                joblib = None

        if tf is None:
            raise RuntimeError(
                "TensorFlow is unavailable, so AI BMS control cannot start. Install dependencies first."
            )
        self.model = tf.keras.models.load_model(self.model_path)
        model_input_shape = self.model.input_shape
        if len(model_input_shape) != 3:
            raise RuntimeError(f"Unsupported AI model input shape: {model_input_shape}")
        self.input_feature_count = int(model_input_shape[-1])

        if self.x_scaler_path:
            if joblib is None:
                raise RuntimeError(
                    "joblib is unavailable, so scaler loading failed. Install dependencies first."
                )
            self.x_scaler = joblib.load(self.x_scaler_path)
        if self.y_scaler_path:
            if joblib is None:
                raise RuntimeError(
                    "joblib is unavailable, so scaler loading failed. Install dependencies first."
                )
            self.y_scaler = joblib.load(self.y_scaler_path)

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if self.x_scaler is not None:
            try:
                return self.x_scaler.transform(x)
            except Exception:
                # Some projects store scalers for wider feature sets.
                # If shape mismatch happens, use deterministic sensor normalization.
                self.x_scaler = None

        # Feature-wise fallback normalization.
        if x.shape[1] == 7:
            norm = np.array([5.0, 200.0, 100.0, 600.0, 100.0, 10.0, 1.0], dtype=float)
        elif x.shape[1] == 3:
            norm = np.array([5.0, 200.0, 100.0], dtype=float)
        else:
            norm = np.ones(x.shape[1], dtype=float)
        return x / norm

    def _build_features(
        self,
        avg_cell_voltage_v: float,
        current_a: float,
        temp_c: float,
        power_kw: float,
        soh_pct: float,
        dt_s: float,
        latency_s: float,
    ) -> np.ndarray:
        full = np.array(
            [avg_cell_voltage_v, current_a, temp_c, power_kw, soh_pct, dt_s, latency_s], dtype=float
        )
        if self.input_feature_count == 7:
            return full
        if self.input_feature_count == 3:
            return full[:3]
        if self.input_feature_count < 7:
            return full[: self.input_feature_count]
        extra = np.zeros(self.input_feature_count - 7, dtype=float)
        return np.concatenate([full, extra], axis=0)

    def predict_soc_pct(
        self,
        avg_cell_voltage_v: float,
        current_a: float,
        temp_c: float,
        power_kw: float,
        soh_pct: float,
        dt_s: float,
        latency_s: float,
    ) -> Optional[float]:
        feature_vector = self._build_features(
            avg_cell_voltage_v=avg_cell_voltage_v,
            current_a=current_a,
            temp_c=temp_c,
            power_kw=power_kw,
            soh_pct=soh_pct,
            dt_s=dt_s,
            latency_s=latency_s,
        )
        raw = np.array([feature_vector], dtype=float)
        scaled = self._scale(raw)[0]
        self.history.append(scaled)
        if len(self.history) > self.sequence_length:
            self.history.pop(0)
        if len(self.history) < self.sequence_length:
            return None

        ai_input = np.array(self.history, dtype=float).reshape(
            1, self.sequence_length, self.input_feature_count
        )
        pred = self.model.predict(ai_input, verbose=0)
        if self.y_scaler is not None:
            try:
                soc = float(self.y_scaler.inverse_transform(pred)[0][0])
            except Exception:
                self.y_scaler = None
                soc = float(pred[0][0])
        else:
            soc = float(pred[0][0])
        if soc <= 1.0:
            soc *= 100.0
        soc = clamp(soc, 0.0, 100.0)
        if self.previous_soc_pct is None:
            self.previous_soc_pct = soc
            return soc
        soc = (1.0 - self.smoothing_alpha) * self.previous_soc_pct + (
            self.smoothing_alpha * soc
        )
        self.previous_soc_pct = soc
        return clamp(soc, 0.0, 100.0)


@dataclass
class BatteryCell:
    capacity_ah: float
    internal_resistance_ohm: float
    soc: float
    temperature_c: float = 25.0
    soh: float = 1.0
    thermal_mass_j_per_c: float = 500.0
    passive_cooling_w_per_c: float = 0.75

    def ocv(self) -> float:
        soc = clamp(self.soc, 0.0, 1.0)
        # Tuned to match the provided dataset's voltage-SOC operating range.
        base = 3.68 + (0.52 * soc) + (0.06 * math.tanh((soc - 0.35) * 4.0))
        temp_effect = -0.0015 * (25.0 - self.temperature_c)
        return clamp(base + temp_effect, 2.8, 4.25)

    def terminal_voltage(self, current_a: float) -> float:
        return self.ocv() - (current_a * self.internal_resistance_ohm)

    def step(self, current_a: float, ambient_c: float, cooling_w: float, dt_s: float) -> float:
        effective_capacity = max(0.4, self.capacity_ah * self.soh)
        delta_ah = current_a * dt_s / 3600.0
        self.soc = clamp(self.soc - (delta_ah / effective_capacity), 0.0, 1.0)

        joule_heat_w = (current_a ** 2) * self.internal_resistance_ohm
        passive_cooling_w = self.passive_cooling_w_per_c * (self.temperature_c - ambient_c)
        net_power_w = joule_heat_w - passive_cooling_w - cooling_w
        self.temperature_c += (net_power_w * dt_s) / self.thermal_mass_j_per_c
        # Guard against non-physical runaway values after fault injections.
        self.temperature_c = clamp(self.temperature_c, ambient_c - 20.0, 90.0)

        throughput_ah = abs(delta_ah)
        self.soh = max(0.7, self.soh - throughput_ah * 2.0e-6)
        return self.terminal_voltage(current_a)


@dataclass
class BatteryPack:
    cells: List[BatteryCell]
    balancing_shunt_a: float = 1.0
    cooling_per_cell_w: float = 8.0

    def mean_soc(self) -> float:
        return sum(cell.soc for cell in self.cells) / len(self.cells)

    def soc_spread(self) -> float:
        socs = [cell.soc for cell in self.cells]
        return max(socs) - min(socs)

    def min_cell_voltage(self, current_a: float = 0.0) -> float:
        return min(cell.terminal_voltage(current_a) for cell in self.cells)

    def max_cell_voltage(self, current_a: float = 0.0) -> float:
        return max(cell.terminal_voltage(current_a) for cell in self.cells)

    def max_temperature_c(self) -> float:
        return max(cell.temperature_c for cell in self.cells)

    def step(
        self,
        pack_current_a: float,
        ambient_c: float,
        dt_s: float,
        balance_mask: List[bool],
        cooling_on: bool,
    ) -> List[float]:
        voltages = []
        for idx, cell in enumerate(self.cells):
            balance_current = 0.0
            if balance_mask[idx]:
                if pack_current_a < 0.0:
                    balance_current = self.balancing_shunt_a
                elif pack_current_a > 0.0:
                    balance_current = self.balancing_shunt_a * 0.6

            cell_current = pack_current_a + balance_current
            cooling_w = self.cooling_per_cell_w if cooling_on else 0.0
            voltages.append(cell.step(cell_current, ambient_c, cooling_w, dt_s))
        return voltages


@dataclass
class BMSLimits:
    min_cell_v: float = 3.0
    max_cell_v: float = 4.2
    max_discharge_a: float = 120.0
    max_charge_a: float = 90.0
    critical_temp_c: float = 58.0
    restart_temp_c: float = 50.0
    cooling_on_temp_c: float = 36.0
    soc_charge_taper_start: float = 0.9
    soc_discharge_taper_end: float = 0.12
    balance_start_spread: float = 0.025


@dataclass
class BMSCommand:
    current_a: float
    cooling_on: bool
    balance_mask: List[bool]
    state: str
    events: List[str] = field(default_factory=list)
    limiting_reasons: List[str] = field(default_factory=list)
    voltage_window_a: tuple[float, float] = (0.0, 0.0)
    soc_window_a: tuple[float, float] = (0.0, 0.0)
    temp_window_a: tuple[float, float] = (0.0, 0.0)
    allowed_window_a: tuple[float, float] = (0.0, 0.0)


@dataclass
class BMSController:
    limits: BMSLimits = field(default_factory=BMSLimits)
    contactor_closed: bool = True
    fault_reason: str = ""
    low_power_mode: bool = False

    def _voltage_current_window(self, pack: BatteryPack) -> tuple[float, float]:
        min_current = -self.limits.max_charge_a
        max_current = self.limits.max_discharge_a
        for cell in pack.cells:
            r = max(cell.internal_resistance_ohm, 1e-6)
            ocv = cell.ocv()
            max_discharge_by_v = (ocv - self.limits.min_cell_v) / r
            min_charge_by_v = (ocv - self.limits.max_cell_v) / r
            max_current = min(max_current, max_discharge_by_v)
            min_current = max(min_current, min_charge_by_v)
        return min_current, max_current

    def _soc_current_window(self, pack: BatteryPack, soc_estimate_pct: Optional[float]) -> tuple[float, float]:
        min_current = -self.limits.max_charge_a
        max_current = self.limits.max_discharge_a
        if soc_estimate_pct is None:
            mean_soc = pack.mean_soc()
        else:
            mean_soc = clamp(soc_estimate_pct / 100.0, 0.0, 1.0)

        if mean_soc >= self.limits.soc_charge_taper_start:
            headroom = max(0.01, 1.0 - self.limits.soc_charge_taper_start)
            charge_scale = clamp((1.0 - mean_soc) / headroom, 0.05, 1.0)
            min_current = -self.limits.max_charge_a * charge_scale

        if mean_soc <= self.limits.soc_discharge_taper_end:
            discharge_scale = clamp(
                mean_soc / max(0.01, self.limits.soc_discharge_taper_end), 0.05, 1.0
            )
            max_current = self.limits.max_discharge_a * discharge_scale

        return min_current, max_current

    def _temperature_current_window(self, pack: BatteryPack) -> tuple[float, float]:
        min_current = -self.limits.max_charge_a
        max_current = self.limits.max_discharge_a
        max_temp = pack.max_temperature_c()

        if max_temp >= self.limits.critical_temp_c:
            self.contactor_closed = False
            self.fault_reason = "OVER_TEMP"
            return 0.0, 0.0

        if not self.contactor_closed:
            if max_temp <= self.limits.restart_temp_c:
                self.contactor_closed = True
                self.fault_reason = ""
            else:
                return 0.0, 0.0

        if max_temp > self.limits.cooling_on_temp_c:
            scale = clamp(
                1.0
                - (
                    (max_temp - self.limits.cooling_on_temp_c)
                    / max(1.0, self.limits.critical_temp_c - self.limits.cooling_on_temp_c)
                ),
                0.2,
                1.0,
            )
            min_current *= scale
            max_current *= scale

        return min_current, max_current

    def _balance_mask(self, pack: BatteryPack, pack_current_a: float) -> List[bool]:
        if pack_current_a > 0.0:
            return [False] * len(pack.cells)
        spread = pack.soc_spread()
        if spread < self.limits.balance_start_spread:
            return [False] * len(pack.cells)
        avg_soc = pack.mean_soc()
        return [cell.soc > (avg_soc + 0.005) for cell in pack.cells]

    def command(
        self,
        requested_current_a: float,
        pack: BatteryPack,
        soc_estimate_pct: Optional[float] = None,
    ) -> BMSCommand:
        events: List[str] = []
        state = "NORMAL"

        voltage_min, voltage_max = self._voltage_current_window(pack)
        soc_min, soc_max = self._soc_current_window(pack, soc_estimate_pct)
        temp_min, temp_max = self._temperature_current_window(pack)

        allowed_min = max(voltage_min, soc_min, temp_min)
        allowed_max = min(voltage_max, soc_max, temp_max)

        if self.low_power_mode:
            allowed_min *= 0.6
            allowed_max *= 0.6
            events.append("LOW_POWER")

        if not self.contactor_closed or allowed_min > allowed_max:
            commanded_current = 0.0
            state = f"FAULT_{self.fault_reason}" if self.fault_reason else "PROTECTIVE_STOP"
            events.append("CONTACTOR_OPEN")
            limiting_reasons = ["CONTACTOR"]
        else:
            commanded_current = clamp(requested_current_a, allowed_min, allowed_max)
            limiting_reasons = []
            if abs(commanded_current - requested_current_a) > 1e-6:
                events.append("CURRENT_LIMITED")
                tol = 1e-4
                if requested_current_a > allowed_max:
                    if abs(allowed_max - voltage_max) < tol:
                        limiting_reasons.append("VOLTAGE")
                    if abs(allowed_max - soc_max) < tol:
                        limiting_reasons.append("SOC")
                    if abs(allowed_max - temp_max) < tol:
                        limiting_reasons.append("TEMPERATURE")
                elif requested_current_a < allowed_min:
                    if abs(allowed_min - voltage_min) < tol:
                        limiting_reasons.append("VOLTAGE")
                    if abs(allowed_min - soc_min) < tol:
                        limiting_reasons.append("SOC")
                    if abs(allowed_min - temp_min) < tol:
                        limiting_reasons.append("TEMPERATURE")
        if soc_estimate_pct is not None:
            events.append("AI_SOC_CONTROL")

        cooling_on = (
            pack.max_temperature_c() >= self.limits.cooling_on_temp_c
            or abs(commanded_current) >= self.limits.max_discharge_a * 0.75
        )
        balance_mask = self._balance_mask(pack, commanded_current)
        if any(balance_mask):
            events.append("BALANCING_ACTIVE")

        return BMSCommand(
            current_a=commanded_current,
            cooling_on=cooling_on,
            balance_mask=balance_mask,
            state=state,
            events=events,
            limiting_reasons=limiting_reasons,
            voltage_window_a=(voltage_min, voltage_max),
            soc_window_a=(soc_min, soc_max),
            temp_window_a=(temp_min, temp_max),
            allowed_window_a=(allowed_min, allowed_max),
        )


def create_simulated_pack(series_cells: int, parallel_groups: int, rng: random.Random) -> BatteryPack:
    cells: List[BatteryCell] = []
    base_capacity = 3.2
    base_resistance = 0.0045
    for _ in range(series_cells):
        cells.append(
            BatteryCell(
                capacity_ah=base_capacity * parallel_groups * (1.0 + rng.uniform(-0.05, 0.05)),
                internal_resistance_ohm=(
                    (base_resistance / parallel_groups) * (1.0 + rng.uniform(-0.12, 0.12))
                ),
                soc=0.65 + rng.uniform(-0.03, 0.03),
                temperature_c=27.0 + rng.uniform(-1.5, 1.5),
            )
        )
    return BatteryPack(cells=cells)


def drive_cycle_current(t_s: float, rng: random.Random) -> float:
    cycle_t = t_s % 900.0
    if cycle_t < 180.0:
        base = 55.0 + 10.0 * math.sin(t_s / 25.0)
    elif cycle_t < 360.0:
        base = 85.0 + 8.0 * math.sin(t_s / 14.0)
    elif cycle_t < 480.0:
        base = 35.0 + 5.0 * math.sin(t_s / 12.0)
    elif cycle_t < 620.0:
        base = -45.0 + 5.0 * math.sin(t_s / 15.0)
    elif cycle_t < 760.0:
        base = 25.0 + 6.0 * math.sin(t_s / 20.0)
    else:
        base = -20.0 + 4.0 * math.sin(t_s / 10.0)
    return base + rng.uniform(-3.0, 3.0)


def run_simulation(
    duration_s: int,
    dt_s: float,
    log_interval_s: int,
    series_cells: int,
    parallel_groups: int,
    csv_path: str,
    seed: int,
    ai_model_path: str,
    ai_x_scaler_path: Optional[str],
    ai_y_scaler_path: Optional[str],
    ai_sequence_length: int,
) -> None:
    rng = random.Random(seed)
    pack = create_simulated_pack(series_cells=series_cells, parallel_groups=parallel_groups, rng=rng)
    bms = BMSController()
    ai_soc_model = AISOCModel(
        model_path=ai_model_path,
        x_scaler_path=ai_x_scaler_path,
        y_scaler_path=ai_y_scaler_path,
        sequence_length=ai_sequence_length,
    )

    rows = []
    interval_steps = max(1, int(log_interval_s / dt_s))
    total_steps = int(duration_s / dt_s)
    energy_out_wh = 0.0
    peak_temp_c = pack.max_temperature_c()
    start_soc_pct = pack.mean_soc() * 100.0
    previous_current = 0.0

    print(
        f"Pack config: {series_cells}s{parallel_groups}p | "
        f"Nominal capacity ~{pack.cells[0].capacity_ah:.1f} Ah group"
    )
    print(
        f"AI model: {ai_model_path}"
        + (f" | x_scaler: {ai_x_scaler_path}" if ai_x_scaler_path else " | x_scaler: none")
        + (f" | y_scaler: {ai_y_scaler_path}" if ai_y_scaler_path else " | y_scaler: none")
    )
    print(
        "t(s)  req(A)  cmd(A)  pack(V)  minV  maxV  soc_true(%)  soc_ai(%)  spread(%)  temp(C)  state  events"
    )
    print("-" * 122)

    for step in range(total_steps + 1):
        t_s = step * dt_s
        ambient_c = 29.0 + 3.0 * math.sin(t_s / 1200.0)

        sensed_avg_cell_v = sum(cell.terminal_voltage(previous_current) for cell in pack.cells) / len(
            pack.cells
        )
        sensed_temp_c = pack.max_temperature_c()
        sensed_cell_current_a = previous_current / max(1, parallel_groups)
        sensed_cell_power_w = sensed_avg_cell_v * sensed_cell_current_a
        ai_soc_pct = ai_soc_model.predict_soc_pct(
            avg_cell_voltage_v=sensed_avg_cell_v,
            current_a=sensed_cell_current_a,
            temp_c=sensed_temp_c,
            power_kw=sensed_cell_power_w,
            soh_pct=(sum(c.soh for c in pack.cells) / len(pack.cells)) * 100.0,
            dt_s=max(2.5, dt_s),
            latency_s=0.005,
        )

        requested_current = drive_cycle_current(t_s, rng)
        command = bms.command(requested_current, pack, soc_estimate_pct=ai_soc_pct)
        cell_voltages = pack.step(
            pack_current_a=command.current_a,
            ambient_c=ambient_c,
            dt_s=dt_s,
            balance_mask=command.balance_mask,
            cooling_on=command.cooling_on,
        )
        previous_current = command.current_a

        pack_voltage = sum(cell_voltages)
        min_cell_v = min(cell_voltages)
        max_cell_v = max(cell_voltages)
        mean_soc_true_pct = pack.mean_soc() * 100.0
        mean_soc_ai_pct = ai_soc_pct if ai_soc_pct is not None else mean_soc_true_pct
        spread_pct = pack.soc_spread() * 100.0
        max_temp_c = pack.max_temperature_c()
        peak_temp_c = max(peak_temp_c, max_temp_c)
        energy_out_wh += pack_voltage * command.current_a * dt_s / 3600.0
        soc_err_pct = mean_soc_ai_pct - mean_soc_true_pct

        event_text = "|".join(command.events) if command.events else "-"
        row = {
            "time_s": round(t_s, 2),
            "ambient_c": round(ambient_c, 2),
            "request_current_a": round(requested_current, 2),
            "command_current_a": round(command.current_a, 2),
            "pack_voltage_v": round(pack_voltage, 3),
            "min_cell_v": round(min_cell_v, 3),
            "max_cell_v": round(max_cell_v, 3),
            "soc_true_pct": round(mean_soc_true_pct, 2),
            "soc_ai_pct": round(mean_soc_ai_pct, 2),
            "soc_error_pct": round(soc_err_pct, 2),
            "soc_spread_pct": round(spread_pct, 3),
            "max_temp_c": round(max_temp_c, 2),
            "cooling_on": int(command.cooling_on),
            "balancing_cells": sum(1 for x in command.balance_mask if x),
            "state": command.state,
            "events": event_text,
        }
        rows.append(row)

        if step % interval_steps == 0:
            print(
                f"{int(t_s):4d}  "
                f"{requested_current:6.1f}  "
                f"{command.current_a:6.1f}  "
                f"{pack_voltage:7.2f}  "
                f"{min_cell_v:4.2f}  "
                f"{max_cell_v:4.2f}  "
                f"{mean_soc_true_pct:10.2f}  "
                f"{mean_soc_ai_pct:8.2f}  "
                f"{spread_pct:8.3f}  "
                f"{max_temp_c:7.2f}  "
                f"{command.state:12s}  "
                f"{event_text}"
            )

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    end_soc_pct = pack.mean_soc() * 100.0
    print("\nSimulation complete")
    print(f"Start SOC: {start_soc_pct:.2f}%")
    print(f"End SOC:   {end_soc_pct:.2f}%")
    print(f"Peak Temp: {peak_temp_c:.2f} C")
    print(f"Net energy out: {energy_out_wh:.2f} Wh (positive means discharge)")
    print(f"CSV log saved: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulated Battery + BMS control loop with balancing and safety limits."
    )
    parser.add_argument("--duration", type=int, default=1200, help="Simulation time in seconds.")
    parser.add_argument("--dt", type=float, default=1.0, help="Simulation step in seconds.")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=30,
        help="Console telemetry interval in seconds.",
    )
    parser.add_argument(
        "--series-cells", type=int, default=12, help="Number of series cells in the simulated pack."
    )
    parser.add_argument(
        "--parallel-groups",
        type=int,
        default=40,
        help="Parallel cell groups per series stage (scales pack capacity).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="bms_simulation_log.csv",
        help="CSV output file path.",
    )
    parser.add_argument(
        "--ai-model",
        type=str,
        default="bms_lstm_model.keras",
        help="Path to the AI model used by BMS SOC estimation.",
    )
    parser.add_argument(
        "--ai-x-scaler",
        type=str,
        default="x_scaler.pkl",
        help="Path to input scaler used by the AI model.",
    )
    parser.add_argument(
        "--ai-y-scaler",
        type=str,
        default="y_scaler.pkl",
        help="Path to output scaler used by the AI model.",
    )
    parser.add_argument(
        "--ai-seq-len",
        type=int,
        default=20,
        help="Sequence length expected by the AI model.",
    )
    parser.add_argument("--seed", type=int, default=25, help="Random seed for repeatability.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        duration_s=args.duration,
        dt_s=args.dt,
        log_interval_s=args.log_interval,
        series_cells=args.series_cells,
        parallel_groups=args.parallel_groups,
        csv_path=args.csv,
        seed=args.seed,
        ai_model_path=args.ai_model,
        ai_x_scaler_path=args.ai_x_scaler,
        ai_y_scaler_path=args.ai_y_scaler,
        ai_sequence_length=args.ai_seq_len,
    )
