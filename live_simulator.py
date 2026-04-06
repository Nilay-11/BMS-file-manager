from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass
from typing import Dict

import pandas as pd

from simulated_bms import BMSController, create_simulated_pack
from soc_ai import RealWorldSocModel


@dataclass
class RuntimeConfig:
    dt_s: float = 1.0
    series_cells: int = 96
    parallel_groups: int = 40
    telemetry_dataset_path: str = "distributed_bms_dataset.csv"
    soc_model_path: str = "models/soc_model_v2.joblib"


class RealWorldDriveProfile:
    def __init__(self, dataset_path: str, rng: random.Random) -> None:
        df = pd.read_csv(dataset_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
        self.df = df.sort_values(["module_id", "timestamp"]).reset_index(drop=True)
        self.rng = rng
        self.index = 0
        self.mean_soc = float(self.df["soc_pct"].mean())
        self.mean_soh = float(self.df["soh_pct"].mean())

    def next_row(self) -> Dict[str, float]:
        row = self.df.iloc[self.index]
        self.index = (self.index + 1) % len(self.df)
        return {
            "temperature_c": float(row["Temperature"]),
            "current_a": float(row["Current"]),
            "power_kw": float(row["Power"]),
            "converter_command_pct": float(row["converter_command_pct"]),
            "latency_ms": float(row["latency_ms"]),
            "anomaly_score_pct": float(row["anomaly_score_pct"]),
        }


class LiveBMSSimulator:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.rng = random.Random(25)
        self.profile = RealWorldDriveProfile(config.telemetry_dataset_path, self.rng)
        self.pack = create_simulated_pack(config.series_cells, config.parallel_groups, self.rng)
        for cell in self.pack.cells:
            cell.soc = max(0.05, min(0.95, self.profile.mean_soc / 100.0 + self.rng.uniform(-0.015, 0.015)))
            cell.soh = max(0.75, min(1.0, self.profile.mean_soh / 100.0 + self.rng.uniform(-0.01, 0.01)))

        self.bms = BMSController()
        self.ai = RealWorldSocModel.load(config.soc_model_path)
        self.lock = threading.Lock()
        self.tick = 0
        self.prev_current_a = 0.0
        self.low_power_mode = False
        self.drive_mode = "auto"
        self.latest_state: Dict[str, object] = {}

    def _requested_current(self, profile_current_a: float) -> float:
        base = abs(profile_current_a)
        if self.drive_mode == "charge":
            return -(base * 3.2 + self.rng.uniform(3.0, 8.0))
        if self.drive_mode == "discharge":
            return base * 4.2 + self.rng.uniform(4.0, 9.0)

        phase = self.tick % 240
        if 170 <= phase < 205:
            return -(base * 1.6 + self.rng.uniform(2.0, 6.0))
        return base * 3.8 + self.rng.uniform(-4.0, 7.0)

    def set_low_power_mode(self, enabled: bool) -> None:
        with self.lock:
            self.low_power_mode = enabled
            self.bms.low_power_mode = enabled

    def set_drive_mode(self, mode: str) -> None:
        if mode not in {"auto", "charge", "discharge"}:
            return
        with self.lock:
            self.drive_mode = mode

    def trigger_scenario(self, scenario: str) -> None:
        with self.lock:
            if scenario == "temp_spike":
                for cell in self.pack.cells:
                    cell.temperature_c += 30.0
            elif scenario == "low_voltage":
                for cell in self.pack.cells:
                    cell.temperature_c = min(cell.temperature_c, 31.0)
                for idx, cell in enumerate(self.pack.cells):
                    if idx % 8 == 0:
                        cell.soc = 0.01
                        cell.internal_resistance_ohm *= 5.0
            elif scenario == "high_voltage":
                for cell in self.pack.cells:
                    cell.temperature_c = min(cell.temperature_c, 31.0)
                for idx, cell in enumerate(self.pack.cells):
                    if idx % 8 == 0:
                        cell.soc = 0.999

    def step(self) -> Dict[str, object]:
        with self.lock:
            t_s = self.tick * self.config.dt_s
            profile = self.profile.next_row()
            ambient_c = profile["temperature_c"] + 0.3 * math.sin(t_s / 240.0)
            requested_current_a = self._requested_current(profile["current_a"])

            avg_cell_v = sum(cell.terminal_voltage(self.prev_current_a) for cell in self.pack.cells) / len(
                self.pack.cells
            )
            max_temp_c = max(self.pack.max_temperature_c(), profile["temperature_c"])
            avg_soh_pct = (sum(c.soh for c in self.pack.cells) / len(self.pack.cells)) * 100.0
            cell_current_a = self.prev_current_a / max(1, self.config.parallel_groups)
            cell_power_kw = max(0.0, (abs(cell_current_a) * avg_cell_v) / 1000.0)

            ai_soc_pct = self.ai.predict_soc_pct(
                voltage_v=avg_cell_v,
                current_a=cell_current_a,
                temp_c=max_temp_c,
                power_kw=cell_power_kw,
                soh_pct=avg_soh_pct,
                dt_s=self.config.dt_s,
                latency_s=max(0.001, profile["latency_ms"] / 1000.0),
                converter_command_pct=profile["converter_command_pct"],
                anomaly_score_pct=profile["anomaly_score_pct"],
            )

            command = self.bms.command(
                requested_current_a=requested_current_a,
                pack=self.pack,
                soc_estimate_pct=ai_soc_pct,
            )
            cell_voltages = self.pack.step(
                pack_current_a=command.current_a,
                ambient_c=ambient_c,
                dt_s=self.config.dt_s,
                balance_mask=command.balance_mask,
                cooling_on=command.cooling_on,
            )
            self.prev_current_a = command.current_a
            self.tick += 1

            pack_voltage_v = sum(cell_voltages)
            min_cell_v = min(cell_voltages)
            max_cell_v = max(cell_voltages)
            max_temp_c = self.pack.max_temperature_c()
            soc_true_pct = self.pack.mean_soc() * 100.0
            soc_ai_pct = ai_soc_pct if ai_soc_pct is not None else soc_true_pct
            power_kw = pack_voltage_v * command.current_a / 1000.0

            if command.current_a > 1.0:
                flow_state = "DISCHARGING"
            elif command.current_a < -1.0:
                flow_state = "CHARGING"
            else:
                flow_state = "IDLE"

            packet: Dict[str, object] = {
                "time_s": round(t_s, 1),
                "ambient_c": round(ambient_c, 2),
                "drive_mode": self.drive_mode,
                "low_power_mode": self.low_power_mode,
                "flow_state": flow_state,
                "request_current_a": round(requested_current_a, 2),
                "command_current_a": round(command.current_a, 2),
                "pack_voltage_v": round(pack_voltage_v, 2),
                "pack_power_kw": round(power_kw, 2),
                "min_cell_v": round(min_cell_v, 3),
                "max_cell_v": round(max_cell_v, 3),
                "max_temp_c": round(max_temp_c, 2),
                "soc_true_pct": round(soc_true_pct, 2),
                "soc_ai_pct": round(soc_ai_pct, 2),
                "soc_error_pct": round(soc_ai_pct - soc_true_pct, 2),
                "soh_pct": round(avg_soh_pct, 2),
                "balancing_cells": sum(1 for x in command.balance_mask if x),
                "cooling_on": command.cooling_on,
                "state": command.state,
                "events": command.events,
                "limiting_reasons": command.limiting_reasons,
                "windows": {
                    "voltage": [round(command.voltage_window_a[0], 2), round(command.voltage_window_a[1], 2)],
                    "soc": [round(command.soc_window_a[0], 2), round(command.soc_window_a[1], 2)],
                    "temp": [round(command.temp_window_a[0], 2), round(command.temp_window_a[1], 2)],
                    "allowed": [round(command.allowed_window_a[0], 2), round(command.allowed_window_a[1], 2)],
                },
                "limits_active": {
                    "current": ("CURRENT_LIMITED" in command.events) or command.state.startswith("FAULT_"),
                    "voltage": "VOLTAGE" in command.limiting_reasons,
                    "temperature": ("TEMPERATURE" in command.limiting_reasons)
                    or command.state.startswith("FAULT_OVER_TEMP"),
                    "soc": "SOC" in command.limiting_reasons,
                },
                "thresholds": {
                    "min_cell_v": self.bms.limits.min_cell_v,
                    "max_cell_v": self.bms.limits.max_cell_v,
                    "critical_temp_c": self.bms.limits.critical_temp_c,
                },
                "dataset_trace": {
                    "base_current_a": round(profile["current_a"], 2),
                    "converter_command_pct": round(profile["converter_command_pct"], 2),
                    "latency_ms": round(profile["latency_ms"], 2),
                },
            }

            self.latest_state = packet
            return packet

    def get_state(self) -> Dict[str, object]:
        with self.lock:
            if self.latest_state:
                return dict(self.latest_state)
        return self.step()
