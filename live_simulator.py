from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass
from typing import Dict

from simulated_bms import AISOCModel, BMSController, create_simulated_pack, drive_cycle_current


@dataclass
class RuntimeConfig:
    dt_s: float = 1.0
    series_cells: int = 96
    parallel_groups: int = 40
    ai_model_path: str = "bms_lstm_model.keras"
    ai_x_scaler_path: str = "x_scaler.pkl"
    ai_y_scaler_path: str = "y_scaler.pkl"
    ai_sequence_length: int = 20


class LiveBMSSimulator:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.rng = random.Random(25)
        self.pack = create_simulated_pack(config.series_cells, config.parallel_groups, self.rng)
        self.bms = BMSController()
        self.ai = AISOCModel(
            model_path=config.ai_model_path,
            x_scaler_path=config.ai_x_scaler_path,
            y_scaler_path=config.ai_y_scaler_path,
            sequence_length=config.ai_sequence_length,
        )
        self.lock = threading.Lock()
        self.tick = 0
        self.prev_current_a = 0.0
        self.low_power_mode = False
        self.drive_mode = "auto"
        self.latest_state: Dict[str, object] = {}

    def _requested_current(self, t_s: float) -> float:
        if self.drive_mode == "charge":
            return -75.0 + self.rng.uniform(-4.0, 4.0)
        if self.drive_mode == "discharge":
            return 95.0 + self.rng.uniform(-5.0, 5.0)
        return drive_cycle_current(t_s, self.rng)

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
                    cell.temperature_c += 15.0
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
            ambient_c = 29.0 + 3.0 * math.sin(t_s / 900.0)
            requested_current_a = self._requested_current(t_s)

            avg_cell_v = sum(cell.terminal_voltage(self.prev_current_a) for cell in self.pack.cells) / len(
                self.pack.cells
            )
            max_temp_c = self.pack.max_temperature_c()
            avg_soh_pct = (sum(c.soh for c in self.pack.cells) / len(self.pack.cells)) * 100.0

            ai_soc_pct = self.ai.predict_soc_pct(
                avg_cell_voltage_v=avg_cell_v,
                current_a=self.prev_current_a,
                temp_c=max_temp_c,
                power_kw=(abs(self.prev_current_a) * avg_cell_v * self.config.series_cells) / 1000.0,
                soh_pct=avg_soh_pct,
                dt_s=self.config.dt_s,
                latency_s=0.10 + self.rng.uniform(0.0, 0.06),
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
                    "current": "CURRENT_LIMITED" in command.events,
                    "voltage": "VOLTAGE" in command.limiting_reasons,
                    "temperature": "TEMPERATURE" in command.limiting_reasons,
                    "soc": "SOC" in command.limiting_reasons,
                },
                "thresholds": {
                    "min_cell_v": self.bms.limits.min_cell_v,
                    "max_cell_v": self.bms.limits.max_cell_v,
                    "critical_temp_c": self.bms.limits.critical_temp_c,
                },
            }

            self.latest_state = packet
            return packet

    def get_state(self) -> Dict[str, object]:
        with self.lock:
            if not self.latest_state:
                return self.step()
            return dict(self.latest_state)
