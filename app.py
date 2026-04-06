from __future__ import annotations

import threading
import time
from typing import Dict, Optional

from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit

from live_simulator import LiveBMSSimulator, RuntimeConfig


config = RuntimeConfig()
simulator = LiveBMSSimulator(config)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


def simulation_worker() -> None:
    while True:
        packet = simulator.step()
        socketio.emit("telemetry", packet)
        time.sleep(config.dt_s)


@app.route("/")
def index() -> str:
    return render_template("dashboard.html")


@app.route("/api/health")
def health() -> object:
    state = simulator.get_state()
    return jsonify({"ok": True, "state": state.get("state", "UNKNOWN")})


@socketio.on("connect")
def on_connect() -> None:
    emit("telemetry", simulator.get_state())


@socketio.on("set_low_power")
def on_set_low_power(data: Optional[Dict[str, object]] = None) -> None:
    enabled = bool((data or {}).get("enabled", False))
    simulator.set_low_power_mode(enabled)


@socketio.on("set_drive_mode")
def on_set_drive_mode(data: Optional[Dict[str, object]] = None) -> None:
    mode = str((data or {}).get("mode", "auto"))
    simulator.set_drive_mode(mode)


@socketio.on("trigger_scenario")
def on_trigger_scenario(data: Optional[Dict[str, object]] = None) -> None:
    scenario = str((data or {}).get("scenario", ""))
    simulator.trigger_scenario(scenario)


if __name__ == "__main__":
    worker = threading.Thread(target=simulation_worker, daemon=True)
    worker.start()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
