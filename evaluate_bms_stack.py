from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

from live_simulator import LiveBMSSimulator, RuntimeConfig


@dataclass
class EvalResult:
    mode: str
    steps: int
    mean_soc_abs_error_pct: float
    p95_soc_abs_error_pct: float
    mean_current_tracking_error_a: float
    current_limited_steps: int
    fault_steps: int
    avg_pack_power_kw: float
    peak_temp_c: float
    mean_command_current_a: float


def run_case(steps: int, use_ai_current_control: bool, use_ai_soc_for_control: bool) -> EvalResult:
    cfg = RuntimeConfig(
        use_ai_for_control=use_ai_soc_for_control,
        use_ai_current_control=use_ai_current_control,
    )
    sim = LiveBMSSimulator(cfg)
    soc_abs_errors = []
    tracking_errors = []
    limited = 0
    faults = 0
    power_hist = []
    temp_hist = []
    current_hist = []

    for _ in range(steps):
        pkt = sim.step()
        soc_abs_errors.append(abs(float(pkt["soc_error_pct"])))
        tracking_errors.append(abs(float(pkt["command_current_a"]) - float(pkt["request_current_a"])))
        current_hist.append(float(pkt["command_current_a"]))
        power_hist.append(float(pkt["pack_power_kw"]))
        temp_hist.append(float(pkt["max_temp_c"]))
        if "CURRENT_LIMITED" in pkt["events"]:
            limited += 1
        if str(pkt["state"]).startswith("FAULT_"):
            faults += 1

    mode_label = "ai_current_control" if use_ai_current_control else "profile_request_only"
    return EvalResult(
        mode=mode_label,
        steps=steps,
        mean_soc_abs_error_pct=float(np.mean(soc_abs_errors)),
        p95_soc_abs_error_pct=float(np.percentile(soc_abs_errors, 95)),
        mean_current_tracking_error_a=float(np.mean(tracking_errors)),
        current_limited_steps=int(limited),
        fault_steps=int(faults),
        avg_pack_power_kw=float(np.mean(power_hist)),
        peak_temp_c=float(np.max(temp_hist)),
        mean_command_current_a=float(np.mean(current_hist)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BMS stack with and without AI current-control."
    )
    parser.add_argument("--steps", type=int, default=900, help="Number of runtime steps per case.")
    parser.add_argument(
        "--use-ai-soc-control",
        action="store_true",
        help="Enable SOC estimator as an active control input in both cases.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="models/bms_stack_eval.json",
        help="Path for saving evaluation JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = run_case(
        steps=args.steps,
        use_ai_current_control=False,
        use_ai_soc_for_control=args.use_ai_soc_control,
    )
    ai_case = run_case(
        steps=args.steps,
        use_ai_current_control=True,
        use_ai_soc_for_control=args.use_ai_soc_control,
    )

    summary: Dict[str, object] = {
        "steps": args.steps,
        "ai_soc_for_control": bool(args.use_ai_soc_control),
        "baseline": asdict(baseline),
        "ai_current_control": asdict(ai_case),
        "delta": {
            "mean_soc_abs_error_pct": ai_case.mean_soc_abs_error_pct - baseline.mean_soc_abs_error_pct,
            "mean_current_tracking_error_a": ai_case.mean_current_tracking_error_a
            - baseline.mean_current_tracking_error_a,
            "fault_steps": ai_case.fault_steps - baseline.fault_steps,
            "current_limited_steps": ai_case.current_limited_steps - baseline.current_limited_steps,
            "avg_pack_power_kw": ai_case.avg_pack_power_kw - baseline.avg_pack_power_kw,
            "peak_temp_c": ai_case.peak_temp_c - baseline.peak_temp_c,
        },
    }
    print(json.dumps(summary, indent=2))

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation report: {args.json_out}")


if __name__ == "__main__":
    main()
