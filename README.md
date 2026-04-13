# EV Battery Manager

This project combines:
- A physics-based EV battery simulator.
- A high-accuracy SOC predictor trained on real telemetry.
- An AI current-controller that actively proposes BMS current requests.
- Live Flask and Streamlit dashboards for SOC, charging/discharging state, and active safety limits.

## 1) Train The Improved SOC Model

```bash
python train_soc_model.py
```

This training pipeline:
- Uses `distributed_bms_dataset.csv` as the real-world telemetry reference dataset.
- Builds lag/rolling time-series features.
- Runs time-aware cross-validation across multiple model families.
- Builds a weighted ensemble artifact.
- Saves:
  - `models/soc_model_v2.joblib`
  - `models/soc_model_v2_metrics.json`

## 2) Train The AI Current Controller

```bash
python train_bms_controller_model.py
```

This trains `models/current_controller_v1.joblib` from expert BMS behavior generated on physics simulation rollouts.

## 3) Evaluate The Full BMS Stack

```bash
python evaluate_bms_stack.py --steps 900
```

This runs two simulations:
- Baseline (profile request only).
- AI current-control enabled.

Report is saved to `models/bms_stack_eval.json`.

## 4) Run Live Flask Dashboard

```bash
python app.py
```

Open `http://127.0.0.1:5000`

What it shows:
- Simulated EV battery telemetry (voltage/current/power/temp/SOH/SOC).
- AI SOC estimate vs true simulated SOC.
- AI current request vs commanded current.
- Charging / discharging / idle state.
- Safety limiting reasons (`VOLTAGE`, `TEMPERATURE`, `SOC`) and current windows.

## 5) Run Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

## 6) Dataset-Based Simulation

`live_simulator.py` uses `distributed_bms_dataset.csv` to drive realistic behavior:
- Ambient temperature trace.
- Current/load profile.
- Converter command and latency profile.

Battery state evolves through the physics model (SOC/voltage/thermal), while AI is used in two places:
- SOC estimation (`models/soc_model_v2.joblib`).
- Current request policy (`models/current_controller_v1.joblib`).

Default runtime mode enables AI current-control if the controller artifact exists.
If you want AI SOC to directly influence BMS SOC-window control, set `RuntimeConfig(use_ai_for_control=True)`.

## 7) Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Open [https://share.streamlit.io](https://share.streamlit.io).
3. Create a new app:
   - Repository: `Nilay-11/BMS-file-manager`
   - Branch: `main`
   - Main file: `streamlit_app.py`
4. Deploy.
