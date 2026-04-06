# EV Battery Manager

This project combines:
- A physics-based EV battery simulator.
- An AI-driven BMS controller trained on telemetry data.
- Live Flask and Streamlit dashboards for SOC, charging/discharging state, and active safety limits.

## 1) Train The Improved SOC Model

```bash
python train_soc_model.py
```

This training pipeline:
- Uses `distributed_bms_dataset.csv` as the real-world telemetry reference dataset.
- Builds lag/rolling time-series features.
- Tries multiple model candidates until the RMSE target is met.
- Saves:
  - `models/soc_model_v2.joblib`
  - `models/soc_model_v2_metrics.json`

Current result in this repo:
- RMSE: `0.228%`
- MAE: `0.185%`

## 2) Run Live Flask Dashboard

```bash
python app.py
```

Open `http://127.0.0.1:5000`

What it shows:
- Simulated EV battery telemetry (voltage/current/power/temp/SOH/SOC).
- AI SOC estimate vs true simulated SOC.
- Charging / discharging / idle state.
- Safety limiting reasons (`VOLTAGE`, `TEMPERATURE`, `SOC`) and current windows.

## 3) Run Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

## 4) Dataset-Based Simulation

`live_simulator.py` uses `distributed_bms_dataset.csv` to drive realistic behavior:
- Ambient temperature trace.
- Current/load profile.
- Converter command and latency profile.

Battery state still evolves through the physics model (SOC/voltage/thermal), while BMS control decisions are AI-driven using the trained model artifact.

## 5) Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Open [https://share.streamlit.io](https://share.streamlit.io).
3. Create a new app:
   - Repository: `Nilay-11/BMS-file-manager`
   - Branch: `main`
   - Main file: `streamlit_app.py`
4. Deploy.
