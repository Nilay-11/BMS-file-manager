# EV Battery Manager

This repository includes a physics-based EV battery simulator and an AI-driven Battery Management System (BMS) controller for SOC-aware safety limiting.

## Simulated BMS (CLI)

Run the standalone simulator:

```bash
python simulated_bms.py --duration 1200 --dt 1 --log-interval 30 --series-cells 12 --parallel-groups 40 --ai-model bms_lstm_model.keras --ai-x-scaler x_scaler.pkl --ai-y-scaler y_scaler.pkl --csv bms_simulation_log.csv
```

What it does:
- Simulates battery cell voltage, SOC, temperature, SOH, and pack behavior.
- Uses an AI model to estimate SOC and feed BMS control logic.
- Applies safety limits for voltage, temperature, SOC, and current.
- Logs live telemetry and writes CSV output.

## Live Flask Dashboard

Run backend + web frontend:

```bash
python app.py
```

Open:

`http://127.0.0.1:5000`

Dashboard capabilities:
- Live EV battery telemetry (voltage, current, power, SOC, SOH, temperature).
- AI SOC vs true SOC visualization.
- Charging/discharging state.
- Safety-limiting reason display (`VOLTAGE`, `TEMPERATURE`, `SOC`).
- Scenario triggers for demonstration of BMS protections.

## Streamlit App

Run locally:

```bash
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Open [https://share.streamlit.io](https://share.streamlit.io).
3. Create a new app with:
   - Repository: `Nilay-11/BMS-file-manager`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. Deploy.
