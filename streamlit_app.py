from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from live_simulator import LiveBMSSimulator, RuntimeConfig


st.set_page_config(page_title="Battery Health Monitor", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #060709; color: #f0f4f8; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1700px; }
    .topline { border-top: 2px solid #0aa4ff; padding-top: 0.8rem; }
    .hdr-title { font-size: 2rem; font-weight: 800; letter-spacing: .06rem; margin: 0; }
    .hdr-sub { color: #8a93a1; font-size: .8rem; letter-spacing: .08rem; text-transform: uppercase; margin-top: .2rem; }
    .online-pill {
      border: 1px solid #1f3d2a; color: #00e160; border-radius: 999px; padding: .45rem .9rem;
      background: #0b130f; font-size: .8rem; text-align: center;
    }
    .panel {
      border: 1px solid #2b2f36; border-radius: 14px; background: linear-gradient(160deg, #0a0b0f, #0c1017);
      padding: .9rem;
    }
    .p-title { color: #8a93a1; font-size: .75rem; letter-spacing: .09rem; text-transform: uppercase; font-weight: 700; margin-bottom: .7rem; }
    .sensor-k { color: #b6c0cf; font-size: 1rem; }
    .sensor-v { font-size: 2rem; font-weight: 700; text-align: right; }
    .log-box { margin-top: .9rem; background: #12151c; border: 1px solid #1e232d; border-radius: 10px; padding: .55rem; }
    .log-line { color: #a0acbf; font-family: Consolas, monospace; font-size: .74rem; margin: 0; line-height: 1.3; }
    .ring-wrap { display: flex; justify-content: center; align-items: center; min-height: 350px; }
    .ring {
      width: 280px; height: 280px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      background: conic-gradient(#00e160 var(--deg), #2c3138 var(--deg));
      padding: 8px;
      box-shadow: 0 0 36px rgba(0,225,96,.17);
    }
    .ring-inner {
      width: 100%; height: 100%; border-radius: 50%; background: #0a0d12; border: 1px solid #2c3138;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .soc-main { font-size: 4rem; font-weight: 800; line-height: 1; }
    .soc-sub { color: #9da8b8; font-size: 1rem; letter-spacing: .08rem; }
    .range-box { text-align: center; margin-top: .7rem; }
    .range-v { font-size: 2.2rem; font-weight: 700; }
    .range-sub { color: #8e99aa; text-transform: uppercase; font-size: .9rem; letter-spacing: .07rem; }
    .mini-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .5rem; margin-top: .7rem; }
    .mini { border: 1px solid #222a34; border-radius: 10px; background: #0f1218; padding: .45rem .55rem; }
    .mini-k { color: #8d98aa; font-size: .62rem; text-transform: uppercase; letter-spacing: .07rem; }
    .mini-v { font-size: .95rem; font-weight: 600; }
    .ok { color: #00e160; } .warn { color: #ffb347; } .danger { color: #ff3d3d; }
    .btn-note { color: #8d98aa; font-size: .78rem; margin-top: .25rem; }
    .safety-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .4rem; margin-top: .6rem; }
    .safety { border: 1px solid #222a34; border-radius: 8px; background: #101318; padding: .4rem; }
    .safety-k { color: #8d98aa; font-size: .62rem; text-transform: uppercase; letter-spacing: .07rem; }
    .safety-v { font-size: .9rem; font-weight: 700; }
    .divider { height: 1px; background: #2b2f36; margin: .85rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def add_log(msg: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs = [f"[{stamp}] {msg}"] + st.session_state.logs[:15]


if "simulator" not in st.session_state:
    st.session_state.simulator = LiveBMSSimulator(RuntimeConfig())
if "history" not in st.session_state:
    st.session_state.history = {
        "time_s": [],
        "request_current_a": [],
        "command_current_a": [],
        "max_temp_c": [],
        "soc_ai_pct": [],
        "soc_true_pct": [],
    }
if "logs" not in st.session_state:
    st.session_state.logs = []
if "low_power" not in st.session_state:
    st.session_state.low_power = False
if "drive_mode" not in st.session_state:
    st.session_state.drive_mode = "auto"
if "show_diag" not in st.session_state:
    st.session_state.show_diag = False
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True

sim = st.session_state.simulator

sim.set_low_power_mode(st.session_state.low_power)
sim.set_drive_mode(st.session_state.drive_mode)
packet = sim.step()

if packet["events"]:
    add_log(" | ".join(packet["events"]))

hist = st.session_state.history
for key in hist:
    hist[key].append(packet[key])
    if len(hist[key]) > 120:
        hist[key].pop(0)

st.markdown('<div class="topline"></div>', unsafe_allow_html=True)
h1, h2 = st.columns([8, 2])
with h1:
    st.markdown('<p class="hdr-title">BATTERY HEALTH MONITOR</p>', unsafe_allow_html=True)
    st.markdown('<p class="hdr-sub">TEAM 25 | HARDWARE PROTOTYPE</p>', unsafe_allow_html=True)
with h2:
    state_text = "• SYSTEM FAULT" if "FAULT" in packet["state"] else "• SYSTEM ONLINE"
    pill_color = "#ff4c4c" if "FAULT" in packet["state"] else "#00e160"
    pill_border = "#5b2727" if "FAULT" in packet["state"] else "#1f3d2a"
    st.markdown(
        f'<div class="online-pill" style="color:{pill_color};border-color:{pill_border};">{state_text}</div>',
        unsafe_allow_html=True,
    )
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

left, center, right = st.columns([1.0, 4.0, 1.0], gap="medium")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="p-title">Live Sensors</div>', unsafe_allow_html=True)
    cell_v = packet["pack_voltage_v"] / max(1, sim.config.series_cells)
    sensor_rows = [
        ("VOLTAGE", f"{cell_v:.2f} V"),
        ("CURRENT", f"{abs(packet['command_current_a']):.2f} A"),
        ("TEMP", f"{packet['max_temp_c']:.1f} °C"),
    ]
    for k, v in sensor_rows:
        c1, c2 = st.columns([1, 1])
        c1.markdown(f'<p class="sensor-k">{k}</p>', unsafe_allow_html=True)
        c2.markdown(f'<p class="sensor-v">{v}</p>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="log-box">', unsafe_allow_html=True)
    st.markdown('<div class="p-title" style="margin:0 0 .4rem;">Diagnostics Log</div>', unsafe_allow_html=True)
    for line in st.session_state.logs[:7]:
        st.markdown(f'<p class="log-line">&gt; {line}</p>', unsafe_allow_html=True)
    if not st.session_state.logs:
        st.markdown('<p class="log-line">&gt; waiting for diagnostic events...</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with center:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    soc = float(packet["soc_ai_pct"])
    deg = int(max(0, min(100, soc)) * 3.6)
    st.markdown(
        f"""
        <div class="ring-wrap">
          <div class="ring" style="--deg:{deg}deg;">
            <div class="ring-inner">
              <div class="soc-main">{round(soc)}</div>
              <div class="soc-sub">% CHARGE</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    est_range = max(0, int(packet["soc_ai_pct"] * 3.8))
    st.markdown(
        f'<div class="range-box"><span class="range-v">{est_range}</span> <span style="color:#8e99aa;">km</span><div class="range-sub">Estimated Range</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="mini-grid">
          <div class="mini"><div class="mini-k">Flow</div><div class="mini-v">{packet['flow_state']}</div></div>
          <div class="mini"><div class="mini-k">AI / True SOC</div><div class="mini-v">{packet['soc_ai_pct']:.1f} / {packet['soc_true_pct']:.1f}</div></div>
          <div class="mini"><div class="mini-k">Pack Power</div><div class="mini-v">{packet['pack_power_kw']:.2f} kW</div></div>
          <div class="mini"><div class="mini-k">SOH</div><div class="mini-v">{packet['soh_pct']:.2f} %</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.show_diag:
        st.markdown("#### Smart Diagnostics")
        diag_df = pd.DataFrame(
            {
                "time_s": hist["time_s"],
                "requested_A": hist["request_current_a"],
                "commanded_A": hist["command_current_a"],
                "max_temp_C": hist["max_temp_c"],
                "ai_soc_%": hist["soc_ai_pct"],
            }
        ).set_index("time_s")
        st.line_chart(diag_df)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="p-title">Control Panel</div>', unsafe_allow_html=True)

    if st.button(
        f"Low Power Mode {'ON' if st.session_state.low_power else 'OFF'}",
        use_container_width=True,
    ):
        st.session_state.low_power = not st.session_state.low_power
        add_log(f"Low power mode {'enabled' if st.session_state.low_power else 'disabled'}")

    if st.button(
        "Run Smart Diagnostics " + ("▲" if st.session_state.show_diag else "▼"),
        use_container_width=True,
    ):
        st.session_state.show_diag = not st.session_state.show_diag

    if st.button("Optimize Charging ⚡", use_container_width=True):
        st.session_state.drive_mode = "charge"
        add_log("Optimize charging activated")

    mode = st.radio(
        "Drive Mode",
        ["auto", "charge", "discharge"],
        index=["auto", "charge", "discharge"].index(st.session_state.drive_mode),
    )
    st.session_state.drive_mode = mode

    if st.button("! SIMULATE FAULT", use_container_width=True):
        sim.trigger_scenario("temp_spike")
        add_log("Fault injected: temperature spike")

    st.markdown(
        f"""
        <div class="safety-grid">
          <div class="safety"><div class="safety-k">Current Limit</div><div class="safety-v {'warn' if packet['limits_active']['current'] else 'ok'}">{'ACTIVE' if packet['limits_active']['current'] else 'SAFE'}</div></div>
          <div class="safety"><div class="safety-k">Voltage Limit</div><div class="safety-v {'danger' if packet['limits_active']['voltage'] else 'ok'}">{'ACTIVE' if packet['limits_active']['voltage'] else 'SAFE'}</div></div>
          <div class="safety"><div class="safety-k">Temp Limit</div><div class="safety-v {'danger' if packet['limits_active']['temperature'] else 'ok'}">{'ACTIVE' if packet['limits_active']['temperature'] else 'SAFE'}</div></div>
          <div class="safety"><div class="safety-k">SOC Limit</div><div class="safety-v {'warn' if packet['limits_active']['soc'] else 'ok'}">{'ACTIVE' if packet['limits_active']['soc'] else 'SAFE'}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="btn-note">Requested {packet["request_current_a"]:.1f} A → Commanded {packet["command_current_a"]:.1f} A</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.session_state.auto_refresh = st.checkbox("Auto refresh", value=st.session_state.auto_refresh)
if st.session_state.auto_refresh:
    time.sleep(max(sim.config.dt_s, 0.6))
    st.rerun()
