from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from live_simulator import LiveBMSSimulator, RuntimeConfig


st.set_page_config(page_title="EV AI-BMS Dashboard", layout="wide")


def style_status(active: bool, danger: bool = False) -> str:
    if active and danger:
        return "red"
    if active:
        return "orange"
    return "green"


if "simulator" not in st.session_state:
    st.session_state.simulator = LiveBMSSimulator(RuntimeConfig())
if "history" not in st.session_state:
    st.session_state.history = {
        "time_s": [],
        "request_current_a": [],
        "command_current_a": [],
        "soc_ai_pct": [],
        "soc_true_pct": [],
        "max_temp_c": [],
    }

sim = st.session_state.simulator
history = st.session_state.history

st.title("EV Battery Simulator + AI BMS")
st.caption("Physics-based battery simulation with AI SOC-driven BMS safety limiting")

col_mode, col_lpm, col_auto = st.columns([2, 1, 1])
with col_mode:
    drive_mode = st.radio("Drive mode", ["auto", "charge", "discharge"], horizontal=True)
with col_lpm:
    low_power = st.toggle("Low power mode", value=sim.low_power_mode)
with col_auto:
    auto_refresh = st.toggle("Auto refresh", value=True)

sim.set_drive_mode(drive_mode)
sim.set_low_power_mode(low_power)

st.markdown("### Scenario Triggers")
sc1, sc2, sc3 = st.columns(3)
if sc1.button("Temp Spike"):
    sim.trigger_scenario("temp_spike")
if sc2.button("Low Voltage"):
    sim.trigger_scenario("low_voltage")
if sc3.button("High Voltage"):
    sim.trigger_scenario("high_voltage")

packet = sim.step()

for key in history:
    history[key].append(packet[key])
    if len(history[key]) > 180:
        history[key].pop(0)

top1, top2, top3, top4, top5, top6 = st.columns(6)
top1.metric("Flow", packet["flow_state"])
top2.metric("AI SOC", f"{packet['soc_ai_pct']:.2f}%")
top3.metric("True SOC", f"{packet['soc_true_pct']:.2f}%")
top4.metric("Pack Voltage", f"{packet['pack_voltage_v']:.2f} V")
top5.metric("Pack Current", f"{packet['command_current_a']:.2f} A")
top6.metric("Max Temp", f"{packet['max_temp_c']:.2f} C")

st.markdown("### Safety Status")
safe1, safe2, safe3, safe4 = st.columns(4)
safe1.markdown(
    f":{style_status(packet['limits_active']['current'])}[Current limiting: {'ACTIVE' if packet['limits_active']['current'] else 'NORMAL'}]"
)
safe2.markdown(
    f":{style_status(packet['limits_active']['voltage'], True)}[Voltage guard: {'ACTIVE' if packet['limits_active']['voltage'] else 'SAFE'}]"
)
safe3.markdown(
    f":{style_status(packet['limits_active']['temperature'], True)}[Temperature guard: {'ACTIVE' if packet['limits_active']['temperature'] else 'SAFE'}]"
)
safe4.markdown(
    f":{style_status(packet['limits_active']['soc'])}[SOC guard: {'ACTIVE' if packet['limits_active']['soc'] else 'SAFE'}]"
)

mid1, mid2 = st.columns(2)
with mid1:
    st.markdown("### Current Control")
    current_df = pd.DataFrame(
        {
            "time_s": history["time_s"],
            "requested_A": history["request_current_a"],
            "commanded_A": history["command_current_a"],
        }
    ).set_index("time_s")
    st.line_chart(current_df)

with mid2:
    st.markdown("### SOC + Temperature")
    soc_df = pd.DataFrame(
        {
            "time_s": history["time_s"],
            "ai_soc_pct": history["soc_ai_pct"],
            "true_soc_pct": history["soc_true_pct"],
            "max_temp_c": history["max_temp_c"],
        }
    ).set_index("time_s")
    st.line_chart(soc_df)

st.markdown("### BMS Windows")
w1, w2, w3, w4 = st.columns(4)
w1.metric("Allowed (A)", f"{packet['windows']['allowed'][0]:.1f} to {packet['windows']['allowed'][1]:.1f}")
w2.metric("Voltage window (A)", f"{packet['windows']['voltage'][0]:.1f} to {packet['windows']['voltage'][1]:.1f}")
w3.metric("Temp window (A)", f"{packet['windows']['temp'][0]:.1f} to {packet['windows']['temp'][1]:.1f}")
w4.metric("SOC window (A)", f"{packet['windows']['soc'][0]:.1f} to {packet['windows']['soc'][1]:.1f}")

st.markdown(
    f"**Events:** {', '.join(packet['events']) if packet['events'] else 'None'}  \n"
    f"**Limiting reasons:** {', '.join(packet['limiting_reasons']) if packet['limiting_reasons'] else 'None'}"
)

if auto_refresh:
    time.sleep(max(sim.config.dt_s, 0.5))
    st.rerun()
