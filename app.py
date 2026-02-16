import time
import numpy as np
import tensorflow as tf
from flask import Flask
from flask_socketio import SocketIO, emit
import threading

# Initialize App
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 1. LOAD YOUR AI MODEL
# Ensure you have run 'train_model.py' first to create this file
try:
    model = tf.keras.models.load_model('bms_model.h5')
    print("✅ AI Model Loaded Successfully")
except:
    print("⚠️ No Model Found. Running in Simulation Mode.")
    model = None

# 2. SYSTEM STATE (The "Memory" of your car)
vehicle_state = {
    # Sensor Readings
    "voltage": 3.7,
    "current": 1.5,
    "temp": 30.0,
    
    # AI Predictions
    "soc": 85.0,        # State of Charge
    "soh": 98.0,        # State of Health (Longevity)
    "range": 320,       # Estimated km
    
    # User Features (Toggled from Dashboard)
    "low_power_mode": False,
    "peak_performance": "STABLE",
    "charging_optimized": False
}

def decision_making_loop():
    """
    The 'Brain' that runs every second to check safety and calculate range.
    """
    while True:
        # --- A. READ SENSORS (Simulated) ---
        # In real life, this comes from Arduino via Serial
        vehicle_state["voltage"] += np.random.uniform(-0.02, 0.02)
        # Keep voltage realistic (3.0V to 4.2V)
        vehicle_state["voltage"] = np.clip(vehicle_state["voltage"], 3.0, 4.2)
        
        # Simulate Temp rising if current is high
        if vehicle_state["current"] > 2.0:
            vehicle_state["temp"] += 0.1
        else:
            vehicle_state["temp"] -= 0.05
        
        # --- B. AI PREDICTION ---
        if model:
            # Prepare input: [[Voltage, Current, Temp]]
            input_data = np.array([[
                vehicle_state["voltage"], 
                vehicle_state["current"], 
                vehicle_state["temp"]
            ]])
            pred = model.predict(input_data, verbose=0)
            vehicle_state["soc"] = float(pred[0][0])
            vehicle_state["soh"] = float(pred[0][1]) # Predicted Life Health
        else:
            # Fallback if no model (Linear approx)
            vehicle_state["soc"] = ((vehicle_state["voltage"] - 3.0) / 1.2) * 100
        
        # --- C. FEATURE LOGIC (The "Three Pillars" Implementation) ---

        # 1. OPTIMIZE FOR LONGER RANGE (Low Power Mode)
        base_efficiency = 3.5 # km per % charge
        if vehicle_state["low_power_mode"]:
            # Boost range calculation by 20% (Simulates turning off AC/Limiting Speed)
            vehicle_state["range"] = int(vehicle_state["soc"] * base_efficiency * 1.2)
        else:
            vehicle_state["range"] = int(vehicle_state["soc"] * base_efficiency)

        # 2. PEAK PERFORMANCE CHECK
        # Protects battery from high stress 
        if vehicle_state["temp"] > 45.0 or vehicle_state["soc"] < 20.0:
            vehicle_state["peak_performance"] = "LIMITED"
        else:
            vehicle_state["peak_performance"] = "STABLE"

        # 3. BATTERY LIFE OPTIMIZATION (Charging Logic)
        # Prevents Over-Charging [cite: 15] and Heat Stress [cite: 17]
        if vehicle_state["charging_optimized"]:
            # If temp gets too high during 'optimization mode', simulate throttling
            if vehicle_state["temp"] > 35.0:
                vehicle_state["current"] = 0.5 # Slow down charging to cool off
            
            # Simulate stopping charge at 80% to extend life
            if vehicle_state["soc"] >= 80.0:
                vehicle_state["current"] = 0.0 # Stop charging

        # --- D. SEND TO DASHBOARD ---
        socketio.emit('update_dashboard', vehicle_state)
        time.sleep(1)

# Start the logic loop in background
thread = threading.Thread(target=decision_making_loop)
thread.daemon = True
thread.start()

# --- E. LISTEN FOR DASHBOARD BUTTONS ---

@socketio.on('toggle_lpm')
def handle_lpm(data):
    vehicle_state["low_power_mode"] = data['active']
    print(f"Low Power Mode set to: {vehicle_state['low_power_mode']}")

@socketio.on('optimize_charge')
def handle_optimize():
    vehicle_state["charging_optimized"] = True
    print("Optimized Charging Routine Started...")

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)