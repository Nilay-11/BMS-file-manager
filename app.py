import time
import serial
import numpy as np
import tensorflow as tf
import joblib
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO

# ================= CONFIGURATION =================
# Standard COM port for Arduino. Change to 'COM3' or '/dev/ttyUSB0' if needed.
SERIAL_PORT = 'COM5' 
BAUD_RATE = 9600

# Initialize Web Server
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ================= 1. LOAD AI MODEL & SCALER =================
print("⏳ Loading AI System...")
try:
    model = tf.keras.models.load_model("bms_model.h5")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model & Scaler Loaded.")
except Exception as e:
    print(f"❌ Error loading AI files: {e}")
    print("⚠️ Ensure 'bms_model.h5' and 'scaler.pkl' are in the folder.")
    exit()

# ================= 2. SERVE YOUR FRONTEND =================
@app.route('/')
def index():
    # This looks for 'dashboard.html' inside the 'templates' folder
    return render_template('dashboard.html')

# ================= 3. HARDWARE LISTENER LOOP =================
def bms_logic():
    print("🔌 Searching for BMS Hardware...")
    
    # Try connecting to Arduino
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"✅ Connected to {SERIAL_PORT}")
    except:
        print(f"⚠️ Could not find {SERIAL_PORT}. Running in SIMULATION MODE.")
        ser = None

    # Storage for the AI window (last 20 readings)
    # We init with zeros to prevent crashing before data arrives
    input_buffer = []

    while True:
        try:
            # --- A. GET DATA (Real or Simulated) ---
            if ser and ser.in_waiting:
                # Read from Arduino: "12.5, 2.1, 35.0"
                raw = ser.readline().decode('utf-8').strip()
                values = list(map(float, raw.split(',')))
            else:
                # Simulation Mode (If hardware isn't connected yet)
                time.sleep(1) 
                values = [12.6, 1.5, 30.0] # Dummy V, I, T

            if len(values) == 3:
                voltage, current, temp = values

                # --- B. PREPARE FOR AI ---
                # Scale the data using your scaler
                features = np.array([[voltage, current, temp]])
                scaled_features = scaler.transform(features)
                
                # Add to buffer (Rolling window of 20 steps)
                input_buffer.append(scaled_features[0])
                if len(input_buffer) > 20: input_buffer.pop(0)

                # --- C. PREDICT SOC ---
                soc = 50.0 # Default
                if len(input_buffer) == 20:
                    # Reshape for LSTM: (1, 20, 3)
                    ai_input = np.array(input_buffer).reshape(1, 20, 3)
                    pred = model.predict(ai_input, verbose=0)
                    soc = float(pred[0][0]) # Assuming output is SOC

                # --- D. SEND TO DASHBOARD ---
                data_packet = {
                    "voltage": voltage,
                    "current": current,
                    "temp": temp,
                    "soc": round(soc, 1),
                    # Simple estimations for Range/SOH based on SOC
                    "range": int(soc * 3.5), 
                    "soh": 98.5 
                }
                
                # Emit to website
                socketio.emit('update_dashboard', data_packet)
                print(f"Sent: {data_packet}")

        except Exception as e:
            print(f"Error in loop: {e}")
            time.sleep(1)

# Start the background thread
thread = threading.Thread(target=bms_logic)
thread.daemon = True
thread.start()

# ================= 4. START SERVER =================
if __name__ == '__main__':
    socketio.run(app, debug=False, port=5000)
