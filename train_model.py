import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- 1. LOAD YOUR DATASET ---
# Replace 'battery_data.csv' with your actual file name
print("Loading dataset...")
df = pd.read_csv('distributed_bms_dataset.csv')

# --- 2. CONFIGURATION ---
# UPDATE THESE STRINGS to match your exact column headers in the CSV
col_voltage = 'Voltage'      # e.g., 'V', 'voltage_v', 'Cell_Voltage'
col_current = 'Current'      # e.g., 'I', 'current_a', 'Current_measured'
col_temp    = 'Temperature'  # e.g., 'T', 'temp_c', 'Temperature_measured'
col_target  = 'SOC'          # The column you want to predict

# Check if columns exist
required_cols = [col_voltage, col_current, col_temp, col_target]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Error: Column '{col}' not found in your CSV. Please check headers!")

# --- 3. PREPROCESSING ---
# Select Features (Inputs) and Labels (Outputs)
X = df[[col_voltage, col_current, col_temp]].values
y = df[col_target].values

# Normalize the data (Scale everything between 0 and 1 for the AI)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 4. BUILD THE MODEL (Updated for Real Data) ---
# We use a deeper model here to capture real-world complexities
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(3,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='linear') # Outputting 1 value: SOC
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- 5. TRAIN ---
print("Training Neural Network on YOUR data...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# --- 6. SAVE ---
model.save('bms_model_real.h5')
print("✅ Model trained on real data and saved as 'bms_model_real.h5'")

# Optional: Test with a sample from your data
test_sample = np.array([[3.7, 1.5, 30]]) # Example: 3.7V, 1.5A, 30C
test_sample_scaled = scaler.transform(test_sample)
prediction = model.predict(test_sample_scaled)
print(f"Test Prediction for 3.7V: {prediction[0][0]:.2f}% SOC")