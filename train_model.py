import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- 1. SETUP & LOAD DATA ---
print("🚀 Starting AI Training Process...")

# Define the file name based on what you uploaded
FILE_NAME = 'distributed_bms_dataset.csv'

# Check if the file exists before trying to load it
if not os.path.exists(FILE_NAME):
    print(f"❌ Error: The file '{FILE_NAME}' was not found.")
    print(f"   Current working directory: {os.getcwd()}")
    print("   Make sure the CSV file is in this folder!")
    exit()

print(f"📂 Loading dataset: {FILE_NAME}...")
df = pd.read_csv(FILE_NAME)

# --- 2. INSPECT & CLEAN COLUMNS ---
print(f"🔍 Raw Columns in CSV: {list(df.columns)}")

# Strip any invisible spaces from column names to avoid errors
df.columns = df.columns.str.strip()

# Map YOUR dataset's specific names to the generic names the AI expects
# Based on the file you uploaded:
# 'cell_voltage' -> 'Voltage'
# 'module_current' -> 'Current'
# 'cell_temp' -> 'Temperature'
# 'soc_pct' -> 'SOC'
rename_map = {
    'cell_voltage': 'Voltage',
    'module_current': 'Current',
    'cell_temp': 'Temperature',
    'soc_pct': 'SOC'
}

print("🔄 Renaming columns...")
df = df.rename(columns=rename_map)

# Verify the columns exist now
required_cols = ['Voltage', 'Current', 'Temperature', 'SOC']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"❌ CRITICAL ERROR: Could not find columns: {missing_cols}")
    print(f"   Current Columns after renaming: {list(df.columns)}")
    print("   Please check the 'rename_map' section in the script matches your CSV headers exactly.")
    exit()

print("✅ Columns mapped successfully!")

# --- 3. PREPARE DATA ---
# Inputs (Features): Voltage, Current, Temperature
X = df[['Voltage', 'Current', 'Temperature']].values
# Output (Target): State of Charge (SOC)
y = df['SOC'].values

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (helps the Neural Network learn faster)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler so the app can use it later
joblib.dump(scaler, 'scaler.pkl')
print("💾 Saved data scaler as 'scaler.pkl'")

# --- 4. BUILD THE AI MODEL ---
print("🧠 Building Neural Network...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)), # Input layer (3 features)
    tf.keras.layers.Dense(32, activation='relu'),                   # Hidden layer
    tf.keras.layers.Dense(16, activation='relu'),                   # Hidden layer
    tf.keras.layers.Dense(1)                                        # Output layer (1 number: SOC)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- 5. TRAIN THE MODEL ---
print("🏋️‍♂️ Training model (this may take a moment)...")
# Training for 20 epochs (rounds)
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

# --- 6. SAVE THE MODEL ---
print("💾 Saving the trained brain...")
model.save('bms_model.h5')
print("\n🎉 SUCCESS! Model saved as 'bms_model.h5'")
print("   You can now run 'python app.py' to start your dashboard.")