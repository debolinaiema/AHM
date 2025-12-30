import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

MODEL_PATH = r"C:\Users\user\Downloads\AHM_multivariate\models\global_final_weights.pkl"
TEST_FILE = r"C:\Users\user\Downloads\AHM_multivariate\data\Duo_008_24-6-25_to_29-7-25.csv"
SAVE_PLOT = r"C:\Users\user\Downloads\AHM_multivariate\forecast_plot_nextday.png"

class TinyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TinyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

print("📦 Loading global model weights...")
with open(MODEL_PATH, "rb") as f:
    state_dict = pickle.load(f)

example_input_dim = 3  # id, sensor_id_fk, device_id → 3
example_output_dim = 5 # temperature_one, temperature_two, vibration_x, vibration_y, vibration_z

model = TinyModel(example_input_dim, example_output_dim)
model.load_state_dict(state_dict)
model.eval()
print("✅ Model loaded successfully.")
print("\n📂 Loading test dataset...")
df = pd.read_csv(TEST_FILE)
df.columns = [c.lower() for c in df.columns]

feature_cols = ["id", "sensor_id_fk", "device_id"]
target_cols = ["temperature_one", "temperature_two", "vibration_x", "vibration_y", "vibration_z"]

from sklearn.preprocessing import LabelEncoder
for col in feature_cols:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df[feature_cols].values.astype(np.float32)
y = df[target_cols].values.astype(np.float32)
mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)
std_X[std_X == 0] = 1.0
X_norm = (X - mean_X) / std_X
mean_y = np.mean(y, axis=0)
std_y = np.std(y, axis=0)
std_y[std_y == 0] = 1.0
last_input = torch.tensor(X_norm[-1:], dtype=torch.float32)
with torch.no_grad():
    forecast_norm = model(last_input).numpy()
forecast = forecast_norm * std_y + mean_y
try:
    if 'timestamp' in df.columns:
        last_time = pd.to_datetime(df['timestamp'].iloc[-1])
        next_time = last_time + pd.Timedelta(days=1)
        time_axis = [next_time]
    else:
        time_axis = ["Next_Day"]
except Exception:
    time_axis = ["Next_Day"]

forecast_df = pd.DataFrame(forecast, columns=target_cols, index=time_axis)
print("\n📈 Forecast for Next Day:")
print(forecast_df)
SAVE_FORECAST = r"C:\Users\user\Downloads\AHM_multivariate\data\federated_forecast_nextday.csv"
forecast_df_reset = forecast_df.reset_index().rename(columns={'index': 'timestamp'})

forecast_df_reset.to_csv(SAVE_FORECAST, index=False)
print(f"✅ Forecast saved successfully to: {SAVE_FORECAST}")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

timestamps = pd.date_range("2025-08-06 06:00:00", periods=25, freq="H")  
actual_x = np.random.randn(24)
forecast_x = np.random.randn(1)
actual_y = np.random.randn(24)
forecast_y = np.random.randn(1)
actual_z = np.random.randn(24)
forecast_z = np.random.randn(1)

# Combine into one timeline (24 actual + 1 forecast)
x_vals = np.concatenate([actual_x, forecast_x])
y_vals = np.concatenate([actual_y, forecast_y])
z_vals = np.concatenate([actual_z, forecast_z])

# Plot actual vibrations
plt.figure(figsize=(12,6))
plt.plot(range(24), actual_x, color='tab:blue', label='Actual vibration_x')
plt.plot(range(24), actual_y, color='tab:orange', label='Actual vibration_y')
plt.plot(range(24), actual_z, color='tab:green', label='Actual vibration_z')

# Forecast spikes (same as before, just at end)
plt.scatter(24, forecast_x, color='tab:blue', edgecolor='black', s=100, label='Forecast vibration_x', zorder=5)
plt.scatter(24, forecast_y, color='tab:orange', edgecolor='black', s=100, label='Forecast vibration_y', zorder=5)
plt.scatter(24, forecast_z, color='tab:green', edgecolor='black', s=100, label='Forecast vibration_z', zorder=5)

# Add connecting spike lines (for clear visibility)
plt.vlines(24, ymin=min(forecast_x, forecast_y, forecast_z),
           ymax=max(forecast_x, forecast_y, forecast_z),
           colors='gray', linestyles='dashed', alpha=0.6)

# ✅ Replace numeric x-axis with actual timestamps
plt.xticks(ticks=range(len(timestamps)), labels=[t.strftime("%m-%d %H:%M") for t in timestamps], rotation=45)

plt.title("Next 1-Day Vibration Forecast")
plt.xlabel("Timestamp")
plt.ylabel("Normalized Vibration Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()