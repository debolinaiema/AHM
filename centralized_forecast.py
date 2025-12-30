# centralized_forecast.py
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# পথগুলো ঠিক করো যদি দরকার হয়
MODEL_PATH = r"C:\Users\user\Downloads\AHM_multivariate\models\global_final_weights.pkl"
TEST_FILE  = r"C:\Users\user\Downloads\AHM_multivariate\data\Duo_008_24-6-25_to_29-7-25.csv"
SAVE_FILE  = r"C:\Users\user\Downloads\AHM_multivariate\forecast_next_day.csv"  # এটাই তোমার NORMAL ফাইল

print("মডেল লোড হচ্ছে...")
class TinyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x): return self.net(x)

# মডেল লোড
with open(MODEL_PATH, "rb") as f:
    state_dict = pickle.load(f)
model = TinyModel(3, 5)
model.load_state_dict(state_dict)
model.eval()

# ডেটা লোড
df = pd.read_csv(TEST_FILE)
df.columns = [c.lower() for c in df.columns]

# ফিচার আর টার্গেট
feature_cols = ["id", "sensor_id_fk", "device_id"]
target_cols = ["temperature_one", "temperature_two", "vibration_x", "vibration_y", "vibration_z"]

# এনকোড
from sklearn.preprocessing import LabelEncoder
for col in feature_cols:
    if df[col].dtype == "object":
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df[feature_cols].values.astype(np.float32)
y = df[target_cols].values.astype(np.float32)

# নরমালাইজ
mean_X = X.mean(axis=0); std_X = X.std(axis=0); std_X[std_X == 0] = 1
X_norm = (X - mean_X) / std_X
mean_y = y.mean(axis=0); std_y = y.std(axis=0); std_y[std_y == 0] = 1

# শেষ রো দিয়ে পরের দিন প্রেডিক্ট
last_input = torch.tensor(X_norm[-1:], dtype=torch.float32)
with torch.no_grad():
    pred_norm = model(last_input).numpy()
pred = pred_norm * std_y + mean_y
pred += np.random.normal(0, 0.005, pred.shape)  # ছোট নয়েজ যোগ করা হলো

# টাইমস্ট্যাম্প
if 'timestamp' in df.columns:
    last_time = pd.to_datetime(df['timestamp'].iloc[-1])
    next_time = last_time + pd.Timedelta(days=1)
else:
    next_time = "Next_Day"

# ফাইল সেভ
forecast_df = pd.DataFrame(pred, columns=target_cols)
forecast_df['timestamp'] = next_time
forecast_df = forecast_df[['timestamp'] + target_cols]
forecast_df.to_csv(SAVE_FILE, index=False)

print(f" YAY! Centralized file is saved{SAVE_FILE}")
print(forecast_df)