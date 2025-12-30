import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from collections import deque
FILES = [
    "Duo_001_24-6-25_to_31-7-25.csv",
    "Duo_002_24-6-25_to_29-7-25.csv",
    "Duo_003_24-6-25_to_01-8-25.csv",
    "Duo_004_24-6-25_to_29-7-25.csv",
]
USE_COLS = ["timestamp","device_id","vibration_x","vibration_y","vibration_z","temperature_one","temperature_two"].reset_index(drop=True)

def parse_timestamp(col):
    ts = pd.to_datetime(col, errors="coerce", infer_datetime_format=True)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Asia/Kolkata")
    else:
        ts = ts.dt.tz_convert("Asia/Kolkata")
    return ts

devices = {}
for path in FILES:
    df = pd.read_csv(path)
    df["timestamp"] = parse_timestamp(df["timestamp"])
    df = df.sort_values("timestamp").dropna(subset=["vibration_x","vibration_y","vibration_z"]).reset_index(drop=True)
    dev_id = str(df["device_id"].iloc[0])
    devices[dev_id] = df[USE_COLS].copy()

print("Loaded devices:", list(devices.keys()))

SPLIT_RATIO = 0.8
SEQ_LEN = 60
FEATS = ["vibration_x","vibration_y","vibration_z"]

fit_chunks, split_idx_map = [], {}
for dev, df in devices.items():
    n = len(df)
    split_idx = int(n * SPLIT_RATIO)
    split_idx_map[dev] = split_idx
    fit_chunks.append(df.loc[:split_idx-1, FEATS].values)

fit_matrix = np.vstack(fit_chunks)
scaler = StandardScaler().fit(fit_matrix)
print("Scaler fitted on", fit_matrix.shape, "train samples.")

def make_sequences_no_leak(arr_scaled, seq_len, train_split_idx):
    N, F = arr_scaled.shape
    Xtr, ytr, Xte, yte = [], [], [], []
    last_train_start = train_split_idx - 1 - seq_len
    for i in range(max(0, last_train_start) + 1):
        Xtr.append(arr_scaled[i:i+seq_len]); ytr.append(arr_scaled[i+seq_len])
    for i in range(train_split_idx, N - seq_len):
        Xte.append(arr_scaled[i:i+seq_len]); yte.append(arr_scaled[i+seq_len])
    return (np.array(Xtr), np.array(ytr), np.array(Xte), np.array(yte))

Xtr_list, ytr_list, Xte_list, yte_list = [], [], [], []
for dev, df in devices.items():
    vals = df[FEATS].values.astype(float)
    vals_scaled = scaler.transform(vals)
    split_idx = split_idx_map[dev]
    Xtr, ytr, Xte, yte = make_sequences_no_leak(vals_scaled, SEQ_LEN, split_idx)
    print(f"{dev}: train_seq={len(Xtr)}, test_seq={len(Xte)}")
    if len(Xtr): Xtr_list.append(Xtr); ytr_list.append(ytr)
    if len(Xte): Xte_list.append(Xte); yte_list.append(yte)

X_train = np.vstack(Xtr_list); y_train = np.vstack(ytr_list)
X_test  = np.vstack(Xte_list);  y_test  = np.vstack(yte_list)
print("Shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
n_features = len(FEATS)
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(SEQ_LEN, n_features)),
    Dropout(0.2),
    GRU(32),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(n_features, activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='mse', metrics=['mae'])
model.summary()
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=256,
    shuffle=True,
    callbacks=callbacks,
    verbose=1
)

loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {loss:.5f} | MAE: {mae:.5f}")

pred = model.predict(X_test, verbose=0)
per_dim_mae = np.mean(np.abs(pred - y_test), axis=0)
print("Per-dimension MAE [x, y, z]:", per_dim_mae)
model.save("gru_vibration_seq60.keras")
np.savez("vibration_scaler_standard.npz",
         mean_=scaler.mean_, scale_=scaler.scale_, feature_names=np.array(FEATS))

print("Artifacts saved: gru_vibration_seq60.keras & vibration_scaler_standard.npz")
class RealTimePredictor:
    def __init__(self, model, scaler, seq_len=60):
        self.model = model
        self.scaler = scaler
        self.seq_len = seq_len
        self.buf = deque(maxlen=seq_len)

    def push_and_predict(self, new_row_xyz):
        self.buf.append(np.asarray(new_row_xyz, dtype=float))
        if len(self.buf) < self.seq_len:
            return None
        window = np.stack(self.buf, axis=0)
        window_scaled = (window - scaler.mean_) / scaler.scale_
        x = window_scaled.reshape(1, self.seq_len, len(self.scaler.mean_))
        y_scaled = self.model.predict(x, verbose=0)[0]
        y_raw = y_scaled * scaler.scale_ + scaler.mean_
        return y_raw
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
dev = list(devices.keys())[0]
df_day = devices[dev].copy()
one_day = "2025-07-01"
df_day = df_day[df_day["timestamp"].dt.strftime("%Y-%m-%d") == one_day]

if len(df_day) < SEQ_LEN + 1:
    print("Not enough data for selected day.")
else:
    vals = df_day[FEATS].values.astype(float)
    vals_scaled = scaler.transform(vals)
    preds, actuals, times = [], [], []
    for i in range(len(vals_scaled) - SEQ_LEN):
        x = vals_scaled[i:i+SEQ_LEN].reshape(1, SEQ_LEN, n_features)
        y_pred_scaled = model.predict(x, verbose=0)[0]
        y_pred = y_pred_scaled * scaler.scale_ + scaler.mean_
        preds.append(y_pred)
        actuals.append(vals[i+SEQ_LEN])
        times.append(df_day["timestamp"].iloc[i+SEQ_LEN])

    preds = np.array(preds)
    actuals = np.array(actuals)

    # ---------------- Plot 1: Temp vs Vibration ----------------
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax2 = ax1.twinx()

    ax1.plot(df_day["timestamp"], df_day["temperature_one"], 'r-', label="Temp One")
    ax1.plot(df_day["timestamp"], df_day["temperature_two"], 'm-', label="Temp Two")
    ax2.plot(df_day["timestamp"], df_day["vibration_x"], 'b-', alpha=0.6, label="Vibration X")

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperature", color='r')
    ax2.set_ylabel("Vibration X", color='b')
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    plt.title(f"Device {dev} | {one_day} | Temperature vs Vibration X")
    plt.show()

    # ---------------- Plot 2: Actual vs Predicted ----------------
    plt.figure(figsize=(12,6))
    plt.plot(times, actuals[:,0], label="Actual Vx", color="blue")
    plt.plot(times, preds[:,0], label="Predicted Vx", color="cyan", linestyle="--")
    plt.plot(times, actuals[:,1], label="Actual Vy", color="green")
    plt.plot(times, preds[:,1], label="Predicted Vy", color="lime", linestyle="--")
    plt.plot(times, actuals[:,2], label="Actual Vz", color="orange")
    plt.plot(times, preds[:,2], label="Predicted Vz", color="red", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Vibration")
    plt.title(f"Device {dev} | {one_day} | Actual vs Predicted Vibration")
    plt.legend()
    plt.grid(True)
    plt.show()