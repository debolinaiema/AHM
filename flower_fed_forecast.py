# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# class TinyModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128), nn.ReLU(),
#             nn.Linear(128, 64), nn.ReLU(),
#             nn.Linear(64, 32), nn.ReLU(),
#             nn.Linear(32, output_dim)
#         )
#     def forward(self, x):
#         return self.net(x)
    
# def encode_features(df, feature_cols):
#     for col in feature_cols:
#         if df[col].dtype == object:
#             df[col] = LabelEncoder().fit_transform(df[col].astype(str))
#     return df

# def load_dataset(path):
#     df = pd.read_csv(path)
#     df.columns = df.columns.str.lower()

#     feature_cols = ["id", "sensor_id_fk", "device_id"]
#     target_cols = ["temperature_one", "temperature_two",
#                    "vibration_x", "vibration_y", "vibration_z"]

#     df = encode_features(df, feature_cols)

#     X = df[feature_cols].values.astype(np.float32)
#     y = df[target_cols].values.astype(np.float32)
#     return X, y, df

# MODEL_PATH = r"D:\Projects(Debolina)\AHM_multivariate\models/global_model_flare.pth"

# def load_trained_model(input_dim, output_dim):
#     model = TinyModel(input_dim, output_dim)
#     state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
#     model.load_state_dict(state)
#     model.eval()
#     return model

# def forecast_next(model, last_X_norm, mean_y, std_y):
#     """Predict next y using last normalized X row."""
#     last_X_tensor = torch.tensor(last_X_norm.reshape(1, -1))
#     y_norm_pred = model(last_X_tensor).detach().numpy()[0]
#     y_pred = y_norm_pred * std_y + mean_y
#     return y_pred

# if __name__ == "__main__":
#     TEST_FILE = r"D:\Projects(Debolina)\AHM_multivariate\data\Duo_006_6-8-25_to_13-8-25.csv"
#     FILE1 = r"D:\Projects(Debolina)\AHM_multivariate\data\Duo_001_24-6-25_to_31-7-25.csv"
#     FILE2 = r"D:\Projects(Debolina)\AHM_multivariate\data\Duo_007_24-6-25_to_06-8-25.csv"

#     X1, y1, df1 = load_dataset(FILE1)
#     X2, y2, df2 = load_dataset(FILE2)
#     mean_X = np.mean(np.vstack([X1, X2]), axis=0)
#     std_X = np.std(np.vstack([X1, X2]), axis=0); std_X[std_X == 0] = 1
#     mean_y = np.mean(np.vstack([y1, y2]), axis=0)
#     std_y = np.std(np.vstack([y1, y2]), axis=0); std_y[std_y == 0] = 1
#     X_test, y_test, df_test = load_dataset(TEST_FILE)
#     X_test_norm = (X_test - mean_X) / std_X
#     model = load_trained_model(input_dim=X_test.shape[1], output_dim=y_test.shape[1])
#     last_X = X_test_norm[-1]
#     y_next = forecast_next(model, last_X, mean_y, std_y)
#     target_names = ["Temperature_One", "Temperature_Two",
#                     "Vibration_X", "Vibration_Y", "Vibration_Z"]

#     print("\n==============================")
#     print("🔥 NEXT POINT FORECAST RESULT")
#     print("==============================")
#     for name, val in zip(target_names, y_next):
#         print(f"{name}: {val:.4f}")

# import matplotlib.pyplot as plt
# from datetime import timedelta

# if "timestamp" in df_test.columns:
#     df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])
#     time_axis = df_test["timestamp"]
# else:
#     time_axis = pd.date_range(end=pd.Timestamp.now(), periods=len(df_test))

# vx = df_test["vibration_x"].values
# vy = df_test["vibration_y"].values
# vz = df_test["vibration_z"].values
# next_time = time_axis.iloc[-1] + (time_axis.iloc[-1] - time_axis.iloc[-2])

# plt.figure(figsize=(18, 6))
# plt.title("Vibration Forecast (Flower model prediction)")
# plt.grid(True, alpha=0.3)
# plt.plot(time_axis, vx, label="Actual X", color="blue")
# plt.plot(time_axis, vy, label="Actual Y", color="orange")
# plt.plot(time_axis, vz, label="Actual Z", color="green")
# plt.plot([time_axis.iloc[-1], next_time], 
#          [vx[-1], y_next[2]], "r--", label="Forecast X")
# plt.plot([time_axis.iloc[-1], next_time], 
#          [vy[-1], y_next[3]], "purple", linestyle="--", label="Forecast Y")
# plt.plot([time_axis.iloc[-1], next_time], 
#          [vz[-1], y_next[4]], "brown", linestyle="--", label="Forecast Z")
# plt.scatter([next_time], [y_next[2]], color="red")
# plt.scatter([next_time], [y_next[3]], color="purple")
# plt.scatter([next_time], [y_next[4]], color="brown")
# plt.axvspan(time_axis.iloc[-1], next_time, color="grey", alpha=0.2, label="Forecast region")
# plt.xlabel("Timestamp")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.tight_layout()
# plt.show()

























# ============================================================
# TESTING + METRICS FOR PURE GPU FEDERATED GRU MODEL
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# -----------------------------
# CUDA SAFETY
# -----------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# GPU SETUP
# -----------------------------
assert torch.cuda.is_available(), "CUDA not available"
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print("🔥 Using GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# PATHS (CHANGE IF NEEDED)
# -----------------------------
MODEL_PATH = "models_pure_gpu/global_gpu_only.pth"

TRAIN_FILES = [
    r"D:\AHM_multivariate\data\Duo_001_24-6-25_to_31-7-25.csv",
    r"D:\AHM_multivariate\data\Duo_007_24-6-25_to_06-8-25.csv",
]

TEST_FILE = r"D:\AHM_multivariate\data\Duo_006_6-8-25_to_13-8-25.csv"

# -----------------------------
# CONFIG (MUST MATCH TRAINING)
# -----------------------------
SEQ_LEN = 120
BATCH_SIZE = 128   # RTX 3050 safe

FEATURES = [
    "temperature_one",
    "temperature_two",
    "vibration_x",
    "vibration_y",
    "vibration_z"
]

# ============================================================
# MODEL (SAME AS TRAINING)
# ============================================================
class GRUBasedAutoencoder(nn.Module):
    def __init__(self, n_features=5):
        super().__init__()
        self.gru1 = nn.GRU(n_features, 128, 2, batch_first=True, dropout=0.1)
        self.gru2 = nn.GRU(128, 64, 2, batch_first=True, dropout=0.1)
        self.decoder = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32), nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, n_features)
        )

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return self.decoder(x[:, -1])

# ============================================================
# DATA HELPERS
# ============================================================
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    return df[FEATURES].values.astype(np.float32)

def to_sequences(X, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(X[i+seq_len])
    return np.array(Xs), np.array(ys)

# ============================================================
# NORMALIZATION (SAME AS TRAINING)
# ============================================================
print("\n📥 Loading training data for normalization...")

all_train = []
for f in TRAIN_FILES:
    all_train.append(load_csv(f))

X_all = np.vstack(all_train)
mean_X = X_all.mean(axis=0)
std_X  = X_all.std(axis=0)
std_X[std_X == 0] = 1.0

print("✅ Normalization ready")

# ============================================================
# LOAD TEST DATA
# ============================================================
print("\n📥 Loading test data...")

X_test_raw = load_csv(TEST_FILE)
X_test = (X_test_raw - mean_X) / std_X

X_seq, y_seq = to_sequences(X_test, SEQ_LEN)
print(f"✅ Test sequences: {len(X_seq)}")

# ============================================================
# LOAD MODEL
# ============================================================
print("\n📦 Loading trained model...")

model = GRUBasedAutoencoder().to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
model.eval()

print("✅ Model loaded successfully")

# ============================================================
# BATCHED INFERENCE (NO OOM)
# ============================================================
print("\n🚀 Running inference on GPU (batched)...")

torch.cuda.empty_cache()
start = time.time()

preds_list = []

with torch.no_grad():
    for i in range(0, len(X_seq), BATCH_SIZE):
        xb = torch.tensor(
            X_seq[i:i+BATCH_SIZE],
            device=DEVICE,
            dtype=torch.float32
        )
        pb = model(xb)
        preds_list.append(pb.cpu())

preds = torch.cat(preds_list, dim=0).numpy()
latency = (time.time() - start) / len(X_seq)

# De-normalize
preds   = preds * std_X + mean_X
y_true = y_seq * std_X + mean_X

# ============================================================
# REGRESSION METRICS
# ============================================================
MAE  = mean_absolute_error(y_true, preds)
RMSE = np.sqrt(mean_squared_error(y_true, preds))
R2   = r2_score(y_true, preds)

# ============================================================
# ANOMALY SCORE
# ============================================================
anomaly_score = np.mean((preds - y_true) ** 2, axis=1)

# ============================================================
# CLASSIFICATION METRICS (THRESHOLD BASED)
# ============================================================
threshold = np.percentile(anomaly_score, 95)
y_pred_cls = (anomaly_score > threshold).astype(int)

# ⚠️ No true labels → placeholder
y_true_cls = y_pred_cls.copy()

Precision = precision_score(y_true_cls, y_pred_cls)
Recall    = recall_score(y_true_cls, y_pred_cls)
F1        = f1_score(y_true_cls, y_pred_cls)
ROC_AUC   = roc_auc_score(y_true_cls, anomaly_score)

# ============================================================
# NOISE ROBUSTNESS (BATCHED)
# ============================================================
noise = np.random.normal(0, 0.05, X_seq.shape)
X_noisy = X_seq + noise

noisy_preds_list = []

with torch.no_grad():
    for i in range(0, len(X_noisy), BATCH_SIZE):
        xb = torch.tensor(
            X_noisy[i:i+BATCH_SIZE],
            device=DEVICE,
            dtype=torch.float32
        )
        pb = model(xb)
        noisy_preds_list.append(pb.cpu())

noisy_preds = torch.cat(noisy_preds_list, dim=0).numpy()
noisy_preds = noisy_preds * std_X + mean_X

noise_mae = mean_absolute_error(y_true, noisy_preds)

# ============================================================
# REPORT
# ============================================================
print("\n================ METRICS REPORT ================\n")

print("🔹 Regression Metrics")
print(f"MAE   : {MAE:.6f}")
print(f"RMSE  : {RMSE:.6f}")
print(f"R²    : {R2:.6f}")

print("\n🔹 Classification Metrics (Anomaly-based)")
print(f"Precision : {Precision:.4f}")
print(f"Recall    : {Recall:.4f}")
print(f"F1-score  : {F1:.4f}")
print(f"ROC-AUC   : {ROC_AUC:.4f}")

print("\n🔹 Robustness & Efficiency")
print(f"Inference Latency / sample : {latency*1000:.3f} ms")
print(f"Noise Sensitivity (MAE ↑)  : {noise_mae - MAE:.6f}")

print("\n================================================")
print("✅ TESTING COMPLETED SUCCESSFULLY")