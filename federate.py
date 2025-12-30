import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import pickle

FILE1 = r"C:\Users\user\Downloads\AHM_multivariate\data\Duo_001_24-6-25_to_31-7-25.csv"
FILE2 = r"C:\Users\user\Downloads\AHM_multivariate\data\Duo_007_24-6-25_to_06-8-25.csv"
SAVE_DIR = r"C:\Users\user\Downloads\AHM_multivariate\models"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_ROUNDS = 5
LOCAL_EPOCHS = 5
LR = 0.0005
BATCH_SIZE = 512

def encode_features(df, feature_cols):
    encoded_df = df.copy()
    for col in feature_cols:
        if encoded_df[col].dtype == "object":
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            print(f"   🔠 Encoded string column '{col}' → numeric labels (0–{encoded_df[col].max()})")
    return encoded_df

def load_data(path):
    print(f"\n📂 Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f" - Shape before selection: {df.shape}")

    df.columns = [c.lower() for c in df.columns]

    feature_cols = ["id", "sensor_id_fk", "device_id"]
    target_cols = ["temperature_one", "temperature_two", "vibration_x", "vibration_y", "vibration_z"]

    missing_targets = [c for c in target_cols if c not in df.columns]
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    df = encode_features(df, feature_cols)

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    print(f" - Features shape: {X.shape}, Targets shape: {y.shape}")
    return X, y


X1, y1 = load_data(FILE1)
X2, y2 = load_data(FILE2)
print("\n✅ Data loaded successfully.")

mean_X = np.mean(np.vstack([X1, X2]), axis=0)
std_X = np.std(np.vstack([X1, X2]), axis=0)
std_X[std_X == 0] = 1.0
X1 = (X1 - mean_X) / std_X
X2 = (X2 - mean_X) / std_X

mean_y = np.mean(np.vstack([y1, y2]), axis=0)
std_y = np.std(np.vstack([y1, y2]), axis=0)
std_y[std_y == 0] = 1.0
y1 = (y1 - mean_y) / std_y
y2 = (y2 - mean_y) / std_y

print("✅ Normalized both inputs and targets successfully.")

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


INPUT_DIM = X1.shape[1]
OUTPUT_DIM = y1.shape[1]
global_model = TinyModel(INPUT_DIM, OUTPUT_DIM)
print(f"\n✅ Model initialized: Input={INPUT_DIM}, Output={OUTPUT_DIM}")

def make_loader(X, y, batch=BATCH_SIZE):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                    torch.tensor(y, dtype=torch.float32)),
                      batch_size=batch, shuffle=True)

loader1 = make_loader(X1, y1)
loader2 = make_loader(X2, y2)
def train_local(model, loader, name):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(LOCAL_EPOCHS):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   {name} epoch {epoch+1}/{LOCAL_EPOCHS} loss={total_loss/len(loader):.6f}")
    return model


def average_weights(weights_list, sizes, client_names):
    print("\n🔢 Weight Aggregation Details:")
    total = sum(sizes)
    avg = {k: torch.zeros_like(v) for k, v in weights_list[0].items()}

    for i, (w, s) in enumerate(zip(weights_list, sizes)):
        weight_factor = s / total
        print(f" - {client_names[i]} contributes {s}/{total} = {weight_factor:.4f}")
        for k in avg.keys():
            avg[k] += w[k] * weight_factor
            if "weight" in k:
                print(f"     Layer {k}: weighted add ×{weight_factor:.4f}")

    print("✅ All client weights aggregated successfully.")
    return avg


def evaluate(model, X, y, label):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).numpy()
    mse = np.mean((preds - y) ** 2)
    print(f"📊 {label} MSE: {mse:.6f}")
    return mse
print("\n🚀 Starting Federated Learning...")

clients = [
    ("client1", loader1, X1.shape[0]),
    ("client2", loader2, X2.shape[0])
]

for rnd in range(1, NUM_ROUNDS + 1):
    print("\n==========================")
    print(f"   ROUND {rnd}/{NUM_ROUNDS}")
    print("==========================")

    local_weights, local_sizes, client_names = [], [], []

    for name, loader, size in clients:
        print(f"\n→ Training on {name} ({size} samples)")
        local_model = TinyModel(INPUT_DIM, OUTPUT_DIM)
        local_model.load_state_dict(global_model.state_dict())
        local_model = train_local(local_model, loader, name)
        local_weights.append(local_model.state_dict())
        local_sizes.append(size)
        client_names.append(name)

        path = os.path.join(SAVE_DIR, f"{name}_round{rnd}_weights.pkl")
        with open(path, "wb") as f:
            pickle.dump(local_model.state_dict(), f)
        print(f"💾 Saved local model to {path}")

    new_state = average_weights(local_weights, local_sizes, client_names)
    global_model.load_state_dict(new_state)

    global_path = os.path.join(SAVE_DIR, f"global_round{rnd}_weights.pkl")
    with open(global_path, "wb") as f:
        pickle.dump(global_model.state_dict(), f)
    print(f"\n🌍 Aggregated and saved global model to {global_path}")

    evaluate(global_model, X1, y1, "Client1")
    evaluate(global_model, X2, y2, "Client2")

print("\n✅ Federated Learning completed successfully.")
final_path = os.path.join(SAVE_DIR, "global_final_weights.pkl")
with open(final_path, "wb") as f:
    pickle.dump(global_model.state_dict(), f)
print(f"\n Final global model saved to: {final_path}")