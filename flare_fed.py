import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple
assert torch.cuda.is_available(), "CUDA not available"
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True

print("🔥 Using GPU:", torch.cuda.get_device_name(0))
FILE1 = r"D:\AHM_multivariate\AHM_multivariate\data\Duo_001_24-6-25_to_31-7-25.csv"
FILE2 = r"D:\AHM_multivariate\AHM_multivariate\data\Duo_007_24-6-25_to_06-8-25.csv"

SEQ_LEN = 120
BATCH_SIZE = 64
LR = 1e-3
LOCAL_EPOCHS = 5
NUM_ROUNDS = 40
VAL_SIZE = 10000

MODEL_DIR = "models_gpu"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "temperature_one",
    "temperature_two",
    "vibration_x",
    "vibration_y",
    "vibration_z"
]
class GRUBasedAutoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.gru1 = nn.GRU(
            n_features, 128, num_layers=2,
            batch_first=True, dropout=0.1
        )
        self.gru2 = nn.GRU(
            128, 64, num_layers=2,
            batch_first=True, dropout=0.1
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, n_features)
        )

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return self.decoder(x[:, -1])

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    X = df[FEATURES].values.astype(np.float32)
    return X

X1_raw = load_csv(FILE1)
X2_raw = load_csv(FILE2)
mean_X = np.mean(np.vstack([X1_raw, X2_raw]), axis=0)
std_X = np.std(np.vstack([X1_raw, X2_raw]), axis=0)
std_X[std_X == 0] = 1.0

X1 = (X1_raw - mean_X) / std_X
X2 = (X2_raw - mean_X) / std_X

def to_sequences(X):
    seq_X, seq_y = [], []
    for i in range(len(X) - SEQ_LEN):
        seq_X.append(X[i:i+SEQ_LEN])
        seq_y.append(X[i+SEQ_LEN])
    return np.array(seq_X), np.array(seq_y)

X1_seq, y1_seq = to_sequences(X1)
X2_seq, y2_seq = to_sequences(X2)

def split_val(X, y):
    X_train = X[:-VAL_SIZE]
    y_train = y[:-VAL_SIZE]
    X_val = X[-VAL_SIZE:]
    y_val = y[-VAL_SIZE:]
    return X_train, y_train, X_val, y_val

X1_tr, y1_tr, X1_val, y1_val = split_val(X1_seq, y1_seq)
X2_tr, y2_tr, X2_val, y2_val = split_val(X2_seq, y2_seq)

def train_local(model, X_np, y_np):
    model = model.to(DEVICE)
    model.train()

    X = torch.tensor(X_np, device=DEVICE)
    y = torch.tensor(y_np, device=DEVICE)

    loss_fn = nn.SmoothL1Loss(beta=0.5).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    for _ in range(LOCAL_EPOCHS):
        idx = torch.randperm(len(X), device=DEVICE)
        Xs, ys = X[idx], y[idx]

        for i in range(0, len(Xs), BATCH_SIZE):
            xb = Xs[i:i+BATCH_SIZE]
            yb = ys[i:i+BATCH_SIZE]

            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
 
def evaluate(model, X_np, y_np):
    model.eval()
    X = torch.tensor(X_np, device=DEVICE)
    y = torch.tensor(y_np, device=DEVICE)

    with torch.no_grad():
        preds = model(X)

    preds = preds.cpu().numpy() * std_X + mean_X
    y = y_np * std_X + mean_X
    return float(np.mean((preds - y) ** 2))
class Client:
    def __init__(self, X_tr, y_tr, X_val, y_val):
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_val = X_val
        self.y_val = y_val
        self.model = GRUBasedAutoencoder(X_tr.shape[2]).to(DEVICE)

    def get_params(self):
        return [p.detach().clone() for p in self.model.state_dict().values()]

    def set_params(self, params):
        sd = self.model.state_dict()
        for k, p in zip(sd.keys(), params):
            sd[k].copy_(p)
        self.model.load_state_dict(sd)

    def fit(self):
        train_local(self.model, self.X_tr, self.y_tr)
        return self.get_params(), len(self.X_tr)

    def evaluate(self):
        return evaluate(self.model, self.X_val, self.y_val), len(self.X_val)
clients = [
    Client(X1_tr, y1_tr, X1_val, y1_val),
    Client(X2_tr, y2_tr, X2_val, y2_val)
]

global_params = clients[0].get_params()

for r in range(1, NUM_ROUNDS + 1):
    print(f"\n🔥 Round {r}/{NUM_ROUNDS}")
    updates = []

    for i, c in enumerate(clients):
        c.set_params(global_params)
        params, n = c.fit()
        updates.append((params, n))
        print(f" Client {i} trained on {n}")
    total = sum(n for _, n in updates)
    new_params = []

    for i in range(len(global_params)):
        p = sum(
            updates[j][0][i] * (updates[j][1] / total)
            for j in range(len(updates))
        )
        new_params.append(p)

    global_params = new_params

    total_loss, total_n = 0, 0
    for i, c in enumerate(clients):
        c.set_params(global_params)
        mse, n = c.evaluate()
        print(f" Client {i} Val MSE = {mse:.6f}")
        total_loss += mse * n
        total_n += n

    print(f" 🌍 Global MSE = {total_loss / total_n:.6f}")

torch.save(
    dict(zip(clients[0].model.state_dict().keys(), global_params)),
    os.path.join(MODEL_DIR, "global_gpu_model.pth")
)

print("\n✅ TRAINING FINISHED — PURE GPU USED")