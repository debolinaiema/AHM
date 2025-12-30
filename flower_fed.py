import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
assert torch.cuda.is_available(), "❌ CUDA GPU not found!"

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print("🔥 FORCED GPU MODE:", torch.cuda.get_device_name(0))
FILES = [
    r"D:\AHM_multivariate\data\Duo_001_24-6-25_to_31-7-25.csv",
    r"D:\AHM_multivariate\data\Duo_007_24-6-25_to_06-8-25.csv",
]
SEQ_LEN = 120
BATCH_SIZE = 128
LOCAL_EPOCHS = 4
NUM_ROUNDS = 40 
VAL_SIZE = 8000
LR = 1e-3
MODEL_DIR = "models_pure_gpu"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "temperature_one",
    "temperature_two",
    "vibration_x",
    "vibration_y",
    "vibration_z"
]
class GRUBasedAutoencoder(nn.Module):
    def __init__(self, n_features=5):
        super().__init__()
        self.gru1 = nn.GRU(n_features, 128, num_layers=2, batch_first=True, dropout=0.1)
        self.gru2 = nn.GRU(128, 64, num_layers=2, batch_first=True, dropout=0.1)
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

def load_clients():
    all_data = []

    for f in FILES:
        df = pd.read_csv(f)
        df.columns = df.columns.str.lower()
        X = df[FEATURES].values.astype(np.float32)
        all_data.append(X)

    X_all = np.vstack(all_data)
    mean = X_all.mean(axis=0)
    std = X_all.std(axis=0)
    std[std == 0] = 1.0

    clients = []
    for X_raw in all_data:
        X = (X_raw - mean) / std
        seq_X, seq_y = [], []

        for i in range(len(X) - SEQ_LEN):
            seq_X.append(X[i:i+SEQ_LEN])
            seq_y.append(X[i+SEQ_LEN])

        seq_X = torch.tensor(np.array(seq_X), device=DEVICE)
        seq_y = torch.tensor(np.array(seq_y), device=DEVICE)

        X_train = seq_X[:-VAL_SIZE]
        y_train = seq_y[:-VAL_SIZE]
        X_val   = seq_X[-VAL_SIZE:]
        y_val   = seq_y[-VAL_SIZE:]

        clients.append((X_train, y_train, X_val, y_val))

    return clients
clients_data = load_clients()

def make_loader(X, y):
    ds = TensorDataset(X, y)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False, 
    )

class Client:
    def __init__(self, data):
        self.X_train, self.y_train, self.X_val, self.y_val = data
        self.model = GRUBasedAutoencoder().cuda()

    def get_params(self):
        return [p.detach().clone() for p in self.model.state_dict().values()]

    def set_params(self, params):
        sd = self.model.state_dict()
        for k, p in zip(sd.keys(), params):
            sd[k].copy_(p)
        self.model.load_state_dict(sd)

    def fit(self):
        loader = make_loader(self.X_train, self.y_train)
        opt = optim.AdamW(self.model.parameters(), lr=LR)
        loss_fn = nn.SmoothL1Loss()

        self.model.train()
        for _ in range(LOCAL_EPOCHS):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()

        return self.get_params(), len(self.X_train)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X_val)
            loss = ((preds - self.y_val) ** 2).mean()
        return loss.item(), len(self.X_val)

clients = [Client(d) for d in clients_data]
global_model = clients[0].model

for r in range(1, NUM_ROUNDS + 1):
    print(f"\n🔥 Round {r}/{NUM_ROUNDS}")

    global_params = clients[0].get_params()
    updates = []

    for i, c in enumerate(clients):
        c.set_params(global_params)
        params, n = c.fit()
        updates.append((params, n))
        print(f" Client {i} trained on {n}")

    total = sum(n for _, n in updates)
    new_params = []

    for i in range(len(global_params)):
        p = sum(updates[j][0][i] * (updates[j][1] / total) for j in range(len(updates)))
        new_params.append(p)

    global_model.load_state_dict(
        dict(zip(global_model.state_dict().keys(), new_params))
    )

    total_loss, total_n = 0, 0
    for i, c in enumerate(clients):
        c.set_params(new_params)
        mse, n = c.evaluate()
        print(f" Client {i} Val MSE = {mse:.6f}")
        total_loss += mse * n
        total_n += n

    print(f" 🌍 Global MSE = {total_loss / total_n:.6f}")

torch.save(global_model.state_dict(), os.path.join(MODEL_DIR, "global_gpu_only.pth"))
print("\n✅ PURE GPU TRAINING FINISHED")