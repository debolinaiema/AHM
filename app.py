import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class GRUBasedAutoencoder(nn.Module):
    def __init__(self, seq_length=60, n_features=3, hidden_dim1=64, hidden_dim2=32):
        super(GRUBasedAutoencoder, self).__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.gru1 = nn.GRU(input_size=n_features, hidden_size=hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.gru2 = nn.GRU(input_size=hidden_dim1, hidden_size=hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim2, 16),
            nn.ReLU(),
            nn.Linear(16, seq_length * n_features),
            nn.Unflatten(1, (seq_length, n_features))
        )
    def forward(self, x):
        out1, _ = self.gru1(x)
        last_t1 = out1[:, -1, :]
        last_t1 = self.dropout1(last_t1)
        out2, _ = self.gru2(last_t1.unsqueeze(1))
        last_t2 = out2.squeeze(1)
        last_t2 = self.dropout2(last_t2)
        decoded = self.decoder(last_t2)
        return decoded
def train_model(df_path, model_path, seq_length=60, epochs=50, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(df_path)
    vibes = df[['vibration_x', 'vibration_y', 'vibration_z']].values.astype(np.float32)
    scaler = StandardScaler()
    vibes_scaled = scaler.fit_transform(vibes)
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)
    sequences = create_sequences(vibes_scaled, seq_length)
    dataset = TensorDataset(torch.tensor(sequences).to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GRUBasedAutoencoder(seq_length=seq_length, n_features=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for data in loader:
            inputs = data[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss/len(loader):.4f}")
    model_save_path = model_path.replace('.pkl', '.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'seq_length': seq_length,
        'n_features': 3,
        'hidden_dim1': 64,
        'hidden_dim2': 32
    }, model_save_path)
    scaler_save_path = model_path.replace('.pkl', '_scaler.pkl')
    joblib.dump(scaler, scaler_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Scaler saved to {scaler_save_path}")
def train_classifier(phase3_path, gearbox_path, classifier_path):
    df_phase3 = pd.read_csv(phase3_path)
    df_gearbox = pd.read_csv(gearbox_path)
    df_phase3['label'] = 0
    df_gearbox['label'] = 1
    df = pd.concat([df_phase3, df_gearbox], ignore_index=True)
    features = ['temperature_one', 'temperature_two']
    X = df[features].values
    y = df['label'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression()
    clf.fit(X_scaled, y)
    joblib.dump({
        'clf': clf,
        'scaler': scaler,
        'features': features
    }, classifier_path)
    print(f"Classifier saved to {classifier_path}")
if __name__ == "__main__":
    train_model(
        'E:\\AHM_multivariate\\data\\Duo_002_24-6-25_to_29-7-25.csv',
        'E:\\AHM_multivariate\\3phase.pkl'
    )
    train_model(
        'E:\\AHM_multivariate\\data\\Duo_005_6-8-25_to_13-8-25.csv',
        'E:\\AHM_multivariate\\gearbox.pkl'
    )
    train_classifier(
        'E:\\AHM_multivariate\\data\\Duo_002_24-6-25_to_29-7-25.csv',
        'E:\\AHM_multivariate\\data\\Duo_005_6-8-25_to_13-8-25.csv',
        'E:\\AHM_multivariate\\classifier.pkl'
    )