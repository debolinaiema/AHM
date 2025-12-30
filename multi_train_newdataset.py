import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, forecast_horizon):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * forecast_horizon)
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  
        _, (h_n, _) = self.lstm(x)  
        x = h_n[-1]  
        x = self.fc(x)  
        x = x.view(-1, self.forecast_horizon, self.output_size)  
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
def load_and_preprocess(csv_path, seq_len=60, horizon=10, test_size=0.2):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M')
    groups = df.groupby('Motor_type')  
    all_X, all_y = [], []
    
    for name, group in groups:
        group = group.sort_values('timestamp')
        features = group[['temperature_one', 'temperature_two', 'vibration_x', 'vibration_y', 'vibration_z']].values
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        X, y = [], []
        for i in range(len(features) - seq_len - horizon + 1):
            X.append(features[i:i+seq_len])  
            y.append(features[i+seq_len:i+seq_len+horizon]) 
        all_X.extend(X)
        all_y.extend(y)
    
    all_X, all_y = np.array(all_X), np.array(all_y)
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=test_size, shuffle=False)
    return X_train, X_test, y_train, y_test
def main():
    input_size = 5  
    hidden_size = 128
    num_layers = 2
    output_size = 5  
    forecast_horizon = 10
    learning_rate = 0.001
    epochs = 50
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = 'new dataset ahm.csv' 
    X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = CNNLSTM(input_size, hidden_size, num_layers, output_size, forecast_horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}')
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            test_loss += criterion(output, y_batch).item()
    print(f'Test Loss: {test_loss / len(test_loader):.4f}')
    torch.save(model.state_dict(), 'cnn_lstm_forecast.pth')
    with torch.no_grad():
        last_seq = torch.tensor(X_test[-1:], dtype=torch.float32).to(device)
        forecast = model(last_seq)
        print("Forecasted values (vibration_x/y/z, temp_one/two):", forecast.cpu().numpy())

if __name__ == '__main__':
    main()