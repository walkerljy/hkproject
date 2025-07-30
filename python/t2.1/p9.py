import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.samples = [data[i:i+9] for i in range(0, len(data), 9)]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        X = np.vstack([sample[0], sample[1:8]])  # 全局变量 + 7个时间步
        y = sample[8]
        return torch.FloatTensor(X), torch.FloatTensor(y)

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(14, 64, batch_first=True)
        self.fc = nn.Linear(64, 7)
        
    def forward(self, x):
        global_vars = x[:, 0]
        time_series = x[:, 1:]
        seq = torch.cat([global_vars.unsqueeze(1).repeat(1,7,1), time_series], dim=2)
        out, _ = self.lstm(seq)
        pred = self.fc(out[:, -1])
        return pred / pred.sum(dim=1, keepdim=True)

def train():
    data = pd.read_csv("python/t2.1/data.csv", header=None).values
    dataset = TimeSeriesDataset(data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = Predictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(100):
        for X, y in loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), "python/t2.1/sequence_predictor.pth")

def predict():
    model = Predictor()
    model.load_state_dict(torch.load("python/t2.1/sequence_predictor.pth"))
    
    test_data = pd.read_csv("python/t2.1/test.csv", header=None).values
    inputs = torch.FloatTensor([test_data[i:i+8] for i in range(0, len(test_data), 8)])
    
    with torch.no_grad():
        preds = model(inputs).numpy()
    
    pd.DataFrame(preds).to_csv("python/t2.1/testresult.csv", header=False, index=False)

if __name__ == "__main__":
    train()
    predict()
