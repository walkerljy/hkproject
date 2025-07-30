import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

class SequenceDataset(Dataset):
    def __init__(self, csv_file):
        raw_data = pd.read_csv(csv_file, header=None)
        self.samples = []
        self.targets = []
        
        data = raw_data.values
        num_groups = len(data) // 9
        
        for i in range(num_groups):
            global_vars = data[i*9]
            time_series = data[i*9+1 : (i+1)*9]
            
            inputs = [np.concatenate([ts, global_vars]) for ts in time_series[:7]]
            target = time_series[7]
            
            self.samples.append(inputs)
            self.targets.append(target)
            
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

class SequencePredictor(nn.Module):
    def __init__(self, input_size=14, hidden_size=128, output_size=7, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        last_output = out[:, -1, :]
        return self.fc(last_output)

config = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 400,
    "test_ratio": 0.2
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化数据集
dataset = SequenceDataset("python/t2.1/data.csv")
train_size = int((1 - config["test_ratio"]) * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

model = SequencePredictor().to(device)
criterion = nn.MSELoss(reduction='sum')  # 关键修改点
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# 修改后的训练循环
for epoch in range(config["num_epochs"]):
    model.train()
    train_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
    
    # 修改结果显示方式
    if (epoch+1) % 20 == 0:
        avg_train = train_loss / (len(train_loader.dataset) * 7)
        avg_test = test_loss / (len(test_loader.dataset) * 7)
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Train SSE: {train_loss:.4f} | Avg per feature: {avg_train:.6f}")
        print(f"Test SSE: {test_loss:.4f} | Avg per feature: {avg_test:.6f}")
        
        # 样本展示
        sample_input, sample_target = next(iter(test_loader))
        sample_input = sample_input.to(device)
        
        with torch.no_grad():
            sample_output = model(sample_input)
        
        print("\nSample Prediction:")
        print("Actual:", sample_target[0].numpy().round(3))
        print("Predicted:", sample_output[0].cpu().numpy().round(3))
        print("-"*50)

torch.save(model.state_dict(), "python/t2.1/sequence_predictor_3.pth")

def predict_sequence(csv_path, model_path, predict_steps=10):
    model = SequencePredictor().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    raw_data = pd.read_csv(csv_path, header=None).values
    if len(raw_data) != 8:
        raise ValueError("Input should contain 8 rows (1 global + 7 time steps)")
    
    global_vars = raw_data[0]
    initial_steps = raw_data[1:8]
    
    current_sequence = np.array([np.concatenate([ts, global_vars]) for ts in initial_steps])
    current_sequence = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(predict_steps):
            pred = model(current_sequence)
            pred_np = pred[0].cpu().numpy()
            predictions.append(pred_np)
            
            new_step = np.concatenate([pred_np, global_vars])
            
            updated_sequence = torch.cat([
                current_sequence[:, 1:, :],
                torch.tensor(new_step, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            ], dim=1)
            
            current_sequence = updated_sequence
    
    # 显示预测的平方和
    print("\nPrediction Details:")
    for i, pred in enumerate(predictions, 1):
        sse = np.sum((pred - raw_data[7])**2)  # 与最后一个已知值的差异
        print(f"Step {i}: {pred.round(3)} | SSE: {sse:.4f}")

# 使用示例
predict_sequence("python/t2.1/test.csv", 
                "python/t2.1/sequence_predictor_3.pth",
                predict_steps=10)
