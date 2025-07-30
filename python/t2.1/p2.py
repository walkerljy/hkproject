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
        # 每组数据包含9行（1全局变量 + 8时间步）
        num_groups = len(data) // 9
        
        for i in range(num_groups):
            # 提取全局变量（第一行）
            global_vars = data[i*9]
            # 提取时间序列（后8行）
            time_series = data[i*9+1 : (i+1)*9]
            
            # 输入为前7个时间步（每个拼接全局变量）
            inputs = [np.concatenate([ts, global_vars]) for ts in time_series[:7]]
            # 目标为第8个时间步（原始数据的第9行）
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
            nn.Softmax(dim=-1)  # 保持和为1的特性
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        last_output = out[:, -1, :]
        return self.fc(last_output)

config = {
    "batch_size": 128,
    "learning_rate": 0.001,
    "num_epochs": 800,
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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# 训练循环
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
    
    # 验证阶段
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
    
    # 每20个epoch打印结果
    if (epoch+1) % 20 == 0:
        sample_input, sample_target = next(iter(test_loader))
        sample_input = sample_input.to(device)
        
        with torch.no_grad():
            sample_output = model(sample_input)
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Test Loss: {test_loss/len(test_loader):.4f}")
        
        # 打印样本信息（显示前7个特征）
        print("\nSample Input (first 7 features):")
        print(sample_input[0, :, :7].cpu().numpy().round(3))
        print("\nPredicted (first 7 features):")
        print(sample_output[0, :7].cpu().numpy().round(3))
        print("Actual (first 7 features):")
        print(sample_target[0, :7].numpy().round(3))
        print("-"*50)

torch.save(model.state_dict(), "python/t2.1/sequence_predictor.pth")

def predict_sequence(csv_path, model_path, predict_steps=10):
    # 加载模型
    model = SequencePredictor().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 读取数据（1全局变量 + 7时间步）
    raw_data = pd.read_csv(csv_path, header=None).values
    if len(raw_data) != 8:
        raise ValueError("Input should contain 8 rows (1 global + 7 time steps)")
    
    global_vars = raw_data[0]
    initial_steps = raw_data[1:8]
    
    # 构建初始序列（每个时间步拼接全局变量）
    current_sequence = np.array([np.concatenate([ts, global_vars]) for ts in initial_steps])
    current_sequence = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(predict_steps):
            # 预测下一个时间步
            pred = model(current_sequence)
            pred_np = pred[0].cpu().numpy()
            predictions.append(pred_np)
            
            # 构建新时间步（预测值 + 全局变量）
            new_step = np.concatenate([pred_np, global_vars])
            
            # 更新序列（滚动更新）
            updated_sequence = torch.cat([
                current_sequence[:, 1:, :],  # 保留后6个时间步
                torch.tensor(new_step, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            ], dim=1)
            
            current_sequence = updated_sequence
    
    # 打印预测结果
    print("\nPrediction Timeline:")
    for i, pred in enumerate(predictions, 1):
        print(f"Step {i}: {pred.round(3)} (Sum: {np.sum(pred):.2f})")

# 使用示例
predict_sequence("python/t2.1/test.csv", 
                "python/t2.1/sequence_predictor.pth",
                predict_steps=10)
