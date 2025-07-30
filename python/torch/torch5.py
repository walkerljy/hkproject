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
        for i in range(len(data)//8):
            sample = data[i*8 : i*8+7]
            target = data[i*8+7]
            self.samples.append(sample)
            self.targets.append(target)
            
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

class SequencePredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=5, num_layers=2):
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
        out, (h_n, c_n) = self.lstm(x)
        last_output = out[:, -1, :]
        return self.fc(last_output)

config = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 200,
    "test_ratio": 0.2
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = SequenceDataset("D:/program development/hkproject/python/torchdata.csv")
train_size = int((1 - config["test_ratio"]) * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

model = SequencePredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# 修改后的训练循环
for epoch in range(config["num_epochs"]):
    model.train()
    train_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
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
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
    
    # 每20个epoch时展示详细测试结果
    if (epoch+1) % 20 == 0:
        # 获取测试集第一个样本用于展示
        sample_input, sample_target = next(iter(test_loader))
        sample_input = sample_input.to(device)
        
        # 添加no_grad上下文管理器
        with torch.no_grad():
            sample_output = model(sample_input)
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Test Loss: {test_loss/len(test_loader):.4f}")
        
        # 打印样本预测结果
        print("\nExample Test Case:")
        print("Input Sequence (7 steps):")
        print(sample_input[0].cpu().numpy().round(3))
        print("\nPredicted Next Step:")
        print(sample_output[0].cpu().numpy().round(3))  # 现在可以安全转换
        print("\nTrue Next Step:")
        print(sample_target[0].numpy().round(3))
        print("-"*50 + "\n")


torch.save(model.state_dict(), "sequence_predictor.pth")

def predict_sequence(csv_path, model_path, predict_steps=10):
    # 加载模型
    model = SequencePredictor().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 读取初始序列
    initial_data = pd.read_csv(csv_path, header=None).values
    if len(initial_data) != 7:
        raise ValueError("Initial data must have exactly 7 rows")
    
    # 初始化预测序列
    current_sequence = torch.tensor(initial_data, dtype=torch.float32).unsqueeze(0).to(device)
    predictions = []
    
    with torch.no_grad():
        for _ in range(predict_steps):
            # 预测下一个时间步
            pred = model(current_sequence)
            predictions.append(pred.cpu().numpy()[0].round(3))
            
            # 更新输入序列：移除最早时间步，添加新预测
            updated_sequence = torch.cat([
                current_sequence[:, 1:, :],  # 保留后6个时间步
                pred.unsqueeze(1)            # 添加新预测作为第7个时间步
            ], dim=1)
            
            current_sequence = updated_sequence
    
    # 输出结果
    print("\nTime Step | Predicted Values")
    print("-----------------------------")
    for i, pred in enumerate(predictions, 1):
        print(f"Step {i:2d}: {pred} (Sum: {np.sum(pred):.2f})")

# 使用示例（预测未来10个时间步）
predict_sequence("D:/program development/hkproject/python/test_data.csv", 
                "sequence_predictor.pth",
                predict_steps=10)
