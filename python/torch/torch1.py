import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# 自定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, seq_length=8, num_features=5):
        """
        Args:
            csv_file (string): CSV文件路径
            seq_length (int): 时间序列长度（默认8个时间点）
            num_features (int): 特征维度（默认5个特征）
        """
        raw_data = pd.read_csv(csv_file)
        
        # 数据预处理：转换为三维张量 (num_samples, seq_length, num_features)
        self.data = []
        for _, row in raw_data.iterrows():
            # 假设数据排列方式为：时间点0的5个特征，时间点1的5个特征...时间点7的5个特征
            sequence = row.values.reshape(seq_length, num_features)
            self.data.append(sequence)
        
        self.data = np.array(self.data)
        
        # 添加归一化的时间特征（0到1均匀分布）
        self.timesteps = np.tile(np.linspace(0, 1, seq_length), (len(self.data), 1))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 输入：时间步（归一化到0-1）
        # 输出：对应的五维特征
        return (
            torch.tensor(self.timesteps[idx], dtype=torch.float32),
            torch.tensor(self.data[idx], dtype=torch.float32)
        )

# 模型定义
class EvolutionModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)  # 确保输出和为1
        )
    
    def forward(self, x):
        # x形状: (batch_size, seq_length)
        # 转换为: (batch_size, seq_length, input_size)
        x = x.unsqueeze(-1)
        # 输出形状: (batch_size, seq_length, output_size)
        return self.net(x)

# 训练参数
config = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 500,
    "train_ratio": 0.8
}

# 数据准备
full_dataset = TimeSeriesDataset("your_data.csv")

# 数据集分割
train_size = int(config["train_ratio"] * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EvolutionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# 训练循环
for epoch in range(config["num_epochs"]):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    if (epoch+1) % 50 == 0:
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {avg_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "evolution_model.pth")

# 测试验证
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        test_loss += criterion(outputs, targets).item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Loss: {avg_test_loss:.4f}")

    # 示例输出对比
    sample_input, sample_target = test_dataset[0]
    sample_input = sample_input.unsqueeze(0).to(device)
    prediction = model(sample_input).cpu().numpy()[0]
    
    print("\nSample Prediction vs Actual:")
    print("Time\tPredicted Features\t\tActual Features")
    for t in range(len(sample_input[0])):
        pred = prediction[t].round(3)
        actual = test_dataset[0][1][t].numpy().round(3)
        print(f"{sample_input[0][t].item():.2f}\t{pred}\t{actual}")
