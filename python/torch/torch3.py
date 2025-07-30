import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# 自定义数据集类
class SequenceDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file: 数据文件路径
        """
        raw_data = pd.read_csv(csv_file, header=None)
        self.samples = []
        self.targets = []
        
        # 将数据按8行一组处理
        data = raw_data.values
        for i in range(len(data)//8):
            # 输入序列：前7行
            sample = data[i*8 : i*8+7]
            # 输出目标：第8行
            target = data[i*8+7]
            self.samples.append(sample)
            self.targets.append(target)
            
        # 转换为Tensor
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

# 序列预测模型
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
            nn.Softmax(dim=-1)  # 保证输出和为1
        )
    
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_output = out[:, -1, :]
        return self.fc(last_output)

# 训练参数
config = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 200,
    "test_ratio": 0.2
}

# 加载数据集
dataset = SequenceDataset("D:/program development/hkproject/python/torchdata.csv")

# 划分训练测试集
train_size = int((1 - config["test_ratio"]) * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SequencePredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# 训练循环
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
    
    # 验证循环
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Test Loss: {test_loss/len(test_loader):.4f}\n")

# 保存模型
torch.save(model.state_dict(), "sequence_predictor.pth")

# 测试续写功能
def predict_sequence(csv_path, model_path):
    # 加载模型
    model = SequencePredictor().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 读取测试数据
    test_data = pd.read_csv(csv_path, header=None).values
    if len(test_data) != 7:
        raise ValueError("Input must have exactly 7 rows")
    
    # 转换为tensor并预测
    input_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # 输出结果
    print("\nPredicted next step:")
    print(pd.DataFrame(prediction.cpu().numpy().round(3)))

# 使用示例
predict_sequence("D:/program development/hkproject/python/test_data.csv", "sequence_predictor.pth")
