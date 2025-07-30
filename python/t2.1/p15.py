import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 配置参数
sequence_length = 7  # 除了全局变量外的时间步数
num_features = 7  # 每行的列数
batch_size = 16
epochs = 50
learning_rate = 0.001

# 自定义Dataset
class SequenceDataset(Dataset):
    def __init__(self, csv_file, is_train=True):
        self.data = pd.read_csv(csv_file, header=None).values
        # 第一行为全局变量
        self.global_vars = self.data[0, :]
        # 数据部分
        if is_train:
            # 训练数据：二到八行作为输入，九行为目标
            self.inputs = self.data[1:8, :]  # 7个时间步
            self.target = self.data[8, :]      # 目标
        else:
            # 测试数据
            self.inputs = self.data[1:8, :]
        # 转换为float
        self.inputs = self.inputs.astype(np.float32)
        if is_train:
            self.target = self.target.astype(np.float32)
            # 等比例缩放目标，使最大值为1
            max_val = np.max(self.target)
            if max_val != 0:
                self.target = self.target / max_val
            self.target = self.target.reshape(-1)  # 转为一维
        # 转换为tensor
        self.inputs = torch.tensor(self.inputs)
        if is_train:
            self.target = torch.tensor(self.target)

    def __len__(self):
        return 1  # 每个文件只有一组数据

    def __getitem__(self, idx):
        if hasattr(self, 'target'):
            return self.inputs, self.target
        else:
            return self.inputs

# 定义模型（示例：简单的LSTM或GRU）
class SequencePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=7):
        super(SequencePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x形状：batch_size, seq_len, features
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # 取最后时间步输出
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# 训练流程
def train():
    train_dataset = SequenceDataset('python/t2.1/data.csv', is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SequencePredictor(input_size=num_features).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            targets = targets.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # 输入应为(batch, seq_len, features)
            inputs = inputs.unsqueeze(0)  # 添加batch维度
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), 'python/t2.1/sequence_predictor.pth')
    print("模型已保存！")

# 验证流程
def validate():
    test_df = pd.read_csv('python/t2.1/test.csv', header=None)
    test_inputs = test_df.values.astype(np.float32)
    # 只用二到八行数据
    test_inputs = test_inputs[1:8, :]
    test_inputs = torch.tensor(test_inputs).unsqueeze(0)  # 添加batch维度
    # 全局变量如果需要可以用
    # 载入模型

train()
validate()