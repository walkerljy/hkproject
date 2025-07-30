import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# 自定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path):
        raw_data = pd.read_csv(csv_path, header=None)
        
        # 计算有效样本数（自动截断多余行）
        total_rows = len(raw_data)
        self.num_samples = total_rows // 9
        valid_rows = self.num_samples * 9
        
        # 截取有效数据行
        self.data = torch.tensor(raw_data.values[:valid_rows], dtype=torch.float32).view(self.num_samples, 9, 7)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'global_vars': sample[0],         # 全局变量 (7,)
            'sequence': sample[1:8],          # 输入序列 (7,7)
            'target': sample[8]               # 目标值 (7,)
        }

# 序列预测模型
class SequencePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64 + 7, 7)  # 合并LSTM输出和全局变量
        
    def forward(self, seq, global_vars):
        lstm_out, _ = self.lstm(seq)          # (batch_size, 7, 64)
        lstm_last = lstm_out[:, -1, :]        # 取最后一个时间步 (batch_size, 64)
        combined = torch.cat([lstm_last, global_vars], dim=1)  # (batch_size, 71)
        output = self.fc(combined)            # (batch_size, 7)
        
        # 等比例归一化（保证和为1）
        return output / output.sum(dim=1, keepdim=True).clamp(min=1e-8)

# 训练函数
def train():
    # 初始化数据集和数据加载器
    dataset = TimeSeriesDataset("python/t2.1/data.csv")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 初始化模型和优化器
    model = SequencePredictor()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 训练循环
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        
        for batch in loader:
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(batch['sequence'], batch['global_vars'])
            
            # 计算损失
            loss = criterion(outputs, batch['target'])
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")
    
    # 保存模型
    torch.save(model.state_dict(), "python/t2.1/sequence_predictor.pth")

# 测试函数
def test():
    # 加载训练好的模型
    model = SequencePredictor()
    model.load_state_dict(torch.load("python/t2.1/sequence_predictor.pth"))
    model.eval()
    
    # 加载测试数据
    raw_test = pd.read_csv("python/t2.1/test.csv", header=None)
    num_samples = len(raw_test) // 8
    test_data = torch.tensor(raw_test.values, dtype=torch.float32).view(num_samples, 8, 7)
    
    # 进行预测
    predictions = []
    with torch.no_grad():
        for sample in test_data:
            # 提取输入数据
            global_vars = sample[0].unsqueeze(0)  # (1,7)
            sequence = sample[1:].unsqueeze(0)    # (1,7,7)
            
            # 预测结果
            pred = model(sequence, global_vars)
            predictions.append(pred.numpy()[0])
    
    # 保存预测结果
    pd.DataFrame(predictions).to_csv("python/t2.1/testresult.csv", header=False, index=False)

if __name__ == "__main__":
    train()
    test()
