import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类
class SequenceDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path, header=None).values
        self.samples = data.reshape(-1, 9, 7)  # 重塑为(N, 9, 7)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 第一行：全局变量
        global_vars = torch.tensor(sample[0], dtype=torch.float32)
        # 第二行到第八行：输入序列 (7个时间步)
        sequence = torch.tensor(sample[1:8], dtype=torch.float32)
        # 第九行：目标输出 (需等比例缩放)
        target = torch.tensor(sample[8], dtype=torch.float32)
        return global_vars, sequence, target

# 等比例缩放层 (替代softmax)
class NormalizeScale(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        # 确保输出和为1 (等比例缩放)
        return x / (x.sum(dim=-1, keepdim=True) + self.eps)

# LSTM序列预测模型
class SequencePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, global_size=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size + global_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 7)
        self.normalize = NormalizeScale()
        
    def forward(self, global_vars, sequences):
        # 将全局变量复制到每个时间步
        global_expanded = global_vars.unsqueeze(1).repeat(1, sequences.size(1), 1)
        # 拼接序列数据和全局变量
        combined = torch.cat((sequences, global_expanded), dim=-1)
        # LSTM处理
        lstm_out, _ = self.lstm(combined)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        # 全连接层
        fc_out = self.fc(last_output)
        # 等比例缩放输出
        return self.normalize(fc_out)

# 训练函数
def train_model():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建数据集和数据加载器
    train_dataset = SequenceDataset("python/t2.1/data.csv")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 初始化模型
    model = SequencePredictor(input_size=7, hidden_size=64).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0.0
        for global_vars, sequences, targets in train_loader:
            global_vars, sequences, targets = (
                global_vars.to(device),
                sequences.to(device),
                targets.to(device)
            )
            
            # 前向传播
            outputs = model(global_vars, sequences)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印每10个epoch的损失
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.6f}")
    
    # 保存模型
    torch.save(model.state_dict(), "python/t2.1/sequence_predictor.pth")
    return model

# 测试和预测函数
def test_and_predict(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载测试数据
    test_data = pd.read_csv("python/t2.1/test.csv", header=None).values
    # 重塑为(M, 8, 7)
    test_samples = test_data.reshape(-1, 8, 7)
    
    # 加载模型
    model = SequencePredictor(input_size=7, hidden_size=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for sample in test_samples:
            # 第一行：全局变量
            global_vars = torch.tensor(sample[0], dtype=torch.float32).unsqueeze(0).to(device)
            
            # 第二行到第八行：输入序列
            sequence = torch.tensor(sample[1:], dtype=torch.float32).unsqueeze(0).to(device)
            
            # 预测
            pred = model(global_vars, sequence)
            predictions.append(pred.cpu().numpy())
    
    # 保存预测结果
    pred_df = pd.DataFrame(np.vstack(predictions))
    pred_df.to_csv("python/t2.1/testresult.csv", index=False, header=False)
    return predictions

# 主执行流程
if __name__ == "__main__":
    # 训练模型
    trained_model = train_model()
    
    # 测试和预测
    test_predictions = test_and_predict("python/t2.1/sequence_predictor.pth")
    print(f"预测结果已保存至 python/t2.1/testresult.csv")