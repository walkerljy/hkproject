import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, filename):
        raw_data = pd.read_csv(filename, header=None).values
        self.num_samples = raw_data.shape[0] // 9  # 每个样本9行
        self.samples = []
        
        # 预处理每个样本
        for i in range(self.num_samples):
            sample = raw_data[i*9 : (i+1)*9]
            global_var = sample[0].astype(np.float32)    # 第一行全局变量
            features = sample[1:8].astype(np.float32)    # 第2-8行作为特征
            target = sample[8].astype(np.float32)         # 第9行作为目标
            
            # 将全局变量拼接到每个时间步
            global_expanded = np.tile(global_var, (7, 1))
            combined = np.concatenate([features, global_expanded], axis=1)
            
            self.samples.append((combined, target))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        features, target = self.samples[idx]
        return torch.tensor(features), torch.tensor(target)

# LSTM模型定义
class SequencePredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 7)

    def forward(self, x):
        out, _ = self.lstm(x)  # out shape: (batch, seq_len, hidden_size)
        return self.fc(out[:, -1, :])  # 取最后一个时间步的输出

# 训练函数
def train_model():
    # 超参数设置
    EPOCHS = 200
    BATCH_SIZE = 32
    HIDDEN_SIZE = 128
    LEARNING_RATE = 0.001
    
    # 创建数据加载器
    dataset = TimeSeriesDataset("python/t2.1/data.csv")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 初始化模型
    model = SequencePredictor(input_size=14, hidden_size=HIDDEN_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for features, targets in loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")
    
    # 保存训练好的模型
    torch.save(model.state_dict(), "python/t2.1/sequence_predictor.pth")

# 测试和预测函数
def predict():
    # 加载训练好的模型
    model = SequencePredictor(input_size=14, hidden_size=128)
    model.load_state_dict(torch.load("python/t2.1/sequence_predictor.pth"))
    model.eval()
    
    # 加载测试数据
    test_data = pd.read_csv("python/t2.1/test.csv", header=None).values
    
    # 获取全局变量（从训练数据第一个样本）
    train_data = pd.read_csv("python/t2.1/data.csv", header=None).values
    global_var = train_data[0].astype(np.float32)
    
    # 初始化滚动窗口（转换为float32类型）
    window = test_data.astype(np.float32)
    
    # 进行预测
    predictions = []
    with torch.no_grad():
        for _ in range(8):
            # 获取特征并拼接全局变量
            features = window[-7:]  # 取最后7个时间步
            global_expanded = np.tile(global_var, (7, 1))
            combined = np.concatenate([features, global_expanded], axis=1)  # 正确水平拼接
            
            # 转换为模型输入格式 (batch_size, seq_len, input_size)
            input_tensor = torch.tensor(combined).unsqueeze(0).float()
            
            # 预测并更新窗口
            pred = model(input_tensor).numpy()[0]
            window = np.vstack([window[1:], pred])
            predictions.append(pred)
    
    # 保存最后8个预测结果
    pd.DataFrame(predictions).to_csv("python/t2.1/testresult.csv", header=False, index=False)

# 执行训练和预测
if __name__ == "__main__":
    train_model()  # 注释掉这行如果只需要预测
    predict()
