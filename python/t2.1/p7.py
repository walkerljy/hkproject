import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# 自定义数据集类（添加标准化）
class TimeSeriesDataset(Dataset):
    def __init__(self, filename):
        raw_data = pd.read_csv(filename, header=None).values
        self.num_samples = raw_data.shape[0] // 9
        
        # 初始化标准化器
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # 预处理数据
        all_features = []
        all_targets = []
        for i in range(self.num_samples):
            sample = raw_data[i*9:(i+1)*9]
            global_var = sample[0].astype(np.float32)
            features = sample[1:8].astype(np.float32)
            target = sample[8].astype(np.float32)
            
            # 保存全局变量并拼接特征
            global_expanded = np.tile(global_var, (7, 1))
            combined = np.concatenate([features, global_expanded], axis=1)
            
            all_features.extend(combined)
            all_targets.append(target)
        
        # 拟合标准化器
        self.feature_scaler.fit(np.vstack(all_features))
        self.target_scaler.fit(np.vstack(all_targets))
        
        # 存储处理后的样本
        self.samples = []
        for i in range(self.num_samples):
            sample = raw_data[i*9:(i+1)*9]
            global_var = sample[0].astype(np.float32)
            features = sample[1:8].astype(np.float32)
            target = sample[8].astype(np.float32)
            
            # 标准化处理
            global_expanded = np.tile(global_var, (7, 1))
            combined = np.concatenate([features, global_expanded], axis=1)
            scaled_features = self.feature_scaler.transform(combined)
            scaled_target = self.target_scaler.transform(target.reshape(1, -1))
            
            self.samples.append((
                torch.tensor(scaled_features, dtype=torch.float32),
                torch.tensor(scaled_target.flatten(), dtype=torch.float32)
            ))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]

# LSTM模型（保持纯线性输出）
class SequencePredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 7)  # 纯线性输出层

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 直接输出线性结果

# 修改后的训练函数
def train_model():
    EPOCHS = 200
    BATCH_SIZE = 32
    HIDDEN_SIZE = 128
    LEARNING_RATE = 0.001
    
    dataset = TimeSeriesDataset("python/t2.1/data.csv")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SequencePredictor(input_size=14, hidden_size=HIDDEN_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
    
    # 保存标准化器和模型
        torch.save({
        'model_state': model.state_dict(),
        'feature_scaler_mean': dataset.feature_scaler.mean_,
        'feature_scaler_scale': dataset.feature_scaler.scale_,
        'target_scaler_mean': dataset.target_scaler.mean_,
        'target_scaler_scale': dataset.target_scaler.scale_
    }, "python/t2.1/sequence_predictor.pth")

# 修改后的预测函数（包含逆标准化）
def predict():
    # 加载模型和标准化参数
    checkpoint = torch.load("python/t2.1/sequence_predictor.pth")
    
    # 初始化标准化器
    feature_scaler = StandardScaler()
    feature_scaler.mean_ = checkpoint['feature_scaler_mean']
    feature_scaler.scale_ = checkpoint['feature_scaler_scale']
    
    target_scaler = StandardScaler()
    target_scaler.mean_ = checkpoint['target_scaler_mean']
    target_scaler.scale_ = checkpoint['target_scaler_scale']

    # 初始化模型
    model = SequencePredictor(input_size=14, hidden_size=128)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 加载测试数据
    test_data = pd.read_csv("python/t2.1/test.csv", header=None).values
    
    # 获取全局变量（假设与训练数据相同）
    train_data = pd.read_csv("python/t2.1/data.csv", header=None).values
    global_var = train_data[0].astype(np.float32)
    
    # 初始化滚动窗口（包含标准化处理）
    window = test_data.astype(np.float32)
    
    predictions = []
    with torch.no_grad():
        for _ in range(8):
            # 准备输入数据
            features = window[-7:]
            global_expanded = np.tile(global_var, (7, 1))
            combined = np.concatenate([features, global_expanded], axis=1)
            
            # 标准化特征
            scaled_features = feature_scaler.transform(combined)
            input_tensor = torch.tensor(scaled_features).unsqueeze(0).float()
            
            # 预测并逆标准化
            pred_scaled = model(input_tensor).numpy()[0]
            pred = target_scaler.inverse_transform(pred_scaled.reshape(1, -1))
            
            # 更新窗口
            window = np.vstack([window[1:], pred[0]])
            predictions.append(pred[0])
    
    # 保存最终预测结果
    pd.DataFrame(predictions).to_csv("python/t2.1/testresult.csv", header=False, index=False)

if __name__ == "__main__":
    train_model()  # 训练时注释掉这行
    predict()
