import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# 确保目录存在
os.makedirs("python/t2.1", exist_ok=True)

# =========== 数据加载与预处理 ===========
def load_data(filepath, is_train=True):
    """加载数据并进行预处理"""
    df = pd.read_csv(filepath, header=None)
    data = df.values
    
    if is_train:
        # 训练集：每9行为一个样本
        num_samples = data.shape[0] // 9
        X = []
        y = []
        for i in range(num_samples):
            start = i * 9
            end = (i + 1) * 9
            sample = data[start:end]
            # 全局变量 + 时间步数据 (第1-7行)
            x = np.concatenate([sample[0], sample[1:8].flatten()])
            X.append(x)
            # 目标值 (第9行)
            y.append(sample[8])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    else:
        # 测试集：每8行为一个样本
        num_samples = data.shape[0] // 8
        X = []
        for i in range(num_samples):
            start = i * 8
            end = (i + 1) * 8
            sample = data[start:end]
            # 全局变量 + 时间步数据 (第1-7行)
            x = np.concatenate([sample[0], sample[1:8].flatten()])
            X.append(x)
        return torch.tensor(X, dtype=torch.float32)

# 加载训练数据
X_train, y_train = load_data("python/t2.1/data.csv")
print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}, Targets: {y_train.shape[1]}")

# =========== 定义模型 ===========
class SequencePredictor(nn.Module):
    def __init__(self, input_dim=56, hidden_dim=128, output_dim=7):
        super(SequencePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
model = SequencePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========== 训练模型 ===========
num_epochs = 1600
best_loss = float('inf')
patience = 200  # 早停耐心值
counter = 0

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # 打印训练进度
    if (epoch+1) % 20 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    # 早停机制
    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), "python/t2.1/sequence_predictor.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# =========== 测试预测 ===========
# 加载测试数据
X_test = load_data("python/t2.1/test.csv", is_train=False)
model.load_state_dict(torch.load("python/t2.1/sequence_predictor.pth"))
model.eval()

with torch.no_grad():
    raw_predictions = model(X_test)
    # 等比例缩放（总和为1）
    scaled_predictions = raw_predictions / raw_predictions.sum(dim=1, keepdim=True)

# 保存预测结果
pd.DataFrame(scaled_predictions.numpy()).to_csv(
    "python/t2.1/testresult.csv",
    index=False,
    header=None
)

print("Prediction completed. Results saved to python/t2.1/testresult.csv")
