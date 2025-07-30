import numpy as np
import pandas as pd
import torch
import csv
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

np.set_printoptions(suppress=True)

# 等比例缩放函数：确保输出总和为1
def proportional_scaling(x):
    """将输入向量进行等比例缩放，使所有元素之和为1"""
    x_sum = torch.sum(x, dim=-1, keepdim=True)
    # 防止除零错误
    x_sum = torch.where(x_sum == 0, torch.tensor(1.0, device=x.device), x_sum)
    return x / x_sum

class SequenceDataset(Dataset):
    def __init__(self, csv_file):
        # 读取数据
        raw_data = pd.read_csv(csv_file, header=None)
        self.data = raw_data.values
        
        # 检查数据维度是否符合要求 (9行7列的倍数)
        if self.data.shape[1] != 7:
            raise ValueError(f"数据列数必须为7，实际为{self.data.shape[1]}")
        if self.data.shape[0] % 9 != 0:
            raise ValueError(f"数据总行数必须是9的倍数，实际为{self.data.shape[0]}")
            
        self.num_samples = self.data.shape[0] // 9
        self.samples = []
        self.targets = []
        
        for i in range(self.num_samples):
            # 提取9行一组的数据
            group = self.data[i*9 : (i+1)*9]
            
            # 第一行为全局变量
            global_vars = group[0]
            
            # 第二到第八行作为输入序列 (7个时间步)
            input_sequence = group[1:8]  # 索引1到7（共7行）
            
            # 第九行作为目标
            target = group[8]
            
            # 对目标进行等比例缩放（确保总和为1）
            target_scaled = target / np.sum(target) if np.sum(target) != 0 else target
            
            # 构建输入样本：每个时间步数据与全局变量拼接
            sample = [np.concatenate([ts, global_vars]) for ts in input_sequence]
            
            self.samples.append(sample)
            self.targets.append(target_scaled)
        
        # 转换为PyTorch张量
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

class SequencePredictor(nn.Module):
    def __init__(self, input_size=14, hidden_size=128, output_size=7, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        # LSTM层
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = out[:, -1, :]
        # 全连接层
        output = self.fc(last_output)
        # 应用等比例缩放（替代softmax）
        return proportional_scaling(output)

# 配置参数
config = {
    "batch_size": 64,
    "learning_rate": 0.0005,
    "num_epochs": 500,
    "test_ratio": 0.2
}

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据集
dataset = SequenceDataset("python/t2.1/data.csv")

# 划分训练集和测试集（按时间顺序划分，不随机打乱）
train_size = int((1 - config["test_ratio"]) * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# 初始化模型、损失函数和优化器
model = SequencePredictor().to(device)
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

# 训练模型
print("开始训练...")
for epoch in range(config["num_epochs"]):
    model.train()
    train_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    # 计算平均训练损失
    avg_train_loss = train_loss / len(train_loader.dataset)
    
    # 在测试集上验证
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item() * inputs.size(0)
    
    # 计算平均测试损失
    avg_test_loss = test_loss / len(test_loader.dataset)
    
    # 调整学习率
    scheduler.step(avg_test_loss)
    
    # 定期打印训练信息
    if (epoch + 1) % 50 == 0:
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"训练损失: {avg_train_loss:.6f} | 测试损失: {avg_test_loss:.6f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.8f}")
        
        # 打印样本预测结果
        sample_input, sample_target = next(iter(test_loader))
        sample_input = sample_input.to(device)
        
        with torch.no_grad():
            sample_output = model(sample_input)
        
        print("\n样本预测:")
        print("实际值:", sample_target[0].numpy().round(4))
        print("预测值:", sample_output[0].cpu().numpy().round(4))
        print("预测值总和:", sample_output[0].cpu().sum().numpy().round(4))  # 验证总和是否为1
        print("-" * 60)

# 保存模型
torch.save(model.state_dict(), "python/t2.1/sequence_predictor.pth")
print(f"模型已保存至 python/t2.1/sequence_predictor.pth")

# 预测函数
def predict_sequence(csv_path, model_path, predict_steps=10):
    # 加载模型
    model = SequencePredictor().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 读取测试数据
    raw_data = pd.read_csv(csv_path, header=None).values
    
    # 检查测试数据格式 (8行7列)
    if raw_data.shape != (8, 7):
        raise ValueError(f"测试数据必须是8行7列，实际为{raw_data.shape}")
    
    # 提取全局变量（第一行）和初始序列（2-8行）
    global_vars = raw_data[0]
    current_sequence = raw_data[1:8]  # 7个时间步
    
    # 存储预测结果
    predictions = []
    
    with torch.no_grad():
        for _ in range(predict_steps):
            # 准备输入：每个时间步数据与全局变量拼接
            input_data = np.array([np.concatenate([ts, global_vars]) for ts in current_sequence])
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 预测
            pred = model(input_tensor)
            pred_np = pred[0].cpu().numpy()
            
            # 保存预测结果
            predictions.append(pred_np)
            
            # 滚动更新序列：新预测值添加到末尾，移除第一个元素
            current_sequence = np.vstack([current_sequence[1:], pred_np])
    
    # 打印预测详情
    print("\n预测结果详情:")
    for i, pred in enumerate(predictions, 1):
        print(f"第{i}步预测: {pred.round(4)} | 总和: {pred.sum().round(4)}")
    
    return predictions

# 执行预测
predictions = predict_sequence(
    "python/t2.1/test.csv", 
    "python/t2.1/sequence_predictor.pth",
    predict_steps=10
)

# 保存预测结果到CSV
with open('python/t2.1/testresult.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for pred in predictions:
        writer.writerow(pred.round(6))  # 保留6位小数

print("预测结果已保存至 python/t2.1/testresult.csv")
    