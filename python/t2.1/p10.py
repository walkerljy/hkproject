import numpy as np
import pandas as pd
import torch, csv
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(suppress=True)

# 1. 改进数据加载与归一化
class SequenceDataset(Dataset):
    def __init__(self, csv_file, scaler=None):
        raw_data = pd.read_csv(csv_file, header=None)
        self.data = raw_data.values
        self.num_groups = len(self.data) // 9
        
        # 分离全局变量和时间序列数据以进行归一化
        all_global = []
        all_time_series = []
        
        for i in range(self.num_groups):
            all_global.append(self.data[i*9])
            all_time_series.extend(self.data[i*9+1 : (i+1)*9])
        
        # 合并所有数据用于拟合归一化器
        all_data = np.vstack(all_time_series + all_global)
        
        # 初始化或使用传入的归一化器
        self.scaler = MinMaxScaler(feature_range=(0, 1)) if scaler is None else scaler
        if scaler is None:
            self.scaler.fit(all_data)
        
        self.samples = []
        self.targets = []
        
        for i in range(self.num_groups):
            global_vars = self.data[i*9]
            time_series = self.data[i*9+1 : (i+1)*9]
            
            # 对数据进行归一化
            scaled_global = self.scaler.transform(global_vars.reshape(1, -1))[0]
            scaled_series = self.scaler.transform(time_series)
            
            # 构建输入序列（前7个时间步）和目标（第8个时间步）
            inputs = [np.concatenate([ts, scaled_global]) for ts in scaled_series[:7]]
            target = scaled_series[7]  # 目标也需要归一化
            
            self.samples.append(inputs)
            self.targets.append(target)
            
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

# 2. 改进模型结构
class SequencePredictor(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, output_size=7, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2  # 添加dropout防止过拟合
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),  # 添加批归一化
            nn.Linear(64, output_size),
            nn.Identity()  # 回归任务使用恒等激活
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        # 使用所有时间步的输出进行平均，而非仅最后一个时间步
        avg_output = torch.mean(out, dim=1)
        return self.fc(avg_output)

# 3. 配置与训练过程改进
config = {
    "batch_size": 128,  # 减小批次大小，适合时序数据
    "learning_rate": 0.0005,  # 降低学习率，避免震荡
    "num_epochs": 800,  # 增加训练轮次
    "test_ratio": 0.2
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集并确保时序划分（而非随机）
def create_sequential_split(dataset, test_ratio):
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size
    # 时序数据应该用后面的数据作为测试集
    return dataset[:train_size], dataset[train_size:]

# 初始化数据集（带归一化）
full_dataset = SequenceDataset("python/t2.1/data.csv")
train_dataset, test_dataset = create_sequential_split(full_dataset, config["test_ratio"])

# 使用训练集的归一化器处理测试集
test_dataset = SequenceDataset("python/t2.1/data.csv", scaler=full_dataset.scaler)
train_dataset, test_dataset = create_sequential_split(test_dataset, config["test_ratio"])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)  # 时序数据不打乱
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

model = SequencePredictor().to(device)
criterion = nn.MSELoss()  # 使用mean而非sum，更稳定
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)  # 添加权重衰减

# 学习率调度器，动态调整学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

# 训练循环改进
for epoch in range(config["num_epochs"]):
    model.train()
    train_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)  # 乘以批次大小
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item() * inputs.size(0)
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    scheduler.step(avg_test_loss)  # 根据测试损失调整学习率
    
    if (epoch+1) % 50 == 0:
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")
        
        # 样本展示
        sample_input, sample_target = next(iter(test_loader))
        sample_input = sample_input.to(device)
        
        with torch.no_grad():
            sample_output = model(sample_input)
        
        # 反归一化以显示真实尺度
        sample_target_original = full_dataset.scaler.inverse_transform(sample_target[0].numpy().reshape(1, -1))
        sample_output_original = full_dataset.scaler.inverse_transform(sample_output[0].cpu().numpy().reshape(1, -1))
        
        print("\nSample Prediction:")
        print("Actual:", sample_target_original.round(3)[0])
        print("Predicted:", sample_output_original.round(3)[0])
        print("-"*50)

torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': full_dataset.scaler
}, "python/t2.1/sequence_predictor_improved.pth")

# 4. 改进预测函数
def predict_sequence(csv_path, model_path, predict_steps=10):
    # 加载模型和归一化器
    checkpoint = torch.load(model_path)
    scaler = checkpoint['scaler']
    
    model = SequencePredictor().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    raw_data = pd.read_csv(csv_path, header=None).values
    if len(raw_data) != 8:
        raise ValueError("Input should contain 8 rows (1 global + 7 time steps)")
    
    # 分离并归一化数据
    global_vars = raw_data[0]
    initial_steps = raw_data[1:8]
    
    scaled_global = scaler.transform(global_vars.reshape(1, -1))[0]
    scaled_initial = scaler.transform(initial_steps)
    
    # 构建初始序列
    current_sequence = np.array([np.concatenate([ts, scaled_global]) for ts in scaled_initial])
    current_sequence = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(predict_steps):
            pred = model(current_sequence)
            pred_scaled = pred[0].cpu().numpy()
            # 反归一化预测结果（只取时间序列部分）
            pred_original = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
            predictions.append(pred_original)
            
            # 更新序列时使用归一化的预测结果
            new_step_scaled = np.concatenate([pred_scaled, scaled_global])
            updated_sequence = torch.cat([
                current_sequence[:, 1:, :],
                torch.tensor(new_step_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            ], dim=1)
            
            current_sequence = updated_sequence
    
    # 显示预测结果
    print("\nPrediction Details:")
    for i, pred in enumerate(predictions, 1):
        print(f"Step {i}: {pred.round(3)}")
    
    return predictions

# 使用示例
predictions = predict_sequence("python/t2.1/test.csv", 
                              "python/t2.1/sequence_predictor_improved.pth",
                              predict_steps=10)

with open('python/t2.1/testresult.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(predictions)
