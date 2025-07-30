import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv

# 配置参数（修复后）
config = {
    "sequence_length": 7,          # 时间步数
    "num_features": 7,             # 特征列数
    "hidden_size": 32,             # 隐藏层大小
    "num_layers": 1,               # 层数保持1
    "batch_size": 16,
    "epochs": 500,                 # 训练轮次
    "learning_rate": 0.0005,       # 学习率
    "test_ratio": 0.2,             # 测试集比例
    "smooth_factor": 0.3,          # 预测平滑系数
    "diff_threshold": 0.15,        # 突变检测阈值
    "scaler_range": (0.05, 0.95)   # 归一化范围
}

# 等比例缩放函数（确保最大值为1）
def max_scaling(x):
    """将输入向量进行缩放，使最大值为1"""
    if isinstance(x, torch.Tensor):
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        max_val = torch.where(max_val == 0, torch.tensor(1.0, device=x.device), max_val)
        return x / max_val
    else:  # numpy数组
        max_val = np.max(x, axis=-1, keepdims=True)
        max_val = 1.0 if max_val == 0 else max_val
        return x / max_val

class EnhancedSequenceDataset(Dataset):
    def __init__(self, csv_file, is_train=True, scaler=None):
        # 读取数据
        self.raw_data = pd.read_csv(csv_file, header=None).values
        self.is_train = is_train
        
        # 数据校验
        if self.raw_data.shape[1] != config["num_features"]:
            raise ValueError(f"数据列数必须为{config['num_features']}，实际为{self.raw_data.shape[1]}")
        
        # 计算样本数量（每组9行）
        self.num_samples = self.raw_data.shape[0] // 9
        
        # 收集所有数据用于归一化
        all_data = []
        for i in range(self.num_samples):
            group = self.raw_data[i*9 : (i+1)*9]
            all_data.extend(group)
        
        # 初始化或使用传入的归一化器
        self.scaler = MinMaxScaler(feature_range=config["scaler_range"]) if scaler is None else scaler
        if scaler is None:
            self.scaler.fit(all_data)
        
        # 准备样本和目标
        self.samples = []
        self.targets = []
        self.global_vars_list = []  # 存储每组数据的全局变量
        self.sequence_stats = []    # 存储序列统计信息
        
        for i in range(self.num_samples):
            group = self.raw_data[i*9 : (i+1)*9]
            global_vars = group[0]          # 第一行：全局变量
            input_sequence = group[1:8]     # 2-8行：输入序列
            target = group[8]               # 第9行：目标
            
            # 归一化处理
            scaled_global = self.scaler.transform(global_vars.reshape(1, -1))[0]
            scaled_input = self.scaler.transform(input_sequence)
            
            # 计算序列统计信息
            seq_mean = np.mean(scaled_input, axis=0)
            seq_std = np.std(scaled_input, axis=0)
            seq_last = scaled_input[-1]  # 最后一个时间步
            self.sequence_stats.append((seq_mean, seq_std, seq_last))
            
            # 构建样本：每个时间步与全局变量拼接
            sample = [np.concatenate([ts, scaled_global]) for ts in scaled_input]
            
            # 处理目标：先归一化再缩放使最大值为1
            scaled_target = self.scaler.transform(target.reshape(1, -1))[0]
            target_scaled = max_scaling(scaled_target)
            
            self.samples.append(sample)
            self.targets.append(target_scaled)
            self.global_vars_list.append(global_vars)
        
        # 转换为PyTorch张量
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.is_train:
            return (self.samples[idx], self.targets[idx], 
                    self.sequence_stats[idx], self.global_vars_list[idx])
        else:
            return (self.samples[idx], self.sequence_stats[idx], 
                    self.global_vars_list[idx])

class OptimizedPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(OptimizedPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层 - 修复：当层数为1时不使用dropout
        dropout = 0.1 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout  # 只有层数>1时才使用dropout
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.Tanh(),  # 更稳定的激活函数
            nn.Linear(16, output_size)
        )
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重以提高稳定性"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=0.5)  # 小增益
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        # x形状：(batch_size, seq_len, features)
        device = x.device
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 使用最后3个时间步的平均输出，增强时序连续性
        last_3_avg = torch.mean(out[:, -3:, :], dim=1)
        
        # 全连接层输出
        out = self.fc(last_3_avg)
        
        # 应用最大值缩放
        return max_scaling(out)

def train_model():
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载完整数据集
    full_dataset = EnhancedSequenceDataset('python/t2.1/data.csv', is_train=True)
    
    # 按时间顺序划分训练集和测试集
    train_size = int((1 - config["test_ratio"]) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 初始化模型
    input_size = config["num_features"] * 2  # 特征+全局变量
    model = OptimizedPredictor(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=config["num_features"]
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-4  # L2正则化
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=50, factor=0.5, min_lr=1e-7
    )
    
    # 记录损失
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    # 训练循环
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            inputs, targets, stats, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 修复：正确处理序列统计信息中的最后一个时间步
            # 提取每个样本的最后一个时间步并转换为张量
            seq_last_list = [s[2] for s in stats]  # 获取列表中的数组
            seq_last = torch.tensor(seq_last_list, dtype=torch.float32).to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 基础损失
            base_loss = criterion(outputs, targets)
            
            # 序列平滑损失（惩罚与上一时间步的剧烈变化）
            smooth_loss = torch.mean(torch.square(outputs - seq_last))
            
            # 组合损失
            loss = base_loss + 0.5 * smooth_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # 在测试集上验证
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets, _, _ = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item() * inputs.size(0)
        
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        
        # 调整学习率
        scheduler.step(avg_test_loss)
        
        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': full_dataset.scaler
            }, 'python/t2.1/sequence_predictor.pth')
        
        # 定期打印信息
        if (epoch + 1) % 50 == 0:
            print(f"\nEpoch {epoch+1}/{config['epochs']}")
            print(f"训练损失: {avg_train_loss:.6f} | 测试损失: {avg_test_loss:.6f}")
            print(f"当前学习率: {optimizer.param_groups[0]['lr']:.8f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.title('训练过程损失变化')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.savefig('python/t2.1/training_loss.png')
    plt.close()
    
    print(f"最佳模型已保存 (最佳测试损失: {best_test_loss:.6f})")

def predict_sequence(predict_steps=10):
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型和归一化器
    checkpoint = torch.load('python/t2.1/sequence_predictor.pth')
    scaler = checkpoint['scaler']
    
    # 初始化模型
    input_size = config["num_features"] * 2
    model = OptimizedPredictor(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=config["num_features"]
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 读取测试数据
    test_data = pd.read_csv('python/t2.1/test.csv', header=None).values
    if test_data.shape != (8, config["num_features"]):
        raise ValueError(f"测试数据必须是8行{config['num_features']}列，实际为{test_data.shape}")
    
    # 提取数据
    global_vars = test_data[0]          # 第一行：全局变量
    current_sequence = test_data[1:8]   # 2-8行：初始序列
    last_known_step = current_sequence[-1].copy()  # 最后一个已知值
    
    # 归一化处理
    scaled_global = scaler.transform(global_vars.reshape(1, -1))[0]
    scaled_sequence = scaler.transform(current_sequence)
    scaled_last_known = scaled_sequence[-1].copy()
    
    # 分析输入序列波动性
    seq_std = np.std(scaled_sequence, axis=0).mean()
    dynamic_smooth = min(0.5, max(0.2, seq_std * 2))  # 动态平滑系数
    
    predictions = []
    
    with torch.no_grad():
        for step in range(predict_steps):
            # 准备输入：每个时间步与全局变量拼接
            input_data = np.array([np.concatenate([ts, scaled_global]) for ts in scaled_sequence])
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 预测
            pred = model(input_tensor)
            pred_scaled = pred[0].cpu().numpy()
            
            # 对第一个预测应用平滑处理
            if step == 0:
                # 计算与最后已知值的差异
                diff = np.abs(pred_scaled - scaled_last_known)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                print(f"\n初始预测差异分析: 最大差异={max_diff:.4f}, 平均差异={mean_diff:.4f}")
                
                # 应用基础平滑
                pred_scaled = (1 - config["smooth_factor"]) * pred_scaled + config["smooth_factor"] * scaled_last_known
                
                # 如果差异过大，应用增强平滑
                if max_diff > config["diff_threshold"]:
                    print(f"检测到潜在突变，应用增强平滑 (阈值={config['diff_threshold']})")
                    pred_scaled = (1 - dynamic_smooth) * pred_scaled + dynamic_smooth * scaled_last_known
                
                # 重新应用最大值缩放
                pred_scaled = max_scaling(pred_scaled)
            
            # 反归一化到原始尺度
            pred_original = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
            # 对原始尺度结果再次应用最大值缩放
            pred_original = max_scaling(pred_original)
            
            predictions.append(pred_original)
            
            # 滚动更新序列
            scaled_sequence = np.vstack([scaled_sequence[1:], pred_scaled])
            scaled_last_known = pred_scaled  # 更新最后已知值
    
    # 打印预测详情
    print("\n预测结果分析:")
    print(f"最后已知值: {last_known_step.round(4)}")
    for i, pred in enumerate(predictions, 1):
        prev_val = last_known_step if i == 1 else predictions[i-2]
        diff = np.abs(pred - prev_val)
        print(f"第{i}步预测: {pred.round(4)} | 与前值差异: {diff.round(4)} | 最大值: {np.max(pred).round(4)}")
    
    # 保存预测结果
    with open('python/t2.1/testresult.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for pred in predictions:
            writer.writerow(pred.round(6))
    
    print("\n预测结果已保存至 python/t2.1/testresult.csv")
    return predictions

if __name__ == "__main__":
    train_model()
    predict_sequence(predict_steps=10)
    