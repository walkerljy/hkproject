import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 自定义数据集类
class SequenceDataset(Dataset):
    def __init__(self, csv_file, seq_length=8, pred_steps=1):
        """
        Args:
            csv_file: 数据文件路径
            seq_length: 输入序列长度（默认8个时间步）
            pred_steps: 需要预测的未来步数（默认1步）
        """
        raw_data = pd.read_csv(csv_file, header=None)
        self.samples = []
        self.targets = []
        
        # 将数据按seq_length分组处理
        data = raw_data.values
        for i in range(len(data) - seq_length - pred_steps + 1):
            # 输入序列：seq_length个时间步
            sample = data[i:i+seq_length]
            # 输出目标：后续pred_steps个时间步
            target = data[i+seq_length:i+seq_length+pred_steps]
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
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        # LSTM处理
        out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = out[:, -1, :]
        # 全连接层生成预测
        return self.fc(last_output)

# 训练参数
config = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 200,
    "test_size": 0.2,
    "pred_steps": 1  # 预测未来1个时间步
}

# 加载数据集
dataset = SequenceDataset("D:/program development/hkproject/python/torchdata.csv", pred_steps=config["pred_steps"])

# 划分训练测试集
train_data, test_data = train_test_split(
    list(zip(dataset.samples, dataset.targets)),
    test_size=config["test_size"],
    shuffle=False
)

train_dataset = [item[0] for item in train_data], [item[1] for item in train_data]
test_dataset = [item[0] for item in test_data], [item[1] for item in test_data]

# 创建DataLoader
train_loader = DataLoader(
    list(zip(*train_dataset)),
    batch_size=config["batch_size"],
    shuffle=True
)

test_loader = DataLoader(
    list(zip(*test_dataset)),
    batch_size=config["batch_size"],
    shuffle=False
)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SequencePredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])


# 续写预测演示
def continue_sequence(model, initial_sequence, steps=3):
    """续写序列生成"""
    model.eval()
    current_seq = initial_sequence.clone().to(device)
    predictions = []
    
    with torch.no_grad():
        for _ in range(steps):
            # 预测下一个时间步
            next_step = model(current_seq)
            predictions.append(next_step.cpu().numpy())
            
            # 更新输入序列（滑动窗口）
            current_seq = torch.cat([
                current_seq[:, 1:], 
                next_step.unsqueeze(1)
            ], dim=1)
    
    return np.concatenate(predictions, axis=0)

# 测试续写功能
sample_input, sample_target = test_dataset[0][0], test_dataset[0][1]
sample_input_tensor = torch.tensor(sample_input).unsqueeze(0).to(device)

# 续写3个时间步
predicted_steps = continue_sequence(model, sample_input_tensor, steps=3)

# 修改后的打印代码（使用numpy转换）
print("初始序列（输入）：")
print(np.round(sample_input.numpy(), 3))  # 转换为numpy数组后处理
 
print("\n真实后续序列：")
print(np.round(sample_target.squeeze(0).numpy(), 3))
 
print("\n预测续写结果：")
print(np.round(predicted_steps, 3))
