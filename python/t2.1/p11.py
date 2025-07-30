import numpy as np
import pandas as pd
import torch, csv
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

np.set_printoptions(suppress=True)

class SequenceDataset(Dataset):
    def __init__(self, csv_file):
        raw_data = pd.read_csv(csv_file, header=None)
        self.samples = []
        self.targets = []
        
        data = raw_data.values
        num_groups = len(data) // 9
        
        for i in range(num_groups):
            global_vars = data[i*9]
            time_series = data[i*9+1 : (i+1)*9]
            
            inputs = [np.concatenate([ts, global_vars]) for ts in time_series[:7]]
            target = time_series[7]
            
            self.samples.append(inputs)
            self.targets.append(target)
            
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)
    
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
            dropout=0.2  # 添加dropout防止过拟合
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)  # 移除Softmax，更适合回归任务
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        last_output = out[:, -1, :]
        return self.fc(last_output)

config = {
    "batch_size": 128,  # 减小batch size
    "learning_rate": 0.0005,  # 降低学习率
    "num_epochs": 600,  # 增加训练轮数
    "test_ratio": 0.2,
    "smooth_factor": 0.2  # 平滑因子，控制预测平滑程度
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化数据集
dataset = SequenceDataset("python/t2.1/data.csv")
train_size = int((1 - config["test_ratio"]) * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

model = SequencePredictor().to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)  # 添加L2正则化

# 训练循环
for epoch in range(config["num_epochs"]):
    model.train()
    train_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 添加序列平滑损失，惩罚剧烈变化
        if inputs.size(1) > 1:
            seq_loss = torch.mean(torch.abs(outputs - inputs[:, -1, :7]))  # 与前一时间步的差异
            loss = loss + 0.1 * seq_loss  # 序列平滑损失权重
            
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
    
    if (epoch+1) % 20 == 0:
        avg_train = train_loss / (len(train_loader.dataset) * 7)
        avg_test = test_loss / (len(test_loader.dataset) * 7)
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss:.3f} | Avg per feature: {avg_train:.5f}")
        print(f"Test Loss: {test_loss:.3f} | Avg per feature: {avg_test:.5f}")
        
        # 样本展示
        sample_input, sample_target = next(iter(test_loader))
        sample_input = sample_input.to(device)
        
        with torch.no_grad():
            sample_output = model(sample_input)
        
        print("\nSample Prediction:")
        print("Actual:", sample_target[0].numpy().round(3))
        print("Predicted:", sample_output[0].cpu().numpy().round(3))
        print("-"*50)

torch.save(model.state_dict(), "python/t2.1/sequence_predictor_3.pth")

opt=[]

def predict_sequence(csv_path, model_path, predict_steps=10):
    model = SequencePredictor().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    raw_data = pd.read_csv(csv_path, header=None).values
    if len(raw_data) != 8:
        raise ValueError("Input should contain 8 rows (1 global + 7 time steps)")
    
    global_vars = raw_data[0]
    initial_steps = raw_data[1:8]
    
    current_sequence = np.array([np.concatenate([ts, global_vars]) for ts in initial_steps])
    current_sequence = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    predictions = []
    prev_pred = initial_steps[-1]  # 记录上一步的值用于平滑
    
    with torch.no_grad():
        for _ in range(predict_steps):
            pred = model(current_sequence)
            pred_np = pred[0].cpu().numpy()
            
            # 应用平滑处理，混合当前预测和上一步预测
            smoothed_pred = (1 - config["smooth_factor"]) * pred_np + config["smooth_factor"] * prev_pred
            predictions.append(smoothed_pred)
            
            # 更新上一步预测值
            prev_pred = smoothed_pred
            
            new_step = np.concatenate([smoothed_pred, global_vars])
            updated_sequence = torch.cat([
                current_sequence[:, 1:, :],
                torch.tensor(new_step, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            ], dim=1)
            
            current_sequence = updated_sequence
    
    # 显示预测的平方和
    print("\nPrediction Details:")
    for i, pred in enumerate(predictions, 1):
        sse = np.sum((pred - raw_data[7])**2)  # 与最后一个已知值的差异
        print(f"Step {i}: {pred.round(3)} | SSE: {sse:.4f}")
        opt.append([j for j in pred])



# 使用示例
predict_sequence("python/t2.1/test.csv", 
                "python/t2.1/sequence_predictor_3.pth",
                predict_steps=10)
with open('python/t2.1/testresult.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(opt)  # 写入多行
    