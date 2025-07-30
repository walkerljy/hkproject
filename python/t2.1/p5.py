import numpy as np
import pandas as pd
import torch, csv
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

class SequenceDataset(Dataset):
    # 保持原有实现不变
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

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx], self.targets[idx]

class SequencePredictor(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, output_size=7, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)  # 移除Softmax
        )
    
    def forward(self, x, hidden=None):  # 添加hidden参数
        out, hidden = self.lstm(x, hidden) 
        return self.fc(out[:, -1, :]), hidden

# 保持训练配置和训练循环不变
config = {
    "batch_size": 256,
    "learning_rate": 0.001,
    "num_epochs": 400,
    "test_ratio": 0.2
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    dataset = SequenceDataset("python/t2.1/data.csv")
    train_size = int((1 - config["test_ratio"]) * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    model = SequencePredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    for epoch in range(config["num_epochs"]):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)  # 忽略训练时的hidden状态
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # 验证和输出逻辑保持原有形式...
    
    torch.save(model.state_dict(), "python/t2.1/sequence_predictor_v2.pth")

def predict_sequence(csv_path, model_path, predict_steps=10):
    model = SequencePredictor().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    raw_data = pd.read_csv(csv_path, header=None).values
    global_vars = raw_data[0]
    initial_steps = raw_data[1:8]  # 取前7个时间步
    
    # 构建初始序列
    seq_buffer = [np.concatenate([ts, global_vars]) for ts in initial_steps]
    current_seq = torch.tensor(seq_buffer, dtype=torch.float32).unsqueeze(0).to(device)
    
    predictions = []
    hidden = None  # 初始化隐藏状态
    
    with torch.no_grad():
        for _ in range(predict_steps):
            pred, hidden = model(current_seq, hidden)
            pred_np = pred[0].cpu().numpy()
            predictions.append(pred_np)
            
            # 滚动更新序列
            new_step = np.concatenate([pred_np, global_vars])
            new_step_tensor = torch.tensor(new_step, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            current_seq = torch.cat([current_seq[:, 1:, :], new_step_tensor], dim=1)
    
    # 输出结果保持原有形式...
    return predictions

# 使用示例
if __name__ == "__main__":
    # 需要先执行训练生成新版模型
    train_model()  
    
    opt = []
    predictions = predict_sequence("python/t2.1/test.csv", 
                                  "python/t2.1/sequence_predictor_v2.pth",
                                  predict_steps=10)
    
    with open('python/t2.1/testresult_v2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(predictions)
