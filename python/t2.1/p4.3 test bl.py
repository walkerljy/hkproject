import numpy as np
import pandas as pd
import torch,csv
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
    def __init__(self, input_size=14, hidden_size=256, output_size=7, num_layers=2):
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
        out, _ = self.lstm(x)
        last_output = out[:, -1, :]
        raw = self.fc(last_output)
        return raw / raw.sum(dim=1, keepdim=True)

config = {
    "batch_size": 256,
    "learning_rate": 0.001,
    "num_epochs": 2000,
    "test_ratio": 0.2
}

class RelativeMSELoss(nn.Module):
    def __init__(self, eps=1e-5, clamp_range=(-1,1)):
        super().__init__()
        self.eps = eps
        self.clamp_min, self.clamp_max = clamp_range

    def forward(self, pred, target):
        # 特征维度归一化（假设输入维度为[B,C],C为特征数）
        pred_norm = pred / (pred.sum(dim=1, keepdim=True) + self.eps)  # 行归一化
        
        # 计算相对误差（分母可调节）
        denominator = target + self.eps  # 原始方案
        # denominator = 0.5*(pred_norm + target) + self.eps  # 替代方案更稳定
        diff = (pred_norm - target) / denominator
        
        # print(diff)
        diff=torch.log(torch.abs(diff)+1)
        # print(diff)
        # 梯度截断
        diff_clamped = torch.clamp(diff, self.clamp_min, self.clamp_max)
        
        # 损失计算
        return torch.mean(diff_clamped ** 2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def predict_sequence(csv_path, model_path, predict_steps=10):
    model = SequencePredictor().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    raw_file=[]
    with open(csv_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            raw_file.append(row)
    raw_data = pd.read_csv(csv_path, header=None).values
    # raw_data = [[i for i in j] for j in raw_data]
    # if len(raw_data) != 8:
    #     raise ValueError("Input should contain 8 rows (1 global + 7 time steps)")
    for i in range(len(raw_data)//9):
        opt=[]
        
        global_vars = raw_data[i*9+0]
        initial_steps = raw_data[i*9+1:i*9+8]
        
        current_sequence = np.array([np.concatenate([ts, global_vars]) for ts in initial_steps])
        current_sequence = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(predict_steps):
                pred = model(current_sequence)
                pred_np = pred[0].cpu().numpy()
                predictions.append(pred_np)
                
                new_step = np.concatenate([pred_np, global_vars])
                # print(current_sequence[:, :, :].cpu().numpy().round(4))
                updated_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    torch.tensor(new_step, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                ], dim=1)
                
                current_sequence = updated_sequence
        
        for k, pred in enumerate(predictions, 1):
            opt.append([j for j in pred])
        # print(opt)
        raw_file[i*9].append(opt[predict_steps-1][0])
        # raw_data[i*9]=np.append(raw_data[i*9],opt[predict_steps-1][0])
        # print(i,raw_data[i*9])
    with open('python/t2.1/fullresult.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(raw_file)  # 写入多行



# 使用示例
predict_sequence("python/t2.1/data.csv", 
                "python/t2.1/sequence_predictor_4.3.pth",
                predict_steps=10)
