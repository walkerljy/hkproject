import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类
class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 提取输入和输出
        # 第一行是全局变量，2-8行是时间序列数据
        inputs = np.concatenate([self.data[idx][0:1], self.data[idx][1:8]])
        # 第9行是结果
        outputs = self.data[idx][8:9]
        
        # 转换为张量
        inputs = torch.FloatTensor(inputs)
        outputs = torch.FloatTensor(outputs)
        
        return inputs, outputs

# 定义模型
class SequencePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SequencePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 展平输入
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        # 等比例缩放使结果和为1（替代softmax）
        row_sums = x.sum(dim=1, keepdim=True)
        x = x / row_sums
        
        return x

# 加载和预处理数据
def load_and_preprocess_data(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path, header=None)
    data = df.values
    
    # 检查数据有效性
    if len(data) % 9 != 0:
        raise ValueError("数据行数必须是9的倍数")
    
    # 分割成多个9x7的方阵
    num_samples = len(data) // 9
    samples = []
    
    for i in range(num_samples):
        start_idx = i * 9
        end_idx = start_idx + 9
        sample = data[start_idx:end_idx]
        samples.append(sample)
    
    return np.array(samples)

# 训练模型
def train_model():
    # 加载数据
    data = load_and_preprocess_data("python/t2.1/data.csv")
    
    # 分割训练集和验证集
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # 创建数据集和数据加载器
    train_dataset = SequenceDataset(train_data)
    val_dataset = SequenceDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 模型参数
    input_size = 8 * 7  # 1行全局变量 + 7行时间序列数据，每行7列
    hidden_size = 128
    output_size = 7     # 输出7列
    
    # 初始化模型、损失函数和优化器
    model = SequencePredictor(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数
    num_epochs = 100
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze())
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "python/t2.1/sequence_predictor.pth")
    
    print(f'最佳验证损失: {best_val_loss:.6f}')
    print('模型已保存至 python/t2.1/sequence_predictor.pth')

# 预测函数
def predict():
    # 加载测试数据
    test_df = pd.read_csv("python/t2.1/test.csv", header=None)
    test_data = test_df.values
    
    # 确保测试数据是8行7列（因为预测时需要前8行来预测第9行）
    if test_data.shape != (8, 7):
        raise ValueError("测试数据必须是8行7列")
    
    # 提取全局变量（假设测试数据的第一行是全局变量）
    global_vars = test_data[0:1]
    # 提取时间序列数据
    sequence_data = test_data[1:8]
    
    # 组合输入
    inputs = np.concatenate([global_vars, sequence_data])
    inputs = torch.FloatTensor(inputs).unsqueeze(0)  # 添加批次维度
    
    # 初始化模型
    input_size = 8 * 7
    hidden_size = 128
    output_size = 7
    model = SequencePredictor(input_size, hidden_size, output_size)
    
    # 加载训练好的模型权重
    model.load_state_dict(torch.load("python/t2.1/sequence_predictor.pth"))
    model.eval()
    
    # 预测
    with torch.no_grad():
        prediction = model(inputs)
    
    # 转换为numpy数组并确保和为1（由于模型中已处理，这里是双重保证）
    prediction = prediction.numpy()[0]
    prediction = prediction / prediction.sum()
    
    # 保存预测结果
    result_df = pd.DataFrame([prediction])
    result_df.to_csv("python/t2.1/testresult.csv", index=False, header=False)
    print('预测结果已保存至 python/t2.1/testresult.csv')

# 主函数
def main():
    # 训练模型
    train_model()
    # 进行预测
    predict()

if __name__ == "__main__":
    main()
