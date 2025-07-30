import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os

# 确保目录存在
os.makedirs("python/t2.1", exist_ok=True)

# 自定义数据集类
class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 提取输入和输出
        # 第一行是全局变量，2-8行是时间序列数据(共7行)
        inputs = np.concatenate([self.data[idx][0:1], self.data[idx][1:8]])
        # 第9行是结果
        outputs = self.data[idx][8:9]
        
        # 转换为张量
        inputs = torch.FloatTensor(inputs)
        outputs = torch.FloatTensor(outputs)
        
        return inputs, outputs

# 定义更稳定的模型
class StableSequencePredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(StableSequencePredictor, self).__init__()
        # 使用批量归一化增加稳定性
        self.bn_input = nn.BatchNorm1d(input_size)
        
        # 构建多层网络
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU(0.01))  # 更稳定的激活函数
            layers.append(nn.Dropout(0.1))  # 轻微 dropout 防止过拟合
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        # 展平输入
        x = x.view(x.size(0), -1)
        x = self.bn_input(x)
        x = self.model(x)
        
        # 等比例缩放使结果和为1 (确保数值稳定性)
        row_sums = x.sum(dim=1, keepdim=True)
        # 防止除零错误
        row_sums = torch.clamp(row_sums, min=1e-8)
        x = x / row_sums
        
        return x

# 加载和预处理数据
def load_and_preprocess_data(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path, header=None)
    data = df.values
    
    # 检查数据有效性
    if len(data) % 9 != 0:
        raise ValueError(f"数据行数必须是9的倍数，当前为{len(data)}行")
    
    # 分割成多个9x7的方阵
    num_samples = len(data) // 9
    samples = []
    
    for i in range(num_samples):
        start_idx = i * 9
        end_idx = start_idx + 9
        sample = data[start_idx:end_idx]
        
        # 验证每行是否符合要求
        for row_idx in range(1, 9):  # 2-9行(索引1-8)
            row_sum = sample[row_idx].sum()
            if not np.isclose(row_sum, 1.0, atol=1e-6):
                raise ValueError(f"样本{i}的第{row_idx+1}行和为{row_sum}，不等于1")
        
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
    
    # 较小的批次大小有助于稳定训练
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 模型参数
    input_size = 8 * 7  # 1行全局变量 + 7行时间序列数据，每行7列
    hidden_sizes = [256, 128, 64]  # 更深的网络结构
    output_size = 7     # 输出7列
    
    # 初始化模型、损失函数和优化器
    model = StableSequencePredictor(input_size, hidden_sizes, output_size)
    criterion = nn.MSELoss()
    # 使用学习率调度器控制学习率
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 训练参数
    num_epochs = 200
    best_val_loss = float('inf')
    patience = 20  # 早停机制
    no_improve_epochs = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            # 确保目标也进行了等比例缩放
            target_scaled = targets.squeeze() / targets.squeeze().sum(dim=1, keepdim=True)
            loss = criterion(outputs, target_scaled)
            
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                target_scaled = targets.squeeze() / targets.squeeze().sum(dim=1, keepdim=True)
                loss = criterion(outputs, target_scaled)
                val_loss += loss.item() * inputs.size(0)
                
                all_predictions.append(outputs.numpy())
                all_targets.append(target_scaled.numpy())
        
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.8f}')
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "python/t2.1/sequence_predictor.pth")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f'早停于第{epoch+1}轮，最佳验证损失: {best_val_loss:.6f}')
                break
    
    print(f'模型已保存至 python/t2.1/sequence_predictor.pth')

# 预测函数，增加平滑处理
def predict():
    # 加载测试数据
    test_df = pd.read_csv("python/t2.1/test.csv", header=None)
    test_data = test_df.values
    
    # 确保测试数据是8行7列
    if test_data.shape != (8, 7):
        raise ValueError(f"测试数据必须是8行7列，当前为{test_data.shape}")
    
    # 提取全局变量和时间序列数据
    global_vars = test_data[0:1]
    sequence_data = test_data[1:8]
    
    # 组合输入
    inputs = np.concatenate([global_vars, sequence_data])
    inputs = torch.FloatTensor(inputs).unsqueeze(0)  # 添加批次维度
    
    # 初始化模型
    input_size = 8 * 7
    hidden_sizes = [256, 128, 64]
    output_size = 7
    model = StableSequencePredictor(input_size, hidden_sizes, output_size)
    
    # 加载训练好的模型权重
    model.load_state_dict(torch.load("python/t2.1/sequence_predictor.pth"))
    model.eval()
    
    # 多次预测取平均，减少随机性
    with torch.no_grad():
        predictions = []
        for _ in range(10):  # 多次预测
            pred = model(inputs)
            predictions.append(pred.numpy()[0])
        
        # 取平均作为最终预测结果，增加稳定性
        prediction = np.mean(predictions, axis=0)
        
        # 再次确保和为1
        prediction = prediction / prediction.sum()
        
        # 平滑处理，减少突变
        # 计算输入序列最后一行的平均值，作为平滑参考
        last_input_row = sequence_data[-1]
        # 加权平均，预测结果占70%，上一时间步占30%
        prediction = 0.7 * prediction + 0.3 * last_input_row
        # 重新归一化
        prediction = prediction / prediction.sum()
    
    # 保存预测结果
    result_df = pd.DataFrame([prediction])
    result_df.to_csv("python/t2.1/testresult.csv", index=False, header=False)
    print('预测结果已保存至 python/t2.1/testresult.csv')
    print('预测结果:', prediction)
    print('预测结果总和:', prediction.sum())

# 主函数
def main():
    # 训练模型
    train_model()
    # 进行预测
    predict()

if __name__ == "__main__":
    main()
    