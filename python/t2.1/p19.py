# sequence_predictor.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

# -------------------------------
# 超参数设置
# -------------------------------
input_size = 7           # 每个时间步7个变量
hidden_size = 64         # LSTM隐藏层大小
num_layers = 2           # LSTM层数
output_size = 7          # 输出7维
sequence_length = 7      # 输入序列长度（第2~8行）
learning_rate = 0.001
num_epochs = 100

# 数据路径
data_path = "python/t2.1/data.csv"
test_path = "python/t2.1/test.csv"
model_path = "python/t2.1/sequence_predictor.pth"
result_path = "python/t2.1/testresult.csv"

# 确保目录存在
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# -------------------------------
# 自定义等比例缩放函数（非 softmax）
# -------------------------------
def scale_to_one(tensor):
    """
    对每一行进行等比例缩放，使和为1
    若某行和为0，则该行保持为0
    """
    row_sum = tensor.sum(dim=1, keepdim=True)  # (batch, 1)
    mask = (row_sum > 0).float()  # 转为 float 进行乘法
    scaled = tensor / (row_sum + 1e-8)
    return scaled * mask

# -------------------------------
# 模型定义
# -------------------------------
class SequencePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SequencePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.global_fc = nn.Linear(input_size + hidden_size, output_size)  # 融合全局变量

    def forward(self, x, global_vars):
        # x: (batch_size, seq_len, input_size)
        # global_vars: (batch_size, input_size)
        batch_size = x.size(0)

        # LSTM 编码序列
        lstm_out, (hn, _) = self.lstm(x)  # 取最后一个时间步
        h_last = hn[-1]  # (batch_size, hidden_size)

        # 融合全局变量
        combined = torch.cat([h_last, global_vars], dim=1)  # (batch, hidden + 7)
        out = self.global_fc(combined)  # 直接映射到输出
        return torch.sigmoid(out)  # 先 sigmoid 保证非负

# -------------------------------
# 数据加载与预处理
# -------------------------------
def load_data(filepath):
    df = pd.read_csv(filepath, header=None)
    data = df.values.astype(np.float32)
    return data

# 读取训练数据
raw_data = load_data(data_path)

# 分割成多个 9x7 样本
assert len(raw_data) % 9 == 0, "训练数据行数必须是9的倍数"
num_samples = len(raw_data) // 9
samples = raw_data.reshape(num_samples, 9, 7)

# 提取数据
global_vars = samples[:, 0, :]  # (N, 7) 第一行是全局变量
inputs = samples[:, 1:8, :]     # (N, 7, 7) 第2~8行作为输入序列
targets = samples[:, 8, :]      # (N, 7) 第9行作为目标

# 对目标进行等比例缩放（使每行和为1）
targets_scaled = []
for tgt in targets:
    s = tgt.sum()
    if s > 0:
        tgt_scaled = tgt / s
    else:
        tgt_scaled = tgt.copy()
    targets_scaled.append(tgt_scaled)
targets_scaled = np.array(targets_scaled)

# 转为 Tensor
X_train = torch.tensor(inputs, dtype=torch.float32)           # (N, 7, 7)
G_train = torch.tensor(global_vars, dtype=torch.float32)      # (N, 7)
y_train = torch.tensor(targets_scaled, dtype=torch.float32)   # (N, 7)

# -------------------------------
# 构建并训练模型
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SequencePredictor(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

X_train = X_train.to(device)
G_train = G_train.to(device)
y_train = y_train.to(device)

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train, G_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# 保存模型
torch.save(model.state_dict(), model_path)
print(f"模型已保存至 {model_path}")

# -------------------------------
# 测试集预测
# -------------------------------
print("开始预测测试集...")

# 读取测试数据（8行 x 7列）
test_data = load_data(test_path)
assert len(test_data) == 8, "测试数据必须是8行"

# 读取对应的全局变量（从训练数据中取第一个，或你需指定来源）
# 假设 test.csv 对应的全局变量与第一个训练样本相同
# 或你可以从其他地方读取，这里用训练集第一个
global_var_test = global_vars[0]  # (7,)
G_test = torch.tensor(global_var_test, dtype=torch.float32).unsqueeze(0).to(device)  # (1,7)

# 构造输入序列：测试数据的第1~7行作为输入（即原第2~8行的位置）
# 注意：你的描述中说“第2~8行作为输入”，所以测试集8行中，前7行是输入
input_test = test_data[:7]  # (7,7)
X_test = torch.tensor(input_test, dtype=torch.float32).unsqueeze(0).to(device)  # (1,7,7)

# 预测
model.eval()
with torch.no_grad():
    pred_raw = model(X_test, G_test)  # (1,7)
    # 等比例缩放使和为1
    pred_scaled = scale_to_one(pred_raw).cpu().numpy()[0]

# 保存预测结果
result_df = pd.DataFrame([pred_scaled])
result_df.to_csv(result_path, index=False, header=False)
print(f"预测结果已保存至 {result_path}")
print("预测结果（已缩放）:", pred_scaled)