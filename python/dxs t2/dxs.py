import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# 读取CSV数据（假设列名分别为E_t1, E_t, G, P）
data = pd.read_csv('python/dxs t2/data.csv')
X = data[['Et', 'G', 'P']].values  # 输入特征
y = data['Et+1'].values             # 目标值

# 转换为PyTorch张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 创建数据集和数据加载器
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型
class NonlinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.randn(1))
        self.beta = torch.nn.Parameter(torch.randn(1))
        self.gamma = torch.nn.Parameter(torch.randn(1))
        # self.delta = torch.nn.Parameter(torch.randn(1))
        self.epsilon = torch.nn.Parameter(torch.randn(1))

    def forward(self, inputs):
        e_t = inputs[:, 0]
        g = inputs[:, 1]
        p = inputs[:, 2]
        
        # 按照公式计算 E_(t+1)
        return e_t*((self.alpha + self.beta * g + self.gamma * g**2 ) * p + self.epsilon)#+ self.delta * g **3

# 初始化模型、损失函数和优化器
model = NonlinearModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# 训练循环
num_epochs = 10000
best = 1
for epoch in range(num_epochs):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    if loss.item() < best and epoch>2000:
        best=loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print('\nFitted parameters:')
        print(f'α: {model.alpha.item():.4f}')
        print(f'β: {model.beta.item():.4f}')
        print(f'γ: {model.gamma.item():.4f}')
        # print(f'δ: {model.delta.item():.4f}')
        print(f'ε: {model.epsilon.item():.4f}')

# 打印拟合参数
print('\nFitted parameters:')
print(f'α: {model.alpha.item():.4f}')
print(f'β: {model.beta.item():.4f}')
print(f'γ: {model.gamma.item():.4f}')
# print(f'δ: {model.delta.item():.4f}')
print(f'ε: {model.epsilon.item():.4f}')
