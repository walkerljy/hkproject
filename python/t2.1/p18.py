# python/t2.1/train_and_predict.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import os

# ---------- 配置 ----------
DATA_PATH      = Path('python/t2.1/data.csv')
TEST_PATH      = Path('python/t2.1/test.csv')
MODEL_PATH     = Path('python/t2.1/sequence_predictor.pth')
RESULT_PATH    = Path('python/t2.1/testresult.csv')
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE     = 64
EPOCHS         = 500
LEARNING_RATE  = 1e-3
SEED           = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
# ---------------------------

# ---------- 工具 ----------
def load_dataset(csv_path):
    """
    将 csv 读成 (N, 9, 7) 的 float32 ndarray
    N 为样本数（自动推断）
    要求 csv 按行展开：先 9 行样本 1，再 9 行样本 2 ...
    """
    df = pd.read_csv(csv_path, header=None)
    mat = df.values.astype(np.float32)
    if mat.size % (9 * 7) != 0:
        raise ValueError('csv 行数必须能被 9 整除')
    N = mat.shape[0] // 9
    mat = mat.reshape(N, 9, 7)
    return mat

def make_loader(mat):
    global_vars = mat[:, 0, :]                  # (N,7)
    seq_x       = mat[:, 1:8, :].reshape(mat.shape[0], -1)  # (N,49)
    seq_y       = mat[:, 8, :]                  # (N,7)
    x = np.concatenate([global_vars, seq_x], axis=1)        # (N,56)
    x = torch.from_numpy(x)
    y = torch.from_numpy(seq_y)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader  = torch.utils.data.DataLoader(dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)
    return loader
# --------------------------

# ---------- 模型 ----------
class Predictor(nn.Module):
    def __init__(self, in_dim=56, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 7)
        )

    def forward(self, x):
        out = self.net(x)          # (B,7)
        # 等比例缩放
        sums = out.sum(dim=1, keepdim=True).clamp(min=1e-8)
        out  = out / sums
        return out
# --------------------------

# ---------- 训练 ----------
def train():
    mat = load_dataset(DATA_PATH)
    loader = make_loader(mat)

    model = Predictor().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            # yb 已经是和为 1 的向量，无需再缩放
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        if epoch % 50 == 0 or epoch == 1:
            print(f'Epoch {epoch:4d} | MSE loss: {total_loss / len(loader.dataset):.6f}')

    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print('Model saved to', MODEL_PATH)
# --------------------------

# ---------- 预测 ----------
def predict():
    if not MODEL_PATH.exists():
        raise FileNotFoundError('请先运行 train() 保存模型')
    mat_test = pd.read_csv(TEST_PATH, header=None).values.astype(np.float32)
    if mat_test.shape != (8, 7):
        raise ValueError('测试集必须是 8 行 7 列')

    global_vars = mat_test[0]           # (7,)
    seq_x       = mat_test[1:].reshape(-1)  # (49,)

    x = np.concatenate([global_vars, seq_x], axis=0)  # (56,)
    x = torch.from_numpy(x).unsqueeze(0).to(DEVICE)   # (1,56)

    model = Predictor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy()  # (7,) 已和为 1

    pd.DataFrame(pred).to_csv(RESULT_PATH, index=False, header=False)
    print('Prediction saved to', RESULT_PATH)
# --------------------------

if __name__ == '__main__':
    # 训练
    train()
    # 预测
    predict()