import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from nmse_zte import nmse_zte
import time
import random

# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(50)

# 数据预处理函数
def preprocess_data(x, y, M=5, pn=False):
    N = len(x)
    X_delayed = np.zeros((N, M + 1), dtype=complex)
    for i in range(M + 1):
        if i < N:
            X_delayed[i:, i] = x[:N-i]

    if pn:
        rk = np.conj(X_delayed[:, 0]) / (np.abs(X_delayed[:, 0]) + 1e-10)
        X_pn = X_delayed * rk[:, np.newaxis]
        y_pn = y * rk
    else:
        X_pn = X_delayed
        y_pn = y
        rk = np.ones(N, dtype=complex)

    I_in = np.real(X_pn)
    Q_in = np.imag(X_pn)
    I_out = np.real(y_pn)
    Q_out = np.imag(y_pn)

    X_in = np.zeros((N, 2 * (M + 1)))
    Y_out = np.zeros((N, 2))

    for n in range(N):
        for m in range(M + 1):
            X_in[n, m] = I_in[n, m] if n - m >= 0 else 0
            X_in[n, M + 1 + m] = Q_in[n, m] if n - m >= 0 else 0
        Y_out[n, 0] = I_out[n]
        Y_out[n, 1] = Q_out[n]

    return X_in, Y_out, rk

# 手动计算 NMSE
def compute_nmse(x, y, a=None, b=None):
    x = np.asarray(x).reshape(-1).astype(complex)
    y = np.asarray(y).reshape(-1).astype(complex)
    if a is not None and b is not None:
        x = x[a:b + 1]
        y = y[a:b + 1]
    elif a is not None:
        x = x[a:]
        y = y[a:]
    error = x - y
    nmse = 10 * np.log10(np.mean(np.abs(error) ** 2) / np.mean(np.abs(x) ** 2))
    return nmse

# 计算验证集 NMSE
def val_NMSE(Y_pred, Y_test, rk=None, pn=False):
    Y_pred = Y_pred.cpu().numpy()
    Y_test = Y_test.cpu().numpy()
    IQ_array = Y_pred[:, 0] + 1j * Y_pred[:, 1]
    IQ_X_array = Y_test[:, 0] + 1j * Y_test[:, 1]
    if pn and rk is not None:
        IQ_array = IQ_array / (rk + 1e-10)
        IQ_X_array = IQ_X_array / (rk + 1e-10)
    nmse_value = nmse_zte(IQ_array, IQ_X_array)
    return nmse_value

# NMSE 损失函数
class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred_complex = torch.complex(y_pred[:, 0], y_pred[:, 1])
        y_true_complex = torch.complex(y_true[:, 0], y_true[:, 1])
        error = y_pred_complex - y_true_complex
        error_power = torch.mean(torch.abs(error) ** 2)
        signal_power = torch.mean(torch.abs(y_true_complex) ** 2)
        nmse_linear = error_power / (signal_power + 1e-10)
        return nmse_linear

# 复数 MSE 损失函数
class ComplexMSELoss(nn.Module):
    def __init__(self):
        super(ComplexMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred_complex = torch.complex(y_pred[:, 0], y_pred[:, 1])
        y_true_complex = torch.complex(y_true[:, 0], y_true[:, 1])
        error = y_pred_complex - y_true_complex
        mse = torch.mean(torch.abs(error) ** 2)
        return mse

# RVTDNN 模型
class RVTDNN(nn.Module):
    def __init__(self, M=5, hidden_layers=None):
        super(RVTDNN, self).__init__()
        self.M = M
        self.input_dim = 2 * (M + 1)
        if hidden_layers is None:
            hidden_layers = [50]
        layers = []
        in_dim = self.input_dim
        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)
    #     self._initialize_weights()
    #
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        x = self.network(x)
        return x

# 计算模型参数总数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 模型训练函数
def train_model(model, X_train, Y_train, X_val, Y_val, rk, pn=False, epochs=1000, batch_size=128, loss_type='mse', patience=50, grad_clip=1.0):
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'nmse':
        criterion = NMSELoss()
    elif loss_type == 'complex_mse':
        criterion = ComplexMSELoss()
    else:
        raise ValueError("loss_type must be 'mse', 'nmse', or 'complex_mse'")

    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_size = len(X_train)
    val_size = len(X_val)
    rk_val = rk[train_size:train_size + val_size]

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
        avg_train_loss = total_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, Y_val)
            val_nmse = val_NMSE(val_outputs, Y_val, rk=rk_val, pn=pn)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}, Val NMSE: {val_nmse:.6f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(best_model_state)
                break

    return model

# 主程序
if __name__ == "__main__":
    # 加载数据
    x = np.load("x.npy")
    y = np.load("y.npy")

    # 调整维度，从 (2100000, 1) 转换为 (2100000,)
    x = np.squeeze(x)
    y = np.squeeze(y)

    # 提取第一个状态的数据（10000-19999）
    samples_per_state = 10000
    state_idx = 0
    start_idx = state_idx * samples_per_state
    end_idx = (state_idx + 1) * samples_per_state
    x_state = x[start_idx:end_idx]
    y_state = y[start_idx:end_idx]

    print("Initial NMSE (nmse_zte) for State 1:", nmse_zte(x_state, y_state))

    # 设置参数
    M = 6
    pn = True
    train_size = 5000
    val_size = 5000

    # 预处理数据
    X, Y, rk = preprocess_data(x_state, y_state, M=M, pn=pn)

    # 划分训练和验证集
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_val = X[train_size:train_size + val_size]
    Y_val = Y[train_size:train_size + val_size]

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    # 初始化模型
    loss_type = 'nmse'
    hidden_layers = [40, 40]
    model = RVTDNN(M=M, hidden_layers=hidden_layers)

    # 训练模型
    start_time = time.time()
    model = train_model(
        model,
        X_train,
        Y_train,
        X_val,
        Y_val,
        rk=rk,
        pn=pn,
        epochs=1000,
        batch_size=128,
        loss_type=loss_type,
        patience=50,
        grad_clip=1.0
    )
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # 输出参数量
    total_params = count_parameters(model)
    print(f"Total number of parameters: {total_params}")

    # 在整个状态数据上评估模型（10000个样本）
    model.eval()
    with torch.no_grad():
        X_all, Y_all, rk_all = preprocess_data(x_state, y_state, M=M, pn=pn)
        X_all = torch.tensor(X_all, dtype=torch.float32)
        Y_all = torch.tensor(Y_all, dtype=torch.float32)
        Y_pred_all = model(X_all)

        # 转换为 numpy 数组
        Y_pred_all = Y_pred_all.cpu().numpy()
        Y_all = Y_all.cpu().numpy()

        IQ_array_all = Y_pred_all[:, 0] + 1j * Y_pred_all[:, 1]
        IQ_X_array_all = Y_all[:, 0] + 1j * Y_all[:, 1]

        # 撤销相位校正
        IQ_array_orig_all = IQ_array_all / (rk_all + 1e-10)
        IQ_X_array_orig_all = IQ_X_array_all / (rk_all + 1e-10)
        y_state_processed = y_state

        print("Predicted IQ_array_orig magnitude range (all):", np.abs(IQ_array_orig_all).min(), np.abs(IQ_array_orig_all).max())
        print("True y_state magnitude range (all):", np.abs(y_state_processed).min(), np.abs(y_state_processed).max())
        print("NMSE (nmse_zte) for all 10000 samples:", nmse_zte(IQ_array_orig_all, y_state_processed))
        print("Manual NMSE for all 10000 samples:", compute_nmse(IQ_array_orig_all, y_state_processed))

    # 输出特征重要性
    first_layer_weights = model.network[0].weight.detach().cpu().numpy()
    feature_importance = np.mean(np.abs(first_layer_weights), axis=0)
    print("Feature importance:", feature_importance)