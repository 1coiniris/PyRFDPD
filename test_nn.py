from operator import concat

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
# from anyio import sleep
from scipy.io import loadmat, savemat
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pyrfdpd.utils import metrics, plot, align
import PA_DVR
import TEST_Plot
import Model.LSTM_DPD as LSTM
import Model.volterra_nn as MCP_NN
import Model.gmp as gmp
import Model.mp as mp
import PA
# NET.MCP_NN()
print(torch.cuda.is_available())

model_train = 1
model_path = "data/MCPNN_Pred_Model.pt"

# 读取PA输入输出信号
data_file = 'data/dataxy400m2G.mat'
# data_file = 'data/signal_100M_NR_fs49152.mat'
data = loadmat(data_file)

# Model_map = ['GMP','MP']
Model_map = ['MCP_NN']

# xorg = x_data[]
xorg = data['x0']
yorg = data['y00']

N = len(xorg)
xnorm = xorg / max(abs(xorg))
# xorg = xorg*0.9
ynorm = yorg / max(abs(yorg))
# yorg = yorg*0.9

# yorg = np.concatenate(([0+0j, 0+0j, 0+0j, 0+0j], PA.PA_Voterra(xorg[4:N-1],xorg[3:N-2],xorg[2:N-3],xorg[1:N-4],xorg[0:N-5]), [0+0j]))
# # yorg = PA_DVR.PA_DVR_v1(xorg).reshape(-1,1)
# yorg = yorg / max(abs(yorg))

x = xorg.squeeze()
y = yorg.squeeze()

# plt.xticks(fontsize=20)
# fig, ax = plt.subplots()
t = np.linspace(0, 1, 200)
plt.plot(t,abs(ynorm[0:200]),label = 'PA_Output')
plt.plot(t,abs(xnorm[0:200]),label = 'PA_Input')
# plt.xlim(0,200)
plt.ylim(0,1)
plt.legend()
plt.savefig('figures/MCP_NN/PA_waveform.png')
# plt.show()
# sleep(5)
plt.close()

fs = 2e9
# import TEST_Plot
plot.psd(
    {"PA input": x,"PA output":y},
    fs=fs,filename='figures/MCP_NN/PA_Spectrum.png'
)
# a = list(xorg)
# TEST_Plot.plot_power_spectrum({"PA input": x,"PA output":y},100e6)
# TEST_Plot.plot_amam(x, y,filename="figures/amam wo DPD.png")
plot.amam(x, {"PAout":y}, "figures/MCP_NN/PA_amam.png")
plot.ampm(x, {"PAout":y}, "figures/MCP_NN/PA_ampm.png")

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 数据预处理函数
def create_dataset(x, y, M):
    # 转换为实部虚部分离格式
    x_real = torch.view_as_real(x).float()  # [N, 2]
    y_real = torch.view_as_real(y).float()  # [N, 2]

    # 创建延迟窗口
    sequences = []
    targets = []
    for i in range(M, len(x) - 1):
        # 输入：x(n-M)到x(n)的实部虚部
        window = x_real[i - M:i + 1].flatten()  # [2*(M+1),]
        # 输出：y(n+1)的实部虚部
        target = y_real[i + 1]  # [2,]
        sequences.append(window)
        targets.append(target)

    return torch.stack(sequences), torch.stack(targets)

# ====================== 数据预处理（添加滑动窗口）======================
def create_sequences(data, seq_length):
    """将数据转换为序列格式"""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

def create_sequences_addmemory(data, seq_length, M = 9):
    """将数据转换为序列格式"""
    sequences = []
    for i in range(len(data) - seq_length - M + 1):
        memory = []
        for j in range(seq_length):
            memory.append(data[i+j:i+j+M+1].reshape(2*(M+1)))
        sequences.append(memory)
    return np.array(sequences)

# 数据预处理：将复数转换为实部+虚部
def complex_to_real(x):
    return np.stack((x.real, x.imag), axis=1)

M = 9
seq_length = 1
y_pred = []
for Model in Model_map:
    if Model == 'LSTM':
        print('------------------------LSTM-------------------------------')
        # 初始化模型
        model = LSTM.LSTMDPD(input_size=2*(M+1), hidden_size=64, num_layers=2, output_size=2).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if model_train == 1:
            # 创建数据集
            X = complex_to_real(x)  # 输入：PA输出信号
            Y = complex_to_real(y)  # 目标：原始信号

            # # 创建序列数据 (seq_length=10)
            # seq_length = 10
            # X_seq = create_sequences(X, seq_length)  # 形状 (10000-9, 10, 2)
            # Y_seq = create_sequences(Y, seq_length)  # 形状 (10000-9, 10, 2)
            X_seq = create_sequences_addmemory(X, seq_length, M=M)
            # Y_seq = create_sequences(X,seq_length)# 形状 (10000-9, 10, 2)
            Y_seq = create_sequences_addmemory(Y, seq_length,M=M)

            X_data = X_seq[:, :, :]  # 输入序列：1o个时间步 (10000-9, 10, 2)
            Y_data = Y_seq[:, -1, -2:]   # 目标值：第10个时间步 (10000-9, 2)

            # ====================== 转换为PyTorch张量 ======================
            X_tensor = torch.FloatTensor(X_data)  # (16375, 10, 2)
            Y_tensor = torch.FloatTensor(Y_data)  # (16375, 2)

            # 划分数据集（保持时序顺序）
            X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=0.2, shuffle=False)

            # 创建数据加载器
            batch_size = 256
            train_dataset = TensorDataset(X_train, Y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            val_dataset = TensorDataset(X_val, Y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            # 训练循环
            num_epochs = 20
            for epoch in range(num_epochs):
                model.train()
                # i = 0
                train_loss = 0
                for inputs, targets in train_loader:
                    # 将数据移动到设备上
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    # i = i + 1
                    # print(i)
                # 验证
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        # 将数据移动到设备上
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, targets).item()

                print(
                    f'Epoch {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f}')

            # 保存模型参数（推荐保存为 .pt 或 .pth 文件）
            torch.save(model.state_dict(), model_path)
        else:
            model.load_state_dict(torch.load(model_path))

        y_pred = model.apply_dpd(x, seq_length,M)

    x_norm = x/max(abs(x))
    y_norm = y/max(abs(y))
    y_pred_norm = 1 #y_pred/max(abs(y_pred))

    # 评估结果（示例）
    NMSE_pred = 10 * np.log10(sum(abs(y_pred_norm - y_norm)**2) / sum(abs(x_norm)**2))
    # NMSE_withoutDPD = 10 * np.log10(sum(abs(y_norm - x_norm)**2) / sum(abs(x_norm)**2))
    print(f"NMSE with {Model}: {NMSE_pred} dB")
    # print(f"NMSE wo   DPD: {NMSE_withoutDPD} dB")

    # fs = 614.4e6
    # print("without DPD")
    # acpr_wo_DPD = metrics.acpr(y,fs,100e6,100e6)
    # print("with DPD")
    # acpr_with_DPD = metrics.acpr(PA_out_withDPD,fs,100e6,100e6)


    plot.psd(
        {"input": x,"pred_output":y_pred,"output":y},
        fs=fs,filename=f'figures/NN/{Model}_spec.png'
    )
    plot.amam(x, {"out":y,"pred":y_pred}, f"figures/NN/{Model}_amam.png")
    plot.ampm(x, {"out": y, "pred": y_pred}, f"figures/NN/{Model}_ampm.png")
    plt.figure()
    t = np.linspace(0, 1, 400)
    plt.plot(t,abs(y_pred[000:400]),label = 'y_pred')
    plt.plot(t,abs(y[000:400]),label = 'y')
    # plt.plot(t,abs(DPD[2000:2200]),label = 'DPD')
    # plt.xlim(0,200)
    plt.ylim(0,1)
    plt.legend()
    plt.savefig(f'figures/NN/{Model}_waveform.png')

A = 1