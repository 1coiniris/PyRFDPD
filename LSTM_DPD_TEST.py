import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# from anyio import sleep
from scipy.io import loadmat
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pyrfdpd.utils import metrics, plot
import Model.LSTM_DPD as LSTM
from function import PA

print(torch.cuda.is_available())

model_train = 1
model_path = "data/LSTM_DPD_Model.pt"

Model_map = ['LSTM']

# 读取PA输入输出信号
data_file = 'data/signal_100M_NR_fs6144.mat'
# data_file = 'data/signal_100M_NR_fs49152.mat'
N = 163840
x_data = loadmat(data_file)
# xorg = x_data[]
xorg = x_data['signal_100M_fs6144'][0:N]
xorg = xorg / max(abs(xorg))
xorg = xorg*0.9

yorg = np.concatenate(([0+0j, 0+0j, 0+0j, 0+0j], PA.PA_Voterra(xorg[4:N - 1], xorg[3:N - 2], xorg[2:N - 3], xorg[1:N - 4], xorg[0:N - 5]), [0 + 0j]))
# yorg = PA_DVR.PA_DVR_v1(xorg).reshape(-1,1)
# yorg = yorg / max(abs(yorg))
# yorg = yorg*0.8
x = xorg.squeeze()
y = yorg.squeeze()

# plt.xticks(fontsize=20)
# fig, ax = plt.subplots()
t = np.linspace(0, 1, 200)
plt.plot(t,abs(y[0:200]),label = 'PA_Output')
plt.plot(t,abs(x[0:200]),label = 'PA_Input')
# plt.xlim(0,200)
plt.ylim(0,1)
plt.legend()
plt.savefig('figures/LSTM/PA_waveform.png')
# plt.show()
# sleep(5)
plt.close()
fs = 614.4e6
# import TEST_Plot
plot.psd(
    {"PA input": x,"PA output":y},
    fs=fs,filename='figures/LSTM/PA_Spectrum.png'
)
# a = list(xorg)
# TEST_Plot.plot_power_spectrum({"PA input": x,"PA output":y},100e6)
# TEST_Plot.plot_amam(x, y,filename="figures/amam wo DPD.png")
plot.amam(x, {"wo DPD":y}, "figures/LSTM/PA_amam.png")
# Plot.plot_power_spectrum(xorg,100e6)

# 假设xorg和yorg是已经对齐的复数信号（numpy数组）
# 这里生成示例数据，实际使用时替换为真实数据
# N = 16384  # 信号长度
# xorg = np.random.randn(N) + 1j*np.random.randn(N)  # 原始信号（PA输入）
# yorg = np.random.randn(N) + 1j*np.random.randn(N)  # PA输出信号


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




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


M = 0
seq_length = 5

for Model in Model_map:
    if Model == 'LSTM':
        print('------------------------LSTM-------------------------------')
        # 初始化模型

        model = LSTM.LSTMDPD(input_size=2*(M+1), hidden_size=64, num_layers=2, output_size=2).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if model_train == 1:
            # 创建数据集
            X = complex_to_real(y)  # 输入：PA输出信号
            Y = complex_to_real(x)  # 目标：原始信号

            # 创建序列数据 (seq_length=10)


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


        # 生成预失真信号
def apply_dpd(model, signal, seq_length=10,M = 9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将模型移动到设备上
    model.to(device)
    model.eval()

    with torch.no_grad():
        # 转换输入格式
        real = signal.real.reshape(-1, 1)
        imag = signal.imag.reshape(-1, 1)
        data = np.hstack((real, imag))  # (N, 2)

        # 创建输入序列（与训练时相同格式）
        # sequences = create_sequences(data, seq_length)[:, :, :]  # (N-seq_length+1, 9, 2)
        sequences = create_sequences_addmemory(data, seq_length, M = M)[:, :, :]  # (N-seq_length+1, 10, 20)
        # 预测
        with torch.no_grad():
            inputs = torch.FloatTensor(sequences).to(device)
            pred = model(inputs).cpu().numpy()  # (M, 2)

        # 重构完整信号（注意处理边界）
        full_pred = np.zeros((len(data), 2))
        full_pred[seq_length + M - 1:] = pred  # 对齐时间戳
        # full_pred[seq_length - 1:] = pred  # 对齐时间戳
    return full_pred[:, 0] + 1j * full_pred[:, 1]


# 使用训练好的模型处理信号
DPD = apply_dpd(model, x, seq_length,M )
#
# DPD = align.align(x,DPD)
DPD = DPD.reshape(-1,1)
# # 评估结果（示例）
# mse = np.mean(np.abs(xorg - predistorted_signal)**2)
# print(f"Prediction MSE: {mse:.6f}")

PA_out_withDPD = np.concatenate(([0+0j, 0+0j, 0+0j, 0+0j], PA.PA_Voterra(DPD[4:N - 1], DPD[3:N - 2], DPD[2:N - 3], DPD[1:N - 4], DPD[0:N - 5]), [0 + 0j]))
# PA_out_withDPD = PA_DVR.PA_DVR_v1(predistorted_signal)
# PA_out_withDPD = PA_out_withDPD / max(abs(PA_out_withDPD))
PA_out_withDPD = PA_out_withDPD.squeeze()
# PA_out_withDPD = align.align(x,PA_out_withDPD)
# 评估结果（示例）

x_norm = x/max(abs(x))
y_norm = y/max(abs(y))
PA_out_withDPD_norm = PA_out_withDPD/max(abs(PA_out_withDPD))

# 评估结果（示例）
mse = np.mean(np.abs(x_norm - PA_out_withDPD_norm)**2)
print(f"MSE with DPD: {mse:.6f}")
mse = np.mean(np.abs(x_norm - y_norm)**2)
print(f"MSE wo   DPD: {mse:.6f}")

NMSE_withDPD = 10 * np.log10(sum(abs(PA_out_withDPD_norm - x_norm)**2) / sum(abs(x_norm)**2))
NMSE_withoutDPD = 10 * np.log10(sum(abs(y_norm - x_norm)**2) / sum(abs(x_norm)**2))
print(f"NMSE with DPD: {NMSE_withDPD} dB")
print(f"NMSE wo   DPD: {NMSE_withoutDPD} dB")

fs = 614.4e6
print("without DPD")
acpr_wo_DPD = metrics.acpr(y,fs,100e6,100e6)
print("with DPD")
acpr_with_DPD = metrics.acpr(PA_out_withDPD,fs,100e6,100e6)

plot.psd(
    {"input": x,"output_with_DPD":PA_out_withDPD,"output_wo_DPD":y},
    fs=fs,filename='figures/LSTM/LSTM_DPD_spec.png'
)
plot.amam(x, {"wo DPD":y,"DPD":DPD,"with DPD":PA_out_withDPD}, "figures/LSTM/LSTM_DPD_amam.png")
# plot.psd(
#     {"PA input": x,"PA output":y},
#     fs=fs,filename='figures/PA_Spectrum.png'
# )
#
# TEST_Plot.plot_power_spectrum(y,100e6)
# TEST_Plot.plot_power_spectrum(xorg,100e6)
plt.figure()
t = np.linspace(0, 1, 200)
plt.plot(t,abs(PA_out_withDPD[0:200]),label = 'PA_Output')
plt.plot(t,abs(x[0:200]),label = 'PA_Input')
plt.plot(t,abs(DPD[0:200]),label = 'DPD')
# plt.xlim(0,200)
plt.ylim(0,1)
plt.legend()
plt.savefig('figures/LSTM/DPD_waveform.png')
# plt.show()
# sleep(5)
plt.close()


a = 1