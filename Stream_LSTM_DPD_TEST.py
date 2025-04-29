from functools import partial

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# from anyio import sleep
from scipy.io import loadmat, savemat
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pyrfdpd.nn as dpdnn
import pyrfdpd.visa as visa
from pyrfdpd.utils import metrics, plot, align
import PA_DVR
import TEST_Plot
from Model import LSTM_DPD as LSTM
import PA

print(torch.cuda.is_available())
if __name__ == '__main__':
    # 读取PA输入输出信号
    data_file = 'data/signal_100M_NR_fs6144.mat'
    # data_file = 'data/signal_100M_NR_fs49152.mat'
    N = 163840
    x_data = loadmat(data_file)
    # xorg = x_data[]
    xorg = x_data['signal_100M_fs6144'][0:N]
    xorg = xorg / max(abs(xorg))
    yorg = np.concatenate(([0+0j, 0+0j, 0+0j, 0+0j], PA.PA_Voterra(xorg[4:N-1],xorg[3:N-2],xorg[2:N-3],xorg[1:N-4],xorg[0:N-5]), [0+0j]))
    # yorg = PA_DVR.PA_DVR_v1(xorg).reshape(-1,1)
    yorg = yorg / max(abs(yorg))

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
    plt.savefig('figures/PA_waveform.png')
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

    # 转换为实部+虚部格式 [N, 2]
    x_real_imag = np.stack([x.real, x.imag], axis=1)
    y_real_imag = np.stack([y.real, y.imag], axis=1)

    # 划分训练集和验证集
    train_size = int(0.8 * N)
    x_train, x_val = x_real_imag[:train_size], x_real_imag[train_size:]
    y_train, y_val = y_real_imag[:train_size], y_real_imag[train_size:]


    # 设置新的序列参数
    seq_length = 512       # 每个序列长度
    total_batches = 256    # 总batch数量
    parallel_batches = 32  # 并行处理的batch数

    # 训练集重整为 [总batch数, seq_length, 特征数]
    x_train = x_train[:total_batches*seq_length].reshape(total_batches, seq_length, 2)
    y_train = y_train[:total_batches*seq_length].reshape(total_batches, seq_length, 2)

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(x_train),torch.FloatTensor(y_train))

    train_loader = DataLoader(
        train_dataset,
        batch_size=parallel_batches,  # 每次处理32个batch
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    # ====================== 训练配置 ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM.ParallelLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ====================== 训练循环 ======================
    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # 每个epoch需要遍历256/32=8次迭代
        for batch_x, batch_y in train_loader:
            # 数据移动到GPU
            batch_x = batch_x.to(device, non_blocking=True)  # [32, 512, 2]
            batch_y = batch_y.to(device, non_blocking=True)  # [32, 512, 2]

            # 前向传播
            outputs = model(batch_x)

            # 计算损失
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ================= 验证阶段 =================
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # 重整验证数据为相同格式
            val_batches = 8  # 验证集batch数
            x_val_reshaped = x_val[:val_batches * seq_length].reshape(val_batches, seq_length, 2)
            y_val_reshaped = y_val[:val_batches * seq_length].reshape(val_batches, seq_length, 2)

            val_x = torch.FloatTensor(x_val_reshaped).to(device)
            val_y = torch.FloatTensor(y_val_reshaped).to(device)

            val_outputs = model(val_x)
            val_loss = criterion(val_outputs, val_y)

        # 打印统计信息
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1:03} | "f"Train Loss: {avg_train_loss:.6f} | "f"Val Loss: {val_loss.item():.6f}")


    # 训练循环

    # for epoch in range(10):
    #     model.train()
    #     model.reset_hidden()
    #
    #     total_loss = 0
    #     for t in range(0, seq_length, truncated_steps):
    #         # 截断梯度反向传播
    #         optimizer.zero_grad()
    #
    #         # 初始化截断区间的隐藏状态
    #         hidden_init = (
    #             model.hidden[0].detach().clone(),
    #             model.hidden[1].detach().clone()
    #         )
    #
    #         # 前向传播（截断区间内）
    #         batch_loss = 0
    #         for step in range(t, min(t + truncated_steps, seq_length)):
    #             # 当前输入 [1,1,2]
    #
    #             inputs = input_stream[step:step + 1]  # shape [1,1,2]
    #             inputs = inputs.to(device)
    #             # 前向计算
    #             preds = model(inputs)
    #
    #             # 计算损失
    #             targets = target_stream[step]
    #             targets = targets.to(device)
    #
    #             loss = criterion(preds, targets)
    #             batch_loss += loss
    #
    #         # 反向传播
    #         batch_loss.backward()
    #         optimizer.step()
    #         total_loss += batch_loss.item()
    #
    #         # 恢复隐藏状态用于下一个截断区间
    #         model.hidden = (hidden_init[0].detach(), hidden_init[1].detach())
    #     print(f'---------------Epoch {epoch + 1}----------------')
    #     print(f"Epoch {epoch + 1} | Train Loss: {total_loss / seq_length:.6f}")
    #
    #     # 验证
    #     model.eval()
    #     val_signal = X_val
    #     val_target = Y_val
    #     val_loss = 0
    #     DPD = np.zeros_like(val_signal, dtype=complex)
    #     with torch.no_grad():
    #         for t in range(len(val_signal)):
    #             # 转换当前输入
    #             current_input = torch.FloatTensor([[[val_signal[t].real, val_signal[t].imag]]]).to(device)  # shape [1,1,2]
    #             targets = torch.FloatTensor([[[val_target[t].real, val_target[t].imag]]]).to(device)
    #             # 前向传播
    #             outputs = model(current_input)
    #             val_loss += criterion(outputs, targets).item()
    #     print(f'Epoch {epoch + 1} | Val Loss: {val_loss / len(val_target):.6f}')
                # outputs = outputs.cpu().numpy()
                # 保存预失真信号
                # DPD[t] = pred[0, 0][0] + 1j * pred[0, 0][1]

        # with torch.no_grad():
        #     for inputs, targets in val_loader:
        #         # 将数据移动到设备上
        #         inputs = inputs.to(device)
        #         targets = targets.to(device)
        #         outputs = model(inputs)
        #         val_loss += criterion(outputs, targets).item()


    # # 生成预失真信号
    # def apply_dpd(model, signal):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #     # 将模型移动到设备上
    #     model.to(device)
    #     model.eval()
    #     model.reset_hidden()
    #
    #     DPD = np.zeros_like(signal, dtype=complex)
    #     with torch.no_grad():
    #         for t in range(len(signal)):
    #             # 转换当前输入
    #             current_input = torch.FloatTensor([[[signal[t].real, signal[t].imag]]]).to(device)  # shape [1,1,2]
    #
    #             # 前向传播
    #             pred = model(current_input)
    #             pred = pred.cpu().numpy()
    #             # 保存预失真信号
    #             DPD[t] = pred[0, 0][0] + 1j * pred[0, 0][1]
    #     return DPD
    # ====================== 推理使用 ======================
    def batch_predict(model, signal, seq_length=512):
        """批量推理函数"""
        model.eval()
        # 将信号重整为 [N_batches, seq_length, 2]
        num_samples = len(signal)
        num_batches = num_samples // seq_length
        signal_trimmed = signal[:num_batches * seq_length]

        signal_tensor = torch.FloatTensor(
            np.stack([signal_trimmed.real, signal_trimmed.imag], axis=1)
        ).reshape(num_batches, seq_length, 2).to(device)

        with torch.no_grad():
            pred = model(signal_tensor)

        # 将输出恢复为连续信号
        return pred.cpu().numpy().reshape(-1, 2)


    # 使用训练好的模型处理信号
    Z = batch_predict(model, x)
    DPD = np.zeros(len(Z), dtype=complex)
    for t in range(len(Z)):
        DPD[t] = Z[t][0] + 1j*Z[t][1]
    DPD = DPD.reshape(-1,1)
    # # 评估结果（示例）
    # mse = np.mean(np.abs(xorg - predistorted_signal)**2)
    # print(f"Prediction MSE: {mse:.6f}")

    PA_out_withDPD = np.concatenate(([0+0j, 0+0j, 0+0j, 0+0j], PA.PA_Voterra(DPD[4:N-1],DPD[3:N-2],DPD[2:N-3],DPD[1:N-4],DPD[0:N-5]), [0+0j]))
    # PA_out_withDPD = PA_DVR.PA_DVR_v1(predistorted_signal)
    PA_out_withDPD = PA_out_withDPD / max(abs(PA_out_withDPD))
    PA_out_withDPD = PA_out_withDPD.squeeze()
    # 评估结果（示例）
    mse = np.mean(np.abs(x - PA_out_withDPD)**2)
    print(f"MSE with DPD: {mse:.6f}")
    mse = np.mean(np.abs(x - y)**2)
    print(f"MSE wo   DPD: {mse:.6f}")

    NMSE_withDPD = 10 * np.log10(sum(abs(PA_out_withDPD - x)**2) / sum(abs(x)**2))
    NMSE_withoutDPD = 10 * np.log10(sum(abs(y - x)**2) / sum(abs(x)**2))
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
    t = np.linspace(0, 1, 200)
    plt.plot(t,abs(PA_out_withDPD[0:200]),label = 'PA_Output')
    plt.plot(t,abs(x[0:200]),label = 'PA_Input')
    plt.plot(t,abs(DPD[0:200]),label = 'DPD')
    # plt.xlim(0,200)
    plt.ylim(0,1)
    plt.legend()
    plt.savefig('figures/DPD_waveform.png')
    # plt.show()
    # sleep(5)
    plt.close()


    a = 1