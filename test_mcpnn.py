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
import logging
import argparse
import time
import os
import Function_Lib as fun
import Function_Calculate as cal
from tqdm import tqdm

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


########################## 信号描述 ################################
fs = 2e9
BW = 400e6
# 读取PA输入输出信号
data_file = 'data/dataxy400m2G.mat'
# data_file = 'data/signal_100M_NR_fs49152.mat'
data = loadmat(data_file)

# xorg = x_data[]
xorg = data['x0']
yorg = data['y00']

x = xorg[0:50000].squeeze()
y = yorg[0:50000].squeeze()

N = len(xorg)
figure_path = 'figures/MCP_NN'
fun.PA_figure(x,y,figure_path)

# 模型设置
# Model_map = ['GMP','MP']
# Model_map = ['RVTD_NN']#,'MCP_NN']
Model_map = ['MCP_NN']
# Model_map = ['MCP_LSTM']

# 参数设置
M = 7  # 记忆深度，可修改
input_size = 2 * (M + 1)  # 输入维度
output_size = 2 * (M + 1)  # 输出维度
# output_size = 2
hidden_size = [20,18,18]  # 隐藏层维度
# hidden_layer = 5

layer_dims = []
layer_dims.append(input_size)
for size in hidden_size:
    layer_dims.append(size)
layer_dims.append(output_size)

learning_rate = 0.001
epochs = 100
batch_size = 512
activation="ReLU" #Tanh ReLU

plotwave = 0
model_train = 1

for Model in Model_map:
    # model_path = 'results/save/MCP_NN_M7_[16, 24, 24, 16]_ReLU_202505091035.pt'

    # logger.info('')
    # dataset = torch.utils.data.TensorDataset(X, Y)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if Model == 'MCP_NN':
        model_path = f"results/save/{Model}_M{M}_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
        filename = f"{Model}_M{M}_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}"
        parser = argparse.ArgumentParser(description='configTemplates')
        parser.add_argument('-log_path', default='./results/log', type=str, help='log file path to save result')
        args = parser.parse_args()
        logger = fun.create_logger(args.log_path, filename)
        logger.info(f'------signal {BW / 1e6}M {fs / 1e6}------')

        # 创建数据集
        X_train, X_val, Y_train, Y_val = fun.create_dataset(x, y, M)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset = TensorDataset(X_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # print(f'------------------------{Model}_M{M}_{layer_dims}_{activation}-------------------------------')
        logger.info(f'------------------------{Model}_M{M}_{layer_dims}_{activation}-------------------------------')
        # 初始化模型
        model = MCP_NN.MCP_NN(layer_dims, M,activation).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        fun.model_structure(model,logger)
        best_metric = float('inf')

        if model_train == 1:
            logger.info(f'------------------------Train Stage-------------------------------')
            train_loss_list = []
            val_loss_list = []
            # 训练循环
            start_time = time.time()  # 记录开始时间
            for epoch in range(epochs):
                model.train()
                i = 0
                train_loss = 0
                for inputs, targets in tqdm(train_loader):
                    # 获取对应的复数信号窗口 [batch, M+1]

                    reshaped_tensor = inputs.view(len(inputs), M+1, 2)
                    batch_x_signal = torch.view_as_complex(reshaped_tensor)
                    plotwave = 0
                    if plotwave == 1:
                        batch_x = np.stack(batch_x_signal[0:200])
                        batch_y = np.stack(torch.view_as_complex(targets)[0:200])#targets[0:200]
                        plt.figure()
                        t = np.linspace(0, 1, 200)
                        plt.plot(t, abs(batch_y[000:200]), label='y')
                        plt.plot(t, abs(batch_x[000:200]), label='x')
                        plt.ylim(0, 1)
                        plt.legend()
                        plt.show()
                        plt.savefig(f'figures/MCP_NN/{Model}_waveform.png')
                    # 将数据移动到设备上
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    batch_x_signal = batch_x_signal.to(device)

                    optimizer.zero_grad()
                    # outputs = model(inputs)
                    outputs = model(inputs, batch_x_signal)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    # i = i + 1
                    # print(f"batch{i} added loss:  {train_loss}")

                # 验证
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, targets in tqdm(val_loader):
                        # 将数据移动到设备上
                        # 获取对应的复数信号窗口 [batch, M+1]
                        reshaped_tensor = inputs.view(len(inputs), M + 1, 2)
                        batch_x_signal = torch.view_as_complex(reshaped_tensor)

                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        batch_x_signal = batch_x_signal.to(device)
                        val_outputs = model(inputs, batch_x_signal)
                        val_loss += criterion(val_outputs, targets).item()

                train_loss_list.append(train_loss / len(train_loader))
                val_loss_list.append(val_loss / len(val_loader))
                # 记录指标
                metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }

                # # 保存当前模型（按周期命名）
                # torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
                save = 0
                # 更新最佳模型
                if val_loss_list[-1] < best_metric:
                    best_metric = val_loss_list[-1]
                    save = 1
                    # 保存模型参数（推荐保存为 .pt 或 .pth 文件）
                    best_model = model.state_dict()
                # print(f'Epoch {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f} | save:{save}')
                logger.info(f'Epoch {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f} | save:{save}')
            torch.save(best_model, model_path)
            logger.info(f"model save: {model_path} ")
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            logger.info(f"model train time: {elapsed_time:.6f} s")
            if 1:
                y_train_loss = train_loss_list  # loss值，即y轴
                x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴

                plt.figure()

                # 去除顶部和右边框框
                ax = plt.axes()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                plt.xlabel('iters')  # x轴标签
                plt.ylabel('loss')  # y轴标签

                # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
                # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
                plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
                plt.legend()
                plt.title('Loss curve')
                plt.savefig(f'figures/MCP_NN/{Model}_loss_curve.png')
                plt.close()
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")

        x = xorg[50001:].squeeze()
        y = yorg[50001:].squeeze()
        start_time = time.time()  # 记录开始时间
        y_pred = model.apply_dpd(x, M)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")

    if Model == 'MCP_LSTM':
        M = 7
        seq_length = 1
        model_path = f"results/save/{Model}_M{M}_seqlen{seq_length}_{time.strftime('%Y%m%d%H%M')}.pt"
        # model_path = f"results/save/MCP_LSTM_M7_[16, 24, 24, 24, 24, 24, 2]_ReLU_202505091446.pt"
        filename = f"{Model}_M{M}_seqlen{seq_length}_{time.strftime('%Y%m%d%H%M')}"
        parser = argparse.ArgumentParser(description='configTemplates')
        parser.add_argument('-log_path', default='./results/log', type=str, help='log file path to save result')
        args = parser.parse_args()
        logger = fun.create_logger(args.log_path, filename)
        logger.info(f'------signal {BW / 1e6}M {fs / 1e6}------')

        logger.info(f'------------------------{Model}_M{M}_seqlen{seq_length}-------------------------------')
        # 初始化模型
        model = MCP_NN.MCP_LSTM( hidden_size=64, num_layers=1, M = M).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        fun.model_structure(model, logger)
        best_metric = float('inf')

        if model_train == 1:
            logger.info(f'------------------------Train Stage-------------------------------')
            train_loss_list = []
            val_loss_list = []
            X_train, X_val, Y_train, Y_val = model.create_dataset(x, y, M,seq_length,0.2)

            # 创建数据加载器
            batch_size = 256
            train_dataset = TensorDataset(X_train, Y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            val_dataset = TensorDataset(X_val, Y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            # 训练循环
            num_epochs = 100

            for epoch in range(num_epochs):
                model.train()
                # i = 0
                train_loss = 0
                for inputs, targets in tqdm(train_loader):
                    batch_x_signal = torch.stack([
                        window[-1][-2:] for window in inputs
                    ])[:inputs.shape[0]]  # 需优化以提高效率
                    # 转换为张量
                    # batch_x_signal = torch.view_as_real(batch_x_signal).to(torch.float32)
                    batch_x_signal = torch.view_as_complex(batch_x_signal)
                    plotwave = 0
                    if plotwave == 1:
                        batch_x = np.stack(batch_x_signal[0:200])
                        batch_y = np.stack(torch.view_as_complex(targets)[0:200])#targets[0:200]
                        plt.figure()
                        t = np.linspace(0, 1, 100)
                        plt.plot(t, abs(batch_y[000:100]), label='y')
                        plt.plot(t, abs(batch_x[000:100]), label='x')
                        plt.ylim(0, 1)
                        plt.legend()
                        # plt.show()
                        plt.savefig(f'figures/MCP_NN/{Model}_waveform.png')
                    # 将数据移动到设备上
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    batch_x_signal = batch_x_signal.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs,batch_x_signal)
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
                    for inputs, targets in tqdm(val_loader):
                        batch_x_signal = torch.stack([
                            window[-1][-2:] for window in inputs
                        ])[:inputs.shape[0]]  # 需优化以提高效率
                        # 转换为张量
                        # batch_x_signal = torch.view_as_real(batch_x_signal).to(torch.float32)
                        batch_x_signal = torch.view_as_complex(batch_x_signal)
                        # 将数据移动到设备上
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        batch_x_signal = batch_x_signal.to(device)
                        outputs = model(inputs, batch_x_signal)
                        val_loss += criterion(outputs, targets).item()

                train_loss_list.append(train_loss / len(train_loader))
                val_loss_list.append(val_loss / len(val_loader))
                # 记录指标
                metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }

                # # 保存当前模型（按周期命名）
                # torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
                save = 0
                # 更新最佳模型
                if val_loss_list[-1] < best_metric:
                    best_metric = val_loss_list[-1]
                    save = 1
                    # 保存模型参数（推荐保存为 .pt 或 .pth 文件）
                    best_model = model.state_dict()
                # print(f'Epoch {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f} | save:{save}')
                logger.info(
                    f'Epoch {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f} | save:{save}')
            torch.save(best_model, model_path)

        model.load_state_dict(torch.load(model_path))
        y_pred = model.apply_dpd(x,seq_length, M)

    if Model == 'RVTD_NN':
        model_path = f"results/save/{Model}_M{M}_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
        filename = f"{Model}_M{M}_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}"
        parser = argparse.ArgumentParser(description='configTemplates')
        parser.add_argument('-log_path', default='./results/log', type=str, help='log file path to save result')
        args = parser.parse_args()
        logger = fun.create_logger(args.log_path, filename)
        logger.info(f'------signal {BW / 1e6}M {fs / 1e6}------')

        # 创建数据集
        X_train, X_val, Y_train, Y_val = fun.create_dataset(x, y, M)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset = TensorDataset(X_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # print(f'------------------------{Model}_M{M}_{layer_dims}_{activation}-------------------------------')
        logger.info(f'------------------------{Model}_M{M}_{layer_dims}_{activation}-------------------------------')
        # 初始化模型
        model = MCP_NN.RVTD_NN(layer_dims, activation).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        fun.model_structure(model,logger)
        best_metric = float('inf')

        if model_train == 1:
            logger.info(f'------------------------Train Stage-------------------------------')
            train_loss_list = []
            val_loss_list = []
            # 训练循环
            for epoch in range(epochs):
                model.train()
                i = 0
                train_loss = 0
                for inputs, targets in tqdm(train_loader):
                    # 获取对应的复数信号窗口 [batch, M+1]
                    # 将数据移动到设备上
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    # outputs = model(inputs)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    # i = i + 1
                    # print(f"batch{i} added loss:  {train_loss}")

                # 验证
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, targets in tqdm(val_loader):
                        # 将数据移动到设备上
                        # 获取对应的复数信号窗口 [batch, M+1]
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        val_outputs = model(inputs)
                        val_loss += criterion(val_outputs, targets).item()

                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)
                # 记录指标
                metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
                # 更新最佳模型
                save = 0
                if val_loss < best_metric:
                    best_metric = val_loss
                    save = 1
                    # 保存模型参数（推荐保存为 .pt 或 .pth 文件）
                    best_model = model.state_dict()
                # print(f'Epoch {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f} | save:{save}')
                logger.info(f'Epoch {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f} | save:{save}')
            torch.save(model.state_dict(), model_path)

        model.load_state_dict(torch.load(model_path))
        y_pred = model.apply_dpd(x, M)


    x_norm = x/max(abs(x))
    y_norm = y/max(abs(y))
    y_pred_norm = y_pred/max(abs(y_pred))

    # 评估结果（示例）
    # NMSE_pred = 10 * np.log10(sum(abs(y_pred_norm - y_norm)**2) / sum(abs(y_norm)**2))
    logger.info(f"signal 400M:")
    NMSE_pred = cal.nmse(x,y,logger)
    ACLR_pred = cal.acpr(y,fs,BW,BW,logger)

    logger.info(f"{Model} modeling:")
    NMSE_pred = cal.nmse(y_norm,y_pred_norm,logger)
    ACLR_pred = cal.acpr(y_pred,fs,BW,BW,logger)

    # logger.info(f"NMSE with {Model}: {NMSE_2:.3f} dB")
    # print(f"NMSE wo   DPD: {NMSE_withoutDPD} dB")
        #
    # fs = 2e9
        # # print("without DPD")
        # # acpr_wo_DPD = metrics.acpr(y,fs,100e6,100e6)
        # # print("with DPD")
        # # acpr_with_DPD = metrics.acpr(PA_out_withDPD,fs,100e6,100e6)
        #
        #

    plot.psd(
        {"input": x,"pred_output":y_pred,"output":y},
        fs=fs,filename=f'figures/MCP_NN/{Model}_spec.png'
    )
    plot.amam(x, {"out":y,"pred":y_pred}, f"figures/MCP_NN/{Model}_amam.png")
    plot.ampm(x, {"out": y, "pred": y_pred}, f"figures/MCP_NN/{Model}_ampm.png")
    plt.figure()
    t = np.linspace(0, 1, 200)
    plt.plot(t,abs(y_pred[300:500]),label = 'y_pred')
    plt.plot(t,abs(y[300:500]),label = 'y')
    # plt.plot(t,abs(DPD[2000:2200]),label = 'DPD')
    # plt.xlim(0,200)
    plt.ylim(0,1)
    plt.legend()
    plt.savefig(f'figures/MCP_NN/{Model}_waveform.png')

A = 1