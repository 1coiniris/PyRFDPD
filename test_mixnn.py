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
import Model.Mixed_NN as MIX_NN
import Model.DVR as DVR
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
# signal = '400M' #'LMBA200M'
signal = 'LMBA200M'
# signal = '400M'
# signal = 'ILC_120M'
# signal = 'ILC'
# 读取PA输入输出信号
if signal == '400M':
    fs = 2e9
    BW = 400e6
    data_file = 'data/dataxy400m2G.mat'
    data = loadmat(data_file)
    xorg = data['x0']
    yorg = data['y00']
elif signal == 'LMBA200M':
    fs = 1e9
    BW = 200e6
    data_file = 'data/LMBA_200M_23G.mat'
    data = loadmat(data_file)
    xorg = data['x']
    yorg = data['y']
elif signal == '100M':
    fs = 983.04e6
    BW = 100e6
    data_file = 'data/signal_100M_fs98304.mat'
    data = loadmat(data_file)
    xorg = data['x'].T
    yorg = data['y'].T
elif signal == 'ILC_120M':
    fs = 1.2288e9
    BW = 120e6
    data_file = 'data/ILC.mat'
    data = loadmat(data_file)
    xorg = data['uBB']
    # ILCOut = data['x']
    yorg = data['xBB']
elif signal == 'ILC':
    fs = 1.2288e9
    BW = 200e6
    data_file = 'data/ILC_[0  1  1  1  0]_G1_forpython.mat'
    data = loadmat(data_file)
    xorg = data['x']
    # ILCOut = data['x']
    yorg = data['y_ILC']
N = len(xorg)
last_train = int(N*0.6-1)

x = xorg[0:last_train].squeeze()
y = yorg[0:last_train].squeeze()

N = len(xorg)
figure_path = 'figures/MCP_NN'
fun.PA_figure(x,y,fs,figure_path)

# 模型设置
# Model = 'DVC_NN'
# Model = 'GMP_NN'
# Model = 'RVTD_NN'
# Model = 'MCP_BASE_NN'
# Model = 'PARA_NN'
# Model = 'DVR'
Model = 'DVR_NN'
# model_map = []
# for M in range(5,7,1):
#     for L in range(M-3,M-1):
#         inputsize = M+2*L+1
#         outputsize = 2*M+2
#         # model_map.append([M, L, 22, 30, 24])
#         for layer1 in range(8,16,2):
#             for layer2 in range(8,16,2):
#                 model_map.append(['ReLU',M, L, layer1, layer2])
#                 model_map.append(['Tanh', M, L, layer1, layer2])
                # for layer3 in range(2*M+6, 2*M+18, 4):
                #     model_map.append([M,L,layer1,layer2,layer3])
# model = [['Tanh',11,0,12,10,12,12],['Tanh',11,0,12]]#,['ReLU',5,3,12,12,10]]
model = ['GELU',3,15] #DVR [K,M]
threshold = np.array([0.2,0.6,0.8])
learning_rate = 0.001
epochs = 200
batch_size = 256
# activation= "ReLU" #Tanh ReLU
# activation= "Tanh"
model_train = 1
total_train = 1

NMSE_list = []

filename = f"{Model}_{time.strftime('%Y%m%d%H')}"
parser = argparse.ArgumentParser(description='configTemplates')
parser.add_argument('-log_path', default='./results/20250525/log/', type=str, help='log file path to save result')
args = parser.parse_args()
logger = fun.create_logger(args.log_path, filename)


# model_path = 'results/save/MCP_NN_M7_[16, 24, 24, 16]_ReLU_202505091035.pt'
if Model == 'DVR_NN':
    # epochs = 30
    # 参数设置
    activation = model[0]
    K = model[1]  # 分段数，可修改
    M = model[2]
    # size = (M + 1 + Lb + Lc)

    model_path = f"results/20250519/save/{Model}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

    trained_model = 'results/20250519/save/DVR_NN_M15_K3_Tanh_202507071808.pt'
    # filename = f"{Model}_M{M}_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}"
    # parser = argparse.ArgumentParser(description='configTemplates')
    # parser.add_argument('-log_path', default='./results/20250517/log/', type=str, help='log file path to save result')
    # args = parser.parse_args()
    # logger = fun.create_logger(args.log_path, filename)
    logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')

    # print(f'------------------------{Model}_M{M}_{layer_dims}_{activation}-------------------------------')
    logger.info(f'------------------------{Model}_M{M}_K{K}_{activation}-------------------------------')
    # 初始化模型
    model = DVR.DVR_NN( K=K, M=M, activation=activation).to(device)
    if total_train == 0:
        model.load_state_dict(torch.load(trained_model))
        optimizer = optim.Adam(model.linear.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    fun.model_structure(model,logger)
    best_metric = float('inf')

    # 训练前的模型参数
    # print("model.amp_linear.weight", model.amp_linear.weight)
    # print("model.DDR_linear.weight", model.DDR_linear.weight)
    # print("model.linear.weight", model.linear.weight)
    # 创建数据集
    x = xorg[0:last_train].squeeze()
    y = yorg[0:last_train].squeeze()
    X_train, X_val, Y_train, Y_val = model.create_dataset(x, y)
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if model_train == 1:
        logger.info(f'------------------------Train Stage-------------------------------')
        train_loss_list = []
        val_loss_list = []
        # 训练循环
        start_time = time.time()  # 记录开始时间
        nosave_count = 0
        for epoch in range(epochs):
            model.train()
            i = 0
            train_loss = 0
            for inputs, targets in tqdm(train_loader):
                # 获取对应的复数信号窗口 [batch, M+1]
                optimizer.zero_grad()

                batch_x_signal = inputs.to(device)
                targets = torch.view_as_real(targets)
                targets = targets.to(device)
                # outputs = model(inputs)
                outputs = model(batch_x_signal)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # PA_out = targets.cpu()
                # PA_out_pred = outputs.cpu()
                # PA_out = torch.view_as_complex(PA_out).detach().numpy()
                # PA_out_pred = torch.view_as_complex(PA_out_pred).detach().numpy()
                # train_nmse = cal.nmse(PA_out, PA_out_pred,logger)

                # i = i + 1
                # print(f"batch{i} added loss:  {train_loss}")

            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader):
                    # 将数据移动到设备上
                    # 获取对应的复数信号窗口 [batch, M+1]
                    batch_x_signal = inputs.to(device)
                    targets = torch.view_as_real(targets)
                    targets = targets.to(device)
                    # outputs = model(inputs)
                    val_outputs = model(batch_x_signal)
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
            if save == 0:
                nosave_count = nosave_count + 1
            else:
                nosave_count = 0
            if nosave_count > 100:
                break

        torch.save(best_model, model_path)
        logger.info(f"model save: {model_path} ")
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model train time: {elapsed_time:.6f} s")
        # 训练前的模型参数
        # print("model.amp_linear.weight", model.amp_linear.weight)
        # print("model.DDR_linear.weight", model.DDR_linear.weight)
        # print("model.linear.weight", model.linear.weight)

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

    x = xorg[last_train+1:].squeeze()
    y = yorg[last_train+1:].squeeze()
    start_time = time.time()  # 记录开始时间
    y_pred = model.apply_dpd(x)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time
    logger.info(f"model prediction time: {elapsed_time:.6f} s")
if Model == 'PARA_NN':
    # epochs = 30
    # 参数设置
    activation = model[0][0]
    M = model[0][1]  # 记忆深度，可修改
    Lb = model[0][2]
    Lc = model[0][2]
    K = 5
    M_len = (M + 1 + Lb + Lc)
    # size = (M + 1 + Lb + Lc)
    input_size = 1*(M + 1 + Lb + Lc) # 输入维度
    output_size = 2*(M+1)  # 输出维度
    # output_size = 2
    hidden_size_1 = []
    hidden_size_2 = []
    for hidden in model[0][3:]:
        hidden_size_1.append(hidden) # 隐藏层维度
    # hidden_layer = 5
    for hidden in model[1][3:]:
        hidden_size_2.append(hidden) # 隐藏层维度

    layer_dim1 = []
    layer_dim2 = []
    layer_dim1.append(input_size)
    layer_dim2.append(input_size)
    layer_dim2.append(2*K)
    for size in hidden_size_1:
        layer_dim1.append(size)
    for size in hidden_size_2:
        layer_dim2.append(size)
    layer_dim1.append(output_size)
    layer_dim2.append(output_size)
    layer_dims = [layer_dim1, layer_dim2]
    # model_path = 'results/save/GMP_NN_M7_[15, 12, 16, 20, 16]_ReLU_202505101439.pt'
    model_path = f"results/20250519/save/{Model}_M{M}_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
    # filename = f"{Model}_M{M}_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}"
    # parser = argparse.ArgumentParser(description='configTemplates')
    # parser.add_argument('-log_path', default='./results/20250517/log/', type=str, help='log file path to save result')
    # args = parser.parse_args()
    # logger = fun.create_logger(args.log_path, filename)
    logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')

    # print(f'------------------------{Model}_M{M}_{layer_dims}_{activation}-------------------------------')
    logger.info(f'------------------------{Model}_M{M}_{layer_dims}_{activation}-------------------------------')
    logger.info(f'M{M}_Lb{Lb}_Lc{Lc}')
    # 初始化模型
    model = MIX_NN.Parallel_NN(layer_dims, [Lb,Lc], M,activation).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    fun.model_structure(model,logger)
    best_metric = float('inf')

    # 创建数据集
    x = xorg[0:last_train].squeeze()
    y = yorg[0:last_train].squeeze()
    X_train, X_val, Y_train, Y_val = model.create_dataset(x, y, [Lb,Lc],M)
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if model_train == 1:
        logger.info(f'------------------------Train Stage-------------------------------')
        train_loss_list = []
        val_loss_list = []
        # 训练循环
        start_time = time.time()  # 记录开始时间
        nosave_count = 0
        for epoch in range(epochs):
            model.train()
            i = 0
            train_loss = 0
            for inputs, targets in tqdm(train_loader):
                # 获取对应的复数信号窗口 [batch, M+1]
                optimizer.zero_grad()

                reshaped_tensor = inputs.view(len(inputs), M_len, 2)
                input_tensor = torch.view_as_complex(reshaped_tensor)
                x_tensor = reshaped_tensor[:,Lb:M+Lb+1,:]
                batch_x_signal = torch.view_as_complex(x_tensor)

                # 将数据移动到设备上
                batch_x_magnitude_1 = abs(input_tensor)
                # batch_x_magnitude_3 = abs(input_tensor)**3
                # batch_x_magnitude_5 = abs(input_tensor)**5
                # batch_x_magnitude = torch.cat((batch_x_magnitude_1, batch_x_magnitude_3, batch_x_magnitude_5),dim=1)
                batch_x_magnitude = batch_x_magnitude_1.to(device)
                # inputs = inputs.to(device)
                targets = targets.to(device)
                batch_x_signal = batch_x_signal.to(device)


                # outputs = model(inputs)
                outputs = model(batch_x_magnitude, batch_x_signal)

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
                    reshaped_tensor = inputs.view(len(inputs), M_len, 2)
                    input_tensor = torch.view_as_complex(reshaped_tensor)
                    x_tensor = reshaped_tensor[:,Lb:M+Lb+1,:]
                    batch_x_signal = torch.view_as_complex(x_tensor)
                    batch_x_magnitude_1 = abs(input_tensor)
                    # batch_x_magnitude_3 = abs(input_tensor) ** 3
                    # batch_x_magnitude_5 = abs(input_tensor) ** 5
                    # batch_x_magnitude = torch.cat((batch_x_magnitude_1, batch_x_magnitude_3, batch_x_magnitude_5), dim=1)
                    batch_x_magnitude = batch_x_magnitude_1.to(device)
                    # inputs = inputs.to(device)
                    targets = targets.to(device)
                    batch_x_signal = batch_x_signal.to(device)
                    val_outputs = model(batch_x_magnitude, batch_x_signal)
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
            if save == 0:
                nosave_count = nosave_count + 1
            else:
                nosave_count = 0
            if nosave_count > 15:
                break

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

    x = xorg[last_train+1:].squeeze()
    y = yorg[last_train+1:].squeeze()
    start_time = time.time()  # 记录开始时间
    y_pred = model.apply_dpd(x)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time
    logger.info(f"model prediction time: {elapsed_time:.6f} s")

if Model == 'DVR':
    # epochs = 30
    # 参数设置
    activation = model[0]
    K = model[1]  # 分段数，可修改
    M = model[2]
    # size = (M + 1 + Lb + Lc)

    model_path = f"results/20250519/save/{Model}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
    # filename = f"{Model}_M{M}_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}"
    # parser = argparse.ArgumentParser(description='configTemplates')
    # parser.add_argument('-log_path', default='./results/20250517/log/', type=str, help='log file path to save result')
    # args = parser.parse_args()
    # logger = fun.create_logger(args.log_path, filename)
    logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')

    # print(f'------------------------{Model}_M{M}_{layer_dims}_{activation}-------------------------------')
    logger.info(f'------------------------{Model}_M{M}_K{K}_{activation}-------------------------------')
    # 初始化模型
    model = DVR.DVR( K=K, M=M, threshold=threshold,activation=activation).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    fun.model_structure(model,logger)
    best_metric = float('inf')

    # 创建数据集
    x = xorg[0:last_train].squeeze()
    y = yorg[0:last_train].squeeze()
    X_train, X_val, Y_train, Y_val = model.create_dataset(x, y)
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if model_train == 1:
        logger.info(f'------------------------Train Stage-------------------------------')
        train_loss_list = []
        val_loss_list = []
        # 训练循环
        start_time = time.time()  # 记录开始时间
        nosave_count = 0
        for epoch in range(epochs):
            model.train()
            i = 0
            train_loss = 0
            for inputs, targets in tqdm(train_loader):
                # 获取对应的复数信号窗口 [batch, M+1]
                optimizer.zero_grad()

                batch_x_signal = inputs.to(device)
                targets = torch.view_as_real(targets)
                targets = targets.to(device)
                # outputs = model(inputs)
                outputs = model(batch_x_signal)

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
                    batch_x_signal = inputs.to(device)
                    targets = torch.view_as_real(targets)
                    targets = targets.to(device)
                    # outputs = model(inputs)
                    val_outputs = model(batch_x_signal)
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
            if save == 0:
                nosave_count = nosave_count + 1
            else:
                nosave_count = 0
            if nosave_count > 15:
                break

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

    x = xorg[last_train+1:].squeeze()
    y = yorg[last_train+1:].squeeze()
    start_time = time.time()  # 记录开始时间
    y_pred = model.apply_dpd(x)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time
    logger.info(f"model prediction time: {elapsed_time:.6f} s")

x_norm = x/max(abs(x))
y_norm = y/max(abs(y))
y_pred_norm = y_pred/max(abs(y_pred))

# 评估结果（示例）
# NMSE_pred = 10 * np.log10(sum(abs(y_pred_norm - y_norm)**2) / sum(abs(y_norm)**2))
logger.info(f"signal {signal}")
NMSE = cal.nmse(x,y,logger)
ACLR = cal.acpr(y,fs,BW,BW,logger)



logger.info(f"{Model} modeling:")
NMSE_pred = cal.nmse(y_norm,y_pred_norm,logger)
ACLR_pred = cal.acpr(y_pred,fs,BW,BW,logger)
NMSE_list.append(NMSE_pred)
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
plot.amam(x, {"out":y,"pred":y_pred}, norm=0, filename=f"figures/MCP_NN/{Model}_amam.png")
plot.ampm(x, {"out": y, "pred": y_pred}, norm=0, filename=f"figures/MCP_NN/{Model}_ampm.png")
plt.figure()
t = np.linspace(0, 1, 200)
plt.plot(t,abs(y_pred[300:500]),label = 'y_pred')
plt.plot(t,abs(y[300:500]),label = 'y')
# plt.plot(t,abs(DPD[2000:2200]),label = 'DPD')
# plt.xlim(0,200)
plt.ylim(0,1)
plt.legend()
plt.savefig(f'figures/MCP_NN/{Model}_waveform.png')


# for para,nmse in zip(model[0],NMSE_list):
#     logger.info(f"{para[0]}_M{para[1]}_L{para[2]}_{para[3:]}_nmse{nmse}dB")
#
# best = min(NMSE_list)
# key = NMSE_list.index(best)
#
# logger.info(f"best NMSE {best} | best para{model[key]}")


A = 1

# DPD_with_GMPNN = model.apply_dpd(xorg)
# print(max(abs(DPD_with_GMPNN)))
# # 保存数据到MAT文件
# file_name = 'data/GMPNN.mat'
# savemat(file_name, {'DPD': DPD_with_GMPNN.T, 'ILC': yorg, 'X': xorg})