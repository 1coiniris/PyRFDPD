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
import Model.Orth_NN as ORTH_NN
import Model.VDTDNN as VDTDNN
import Model.RVTDNN as RVTDNN
import Model.PARA_VD_NN as VD_NN
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


plot_swich = 0
NMSE_state_list = []
########################## 信号描述 ################################
# signal = '400M' #'LMBA200M'
# signal = 'LMBA200M'
# signal = '100M'
# signal = 'ILC_120M'
# signal = 'ILC'
signal = 'YU'

for state in range(24):
    x_train, y_train, x, y, fs, BW = fun.get_data(signal,state = state)
    figure_path = 'figures/MCP_NN'
    if plot_swich:
        fun.PA_figure(x,y,fs,figure_path)

    # 模型设置
    # Model = 'DVC_NN'
    # Model = 'GMP_NN'
    # Model = 'RVTD_NN'
    # Model = 'MCP_BASE_NN'
    # Model = 'PARA_NN'
    # Model = 'DVR'
    # Model = 'ADVR_NN'
    # Model = 'OB_DVR_NN'
    # Model = 'VD_DVR_NN'
    # Model = 'VDTDNN'
    Model = 'RVTDNN'
    # Model = 'DVR_NN'

    model = ['Tanh',3,30,40] #DVR [K,M]
    threshold = np.array([0.2,0.6,0.8])
    learning_rate = 0.001
    epochs = 400
    batch_size = 1024
    # activation= "ReLU" #Tanh ReLU

    total_train = 1

    NMSE_list = []

    filename = f"{Model}_{time.strftime('%Y%m%d%H')}"
    if state==0:
        parser = argparse.ArgumentParser(description='configTemplates')
        parser.add_argument('-log_path', default='./results/20250525/log/', type=str, help='log file path to save result')
        args = parser.parse_args()
        logger = fun.create_logger(args.log_path, filename)


    # model_path = 'results/save/MCP_NN_M7_[16, 24, 24, 16]_ReLU_202505091035.pt'
    if Model == 'VD_DVR_NN':
        # 参数设置
        activation = model[0]
        K = model[1]  # 分段数，可修改
        M = model[2]
        M2 = model[3]

        model_path = f"results/20250519/save/{Model}_M{M}_M2{M2}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        # trained_model = 'results/20250519/save/OB_DVR_NN_M15_K3_Tanh_202507201242.pt'# 好用
        trained_model = 'results/20250519/save/VD_DVR_NN_M30_M231_K3_Tanh_202508051529.pt' #很好
        # trained_model = 'results/20250519/save/VD_DVR_NN_M10_K1_Tanh_202508031410.pt' #LMBA
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        logger.info(f'------------------------{Model}_M{M}_M2{M2}_K{K}_{activation}-------------------------------')
        # 初始化模型
        model = VD_NN.VD_DVR_NN( K=K, M=M,M2=M2, activation=activation).to(device)
        fun.model_structure(model, logger)


        if total_train == 1:
            # model.load_state_dict(torch.load(trained_model))
            model.model_train(x_train,y_train,model_path, logger,1)
            model.load_state_dict(torch.load(model_path))
            logger.info(f"-------------------load model: {model_path}---------------------")
            start_time = time.time()  # 记录开始时间
            # 提取参数
            x_coef = x_train[:]
            y_coef = y_train[:]
            x_coef_tensor = torch.from_numpy(x_coef).to(device)
            y_coef_tensor = torch.from_numpy(y_coef).to(device)
            sequences = MCP_NN.create_memory_seq(x_coef, M2)  # [N, M+1]
            x_coef_window = torch.from_numpy(sequences).to(device)
            y_sequences = MCP_NN.create_memory_seq(y_coef, M2)  # [N, M+1]
            y_coef_window = torch.from_numpy(y_sequences).to(device)
            model.coef = model.DVR_NN_e(x_coef_window,y_coef_tensor)
            y_pred = model.apply_dpd(x,model.coef)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            logger.info(f"model prediction time: {elapsed_time:.6f} s")
        else:
            N_train = min(50000,len(x_train))
            model.load_state_dict(torch.load(trained_model))
            # 提取参数
            x_coef = x_train[0:N_train - 1]
            y_coef = y_train[0:N_train - 1]
            x_coef_tensor = torch.from_numpy(x_coef).to(device)
            y_coef_tensor = torch.from_numpy(y_coef).to(device)
            sequences = MCP_NN.create_memory_seq(x_coef, M2)  # [N, M+1]
            x_coef_window = torch.from_numpy(sequences).to(device)
            y_sequences = MCP_NN.create_memory_seq(y_coef, M)  # [N, M+1]
            y_coef_window = torch.from_numpy(y_sequences).to(device)
            coef = model.DVR_NN_e(x_coef_window,y_coef_tensor,alpha=5e-3,pri=1)
            logger.info(f"model coef number: {len(coef)} ")
            # 建模
            sequences = MCP_NN.create_memory_seq(x, M2)  # [N, M+1]
            x_window = torch.from_numpy(sequences).to(device)
            y_pred = model.DVR_NN_v(x_window,coef).cpu().detach().numpy()
            X = model.get_basis(x_window)
            # # DPD建模
            # DPD = ORTH_NN.Orth_Basis_DVR_NN( K=K, M=M, activation=activation).to(device)
            # DPD.load_state_dict(torch.load(trained_model))
            # DPD_coef = DPD.DVR_NN_e(y_coef_window, x_coef_tensor)
            # # DPD
            # sequences = MCP_NN.create_memory_seq(x, M)  # [N, M+1]
            # x_window = torch.from_numpy(sequences).to(device)
            # DPD_out = DPD.DVR_NN_v(x_window,DPD_coef).cpu().detach().numpy()
            # # PA
            # sequences = MCP_NN.create_memory_seq(DPD_out, M)  # [N, M+1]
            # x_window = torch.from_numpy(sequences).to(device)
            # PA_out = model.DVR_NN_v(x_window,coef).cpu().detach().numpy()
            # NMSE_with_DPD = cal.nmse(x,PA_out,logger,1)
            # plot.amam(x, {"out": y, "dpd": PA_out}, norm=0, filename=f"figures/MCP_NN/{Model}_amam.png")

    if Model == 'OB_DVR_NN':
        # 参数设置
        activation = model[0]
        K = model[1]  # 分段数，可修改
        M = model[2]

        model_path = f"results/20250519/save/{Model}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        # trained_model = 'results/20250519/save/OB_DVR_NN_M15_K3_Tanh_202507201242.pt'# 好用
        # trained_model = 'results/20250519/save/OB_DVR_NN_M30_K1_Tanh_202507262149.pt' #很好
        trained_model = 'results/20250519/save/OB_DVR_NN_M30_K1_Tanh_202507262348.pt' #LMBA
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        logger.info(f'------------------------{Model}_M{M}_K{K}_{activation}-------------------------------')
        # 初始化模型
        model = ORTH_NN.Orth_Basis_DVR_NN( K=K, M=M, activation=activation).to(device)
        fun.model_structure(model, logger)


        if total_train == 1:
            # model.load_state_dict(torch.load(trained_model))
            model.model_train(x_train,y_train,model_path, logger,1)
            model.load_state_dict(torch.load(model_path))
            logger.info(f"-------------------load model: {model_path}---------------------")
            start_time = time.time()  # 记录开始时间
            # 提取参数
            x_coef = x_train[:]
            y_coef = y_train[:]
            x_coef_tensor = torch.from_numpy(x_coef).to(device)
            y_coef_tensor = torch.from_numpy(y_coef).to(device)
            sequences = MCP_NN.create_memory_seq(x_coef, M)  # [N, M+1]
            x_coef_window = torch.from_numpy(sequences).to(device)
            y_sequences = MCP_NN.create_memory_seq(y_coef, M)  # [N, M+1]
            y_coef_window = torch.from_numpy(y_sequences).to(device)
            model.coef = model.DVR_NN_e(x_coef_window,y_coef_tensor)
            y_pred = model.apply_dpd(x,model.coef)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            logger.info(f"model prediction time: {elapsed_time:.6f} s")
            print(f'COEF number: {len(model.coef)} ')
        else:
            N_train = min(50000,len(x_train))
            model.load_state_dict(torch.load(trained_model))
            # 提取参数
            x_coef = x_train[0:N_train - 1]
            y_coef = y_train[0:N_train - 1]
            x_coef_tensor = torch.from_numpy(x_coef).to(device)
            y_coef_tensor = torch.from_numpy(y_coef).to(device)
            sequences = MCP_NN.create_memory_seq(x_coef, M)  # [N, M+1]
            x_coef_window = torch.from_numpy(sequences).to(device)
            y_sequences = MCP_NN.create_memory_seq(y_coef, M)  # [N, M+1]
            y_coef_window = torch.from_numpy(y_sequences).to(device)
            coef = model.DVR_NN_e(x_coef_window,y_coef_tensor,alpha=1e-2,pri=1)
            # 建模
            sequences = MCP_NN.create_memory_seq(x, M)  # [N, M+1]
            x_window = torch.from_numpy(sequences).to(device)
            y_pred = model.DVR_NN_v(x_window,coef).cpu().detach().numpy()
            X = model.get_basis(x_window)
            # # DPD建模
            # DPD = ORTH_NN.Orth_Basis_DVR_NN( K=K, M=M, activation=activation).to(device)
            # DPD.load_state_dict(torch.load(trained_model))
            # DPD_coef = DPD.DVR_NN_e(y_coef_window, x_coef_tensor)
            # # DPD
            # sequences = MCP_NN.create_memory_seq(x, M)  # [N, M+1]
            # x_window = torch.from_numpy(sequences).to(device)
            # DPD_out = DPD.DVR_NN_v(x_window,DPD_coef).cpu().detach().numpy()
            # # PA
            # sequences = MCP_NN.create_memory_seq(DPD_out, M)  # [N, M+1]
            # x_window = torch.from_numpy(sequences).to(device)
            # PA_out = model.DVR_NN_v(x_window,coef).cpu().detach().numpy()
            # NMSE_with_DPD = cal.nmse(x,PA_out,logger,1)
            # plot.amam(x, {"out": y, "dpd": PA_out}, norm=0, filename=f"figures/MCP_NN/{Model}_amam.png")

    if Model == 'ADVR_NN':
        # 参数设置
        activation = model[0]
        K = model[1]  # 分段数，可修改
        M = model[2]

        model_path = f"results/20250519/save/{Model}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        # trained_model = 'results/20250519/save/OB_DVR_NN_M15_K3_Tanh_202507201242.pt'# 好用
        # trained_model = 'results/20250519/save/OB_DVR_NN_M30_K1_Tanh_202507262149.pt' #很好
        trained_model = 'results/20250519/save/ADVR_NN_M30_K1_Tanh_202507272253.pt'  # 400M
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        logger.info(f'------------------------{Model}_M{M}_K{K}_{activation}-------------------------------')
        # 初始化模型
        model = DVR.ADVR_NN(K=K, M=M, activation=activation).to(device)
        fun.model_structure(model, logger)

        if total_train == 1:
            # model.load_state_dict(torch.load(trained_model))
            model.model_train(x_train, y_train, model_path, logger, 1)
            model.load_state_dict(torch.load(model_path))
            logger.info(f"-------------------load model: {model_path}---------------------")
            start_time = time.time()  # 记录开始时间
            # 提取参数
            x_coef = x_train[:]
            y_coef = y_train[:]
            x_coef_tensor = torch.from_numpy(x_coef).to(device)
            y_coef_tensor = torch.from_numpy(y_coef).to(device)
            sequences = MCP_NN.create_memory_seq(x_coef, M)  # [N, M+1]
            x_coef_window = torch.from_numpy(sequences).to(device)
            y_sequences = MCP_NN.create_memory_seq(y_coef, M)  # [N, M+1]
            y_coef_window = torch.from_numpy(y_sequences).to(device)
            model.coef = model.DVR_NN_e(x_coef_window, y_coef_tensor)
            y_pred = model.apply_dpd(x, model.coef)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            logger.info(f"model prediction time: {elapsed_time:.6f} s")
        else:
            N_train = min(50000, len(x_train))
            model.load_state_dict(torch.load(trained_model))
            # 提取参数
            x_coef = x_train[0:N_train - 1]
            y_coef = y_train[0:N_train - 1]
            x_coef_tensor = torch.from_numpy(x_coef).to(device)
            y_coef_tensor = torch.from_numpy(y_coef).to(device)
            sequences = MCP_NN.create_memory_seq(x_coef, M)  # [N, M+1]
            x_coef_window = torch.from_numpy(sequences).to(device)
            y_sequences = MCP_NN.create_memory_seq(y_coef, M)  # [N, M+1]
            y_coef_window = torch.from_numpy(y_sequences).to(device)
            coef = model.DVR_NN_e(x_coef_window, y_coef_tensor, alpha=1e-2, pri=1)
            # 建模
            sequences = MCP_NN.create_memory_seq(x, M)  # [N, M+1]
            x_window = torch.from_numpy(sequences).to(device)
            y_pred = model.DVR_NN_v(x_window, coef).cpu().detach().numpy()
            X = model.get_basis(x_window)
            # # DPD建模
            # DPD = ORTH_NN.Orth_Basis_DVR_NN( K=K, M=M, activation=activation).to(device)
            # DPD.load_state_dict(torch.load(trained_model))
            # DPD_coef = DPD.DVR_NN_e(y_coef_window, x_coef_tensor)
            # # DPD
            # sequences = MCP_NN.create_memory_seq(x, M)  # [N, M+1]
            # x_window = torch.from_numpy(sequences).to(device)
            # DPD_out = DPD.DVR_NN_v(x_window,DPD_coef).cpu().detach().numpy()
            # # PA
            # sequences = MCP_NN.create_memory_seq(DPD_out, M)  # [N, M+1]
            # x_window = torch.from_numpy(sequences).to(device)
            # PA_out = model.DVR_NN_v(x_window,coef).cpu().detach().numpy()
            # NMSE_with_DPD = cal.nmse(x,PA_out,logger,1)
            # plot.amam(x, {"out": y, "dpd": PA_out}, norm=0, filename=f"figures/MCP_NN/{Model}_amam.png")

    if Model == 'DVR_NN':
        # epochs = 30
        # 参数设置
        activation = model[0]
        K = model[1]  # 分段数，可修改
        M = model[2]
        # size = (M + 1 + Lb + Lc)

        model_path = f"results/20250519/save/{Model}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        trained_model = 'results/20250519/save/DVR_NN_M15_K3_Tanh_202507201202.pt'
        # trained_model = 'results/20250519/save/DVR_NN_M15_K3_Tanh_202507181901.pt'
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        logger.info(f'------------------------{Model}_M{M}_K{K}_{activation}-------------------------------')
        # 初始化模型
        model = DVR.DVR_NN( K=K, M=M, activation=activation).to(device)


        if total_train == 1:
            # model.load_state_dict(torch.load(trained_model))
            # 训练前的模型参数
            # print("model.amp_linear.weight", model.amp_linear.weight)
            # print("model.DDR_linear.weight", model.DDR_linear.weight)
            # print("model.linear.weight", model.linear.weight)
            model.model_train(x_train,y_train,model_path, logger,1)
            # 训练后的模型参数
            # print("model.amp_linear.weight", model.amp_linear.weight)
            # print("model.DDR_linear.weight", model.DDR_linear.weight)
            # print("model.linear.weight", model.linear.weight)
            model.load_state_dict(torch.load(model_path))
            logger.info(f"-------------------load model: {model_path}---------------------")
            start_time = time.time()  # 记录开始时间
            y_pred = model.apply_dpd(x)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            logger.info(f"model prediction time: {elapsed_time:.6f} s")
        else:
            N_train = min(5000,len(x_train))
            model.load_state_dict(torch.load(trained_model))
            coef = model.DVR_NN_e(x_train[0:N_train-1],y_train[0:N_train-1])
            y_pred = model.DVR_NN_v(x,coef)
            sequences = MCP_NN.create_memory_seq(x, M)  # [N, M+1]
            x_window = torch.from_numpy(sequences).to(device)
            X = model.get_basis(x_window)

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

        X_train, X_val, Y_train, Y_val = model.create_dataset(x, y, [Lb,Lc],M)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset = TensorDataset(X_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if total_train == 1:
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
        X_train, X_val, Y_train, Y_val = model.create_dataset(x, y)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset = TensorDataset(X_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if total_train == 1:
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

        start_time = time.time()  # 记录开始时间
        y_pred = model.apply_dpd(x)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")

    if Model == 'VDTDNN':
        # 参数设置
        model_map = ['Tanh', 30, 7,32, 32, 32]
        activation = model_map[0]
        # K = model_map[1]  # 分段数，可修改
        M = model_map[1]
        K = model_map[2]
        layer_dims = []

        input_size = K * (M + 1)  # 输入维度
        output_size = 1 * (M + 1)  # 输出维度
        layer_dims.append(input_size)
        for size in model_map[3:]:
            layer_dims.append(size)
        layer_dims.append(output_size)

        model_path = f"results/20250519/save/{Model}_M{M}_K{K}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        # trained_model = 'results/20250519/save/OB_DVR_NN_M15_K3_Tanh_202507201242.pt'# 好用
        # trained_model = 'results/20250519/save/OB_DVR_NN_M30_K1_Tanh_202507262149.pt' #很好
        trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        logger.info(f'------------------------{Model}_M{M}_K{K}_{activation}-------------------------------')
        # 初始化模型
        model = VDTDNN.VDTDNN(layer_dims, M=M, K=K, activation=activation).to(device)
        fun.model_structure(model, logger)

        if total_train == 1:
            # model.load_state_dict(torch.load(trained_model))
            model.model_train(x_train, y_train, model_path, logger, 1)
            model.load_state_dict(torch.load(model_path))
            logger.info(f"-------------------load model: {model_path}---------------------")
            start_time = time.time()  # 记录开始时间
            y_pred = model.apply_dpd(x)#, model.coef)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            logger.info(f"model prediction time: {elapsed_time:.6f} s")
        else:
            # N_train = min(50000, len(x_train))
            model.load_state_dict(torch.load(trained_model))
            logger.info(f"-------------------load model: {model_path}---------------------")
            start_time = time.time()  # 记录开始时间
            y_pred = model.apply_dpd(x)#, model.coef)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            logger.info(f"model prediction time: {elapsed_time:.6f} s")


    if Model == 'RVTDNN':
        # 参数设置
        model_map = ['Tanh', 20, 7, 24, 24]
        activation = model_map[0]
        # K = model_map[1]  # 分段数，可修改
        M = model_map[1]
        K = model_map[2]
        layer_dims = []

        input_size = 2 * K *(M + 1)  # 输入维度
        output_size = 2  # 输出维度
        layer_dims.append(input_size)
        for size in model_map[3:]:
            layer_dims.append(size)
        layer_dims.append(output_size)

        model_path = f"results/20250519/save/{Model}_M{M}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        # trained_model = 'results/20250519/save/OB_DVR_NN_M15_K3_Tanh_202507201242.pt'# 好用
        # trained_model = 'results/20250519/save/OB_DVR_NN_M30_K1_Tanh_202507262149.pt' #很好
        trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        logger.info(f'------------------------{Model}_M{M}_{activation}-------------------------------')
        # 初始化模型
        model = RVTDNN.RVTDNN(layer_dims, M=M, K=K,activation=activation).to(device)
        fun.model_structure(model, logger)

        if total_train == 1:
            # model.load_state_dict(torch.load(trained_model))
            model.model_train(x_train, y_train, model_path, logger, 1)
            model.load_state_dict(torch.load(model_path))
            logger.info(f"-------------------load model: {model_path}---------------------")
            start_time = time.time()  # 记录开始时间
            y_pred = model.apply_dpd(x)#, model.coef)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            logger.info(f"model prediction time: {elapsed_time:.6f} s")
        else:
            # N_train = min(50000, len(x_train))
            model.load_state_dict(torch.load(trained_model))
            logger.info(f"-------------------load model: {model_path}---------------------")
            start_time = time.time()  # 记录开始时间
            y_pred = model.apply_dpd(x)#, model.coef)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            logger.info(f"model prediction time: {elapsed_time:.6f} s")



    x_norm = x/max(abs(x))
    y_norm = y/max(abs(y))
    y_pred_norm = y_pred/max(abs(y_pred))

    # 评估结果（示例）
    # NMSE_pred = 10 * np.log10(sum(abs(y_pred_norm - y_norm)**2) / sum(abs(y_norm)**2))
    logger.info(f"signal {signal}")
    NMSE = cal.nmse(x,y,logger,1)
    ACLR = cal.acpr(y,fs,BW,BW,logger)



    logger.info(f"{Model} modeling:")
    NMSE_pred = cal.nmse(y_norm,y_pred_norm,logger,1)
    ACLR_pred = cal.acpr(y_pred,fs,BW,BW,logger)
    NMSE_list.append(NMSE_pred)
    NMSE_state_list.append(NMSE_pred)
    if plot_swich:
        plot.psd({"input": x,"pred_output":y_pred,"output":y},fs=fs,filename=f'figures/MCP_NN/{Model}_spec.png')
        plot.amam(x, {"out":y,"pred":y_pred}, norm=0, filename=f"figures/MCP_NN/{Model}_amam.png")
        plot.ampm(x, {"out": y, "pred": y_pred}, norm=0, filename=f"figures/MCP_NN/{Model}_ampm.png")
        plot.waveform({"y":y,"y_pred":y_pred},200,offset=200,norm=0,filename=f"figures/MCP_NN/{Model}_waveform.png")


# for para,nmse in zip(model[0],NMSE_list):
#     logger.info(f"{para[0]}_M{para[1]}_L{para[2]}_{para[3:]}_nmse{nmse}dB")
#
# best = min(NMSE_list)
# key = NMSE_list.index(best)
#
# logger.info(f"best NMSE {best} | best para{model[key]}")




# 计算X^H X
B = 1
# X = model.get_basis_amp(x_window)
# X = X.cpu().detach().numpy()
# lambda_reg = 0
# A = X.conj().T @ X + lambda_reg * np.eye(X.shape[1])
#
# # 计算2-范数条件数
# cond_number = np.linalg.cond(A)
#
# print(f'cond_number: {cond_number}')


# DPD_with_GMPNN = model.apply_dpd(xorg)
# print(max(abs(DPD_with_GMPNN)))
# # 保存数据到MAT文件
# file_name = 'data/GMPNN.mat'
# savemat(file_name, {'DPD': DPD_with_GMPNN.T, 'ILC': yorg, 'X': xorg})