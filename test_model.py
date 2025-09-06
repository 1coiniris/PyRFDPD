import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
# from anyio import sleep
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat
from torch.utils.data import DataLoader, TensorDataset
from pyrfdpd.utils import plot
import Model.volterra_nn as MCP_NN
import Model.Mixed_NN as MIX_NN
import Model.DVR as DVR
import Model.gmp as gmp
import Model.DVR_NN as DVR_NN
import Model.Orth_NN as ORTH_NN
import Model.VDTDNN as VDTDNN
import Model.RVTDNN as RVTDNN
import Model.PNRVTDNN as PNRVTDNN
import Model.KFC_NN as KFCNN
import argparse
import time
from function import Function_Calculate as cal, Function_Lib as fun
from tqdm import tqdm



# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


plot_swich = 1
NMSE_state_list = []
########################## 信号描述 ################################
# signal = '400M' #'LMBA200M'
# signal = 'LMBA200M'
# signal = '100M'
# signal = 'ILC_120M'
signal = 'ILC_100M'
# signal = 'YU'
count = 0
for state in range(1):
    x_train, y_train, x, y, fs, BW = fun.get_data(signal,state = state)
    figure_path = 'figures/Modeling'
    if plot_swich:
        fun.PA_figure(x,y,fs,figure_path)

    # 模型设置
    test_map = [
        # 'PNRVTDNN_Tanh_30_5_32_32',
        # 'RVTDNN_Tanh_30_5_32_32',
        # 'VDTDNN_Tanh_30_5_32_32',
        # 'RVTDNN_Tanh_30_5_32_32',
        # 'RVTDNN_Tanh_30_5_16_16',
        # 'RVTDNN_Tanh_20_5_32_32',
        # 'RVTDNN_Tanh_20_5_16_16',
        #
        # 'RVTDNN_Tanh_30_5_32_32_32',
        # 'RVTDNN_Tanh_30_5_16_16_16',
        # 'RVTDNN_Tanh_20_5_32_32_32',
        # 'RVTDNN_Tanh_20_5_16_16_16',
        #
        # 'VDTDNN_Tanh_30_5_32_32_32',
        # 'VDTDNN_Tanh_30_5_16_16_16',
        # 'VDTDNN_Tanh_20_5_32_32_32',
        # 'VDTDNN_Tanh_20_5_16_16_16',
        #
        # 'VDTDNN_Tanh_30_5_32_32',
        # 'VDTDNN_Tanh_30_5_16_16',
        # 'VDTDNN_Tanh_20_5_32_32',
        # 'VDTDNN_Tanh_20_5_16_16',

        # 'DVRNN_ReLU_9_30_20_20',
        # 'DVRNN_ReLU_9_30_10_20',
        # 'DVRNN_ReLU_9_30_10_10',
        # 'DVRNN_ReLU_9_30_10',
        # 'DVRNN_ReLU_7_30_10',
        # 'DVRNN_ReLU_5_30_10',
        #
        # 'DVRNN_Tanh_9_30_10',
        # 'DVRNN_Tanh_7_30_10',
        # 'DVRNN_Tanh_5_30_10',
        # 'KFCNN_Tanh_9_40_30',
        # 'KFCNN_Tanh_9_30_30',
        # 'KFCNN_Tanh_9_20_30',
        # 'KFCNN_Tanh_9_10_30',
        # 'KFCNN_Tanh_9_5_30',
        # 'KFCNN_Tanh_7_40_30',
        # 'KFCNN_Tanh_7_30_30',
        # 'KFCNN_Tanh_7_20_30',
        # 'KFCNN_Tanh_7_10_30',
        # 'KFCNN_Tanh_7_5_30',
        # 'KFCNN_Tanh_3_10_30',
        # 'KFCNN_Tanh_3_5_30',
        # 'KFCNN_Tanh_5_20_20',
        # 'KFCNN_Tanh_5_10_30',
        # 'KFCNN_Tanh_5_10_20',
        # 'KFCNN_Tanh_5_10_10',
        # 'KFCNN_Tanh_5_5_30',
        # 'KFCNN_Tanh_5_5_20',
        # 'KFCNN_Tanh_5_5_10',
        # 'KFCNN_Tanh_3_10_30',
        # 'KFCNN_Tanh_3_10_20',
        # 'KFCNN_Tanh_3_10_10',
        # 'KFCNN_Tanh_3_5_30',
        # 'KFCNN_Tanh_3_5_20',
        # 'KFCNN_Tanh_3_5_10',
        # 'KFCNN_Tanh_1_5_30',
        # 'KFCNN_Tanh_1_5_20',
        # 'KFCNN_Tanh_1_5_10',
        # 'OBDVRNN_ReLU_4_10_20_20',
        # 'OBDVRNN_Tanh_4_10_20_20',
        # 'OBDVRNN_ReLU_4_20_20_20',
        # 'OBDVRNN_Tanh_4_20_20_20',
        # 'OBDVRNN_ReLU_4_20_20',
        # 'OBDVRNN_Tanh_4_20_20',

        'DVRNN_ReLU_4_20_20',
        'DVRNN_Tanh_4_20_20',
        'OBDVRNN_ReLU_4_20',
        'OBDVRNN_Tanh_4_20',
        # 'OBDVRNN_ReLU_4_30_20_20',



        # 'OBDVRNN_ReLU_9_30_20_20',
        # 'OBDVRNN_ReLU_9_20_20_20',
        # 'OBDVRNN_ReLU_9_10_20_20',
        # 'OBDVRNN_ReLU_9_5_20_20',
        # 'OBDVRNN_ReLU_7_30_20_20',
        # 'OBDVRNN_ReLU_7_20_20_20',
        # 'OBDVRNN_ReLU_7_10_20_20',
        # 'OBDVRNN_ReLU_7_5_20_20',
        #
        # # 'OBDVRNN_Tanh_9_30_20_20',


        # # 'OBDVRNN_Tanh_9_5_20_20',
        # # 'OBDVRNN_Tanh_7_30_20_20',
        # 'OBDVRNN_Tanh_7_20_20_20',
        # 'OBDVRNN_Tanh_7_10_20_20',
        # 'OBDVRNN_Tanh_7_5_20_20',

        # 'DVRNN_Tanh_5_30_10',
        # 'DVRNN_Tanh_7_30_10',
        # 'DVRNN_Tanh_9_30_10'

    ]

    total_train = 1

    NMSE_list = []



    for test_state in test_map:
        count = count + 1
        Model = test_state.split('_')
        filename = f"Modeling_{time.strftime('%Y%m%d%H')}"
        if count ==1:
            parser = argparse.ArgumentParser(description='configTemplates')
            parser.add_argument('-log_path', default='./results/20250525/log/', type=str, help='log file path to save result')
            args = parser.parse_args()
            logger = fun.create_logger(args.log_path, filename)

        if Model[0] == 'DVR':  # DVR [K,M]
            # 参数设置
            M = ast.literal_eval(Model[2])
            threshold = ast.literal_eval(Model[3])
            K = len(threshold)

            filename = f"DVR_K{K}_M{M}_threshold{threshold}_{time.strftime('%Y%m%d%H%M')}"
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')

            # print(f'------------------------{Model}_M{M}_{layer_dims}_{activation}-------------------------------')
            logger.info(f'------------------------{Model[0]}_M{M}_K{K}-------------------------------')
            # 初始化模型
            dvr = DVR.DVR(M=M, threshold=threshold)
            coef = dvr.DVR_e(x_train, y_train, alpha=1e-3)
            y_pred = dvr.DVR_v(x_train, coef)
            logger.info(f'{Model[0]} train NMSE:')
            NMSE = cal.nmse(y_train[M + 11:], y_pred[M + 11:], logger, 1)
            y_pred = dvr.DVR_v(x, coef)
            logger.info(f'{Model[0]} coef num {len(coef)}')

        if Model[0] == 'GMP':
            K = ast.literal_eval(Model[1])  # [int(num) for num in re.findall(r'\d+', Model[1])]
            L = ast.literal_eval(Model[2])  # [int(num) for num in re.findall(r'\d+', Model[2])]
            M = ast.literal_eval(Model[3])  # [int(num) for num in re.findall(r'\d+', Model[3])]
            GMP = gmp.GMP(K, L, M)
            filename = f"GMP_K{K}_L{L}_M{M}_{time.strftime('%Y%m%d%H%M')}"
            logger.info(f'----------------{Model[0]}_K{K}_L{L}_M{M}--------------------')
            start_time = time.time()  # 记录开始时间
            coef = GMP.model_e(x_train, y_train)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            # logger.info(f"model train time: {elapsed_time:.6f} s")
            y_pred = GMP.model_v(x_train, coef)
            logger.info(f'{Model[0]} train NMSE:')
            NMSE = cal.nmse(y_train[M + 11:], y_pred[M + 11:], logger, 1)
            logger.info(f'{Model[0]} coef num {len(coef)}')
            y_pred = GMP.model_v(x, coef)


        if Model[0] == 'KFCNN':
            # 参数设置
            # model = ['Tanh', 5, 20, 20]
            activation = Model[1]
            K = ast.literal_eval(Model[2])  # 分段数，可修改
            M = ast.literal_eval(Model[3])
            M2 = ast.literal_eval(Model[4])

            model_path = f"results/20250519/save/{Model[0]}_M{M}_M2{M2}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

            # trained_model = 'results/20250519/save/OB_DVR_NN_M15_K3_Tanh_202507201242.pt'# 好用
            trained_model = 'results/20250519/save/VD_DVR_NN_M30_M231_K3_Tanh_202508051529.pt' #很好
            # trained_model = 'results/20250519/save/VD_DVR_NN_M10_K1_Tanh_202508031410.pt' #LMBA
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
            logger.info(f'------------------------{Model[0]}_M{M}_M2{M2}_K{K}_{activation}-------------------------------')
            # 初始化模型
            model = KFCNN.KFC_NN( K=K, M=M,M2=M2, activation=activation).to(device)
            fun.model_structure(model, logger)


            if total_train == 1:
                # model.load_state_dict(torch.load(trained_model))
                model.model_train(x_train,y_train,model_path, logger,1,para=[0.001,150,512])
                model.load_state_dict(torch.load(model_path))
                logger.info(f"-------------------load model: {model_path}---------------------")
                start_time = time.time()  # 记录开始时间
                # 提取参数
                x_coef = x_train[:]
                y_coef = y_train[:]
                x_coef_tensor = torch.from_numpy(x_coef).to(device)
                y_coef_tensor = torch.from_numpy(y_coef).to(device)
                sequences = fun.create_memory_seq(x_coef, M2)  # [N, M+1]
                x_coef_window = torch.from_numpy(sequences).to(device)
                y_sequences = fun.create_memory_seq(y_coef, M2)  # [N, M+1]
                y_coef_window = torch.from_numpy(y_sequences).to(device)
                model.coef = model.DVR_NN_e(x_coef_window,y_coef_tensor,alpha=1e-2,pri=1)
                y_pred = model.apply_dpd(x,model.coef)
                end_time = time.time()  # 记录结束时间
                elapsed_time = end_time - start_time
                logger.info(f"model prediction time: {elapsed_time:.6f} s")
                logger.info(f'COEF number: {len(model.coef)} ')
            else:
                N_train = min(50000,len(x_train))
                model.load_state_dict(torch.load(trained_model))
                # 提取参数
                x_coef = x_train[0:N_train - 1]
                y_coef = y_train[0:N_train - 1]
                x_coef_tensor = torch.from_numpy(x_coef).to(device)
                y_coef_tensor = torch.from_numpy(y_coef).to(device)
                sequences = fun.create_memory_seq(x_coef, M2)  # [N, M+1]
                x_coef_window = torch.from_numpy(sequences).to(device)
                y_sequences = fun.create_memory_seq(y_coef, M)  # [N, M+1]
                y_coef_window = torch.from_numpy(y_sequences).to(device)
                coef = model.DVR_NN_e(x_coef_window,y_coef_tensor,alpha=5e-3,pri=1)
                logger.info(f"model coef number: {len(coef)} ")
                # 建模
                sequences = fun.create_memory_seq(x, M2)  # [N, M+1]
                x_window = torch.from_numpy(sequences).to(device)
                y_pred = model.DVR_NN_v(x_window,coef).cpu().detach().numpy()
                X = model.get_basis(x_window)

        if Model[0] == 'OBDVRNN':
            # 参数设置
            activation = Model[1]
            K = ast.literal_eval(Model[2])
            M = ast.literal_eval(Model[3])

            layer_dims = []

            input_size = M+1  # 输入维度
            output_size = (M+1)*K  # 输出维度
            layer_dims.append(input_size)
            for size_str in Model[4:]:
                size = ast.literal_eval(size_str)
                layer_dims.append(size)
            layer_dims.append(output_size)

            model_path = f"results/20250519/save/{Model}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

            # trained_model = 'results/20250519/save/OB_DVR_NN_M15_K3_Tanh_202507201242.pt'# 好用
            # trained_model = 'results/20250519/save/OB_DVR_NN_M30_K1_Tanh_202507262149.pt' #很好
            trained_model = 'results/20250519/save/OB_DVR_NN_M20_K3_Tanh_202509011901.pt' #LMBA
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
            logger.info(f'------------------------{Model}_M{M}_K{K}_{activation}-------------------------------')
            # 初始化模型
            model = ORTH_NN.Orth_Basis_DVR_NN(layer_dims, K=K, M=M, activation=activation).to(device)
            fun.model_structure(model, logger)


            if total_train == 1:
                # model.load_state_dict(torch.load(trained_model))
                model.model_train(x_train,y_train,model_path, logger,1,para = [0.001,250,4096])
                model.load_state_dict(torch.load(model_path))
                logger.info(f"-------------------load model: {model_path}---------------------")
                start_time = time.time()  # 记录开始时间
                # 提取参数
                x_coef = x_train[:]
                y_coef = y_train[:]
                x_coef_tensor = torch.from_numpy(x_coef).to(device)
                y_coef_tensor = torch.from_numpy(y_coef).to(device)
                sequences = fun.create_memory_seq(x_coef, M)  # [N, M+1]
                x_coef_window = torch.from_numpy(sequences).to(device)
                y_sequences = fun.create_memory_seq(y_coef, M)  # [N, M+1]
                y_coef_window = torch.from_numpy(y_sequences).to(device)
                model.coef = model.model_e(x_coef_window,y_coef_tensor)
                y_pred = model.apply_dpd(x,model.coef)
                end_time = time.time()  # 记录结束时间
                elapsed_time = end_time - start_time
                logger.info(f"model prediction time: {elapsed_time:.6f} s")
                logger.info(f'COEF number: {len(model.coef)} ')
            else:
                N_train = min(50000,len(x_train))
                model.load_state_dict(torch.load(trained_model))
                # 提取参数
                x_coef = x_train[0:N_train - 1]
                y_coef = y_train[0:N_train - 1]
                x_coef_tensor = torch.from_numpy(x_coef).to(device)
                y_coef_tensor = torch.from_numpy(y_coef).to(device)
                sequences = fun.create_memory_seq(x_coef, M)  # [N, M+1]
                x_coef_window = torch.from_numpy(sequences).to(device)
                y_sequences = fun.create_memory_seq(y_coef, M)  # [N, M+1]
                y_coef_window = torch.from_numpy(y_sequences).to(device)
                coef = model.model_e(x_coef_window,y_coef_tensor,alpha=1e-2,pri=1)
                logger.info(f'COEF number: {len(model.coef)} ')
                # 建模
                sequences = fun.create_memory_seq(x, M)  # [N, M+1]
                x_window = torch.from_numpy(sequences).to(device)
                y_pred = model.model_v(x_window,coef).cpu().detach().numpy()
                X = model.get_basis(x_window)


        if Model[0] == 'DVRNN':
            # 参数设置
            activation = Model[1]
            K = ast.literal_eval(Model[2])
            M = ast.literal_eval(Model[3])

            layer_dims = []

            input_size = M+1  # 输入维度
            output_size = (M+1)*K  # 输出维度
            layer_dims.append(input_size)
            for size_str in Model[4:]:
                size = ast.literal_eval(size_str)
                layer_dims.append(size)
            layer_dims.append(output_size)

            model_path = f"results/20250519/save/{Model[0]}_M{M}_K{K}_layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
            trained_model = 'tests/20250826/model/OB_DVR_NN_M30_K3_Tanh_202508261803.pt'  # LMBA
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')

            # 初始化模型
            model = DVR_NN.DVR_NN(layer_dims, K=K, M=M, activation=activation).to(device)

            if total_train == 1:
                # model.load_state_dict(torch.load(trained_model))
                # 训练前的模型参数
                model.model_train(x_train,y_train,model_path, logger,1,para = [0.001,250,512])
                # 训练后的模型参数
                model.load_state_dict(torch.load(model_path))
                logger.info(f"-------------------load model: {model_path}---------------------")
                coef = model.DVR_NN_e(x_train, y_train)
                start_time = time.time()  # 记录开始时间
                # y_pred = model.apply_dpd(x)
                y_pred = model.DVR_NN_v(x,coef)
                end_time = time.time()  # 记录结束时间
                elapsed_time = end_time - start_time
                logger.info(f"model prediction time: {elapsed_time:.6f} s")
                logger.info(f'COEF number: {len(coef)} ')
            else:
                N_train = min(5000,len(x_train))
                model.load_state_dict(torch.load(trained_model))
                coef = model.DVR_NN_e(x_train[0:N_train-1],y_train[0:N_train-1])
                y_pred = model.DVR_NN_v(x,coef)
                sequences = fun.create_memory_seq(x, M)  # [N, M+1]
                x_window = torch.from_numpy(sequences).to(device)
                X = model.get_basis(x_window)

        if Model[0] == 'PNRVTDNN':
            activation = Model[1]
            M = ast.literal_eval(Model[2])
            K = ast.literal_eval(Model[3])
            layer_dims = []

            input_size = 2 * (M + 1) + K * (M + 1)-1  # 输入维度
            output_size = 2  # 输出维度
            layer_dims.append(input_size)
            for size_str in Model[4:]:
                size = ast.literal_eval(size_str)
                layer_dims.append(size)
            layer_dims.append(output_size)
            model_path = f"results/20250519/save/{Model[0]}_M{M}_K{K}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

            trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
            logger.info(f'------------------------{Model[0]}_M{M}_K{K}_layer{layer_dims}_{activation}-------------------------------')
            # 初始化模型
            model = PNRVTDNN.PNRVTDNN(layer_dims, M=M, K=K, activation=activation).double().to(device)
            fun.model_structure(model, logger)

            if total_train == 1:
                # model.load_state_dict(torch.load(trained_model))
                model.model_train(x_train, y_train, model_path, logger, [0.001,350,512])
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

        if Model[0] == 'VDTDNN':
            activation = Model[1]
            M = ast.literal_eval(Model[2])
            K = ast.literal_eval(Model[3])
            layer_dims = []

            input_size = K * (M + 1)  # 输入维度
            output_size = 1 * (M + 1)  # 输出维度
            layer_dims.append(input_size)
            for size_str in Model[4:]:
                size = ast.literal_eval(size_str)
                layer_dims.append(size)
            layer_dims.append(output_size)

            model_path = f"results/20250519/save/{Model[0]}_M{M}_K{K}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

            # trained_model = 'results/20250519/save/OB_DVR_NN_M15_K3_Tanh_202507201242.pt'# 好用
            # trained_model = 'results/20250519/save/OB_DVR_NN_M30_K1_Tanh_202507262149.pt' #很好
            trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
            logger.info(f'------------------------{Model[0]}_M{M}_K{K}_layer{layer_dims}_{activation}-------------------------------')
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


        if Model[0] == 'RVTDNN':
            activation = Model[1]
            M = ast.literal_eval(Model[2])
            K = ast.literal_eval(Model[3])
            layer_dims = []

            input_size = 2 * (M + 1) + K * (M + 1)  # 输入维度
            output_size = 2  # 输出维度
            layer_dims.append(input_size)
            for size_str in Model[4:]:
                size = ast.literal_eval(size_str)
                layer_dims.append(size)
            layer_dims.append(output_size)

            model_path = f"results/20250519/save/{Model[0]}_M{M}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

            trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
            logger.info(f'------------------------{Model[0]}_M{M}_K{K}_layer{layer_dims}_{activation}-------------------------------')
            # 初始化模型
            model = RVTDNN.RVTDNN(layer_dims, M=M, K=K,activation=activation).double().to(device)
            fun.model_structure(model, logger)

            if total_train == 1:
                # model.load_state_dict(torch.load(trained_model))
                model.model_train(x_train, y_train, model_path, logger,para = [0.001,450,512])
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


        NMSE, ACLR, NMSE_pred, ACLR_pred = fun.calculate_CRZ(x, y, y_pred, fs, BW, figure_path, Model[0], 1, logger,type='model')
        # x_norm = x/max(abs(x))
        # y_norm = y/max(abs(y))
        # y_pred_norm = y_pred/max(abs(y_pred))
        #
        # # 评估结果（示例）
        # # NMSE_pred = 10 * np.log10(sum(abs(y_pred_norm - y_norm)**2) / sum(abs(y_norm)**2))
        # logger.info(f"signal {signal}")
        # NMSE = cal.nmse(x,y,logger,1)
        # ACLR = cal.acpr(y,fs,BW,BW,logger)
        #
        #
        #
        # logger.info(f"{Model[0]} modeling:")
        # NMSE_pred = cal.nmse(y_norm,y_pred_norm,logger,1)
        # ACLR_pred = cal.acpr(y_pred,fs,BW,BW,logger)
        # NMSE_list.append(NMSE_pred)
        # NMSE_state_list.append(NMSE_pred)
        # if plot_swich:
        #     plot.psd({"input": x,"pred_output":y_pred,"output":y},fs=fs,filename=f'figures/MCP_NN/{Model[0]}_spec.png')
        #     plot.amam(x, {"out":y,"pred":y_pred}, norm=0, filename=f"figures/MCP_NN/{Model[0]}_amam.png")
        #     plot.ampm(x, {"out": y, "pred": y_pred}, norm=0, filename=f"figures/MCP_NN/{Model[0]}_ampm.png")
        #     plot.waveform({"y":y,"y_pred":y_pred},200,offset=200,norm=0,filename=f"figures/MCP_NN/{Model[0]}_waveform.png")


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