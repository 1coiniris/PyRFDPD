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
import Model.DVR_NN as DVR_NN
import Model.Orth_NN as ORTH_NN
import Model.VDTDNN as VDTDNN
import Model.RVTDNN as RVTDNN
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
signal = '400M' #'LMBA200M'
# signal = 'LMBA200M'
# signal = '100M'
# signal = 'ILC_120M'
# signal = 'ILC'
# signal = 'YU'
count = 0
for state in range(1):
    x_train, y_train, x, y, fs, BW = fun.get_data(signal,state = state)
    figure_path = 'figures/MCP_NN'
    if plot_swich:
        fun.PA_figure(x,y,fs,figure_path)

    # 模型设置
    test_map = [
        'KFCNN_Tanh_7_30_30',
        'KFCNN_Tanh_7_20_30',
        'KFCNN_Tanh_7_20_20',
        'KFCNN_Tanh_7_15_20',
        'KFCNN_Tanh_7_15_15',
        'KFCNN_Tanh_7_10_20',
        'KFCNN_Tanh_7_10_15',
        'KFCNN_Tanh_7_10_10',
        'KFCNN_Tanh_7_5_20',
        'KFCNN_Tanh_7_5_15',
        'KFCNN_Tanh_7_5_10',
        'KFCNN_Tanh_7_5_5',
        'KFCNN_Tanh_5_20_20',
        'KFCNN_Tanh_5_15_20',
        'KFCNN_Tanh_5_15_15',
        'KFCNN_Tanh_5_10_20',
        'KFCNN_Tanh_5_10_15',
        'KFCNN_Tanh_5_10_10',
        'KFCNN_Tanh_5_5_20',
        'KFCNN_Tanh_5_5_15',
        'KFCNN_Tanh_5_5_10',
        'KFCNN_Tanh_5_5_5',
        'KFCNN_Tanh_3_20_20',
        'KFCNN_Tanh_3_15_20',
        'KFCNN_Tanh_3_15_15',
        'KFCNN_Tanh_3_10_20',
        'KFCNN_Tanh_3_10_15',
        'KFCNN_Tanh_3_10_10',
        'KFCNN_Tanh_3_5_20',
        'KFCNN_Tanh_3_5_15',
        'KFCNN_Tanh_3_5_10',
        'KFCNN_Tanh_3_5_5',
        'KFCNN_Tanh_2_20_20',
        'KFCNN_Tanh_2_15_20',
        'KFCNN_Tanh_2_15_15',
        'KFCNN_Tanh_2_10_20',
        'KFCNN_Tanh_2_10_15',
        'KFCNN_Tanh_2_10_10',
        'KFCNN_Tanh_2_5_20',
        'KFCNN_Tanh_2_5_15',
        'KFCNN_Tanh_2_5_10',
        'KFCNN_Tanh_2_5_5'
    ]
    # Model = 'DVC_NN'
    # Model = 'GMP_NN'
    # Model = 'RVTDNN'
    # Model = 'MCP_BASE_NN'
    # Model = 'PARA_NN'
    # Model = 'DVR'
    # Model = 'ADVR_NN'
    # Model = 'OB_DVR_NN'
    # Model = 'KFC_NN'
    # Model = 'VDTDNN'
    # Model = 'RVTDNN'
    # Model = 'DVR_NN'

    total_train = 1

    NMSE_list = []



    for test_state in test_map:
        count = count + 1
        Model = test_state.split('_')
        filename = f"{Model}_{time.strftime('%Y%m%d%H')}"
        if count ==1:
            parser = argparse.ArgumentParser(description='configTemplates')
            parser.add_argument('-log_path', default='./results/20250525/log/', type=str, help='log file path to save result')
            args = parser.parse_args()
            logger = fun.create_logger(args.log_path, filename)
    # model_path = 'results/save/MCP_NN_M7_[16, 24, 24, 16]_ReLU_202505091035.pt'
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
                model.model_train(x_train,y_train,model_path, logger,1)
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

        if Model[0] == 'OB_DVR_NN':
            # 参数设置
            model = ['Tanh',3,20,21]
            activation = model[0]
            K = model[1]  # 分段数，可修改
            M = model[2]

            model_path = f"results/20250519/save/{Model}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

            # trained_model = 'results/20250519/save/OB_DVR_NN_M15_K3_Tanh_202507201242.pt'# 好用
            # trained_model = 'results/20250519/save/OB_DVR_NN_M30_K1_Tanh_202507262149.pt' #很好
            trained_model = 'results/20250519/save/OB_DVR_NN_M20_K3_Tanh_202509011901.pt' #LMBA
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

        if Model[0] == 'ADVR_NN':
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
            model = DVR_NN.ADVR_NN(K=K, M=M, activation=activation).to(device)
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

        if Model[0] == 'DVR_NN':
            # epochs = 30
            # 参数设置
            model = ['Tanh', 3, 30]
            activation = model[0]
            K = model[1]  # 分段数，可修改
            M = model[2]
            # size = (M + 1 + Lb + Lc)

            model_path = f"results/20250519/save/{Model}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

            trained_model = 'results/20250519/save/DVR_NN_M30_K3_Tanh_202509012028.pt'
            # trained_model = 'results/20250519/save/DVR_NN_M15_K3_Tanh_202507181901.pt'
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
            logger.info(f'------------------------{Model}_M{M}_K{K}_{activation}-------------------------------')
            # 初始化模型
            model = DVR_NN.DVR_NN( K=K, M=M, activation=activation).to(device)


            if total_train == 1:
                # model.load_state_dict(torch.load(trained_model))
                # 训练前的模型参数
                model.model_train(x_train,y_train,model_path, logger,1)
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

        if Model[0] == 'VDTDNN':
            # 参数设置
            model_map = ['Tanh', 30, 1,32, 32, 32]
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


        if Model[0] == 'RVTDNN':
            # 参数设置
            model_map = ['ReLU', 30, 3, 32,32,32]
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
            model = RVTDNN.RVTDNN(layer_dims, M=M, K=K,activation=activation).double().to(device)
            fun.model_structure(model, logger)

            if total_train == 1:
                # model.load_state_dict(torch.load(trained_model))
                model.model_train(x_train, y_train, model_path, logger)
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


        count = count+1
        x_norm = x/max(abs(x))
        y_norm = y/max(abs(y))
        y_pred_norm = y_pred/max(abs(y_pred))

        # 评估结果（示例）
        # NMSE_pred = 10 * np.log10(sum(abs(y_pred_norm - y_norm)**2) / sum(abs(y_norm)**2))
        logger.info(f"signal {signal}")
        NMSE = cal.nmse(x,y,logger,1)
        ACLR = cal.acpr(y,fs,BW,BW,logger)



        logger.info(f"{Model[0]} modeling:")
        NMSE_pred = cal.nmse(y_norm,y_pred_norm,logger,1)
        ACLR_pred = cal.acpr(y_pred,fs,BW,BW,logger)
        NMSE_list.append(NMSE_pred)
        NMSE_state_list.append(NMSE_pred)
        if plot_swich:
            plot.psd({"input": x,"pred_output":y_pred,"output":y},fs=fs,filename=f'figures/MCP_NN/{Model[0]}_spec.png')
            plot.amam(x, {"out":y,"pred":y_pred}, norm=0, filename=f"figures/MCP_NN/{Model[0]}_amam.png")
            plot.ampm(x, {"out": y, "pred": y_pred}, norm=0, filename=f"figures/MCP_NN/{Model[0]}_ampm.png")
            plot.waveform({"y":y,"y_pred":y_pred},200,offset=200,norm=0,filename=f"figures/MCP_NN/{Model[0]}_waveform.png")


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