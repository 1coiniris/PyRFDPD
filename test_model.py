import matplotlib
matplotlib.use('Agg')
# from matplotlib import pyplot as plt
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
# import Model.volterra_nn as MCP_NN
# import Model.Mixed_NN as MIX_NN
import Model.DVR as DVR
import Model.gmp as gmp
import Model.DDR as DDR
import Model.DVR_NN as DVR_NN
import Model.Orth_NN as ORTH_NN
import Model.VDTDNN as VDTDNN
import Model.RVTDNN as RVTDNN
import Model.PNRVTDNN as PNRVTDNN
import Model.AKPTDNN as AKPTDNN
import Model.KFC_NN as KFCNN
import argparse
import time
from function import Function_Calculate as cal, Function_Lib as fun
# from function import FunctionAnalyzer as analyzer
# from tqdm import tqdm



# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

file_path = 'results/modeling/'

plot_swich = 1
NMSE_state_list = []
########################## 信号描述 ################################
# signal = '400M' #'LMBA200M'
# signal = 'LMBA200M'
# signal = '100M'
# signal = 'ILC_100M'
# signal = 'ILC_400M'
signal = 'NXP_100M'
# signal = 'YU'
count = 0
for state in range(3):
    x_train, y_train, x, y, fs, BW = fun.get_data(signal,state = state)
    figure_path = 'results/modeling/figure'
    if plot_swich:
        fun.PA_figure(x,y,fs,figure_path)

    # test_map = []
    # for prefix in ['DVRNN_ReLU_2_10_',]:#'DVRNN_Tanh_5_10','DVRNN_Tanh_7_10','DVRNN_Tanh_7_15']: #'PNRVTDNN_ReLU_10_3_','PNRVTDNN_ReLU_10_5_',
    #     NN_map = fun.generate_NN_map(prefix,2,[[8,8],[12,12],[10,12]],2)
    #     test_map.extend(NN_map)
    # 模型设置
    test_map = [
        # 'DVRNN_Sigmoid_1_10',
        # 'DVRNN_Sigmoid_1_10_12',
        # 'DVRNN_Sigmoid_2_10_8',
        # 'DVRNN_Sigmoid_2_10_8_8',
        # 'DVRNN_Sigmoid_2_10_12_8',
        # 'DVRNN_Sigmoid_3_10_8',
        # 'DVRNN_Sigmoid_3_10_12',
        # 'DVRNN_Sigmoid_3_10_8_8',
        # 'AKPTDNN_ReLU_10_6_4',
        # 'AKPTDNN_Tanh_10_6_4',

        'DVRNN_Tanh_1_10_6',
        'DVRNN_Tanh_1_10_6_12',
        'DVRNN_Tanh_2_10_6_8',
        'DVRNN_Tanh_2_10_6_8_8',
        'DVRNN_Tanh_3_10_6_8_8',
        'DVRNN_Tanh_3_10_6_12_8',

        'DVRNN_Tanh_1_10_7',
        'DVRNN_Tanh_1_10_7_12',
        'DVRNN_Tanh_2_10_7_8',
        'DVRNN_Tanh_2_10_7_8_8',
        'DVRNN_Tanh_3_10_7_8_8',
        'DVRNN_Tanh_3_10_7_12_8',

        'DVRNN_Tanh_1_10_5',
        'DVRNN_Tanh_1_10_5_12',
        'DVRNN_Tanh_2_10_5_8',
        'DVRNN_Tanh_2_10_5_8_8',
        'DVRNN_Tanh_3_10_5_8_8',
        'DVRNN_Tanh_3_10_5_12_8',

        'DVRNN_Tanh_1_10_1',
        'DVRNN_Tanh_1_10_1_12',
        'DVRNN_Tanh_2_10_1_8',
        'DVRNN_Tanh_2_10_1_8_8',
        'DVRNN_Tanh_3_10_1_8_8',
        'DVRNN_Tanh_3_10_1_12_8',

        'DVRNN_Tanh_1_10_0',
        'DVRNN_Tanh_1_10_0_12',
        'DVRNN_Tanh_2_10_0_8',
        'DVRNN_Tanh_2_10_0_8_8',
        'DVRNN_Tanh_3_10_0_8_8',
        'DVRNN_Tanh_3_10_0_12_8',
        # 'DVRNN_ReLU_1_10',
        # 'DVRNN_ReLU_1_10_12',
        # 'DVRNN_ReLU_2_10_8',
        # 'DVRNN_ReLU_2_10_8_8',
        # 'DVRNN_ReLU_2_10_12_8',
        # 'DVRNN_ReLU_3_10_8',
        # 'DVRNN_ReLU_3_10_12',
        # 'DVRNN_ReLU_3_10_8_8',



        # 'DVRNN_ReLU_1_10_8_8',
        # 'DVRNN_ReLU_1_10_12_8',

        # 'PNRVTDNN_Tanh_10_3_8',
        # 'PNRVTDNN_Tanh_10_3_8_8',
        # 'PNRVTDNN_Tanh_10_3_12_8',
        # # 'PNRVTDNN_Tanh_10_3_12_8_8',
        # 'PNRVTDNN_Tanh_10_3_12_12_8',
        # 'PNRVTDNN_Tanh_10_3_16_12',
        # 'PNRVTDNN_Tanh_10_3_16_12_8',
        # # 'PNRVTDNN_Tanh_10_3_16_12_12',
        # #
        # 'PNRVTDNN_Sigmoid_10_3_8',
        # 'PNRVTDNN_Sigmoid_10_3_8_8',
        # 'PNRVTDNN_Sigmoid_10_3_12_8',
        # # 'PNRVTDNN_Sigmoid_10_3_12_8_8',
        # 'PNRVTDNN_Sigmoid_10_3_12_12_8',
        # 'PNRVTDNN_Sigmoid_10_3_16_12',
        # 'PNRVTDNN_Sigmoid_10_3_16_12_8',
        # # 'PNRVTDNN_Sigmoid_10_3_16_12_12',
        # #
        # 'PNRVTDNN_ReLU_10_3_8',
        # 'PNRVTDNN_ReLU_10_3_8_8',
        # 'PNRVTDNN_ReLU_10_3_12_8',
        # # 'PNRVTDNN_ReLU_10_3_12_8_8',
        # 'PNRVTDNN_ReLU_10_3_12_12_8',
        # 'PNRVTDNN_ReLU_10_3_16_12',
        # 'PNRVTDNN_ReLU_10_3_16_12_8',
        # 'PNRVTDNN_ReLU_10_3_16_12_12',

        # 'VDTDNN_Tanh_10_3_8',
        # 'VDTDNN_Tanh_10_3_12',
        # 'VDTDNN_Tanh_10_3_12_12',
        # 'VDTDNN_Tanh_10_5_12_12',
        # 'VDTDNN_Tanh_10_5_12_12_12',
        # 'VDTDNN_Tanh_10_5_16_16_12',
        # 'VDTDNN_Tanh_10_5_16_12_12',
        #
        # 'VDTDNN_Sigmoid_10_3_8',
        # 'VDTDNN_Sigmoid_10_3_12',
        # 'VDTDNN_Sigmoid_10_3_12_12',
        # 'VDTDNN_Sigmoid_10_5_12_12',
        # 'VDTDNN_Sigmoid_10_5_12_12_12',
        # 'VDTDNN_Sigmoid_10_5_16_16_12',
        # 'VDTDNN_Sigmoid_10_5_16_12_12',
        #
        # 'VDTDNN_ReLU_10_3_8',
        # 'VDTDNN_ReLU_10_3_12',
        # 'VDTDNN_ReLU_10_3_12_12',
        # 'VDTDNN_ReLU_10_5_12_12',
        # 'VDTDNN_ReLU_10_5_12_12_12',
        # 'VDTDNN_ReLU_10_5_16_16_12',
        # 'VDTDNN_ReLU_10_5_16_12_12',

        # 'DDR_2_9_7',
        # 'DDR_7_7',
        # 'DDR_9_7',
        # 'GMP_[5, 5, 5]_[5, 5, 5]_[2, 2]',
        # 'GMP_[7, 5, 5]_[5, 5, 5]_[2, 2]',
        # 'GMP_[7, 3, 3]_[5, 3, 3]_[2, 2]',
        # 'GMP_[9, 3, 3]_[5, 3, 3]_[2, 2]',
        # 'GMP_[9, 5, 5]_[5, 5, 5]_[2, 2]',
        # 'GMP_[9, 7, 7]_[5, 5, 5]_[2, 2]',
        # 'GMP_[9, 7, 7]_[5, 5, 5]_[3, 3]',
        # 'GMP_[9, 9, 9]_[7, 5, 5]_[3, 3]',
        # 'GMP_[11, 3, 3]_[5, 3, 3]_[2, 2]',
        # 'GMP_[11, 5, 5]_[5, 3, 3]_[2, 2]',
        # 'GMP_[11, 7, 7]_[5, 3, 3]_[3, 3]',
        # 'GMP_[11, 9, 9]_[5, 5, 5]_[3, 3]',
        # 'GMP_[11, 9, 9]_[5, 5, 5]_[4, 4]',
        # 'GMP_[5, 5, 5]_[3, 3, 3]_[2, 2]',
        # 'GMP_[7, 5, 5]_[3, 3, 3]_[2, 2]',
        # 'GMP_[7, 3, 3]_[3, 3, 3]_[2, 2]',
        # 'GMP_[9, 3, 3]_[3, 3, 3]_[2, 2]',
        # 'GMP_[9, 5, 5]_[3, 3, 3]_[2, 2]',
        # 'GMP_[9, 7, 7]_[3, 3, 3]_[2, 2]',
        # 'GMP_[9, 7, 7]_[3, 3, 3]_[3, 3]',
        # 'GMP_[9, 9, 9]_[3, 3, 3]_[3, 3]',
        # 'GMP_[11, 3, 3]_[3, 3, 3]_[2, 2]',
        # 'GMP_[11, 5, 5]_[3, 3, 3]_[2, 2]',
        # 'GMP_[11, 7, 7]_[3, 3, 3]_[3, 3]',
        # 'GMP_[11, 9, 9]_[3, 3, 3]_[3, 3]',
        # 'GMP_[11, 9, 9]_[3, 3, 3]_[4, 4]',
        # 'DVR_1_10_[0.5]',
        # 'DVR_2_10_[0.3,0.7]',
        # 'DVR_3_10_[0.2,0.5,0.8]',
        # 'DVR_4_10_[0.2,0.4,0.6,0.8]',
        # 'DVR_5_10_[0.2,0.4,0.6,0.7,0.8]',
        # 'DVR_6_10_[0.2,0.4,0.6,0.7,0.8,0.9]',
        # 'DVR_7_10_[0.1,0.2,0.4,0.5,0.6,0.7,0.8]',

        # 'DVR_1_7_[0.5]',
        # 'DVR_2_7_[0.3,0.7]',
        # 'DVR_3_7_[0.2,0.5,0.8]',
        # 'DVR_4_7_[0.2,0.4,0.6,0.8]',
        # 'DVR_5_7_[0.2,0.4,0.6,0.7,0.8]',
        # 'DVR_6_7_[0.2,0.4,0.6,0.7,0.8,0.9]',
        # 'DVR_7_7_[0.1,0.2,0.4,0.5,0.6,0.7,0.8]',
        #
        # 'DVR_1_5_[0.5]',
        # 'DVR_2_5_[0.3,0.7]',
        # 'DVR_3_5_[0.2,0.5,0.8]',
        # 'DVR_4_5_[0.2,0.4,0.6,0.8]',
        # 'DVR_5_5_[0.2,0.4,0.6,0.7,0.8]',
        # 'DVR_6_5_[0.2,0.4,0.6,0.7,0.8,0.9]',
        # 'DVR_7_5_[0.1,0.2,0.4,0.5,0.6,0.7,0.8]',
        # 'DVR_3_5_[0.2,0.5,0.8]',
        # 'DVR_4_5_[0.2,0.4,0.6,0.8]',
        #
        # 'DVR_3_7_[0.2,0.5,0.8]',
        # 'DVR_4_7_[0.2,0.4,0.6,0.8]',
        ]
    #     # 'KFCNN_ReLU_1_7_7',
    #     # 'KFCNN_ReLU_1_7_7_6_10',
    #     # 'KFCNN_ReLU_2_7_7',
    #     # 'KFCNN_ReLU_1_10_10',
    #     # 'KFCNN_ReLU_3_7_7',
    #     # 'KFCNN_ReLU_4_7_7',
    #     # 'KFCNN_ReLU_2_10_10',
    #     # 'KFCNN_ReLU_3_10_10',
    #
        # 'DVRNN_ReLU_1_10_6',
        # 'DVRNN_ReLU_1_10_12',
        # 'DVRNN_ReLU_2_10_6',
        # 'DVRNN_ReLU_3_10_6',
        # 'DVRNN_ReLU_2_10_12',
        # 'DVRNN_ReLU_3_10_8_8',
        # 'DVRNN_ReLU_3_10_10_12',
        # 'DVRNN_ReLU_3_10_10_12',
        # 'DVRNN_ReLU_1_20_6',
        # 'DVRNN_ReLU_2_10_6',
        # 'DVRNN_ReLU_1_10_12',
        # 'DVRNN_ReLU_3_10_6',
        # 'DVRNN_ReLU_2_10_12',
        # 'DVRNN_ReLU_3_10_8_8',
        # 'DVRNN_ReLU_3_10_10_12',
        # ]
        # DVRNN_M10_K1_layer[11, 6, 11]_ReLU
        # DVRNN_M10_K1_layer[11, 12, 11]_ReLU
        # DVRNN_M10_K2_layer[11, 6, 22]_ReLU
        # DVRNN_M10_K2_layer[11, 12, 22]_ReLU
        # DVRNN_M10_K3_layer[11, 6, 33]_ReLU
        # DVRNN_M10_K3_layer[11, 8, 8, 33]_ReLU
        # DVRNN_M10_K3_layer[11, 10, 12, 33]_ReLU

        # 'DVRNN_ReLU_1_8',
        # 'DVRNN_ReLU_1_10',
        # 'DVRNN_ReLU_3_10_6',
        # 'DVR_7_9_[0.1, 0.2, 0.3,0.4,0.6,0.7, 0.8]',
        # 'DVR_7_10_[0.1, 0.2, 0.4,0.5,0.6,0.7, 0.8]',
        # 'VDTDNN_ReLU_5_2_8',
        # 'VDTDNN_ReLU_5_2_6_10',
        # 'VDTDNN_ReLU_5_3_8_10',
        # 'VDTDNN_ReLU_5_2_12_10',
        # 'VDTDNN_ReLU_15_1_8_10',
        # 'VDTDNN_ReLU_10_3_8_20',
        # 'VDTDNN_ReLU_10_3_16_16',
        # 'VDTDNN_Tanh_20_3_16',
        # 'VDTDNN_ReLU_20_3_16_16',
        # 'DVR_2_7_[0.2,0.5]',
        # 'DVR_4_5_[0.2,0.4,0.6,0.8]',
        # 'DVR_3_8_[0.2,0.5,0.8]',
        # 'DVR_3_10_[0.2,0.5,0.8]',
        # 'DVR_5_8_[0.2,0.4,0.6,0.7,0.8]',
        # 'DVR_5_10_[0.2,0.4,0.6,0.7,0.8]',
        # 'DVR_7_8_[0.1,0.2,0.3,0.4,0.5,0.6,0.8]',
    #
    #
    # ]

    total_train = 1

    NMSE_list = []



    for test_state in test_map:
        count = count + 1
        Model = test_state.split('_')
        filename = f"Modeling_{time.strftime('%Y%m%d%H')}"
        if count ==1:
            parser = argparse.ArgumentParser(description='configTemplates')
            parser.add_argument('-log_path', default=f'{file_path}log/', type=str, help='log file path to save result')
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
            coef = dvr.DVR_e(x_train, y_train, alpha=1e-9)
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
            coef = GMP.model_e(x_train, y_train,alpha=1e-8)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            # logger.info(f"model train time: {elapsed_time:.6f} s")
            y_pred = GMP.model_v(x_train, coef)
            logger.info(f'{Model[0]} train NMSE:')
            NMSE = cal.nmse(y_train[max(M)+max(L) + 11:], y_pred[max(M)+max(L)  + 11:], logger, 1)
            logger.info(f'{Model[0]} coef num {len(coef)}')
            y_pred = GMP.model_v(x, coef)

        if Model[0] == 'DDR':
            r = ast.literal_eval(Model[1])
            K = ast.literal_eval(Model[2])  # [int(num) for num in re.findall(r'\d+', Model[1])]
            M = ast.literal_eval(Model[3])  # [int(num) for num in re.findall(r'\d+', Model[3])]
            ddr = DDR.DDR(r, K, M)

            filename = f"DDR_R{r}_K{K}_M{M}_{time.strftime('%Y%m%d%H%M')}"
            logger.info(f'----------------{Model[0]}_R{r}_K{K}_M{M}--------------------')
            start_time = time.time()  # 记录开始时间
            coef = ddr.model_e(x_train, y_train,alpha=1e-9)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            # logger.info(f"model train time: {elapsed_time:.6f} s")
            y_pred = ddr.model_v(x_train)
            logger.info(f'{Model[0]} train NMSE:')
            NMSE = cal.nmse(y_train[M + 11:], y_pred[M + 11:], logger, 1)
            logger.info(f'{Model[0]} coef num {len(coef)}')
            y_pred = ddr.model_v(x)

        if Model[0] == 'AKPTDNN':
            # AKPTDNN: M_taps, L, P  (P * L = 24 for complexity control)
            # Config: 'AKPTDNN_10_24_1' -> M_taps=10, L=24, P=1
            #         'AKPTDNN_10_12_2' -> M_taps=10, L=12, P=2
            #         'AKPTDNN_10_8_3'  -> M_taps=10, L=8,  P=3
            #         'AKPTDNN_10_6_4'  -> M_taps=10, L=6,  P=4
            activation = Model[1]
            M_taps = ast.literal_eval(Model[2])
            L = ast.literal_eval(Model[3])
            P = ast.literal_eval(Model[4])

            model_path = f"{file_path}save/{Model[0]}_M{M_taps}_L{L}_P{P}_{time.strftime('%Y%m%d%H%M')}.pt"
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
            logger.info(f'------------------------{Model[0]}_M{M_taps}_L{L}_P{P}-------------------------------')

            # Initialize model
            model = AKPTDNN.AKPTDNN(M_taps=M_taps, L=L, P=P, activation=activation).to(device)
            fun.model_structure(model, logger)

            # Compute FLOPs and parameters
            flops, n_params = model.get_FLOPs_and_params()
            logger.info(f'AKPTDNN FLOPs: {flops}, Parameters: {n_params}')

            if total_train == 1:
                # Train from scratch (paper: lr=1e-4, epochs=2000, batch_size=2000)
                model.model_train(
                    x_train, y_train, model_path, logger, 1,
                    para=[0.001,1000,512]
                )
                model.load_state_dict(torch.load(model_path))
                logger.info(f"-------------------load model: {model_path}---------------------")

                start_time = time.time()
                y_pred = model.apply_dpd(x)
                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.info(f"model prediction time: {elapsed_time:.6f} s")
            else:
                # Load pretrained model
                model.load_state_dict(torch.load(model_path))
                logger.info(f"-------------------load model: {model_path}---------------------")
                start_time = time.time()
                y_pred = model.apply_dpd(x)
                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.info(f"model prediction time: {elapsed_time:.6f} s")


        if Model[0] == 'KFCNN':
            # 参数设置
            # model = ['Tanh', 5, 20, 20]
            activation = Model[1]
            K = ast.literal_eval(Model[2])  # 分段数，可修改
            M = ast.literal_eval(Model[3])
            M2 = ast.literal_eval(Model[4])

            layer_dims = []

            input_size = M2+1  # 输入维度
            output_size = (M+1)*K  # 输出维度
            layer_dims.append(input_size)
            for size_str in Model[5:]:
                size = ast.literal_eval(size_str)
                layer_dims.append(size)
            layer_dims.append(output_size)
            layer_dims_phase = [2 * (M2 + 1), M2 + 1, (M + 1) * K * 2]
            model_path = f"{file_path}save/{Model[0]}_M{M}_M2{M2}_K{K}_layer{layer_dims}_Phase{layer_dims_phase}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

            # trained_model = 'results/20250519/save/OB_DVR_NN_M15_K3_Tanh_202507201242.pt'# 好用
            trained_model = 'results/20250519/save/VD_DVR_NN_M30_M231_K3_Tanh_202508051529.pt' #很好
            # trained_model = 'results/20250519/save/VD_DVR_NN_M10_K1_Tanh_202508031410.pt' #LMBA
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
            logger.info(f'------------------------{Model[0]}_M{M}_M2{M2}_K{K}_layer{layer_dims}_Phase{layer_dims_phase}_{activation}-------------------------------')
            # 初始化模型
            model = KFCNN.KFC_NN(layer_dims=layer_dims,layer_dims_phase=layer_dims_phase, K=K, M=M,M2=M2, activation=activation).to(device)
            fun.model_structure(model, logger)


            if total_train == 1:
                # model.load_state_dict(torch.load(trained_model))
                model.model_train(x_train,y_train,model_path, logger,1,para=[0.001,300,512])
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

                y_pred = model.apply_dpd(x_train,model.coef)
                logger.info(f'{Model[0]} train NMSE:')
                NMSE = cal.nmse(y_train[M + 11:], y_pred[M + 11:], logger, 1)

                y_pred = model.apply_dpd(x,model.coef)
                end_time = time.time()  # 记录结束时间
                elapsed_time = end_time - start_time
                logger.info(f"model prediction time: {elapsed_time:.6f} s")
                logger.info(f'COEF number: {len(model.coef)} ')
                # NMSE, ACLR, NMSE_pred, ACLR_pred = fun.calculate_CRZ(x, y, y_pred, fs, BW, figure_path, Model[0], 1,logger, type='model')
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


        if Model[0] == 'DVRNN':
            # 参数设置
            activation = Model[1]
            K = ast.literal_eval(Model[2])
            M = ast.literal_eval(Model[3])
            term = ast.literal_eval(Model[4])
            layer_dims = []

            P = 1
            input_size = (M+1)   # 输入维度
            output_size = (M+1)*K*P  # 输出维度
            layer_dims.append(input_size)
            for size_str in Model[5:]:
                size = ast.literal_eval(size_str)
                layer_dims.append(size)
            layer_dims.append(output_size)

            model_path = f"{file_path}save/{Model[0]}_M{M}_K{K}_layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
            trained_model = 'tests/20250826/model/OB_DVR_NN_M30_K3_Tanh_202508261803.pt'  # LMBA
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
            logger.info(f'------------------------{Model[0]}_M{M}_K{K}_term{term}_layer{layer_dims}_{activation}-------------------------------')

            # 初始化模型
            model = DVR_NN.DVR_NN(layer_dims, K=K, M=M, activation=activation,term=term).to(device)
            fun.model_structure(model, logger)

            if total_train == 1:
                # model.load_state_dict(torch.load(trained_model))
                # 训练前的模型参数
                model.model_train(x_train,y_train,model_path, logger,1,para = [0.001,300,512,1e-5])
                # 训练后的模型参数
                model.load_state_dict(torch.load(model_path))
                logger.info(f"-------------------load model: {model_path}--------------------  -")
                # 提取参数
                x_coef = x_train[:]
                y_coef = y_train[:]
                x_coef_tensor = torch.from_numpy(x_coef).to(device)
                y_coef_tensor = torch.from_numpy(y_coef).to(device)
                sequences = fun.create_memory_seq(x_coef, M)  # [N, M+1]
                x_coef_window = torch.from_numpy(sequences).to(device)
                y_sequences = fun.create_memory_seq(y_coef, M)  # [N, M+1]
                y_coef_window = torch.from_numpy(y_sequences).to(device)
                model.coef = model.DVR_NN_e(x_coef_window,y_coef_tensor,alpha=1e-5,pri=1)
                # coef = model.DVR_NN_e(x_train, y_train,alpha=5e-2)
                # y_pred = model.DVR_NN_v(x_train,coef)
                y_pred = model.DVR_NN_v(x_coef_window,model.coef).cpu().detach().numpy()
                # y_pred = model.apply_dpd(x_train,model.coef)
                logger.info(f'{Model[0]} train NMSE:')
                NMSE = cal.nmse(y_train[M + 11:], y_pred[M + 11:], logger, 1)

                start_time = time.time()  # 记录开始时间
                # y_pred = model.apply_dpd(x)
                # y_pred = model.DVR_NN_v(x,coef)
                x_val = x[:]
                sequences = fun.create_memory_seq(x_val, M)  # [N, M+1]
                x_val_window = torch.from_numpy(sequences).to(device)

                y_pred = model.DVR_NN_v(x_val_window,model.coef).cpu().detach().numpy()
                end_time = time.time()  # 记录结束时间
                elapsed_time = end_time - start_time
                logger.info(f"model prediction time: {elapsed_time:.6f} s")
                logger.info(f'COEF number: {len(model.coef)} ')
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
            model_path = f"{file_path}save/{Model[0]}_M{M}_K{K}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

            trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
            logger.info(f'------------------------{Model[0]}_M{M}_K{K}_layer{layer_dims}_{activation}-------------------------------')
            # 初始化模型
            model = PNRVTDNN.PNRVTDNN(layer_dims, M=M, K=K, activation=activation).double().to(device)
            fun.model_structure(model, logger)

            if total_train == 1:
                # model.load_state_dict(torch.load(trained_model))
                model.model_train(x_train, y_train, model_path, logger, [0.001,300,512])
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

            model_path = f"{file_path}save/{Model[0]}_M{M}_K{K}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

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
                model.model_train(x_train, y_train, model_path, logger, 1,para = [0.001,500,512])
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

            model_path = f"{file_path}save/{Model[0]}_M{M}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

            trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
            logger.info(f'------------------------{Model[0]}_M{M}_K{K}_layer{layer_dims}_{activation}-------------------------------')
            # 初始化模型
            model = RVTDNN.RVTDNN(layer_dims, M=M, K=K,activation=activation).double().to(device)
            fun.model_structure(model, logger)

            if total_train == 1:
                # model.load_state_dict(torch.load(trained_model))
                model.model_train(x_train, y_train, model_path, logger,para = [0.003,300,512])
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



# 假设你已经训练好模型
# model = ADVR_NN(K=7, M=5, activation="ReLU")
# model.load_state_dict(torch.load('trained_model.pth'))
# model.eval()
#
# # 创建分析器
# analyzer = analyzer.BasisFunctionAnalyzer(model, M=5, K=7)
#
# # 生成测试信号
# test_signals = analyzer.generate_test_signals(num_samples=5000)
#
# # 分析基函数
# analysis_results = analyzer.analyze_basis_functions(test_signals)
#
# # 可视化结果
# analyzer.visualize_basis_functions(analysis_results, top_n=12)
#
# # 分析相位恢复项
# analyzer.analyze_phase_recovery_terms()



# if __name__ == "__main__":
#     analysis_results = main()
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