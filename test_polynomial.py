import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from anyio import sleep
import numpy as np
import re
import ast
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# import pyrfdpd.nn as dpdnn
# import pyrfdpd.visa as visa
import Model.gmp as gmp
import Model.mp as mp
import Model.DVR as DVR
import argparse
import time
from function import Function_Calculate as cal, Function_Lib as fun

########################## 信号描述 ################################
# signal = '100M' #'LMBA200M'
# signal = 'LMBA200M'
# signal = '400M'
# signal = 'ILC_120M'
signal = 'ILC_100M'
# signal = 'YU'

plot_swich = 1
NMSE_list = []
count = 0
for state in range(1):
    x_train, y_train, x, y, fs, BW = fun.get_data(signal,state = state)
    # N = len(xorg)
    figure_path = 'figures/polynomial'
    if plot_swich:
        fun.PA_figure(x,y,fs,figure_path)



    # Model_map = ['GMP','MP']
    test_map = [
        # 'GMP_[17, 17, 17]_[17, 17, 17]_[10, 10]',
        # 'GMP_[11, 11, 11]_[11, 7, 7]_[5, 5]',
        # 'GMP_[11, 11, 11]_[11, 7, 7]_[5, 5]',
        # 'GMP_[11, 7, 7]_[7, 7, 7]_[5, 5]',
        # 'GMP_[7, 7, 7]_[7, 7, 7]_[5, 5]',
        # 'GMP_[7, 7, 7]_[6, 6, 5]_[5, 5]',
        # 'GMP_[7, 7, 7]_[6, 5, 5]_[5, 3]',
        # 'GMP_[7, 7, 7]_[5, 5, 5]_[5, 5]',
        # 'GMP_[7, 7, 7]_[5, 3, 3]_[5, 5]',
        # 'GMP_[7, 7, 7]_[5, 5, 5]_[2, 2]',
        # 'GMP_[7, 5, 5]_[5, 5, 5]_[3, 3]',
        # 'GMP_[5, 5, 5]_[5, 5, 5]_[3, 3]',
        # 'GMP_[5, 3, 3]_[5, 5, 5]_[3, 3]',
        # 'GMP_[5, 3, 3]_[5, 3, 3]_[2, 2]',
        # 'GMP_[5, 3, 3]_[3, 3, 3]_[2, 2]',
        # 'GMP_[3, 1, 1]_[3, 3, 3]_[2, 2]',
        # 'GMP_[3, 1, 1]_[3, 1, 1]_[2, 2]'
        # 'DVR_9_30_[0.1,0.2, 0.3,0.4,0.5,0.6, 0.7, 0.8,0.9]',
        # 'DVR_9_20_[0.1,0.2, 0.3,0.4,0.5,0.6, 0.7, 0.8,0.9]',
        # 'DVR_9_10_[0.1,0.2, 0.3,0.4,0.5,0.6, 0.7, 0.8,0.9]',
        # 'DVR_9_5_[0.1,0.2, 0.3,0.4,0.5,0.6, 0.7, 0.8,0.9]',
        # 'DVR_7_30_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
        # 'DVR_7_20_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
        # 'DVR_7_10_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
        # 'DVR_7_5_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
        # 'DVR_7_30_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
        'DVR_4_40_[0.2,0.3,0.4, 0.5,0.55,0.6,0.65,0.7,0.75, 0.8,0.81,0.83,0.84,0.85,0.87,0.88,0.9, 0.92,0.94,0.95,0.97,1]',
        'DVR_4_30_[0.2,0.3,0.4, 0.5,0.55,0.6,0.65,0.7,0.75, 0.8,0.81,0.83,0.84,0.85,0.87,0.88,0.9, 0.92,0.94,0.95,0.97,1]',
        'DVR_4_20_[0.2,0.3,0.4, 0.5,0.55,0.6,0.65,0.7,0.75, 0.8,0.81,0.83,0.84,0.85,0.87,0.88,0.9, 0.92,0.94,0.95,0.97,1]',
        'DVR_4_15_[0.2,0.3,0.4, 0.5,0.55,0.6,0.65,0.7,0.75, 0.8,0.81,0.83,0.84,0.85,0.87,0.88,0.9, 0.92,0.94,0.95,0.97,1]',
        'DVR_4_10_[0.2,0.3,0.4, 0.5,0.55,0.6,0.65,0.7,0.75, 0.8,0.81,0.83,0.84,0.85,0.87,0.88,0.9, 0.92,0.94,0.95,0.97,1]'
        # 'DVR_4_20_[0.1, 0.3, 0.7, 0.8]',
        # 'DVR_4_15_[0.1, 0.3, 0.7, 0.8]',
        # 'DVR_2_30_[0.1, 0.7]',
        # 'DVR_2_20_[0.1, 0.7]',
        # 'DVR_2_15_[0.1, 0.7]'
        # 'DVR_2_15_[0.1, 0.7]',
        # 'DVR_2_10_[0.1, 0.7]',
        # 'DVR_2_5_[0.1, 0.7]',
        # 'DVR_2_3_[0.1, 0.7]'
        # 'DVR_2_10_[0.2, 0.6]'
    ]

    # last_train = int(N*0.6-1)
    PA_in  = x_train
    PA_out = y_train



    for test_state in test_map:
        count = count + 1
        Model = test_state.split('_')
        if Model[0] == 'DVR': # DVR [K,M]
            # 参数设置
            M = ast.literal_eval(Model[2])
            threshold = ast.literal_eval(Model[3])
            K = len(threshold)

            filename = f"DVR_K{K}_M{M}_threshold{threshold}_{time.strftime('%Y%m%d%H%M')}"
            if count==1:
                parser = argparse.ArgumentParser(description='configTemplates')
                parser.add_argument('-log_path', default='./results/log', type=str, help='log file path to save result')
                args = parser.parse_args()
                logger = fun.create_logger(args.log_path, filename)
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')

            # print(f'------------------------{Model}_M{M}_{layer_dims}_{activation}-------------------------------')
            logger.info(f'------------------------{Model[0]}_M{M}_K{K}-------------------------------')
            # 初始化模型
            dvr = DVR.DVR(M=M, threshold=threshold)
            coef = dvr.DVR_e(x_train, y_train,alpha=1e-3)
            y_pred = dvr.DVR_v(x_train, coef)
            logger.info(f'{Model[0]} train NMSE:')
            NMSE = cal.nmse(y_train[M+11:],y_pred[M+11:], logger,1)
            y_pred = dvr.DVR_v(x, coef)
            logger.info(f'{Model[0]} coef num {len(coef)}')

        elif Model[0] == 'GMP':
            K = ast.literal_eval(Model[1])#[int(num) for num in re.findall(r'\d+', Model[1])]
            L = ast.literal_eval(Model[2])#[int(num) for num in re.findall(r'\d+', Model[2])]
            M = ast.literal_eval(Model[3])#[int(num) for num in re.findall(r'\d+', Model[3])]
            GMP = gmp.GMP(K,L,M)
            filename = f"GMP_K{K}_L{L}_M{M}_{time.strftime('%Y%m%d%H%M')}"
            if count==1:
                parser = argparse.ArgumentParser(description='configTemplates')
                parser.add_argument('-log_path', default='./results/log', type=str, help='log file path to save result')
                args = parser.parse_args()
                logger = fun.create_logger(args.log_path, filename)
            logger.info(f'----------------{Model[0]}_K{K}_L{L}_M{M}--------------------')
            start_time = time.time()  # 记录开始时间
            coef = GMP.model_e(PA_in, PA_out)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            # logger.info(f"model train time: {elapsed_time:.6f} s")
            y_pred = GMP.model_v(PA_in, coef)
            logger.info(f'{Model[0]} train NMSE:')

            x_norm = PA_in / max(abs(PA_in))
            y_norm = PA_out.squeeze() / max(abs(PA_out))
            y_pred_norm = y_pred / max(abs(y_pred))
            NMSE = cal.nmse(y_norm,y_pred_norm, logger,1)
            logger.info(f'{Model[0]} coef num {len(coef)}')

            PA_in  = x
            PA_out = y
            start_time = time.time()  # 记录开始时间
            y_pred = GMP.model_v(PA_in, coef)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            # logger.info(f"model prediction time: {elapsed_time:.6f} s")

            X = GMP.get_basis(PA_in)
            # X = X.cpu().detach().numpy()


            # y_pred = y_pred.reshape(-1,1)
        elif Model == 'MP':
            logger.info('----------------MP--------------------')
            coef = mp.MP_e(PA_in, PA_out, 5, 7)
            logger.info(f'{Model} coef num {len(coef)}')
            y_pred = mp.MP_v(PA_in, coef, 5, 7)
            # y_pred = y_pred.reshape(-1, 1)


        x_val = x[20:-30]
        y_val = y[20:-30]
        y_pred = y_pred[20:-30]

        x_norm = x_val/max(abs(x_val))
        y_norm = y_val/max(abs(y_val))
        y_pred_norm = y_pred/max(abs(y_pred))

        filepath = 'figures/polynomial'
        NMSE,ACLR,NMSE_pred,ACLR_pred = fun.calculate_CRZ(x_val,y_val,y_pred_norm,fs,BW,filepath,Model[0],plot_swich,logger)

        # logger.info('NMSE Model:')
        # NMSE_model = cal.nmse(y_val, y_pred_norm, logger, 1)

    NMSE_list.append(NMSE_pred)


    # 计算2-范数条件数
    # cond_number = np.linalg.cond(A)
    #
    # print(f'cond_number: {cond_number}')
    # DPD_with_GMP = gmp.GMP_v(xorg, coef, K=K, L=L, M=M)
    # max(abs(DPD_with_GMP))
    # # 保存数据到MAT文件
    # file_name = 'data/GMP.mat'
    # savemat(file_name, {'DPD': DPD_with_GMP.T, 'ILC': yorg, 'X': xorg})

A = 1