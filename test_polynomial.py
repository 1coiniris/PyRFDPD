

import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from anyio import sleep
import numpy as np
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
# signal = 'ILC'
plot_swich = 0
NMSE_list = []
count = 0
for state in range(24):
    signal = 'YU'
    x_train, y_train, x, y, fs, BW = fun.get_data(signal,state = state)
    # N = len(xorg)
    figure_path = 'figures/polynomial'
    if plot_swich:
        fun.PA_figure(x,y,fs,figure_path)



    # Model_map = ['GMP','MP']
    Model_map = ['DVR']

    # last_train = int(N*0.6-1)
    PA_in  = x_train
    PA_out = y_train



    for Model in Model_map:
        if Model == 'DVR': # DVR [K,M]
            # 参数设置
            K = 3  # 分段数，可修改
            M = 20
            threshold = np.array([0.2,0.4, 0.6, 0.8])
            # size = (M + 1 + Lb + Lc)
            filename = f"DVR_K{K}_M{M}_threshold{threshold}_{time.strftime('%Y%m%d%H%M')}"
            if count==0:
                parser = argparse.ArgumentParser(description='configTemplates')
                parser.add_argument('-log_path', default='./results/log', type=str, help='log file path to save result')
                args = parser.parse_args()
                logger = fun.create_logger(args.log_path, filename)
            print(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')

            # print(f'------------------------{Model}_M{M}_{layer_dims}_{activation}-------------------------------')
            print(f'------------------------{Model}_M{M}_K{K}-------------------------------')
            # 初始化模型
            dvr = DVR.DVR(M=M, threshold=threshold)
            coef = dvr.DVR_e(x_train, y_train)
            y_pred = dvr.DVR_v(x, coef)


        elif Model == 'GMP':
            K = [11, 11, 11]
            L = [11, 7, 7]
            M = [5, 5]
            GMP = gmp.GMP(K,L,M)
            filename = f"GMP_K{K}_L{L}_M{M}_{time.strftime('%Y%m%d%H%M')}"
            if state==0:
                parser = argparse.ArgumentParser(description='configTemplates')
                parser.add_argument('-log_path', default='./results/log', type=str, help='log file path to save result')
                args = parser.parse_args()
                logger = fun.create_logger(args.log_path, filename)
            logger.info(f'----------------{Model}_K{K}_L{L}_M{M}--------------------')
            start_time = time.time()  # 记录开始时间
            coef = GMP.GMP_e(PA_in, PA_out)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            logger.info(f"model train time: {elapsed_time:.6f} s")
            y_pred = GMP.GMP_v(PA_in, coef)
            logger.info(f'{Model} train NMSE:')

            x_norm = PA_in / max(abs(PA_in))
            y_norm = PA_out.squeeze() / max(abs(PA_out))
            y_pred_norm = y_pred / max(abs(y_pred))
            NMSE = cal.nmse(y_norm,y_pred_norm, logger)
            logger.info(f'{Model} coef num {len(coef)}')

            PA_in  = x
            PA_out = y
            start_time = time.time()  # 记录开始时间
            y_pred = GMP.GMP_v(PA_in, coef)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            logger.info(f"model prediction time: {elapsed_time:.6f} s")

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
        NMSE,ACLR,NMSE_pred,ACLR_pred = fun.calculate_CRZ(x_val,y_val,y_pred_norm,fs,BW,filepath,Model,plot_swich,logger)

        logger.info('NMSE Model:')
        NMSE_model = cal.nmse(y_val, y_pred_norm, logger, 1)

    NMSE_list.append(NMSE_model)
    count = count + 1

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