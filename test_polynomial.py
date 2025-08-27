

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
for state in range(24):
    signal = 'YU'
    x_train, y_train, x, y, fs, BW = fun.get_data(signal,state = state)
    # 读取PA输入输出信号
    # if signal == '400M':
    #     fs = 2e9
    #     BW = 400e6
    #     data_file = 'data/dataxy400m2G.mat'
    #     data = loadmat(data_file)
    #     xorg = data['x0']
    #     yorg = data['y00']
    # elif signal == 'LMBA200M':
    #     fs = 1e9
    #     BW = 200e6
    #     data_file = 'data/LMBA_200M_23G.mat'
    #     data = loadmat(data_file)
    #     xorg = data['x']
    #     yorg = data['y']
    # elif signal == '100M':
    #     fs = 983.04e6
    #     BW = 100e6
    #     data_file = 'data/PA_100M_98304.mat'
    #     data = loadmat(data_file)
    #     xorg = data['x']
    #     yorg = data['y']
    # elif signal == 'ILC_120M':
    #     fs = 1.2288e9
    #     BW = 120e6
    #     data_file = 'data/ILC.mat'
    #     data = loadmat(data_file)
    #     xorg = data['uBB']
    #     # ILCOut = data['x']
    #     yorg = data['xBB']
    # elif signal == 'ILC':
    #     fs = 1.2288e9
    #     BW = 200e6
    #     data_file = 'data/ILC_[0  1  1  1  0]_G1_forpython.mat'
    #     data = loadmat(data_file)
    #     xorg = data['x']
    #     # ILCOut = data['x']
    #     yorg = data['y_ILC']

    # x = xorg.squeeze()
    # y = yorg.squeeze()



    # N = len(xorg)
    figure_path = 'figures/polynomial'
    if plot_swich:
        fun.PA_figure(x,y,fs,figure_path)



    # Model_map = ['GMP','MP']
    Model_map = ['GMP']

    # last_train = int(N*0.6-1)
    PA_in  = x_train
    PA_out = y_train



    for Model in Model_map:
        if Model == 'DVR':
            M = 15
            threshold = [0.2,0.4,0.6,0.8]

            filename = f"DVR_M{M}_THRESHOLD{threshold}_{time.strftime('%Y%m%d%H%M')}"
            parser = argparse.ArgumentParser(description='configTemplates')
            parser.add_argument('-log_path', default='./results/log', type=str, help='log file path to save result')
            args = parser.parse_args()
            logger = fun.create_logger(args.log_path, filename)
            logger.info(f'----------------{Model}_M{M}_THRESHOLD{threshold}--------------------')
            start_time = time.time()  # 记录开始时间
            DVR = DVR.DVR(M,threshold)
            coef = DVR.DVR_e(PA_in,PA_out)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            logger.info(f"model train time: {elapsed_time:.6f} s")

            y_pred = DVR.DVR_v(PA_in, coef)

            x_norm = PA_in / max(abs(PA_in))
            y_norm = PA_out.squeeze() / max(abs(PA_out))
            y_pred_norm = y_pred / max(abs(y_pred))
            logger.info(f'{Model} train NMSE:')
            NMSE = cal.nmse(y_norm,y_pred_norm, logger)
            logger.info(f'{Model} coef num {len(coef)}')

            PA_in  = x
            PA_out = y
            start_time = time.time()  # 记录开始时间
            y_pred = DVR.DVR_v(PA_in, coef)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            logger.info(f"model prediction time: {elapsed_time:.6f} s")

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

        # PA_out_withDPD = np.concatenate(([0+0j, 0+0j, 0+0j, 0+0j], PA.PA_Voterra(DPD[4:N-1],DPD[3:N-2],DPD[2:N-3],DPD[1:N-4],DPD[0:N-5]), [0+0j]))
        # PA_out_withDPD = PA_out_withDPD / max(abs(PA_out_withDPD))
        # PA_out_withDPD = PA_out_withDPD.squeeze()
        # PA_out_withDPD = align.align(x,PA_out_withDPD)
        x_val = x[20:]
        y_val = y[20:]
        y_pred = y_pred[20:]

        x_norm = x_val/max(abs(x_val))
        y_norm = y_val/max(abs(y_val))
        y_pred_norm = y_pred/max(abs(y_pred))

        filepath = 'figures/polynomial'
        NMSE,ACLR,NMSE_pred,ACLR_pred = fun.calculate_CRZ(x_val,y_val,y_pred_norm,fs,BW,filepath,'GMP',1,logger)
        # # 评估结果（示例）
        # logger.info(f"signal {signal}")
        # NMSE = cal.nmse(x_val, y_val, logger,1)
        # ACLR = cal.acpr(y_val, fs, BW, BW, logger)
        #
        # # 评估结果（示例）
        # logger.info(f"with {Model}:")
        # NMSE_pred = cal.nmse(y_norm,y_pred_norm,logger,1)
        # ACLR_pred = cal.acpr(y_pred,fs,BW,BW,logger)
        #
        # # fs = 614.4e6
        # # print("without DPD")
        # # acpr_wo_DPD = metrics.acpr(y,fs,100e6,100e6)
        # # print("with DPD")
        # # acpr_with_DPD = metrics.acpr(PA_out_withDPD,fs,100e6,100e6)
        #
        # if plot_swich:
        #     plot.psd(
        #         {"input": x_val,"pred_output":y_pred,"output":y},
        #         fs=fs,filename=f'figures/polynomial/{Model}_spec.png'
        #     )
        #     plot.amam(x_val, {"out":y_val,"pred":y_pred}, norm=0, filename=f"figures/polynomial/{Model}_amam.png")
        #     plot.ampm(x_val, {"out": y_val, "pred": y_pred}, norm=0, filename=f"figures/polynomial/{Model}_ampm.png")
        #
        #     plt.figure()
        #     t = np.linspace(0, 1, 100)
        #     plt.plot(t,abs(y_val[1000:1100]),label = 'y')
        #     plt.plot(t,abs(y_pred[1000:1100]),label = 'y_pred')
        #     # plt.plot(t,abs(DPD[2000:2200]),label = 'DPD')
        #     # plt.xlim(0,200)
        #     plt.ylim(0,1)
        #     plt.legend()
        #     plt.savefig(f'figures/polynomial/{Model}_waveform.png')

    NMSE_list.append(NMSE_pred)
    A = 1
    lambda_reg = 1e-2
    A = X.conj().T @ X + lambda_reg * np.eye(X.shape[1])

    # 计算2-范数条件数
    cond_number = np.linalg.cond(A)

    print(f'cond_number: {cond_number}')
    # DPD_with_GMP = gmp.GMP_v(xorg, coef, K=K, L=L, M=M)
    # max(abs(DPD_with_GMP))
    # # 保存数据到MAT文件
    # file_name = 'data/GMP.mat'
    # savemat(file_name, {'DPD': DPD_with_GMP.T, 'ILC': yorg, 'X': xorg})

A = 1