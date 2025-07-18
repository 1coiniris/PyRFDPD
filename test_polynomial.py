from operator import concat

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
# from anyio import sleep
from scipy.io import loadmat, savemat
import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# import pyrfdpd.nn as dpdnn
# import pyrfdpd.visa as visa
from pyrfdpd.utils import metrics, plot, align
import PA_DVR
import TEST_Plot
import Model.LSTM_DPD as model
import Model.gmp as gmp
import Model.mp as mp
import PA
import argparse
import time
import Function_Lib as fun
import Function_Calculate as cal


########################## 信号描述 ################################
# signal = '100M' #'LMBA200M'
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

x = xorg.squeeze()
y = yorg.squeeze()



N = len(xorg)
figure_path = 'figures/polynomial'
fun.PA_figure(x,y,fs,figure_path)



# Model_map = ['GMP','MP']
Model_map = ['GMP']

last_train = int(N*0.6-1)
PA_in  = xorg[0:last_train]
PA_out = yorg[0:last_train]

K = [11, 11, 11]
L = [11, 7, 7]
M = [5, 5]

filename = f"GMP_K{K}_L{L}_M{M}_{time.strftime('%Y%m%d%H%M')}"
parser = argparse.ArgumentParser(description='configTemplates')
parser.add_argument('-log_path', default='./results/log', type=str, help='log file path to save result')
args = parser.parse_args()
logger = fun.create_logger(args.log_path, filename)

for Model in Model_map:
    if Model == 'GMP':
        logger.info(f'----------------{Model}_K{K}_L{L}_M{M}--------------------')
        start_time = time.time()  # 记录开始时间
        coef = gmp.GMP_e(PA_in, PA_out, K=K,L=L, M=M)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model train time: {elapsed_time:.6f} s")
        y_pred = gmp.GMP_v(PA_in, coef, K=K, L=L, M=M)
        logger.info(f'{Model} train NMSE:')

        x_norm = PA_in / max(abs(PA_in))
        y_norm = PA_out.squeeze() / max(abs(PA_out))
        y_pred_norm = y_pred / max(abs(y_pred))
        NMSE = cal.nmse(y_norm,y_pred_norm, logger)
        logger.info(f'{Model} coef num {len(coef)}')

        PA_in  = xorg[last_train+1:]
        PA_out = yorg[last_train+1:]
        start_time = time.time()  # 记录开始时间
        y_pred = gmp.GMP_v(PA_in, coef, K=K, L=L, M=M)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")

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
    x = x[last_train+20:]
    y = y[last_train+20:]
    y_pred = y_pred[19:]

    x_norm = x/max(abs(x))
    y_norm = y/max(abs(y))
    y_pred_norm = y_pred/max(abs(y_pred))

    # 评估结果（示例）
    logger.info(f"signal {signal}")
    NMSE = cal.nmse(x, y, logger)
    ACLR = cal.acpr(y, fs, BW, BW, logger)

    # 评估结果（示例）
    logger.info(f"with {Model}:")
    NMSE_pred = cal.nmse(y_norm,y_pred_norm,logger)
    ACLR_pred = cal.acpr(y_pred,fs,BW,BW,logger)

    # fs = 614.4e6
    # print("without DPD")
    # acpr_wo_DPD = metrics.acpr(y,fs,100e6,100e6)
    # print("with DPD")
    # acpr_with_DPD = metrics.acpr(PA_out_withDPD,fs,100e6,100e6)


    plot.psd(
        {"input": x,"pred_output":y_pred,"output":y},
        fs=fs,filename=f'figures/polynomial/{Model}_spec.png'
    )
    plot.amam(x, {"out":y,"pred":y_pred}, norm=0, filename=f"figures/polynomial/{Model}_amam.png")
    plot.ampm(x, {"out": y, "pred": y_pred}, norm=0, filename=f"figures/polynomial/{Model}_ampm.png")

    plt.figure()
    t = np.linspace(0, 1, 400)
    plt.plot(t,abs(y_pred[2000:2400]),label = 'y_pred')
    plt.plot(t,abs(y[2000:2400]),label = 'y')
    # plt.plot(t,abs(DPD[2000:2200]),label = 'DPD')
    # plt.xlim(0,200)
    plt.ylim(0,1)
    plt.legend()
    plt.savefig(f'figures/polynomial/{Model}_waveform.png')

A = 1

DPD_with_GMP = gmp.GMP_v(xorg, coef, K=K, L=L, M=M)
max(abs(DPD_with_GMP))
# 保存数据到MAT文件
file_name = 'data/GMP.mat'
savemat(file_name, {'DPD': DPD_with_GMP.T, 'ILC': yorg, 'X': xorg})