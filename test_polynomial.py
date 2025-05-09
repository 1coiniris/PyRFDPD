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

K = [11, 11, 11]
L = [13, 13, 13]
M = [7, 4]

filename = f"GMP_K{K}_L{L}_M{M}"
parser = argparse.ArgumentParser(description='configTemplates')
parser.add_argument('-log_path', default='./results/log', type=str, help='log file path to save result')
args = parser.parse_args()
logger = fun.create_logger(args.log_path, filename)

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
x = xorg.squeeze()
y = yorg.squeeze()

N = len(xorg)
figure_path = 'figures/polynomial'
fun.PA_figure(x,y,figure_path)
# 评估结果（示例）
logger.info(f"signal 400M:")
NMSE_pred = cal.nmse(x, y, logger)
ACLR_pred = cal.acpr(y, fs, BW, BW, logger)

# Model_map = ['GMP','MP']
Model_map = ['GMP']


PA_in = xorg[0:50000]
PA_out = yorg[0:50000]
for Model in Model_map:
    if Model == 'GMP':
        logger.info(f'----------------{Model}_K{K}_L{L}_M{M}--------------------')
        start_time = time.time()  # 记录开始时间
        coef = gmp.GMP_e(PA_in, PA_out, K=K,L=L, M=M)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model train time: {elapsed_time:.6f} 秒")

        logger.info(f'{Model} coef num {len(coef)}')

        PA_in = xorg[50001:]
        PA_out = yorg[50001:]
        start_time = time.time()  # 记录开始时间
        y_pred = gmp.GMP_v(PA_in, coef, K=K, L=L, M=M)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} 秒")

        # y_pred = y_pred.reshape(-1,1)
    elif Model == 'MP':
        print('----------------MP--------------------')
        coef = mp.MP_e(PA_in, PA_out, 5, 7)
        print(f'{Model} coef num {len(coef)}')
        y_pred = mp.MP_v(PA_in, coef, 5, 7)
        # y_pred = y_pred.reshape(-1, 1)

    # PA_out_withDPD = np.concatenate(([0+0j, 0+0j, 0+0j, 0+0j], PA.PA_Voterra(DPD[4:N-1],DPD[3:N-2],DPD[2:N-3],DPD[1:N-4],DPD[0:N-5]), [0+0j]))
    # PA_out_withDPD = PA_out_withDPD / max(abs(PA_out_withDPD))
    # PA_out_withDPD = PA_out_withDPD.squeeze()
    # PA_out_withDPD = align.align(x,PA_out_withDPD)
    x = x[50001:]
    y = y[50001:]
    y_pred[0:20] = 0
    x_norm = x/max(abs(x))
    y_norm = y/max(abs(y))
    y_pred_norm = y_pred/max(abs(y_pred))

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
    plot.amam(x, {"out":y,"pred":y_pred}, f"figures/polynomial/{Model}_amam.png")
    plot.ampm(x, {"out": y, "pred": y_pred}, f"figures/polynomial/{Model}_ampm.png")

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