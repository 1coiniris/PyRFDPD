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
# signal = '400M' #'LMBA200M'
signal = 'LMBA200M'

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

K = [7, 7, 7]
L = [7, 5, 5]
M = [3, 3]

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
    x = x[last_train+1:]
    y = y[last_train+1:]
    y_pred = y_pred[0:]

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