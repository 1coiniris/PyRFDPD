import matplotlib
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
from Instrument import VSA, VSG
from tests.test_volterra import iteration

filename = f"Test_{time.strftime('%Y%m%d%H')}"
parser = argparse.ArgumentParser(description='configTemplates')
parser.add_argument('-log_path', default='./results/20250525/log/', type=str, help='log file path to save result')
args = parser.parse_args()
logger = fun.create_logger(args.log_path, filename)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

plot_swich = 0
NMSE_state_list = []

########################## 信号描述 ################################
signal = '400M' #'LMBA200M'
# signal = 'LMBA200M'
# signal = '100M'
# signal = 'ILC_120M'
# signal = 'ILC'
# signal = 'YU'

x_train, x, fs, BW = fun.get_waveform(signal)


VSG_IP = "192.168.1.30"
fc = 3.5e9
power = -20
wave_filename = "waveform_crz"
SMW_200A = VSG.VSG(VSG_IP)

VSA_IP = "192.168.1.30"
att = 15
Keysight_9030B = VSA.VSA(VSA_IP)


SMW_200A.down_signal(brand="rohde-schwarz", x=x_train, fc=fc,fs=fs, power=power, file_name=wave_filename,logger=logger)

y_raw = Keysight_9030B.collect_signal(name="keysight",fc=fc, fs=fs, att=att,logger=logger)
y_train = align.align(x_train, y_raw)

figure_path = 'figures/test'
if plot_swich:
    fun.PA_figure(x_train, y_train, fs, figure_path)
# for state in range(24):
#     x_train, y_train, x, y, fs, BW = fun.get_data(signal,state = state)
#     figure_path = 'figures/MCP_NN'
#     if plot_swich:
#         fun.PA_figure(x,y,fs,figure_path)

# 模型设置
Model = "GMP"
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
# Model = 'RVTDNN'
# Model = 'DVR_NN'

logger.info(f'----------------{Model}--------------------')

iteration = 5
K = [11, 11, 11]
L = [11, 7, 7]
M = [5, 5]

GMP = gmp.GMP(K,L,M)
# DPD iteration
for idx in range(iteration):
    logger.debug(f"Start the {idx+1}th iteration")
    coef = GMP.GMP_e(y_train,x_train)
    pa_input = GMP.GMP_v(x_train, coef)
    SMW_200A.down_signal(brand="rohde-schwarz", x=pa_input, fc=fc, fs=fs, power=power, file_name=wave_filename,logger=logger)
    pa_output = Keysight_9030B.collect_signal(name="keysight", fc=fc, fs=fs, att=att, logger=logger)
    pa_output = align.align(x_train, pa_output)

logger.debug("DPD done!")

# pa_input = GMP.GMP_v(x, coef)
# SMW_200A.down_signal(brand="rohde-schwarz", x=pa_input, fc=fc, fs=fs, power=power, file_name=wave_filename,logger=logger)
# pa_output = Keysight_9030B.collect_signal(name="keysight", fc=fc, fs=fs, att=att, logger=logger)
# pa_output = align.align(x, pa_output)

filepath = 'figures/test'
NMSE, ACLR, NMSE_pred, ACLR_pred = fun.calculate_CRZ(x_train, y_train, pa_output, fs, BW, filepath, 'GMP', 1, logger)