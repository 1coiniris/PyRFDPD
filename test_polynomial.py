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

# 读取PA输入输出信号
data_file = 'data/dataxy400m2G.mat'
# data_file = 'data/signal_100M_NR_fs49152.mat'
data = loadmat(data_file)

Model_map = ['GMP','MP']
# Model_map = ['GMP']

# xorg = x_data[]
xorg = data['x0']
yorg = data['y00']

N = len(xorg)
xorg = xorg / max(abs(xorg))
xorg = xorg*0.9
yorg = yorg / max(abs(yorg))
yorg = yorg*0.9

# yorg = np.concatenate(([0+0j, 0+0j, 0+0j, 0+0j], PA.PA_Voterra(xorg[4:N-1],xorg[3:N-2],xorg[2:N-3],xorg[1:N-4],xorg[0:N-5]), [0+0j]))
# # yorg = PA_DVR.PA_DVR_v1(xorg).reshape(-1,1)
# yorg = yorg / max(abs(yorg))

x = xorg.squeeze()
y = yorg.squeeze()

# plt.xticks(fontsize=20)
# fig, ax = plt.subplots()
t = np.linspace(0, 1, 200)
plt.plot(t,abs(y[0:200]),label = 'PA_Output')
plt.plot(t,abs(x[0:200]),label = 'PA_Input')
# plt.xlim(0,200)
plt.ylim(0,1)
plt.legend()
plt.savefig('figures/polynomial/PA_waveform.png')
# plt.show()
# sleep(5)
plt.close()

fs = 2e9
# import TEST_Plot
plot.psd(
    {"PA input": x,"PA output":y},
    fs=fs,filename='figures/polynomial/PA_Spectrum.png'
)
# a = list(xorg)
# TEST_Plot.plot_power_spectrum({"PA input": x,"PA output":y},100e6)
# TEST_Plot.plot_amam(x, y,filename="figures/amam wo DPD.png")
plot.amam(x, {"PAout":y}, "figures/polynomial/PA_amam.png")
plot.ampm(x, {"PAout":y}, "figures/polynomial/PA_ampm.png")

PA_in = xorg
PA_out = yorg
for Model in Model_map:
    if Model == 'GMP':
        print('----------------GMP--------------------')
        K = [5,5,5]
        L = [10,10,10]
        # M = [10,10]
        M = [2, 2]
        coef = gmp.GMP_e(PA_in, PA_out, K=K,L=L, M=M)
        print(f'{Model} coef num {len(coef)}')
        y_pred = gmp.GMP_v(PA_in, coef,K=K,L=L, M=M)
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

    x_norm = x/max(abs(x))
    y_norm = y/max(abs(y))
    y_pred_norm = y_pred/max(abs(y_pred))

    # 评估结果（示例）
    NMSE_withDPD = 10 * np.log10(sum(abs(y_pred_norm - y_norm)**2) / sum(abs(x_norm)**2))
    # NMSE_withoutDPD = 10 * np.log10(sum(abs(y_norm - x_norm)**2) / sum(abs(x_norm)**2))
    print(f"NMSE with {Model}: {NMSE_withDPD} dB")
    # print(f"NMSE wo   DPD: {NMSE_withoutDPD} dB")

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