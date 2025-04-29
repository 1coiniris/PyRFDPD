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
import PA

# 读取PA输入输出信号
data_file = 'data/signal_100M_NR_fs6144.mat'
# data_file = 'data/signal_100M_NR_fs49152.mat'
N = 163840
x_data = loadmat(data_file)
# xorg = x_data[]
xorg = x_data['signal_100M_fs6144'][0:N]
xorg = xorg / max(abs(xorg))
xorg = xorg*0.9

yorg = np.concatenate(([0+0j, 0+0j, 0+0j, 0+0j], PA.PA_Voterra(xorg[4:N-1],xorg[3:N-2],xorg[2:N-3],xorg[1:N-4],xorg[0:N-5]), [0+0j]))
# yorg = PA_DVR.PA_DVR_v1(xorg).reshape(-1,1)
yorg = yorg / max(abs(yorg))

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
plt.savefig('figures/GMP/PA_waveform.png')
# plt.show()
# sleep(5)
plt.close()

fs = 614.4e6
# import TEST_Plot
plot.psd(
    {"PA input": x,"PA output":y},
    fs=fs,filename='figures/GMP/PA_Spectrum.png'
)
# a = list(xorg)
# TEST_Plot.plot_power_spectrum({"PA input": x,"PA output":y},100e6)
# TEST_Plot.plot_amam(x, y,filename="figures/amam wo DPD.png")
plot.amam(x, {"wo DPD":y}, "figures/GMP/PA_amam.png")

PA_in = xorg
PA_out = yorg
coef = gmp.GMP_e(PA_out, PA_in, K=[5,5,5],L=[7,7,7], M=[2,2])
print(coef)
DPD = gmp.GMP_v(PA_in, coef,K=[5,5,5],L=[7,7,7], M=[2,2])
DPD = DPD.reshape(-1,1)
# plt.plot(np.abs(PA_in[0:1000]), label="PA original input")
# plt.plot(np.abs(PA_out[0:1000]), label="PA original output")
# plt.plot(np.abs(PA_exp[0:1000]), label="PA expected output")
# plt.legend()
# plt.show()

PA_out_withDPD = np.concatenate(([0+0j, 0+0j, 0+0j, 0+0j], PA.PA_Voterra(DPD[4:N-1],DPD[3:N-2],DPD[2:N-3],DPD[1:N-4],DPD[0:N-5]), [0+0j]))
# PA_out_withDPD = PA_out_withDPD / max(abs(PA_out_withDPD))
PA_out_withDPD = PA_out_withDPD.squeeze()
# PA_out_withDPD = align.align(x,PA_out_withDPD)

x_norm = x/max(abs(x))
y_norm = y/max(abs(y))
PA_out_withDPD_norm = PA_out_withDPD/max(abs(PA_out_withDPD))
# 评估结果（示例）
mse = np.mean(np.abs(x_norm - PA_out_withDPD_norm)**2)
print(f"MSE with DPD: {mse:.6f}")
mse = np.mean(np.abs(x_norm - y_norm)**2)
print(f"MSE wo   DPD: {mse:.6f}")

NMSE_withDPD = 10 * np.log10(sum(abs(PA_out_withDPD_norm - x_norm)**2) / sum(abs(x_norm)**2))
NMSE_withoutDPD = 10 * np.log10(sum(abs(y_norm - x_norm)**2) / sum(abs(x_norm)**2))
print(f"NMSE with DPD: {NMSE_withDPD} dB")
print(f"NMSE wo   DPD: {NMSE_withoutDPD} dB")

fs = 614.4e6
print("without DPD")
acpr_wo_DPD = metrics.acpr(y,fs,100e6,100e6)
print("with DPD")
acpr_with_DPD = metrics.acpr(PA_out_withDPD,fs,100e6,100e6)


plot.psd(
    {"input": x,"output_with_DPD":PA_out_withDPD,"output_wo_DPD":y},
    fs=fs,filename='figures/GMP/GMP_DPD_spec.png'
)
plot.amam(x, {"wo DPD":y,"DPD":DPD,"with DPD":PA_out_withDPD}, "figures/GMP/GMP_DPD_amam.png")

plt.figure()
t = np.linspace(0, 1, 20)
plt.plot(t,abs(PA_out_withDPD[2000:2020]),label = 'PA_Output')
plt.plot(t,abs(x[2000:2020]),label = 'PA_Input')
# plt.plot(t,abs(DPD[2000:2200]),label = 'DPD')
# plt.xlim(0,200)
plt.ylim(0,1)
plt.legend()
plt.savefig('figures/GMP/DPD_waveform.png')

A = 1