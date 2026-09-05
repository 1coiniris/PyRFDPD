# -*- coding: utf-8 -*-
"""
test_OFDM_main.py — OFDM 传输 + DPD 建模主程序 (Python 版)
=================================================================
对应 CODE_MATLAB/OFDM_transmit/main_ofdm_zip_transmit.mlx, 并扩展:
    ZIP 文件 -> 第 k 包 OFDM 信号 (FEC->加扰->QAM->OFDM->滤波->归一化->CAF)
    -> PA 信道 -> 解调 (EVM / 纠错前后误码率 / 星座图)
    -> ILC DPD -> 各模型 DPD (GMP/DVR/DDR/KFCNN/OBDVRNN/DVRNN/
      PNRVTDNN/VDTDNN/RVTDNN, 与 test_main.py 相同)
    -> 保存 无DPD / ILC DPD / 建模后 DPD 数据 (savemat)

依赖:
    sklearn_stub            (最小 sklearn 兼容层, Model/*.py import 需要)
    function/OFDM_transmit.py, function/PA_DVR.py
    Model/*.py              (test_main.py 同款 DPD 模型)

运行: 在 PyRFDPD 目录下执行  python test_OFDM_main.py
输出: tests/20260829/100M/ 下 log / figure / data
"""

import sklearn_stub          # 必须在 import Model 之前 (提供最小 train_test_split)

import os
import time
import argparse
import ast
import numpy as np
import torch
from scipy.io import savemat, loadmat
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

from function import Function_Calculate as cal, Function_Lib as fun
from function import OFDM_transmit as otx
from function import PA_DVR as pa

# ---- DPD 模型 (test_main.py 同款) ----
import Model.gmp as gmp
import Model.DVR as DVR
import Model.DDR as DDR
import Model.KFC_NN as KFCNN
import Model.Orth_NN as ORTH_NN
import Model.DVR_NN as DVR_NN
import Model.VDTDNN as VDTDNN
import Model.RVTDNN as RVTDNN
import Model.PNRVTDNN as PNRVTDNN
import Model.volterra_nn as MCP_NN


filepath = f'tests/20260829/100M'
figure_path = f'{filepath}/figure'
mat_path = f'{filepath}/data'
log_path = f'{filepath}/log'
logger_filename = f"test_OFDM_{time.strftime('%Y%m%d%H')}"

os.makedirs(log_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)
os.makedirs(mat_path, exist_ok=True)
os.makedirs(f'{filepath}/model', exist_ok=True)


parser = argparse.ArgumentParser(description='configTemplates')
parser.add_argument('-log_path', default=f'{filepath}/log/', type=str,
                    help='log file path to save result')
args = parser.parse_args()
logger = fun.create_logger(args.log_path, logger_filename)


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(device)

# ======================================================================
# 配置
# ======================================================================
zipFile = 'data/Headshot.zip'          # 待传输文件
M = 16384                              # QAM 阶数
BW = 100e6                             # 信号带宽 (Hz)
fs = 5 * BW                            # 采样率 = 500 MHz (5 倍过采样)
cpRatio = 1 / 8                        # 循环前缀占比
maxLenPkt = 80000                      # 每包 OFDM 信号长度上限 (采样点)
scrSeed = 20240517                     # 扰码种子 (收发同种子)
thr_dB = 9.5                           # 削峰门限 (CAF_new 削峰用, dB)

fec = {'type': 'ldpc'}                 # LDPC (对应 mlx 推荐配置)
fec['rate'] = 3 / 4                    # 码率 3/4 (冗余 25%)
# fec = {'type': 'ldpc', 'rate': 9/10}  # 码率 9/10 (冗余 10%)
# fec = {'k': 223}                      # 或用 RS(255,223)

SNR_dB = -55                           # 信道配置 (评估小节):
                                       # >0 = Es/N0(dB) 加 AWGN; Inf = 无噪声;
                                       # <=0 = 过 PA 模型 (PA_DVR_v1) + 噪声

caf_cfg = {'thr_dB': thr_dB, 'T': 10, 'hard_clipping': 0,
           'thr_new': thr_dB + 0.3}

logger.info('=' * 70)
logger.info(f' file: {zipFile}')
logger.info(f' modulation: {M}-QAM')
logger.info(f' bandwidth: {BW/1e6:.0f} MHz')
logger.info(f' fs: {fs/1e6:.0f} MHz')
logger.info(f' Es/N0 = {SNR_dB} dB')
logger.info('=' * 70)

p = otx.ofdm_params(BW, M, fs, cpRatio)
assert p['nPilot'] > 0, 'ofdm_params 版本过旧 (无导频)'


# ======================================================================
# 封装函数: zip 文件 -> 第 k 包 OFDM 信号 (含 TX 星座图 + PAPR 计算)
# ======================================================================
def ofdm_tx_kpkt(zipFile, kpkt, p, fec, caf_cfg, scrSeed, figure_path,
                 logger):
    """从 zip 文件读取并生成第 kpkt 个数据包的 OFDM 信号

    参数:
        zipFile     : 待传输文件路径
        kpkt        : 要生成的包号 (1 基, 越界自动夹取)
        p           : ofdm_params 参数字典
        fec         : FEC 配置 dict
        caf_cfg     : 削峰配置 dict (None = 不削峰)
        scrSeed     : 扰码种子
        figure_path : TX 星座图保存目录
        logger      : logging logger

    返回:
        dict: {'x1': 归一化 OFDM 信号, 'sym1': QAM 符号, 'info1': 解调参考,
               'gNorm1': 峰值归一化增益, 'encBits': 编码比特数,
               'papr_pre': 削峰前 PAPR, 'papr_post': 削峰后 PAPR,
               'b_pkt0': 包起始字节, 'b_pkt1': 包结束字节(不含),
               'n_pkt': 总包数, 'bytes0': 原始文件字节}
    """
    with open(zipFile, 'rb') as f:
        bytes0 = np.frombuffer(f.read(), dtype=np.uint8)
    nBytes = bytes0.size

    fec = dict(fec or {})
    useLDPC = bool(fec.get('type')) and str(fec.get('type')).lower() == 'ldpc'
    if useLDPC:
        nFEC, kFEC, codeUsed = otx.ldpc_pcm(fec.get('code'), fec.get('rate'))
    elif fec:
        nFEC, kFEC = 255, fec.get('k', 223)
    else:
        nFEC = kFEC = p['k']
    L = p['N_fft'] + p['N_cp']
    nSymPkt = max(1, int(np.floor(maxLenPkt / L)))
    bytesPkt = int(np.floor(np.floor(nSymPkt * p['N_data'] * p['k'] / nFEC)
                            * kFEC / 8))
    nPkt = max(1, int(np.ceil(nBytes / bytesPkt)))
    logger.info(f'文件共 {nBytes/1e6:.2f} MB = {nBytes} 字节, '
                f'分包 {nPkt} 个, 每包 {bytesPkt} 字节')

    kpkt = min(max(int(kpkt), 1), nPkt)
    b_pkt0 = (kpkt - 1) * bytesPkt
    b_pkt1 = min(kpkt * bytesPkt, nBytes)

    # 发射链路 (FEC -> 加扰 -> QAM -> OFDM -> 滤波 -> 归一化 -> CAF)
    x1, info1 = otx.ofdm_tx_packet(bytes0[b_pkt0:b_pkt1], p, fec,
                                   scr_seed=scrSeed, caf=caf_cfg)
    gNorm1 = info1['g_norm']
    encBits = info1['enc_bits']
    sym1 = info1['tx_sym']

    # ---- TX 星座图 ----
    nShowPt = min(50000, sym1.size)
    idxP = np.random.choice(sym1.size, nShowPt, replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.real(sym1[idxP]), np.imag(sym1[idxP]), '.', markersize=6)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_title(f'TX Constellation ({M}-QAM)')
    fig.tight_layout()
    fig.savefig(f'{figure_path}/constellation_TX.png', dpi=150)
    plt.close(fig)
    logger.info(f'[第 {kpkt} 包] TX 星座图 -> {figure_path}/constellation_TX.png')

    # ---- PAPR (削峰前 / 削峰后) ----
    if caf_cfg:                       # 削峰前信号 = 未削峰的归一化信号
        x_pre = x1 / np.max(np.abs(x1)) * gNorm1
    else:
        x_pre = x1
    papr_pre = pa.papr(x_pre)[0]
    papr_post = pa.papr(x1)[0]
    logger.info(f'[第 {kpkt} 包] PAPR: 削峰前 {papr_pre:.2f} dB'
                f' -> 削峰后 {papr_post:.2f} dB')

    return {'x1': x1, 'sym1': sym1, 'info1': info1, 'gNorm1': gNorm1,
            'encBits': encBits, 'papr_pre': papr_pre, 'papr_post': papr_post,
            'b_pkt0': b_pkt0, 'b_pkt1': b_pkt1, 'n_pkt': nPkt,
            'bytes0': bytes0}


# ======================================================================
# 封装函数: 信道 + 解调 (输出 EVM / 纠错前后误码率 / 星座图)
# ======================================================================
def channel_and_demod(x1, gNorm1, p, info1, fec, scr_seed, SNR_dB,
                      label='RX', plot=True, savepath=None, logger=None):
    """过信道 (PA 模型 / AWGN) 并解调评估 (星座图内嵌于 ofdm_rx_packet)

    返回: (y, res)  y 为 PA/信道输出, res 为解调结果 dict
    """
    # ---- 信道 ----
    if np.isinf(SNR_dB):
        rx1 = x1
        logger.info(f'信道: 无噪声')
    elif SNR_dB > 0:
        N0 = 1 / 10 ** (SNR_dB / 10)
        sigma = np.sqrt(N0 / (2 * p['N_fft'])) * gNorm1
        rx1 = x1 + sigma * (np.random.randn(x1.size) +
                            1j * np.random.randn(x1.size))
        logger.info(f'信道: AWGN, Es/N0 = {SNR_dB:.1f} dB')
    else:
        rx1 = pa.PA_DVR_v1(x1)
        N0 = 1 / 10 ** (-SNR_dB / 10)
        sigma = np.sqrt(N0 / (2 * p['N_fft'])) * gNorm1
        rx1 = rx1 + sigma * (np.random.randn(x1.size) +
                             1j * np.random.randn(x1.size))
        logger.info(f'信道: PA_DVR_v1 + 噪声')
    y = rx1 / np.max(np.abs(rx1))

    # ---- 解调 (EVM / 纠错前后误码率 / 星座图) ----
    res = otx.ofdm_rx_packet(y, p, info1, fec=fec, scr_seed=scr_seed,
                             plot=plot, savepath=savepath, label=label)
    logger.info(f'[{label}] 信道原始误码 (FEC 译码前): '
                f'{res["n_err_raw"]} / {info1["enc_bits"]} 比特 '
                f'(BER = {res["ber_raw"]:.3e})')
    logger.info(f'[{label}] 接收端 EVM (RMS) = {res["evm"]:.3f} %')
    logger.info(f'[{label}] FEC 译码后比特错误 = {res["n_err_after"]} '
                f'(BER = {res["ber_after"]:.3e})')
    return y, res


# ======================================================================
# 第 k 包 OFDM 信号生成 (封装函数调用)
# ======================================================================
kpkt = 1                                # 要单独评估的包号 (1..nPkt)
tx_out = ofdm_tx_kpkt(zipFile, kpkt, p, fec, caf_cfg, scrSeed,
                      figure_path, logger)
x1 = tx_out['x1']
sym1 = tx_out['sym1']
info1 = tx_out['info1']
gNorm1 = tx_out['gNorm1']
encBits1 = tx_out['encBits']
bPkt0 = tx_out['b_pkt0']
bPkt1 = tx_out['b_pkt1']
bytes0 = tx_out['bytes0']
nPkt = tx_out['n_pkt']

# ======================================================================
# 无 DPD: 信道 + 解调评估
# ======================================================================
logger.info('\n================ 无 DPD 传输评估 ================')
y_woDPD, res_woDPD = channel_and_demod(
    x1, gNorm1, p, info1, fec, scrSeed, SNR_dB, label='woDPD',
    plot=True, savepath=f'{figure_path}/constellation_woDPD.png',
    logger=logger)
rxSym_woDPD = res_woDPD['rx_sym']

# ---- 频谱 / AMAM / AMPM ----
acpr_y_l, acpr_y_r = pa.ACLR_ZTE(y_woDPD, BW * 0.93, BW, fs)
logger.info(f'ACLR(PA 输出) = {acpr_y_l:.2f} / {acpr_y_r:.2f} dBc')
nmse_woDPD = pa.NMSE_ZTE(x1, y_woDPD)
logger.info(f'NMSE (x1 vs y_woDPD) = {nmse_woDPD:.2f} dB')

fig, ax = plt.subplots(figsize=(8, 5))
pa.psd_crz(x1, fs, 512, 'b', ax=ax)
pa.psd_crz(y_woDPD, fs, 512, 'r', ax=ax)
ax.set_ylim(-60, 10)
ax.set_title('PSD: TX (blue) vs PA output (red)')
fig.tight_layout()
fig.savefig(f'{figure_path}/psd_woDPD.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
pa.amam(x1, y_woDPD, 'r', ax=ax)
ax.set_title('AM/AM: woDPD')
fig.tight_layout()
fig.savefig(f'{figure_path}/amam_woDPD.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
pa.ampm(x1, y_woDPD, 'r', ax=ax)
ax.set_title('AM/PM: woDPD')
fig.tight_layout()
fig.savefig(f'{figure_path}/ampm_woDPD.png', dpi=150)
plt.close(fig)

# ---- 保存无 DPD 数据 ----
N = len(x1)
x_train = x1[150:int(N * 0.6 - 1)]
y_train = y_woDPD[150:int(N * 0.6 - 1)]
savemat(f"{mat_path}/PA_inout_OFDM_{time.strftime('%Y%m%d%H%M')}.mat",
        {'x_train': x_train, 'y_train': y_train, 'x': x1, 'y': y_woDPD})
logger.info(f"save PA file to {mat_path}/PA_inout_OFDM_"
            f"{time.strftime('%Y%m%d%H%M')}.mat")

# ======================================================================
# ILC DPD (迭代学习控制, 对应 mlx ILC 小节)
# ======================================================================
logger.info('\n================ ILC ================')
y_d = x1                                       # desired output
u_k = x1                                       # initial input
nIterations = 40
type_ilc = 'linear'                            # instantaneous_gain / linear
eta = 0.3

In_ilc = []
Out_ilc = []
u_ideal = u_k.copy()
NMSE_ilc = np.zeros(nIterations)
for k_ilc in range(nIterations):
    y_k = pa.PA_DVR_v1(u_k)
    N0 = 1 / 10 ** (-SNR_dB / 10)
    sigma = np.sqrt(N0 / (2 * p['N_fft'])) * gNorm1
    y_k = y_k + sigma * (np.random.randn(x1.size) + 1j * np.random.randn(x1.size))
    y_k = y_k / np.linalg.norm(y_k) * np.linalg.norm(u_k)
    if k_ilc == 0:
        PA_Out = y_k                           # output for learning

    e_k = y_d - y_k
    nmse_k = pa.NMSE_ZTE(y_d[10:-10], y_k[10:-10])
    NMSE_ilc[k_ilc] = nmse_k
    In_ilc.append(u_ideal.copy())
    Out_ilc.append(y_k.copy())

    if type_ilc == 'instantaneous_gain':
        learning_matrix = np.diag(y_k / u_k)
        u_k = u_k + np.linalg.solve(learning_matrix, e_k)
    elif type_ilc == 'linear':
        u_k = u_k + eta * e_k
    u_ideal = u_k.copy()
    if (k_ilc + 1) % 10 == 0:
        logger.info(f'ILC iter {k_ilc+1:2d}/{nIterations}  '
                    f'NMSE = {nmse_k:.2f} dB')

k_opt = int(np.argmin(NMSE_ilc))
u_ideal = In_ilc[k_opt]
ILC_final = Out_ilc[k_opt]
y_ideal = ILC_final
logger.info(f'ILC 最优迭代 k_opt = {k_opt+1}, NMSE = {NMSE_ilc[k_opt]:.2f} dB')

# ---- ILC 输出解调评估 (ILC 输出已是 PA 输出, 直接解调, 不再过信道) ----
y_ilc_norm = y_ideal / np.max(np.abs(y_ideal))
res_ilc = otx.ofdm_rx_packet(y_ilc_norm, p, info1, fec=fec,
                             scr_seed=scrSeed, plot=True,
                             savepath=f'{figure_path}/constellation_ILC.png',
                             label='ILC')
logger.info(f'[ILC] 信道原始误码 (FEC 译码前): '
            f'{res_ilc["n_err_raw"]} / {encBits1} 比特 '
            f'(BER = {res_ilc["ber_raw"]:.3e})')
logger.info(f'[ILC] 接收端 EVM (RMS) = {res_ilc["evm"]:.3f} %')
logger.info(f'[ILC] FEC 译码后比特错误 = {res_ilc["n_err_after"]} '
            f'(BER = {res_ilc["ber_after"]:.3e})')
nmse_ilc_out = pa.NMSE_ZTE(x1, y_ideal)
logger.info(f'NMSE (x1 vs ILC_final) = {nmse_ilc_out:.2f} dB')

fig, ax = plt.subplots(figsize=(8, 5))
pa.psd_b(y_woDPD, fs, 512, 'woDPD', ax=ax)
pa.psd_b(y_ideal, fs, 512, 'ILC', ax=ax)
ax.set_ylim(-55, 10)
ax.set_title('PSD: woDPD vs ILC')
ax.legend()
fig.tight_layout()
fig.savefig(f'{figure_path}/psd_ilc.png', dpi=150)
plt.close(fig)

# ---- 保存 ILC DPD 数据 ----
ilc_out = {
    'u_k': u_k, 'PA_Out': PA_Out, 'u_ideal': u_ideal,
    'ILC_final': ILC_final, 'NMSE': NMSE_ilc, 'k_opt': k_opt + 1,
}
savemat(f"{mat_path}/ILCOUT_OFDM_{time.strftime('%Y%m%d%H%M')}.mat",
        ilc_out)
logger.info(f"save ilc file to {mat_path}/ILCOUT_OFDM_"
            f"{time.strftime('%Y%m%d%H%M')}.mat")

ilc_out_train = u_ideal[150:int(N * 0.6 - 1)]   # 模型训练目标 (与 test_main.py 一致)

# ======================================================================
# 建模 DPD: 遍历各模型 (test_main.py 同款)
# ======================================================================
test_map = [
    # ---- 传统多项式模型 (快) ----
    'GMP_[5, 5, 5]_[8, 5, 5]_[2, 2]',
    'DVR_3_7_[0.2,0.5,0.8]',
    'DDR_1_5_10',
    # ---- 神经网络模型 (训练耗时, 按需启用) ----
    # 'KFCNN_Tanh_3_10_12_8',
    # 'OBDVRNN_Tanh_10_3_8_8',
    # 'DVRNN_Tanh_10_3_8_8',
    # 'PNRVTDNN_Tanh_10_3_8_8',
    # 'VDTDNN_Tanh_10_5_16_12_8',
    # 'RVTDNN_Tanh_10_3_8_8',
]

for test_state in test_map:
    Model = test_state.split('_')
    logger.info(f'----------------{test_state}--------------------')

    if Model[0] == 'GMP':
        K = ast.literal_eval(Model[1])
        L = ast.literal_eval(Model[2])
        M_m = ast.literal_eval(Model[3])
        model = gmp.GMP(K, L, M_m)
        model.coef = model.model_e(x_train, ilc_out_train, alpha=5e-3)
        pa_input = model.model_v(x1, model.coef)
        logger.info(f'COEF number: {len(model.coef)} ')
        pa_output = pa.PA_DVR_v1(pa_input)
        savemat(f"{mat_path}/{Model[0]}_K{K}_L{L}_M{M_m}_"
                f"{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x1, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_L{L}_"
                    f"M{M_m}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'DVR':
        K = ast.literal_eval(Model[1])
        M_m = ast.literal_eval(Model[2])
        threshold = ast.literal_eval(Model[3])
        model = DVR.DVR(M=M_m, threshold=threshold)
        model.coef = model.DVR_e(x_train, ilc_out_train, alpha=1)
        pa_input = model.DVR_v(x1, model.coef)
        logger.info(f'COEF number: {len(model.coef)} ')
        pa_output = pa.PA_DVR_v1(pa_input)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M_m}_thres{threshold}_"
                f"{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x1, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M_m}_"
                    f"thres{threshold}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'DDR':
        r = ast.literal_eval(Model[1])
        K = ast.literal_eval(Model[2])
        M_m = ast.literal_eval(Model[3])
        ddr = DDR.DDR(r, K, M_m)
        coef = ddr.model_e(x_train, ilc_out_train, alpha=5e-2)
        pa_input = ddr.model_v(x1)
        logger.info(f'COEF number: {len(coef)} ')
        pa_output = pa.PA_DVR_v1(pa_input)
        savemat(f"{mat_path}/{Model[0]}_R{r}_K{K}_M{M_m}_"
                f"{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x1, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_R{r}_K{K}_"
                    f"M{M_m}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'KFCNN':
        activation = Model[1]
        K = ast.literal_eval(Model[2])
        M_m = ast.literal_eval(Model[3])
        M2 = ast.literal_eval(Model[4])
        layer_dims = [M2 + 1]
        for size_str in Model[5:]:
            layer_dims.append(ast.literal_eval(size_str))
        layer_dims.append((M_m + 1) * K)
        layer_dims_phase = [2 * (M2 + 1), M2 + 1, (M_m + 1) * K * 2]
        model_path = (f"{filepath}/model/{Model[0]}_M{M_m}_M2{M2}_K{K}_"
                      f"layer{layer_dims}_{activation}_"
                      f"{time.strftime('%Y%m%d%H%M')}.pt")
        logger.info(f'------signal BW{BW/1e6}M fs{fs/1e6}MHz------')
        model = KFCNN.KFC_NN(layer_dims=layer_dims,
                             layer_dims_phase=layer_dims_phase,
                             K=K, M=M_m, M2=M2,
                             activation=activation, alpha=1e-3).to(device)
        fun.model_structure(model, logger)
        model.model_train(x_train, ilc_out_train, model_path, logger, 1,
                          para=[0.002, 350, 1024])
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}----------")
        start_time = time.time()
        sequences = fun.create_memory_seq(x_train, M2)
        x_coef_window = torch.from_numpy(sequences).to(device)
        y_sequences = fun.create_memory_seq(ilc_out_train, M2)
        y_coef_window = torch.from_numpy(y_sequences).to(device)
        model.coef = model.DVR_NN_e(x_coef_window, y_coef_window,
                                    alpha=1e-3, pri=1)
        pa_input = model.apply_dpd(x1, model.coef)
        logger.info(f"model prediction time: {time.time()-start_time:.6f} s")
        logger.info(f'COEF number: {len(model.coef)} ')
        pa_output = pa.PA_DVR_v1(pa_input)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M_m}_"
                f"{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x1, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M_m}_"
                    f"{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'OBDVRNN':
        activation = Model[1]
        K = ast.literal_eval(Model[2])
        M_m = ast.literal_eval(Model[3])
        layer_dims = [M_m + 1]
        for size_str in Model[4:]:
            layer_dims.append(ast.literal_eval(size_str))
        layer_dims.append((M_m + 1) * K)
        model_path = (f"{filepath}/model/{Model[0]}_M{M_m}_K{K}_"
                      f"layer{layer_dims}_{activation}_"
                      f"{time.strftime('%Y%m%d%H%M')}.pt")
        logger.info(f'------signal BW{BW/1e6}M fs{fs/1e6}MHz------')
        model = ORTH_NN.Orth_Basis_DVR_NN(layer_dims, K=K, M=M_m,
                                          activation=activation).to(device)
        fun.model_structure(model, logger)
        model.model_train(x_train, ilc_out_train, model_path, logger, 1,
                          para=[0.001, 100, 512])
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}----------")
        start_time = time.time()
        coef = model.model_e(x_train, ilc_out_train, pri=1, alpha=1e-2)
        pa_input = model.model_v(x1, coef)
        logger.info(f"model prediction time: {time.time()-start_time:.6f} s")
        logger.info(f'COEF number: {len(coef)} ')
        pa_output = pa.PA_DVR_v1(pa_input)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M_m}_"
                f"{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x1, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M_m}_"
                    f"{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'DVRNN':
        activation = Model[1]
        K = ast.literal_eval(Model[2])
        M_m = ast.literal_eval(Model[3])
        layer_dims = [(M_m + 1) * 1]
        for size_str in Model[4:]:
            layer_dims.append(ast.literal_eval(size_str))
        layer_dims.append((M_m + 1) * K)
        model_path = (f"{filepath}/model/{Model[0]}_M{M_m}_K{K}_"
                      f"{activation}_{time.strftime('%Y%m%d%H%M')}.pt")
        logger.info(f'------signal BW{BW/1e6}M fs{fs/1e6}MHz------')
        model = DVR_NN.DVR_NN(layer_dims, K=K, M=M_m,
                              activation=activation).to(device)
        fun.model_structure(model, logger)
        model.model_train(x_train, ilc_out_train, model_path, logger, 1,
                          para=[0.001, 1000, 512, 1e-1])
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}----------")
        start_time = time.time()
        sequences = MCP_NN.create_memory_seq(x_train, M_m)
        x_coef_window = torch.from_numpy(sequences).to(device)
        y_sequences = MCP_NN.create_memory_seq(ilc_out_train, M_m)
        y_coef_window = torch.from_numpy(y_sequences).to(device)
        coef = model.DVR_NN_e(x_coef_window, y_coef_window, alpha=1e-3)
        sequences = MCP_NN.create_memory_seq(x1, M_m)
        x_window = torch.from_numpy(sequences).to(device)
        pa_input = model.DVR_NN_v(x_window, coef).cpu().detach().numpy()
        pa_input[0:150] = x1[0:150]
        logger.info(f"model prediction time: {time.time()-start_time:.6f} s")
        logger.info(f'COEF number: {len(coef)} ')
        pa_output = pa.PA_DVR_v1(pa_input)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M_m}_layer{layer_dims}_"
                f"{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x1, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M_m}_"
                    f"layer{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'PNRVTDNN':
        activation = Model[1]
        M_m = ast.literal_eval(Model[2])
        K = ast.literal_eval(Model[3])
        layer_dims = [2 * (M_m + 1) + K * (M_m + 1) - 1]
        for size_str in Model[4:]:
            layer_dims.append(ast.literal_eval(size_str))
        layer_dims.append(2)
        model_path = (f"{filepath}/model/{Model[0]}_M{M_m}_K{K}_"
                      f"{activation}_{time.strftime('%Y%m%d%H%M')}.pt")
        logger.info(f'------signal BW{BW/1e6}M fs{fs/1e6}MHz------')
        model = PNRVTDNN.PNRVTDNN(layer_dims, M=M_m, K=K,
                                  activation=activation).double().to(device)
        fun.model_structure(model, logger)
        model.model_train(x_train, ilc_out_train, model_path, logger,
                          [0.001, 1000, 512])
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}----------")
        start_time = time.time()
        pa_input = model.apply_dpd(x1)
        logger.info(f"model prediction time: {time.time()-start_time:.6f} s")
        pa_output = pa.PA_DVR_v1(pa_input)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M_m}_Layer_{layer_dims}_"
                f"{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x1, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M_m}_"
                    f"Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'VDTDNN':
        activation = Model[1]
        M_m = ast.literal_eval(Model[2])
        K = ast.literal_eval(Model[3])
        layer_dims = [K * (M_m + 1)]
        for size_str in Model[4:]:
            layer_dims.append(ast.literal_eval(size_str))
        layer_dims.append(1 * (M_m + 1))
        model_path = (f"{filepath}/model/{Model[0]}_M{M_m}_K{K}_"
                      f"Layer_{layer_dims}_{activation}_"
                      f"{time.strftime('%Y%m%d%H%M')}.pt")
        logger.info(f'------signal BW{BW/1e6}M fs{fs/1e6}MHz------')
        model = VDTDNN.VDTDNN(layer_dims, M=M_m, K=K,
                              activation=activation).to(device)
        fun.model_structure(model, logger)
        model.model_train(x_train, ilc_out_train, model_path, logger, 1,
                          para=[0.002, 1400, 512])
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}----------")
        start_time = time.time()
        pa_input = model.apply_dpd(x1)
        logger.info(f"model prediction time: {time.time()-start_time:.6f} s")
        pa_output = pa.PA_DVR_v1(pa_input)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M_m}_Layer_{layer_dims}_"
                f"{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x1, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M_m}_"
                    f"Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'RVTDNN':
        activation = Model[1]
        M_m = ast.literal_eval(Model[2])
        K = ast.literal_eval(Model[3])
        layer_dims = [2 * (M_m + 1) + (M_m + 1) * K]
        for size_str in Model[4:]:
            layer_dims.append(ast.literal_eval(size_str))
        layer_dims.append(2)
        model_path = (f"{filepath}/model/{Model[0]}_M{M_m}_Layer_"
                      f"{layer_dims}_{activation}_"
                      f"{time.strftime('%Y%m%d%H%M')}.pt")
        logger.info(f'------signal BW{BW/1e6}M fs{fs/1e6}MHz------')
        model = RVTDNN.RVTDNN(layer_dims, M=M_m, K=K,
                              activation=activation).double().to(device)
        fun.model_structure(model, logger)
        model.model_train(x_train, ilc_out_train, model_path, logger,
                          para=[0.001, 350, 512])
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}----------")
        start_time = time.time()
        pa_input = model.apply_dpd(x1)
        logger.info(f"model prediction time: {time.time()-start_time:.6f} s")
        pa_output = pa.PA_DVR_v1(pa_input)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M_m}_Layer_{layer_dims}_"
                f"{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x1, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M_m}_"
                    f"Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat")

    # ---- 建模 DPD 解调评估 (星座图) ----
    y_withDPD = pa_output / np.max(np.abs(pa_output))
    nmse_dpd = pa.NMSE_ZTE(x1, y_withDPD)
    acpr_dpd_l, acpr_dpd_r = pa.ACLR_ZTE(y_withDPD, BW * 0.93, BW, fs)
    logger.info(f'[{Model[0]}] NMSE = {nmse_dpd:.2f} dB | '
                f'ACLR = {acpr_dpd_l:.2f} / {acpr_dpd_r:.2f} dBc')

    res_dpd = otx.ofdm_rx_packet(
        y_withDPD, p, info1, fec=fec, scr_seed=scrSeed,
        plot=True, savepath=f'{figure_path}/constellation_DPD_{Model[0]}.png',
        label=Model[0])
    logger.info(f'[{Model[0]}] EVM = {res_dpd["evm"]:.3f} % | '
                f'BER(纠错前) = {res_dpd["ber_raw"]:.3e} | '
                f'BER(纠错后) = {res_dpd["ber_after"]:.3e}')

    # ---- AM/AM 与 AM/PM 对比 ----
    fig, ax = plt.subplots(figsize=(6, 5))
    pa.amam(x1, y_woDPD, 'r', ax=ax)
    pa.amam(x1, y_withDPD, 'b', ax=ax)
    ax.set_title(f'AM/AM: woDPD(red) vs {Model[0]}(blue)')
    fig.tight_layout()
    fig.savefig(f'{figure_path}/amam_{Model[0]}.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    pa.ampm(x1, y_woDPD, 'r', ax=ax)
    pa.ampm(x1, y_withDPD, 'b', ax=ax)
    ax.set_title(f'AM/PM: woDPD(red) vs {Model[0]}(blue)')
    fig.tight_layout()
    fig.savefig(f'{figure_path}/ampm_{Model[0]}.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    pa.psd_crz(y_woDPD, fs, 512, 'b', ax=ax)
    pa.psd_crz(y_withDPD, fs, 512, 'r', ax=ax)
    ax.set_ylim(-55, 10)
    ax.set_title(f'PSD: woDPD(blue) vs {Model[0]}(red)')
    fig.tight_layout()
    fig.savefig(f'{figure_path}/psd_{Model[0]}.png', dpi=150)
    plt.close(fig)

logger.debug("OFDM DPD done!")
