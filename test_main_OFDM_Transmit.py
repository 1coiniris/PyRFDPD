import torch
import ast
import numpy as np
from scipy.io import savemat, loadmat
import Model.gmp as gmp
import Model.volterra_nn as MCP_NN
import Model.Mixed_NN as MIX_NN
import Model.DVR as DVR
import Model.DDR as DDR
import Model.Orth_NN as ORTH_NN
import Model.DVR_NN as DVR_NN
import Model.VDTDNN as VDTDNN
import Model.RVTDNN as RVTDNN
import Model.KFC_NN as KFCNN
import Model.PNRVTDNN as PNRVTDNN
import Model.Full_Model as Full_Model
import argparse
import time
import os
import function.demodulation as demod
from function import align
from function import Function_Calculate as cal, Function_Lib as fun
from function import OFDM_transmit as OFDM
from function import OFDM_rx as OFDM_rx
from function import Single_Band_PA, ILC, ILA,Board_PA
from Instrument import VSA, VSG
from matplotlib import pyplot as plt
from scipy.io import loadmat
import matplotlib


filepath = 'tests/20261013/100M'
figure_path = f'{filepath}/figure'
mat_path = f'{filepath}/data'
log_path = f'{filepath}/log'
model_path = f'{filepath}/model'
logger_filename = f"test_OFDM_{time.strftime('%Y%m%d%H')}"

os.makedirs(filepath, exist_ok=True)
os.makedirs(log_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)
os.makedirs(mat_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

parser = argparse.ArgumentParser(description='configTemplates')
parser.add_argument('-log_path', default=log_path, type=str, help='log file path to save result')
parser.add_argument('-simulate', type=int, default=0, help='1 = loopback without instruments (debug demod chain)')
args = parser.parse_args()
logger = fun.create_logger(args.log_path, logger_filename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

plot_swich = 1
NMSE_state_list = []
demodulate = 1


matplotlib.use('Agg')

MAT_FILE = 'data/OFDM_16384QAM/signal_100M_fs500_OFDM_16384QAM.mat'
QAM_NPZ  = 'data/OFDM_16384QAM/qam16384_ref.npz'
H_MAT_DIR = 'data/OFDM_16384QAM'
# LDPC 校验矩阵按码率区分 (kFEC = 每码字信息位长):
#   rate 3/4 -> kFEC=48600, rate 5/6 -> kFEC=54000 (n 恒为 64800)
H_FILES_BY_K = {
    48600: 'dvb2_64800_r075_H.mat',   # DVB-S2 64800 rate 3/4
    54000: 'dvb2_64800_r056_H.mat',   # DVB-S2 64800 rate 5/6
}


logger.info('===========================load signal================================')

signal = 'OFDM_100M_16384QAM'

xorg, x_2, fs, BW = fun.get_waveform(signal, rate=1)

if demodulate == 1:
    # 从 OFDM .mat 文件加载信号及解调参数
    mat = loadmat('data/OFDM_16384QAM/signal_100M_fs500_OFDM_16384QAM.mat')

    sym1 = np.asarray(mat['sym1']).ravel()  # 发射 QAM 符号 (含哑元)
    txBits1 = np.asarray(mat['txBits1']).ravel().astype(np.uint8)  # 加扰后编码比特
    encBits1 = int(np.asarray(mat['encBits1']).ravel()[0])  # 预FEC比特数
    scrSeed = int(np.asarray(mat['scrSeed']).ravel()[0])  # 扰码种子
    bytes0 = np.asarray(mat['bytes']).ravel().astype(np.uint8)  # 本包原始字节
    b1e = int(np.asarray(mat['b1e']).ravel()[0])  # 本包字节数
    nFEC = int(np.asarray(mat['nFEC']).ravel()[0])  # LDPC 码长 64800
    kFEC = int(np.asarray(mat['kFEC']).ravel()[0])  # LDPC 信息位 (与码率对应)
    Mq = int(np.asarray(mat['M']).ravel()[0])  # 16384
    nPadBits = (-b1e * 8) % kFEC  # 编码填充比特数

    p_demod = OFDM_rx.matlab_p_to_python(mat['p'])
    qam_tbl = OFDM_rx.load_qam_table(Mq, QAM_NPZ)

    # 按 .mat 中实际码率选择对应的 DVB-S2 校验矩阵文件
    _hname = H_FILES_BY_K.get(kFEC, f'dvb2_64800_k{kFEC}_H.mat')
    _hfull = os.path.join(H_MAT_DIR, _hname)
    if not os.path.exists(_hfull):
        raise FileNotFoundError(
            f'缺少 LDPC 校验矩阵文件 {_hfull} (nFEC={nFEC}, kFEC={kFEC})。\n'
            f'请在 MATLAB 中执行: H = dvbs2ldpc({kFEC / nFEC:.6g}, ''sparse''); '
            f'save(''{_hfull}'', ''H'', ''-v7'')  后重试。')
    _hm = loadmat(_hfull)
    H_ldpc = _hm['H'].tocsr()
    nL, kL = int(np.asarray(_hm['n']).ravel()[0]), \
        int(np.asarray(_hm['k']).ravel()[0])
    assert (nL, kL) == (nFEC, kFEC), \
        f'H 文件码率不匹配: {_hname} 是 (n={nL}, k={kL}), .mat 需要 (n={nFEC}, k={kFEC})'
    prep = OFDM_rx._dvb_bp_prep(H_ldpc)

    logger.info(f'LDPC: dvb2 (n={nFEC}, k={kFEC}), H file={_hname}, H nnz={H_ldpc.nnz}')

    res_x = OFDM.demodulate_ofdm_transmit(xorg,p_demod=p_demod,sym1=sym1,qam_tbl=qam_tbl,encBits1=encBits1,
                                          txBits1=txBits1,scrSeed=scrSeed,H_ldpc=H_ldpc,nFEC=nFEC,kFEC=kFEC,
                                          nPadBits=nPadBits,bytes0=bytes0,b1e=b1e,logger=logger,figure_path=figure_path,label='origin')
    logger.info(f'EVM_origin: {res_x["evm"]:.4f} %')

# ---- 4) 传输平台配置 (真实仪器; 按实际台架修改 IP/功率/频率/衰减) ----
logger.info('===========================PA Transmit================================')

params = {
    'pow': -11.5,             # VSG 输出功率 dBm (参考 MATLAB 实验 pow=-13)
    'VSG_IP': '192.168.2.22',  # ← 按台架修改
    'VSA_IP': '192.168.1.30',  # ← 按台架修改
    'VSA_type': 'fsw',      # 'fsw' / 'keysight' / 'rs'
    'waveformfile': 'waveform_ofdm_16384qam',
    'fs': fs,               # 500 MHz
    'fc_1': 3.6e9,            # 载波频率 (与仪器混频一致)
    'fc_2': 3.6e9,          # 载波频率 (与仪器混频一致)
    'att': 3,              # VSA 输入衰减 dB
    'type': 1               # 1=真实PA 0=PA模型(PA_DVR_v1)
}

PA_board = Board_PA.BoardPA(params)
logger.info(f"power: {params['pow']} dB, fc_1: {params['fc_1']/1e9:.2f} GHz, fc_2: {params['fc_2']/1e9:.2f} GHz")

yorg = PA_board.transmit(xorg, logger,avg_noise=1)

NMSE_woDPD = cal.nmse(xorg, yorg)
logger.info(f'NMSE_WO_DPD: {NMSE_woDPD} dB')
ACP = cal.acpr(yorg, fs, BW, BW * 0.93, logger)

if plot_swich:
    fun.PA_figure(xorg, yorg, fs, figure_path)
    if demodulate == 1:
        res_woDPD = OFDM.demodulate_ofdm_transmit(yorg,p_demod=p_demod,sym1=sym1,qam_tbl=qam_tbl,encBits1=encBits1,
                                              txBits1=txBits1,scrSeed=scrSeed,H_ldpc=H_ldpc,nFEC=nFEC,kFEC=kFEC,
                                              nPadBits=nPadBits,bytes0=bytes0,b1e=b1e,logger=logger,figure_path=figure_path,label='woDPD')
        logger.info(f'EVM_woDPD: {res_woDPD["evm"]:.4f} %')

N = len(xorg)
x_train = xorg[150:int(N * 0.6 - 1)]
y_train = yorg[150:int(N * 0.6 - 1)]
# x = xorg[int(N * 0.6):]
# y = yorg[0:int(N * 0.6):]
x = xorg
y = yorg

savemat(f"{mat_path}/PA_inout_{time.strftime('%Y%m%d%H%M')}.mat", {'x_train':x_train,'y_train':y_train,'x':xorg,'y':yorg,})
logger.info(f"save PA file to {mat_path}/PA_inout_{time.strftime('%Y%m%d%H%M')}.mat")


logger.info('===========================ILC================================')
ilc_in = {
    'y_d': xorg,
    'u_k': xorg,
    'fs': fs,
    'BW': BW,
    'nIterations': 25,
    'type': 'linear',
    'eta': 0.15
}

# ilc_out,k_opt = ILC.ILC(PA_board,ilc_in,logger)
#
# NMSE, ACLR, NMSE_ILC, ACLR_ILC = fun.calculate_CRZ(xorg, yorg, ilc_out['ILC_final'], fs, BW, figure_path, 'ILC', 1, logger,type='DPD')
#
# if demodulate == 1:
#     res_ILC = OFDM.demodulate_ofdm_transmit(ilc_out['ILC_final'], p_demod=p_demod, sym1=sym1, qam_tbl=qam_tbl, encBits1=encBits1,
#                                               txBits1=txBits1, scrSeed=scrSeed, H_ldpc=H_ldpc, nFEC=nFEC, kFEC=kFEC,
#                                               nPadBits=nPadBits, bytes0=bytes0, b1e=b1e, logger=logger,
#                                               figure_path=figure_path, label='ILCDPD')
#     logger.info(f'EVM_origin: {res_ILC["evm"]:.4f} %')
#
# mat_filename = f"{mat_path}/ILCOUT_{time.strftime('%Y%m%d%H%M')}.mat"
# savemat(mat_filename, ilc_out)
# logger.info(f"save ilc file to {mat_path}/ILCOUT_{time.strftime('%Y%m%d%H%M')}.mat")
#
#
# logger.info(f"================ ILC Done ================")


data_file = './tests/20261013/100M/data/ILCOUT_202610171509.mat'
ilc_out = loadmat(data_file)
ilc_out['u_ideal'] = ilc_out['u_ideal'].T.squeeze()
# y_ilc = PA_board.transmit(ilc_out['u_ideal'], logger,avg_noise=48)
# NMSE, ACLR, NMSE_ILC, ACLR_ILC = fun.calculate_CRZ(xorg, yorg, y_ilc, fs, BW, figure_path, 'ILC', 1, logger,type='DPD')
#
# if demodulate == 1:
#     res_ILC = OFDM.demodulate_ofdm_transmit(y_ilc, p_demod=p_demod, sym1=sym1, qam_tbl=qam_tbl, encBits1=encBits1,
#                                               txBits1=txBits1, scrSeed=scrSeed, H_ldpc=H_ldpc, nFEC=nFEC, kFEC=kFEC,
#                                               nPadBits=nPadBits, bytes0=bytes0, b1e=b1e, logger=logger,
#                                               figure_path=figure_path, label='ILCDPD')
#     logger.info(f'EVM_origin: {res_ILC["evm"]:.4f} %')


ilc_out_train = ilc_out['u_ideal'][150:int(N * 0.6 - 1)]

# 模型设置
test_map = [
    # 'DVR_3_20_[0.2,0.5,0.8]_1e-9',
    # 'DVR_4_20_[0.2,0.4,0.6,0.8]_1e-9',
    # 'DVR_5_20_[0.2,0.4,0.6,0.7,0.8]_1e-9',
    # 'DVR_6_20_[0.2,0.4,0.6,0.7,0.8,0.9]_1e-9',
    # 'DVR_9_20_[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]_1e-9',
    # 'DVR_9_30_[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]_1e-9',
    # 'DVR_9_40_[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]_1e-9',
    # 'DVR_9_45_[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]_1e-9',
    # 'DVR_9_50_[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]_1e-9',
    # 'GMP_[9, 7, 7]_[40, 7, 7]_[5, 5]',
    # 'GMP_[9, 7, 7]_[40, 10, 10]_[5, 5]',
    # 'GMP_[9, 7, 7]_[40, 20, 20]_[5, 5]',
    'DVRNN_Tanh_5_40_6_8',
    # 'DVRNN_Tanh_3_55_6_12_8_8',

    # 'PNRVTDNN_Tanh_40_3_16_16_12_8',

]


for test_state in test_map:
    Model = test_state.split('_')
    logger.info(f'----------------{test_state}--------------------')

    iteration = 1

    if Model[0] == 'GMP':
        K = ast.literal_eval(Model[1])  # [int(num) for num in re.findall(r'\d+', Model[1])]
        L = ast.literal_eval(Model[2])  # [int(num) for num in re.findall(r'\d+', Model[2])]
        M = ast.literal_eval(Model[3])  # [int(num) for num in re.findall(r'\d+', Model[3])]
        model = gmp.GMP(K,L,M)
        # coef = GMP.GMP_e(x_train,ilc_out['u_ideal'])
        model.coef = model.model_e(x_train,ilc_out_train,alpha=1e-5)
        pa_input = model.model_v(x, model.coef)
        logger.info(f'COEF number: {len(model.coef)} ')
        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_L{L}_M{M}_{time.strftime('%Y%m%d%H%M')}.mat", {'x':x,'u':pa_input,'y_withDPD':pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_L{L}_M{M}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'DVR':
        # 参数设置
        K = ast.literal_eval(Model[1])
        M = ast.literal_eval(Model[2])
        threshold = ast.literal_eval(Model[3])

        model = DVR.DVR(M=M, threshold=threshold)
        model.coef = model.DVR_e(x_train, ilc_out_train,alpha=1e-5)
        pa_input = model.DVR_v(x, model.coef)
        logger.info(f'COEF number: {len(model.coef)} ')
        pa_output = PA_board.transmit(pa_input, logger,avg_noise=48)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_thres{threshold}_{time.strftime('%Y%m%d%H%M')}.mat", {'x':x,'u':pa_input,'y_withDPD':pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_thres{threshold}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'DDR':
        r = ast.literal_eval(Model[1])
        K = ast.literal_eval(Model[2])  # [int(num) for num in re.findall(r'\d+', Model[1])]
        M = ast.literal_eval(Model[3])  # [int(num) for num in re.findall(r'\d+', Model[3])]
        ddr = DDR.DDR(r, K, M)

        coef = ddr.model_e(x_train, ilc_out_train,alpha=5e-2)

        y_pred = ddr.model_v(x_train)
        logger.info(f'{Model[0]} train NMSE:')
        NMSE = cal.nmse(ilc_out_train[M + 11:], y_pred[M + 11:], logger, 1)

        pa_input = ddr.model_v(x)
        logger.info(f'COEF number: {len(coef)} ')
        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_R{r}_K{K}_M{M}_{time.strftime('%Y%m%d%H%M')}.mat", {'x':x,'u':pa_input,'y_withDPD':pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_R{r}_K{K}_M{M}_{time.strftime('%Y%m%d%H%M')}.mat")


    if Model[0] == 'KFCNN':
        # 参数设置
        # model = ['Tanh', 5, 20, 20]
        activation = Model[1]
        K = ast.literal_eval(Model[2])  # 分段数，可修改
        M = ast.literal_eval(Model[3])
        M2 = ast.literal_eval(Model[4])

        layer_dims = []

        input_size = M2 + 1  # 输入维度
        output_size = (M + 1) * K  # 输出维度
        layer_dims.append(input_size)
        for size_str in Model[5:]:
            size = ast.literal_eval(size_str)
            layer_dims.append(size)
        layer_dims.append(output_size)
        layer_dims_phase = [2 * (M2 + 1), M2 + 1, (M + 1) * K * 2]
        model_path = f"{filepath}/model/{Model[0]}_M{M}_M2{M2}_K{K}_layer{layer_dims}_Phase{layer_dims_phase}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
        trained_model = 'results/20250519/save/VD_DVR_NN_M30_M231_K3_Tanh_202508051529.pt'  # 很好
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        # 初始化模型
        model = KFCNN.KFC_NN(layer_dims=layer_dims, layer_dims_phase=layer_dims_phase, K=K, M=M, M2=M2,activation=activation, alpha=1e-3).to(device)
        fun.model_structure(model, logger)

        model.model_train(x_train, ilc_out_train, model_path, logger, 1,para=[0.002,350,1024])
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")
        start_time = time.time()  # 记录开始时间
        # 提取参数
        x_coef = x_train[:]
        y_coef = ilc_out_train[:]
        x_coef_tensor = torch.from_numpy(x_coef).to(device)
        y_coef_tensor = torch.from_numpy(y_coef).to(device)
        sequences = fun.create_memory_seq(x_coef, M2)  # [N, M+1]
        x_coef_window = torch.from_numpy(sequences).to(device)
        y_sequences = fun.create_memory_seq(y_coef, M2)  # [N, M+1]
        y_coef_window = torch.from_numpy(y_sequences).to(device)
        model.coef = model.DVR_NN_e(x_coef_window, y_coef_tensor, alpha=1e-3, pri=1)
        pa_input = model.apply_dpd(x, model.coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")
        logger.info(f'COEF number: {len(model.coef)} ')

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_{time.strftime('%Y%m%d%H%M')}.mat", {'x':x,'u':pa_input,'y_withDPD':pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_M2_{M2}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'OBDVRNN':
        # 参数设置
        activation = Model[1]
        K = ast.literal_eval(Model[2])
        M = ast.literal_eval(Model[3])

        layer_dims = []

        input_size = M + 1  # 输入维度
        output_size = (M + 1) * K  # 输出维度
        layer_dims.append(input_size)
        for size_str in Model[4:]:
            size = ast.literal_eval(size_str)
            layer_dims.append(size)
        layer_dims.append(output_size)

        model_path = f"{filepath}/model/{Model[0]}_M{M}_K{K}_layer{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
        trained_model = 'tests/20250826/model/OB_DVR_NN_M30_K3_Tanh_202508261803.pt'  # LMBA
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        # 初始化模型
        model = ORTH_NN.Orth_Basis_DVR_NN(layer_dims, K=K, M=M, activation=activation).to(device)
        fun.model_structure(model, logger)

        # model.load_state_dict(torch.load(trained_model))
        model.model_train(x_train, ilc_out_train, model_path, logger, 1,para=[0.001,100,512])
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")
        start_time = time.time()  # 记录开始时间
        # 提取参数
        x_coef = x_train[:]
        y_coef = ilc_out_train[:]
        x_coef_tensor = torch.from_numpy(x_coef).to(device)
        y_coef_tensor = torch.from_numpy(y_coef).to(device)
        sequences = fun.create_memory_seq(x_coef, M)  # [N, M+1]
        x_coef_window = torch.from_numpy(sequences).to(device)
        y_sequences = fun.create_memory_seq(y_coef, M)  # [N, M+1]
        y_coef_window = torch.from_numpy(y_sequences).to(device)
        coef = model.model_e(x_train,ilc_out_train,pri=1,alpha=1e-2)
        pa_input = model.model_v(x, coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")
        logger.info(f'COEF number: {len(coef)} ')

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_{time.strftime('%Y%m%d%H%M')}.mat", {'x':x,'u':pa_input,'y_withDPD':pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'DVRNN':
        # 参数设置
        activation = Model[1]
        K = ast.literal_eval(Model[2])
        M = ast.literal_eval(Model[3])
        term = ast.literal_eval(Model[4])
        layer_dims = []

        P = 1
        input_size = (M + 1)  # 输入维度
        output_size = (M + 1) * K * P  # 输出维度
        layer_dims.append(input_size)
        for size_str in Model[5:]:
            size = ast.literal_eval(size_str)
            layer_dims.append(size)
        layer_dims.append(output_size)

        model_path = f"{filepath}/model/{Model[0]}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
        trained_model = 'tests/20250826/model/OB_DVR_NN_M30_K3_Tanh_202508261803.pt'  # LMBA
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        logger.info(f'------------------------{Model[0]}_M{M}_K{K}_term{term}_layer{layer_dims}_{activation}-------------------------------')

        # 初始化模型
        model = DVR_NN.DVR_NN(layer_dims, K=K, M=M, activation=activation, term=term).to(device)
        fun.model_structure(model, logger)


        # model.load_state_dict(torch.load(trained_model))
        model.model_train(x_train, ilc_out_train, model_path, logger, 1,para=[0.001,500,512,1e-1])
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")
        start_time = time.time()  # 记录开始时间
        # 提取参数
        x_coef = x_train[:]
        y_coef = ilc_out_train
        x_coef_tensor = torch.from_numpy(x_coef).to(device)
        y_coef_tensor = torch.from_numpy(y_coef).to(device)
        sequences = MCP_NN.create_memory_seq(x_coef, M)  # [N, M+1]
        x_coef_window = torch.from_numpy(sequences).to(device)
        y_sequences = MCP_NN.create_memory_seq(y_coef, M)  # [N, M+1]
        y_coef_window = torch.from_numpy(y_sequences).to(device)
        coef = model.DVR_NN_e(x_coef_window,y_coef_tensor,alpha=1e-3)

        sequences = MCP_NN.create_memory_seq(x, M)  # [N, M+1]
        x_window = torch.from_numpy(sequences).to(device)
        pa_input = model.DVR_NN_v(x_window, coef).cpu().detach().numpy()
        # pa_input[0:150] = x[0:150]
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")
        logger.info(f'COEF number: {len(coef)} ')

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_layer{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat", {'x':x,'u':pa_input,'y_withDPD':pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_term{term}_layer{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'PNRVTDNN':
        activation = Model[1]
        M = ast.literal_eval(Model[2])
        K = ast.literal_eval(Model[3])
        layer_dims = []

        input_size = 2 * (M + 1) + K * (M + 1) -1  # 输入维度
        output_size = 2  # 输出维度
        layer_dims.append(input_size)
        for size_str in Model[4:]:
            size = ast.literal_eval(size_str)
            layer_dims.append(size)
        layer_dims.append(output_size)
        model_path = f"{filepath}/model/{Model[0]}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        logger.info(
            f'------------------------{Model[0]}_M{M}_K{K}_layer{layer_dims}_{activation}-------------------------------')
        # 初始化模型
        model = PNRVTDNN.PNRVTDNN(layer_dims, M=M, K=K, activation=activation).double().to(device)
        fun.model_structure(model, logger)

        # model.load_state_dict(torch.load(trained_model))
        model.model_train(x_train, ilc_out_train, model_path, logger, [0.001, 1000, 512])
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")
        start_time = time.time()  # 记录开始时间
        pa_input = model.apply_dpd(x)  # , model.coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat",{'x': x, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat")


    if Model[0] == 'VDTDNN':
        # 参数设置
        activation = Model[1]
        M = ast.literal_eval(Model[2])
        K = ast.literal_eval(Model[3])
        layer_dims = []

        input_size = K * (M + 1)  # 输入维度
        output_size = 1 * (M + 1)  # 输出维度
        layer_dims.append(input_size)
        for size_str in Model[4:]:
            size = ast.literal_eval(size_str)
            layer_dims.append(size)
        layer_dims.append(output_size)

        model_path = f"{filepath}/model/{Model[0]}_M{M}_K{K}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')

        model = VDTDNN.VDTDNN(layer_dims, M=M, K=K, activation=activation).to(device)
        fun.model_structure(model, logger)

        model.model_train(x_train, ilc_out_train, model_path, logger, 1,para=[0.002,1400,512])
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")
        start_time = time.time()  # 记录开始时间
        pa_input = model.apply_dpd(x)  # , model.coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat", {'x':x,'u':pa_input,'y_withDPD':pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'RVTDNN':
        activation = Model[1]
        M = ast.literal_eval(Model[2])
        K = ast.literal_eval(Model[3])
        layer_dims = []

        input_size = 2  * (M + 1) + (M+1) * K  # 输入维度
        output_size = 2  # 输出维度
        layer_dims.append(input_size)
        for size_str in Model[4:]:
            size = ast.literal_eval(size_str)
            layer_dims.append(size)
        layer_dims.append(output_size)

        model_path = f"{filepath}/model/{Model[0]}_M{M}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        # 初始化模型
        model = RVTDNN.RVTDNN(layer_dims, M=M, K=K, activation=activation).double().to(device)
        fun.model_structure(model, logger)

        model.model_train(x_train, ilc_out_train, model_path, logger,para=[0.001,350,512])
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")
        start_time = time.time()  # 记录开始时间
        pa_input = model.apply_dpd(x)  # , model.coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat", {'x':x,'u':pa_input,'y_withDPD':pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat")



    NMSE, ACLR, NMSE_pred, ACLR_pred = fun.calculate_CRZ(x, y, pa_output, fs, BW, figure_path, Model[0], 1, logger,type='DPD')

    if demodulate == 1:
        res_DPD = OFDM.demodulate_ofdm_transmit(pa_output, p_demod=p_demod, sym1=sym1, qam_tbl=qam_tbl,
                                                encBits1=encBits1,
                                                txBits1=txBits1, scrSeed=scrSeed, H_ldpc=H_ldpc, nFEC=nFEC, kFEC=kFEC,
                                                nPadBits=nPadBits, bytes0=bytes0, b1e=b1e, logger=logger,
                                                figure_path=figure_path, label=f'{Model[0]}_DPD')
        logger.info(f'EVM {Model[0]} DPD: {res_DPD["evm"]:.4f} %')


# DPD iteration
# for idx in range(iteration):
#     logger.info(f"Start the {idx+1}th iteration")
#     coef = GMP.GMP_e(y_train,x_train)
#     pa_input = GMP.GMP_v(x_train, coef)
#     pa_output = PA_board.transmit(pa_input, logger)
#     SMW_200A.down_signal(brand="rohde-schwarz", x=pa_input, fc=fc, fs=fs, power=power, file_name=wave_filename,logger=logger)
#     pa_output = Keysight_9030B.collect_signal(name="keysight", fc=fc, fs=fs, att=att, logger=logger)
#     pa_output = align.align(x_train, pa_output)

logger.debug("DPD done!")

# for test_state in test_map:
#     Model = test_state.split('_')
#     logger.info(f'----------------{test_state}--------------------')
#
#     iteration = 1
#     pa_output = Full_Model.Model_DPD(Model,PA_board,x_train,ilc_out_train,x,mat_path, filepath, logger)
#     NMSE, ACLR, NMSE_pred, ACLR_pred = fun.calculate_CRZ(x, y, pa_output, fs, BW, figure_path, Model[0],
#                                                          1, logger,type='DPD')
#
#     if demodulate == 1:
#         res_DPD = OFDM.demodulate_ofdm_transmit(pa_output, p_demod=p_demod, sym1=sym1, qam_tbl=qam_tbl,
#                                                 encBits1=encBits1,
#                                                 txBits1=txBits1, scrSeed=scrSeed, H_ldpc=H_ldpc, nFEC=nFEC, kFEC=kFEC,
#                                                 nPadBits=nPadBits, bytes0=bytes0, b1e=b1e, logger=logger,
#                                                 figure_path=figure_path, label=f'{Model[0]}_DPD')
#         logger.info(f'EVM {Model[0]} DPD: {res_DPD["evm"]:.4f} %')
#
#
#
#
# logger.debug("DPD done!")

# pa_output = ilc_out['ILC_final']
# NMSE_withDPD = cal.nmse(x_train,pa_output)
# logger.info(f'NMSE_WITH_DPD: {NMSE_withDPD} dB')


# ila_out = ILA.ILA(PA_board,model,x_train,10,logger)
# mat_filename = f"{mat_path}/ILAOUT_{model.name}_{time.strftime('%Y%m%d%H%M')}.mat"
# savemat(mat_filename, {'PA_out':ila_out['PA_Out'],'NMSE':ila_out['NMSE']})
# logger.info(f"save ilc file to {mat_filename}")

#
# pa_output = ilc_out['ILC_final']
# NMSE_withDPD = cal.nmse(x_train,pa_output)
# logger.info(f'NMSE_WITH_DPD: {NMSE_withDPD} dB')
#
#
# ila_out = ILA.ILA(PA_board,model,x_train,10,logger)
# mat_filename = f"{mat_path}/ILAOUT_{model.name}_{time.strftime('%Y%m%d%H%M')}.mat"
# savemat(mat_filename, {'PA_out':ila_out['PA_Out'],'NMSE':ila_out['NMSE']})
# logger.info(f"save ilc file to {mat_filename}")

A = 1