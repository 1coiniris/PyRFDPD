# import matplotlib.pyplot as plt
# from anyio import sleep
import torch
import numpy as np
from scipy.io import savemat
import Model.gmp as gmp
import Model.volterra_nn as MCP_NN
import Model.Mixed_NN as MIX_NN
import Model.DVR as DVR
import Model.Orth_NN as ORTH_NN
import Model.VDTDNN as VDTDNN
import Model.RVTDNN as RVTDNN
import Model.PARA_VD_NN as VD_NN
import argparse
import time
from function import align
from function import Function_Calculate as cal, Function_Lib as fun
from function import Single_Band_PA, ILC
from Instrument import VSA, VSG
from matplotlib import pyplot as plt


filepath = f'tests/20250826/40M'
figure_path = f'{filepath}/figure'
mat_path = f'{filepath}/data'
logger_filename = f"test_{time.strftime('%Y%m%d%H')}"


parser = argparse.ArgumentParser(description='configTemplates')
parser.add_argument('-log_path', default=f'{filepath}/log/', type=str, help='log file path to save result')
args = parser.parse_args()
logger = fun.create_logger(args.log_path, logger_filename)



# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

plot_swich = 1
NMSE_state_list = []

########################## 信号描述 ################################
# signal = '400M' #'LMBA200M'
# signal = 'LMBA200M'
# signal = '100M'
# signal = 'ILC_120M'
# signal = 'ILC'
signal = 'YU'
# signal = '20M'
# signal = '40M'

x_train, x, fs, BW = fun.get_waveform(signal,rate=0.9)

params = {
    'pow': -10,  # output power in dB
    'VSG_IP': '192.168.1.30',  # IP of Vector Signal Generator (VSG)
    'VSA_IP': '192.168.1.36',  # IP of Vector Signal Analyzer (VSA)
    'VSA_type': 'k',  # Type of Vector Signal Analyzer (VSA) 'rs' or 'k'
    'waveformfile': 'waveform_crz',
    'fs': fs,  # sampling rate = 160 MHz
    'fc': 3.5e9,  # carrier frequency = 2.14 GHz
    'att': 10,  # attenuation level of (VSA) in dB
    'type': 1  # test type
}


PA_board = Single_Band_PA.SingleBandPA(params)

y_train = PA_board.transmit(x_train,logger)

NMSE_woDPD = cal.nmse(x_train,y_train)
logger.info(f'NMSE_WO_DPD: {NMSE_woDPD} dB')
ACP = cal.acpr(y_train,fs,BW,BW,logger)

if plot_swich:
    fun.PA_figure(x_train, y_train, fs, figure_path)


savemat(f'{mat_path}/PA_inout.mat', {'x':x_train,'y':y_train})


ilc_in = {
    'y_d': x_train,
    'u_k': x_train,
    'nIterations': 30,
    'type': 'linear',
    'eta': 0.1
}

ilc_out,k_opt = ILC.ILC(PA_board,ilc_in,logger)

mat_filename = f'{mat_path}/ILCOUT.mat'
savemat(mat_filename, ilc_out)


# 模型设置
Model_map = ['DVR_NN','GMP','VDTDNN']
# Model = "GMP"
# Model = 'DVC_NN'
# Model = 'GMP_NN'
# Model = 'RVTD_NN'
# Model = 'MCP_BASE_NN'
# Model = 'PARA_NN'
# Model = 'DVR'
# Model = 'ADVR_NN'
# Model = 'DVR_NN'
# Model = 'VD_DVR_NN'
# Model = 'VDTDNN'
# Model = 'RVTDNN'
# Model = 'DVR_NN'

for Model in Model_map:
    logger.info(f'----------------{Model}--------------------')

    iteration = 1
    if Model == 'GMP':
        K = [7, 7, 7]
        L = [7, 3, 3]
        M = [3, 3]
        GMP = gmp.GMP(K,L,M)
        # coef = GMP.GMP_e(x_train,ilc_out['u_ideal'])
        coef = GMP.GMP_e(x_train,ilc_out['u_ideal'])
        pa_input = GMP.GMP_v(x_train, coef)
        print(f'COEF number: {len(coef)} ')
        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f'{mat_path}/{Model}_K{K}_L{L}_M{M}.mat', {'x':x_train,'u':pa_input,'y_withDPD':pa_output})

    if Model == 'DVR_NN':
        model = ['Tanh', 2, 12, 40]
        # 参数设置
        activation = model[0]
        K = model[1]  # 分段数，可修改
        M = model[2]

        model_path = f"{filepath}/model/{Model}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
        trained_model = 'tests/20250826/model/OB_DVR_NN_M30_K3_Tanh_202508261803.pt'  # LMBA
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        logger.info(f'------------------------{Model}_M{M}_K{K}_{activation}-------------------------------')
        # 初始化模型
        model = ORTH_NN.Orth_Basis_DVR_NN(K=K, M=M, activation=activation).to(device)
        fun.model_structure(model, logger)

        # model.load_state_dict(torch.load(trained_model))
        model.model_train(x_train, ilc_out['u_ideal'], model_path, logger, 1)
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")
        start_time = time.time()  # 记录开始时间
        # 提取参数
        x_coef = x_train[:]
        y_coef = ilc_out['u_ideal'][:]
        x_coef_tensor = torch.from_numpy(x_coef).to(device)
        y_coef_tensor = torch.from_numpy(y_coef).to(device)
        sequences = MCP_NN.create_memory_seq(x_coef, M)  # [N, M+1]
        x_coef_window = torch.from_numpy(sequences).to(device)
        y_sequences = MCP_NN.create_memory_seq(y_coef, M)  # [N, M+1]
        y_coef_window = torch.from_numpy(y_sequences).to(device)
        model.coef = model.DVR_NN_e(x_coef_window,y_coef_tensor)
        pa_input = model.apply_dpd(x_train, model.coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")
        print(f'COEF number: {len(model.coef)} ')

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f'{mat_path}/{Model}_K{K}_M{M}.mat', {'x':x_train,'u':pa_input,'y_withDPD':pa_output})

    if Model == 'VDTDNN':
        # 参数设置
        model = ['Tanh', 10, 5, 12, 12]
        activation = model[0]
        # K = model_map[1]  # 分段数，可修改
        M = model[1]
        K = model[2]
        layer_dims = []

        input_size = K * (M + 1)  # 输入维度
        output_size = 1 * (M + 1)  # 输出维度
        layer_dims.append(input_size)
        for size in model[3:]:
            layer_dims.append(size)
        layer_dims.append(output_size)

        model_path = f"{filepath}/model/{Model}_M{M}_K{K}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        # trained_model = 'results/20250519/save/OB_DVR_NN_M15_K3_Tanh_202507201242.pt'# 好用
        # trained_model = 'results/20250519/save/OB_DVR_NN_M30_K1_Tanh_202507262149.pt' #很好
        trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        logger.info(f'------------------------{Model}_M{M}_K{K}_{activation}-------------------------------')
        # 初始化模型
        model = VDTDNN.VDTDNN(layer_dims, M=M, K=K, activation=activation).to(device)
        fun.model_structure(model, logger)

        # model.load_state_dict(torch.load(trained_model))
        model.model_train(x_train, ilc_out['u_ideal'], model_path, logger, 1)
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")
        start_time = time.time()  # 记录开始时间
        pa_input = model.apply_dpd(x_train)  # , model.coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f'{mat_path}/{Model}_K{K}_M{M}_Layer_{layer_dims}.mat', {'x':x_train,'u':pa_input,'y_withDPD':pa_output})

    if Model == 'RVTDNN':
        # 参数设置
        model = ['Tanh', 10, 7, 24, 24]
        activation = model[0]
        # K = model_map[1]  # 分段数，可修改
        M = model[1]
        K = model[2]
        layer_dims = []

        input_size = 2 * K * (M + 1)  # 输入维度
        output_size = 2  # 输出维度
        layer_dims.append(input_size)
        for size in model[3:]:
            layer_dims.append(size)
        layer_dims.append(output_size)

        model_path = f"{filepath}/model/{Model}_M{M}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        # trained_model = 'results/20250519/save/OB_DVR_NN_M15_K3_Tanh_202507201242.pt'# 好用
        # trained_model = 'results/20250519/save/OB_DVR_NN_M30_K1_Tanh_202507262149.pt' #很好
        trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        logger.info(f'------------------------{Model}_M{M}_{activation}-------------------------------')
        # 初始化模型
        model = RVTDNN.RVTDNN(layer_dims, M=M, K=K, activation=activation).to(device)
        fun.model_structure(model, logger)

        # model.load_state_dict(torch.load(trained_model))
        model.model_train(x_train, ilc_out['u_ideal'], model_path, logger, 1)
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")
        start_time = time.time()  # 记录开始时间
        pa_input = model.apply_dpd(x_train)  # , model.coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f'{mat_path}/{Model}_K{K}_M{M}_Layer_{layer_dims}.mat', {'x':x_train,'u':pa_input,'y_withDPD':pa_output})



    NMSE, ACLR, NMSE_pred, ACLR_pred = fun.calculate_CRZ(x_train, y_train, pa_output, fs, BW, figure_path, Model, 1, logger)


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

# pa_output = ilc_out['ILC_final']
# NMSE_withDPD = cal.nmse(x_train,pa_output)
# logger.info(f'NMSE_WITH_DPD: {NMSE_withDPD} dB')



A = 1


# # 保存数据到MAT文件
# savemat(file_name, {'DPD': DPD_with_GMPNN.T, 'ILC': yorg, 'X': xorg})