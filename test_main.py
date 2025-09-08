# import matplotlib.pyplot as plt
# from anyio import sleep
import torch
import ast
import numpy as np
from scipy.io import savemat, loadmat
import Model.gmp as gmp
import Model.volterra_nn as MCP_NN
import Model.Mixed_NN as MIX_NN
import Model.DVR as DVR
import Model.Orth_NN as ORTH_NN
import Model.DVR_NN as DVR_NN
import Model.VDTDNN as VDTDNN
import Model.RVTDNN as RVTDNN
import Model.KFC_NN as KFCNN
import Model.PNRVTDNN as PNRVTDNN
import argparse
import time
from function import align
from function import Function_Calculate as cal, Function_Lib as fun
from function import Single_Band_PA, ILC, ILA
from Instrument import VSA, VSG
from matplotlib import pyplot as plt
from scipy.io import loadmat

filepath = f'tests/20250907/100M'
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
signal = '100M'
# signal = 'ILC_120M'
# signal = 'ILC'
# signal = 'YU'
# signal = '20M'
# signal = '40M'

xorg, x_2, fs, BW = fun.get_waveform(signal,rate=0.5)


params = {
    'pow': -11.8,  # output power in dB
    'VSG_IP': '192.168.1.30',  # IP of Vector Signal Generator (VSG)
    'VSA_IP': '192.168.1.40',  # IP of Vector Signal Analyzer (VSA)
    'VSA_type': 'fsw',#'keysight',  # Type of Vector Signal Analyzer (VSA) 'rs' or 'k'
    'waveformfile': 'waveform_crz',
    'fs': fs,  # sampling rate = 160 MHz
    'fc': 3.5e9,  # carrier frequency = 2.14 GHz
    'att': 10,  # attenuation level of (VSA) in dB
    'type': 1  # test type
}


PA_board = Single_Band_PA.SingleBandPA(params)

yorg = PA_board.transmit(xorg,logger)

# y = PA_board.transmit(x,logger)

NMSE_woDPD = cal.nmse(xorg,yorg)
logger.info(f'NMSE_WO_DPD: {NMSE_woDPD} dB')
ACP = cal.acpr(yorg,fs,BW,BW*0.98,logger)

if plot_swich:
    fun.PA_figure(xorg, yorg, fs, figure_path)

N = len(xorg)
x_train = xorg[0:int(N * 0.6 - 1)]
y_train = yorg[0:int(N * 0.6 - 1)]
x = xorg[int(N * 0.6):]
y = yorg[int(N * 0.6):]


savemat(f"{mat_path}/PA_inout_{time.strftime('%Y%m%d%H%M')}.mat", {'x_train':x_train,'y_train':y_train,'x':x,'y':y,})
logger.info(f"save ilc file to {mat_path}/PA_inout_{time.strftime('%Y%m%d%H%M')}.mat")


ilc_in = {
    'y_d': xorg,
    'u_k': xorg,
    'fs': fs,
    'BW': BW,
    'nIterations': 30,
    'type': 'linear',
    'eta': 0.2
}

ilc_out,k_opt = ILC.ILC(PA_board,ilc_in,logger)

mat_filename = f"{mat_path}/ILCOUT_{time.strftime('%Y%m%d%H%M')}.mat"
savemat(mat_filename, ilc_out)
logger.info(f"save ilc file to {mat_path}/ILCOUT_{time.strftime('%Y%m%d%H%M')}.mat")


NMSE, ACLR, NMSE_ILC, ACLR_ILC = fun.calculate_CRZ(xorg, yorg, ilc_out['ILC_final'], fs, BW, figure_path, 'ILC', 1, logger,type='DPD')
logger.info(f"================ ILC Done ================")

# data_file = './tests/20250907/100M/data/ILCOUT_202509072055.mat'
# ilc_out = loadmat(data_file)
# #
# ilc_out['u_ideal'] = ilc_out['u_ideal'].T.squeeze()

ilc_out_train = ilc_out['u_ideal'][0:int(N * 0.6 - 1)]

# 模型设置
test_map = [
    # 'DVR_9_30_[0.1,0.2, 0.3,0.4,0.5,0.6, 0.7, 0.8,0.9]',
    # 'DVR_9_20_[0.1,0.2, 0.3,0.4,0.5,0.6, 0.7, 0.8,0.9]',
    # 'DVR_9_10_[0.1,0.2, 0.3,0.4,0.5,0.6, 0.7, 0.8,0.9]',
    # 'DVR_9_5_[0.1,0.2, 0.3,0.4,0.5,0.6, 0.7, 0.8,0.9]',
    # 'DVR_7_30_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
    # 'DVR_7_20_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
    # 'DVR_7_10_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
    # 'DVR_7_5_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
    # 'DVR_7_30_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
    # 'DVR_4_40_[0.1, 0.3, 0.7, 0.8]',
    # 'DVR_4_30_[0.1, 0.3, 0.7, 0.8]',
    # 'DVR_4_20_[0.1, 0.3, 0.7, 0.8]',
    # 'DVR_4_15_[0.1, 0.3, 0.7, 0.8]',
    # 'DVR_2_30_[0.3, 0.7]',
    # 'DVR_2_20_[0.3, 0.7]',
    # 'DVR_2_10_[0.3, 0.7]',
    # 'DVR_2_5_[0.3, 0.7]',
    # 'DVR_1_30_[ 0.5]',
    # 'DVR_1_20_[ 0.5]',
    # 'DVR_1_10_[ 0.5]',
    # 'GMP_[5, 5, 5]_[3, 3, 3]_[3, 3]',
    # 'GMP_[5, 5, 5]_[5, 5, 5]_[3, 3]',
    # 'GMP_[7, 7, 7]_[5, 5, 5]_[3, 3]',
    # 'GMP_[7, 7, 7]_[6, 6, 5]_[5, 5]',
    # 'GMP_[11, 7, 7]_[11, 7, 7]_[5, 5]',
    # 'GMP_[11, 11, 11]_[11, 7, 7]_[5, 5]',
    # 'GMP_[11, 11, 11]_[11, 11, 11]_[7, 7]',
    # 'GMP_[11, 11, 11]_[11, 11, 11]_[11, 11]',


    # 'DVR_2_5_[0.3, 0.7]',
    # 'DVRNN_ReLU_2_5_10',
    # 'VDTDNN_Tanh_10_1_16_16',
    # 'VDTDNN_Tanh_10_3_16',
    # 'VDTDNN_Tanh_20_3_16',
    # 'VDTDNN_Tanh_20_5_16',
    # 'VDTDNN_Tanh_10_3_16_16',
    # 'VDTDNN_Tanh_20_3_16_16',
    # 'VDTDNN_Tanh_20_5_16_16',
    # 'VDTDNN_Tanh_25_3_16_16',
    # 'VDTDNN_Tanh_25_5_16_16',
    # 'VDTDNN_Tanh_20_5_32_32',
    # 'VDTDNN_Tanh_20_7_32_16_16',
    'VDTDNN_Tanh_20_7_16_32_32',
    # 'VDTDNN_Tanh_20_5_32_32_32',

    # 'VDTDNN_Tanh_30_7_32_32_32_32',
    # 'VDTDNN_ReLU_30_7_32_32_32_32_32',
    # 'VDTDNN_ReLU_35_7_32_64_64_32_32',

    # 'RVTDNN_Tanh_10_3_16',
    # 'RVTDNN_Tanh_10_5_16',
    # 'RVTDNN_ReLU_10_3_16_16',
    # 'RVTDNN_ReLU_10_5_16_16',
    # 'RVTDNN_ReLU_20_3_16_16',
    # 'RVTDNN_ReLU_20_5_16_16',
    # 'RVTDNN_ReLU_20_5_32_32',
    # 'RVTDNN_ReLU_20_5_16_16_16',
    # 'RVTDNN_ReLU_20_5_32_32_32',
    #
    # 'PNRVTDNN_Tanh_10_3_16',
    # 'PNRVTDNN_Tanh_10_5_16',
    # 'PNRVTDNN_Tanh_10_3_16_16',
    # 'PNRVTDNN_Tanh_10_5_16_16',
    # 'PNRVTDNN_Tanh_20_5_16_16',
    # 'PNRVTDNN_Tanh_20_5_32_32',
    # 'PNRVTDNN_Tanh_20_5_32_32_32',

    # 'DVR_2_10_[0.3, 0.7]',
    # 'DVRNN_ReLU_2_10_10',
    # 'DVR_2_20_[0.3, 0.7]',
    # 'DVRNN_ReLU_2_20_10',
    # 'DVR_4_5_[0.2, 0.4,0.6, 0.8]',
    # 'DVRNN_ReLU_4_5_10',
    # 'DVR_4_10_[0.2, 0.4,0.6, 0.8]',
    # 'DVRNN_ReLU_4_10_10',
    # 'DVR_4_20_[0.2, 0.4,0.6, 0.8]',
    # 'DVRNN_ReLU_4_20_10',
    # 'DVR_7_5_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
    # 'DVRNN_ReLU_7_5_10',
    # 'DVR_7_10_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
    # 'DVRNN_ReLU_7_10_10',
    # 'DVR_7_15_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
    # 'DVRNN_ReLU_7_15_10',
    # 'DVR_7_20_[0.1,0.2, 0.3,0.4,0.6, 0.7, 0.8]',
    # 'DVRNN_ReLU_7_20_10',
    # 'DVR_9_10_[0.1,0.2, 0.3,0.4,0.5,0.6, 0.7, 0.8,0.9]',
    # 'DVRNN_ReLU_9_10_10',
    # 'DVR_9_25_[0.1,0.2, 0.3,0.4,0.5,0.55,0.6,0.65, 0.7, 0.75,0.8,0.9]',
    # 'DVRNN_ReLU_9_20_10',
    # 'DVR_9_20_[0.1,0.2, 0.3,0.4,0.5,0.6, 0.7, 0.8,0.9]',
    # 'DVRNN_ReLU_9_30_10',
    # 'DVRNN_ReLU_9_20',
    # 'DVRNN_ReLU_9_10',
    # 'DVRNN_ReLU_9_5',
    # 'DVRNN_ReLU_7_20',


    # 'KFCNN_Tanh_3_10_30',
    # 'VDTDNN_Tanh_10_3_16_16',
    # 'RVTDNN_Tanh_10_3_16_16'

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
        model.coef = model.model_e(x_train,ilc_out_train,alpha=1e-3)
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
        model.coef = model.DVR_e(x_train, ilc_out_train,alpha=5e-1)
        pa_input = model.DVR_v(x, model.coef)
        logger.info(f'COEF number: {len(model.coef)} ')
        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_thres{threshold}_{time.strftime('%Y%m%d%H%M')}.mat", {'x':x,'u':pa_input,'y_withDPD':pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_thres{threshold}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'KFCNN':
        # 参数设置
        # model = ['Tanh', 5, 20, 20]
        activation = Model[1]
        K = ast.literal_eval(Model[2])  # 分段数，可修改
        M = ast.literal_eval(Model[3])
        M2 = ast.literal_eval(Model[4])

        model_path = f"{filepath}/model/{Model[0]}_M{M}_M2{M2}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
        trained_model = 'results/20250519/save/VD_DVR_NN_M30_M231_K3_Tanh_202508051529.pt'  # 很好
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        # 初始化模型
        model = KFCNN.KFC_NN(K=K, M=M, M2=M2, activation=activation).to(device)
        fun.model_structure(model, logger)

        model.model_train(x_train, ilc_out_train, model_path, logger, 1,para=[0.001,250,256])
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
        model.coef = model.DVR_NN_e(x_coef_window, y_coef_tensor, alpha=1e-2, pri=1)
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

        layer_dims = []

        input_size = M + 1  # 输入维度
        output_size = (M + 1) * K  # 输出维度
        layer_dims.append(input_size)
        for size_str in Model[4:]:
            size = ast.literal_eval(size_str)
            layer_dims.append(size)
        layer_dims.append(output_size)

        model_path = f"{filepath}/model/{Model[0]}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
        trained_model = 'tests/20250826/model/OB_DVR_NN_M30_K3_Tanh_202508261803.pt'  # LMBA
        logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        # 初始化模型
        model = DVR_NN.DVR_NN(layer_dims,K=K, M=M, activation=activation).to(device)
        fun.model_structure(model, logger)

        # model.load_state_dict(torch.load(trained_model))
        model.model_train(x_train, ilc_out_train, model_path, logger, 1,para=[0.001,250,512])
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")
        start_time = time.time()  # 记录开始时间
        # 提取参数
        # x_coef = x_train[:]
        # y_coef = ilc_out['u_ideal'][:]
        # x_coef_tensor = torch.from_numpy(x_coef).to(device)
        # y_coef_tensor = torch.from_numpy(y_coef).to(device)
        # sequences = MCP_NN.create_memory_seq(x_coef, M)  # [N, M+1]
        # x_coef_window = torch.from_numpy(sequences).to(device)
        # y_sequences = MCP_NN.create_memory_seq(y_coef, M)  # [N, M+1]
        # y_coef_window = torch.from_numpy(y_sequences).to(device)
        coef = model.DVR_NN_e(x_train,ilc_out_train,alpha=1e-6)
        pa_input = model.DVR_NN_v(x, coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")
        logger.info(f'COEF number: {len(coef)} ')

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_layer{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat", {'x':x,'u':pa_input,'y_withDPD':pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_{time.strftime('%Y%m%d%H%M')}.mat")

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
        model.model_train(x_train, ilc_out_train, model_path, logger, [0.001, 350, 512])
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

        model.model_train(x_train, ilc_out_train, model_path, logger, 1,para=[0.001,300,512])
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


# ila_out = ILA.ILA(PA_board,model,x_train,10,logger)
# mat_filename = f"{mat_path}/ILAOUT_{model.name}_{time.strftime('%Y%m%d%H%M')}.mat"
# savemat(mat_filename, {'PA_out':ila_out['PA_Out'],'NMSE':ila_out['NMSE']})
# logger.info(f"save ilc file to {mat_filename}")

A = 1

# ILA.ILA(PA_board,model,x_train,5,logger)
# # 保存数据到MAT文件
# savemat(file_name, {'DPD': DPD_with_GMPNN.T, 'ILC': yorg, 'X': xorg})