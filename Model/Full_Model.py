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
import Model.AKPTDNN as AKPTDNN
import Model.DVR_NN as DVR_NN
import Model.VDTDNN as VDTDNN
import Model.RVTDNN as RVTDNN
import Model.KFC_NN as KFCNN
import Model.PNRVTDNN as PNRVTDNN
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

def Model_DPD(Model,PA_board,x_train,ilc_out_train,x,mat_path,filepath,logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Model[0] == 'GMP':
        K = ast.literal_eval(Model[1])  # [int(num) for num in re.findall(r'\d+', Model[1])]
        L = ast.literal_eval(Model[2])  # [int(num) for num in re.findall(r'\d+', Model[2])]
        M = ast.literal_eval(Model[3])  # [int(num) for num in re.findall(r'\d+', Model[3])]
        para = ast.literal_eval(Model[-1])
        model = gmp.GMP(K, L, M)
        # coef = GMP.GMP_e(x_train,ilc_out['u_ideal'])
        model.coef = model.model_e(x_train, ilc_out_train, alpha=para)
        pa_input = model.model_v(x, model.coef)
        logger.info(f'COEF number: {len(model.coef)} ')
        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_L{L}_M{M}_{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_L{L}_M{M}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'DVR':
        # 参数设置
        K = ast.literal_eval(Model[1])
        M = ast.literal_eval(Model[2])
        threshold = ast.literal_eval(Model[3])
        para = ast.literal_eval(Model[-1])

        model = DVR.DVR(M=M, threshold=threshold)
        model.coef = model.DVR_e(x_train, ilc_out_train, alpha=para)
        pa_input = model.DVR_v(x, model.coef)
        logger.info(f'COEF number: {len(model.coef)} ')
        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_thres{threshold}_{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(
            f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_thres{threshold}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'DDR':
        r = ast.literal_eval(Model[1])
        K = ast.literal_eval(Model[2])  # [int(num) for num in re.findall(r'\d+', Model[1])]
        M = ast.literal_eval(Model[3])  # [int(num) for num in re.findall(r'\d+', Model[3])]
        para = ast.literal_eval(Model[-1])

        ddr = DDR.DDR(r, K, M)

        coef = ddr.model_e(x_train, ilc_out_train, alpha=para)

        y_pred = ddr.model_v(x_train)
        logger.info(f'{Model[0]} train NMSE:')
        NMSE = cal.nmse(ilc_out_train[M + 11:], y_pred[M + 11:], logger, 1)

        pa_input = ddr.model_v(x)
        logger.info(f'COEF number: {len(coef)} ')
        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_R{r}_K{K}_M{M}_{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_R{r}_K{K}_M{M}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'KFCNN':
        # 参数设置
        # model = ['Tanh', 5, 20, 20]
        activation = Model[1]
        K = ast.literal_eval(Model[2])  # 分段数，可修改
        M = ast.literal_eval(Model[3])
        M2 = ast.literal_eval(Model[4])
        para = ast.literal_eval(Model[-1])

        layer_dims = []

        input_size = M2 + 1  # 输入维度
        output_size = (M + 1) * K  # 输出维度
        layer_dims.append(input_size)
        for size_str in Model[5:-1]:
            size = ast.literal_eval(size_str)
            layer_dims.append(size)
        layer_dims.append(output_size)
        layer_dims_phase = [2 * (M2 + 1), M2 + 1, (M + 1) * K * 2]
        model_path = f"{filepath}/model/{Model[0]}_M{M}_M2{M2}_K{K}_layer{layer_dims}_Phase{layer_dims_phase}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
        trained_model = 'results/20250519/save/VD_DVR_NN_M30_M231_K3_Tanh_202508051529.pt'  # 很好
        # logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        # 初始化模型
        model = KFCNN.KFC_NN(layer_dims=layer_dims, layer_dims_phase=layer_dims_phase, K=K, M=M, M2=M2,
                             activation=activation, alpha=1e-3).to(device)
        fun.model_structure(model, logger)

        model.model_train(x_train, ilc_out_train, model_path, logger, 1, para=para)#[0.002, 350, 1024]
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
        model.coef = model.DVR_NN_e(x_coef_window, y_coef_tensor, alpha=para[-1], pri=1)
        pa_input = model.apply_dpd(x, model.coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")
        logger.info(f'COEF number: {len(model.coef)} ')

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_M2_{M2}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'DVRNN':
        # 参数设置
        activation = Model[1]
        K = ast.literal_eval(Model[2])
        M = ast.literal_eval(Model[3])
        para = ast.literal_eval(Model[-1])
        layer_dims = []

        input_size = (M + 1) * 1  # 输入维度
        output_size = (M + 1) * K  # 输出维度
        layer_dims.append(input_size)
        for size_str in Model[4:-1]:
            size = ast.literal_eval(size_str)
            layer_dims.append(size)
        layer_dims.append(output_size)

        model_path = f"{filepath}/model/{Model[0]}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"
        trained_model = 'tests/20260430/100M/model/DVRNN_M10_K3_Tanh_202605011832.pt'  # LMBA
        # logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        # 初始化模型
        model = DVR_NN.DVR_NN(layer_dims, K=K, M=M, activation=activation).to(device)
        fun.model_structure(model, logger)

        # model.load_state_dict(torch.load(trained_model))
        model.model_train(x_train, ilc_out_train, model_path, logger, 1, para=para)
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
        coef = model.DVR_NN_e(x_coef_window, y_coef_tensor, alpha=para[-1])

        sequences = MCP_NN.create_memory_seq(x, M)  # [N, M+1]
        x_window = torch.from_numpy(sequences).to(device)
        pa_input = model.DVR_NN_v(x_window, coef).cpu().detach().numpy()
        pa_input[0:150] = x[0:150]
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")
        logger.info(f'COEF number: {len(coef)} ')

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_layer{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(
            f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_layer{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'PNRVTDNN':
        activation = Model[1]
        M = ast.literal_eval(Model[2])
        K = ast.literal_eval(Model[3])
        para = ast.literal_eval(Model[-1])
        layer_dims = []

        input_size = 2 * (M + 1) + K * (M + 1) - 1  # 输入维度
        output_size = 2  # 输出维度
        layer_dims.append(input_size)
        for size_str in Model[4:-1]:
            size = ast.literal_eval(size_str)
            layer_dims.append(size)
        layer_dims.append(output_size)
        model_path = f"{filepath}/model/{Model[0]}_M{M}_K{K}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
        # logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        logger.info(
            f'------------------------{Model[0]}_M{M}_K{K}_layer{layer_dims}_{activation}-------------------------------')
        # 初始化模型
        model = PNRVTDNN.PNRVTDNN(layer_dims, M=M, K=K, activation=activation).double().to(device)
        fun.model_structure(model, logger)

        # model.load_state_dict(torch.load(trained_model))
        model.model_train(x_train, ilc_out_train, model_path, logger, para=para)
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")
        start_time = time.time()  # 记录开始时间
        pa_input = model.apply_dpd(x)  # , model.coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(
            f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'VDTDNN':
        # 参数设置
        activation = Model[1]
        M = ast.literal_eval(Model[2])
        K = ast.literal_eval(Model[3])
        para = ast.literal_eval(Model[-1])
        layer_dims = []

        input_size = K * (M + 1)  # 输入维度
        output_size = 1 * (M + 1)  # 输出维度
        layer_dims.append(input_size)
        for size_str in Model[4:-1]:
            size = ast.literal_eval(size_str)
            layer_dims.append(size)
        layer_dims.append(output_size)

        model_path = f"{filepath}/model/{Model[0]}_M{M}_K{K}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
        # logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')

        model = VDTDNN.VDTDNN(layer_dims, M=M, K=K, activation=activation).to(device)
        fun.model_structure(model, logger)

        model.model_train(x_train, ilc_out_train, model_path, logger, 1, para=para)
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")
        start_time = time.time()  # 记录开始时间
        pa_input = model.apply_dpd(x)  # , model.coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(
            f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'RVTDNN':
        activation = Model[1]
        M = ast.literal_eval(Model[2])
        K = ast.literal_eval(Model[3])
        para = ast.literal_eval(Model[-1])
        layer_dims = []

        input_size = 2 * (M + 1) + (M + 1) * K  # 输入维度
        output_size = 2  # 输出维度
        layer_dims.append(input_size)
        for size_str in Model[4:-1]:
            size = ast.literal_eval(size_str)
            layer_dims.append(size)
        layer_dims.append(output_size)

        model_path = f"{filepath}/model/{Model[0]}_M{M}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

        trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
        # logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        # 初始化模型
        model = RVTDNN.RVTDNN(layer_dims, M=M, K=K, activation=activation).double().to(device)
        fun.model_structure(model, logger)

        model.model_train(x_train, ilc_out_train, model_path, logger, para=para)
        model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")
        start_time = time.time()  # 记录开始时间
        pa_input = model.apply_dpd(x)  # , model.coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(
            f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat")

    if Model[0] == 'AKPTDNN':
        # AKPTDNN: M_taps, L, P  (P * L = 24 for complexity control)
        # Config: 'AKPTDNN_10_24_1' -> M_taps=10, L=24, P=1
        #         'AKPTDNN_10_12_2' -> M_taps=10, L=12, P=2
        #         'AKPTDNN_10_8_3'  -> M_taps=10, L=8,  P=3
        #         'AKPTDNN_10_6_4'  -> M_taps=10, L=6,  P=4
        activation = Model[1]
        M_taps = ast.literal_eval(Model[2])
        L = ast.literal_eval(Model[3])
        P = ast.literal_eval(Model[4])
        para = ast.literal_eval(Model[-1])

        model_path = f"{filepath}save/{Model[0]}_M{M_taps}_L{L}_P{P}_{time.strftime('%Y%m%d%H%M')}.pt"
        # logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
        logger.info(f'------------------------{Model[0]}_M{M_taps}_L{L}_P{P}-------------------------------')

        # Initialize model
        model = AKPTDNN.AKPTDNN(M_taps=M_taps, L=L, P=P, activation=activation).to(device)
        fun.model_structure(model, logger)

        # Compute FLOPs and parameters
        flops, n_params = model.get_FLOPs_and_params()
        logger.info(f'AKPTDNN FLOPs: {flops}, Parameters: {n_params}')


        model.model_train(x_train, ilc_out_train, model_path, logger, 1,para=para)
        model.load_state_dict(torch.load(model_path))
        # # Train from scratch (paper: lr=1e-4, epochs=2000, batch_size=2000)
        # model.model_train(
        #     x_train, y_train, model_path, logger, 1,
        #     para=[0.001, 1000, 512]
        # )
        # model.load_state_dict(torch.load(model_path))
        logger.info(f"-------------------load model: {model_path}---------------------")

        start_time = time.time()  # 记录开始时间
        pa_input = model.apply_dpd(x)  # , model.coef)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model prediction time: {elapsed_time:.6f} s")

        pa_output = PA_board.transmit(pa_input, logger)
        savemat(f"{mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat",
                {'x': x, 'u': pa_input, 'y_withDPD': pa_output})
        logger.info(
            f"save mat_file to {mat_path}/{Model[0]}_K{K}_M{M}_Layer_{layer_dims}_{time.strftime('%Y%m%d%H%M')}.mat")

    return pa_output
