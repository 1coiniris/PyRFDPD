import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
# from anyio import sleep
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat
from torch.utils.data import DataLoader, TensorDataset
from pyrfdpd.utils import plot
import Model.volterra_nn as MCP_NN
import Model.Mixed_NN as MIX_NN
import Model.DVR as DVR
import Model.gmp as gmp
import Model.DVR_NN as DVR_NN
import Model.Orth_NN as ORTH_NN
import Model.VDTDNN as VDTDNN
import Model.RVTDNN_LU as RVTDNN
import Model.PNRVTDNN as PNRVTDNN
import Model.KFC_NN as KFCNN
import argparse
import time
from function import Function_Calculate as cal, Function_Lib as fun
from function import FunctionAnalyzer as analyzer
from tqdm import tqdm



# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

file_path = 'results/modeling/'

plot_swich = 1
NMSE_state_list = []
########################## 信号描述 ################################
# signal = '400M' #'LMBA200M'
# signal = 'LMBA200M'
# signal = '100M'
# signal = 'ILC_120M'
# signal = 'ILC_400M'
signal = 'NXP_100M'
# signal = 'YU'
count = 0
for state in range(1):
    x_train, y_train, x, y, fs, BW = fun.get_data(signal,state = state)
    figure_path = 'results/modeling/figure'
    if plot_swich:
        fun.PA_figure(x,y,fs,figure_path)

    # test_map = []
    # for prefix in ['RVTDNN_LeakyReLU_10_1_',]:#'DVRNN_Tanh_5_10','DVRNN_Tanh_7_10','DVRNN_Tanh_7_15']: #'PNRVTDNN_ReLU_10_3_','PNRVTDNN_ReLU_10_5_',
    #     NN_map = fun.generate_NN_map(prefix,1,[[6,12],[10,12],[6,12]],2)
    #     test_map.extend(NN_map)
    # 模型设置
    test_map = ['RVTDNN_LeakyReLU_10_1_10_10']
    #     # 'KFCNN_ReLU_1_7_7',


    total_train = 1

    NMSE_list = []



    for test_state in test_map:
        count = count + 1
        Model = test_state.split('_')
        filename = f"Modeling_{time.strftime('%Y%m%d%H')}"
        if count ==1:
            parser = argparse.ArgumentParser(description='configTemplates')
            parser.add_argument('-log_path', default=f'{file_path}log/', type=str, help='log file path to save result')
            args = parser.parse_args()
            logger = fun.create_logger(args.log_path, filename)

        if Model[0] == 'RVTDNN':
            activation = Model[1]
            M = ast.literal_eval(Model[2])
            K = ast.literal_eval(Model[3])
            layer_dims = []

            input_size = 2 * (M + 1)  # 输入维度
            output_size = 2  # 输出维度
            layer_dims.append(input_size)
            for size_str in Model[4:]:
                size = ast.literal_eval(size_str)
                layer_dims.append(size)
            layer_dims.append(output_size)

            model_path = f"{file_path}save/{Model[0]}_M{M}_Layer_{layer_dims}_{activation}_{time.strftime('%Y%m%d%H%M')}.pt"

            trained_model = 'results/20250519/save/VDTDNN_M15_Tanh_202507292246.pt'  # 400M
            logger.info(f'------signal BW{BW / 1e6}M fs{fs / 1e6}MHz------')
            logger.info(f'------------------------{Model[0]}_M{M}_K{K}_layer{layer_dims}_{activation}-------------------------------')
            # 初始化模型
            model = RVTDNN.RVTDNN_LU(layer_dims, M=M, K=K,activation=activation).double().to(device)
            fun.model_structure(model, logger)

            if total_train == 1:
                loss = model.model_train(
                    x_train, y_train,
                    model_path="model.pth",
                    logger=logger,
                    para=[0.001, 50, 512],  # iterations=50
                    method="layerwise"
                )
                # model.load_state_dict(torch.load(trained_model))
                # model.model_train(x_train, y_train, model_path, logger,para = [0.001,450,512])
                # model.load_state_dict(torch.load(model_path))
                # logger.info(f"-------------------load model: {model_path}---------------------")
                # start_time = time.time()  # 记录开始时间
                y_pred = model.apply_dpd(x)#, model.coef)
                # end_time = time.time()  # 记录结束时间
                # elapsed_time = end_time - start_time
                # logger.info(f"model prediction time: {elapsed_time:.6f} s")


        NMSE, ACLR, NMSE_pred, ACLR_pred = fun.calculate_CRZ(x, y, y_pred, fs, BW, figure_path, Model[0], 1, logger,type='model')
        # x_norm = x/max(abs(x))
        # y_norm = y/max(abs(y))
        # y_pred_norm = y_pred/max(abs(y_pred))
        #
        # # 评估结果（示例）
        # # NMSE_pred = 10 * np.log10(sum(abs(y_pred_norm - y_norm)**2) / sum(abs(y_norm)**2))
        # logger.info(f"signal {signal}")
        # NMSE = cal.nmse(x,y,logger,1)
        # ACLR = cal.acpr(y,fs,BW,BW,logger)
        #
        #
        #
        # logger.info(f"{Model[0]} modeling:")
        # NMSE_pred = cal.nmse(y_norm,y_pred_norm,logger,1)
        # ACLR_pred = cal.acpr(y_pred,fs,BW,BW,logger)
        # NMSE_list.append(NMSE_pred)
        # NMSE_state_list.append(NMSE_pred)
        # if plot_swich:
        #     plot.psd({"input": x,"pred_output":y_pred,"output":y},fs=fs,filename=f'figures/MCP_NN/{Model[0]}_spec.png')
        #     plot.amam(x, {"out":y,"pred":y_pred}, norm=0, filename=f"figures/MCP_NN/{Model[0]}_amam.png")
        #     plot.ampm(x, {"out": y, "pred": y_pred}, norm=0, filename=f"figures/MCP_NN/{Model[0]}_ampm.png")
        #     plot.waveform({"y":y,"y_pred":y_pred},200,offset=200,norm=0,filename=f"figures/MCP_NN/{Model[0]}_waveform.png")


# for para,nmse in zip(model[0],NMSE_list):
#     logger.info(f"{para[0]}_M{para[1]}_L{para[2]}_{para[3:]}_nmse{nmse}dB")
#
# best = min(NMSE_list)
# key = NMSE_list.index(best)
#
# logger.info(f"best NMSE {best} | best para{model[key]}")




# 计算X^H X
B = 1






# if __name__ == "__main__":
#     analysis_results = main()
# X = model.get_basis_amp(x_window)
# X = X.cpu().detach().numpy()
# lambda_reg = 0
# A = X.conj().T @ X + lambda_reg * np.eye(X.shape[1])
#
# # 计算2-范数条件数
# cond_number = np.linalg.cond(A)
#
# print(f'cond_number: {cond_number}')


# DPD_with_GMPNN = model.apply_dpd(xorg)
# print(max(abs(DPD_with_GMPNN)))
# # 保存数据到MAT文件
# file_name = 'data/GMPNN.mat'
# savemat(file_name, {'DPD': DPD_with_GMPNN.T, 'ILC': yorg, 'X': xorg})