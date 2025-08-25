import matplotlib
matplotlib.use('Agg')
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
import Model.gmp as gmp
import Model.mp as mp
import PA
import logging
import argparse
import time
import os

def get_data(signal,rate=0.6,state=3):
    # 读取PA输入输出信号
    if signal == 'YU':
        state_list = [3, 5, 6, 7, 8, 11 ,15, 18, 20 ]
        fs = 491.52e6
        BW = 100e6
        print(f'state {state}')
        data_file = 'data/Data_1104_24_3.mat'
        data = loadmat(data_file)
        xorg = data['x00']
        yorg = data['y00']
        # state = 3
        L = len(xorg)
        val_start = int(L * 0.375 + L * 0.625 / 24 * state + 1)
        val_end = int(L * 0.375 + L * 0.625 / 24 * (state + 1))
        train_start = int(L * 0.375 / 24 * state + 1)
        train_end = int(L * 0.375 / 24 * (state + 1))
        x_train = xorg[train_start:train_end].squeeze()
        y_train = yorg[train_start:train_end].squeeze()
        x = xorg[val_start:val_end].squeeze()
        y = yorg[val_start:val_end].squeeze()
    else:
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
        elif signal == '100M':
            fs = 983.04e6
            BW = 100e6
            data_file = 'data/PA_100M_98304.mat'
            data = loadmat(data_file)
            xorg = data['x']
            yorg = data['y']
        elif signal == 'ILC_120M':
            fs = 1.2288e9
            BW = 120e6
            data_file = 'data/ILC.mat'
            data = loadmat(data_file)
            xorg = data['uBB']
            # ILCOut = data['x']
            yorg = data['xBB']
        elif signal == 'ILC':
            fs = 1.2288e9
            BW = 200e6
            data_file = 'data/ILC_[0  1  1  1  0]_G1_forpython.mat'
            data = loadmat(data_file)
            xorg = data['x']
            # ILCOut = data['x']
            yorg = data['y_ILC']

        N = len(xorg)
        last_train = int(N * rate - 1)
        # 创建数据集
        x_train = xorg[0:last_train].squeeze()
        y_train = yorg[0:last_train].squeeze()
        x = xorg[last_train + 1:].squeeze()
        y = yorg[last_train + 1:].squeeze()

    return x_train, y_train, x, y, fs, BW



def create_logger(logger_file_path,filename):

    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    log_name = '{}.log'.format(filename)
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    file_handler = logging.FileHandler(final_log_file)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

'''方法1，自定义函数 参考自 https://blog.csdn.net/qq_33757398/article/details/109210240'''
def model_structure(model,logger = None):
    blank = ' '
    if logger == None:
        print('-' * 90)
        print('|' + ' ' * 20 + 'weight name' + ' ' * 30 + '|' \
              + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
              + ' ' * 3 + 'number' + ' ' * 3 + '|')
        print('-' * 90)

        num_para = 0
        type_size = 1  # 如果是浮点数就是4
        for index, (key, w_variable) in enumerate(model.named_parameters()):
            if len(key) <= 30:
                key = key + (30 - len(key)) * blank
            shape = str(w_variable.shape)
            if len(shape) <= 40:
                shape = shape + (40 - len(shape)) * blank
            each_para = 1
            for k in w_variable.shape:
                each_para *= k
            num_para += each_para
            str_num = str(each_para)
            if len(str_num) <= 10:
                str_num = str_num + (10 - len(str_num)) * blank

            print('| {} | {} | {} |'.format(key, shape, str_num))
        print('-' * 90)
        print('The total number of parameters: ' + str(num_para))
        print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
        print('-' * 90)
    else:
        logger.info('-' * 120)
        logger.info('|' + ' ' * 20 + 'weight name' + ' ' * 30 + '|' \
                    + ' ' * 10 + 'weight shape' + ' ' * 10 + '|' \
                    + ' ' * 5 + 'number' + ' ' * 5 + '|')
        logger.info('-' * 120)

        num_para = 0
        type_size = 1  # 如果是浮点数就是4
        for index, (key, w_variable) in enumerate(model.named_parameters()):
            if len(key) <= 30:
                key = key + (50 - len(key)) * blank
            shape = str(w_variable.shape)
            if len(shape) <= 40:
                shape = shape + (20 - len(shape)) * blank
            each_para = 1
            for k in w_variable.shape:
                each_para *= k
            num_para += each_para
            str_num = str(each_para)
            if len(str_num) <= 10:
                str_num = str_num + (10 - len(str_num)) * blank

            logger.info('| {} | {} | {} |'.format(key, shape, str_num))
        logger.info('-' * 120)
        logger.info('The total number of parameters: ' + str(num_para))
        logger.info('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
        logger.info('-' * 120)

# 数据预处理函数
def create_dataset(x, y, M, test_size = 0.2):
    # 转换为实部虚部分离格式
    # x_real = torch.view_as_real(x).float()  # [N, 2]
    # y_real = torch.view_as_real(y).float()  # [N, 2]
    X = complex_to_real(x)
    Y = complex_to_real(y)

    # 创建延迟窗口
    sequences = []
    targets = []
    for i in range(M, len(x) - 1):
        # 输入：x(n-M)到x(n)的实部虚部
        window = X[i - M:i + 1].flatten()  # [2*(M+1),]
        # 输出：y(n+1)的实部虚部
        target = Y[i]  # [2,]
        sequences.append(window)
        targets.append(target)
    # if 1:
    #     batch_x_signal = np.stack([a[-2:] for a in sequences])  # 需优化以提高效率
    #     batch_x_signal = torch.from_numpy(batch_x_signal)
    #     # 转换为张量
    #     # batch_x_signal = torch.view_as_real(batch_x_signal).to(torch.float32)
    #     batch_x_signal = torch.view_as_complex(batch_x_signal)
    #     batch_x = np.stack(batch_x_signal[0:200])
    #     batch_y = np.stack([a[0]+1j*a[1] for a in targets[0:200]])  # targets[0:200]
    #     plt.figure()
    #     t = np.linspace(0, 1, 200)
    #     plt.plot(t, abs(batch_y[000:200]), label='y')
    #     plt.plot(t, abs(batch_x[000:200]), label='x')
    #     plt.ylim(0, 1)
    #     plt.legend()
    #     plt.show()
    #     plt.savefig(f'figures/MCP_NN/Dataset_waveform.png')
    X_tensor = torch.FloatTensor(sequences)  # (16375, 2M+2)
    Y_tensor = torch.FloatTensor(targets)  # (16375, 2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=test_size, shuffle=False)
    return [X_train,X_val,Y_train,Y_val]

# ====================== 数据预处理（添加滑动窗口）======================
def create_sequences(data, seq_length):
    """将数据转换为序列格式"""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

def create_sequences_addmemory(data, seq_length, M = 9):
    """将数据转换为序列格式"""
    sequences = []
    for i in range(len(data) - seq_length - M + 1):
        memory = []
        for j in range(seq_length):
            memory.append(data[i+j:i+j+M+1].reshape(2*(M+1)))
        sequences.append(memory)
    return np.array(sequences)

# 数据预处理：将复数转换为实部+虚部
def complex_to_real(x):
    return np.stack((x.real, x.imag), axis=1)

def PA_figure(x,y,fs,filepath):
    N = len(x)
    xnorm = x / max(abs(x))
    # xorg = xorg*0.9
    ynorm = y / max(abs(y))
    # plt.xticks(fontsize=20)
    # fig, ax = plt.subplots()
    # 设置坐标轴线条宽度（粗细）
    plt.rcParams['axes.linewidth'] = 2.0  # 默认值为 0.8
    # 全局设置
    plt.rcParams['xtick.labelsize'] = 12  # X轴刻度标签字体大小
    plt.rcParams['ytick.labelsize'] = 12  # Y轴刻度标签字体大小

    t = np.linspace(0, 1, 200)
    plt.plot(t,abs(ynorm[0:200]),label = 'PA_Output')
    plt.plot(t,abs(xnorm[0:200]),label = 'PA_Input')
    # plt.xlim(0,200)
    plt.ylim(0,1)
    plt.legend()
    plt.savefig(f'{filepath}/PA_waveform.png')
    # plt.show()
    # sleep(5)
    plt.close()

    # fs = 2e9
    # import TEST_Plot
    plot.psd(
        {"PA input": x,"PA output":y},
        fs=fs,filename=f'{filepath}/PA_Spectrum.png'
    )
    # a = list(xorg)
    # TEST_Plot.plot_power_spectrum({"PA input": x,"PA output":y},100e6)
    # TEST_Plot.plot_amam(x, y,filename="figures/amam wo DPD.png")
    plot.amam(x, {"PAout":y}, filename=f"{filepath}/PA_amam.png")
    plot.ampm(x, {"PAout":y}, filename=f"{filepath}/PA_ampm.png")

# def calculate_metrics(args: argparse.Namespace, stat: Dict[str, Any], prediction: np.ndarray, ground_truth: np.ndarray):
#     stat['NMSE'] = metrics.NMSE(prediction, ground_truth)
#     stat['EVM'] = metrics.EVM(prediction, ground_truth, bw_main_ch=args.bw_main_ch, n_sub_ch=args.n_sub_ch, nperseg=args.nperseg)
#     ACLR_L = []
#     ACLR_R = []
#     ACLR_left, ACLR_right = metrics.ACLR(prediction, fs=args.input_signal_fs, nperseg=args.nperseg,
#                                          bw_main_ch=args.bw_main_ch, n_sub_ch=args.n_sub_ch)
#     ACLR_L.append(ACLR_left)
#     ACLR_R.append(ACLR_right)
#     stat['ACLR_L'] = np.mean(ACLR_L)
#     stat['ACLR_R'] = np.mean(ACLR_R)
#     stat['ACLR_AVG'] = (stat['ACLR_L'] + stat['ACLR_R']) / 2
#     return stat