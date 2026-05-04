import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import pinv
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time
from function import Function_Calculate as cal, Function_Lib as fun
from tqdm import tqdm
from matplotlib import pyplot as plt
import Model.volterra_nn as volterra_nn




class DVR():
    def __init__(self, M,threshold):
        super(DVR, self).__init__()
        self.name = 'DVR'
        self.M = M
        self.threshold = threshold
        self.K = len(self.threshold)

    def DVR_e(self,x, y,alpha=1e-5):
        M = self.M
        # 设置起始索引（跳过前M+10个样本）
        start = M + 1 + 10
        x1 = x[start:]
        y1 = y[start:]
        X1 = self.DVR_get_basis(x1)

        # X1 = X[start:, :]
        # y1 = y[start:].reshape(-1, 1)

        # X1H = np.conjugate(X1.T)
        # coef = np.linalg.pinv(X1H.dot(X1) + 0.001 * np.eye(X1.shape[1])).dot(X1H).dot(y1)

        # 正则化最小二乘求解
        I = np.eye(X1.shape[1])
        coef = pinv(X1.conj().T @ X1 + alpha * I) @ X1.conj().T @ y1 #np.linalg.inv
        y_model = self.DVR_v(x1, coef)
        NMSE = cal.nmse(y1.squeeze(), y_model)
        print(f'NMSE-with-model = {NMSE:.6f} dB')
        return coef


    def DVR_v(self,x,coef):
        M = self.M
        X = self.DVR_get_basis(x)

        # 使用系数进行预测
        y = X @ coef.reshape(-1, 1).flatten()
        # y = X.dot(coef)

        y[0:M+1+10] = x[0:M+1+10]
        # y[abs(y) > 1] = x[abs(y) > 1]
        # a = np.where(abs(y) > 1)
        # indices = np.where(a == 1)
        return y

    def DVR_get_basis(self,x):
        thold = self.threshold
        K = len(thold)
        M = self.M

        xi = x.reshape(-1, 1)  # 转换为列向量
        xip = np.angle(x).reshape(-1, 1)  # 计算相位角（弧度）

        # 初始化特征矩阵列表
        X_lin_list = []
        X_1_list = []
        X_21_list = []
        X_22_list = []
        X_23_list = []
        X_ddr_1_list = []
        X_ddr_2_list = []

        # 循环创建特征
        for m in range(M + 1):  # m从0到M
            xi_shift = np.roll(xi, m, axis=0)  # 循环移位（注意负号方向）
            xip_shift = np.roll(xip, m, axis=0)

            # 线性特征
            X_lin_list.append(xi_shift)

            for k in range(K):  # k从0到K-1
                # 核心非线性项
                xi_core = np.abs(np.abs(xi_shift) - thold[k])

                # 特征组1
                xi_1_shift = xi_core * np.exp(1j * xip_shift)
                X_1_list.append(xi_1_shift)

                # 特征组21
                xi_21_shift = xi_1_shift * np.abs(xi)
                X_21_list.append(xi_21_shift)

                # 仅当m>0时添加以下特征
                if m > 0:
                    # 特征组22
                    xi_22_shift = xi_core * xi
                    X_22_list.append(xi_22_shift)

                    # 特征组23
                    xi_23_shift = xi_core * xi_shift
                    X_23_list.append(xi_23_shift)

                    # DDR特征1
                    xi_ddr_core = np.abs(np.abs(xi) - thold[k])
                    xi_ddr_1 = xi_ddr_core * xi_shift
                    X_ddr_1_list.append(xi_ddr_1)

                    # DDR特征2
                    xi_ddr_2 = xi_ddr_core * xi * xi * np.conj(xi_shift)
                    X_ddr_2_list.append(xi_ddr_2)

        # 将所有特征水平拼接
        X_lin = np.hstack(X_lin_list) if X_lin_list else np.zeros((len(x), 0))
        X_1 = np.hstack(X_1_list) if X_1_list else np.zeros((len(x), 0))
        X_21 = np.hstack(X_21_list) if X_21_list else np.zeros((len(x), 0))
        X_22 = np.hstack(X_22_list) if X_22_list else np.zeros((len(x), 0))
        X_23 = np.hstack(X_23_list) if X_23_list else np.zeros((len(x), 0))
        X_ddr_1 = np.hstack(X_ddr_1_list) if X_ddr_1_list else np.zeros((len(x), 0))
        X_ddr_2 = np.hstack(X_ddr_2_list) if X_ddr_2_list else np.zeros((len(x), 0))

        # 组合所有特征 , X_ddr_2
        X = np.hstack([X_lin, X_1, X_21, X_22, X_23, X_ddr_1, X_ddr_2])
        # X[np.isnan(X)] = 0

        return X


class DVR_2(nn.Module):
    def __init__(self, K, M, threshold, activation="ReLU"):
        super().__init__()
        self.M = M
        self.K = K
        self.threshold = threshold
        assert activation == "ReLU" or "Tanh" or "ELU" or "None"
        self.linear = nn.Linear((K*(M+1)*6+M+1)*2, 2,bias=False).double()
        # self.act = nn.Tanh()
        # self.linear_2 = nn.Linear(12, 2, bias=False).double()
        # self.main_layers = nn.Sequential()
        # self.para_layers = nn.Sequential()
        # self.main_nn = volterra_nn.GMP_NN(self.layer_dim1, L, M = self.M,activation=activation)
        # self.para_nn = volterra_nn.MCP_BASE_NN(self.layer_dim2, L, M = self.M, K = 5,activation=activation)
        # self.out_layer = nn.Linear(2*M+2, 2,bias=False)

    def forward(self, x_window):
        """
        x_window: 当前窗口的记忆输入 [batch, (M+1)]
        """
        M = self.M
        K = self.K
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # threshold
        threshold = torch.from_numpy(self.threshold)
        threshold_matrix = threshold.unsqueeze(1).t().expand(M + 1, -1)  # (M+1,K) K维度延拓 -1 表示保持K维度不变
        threshold_matrix = threshold_matrix.to(device) # x_windows已移动到设备
        # x(n)
        xn = x_window[:,-1] #当前信号x(n) (16384,1)
        xn_amp = torch.abs(xn)  #当前信号|x(n)| (16384,1)
        xn_amp_matrix = xn_amp.unsqueeze(1).repeat(1, M + 1).unsqueeze(2).repeat(1, 1, K)  # 当前信号|x(n)| (16384,M+1,K)
        xn_matrix = xn.unsqueeze(1).repeat(1, M + 1).unsqueeze(2).repeat(1, 1, K)  # (16384,(M+1)*K)
        DVR_xn = xn_matrix.view(len(xn), -1)
        DVR_xn_amp = xn_amp_matrix.view(len(xn), -1)
        # x(n-i)
        X_tensor = x_window #记忆信号x(n-i) (16384,M+1)
        X_matrix = X_tensor.unsqueeze(2).repeat(1, 1, K)  # K维度延拓 x(n-i) (16384,M+1,K)
        DVR_X = X_matrix.view(len(X_matrix), -1)    # 展平 x(n-i) (16384,(M+1)*K)
        # |x(n-i)| DVR CORE
        X_amp = torch.abs(X_tensor)  #|x(n-i)| (16384,M+1)
        amp_matrix = X_amp.unsqueeze(2).repeat(1, 1, K)  # K维度延拓 |x(n-i)| (16384,M+1,K)
        DVR_amp_matrix = amp_matrix - threshold_matrix  # (16384,M+1,K)
        ABS_DVR_amp = torch.abs(DVR_amp_matrix)  # (16384,M+1,K)
        DVR_core = ABS_DVR_amp.view(len(ABS_DVR_amp), -1)    # (16384,(M+1)*K)
        # theta(n-i) DVR PHASE
        X_phase = torch.angle(X_tensor)  #theta(n-i) (16384,M+1)
        e_j_phase = torch.exp(1j * X_phase)  # (16384,M+1)
        e_j_phase_matrix = e_j_phase.unsqueeze(2).repeat(1, 1, K)  # (16384,M+1,K)
        # DVR_phase = torch.flatten(e_j_phase_matrix)  # (16384,(M+1)*K)
        DVR_phase = e_j_phase_matrix.view(len(e_j_phase_matrix), -1)
        # ABS(|x(n)|-beta) DDR CORE
        DDR_amp_matrix = xn_amp_matrix - threshold_matrix #(16384,M+1,K)
        ABS_DDR_amp = torch.abs(DDR_amp_matrix)     # (16384,M+1,K)
        DDR_core = ABS_DDR_amp.view(len(ABS_DDR_amp), -1) # (16384,(M+1)*K)

        # DVR basis
        DVR_out_linear = X_tensor  # (16384,M+1)
        DVR_out_1 = DVR_core * DVR_phase  # (16384,(M+1)*K)
        DVR_out_21 = DVR_core * DVR_phase * DVR_xn_amp  # (16384,(M+1)*K)
        DVR_out_22 = DVR_core * DVR_xn
        DVR_out_23 = DVR_core * DVR_X
        DVR_out_DDR_1 = DDR_core * DVR_X
        DVR_out_DDR_2 = DDR_core * DVR_xn * DVR_xn * torch.conj(DVR_X)

        DVR_out_full = torch.concatenate((DVR_out_linear, DVR_out_1, DVR_out_21, DVR_out_22, DVR_out_23, DVR_out_DDR_1,DVR_out_DDR_2), dim=1)
        DVR_out_full_real = torch.view_as_real(DVR_out_full)
        DVR = DVR_out_full_real.view(len(DVR_out_full_real), -1)

        y_pred = self.linear(DVR)
        # out = self.linear(DVR)
        # out_act = self.act(out)
        # y_pred = self.linear_2(out_act)

        return y_pred


    # 数据预处理函数
    def create_dataset(self,x, y, test_size=0.2):
        M = self.M
        # 转换为实部虚部分离格式
        # X = volterra_nn.complex_to_real(x)
        # Y = volterra_nn.complex_to_real(y)

        sequences = volterra_nn.create_memory_seq(x,M)  # [N, M+1]

        X_tensor = torch.from_numpy(sequences)
        # targets = volterra_nn.complex_to_real(y)    # [N, 1]
        targets = y

        Y_tensor = torch.from_numpy(y) # (N, 2)
        # Y_tensor = torch.view_as_complex(Y_tensor)
        X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=test_size, shuffle=False)
        return [X_train, X_val, Y_train, Y_val]

    def apply_dpd(self, signal):
        M = self.M
        threshold = self.threshold
        K = self.K
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 将模型移动到设备上
        self.to(device)
        self.eval()
        sequences = volterra_nn.create_memory_seq(signal, M)  # [N, M+1]
        x = torch.from_numpy(sequences).to(device)

        out = self(x)
        out_complex = torch.view_as_complex(out).cpu()
        y_pred = out_complex.detach().numpy()
        # y_pred = np.stack()
        return y_pred
    # def train(self):
    #     self.train()
    
# class DVR():
#     def __init__(self, M, threshold):
#         super(DVR, self).__init__()
#         self.name = 'DVR'
#         self.M = M
#         self.threshold = threshold
#         self.K = len(self.threshold)
#
#     def DVR_e(self, x, y, print_nmse=False):
#         M = self.M
#         thold = self.threshold
#         K = self.K
#
#         # 构建特征矩阵（与MATLAB相同的方式）
#         xi = x.reshape(-1, 1)
#         xip = np.angle(x).reshape(-1, 1)
#
#         X_lin = []
#         X_1 = []
#         X_21 = []
#         X_22 = []
#         X_23 = []
#         X_ddr_1 = []
#         X_ddr_2 = []
#
#         for m in range(M + 1):
#             xi_shift = np.roll(xi, m)  # 注意：与MATLAB的circshift方向一致
#             xip_shift = np.roll(xip, m)
#
#             X_lin.append(xi_shift)
#
#             for k in range(K):
#                 xi_core = np.abs(np.abs(xi_shift) - thold[k])
#
#                 xi_1_shift = xi_core * np.exp(1j * xip_shift)
#                 X_1.append(xi_1_shift)
#
#                 xi_21_shift = xi_core * np.exp(1j * xip_shift) * np.abs(xi)
#                 X_21.append(xi_21_shift)
#
#                 if m > 0:
#                     xi_22_shift = xi_core * xi
#                     X_22.append(xi_22_shift)
#
#                     xi_23_shift = xi_core * xi_shift
#                     X_23.append(xi_23_shift)
#
#                     xi_ddr_core = np.abs(np.abs(xi) - thold[k])
#                     xi_ddr_1 = xi_ddr_core * xi_shift
#                     X_ddr_1.append(xi_ddr_1)
#
#                     xi_ddr_2 = xi_ddr_core * xi * xi * np.conj(xi_shift)
#                     X_ddr_2.append(xi_ddr_2)
#
#         # 水平拼接所有特征（与MATLAB相同）
#         X = np.hstack([
#             np.hstack(X_lin) if X_lin else np.zeros((len(x), 0)),
#             np.hstack(X_1) if X_1 else np.zeros((len(x), 0)),
#             np.hstack(X_21) if X_21 else np.zeros((len(x), 0)),
#             np.hstack(X_22) if X_22 else np.zeros((len(x), 0)),
#             np.hstack(X_23) if X_23 else np.zeros((len(x), 0)),
#             np.hstack(X_ddr_1) if X_ddr_1 else np.zeros((len(x), 0)),
#             np.hstack(X_ddr_2) if X_ddr_2 else np.zeros((len(x), 0))
#         ])
#
#         # 设置起始索引
#         start = M + 1 + 10
#         X1 = X[start:, :]
#         y1 = y[start:].reshape(-1, 1)
#
#         # 使用与MATLAB相同的正则化参数和伪逆方法
#         I = np.eye(X1.shape[1])
#         coef = pinv(X1.conj().T @ X1 + 1e-1 * I) @ X1.conj().T @ y1
#
#         # 计算模型输出和NMSE
#         y_model = X @ coef
#         y_model = y_model.flatten()
#
#         # 应用与MATLAB相同的后处理
#         y_model[abs(y_model) > 1] = x[abs(y_model) > 1]
#
#         if print_nmse:
#             NMSE = cal.nmse(y[start:], y_model[start:])
#             print(f'NMSE-with-model = {NMSE:.6f} dB')
#
#         return coef.flatten()
#
#     def DVR_v(self, x, coef):
#         # 构建特征矩阵（与DVR_e相同）
#         M = self.M
#         thold = self.threshold
#         K = self.K
#
#         xi = x.reshape(-1, 1)
#         xip = np.angle(x).reshape(-1, 1)
#
#         X_lin = []
#         X_1 = []
#         X_21 = []
#         X_22 = []
#         X_23 = []
#         X_ddr_1 = []
#         X_ddr_2 = []
#
#         for m in range(M + 1):
#             xi_shift = np.roll(xi, m)
#             xip_shift = np.roll(xip, m)
#
#             X_lin.append(xi_shift)
#
#             for k in range(K):
#                 xi_core = np.abs(np.abs(xi_shift) - thold[k])
#
#                 xi_1_shift = xi_core * np.exp(1j * xip_shift)
#                 X_1.append(xi_1_shift)
#
#                 xi_21_shift = xi_core * np.exp(1j * xip_shift) * np.abs(xi)
#                 X_21.append(xi_21_shift)
#
#                 if m > 0:
#                     xi_22_shift = xi_core * xi
#                     X_22.append(xi_22_shift)
#
#                     xi_23_shift = xi_core * xi_shift
#                     X_23.append(xi_23_shift)
#
#                     xi_ddr_core = np.abs(np.abs(xi) - thold[k])
#                     xi_ddr_1 = xi_ddr_core * xi_shift
#                     X_ddr_1.append(xi_ddr_1)
#
#                     xi_ddr_2 = xi_ddr_core * xi * xi * np.conj(xi_shift)
#                     X_ddr_2.append(xi_ddr_2)
#
#         X = np.hstack([
#             np.hstack(X_lin) if X_lin else np.zeros((len(x), 0)),
#             np.hstack(X_1) if X_1 else np.zeros((len(x), 0)),
#             np.hstack(X_21) if X_21 else np.zeros((len(x), 0)),
#             np.hstack(X_22) if X_22 else np.zeros((len(x), 0)),
#             np.hstack(X_23) if X_23 else np.zeros((len(x), 0)),
#             np.hstack(X_ddr_1) if X_ddr_1 else np.zeros((len(x), 0)),
#             np.hstack(X_ddr_2) if X_ddr_2 else np.zeros((len(x), 0))
#         ])
#
#         # 使用系数进行预测
#         y = X @ coef.reshape(-1, 1)
#         y = y.flatten()
#
#         # 应用与MATLAB相同的后处理
#         y[abs(y) > 1] = x[abs(y) > 1]
#
#         return y
#
#     # def nmse(self, y_true, y_pred):
#     #     """计算NMSE（与MATLAB的nmse_max函数相同）"""
#     #     return 10 * np.log10(np.sum(np.abs(y_true - y_pred) ** 2) / np.sum(np.abs(y_true) ** 2))