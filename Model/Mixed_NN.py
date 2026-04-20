import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import Model.volterra_nn as volterra_nn



class Parallel_NN(nn.Module):
    def __init__(self, layer_dims, L, M, activation="ReLU"):
        super().__init__()
        self.M = M
        self.layer_dim1 = layer_dims[0]
        self.layer_dim2 = layer_dims[1]
        self.L = L
        assert activation == "ReLU" or "Tanh" or "ELU" or "None"
        # self.main_layers = nn.Sequential()
        # self.para_layers = nn.Sequential()
        self.main_nn = volterra_nn.GMP_NN(self.layer_dim1, L, M = self.M,activation=activation)
        self.para_nn = volterra_nn.MCP_BASE_NN(self.layer_dim2, L, M = self.M, K = 5,activation=activation)
        # self.out_layer = nn.Linear(2*M+2, 2,bias=False)

    def forward(self, x_window, x_signal):
        """
        x_window: 当前窗口的实值输入 [batch, (M+1)]
        x_signal: 对应的复数信号窗口 [batch, M+1] (复数)
        """
        # 前向传播
        main_out = self.main_nn(x_window, x_signal)  # [batch, (M+1)]
        para_out = self.para_nn(x_window, x_signal)  # [batch, (M+1)]
        # phase_out = phase_x_window
        # 计算 e^{jθ}
        # complex_phase = torch.exp(1j * phase_out)
        # y_pred = torch.sum(amp_out * complex_phase, dim=1)  # [batch,]
        # return torch.view_as_real(y_pred)
        y_pred = main_out + para_out

        return y_pred

        # complex_phase = torch.exp(1j * phase_out)
        # complex_phase = torch.exp(1j * phase_x_window[])
        # coeffs = torch.view_as_complex(
        #     out.view(-1, self.M + 1, 2)  # [batch, M+1, 2]
        # )  # [batch, M+1] (复数)
        # x_signal = x_signal.reshape(-1,1)
        # 计算预测值 y_pred = sum(coeffs * x_window_signals)
        # y_pred = torch.sum(amp_out * complex_phase, dim=1)  # [batch,]
        # a = torch.view_as_real(y_pred)


    # 数据预处理函数
    def create_dataset(self,x, y, L, M, test_size=0.2):
        # 转换为实部虚部分离格式
        # x_real = torch.view_as_real(x).float()  # [N, 2]
        # y_real = torch.view_as_real(y).float()  # [N, 2]
        X = volterra_nn.complex_to_real(x)
        Y = volterra_nn.complex_to_real(y)

        Lb = self.L[0]
        Lc = self.L[1]

        # 创建延迟窗口
        sequences = []
        targets = []
        for i in range(self.M + Lb, len(x) - Lc -2):
            # 输入：x(n-M)到x(n)的实部虚部
            window = X[i - self.M - Lb : i + Lc + 1].flatten()  # [2*(M+1+Lb+Lc),]
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
        return [X_train, X_val, Y_train, Y_val]

    def apply_dpd(self, signal):
        M = self.M
        Lb = self.L[0]
        Lc = self.L[1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 将模型移动到设备上
        self.to(device)
        self.eval()
        X = volterra_nn.complex_to_real(signal)  # (16384,2)
        # 创建延迟窗口
        sequences = []
        for i in range(M + Lb, len(signal) - Lc - 2):
            # 输入：x(n-M)到x(n)的实部虚部
            window = X[i - M - Lb : i + Lc + 1].flatten()  # [2*(M+1),]
            # 输出：y(n+1)的实部虚部
            sequences.append(window)

        X_tensor = torch.FloatTensor(sequences)  # (16375, 2M+2)

        reshaped_tensor = X_tensor.view(len(X_tensor), M + 1 + Lb + Lc, 2)
        input_tensor = torch.view_as_complex(reshaped_tensor)
        x_tensor = reshaped_tensor[:, Lb:M + Lb + 1, :]
        batch_x_signal = torch.view_as_complex(x_tensor)
        # batch_x_signal = torch.view_as_complex(reshaped_tensor)
        batch_x_magnitude = abs(input_tensor)
        with torch.no_grad():
            # inputs = X_tensor.to(device)
            batch_x_magnitude_1 = abs(input_tensor)
            # batch_x_magnitude_3 = abs(input_tensor) ** 3
            # batch_x_magnitude_5 = abs(input_tensor) ** 5
            # batch_x_magnitude = torch.cat((batch_x_magnitude_1, batch_x_magnitude_3, batch_x_magnitude_5), dim=1)
            # batch_x_magnitude = batch_x_magnitude.to(device)
            batch_x_magnitude = batch_x_magnitude_1.to(device)
            batch_x_signal = batch_x_signal.to(device)
            outputs = self(batch_x_magnitude, batch_x_signal)

            ypred = torch.view_as_complex(outputs).cpu()
        full_pred = np.complex128(np.zeros((len(signal))))
        full_pred[M + Lb : -Lc-2] = np.stack(ypred[0:])  # 对齐时间戳
        return full_pred
    # def train(self):
    #     self.train()
