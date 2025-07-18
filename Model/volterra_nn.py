import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

# 数据预处理：将复数转换为实部+虚部
def complex_to_real(x):
    return np.stack((x.real, x.imag), axis=1)

def create_sequences(data, seq_length):
    """将数据转换为序列格式"""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

def create_memory_seq(data, M):
    """将数据转换为记忆序列格式"""
    L = len(data)
    data_new = np.concatenate((data[ L-M : L],data),axis=0)
    sequences = []
    for i in range(M,len(data_new)):
        sequences.append(data_new[i-M:i+1])
        a = data_new[i-M:i+1]
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


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation="ReLU"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = self._get_activation(activation)
        self.shortcut = nn.Identity()

        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim, bias=False)

    def _get_activation(self, name):
        activations = {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "ELU": nn.ELU(alpha=1),
            "None": nn.Identity()
        }
        return activations.get(name, nn.Identity())

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.linear(x)
        out = self.activation(out)
        out = out + identity  # 关键修改：避免 inplace 操作！
        return out

class DVC_NN(nn.Module):
    def __init__(self, layer_dims,phase_dims,L, M, activation="ReLU"):
        super().__init__()
        self.M = M
        self.L = L
        assert activation == "ReLU" or "Tanh" or "ELU" or "None"
        self.layers = nn.Sequential()
        self.phase_layers = nn.Sequential()
        # self.dropout = nn.Dropout(0.5)
        # 幅度网络
        for index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            # self.layers.add_module(
            #     f"res_{index}",
            #     ResidualBlock(in_dim, out_dim, activation)
            # )
            self.layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim))
            if activation == "ReLU":
                self.layers.add_module("actFunc " + str(index), nn.ReLU())
            elif activation == "Tanh":
                self.layers.add_module("actFunc " + str(index), nn.Tanh())
            elif activation == "GELU":
                self.layers.add_module("actFunc " + str(index), nn.GELU())
            elif activation == "None":
                pass
            # self.layers.add_module("dropout" + str(index), nn.Dropout(0.2))
        if activation != "None":
            self.layers = self.layers[:-1]  # remove the last activation layer
        # 相位网络
        for index, (in_dim, out_dim) in enumerate(zip(phase_dims[:-1], phase_dims[1:])):
            # self.phase_layers.add_module(
            #     f"phase_res_{index}",
            #     ResidualBlock(in_dim, out_dim, activation="None")  # 相位网络默认不激活
            # )
            self.phase_layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim))
            # self.phase_layers.add_module("dropout" + str(index), nn.Dropout(0.2))
            # self.phase_layers.add_module("actFunc " + str(index), nn.ReLU())
        # self.phase_layers = self.phase_layers[:-1]
        self.out_layer = nn.Linear(2*M+2, 2,bias=False)

    def forward(self, amp_x_window,phase_x_window):
        """
        amp_x_window: 当前窗口的幅度输入 [batch, (M+1)]
        phase_x_window: 当前窗口的相位输入 [batch, (M+1)]
        """
        # 前向传播
        amp_out = self.layers(amp_x_window)  # [batch, (M+1)]
        phase_out = self.phase_layers(phase_x_window)
        # phase_out = phase_x_window
        # 计算 e^{jθ}
        # complex_phase = torch.exp(1j * phase_out)
        # y_pred = torch.sum(amp_out * complex_phase, dim=1)  # [batch,]
        # return torch.view_as_real(y_pred)
        cos_phase = torch.cos(phase_out)
        sin_phase = torch.sin(phase_out)

        i_m = cos_phase*amp_out
        q_m = sin_phase*amp_out

        m_out = torch.concat([i_m, q_m],dim=1)

        y_pred = self.out_layer(m_out)
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
        X = complex_to_real(x)
        Y = complex_to_real(y)

        Lb = self.L[0]
        Lc = self.L[1]

        # 创建延迟窗口
        sequences = []
        targets = []
        for i in range(M + Lb, len(x) - Lc -2):
            # 输入：x(n-M)到x(n)的实部虚部
            window = X[i - M - Lb : i + Lc + 1].flatten()  # [2*(M+1+Lb+Lc),]
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
        X = complex_to_real(signal)  # (16384,2)
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
        batch_x_magnitude = torch.abs(input_tensor)
        batch_x_phase = torch.angle(input_tensor)
        with torch.no_grad():
            # inputs = X_tensor.to(device)
            batch_x_magnitude = batch_x_magnitude.to(device)
            batch_x_phase = batch_x_phase.to(device)
            outputs = self(batch_x_magnitude, batch_x_phase)

            ypred = torch.view_as_complex(outputs).cpu()
        full_pred = np.complex128(np.zeros((len(signal))))
        full_pred[M + Lb : -Lc-2] = np.stack(ypred[0:])  # 对齐时间戳
        return full_pred

class GMP_NN_2(nn.Module):
    def __init__(self, layer_dims,L, M, activation="ReLU"):
        super().__init__()
        self.M = M
        self.L = L
        assert activation == "ReLU" or "Tanh" or "ELU" or "None"
        self.layers = nn.Sequential()
        for index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim))
            if activation == "ReLU":
                self.layers.add_module("actFunc " + str(index), nn.ReLU())
            elif activation == "Tanh":
                self.layers.add_module("actFunc " + str(index), nn.Tanh())
            elif activation == "GELU":
                self.layers.add_module("actFunc " + str(index), nn.GELU())
            elif activation == "None":
                pass
        if activation != "None":
            self.layers = self.layers[:-1]  # remove the last activation layer
        self.out_layer = nn.Linear(2 * M + 2, 2, bias=False)
    # def __init__(self, input_size, hidden_size, M):
    #     super().__init__()
    #     self.M = M
    #     self.fc = nn.Sequential(
    #         nn.Linear(input_size, hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(hidden_size, hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(hidden_size, 2 * (M + 1))  # 输出实部虚部分离
    #     )

    def forward(self, x_window, x_signal):
        """
        x_window: 当前窗口的实值输入 [batch, (M+1)]
        x_signal: 对应的复数信号窗口 [batch, M+1] (复数)
        """
        # 前向传播
        out = self.layers(x_window)  # [batch, (M+1)]

        # 重组复数系数
        coeffs = out
        # coeffs = torch.view_as_complex(
        #     out.view(-1, self.M + 1, 2)  # [batch, M+1, 2]
        # )  # [batch, M+1] (复数)
        # x_signal = x_signal.reshape(-1,1)
        # 计算预测值 y_pred = sum(coeffs * x_window_signals)
        I = torch.real(coeffs*x_signal)
        Q = torch.imag(coeffs*x_signal)
        # i_m = cos_phase*amp_out
        # q_m = sin_phase*amp_out

        m_out = torch.concat([I, Q],dim=1)
        y_pred = self.out_layer(m_out)
        # y_pred = torch.sum(coeffs * x_signal, dim=1)  # [batch,]
        # a = torch.view_as_real(y_pred)
        return y_pred   #torch.view_as_real(y_pred)

    # 数据预处理函数
    def create_dataset(self,x, y, L, M, test_size=0.2):
        # 转换为实部虚部分离格式
        # x_real = torch.view_as_real(x).float()  # [N, 2]
        # y_real = torch.view_as_real(y).float()  # [N, 2]
        X = complex_to_real(x)
        Y = complex_to_real(y)

        Lb = self.L[0]
        Lc = self.L[1]

        # 创建延迟窗口
        sequences = []
        targets = []
        for i in range(M + Lb, len(x) - Lc -2):
            # 输入：x(n-M)到x(n)的实部虚部
            window = X[i - M - Lb : i + Lc + 1].flatten()  # [2*(M+1+Lb+Lc),]
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
        X = complex_to_real(signal)  # (16384,2)
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

class GMP_NN(nn.Module):
    def __init__(self, layer_dims,L, M, activation="ReLU"):
        super().__init__()
        self.M = M
        self.L = L
        assert activation == "ReLU" or "Tanh" or "ELU" or "None"
        self.layers = nn.Sequential()
        for index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim))
            if activation == "ReLU":
                self.layers.add_module("actFunc " + str(index), nn.ReLU())
            elif activation == "Tanh":
                self.layers.add_module("actFunc " + str(index), nn.Tanh())
            elif activation == "GELU":
                self.layers.add_module("actFunc " + str(index), nn.GELU())
            elif activation == "None":
                pass

        if activation != "None":
            self.layers = self.layers[:-1]  # remove the last activation layer

    # def __init__(self, input_size, hidden_size, M):
    #     super().__init__()
    #     self.M = M
    #     self.fc = nn.Sequential(
    #         nn.Linear(input_size, hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(hidden_size, hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(hidden_size, 2 * (M + 1))  # 输出实部虚部分离
    #     )

    def forward(self, x_window, x_signal):
        """
        x_window: 当前窗口的实值输入 [batch, (M+1)]
        x_signal: 对应的复数信号窗口 [batch, M+1] (复数)
        """
        # 前向传播
        out = self.layers(x_window)  # [batch, 2*(M+1)]

        # 重组复数系数
        coeffs = torch.view_as_complex(
            out.view(-1, self.M + 1, 2)  # [batch, M+1, 2]
        )  # [batch, M+1] (复数)
        # x_signal = x_signal.reshape(-1,1)
        # 计算预测值 y_pred = sum(coeffs * x_window_signals)
        y_pred = torch.sum(coeffs * x_signal, dim=1)  # [batch,]
        # a = torch.view_as_real(y_pred)
        return torch.view_as_real(y_pred)

    # 数据预处理函数
    def create_dataset(self,x, y, L, M, test_size=0.2):
        # 转换为实部虚部分离格式
        # x_real = torch.view_as_real(x).float()  # [N, 2]
        # y_real = torch.view_as_real(y).float()  # [N, 2]
        X = complex_to_real(x)
        Y = complex_to_real(y)

        Lb = self.L[0]
        Lc = self.L[1]

        # 创建延迟窗口
        sequences = []
        targets = []
        for i in range(M + Lb, len(x) - Lc -2):
            # 输入：x(n-M)到x(n)的实部虚部
            window = X[i - M - Lb : i + Lc + 1].flatten()  # [2*(M+1+Lb+Lc),]
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
        X = complex_to_real(signal)  # (16384,2)
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


class MCP_BASE_NN(nn.Module):
    def __init__(self, layer_dims,L, M,K, activation="ReLU"):
        super().__init__()
        self.M = M
        self.L = L
        self.K = K
        assert activation == "ReLU" or "Tanh" or "ELU" or "None"

        self.layers = nn.Sequential()
        for index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            if index == 0:
                self.poly_layer = nn.Linear(in_dim, out_dim, bias=False)
            else:
                self.layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim))
                if activation == "ReLU":
                    self.layers.add_module("actFunc " + str(index), nn.ReLU())
                elif activation == "Tanh":
                    self.layers.add_module("actFunc " + str(index), nn.Tanh())
                elif activation == "ELU":
                    self.layers.add_module("actFunc " + str(index), nn.ELU(alpha=1))
                elif activation == "None":
                    pass
        if activation != "None":
            self.layers = self.layers[:-1]  # remove the last activation layer

    # def __init__(self, input_size, hidden_size, M):
    #     super().__init__()
    #     self.M = M
    #     self.fc = nn.Sequential(
    #         nn.Linear(input_size, hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(hidden_size, hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(hidden_size, 2 * (M + 1))  # 输出实部虚部分离
    #     )

    def forward(self, x_window, x_signal):
        """
        x_window: 当前窗口的实值输入 [batch, (M+1)]
        x_signal: 对应的复数信号窗口 [batch, M+1] (复数)
        """
        # 前向传播
        poly = self.poly_layer(x_window) #[batch, K]
        nonlinear = []
        for k in range(self.K):
            x_k = poly[:, 2*k]**(k+1)
            x_k_2 = poly[:, 2*k+1 ] ** (k+1)
            # nonlinear = torch.cat(x_k, dim=1)
            nonlinear.append(x_k.unsqueeze(1))
            nonlinear.append(x_k_2.unsqueeze(1))
            # nonlinear.append(poly[:, k]**k)
        nonlinear_layer = torch.cat(nonlinear[0:], dim=1)
        # for x_k in nonlinear:
        #     nonlinear_layer = torch.cat(x_k, dim=1)
        out = self.layers(nonlinear_layer)  # [batch, 2*(M+1)]

        # 重组复数系数
        coeffs = torch.view_as_complex(
            out.view(-1, self.M + 1, 2)  # [batch, M+1, 2]
        )  # [batch, M+1] (复数)
        # x_signal = x_signal.reshape(-1,1)
        # 计算预测值 y_pred = sum(coeffs * x_window_signals)
        y_pred = torch.sum(coeffs * x_signal, dim=1)  # [batch,]
        # a = torch.view_as_real(y_pred)
        return torch.view_as_real(y_pred)

    # 数据预处理函数
    def create_dataset(self,x, y, L, M, test_size=0.2):
        # 转换为实部虚部分离格式
        # x_real = torch.view_as_real(x).float()  # [N, 2]
        # y_real = torch.view_as_real(y).float()  # [N, 2]
        X = complex_to_real(x)
        Y = complex_to_real(y)

        Lb = self.L[0]
        Lc = self.L[1]

        # 创建延迟窗口
        sequences = []
        targets = []
        for i in range(M + Lb, len(x) - Lc -2):
            # 输入：x(n-M)到x(n)的实部虚部
            window = X[i - M - Lb : i + Lc + 1].flatten()  # [2*(M+1+Lb+Lc),]
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
        X = complex_to_real(signal)  # (16384,2)
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




class MCP_NN(nn.Module):
    def __init__(self, layer_dims, M, activation="ReLU"):
        super().__init__()
        self.M = M
        assert activation == "ReLU" or "Tanh" or "ELU" or "None"
        self.layers = nn.Sequential()
        for index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim))
            if activation == "ReLU":
                self.layers.add_module("actFunc " + str(index), nn.ReLU())
            elif activation == "Tanh":
                self.layers.add_module("actFunc " + str(index), nn.Tanh())
            elif activation == "ELU":
                self.layers.add_module("actFunc " + str(index), nn.ELU(alpha=1))
            elif activation == "None":
                pass
        if activation != "None":
            self.layers = self.layers[:-1]  # remove the last activation layer

    # def __init__(self, input_size, hidden_size, M):
    #     super().__init__()
    #     self.M = M
    #     self.fc = nn.Sequential(
    #         nn.Linear(input_size, hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(hidden_size, hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(hidden_size, 2 * (M + 1))  # 输出实部虚部分离
    #     )

    def forward(self, x_window, x_signal):
        """
        x_window: 当前窗口的实值输入 [batch, 2*(M+1)]
        x_signal: 对应的复数信号窗口 [batch, M+1] (复数)
        """
        # 前向传播
        out = self.layers(x_window)  # [batch, 2*(M+1)]

        # 重组复数系数
        coeffs = torch.view_as_complex(
            out.view(-1, self.M + 1, 2)  # [batch, M+1, 2]
        )  # [batch, M+1] (复数)
        # x_signal = x_signal.reshape(-1,1)
        # 计算预测值 y_pred = sum(coeffs * x_window_signals)
        y_pred = torch.sum(coeffs * x_signal, dim=1)  # [batch,]
        # a = torch.view_as_real(y_pred)
        return torch.view_as_real(y_pred)

    def apply_dpd(self, signal, M=9):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 将模型移动到设备上
        self.to(device)
        self.eval()
        X = complex_to_real(signal)  # (16384,2)
        # 创建延迟窗口
        sequences = []
        for i in range(M, len(signal) - 1):
            # 输入：x(n-M)到x(n)的实部虚部
            window = X[i - M:i + 1].flatten()  # [2*(M+1),]
            # 输出：y(n+1)的实部虚部
            sequences.append(window)

        X_tensor = torch.FloatTensor(sequences)  # (16375, 2M+2)

        reshaped_tensor = X_tensor.view(len(X_tensor), M + 1, 2)
        batch_x_signal = torch.view_as_complex(reshaped_tensor)

        with torch.no_grad():
            inputs = X_tensor.to(device)
            batch_x_signal = batch_x_signal.to(device)
            outputs = self(inputs, batch_x_signal)

            ypred = torch.view_as_complex(outputs).cpu()
        full_pred = np.complex128(np.zeros((len(signal))))
        full_pred[M:-1] = np.stack(ypred[0:])  # 对齐时间戳
        return full_pred

class RVTD_NN(nn.Module):
    r"""
    The real-valued time-delay neural network implementation in PyTorch.

    Reference:
    [Dynamic Behavioral Modeling of 3G Power Amplifiers Using Real-Valued Time-Delay Neural Networks](http://ieeexplore.ieee.org/document/1273746/)

    Parameters:
    - layer_dims: The network structure, e.g. [6, 32, 32, 2]
    - activation: The activation function, e.g. "ReLU", "Tanh", "ELU" or "None"
    """

    def __init__(self, layer_dims, activation="ReLU"):
        super().__init__()
        assert activation == "ReLU" or "Tanh" or "ELU" or "None"
        self.layers = nn.Sequential()
        for index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim))
            if activation == "ReLU":
                self.layers.add_module("actFunc " + str(index), nn.ReLU())
            elif activation == "Tanh":
                self.layers.add_module("actFunc " + str(index), nn.Tanh())
            elif activation == "ELU":
                self.layers.add_module("actFunc " + str(index), nn.ELU(alpha=1))
            elif activation == "None":
                pass
        if activation != "None":
            self.layers = self.layers[:-1]  # remove the last activation layer

    def forward(self, x):
        return self.layers(x)

    def apply_dpd(self, signal, M=9):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 将模型移动到设备上
        self.to(device)
        self.eval()
        X = complex_to_real(signal)  # (16384,2)
        # 创建延迟窗口
        sequences = []
        for i in range(M, len(signal) - 1):
            # 输入：x(n-M)到x(n)的实部虚部
            window = X[i - M:i + 1].flatten()  # [2*(M+1),]
            # 输出：y(n+1)的实部虚部
            sequences.append(window)

        X_tensor = torch.FloatTensor(sequences)  # (16375, 2M+2)
        with torch.no_grad():
            inputs = X_tensor.to(device)
            outputs = self(inputs)
            ypred = torch.view_as_complex(outputs).cpu()
        full_pred = np.complex128(np.zeros((len(signal)))).squeeze()
        full_pred[M:-1] = np.stack(ypred[0:])  # 对齐时间戳
        return full_pred

class MCP_LSTM(nn.Module):
    def __init__(self,hidden_size=64, num_layers=1, M=7):
        input_size = 2*(M+1)
        output_size = 2*(M+1)
        self.M = M
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.5)  # 添加dropout层，p是丢弃概率

    def forward(self, x_window, x_signal):
        """
        x_window: 当前窗口的实值输入 [batch, 2*(M+1)]
        x_signal: 对应的复数信号窗口 [batch, M+1] (复数)
        """
        # 前向传播
        # out = self.layers(x_window)  # [batch, 2*(M+1)]
        out, _ = self.lstm(x_window)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出

        # 重组复数系数
        coeffs = torch.view_as_complex(
            out.view(-1, self.M + 1, 2)  # [batch, M+1, 2]
        )  # [batch, M+1] (复数)
        x_signal = x_signal.reshape(-1,1)
        # 计算预测值 y_pred = sum(coeffs * x_window_signals)
        y_pred = torch.sum(coeffs * x_signal, dim=1)  # [batch,]
        # a = torch.view_as_real(y_pred)
        return torch.view_as_real(y_pred)

    # def forward(self, x):
    #     # 添加序列维度（batch_size, sequence_length=10, input_size）
    #     # x = x.unsqueeze(1)
    #     out, _ = self.lstm(x)
    #     out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
    #
    #     return out

    # 数据预处理函数
    def create_dataset(self, x, y, M, seq_length=1, test_size=0.2):
        # 转换为实部虚部分离格式
        # x_real = torch.view_as_real(x).float()  # [N, 2]
        # y_real = torch.view_as_real(y).float()  # [N, 2]
        X = complex_to_real(x)  # 输入：PA输出信号
        Y = complex_to_real(y)  # 目标：原始信号
        # seq_length = 1
        # # 创建序列数据 (seq_length=10)
        # seq_length = 10
        # X_seq = create_sequences(X, seq_length)  # 形状 (10000-M, seq_len, 2M+2)
        X_seq = create_sequences_addmemory(X, seq_length, M=M)
        # Y_seq = create_sequences(X,seq_length)# 形状 (10000-9, 10, 2)
        Y_seq = create_sequences_addmemory(Y, seq_length, M=M)

        X_data = X_seq[:, :, :]  # 输入序列：10个时间步 (10000-9, 10, 2)
        Y_data = Y_seq[:, -1, -2:]  # 目标值：第10个时间步 (10000-9, 2)

        # ====================== 转换为PyTorch张量 ======================
        X_tensor = torch.FloatTensor(X_data)  # (16375, 10, 2)
        Y_tensor = torch.FloatTensor(Y_data)  # (16375, 2)

        # 划分数据集（保持时序顺序）
        X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=0.2, shuffle=False)

        return [X_train, X_val, Y_train, Y_val]

    # 生成预失真信号
    def apply_dpd(self, signal, seq_length=10, M=9):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将模型移动到设备上
        self.to(device)
        self.eval()
        # 转换输入格式
        data = complex_to_real(signal)  # (16384,2)

        # 创建输入序列（与训练时相同格式）
        sequences = create_sequences_addmemory(data, seq_length, M=M)[:, :, :]  # (N-seq_length+1, seq_len, 2M+2)
        X_tensor = torch.FloatTensor(sequences).to(device)

        x_signal = torch.stack([
            window[-1][-2:] for window in X_tensor
        ])[:X_tensor.shape[0]]  # 需优化以提高效率
        # 转换为张量
        batch_x_signal = torch.view_as_complex(x_signal)

        with torch.no_grad():
            inputs = X_tensor.to(device)
            batch_x_signal = batch_x_signal.to(device)
            outputs = self(inputs, batch_x_signal)
            ypred = torch.view_as_complex(outputs).cpu()
        full_pred = np.complex128(np.zeros((len(signal))))
        full_pred[seq_length + M - 1:] = np.stack(ypred[0:])  # 对齐时间戳
        return full_pred

        # with torch.no_grad():
        #     # 预测
        #     pred = self(inputs).cpu().numpy()  # (M, 2)
        #     # 重构完整信号（注意处理边界）
        #     full_pred = np.zeros((len(data), 2))
        #     full_pred[seq_length + M - 1:] = pred  # 对齐时间戳
        #
        # return full_pred[:, 0] + 1j * full_pred[:, 1]

class NeuralBasisDPD(nn.Module):
    def __init__(self, memory_depth=3, hidden_dim=32):
        super().__init__()
        self.memory_depth = memory_depth
        # 基函数生成网络（每个时延独立）
        self.basis_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden_dim),  # 输入：x(n-m)的实部和虚部
                nn.ReLU(),
                nn.Linear(hidden_dim, 8)   # 输出8个基函数
            ) for _ in range(memory_depth + 1)
        ])
        # 非线性组合网络
        self.combiner = nn.Sequential(
            nn.Linear(8 * (memory_depth + 1), 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出预失真信号
        )

    def forward(self, x_history):
        # x_history形状：(batch_size, memory_depth+1, 2) [实部+虚部]
        batch_size, seq_len, _ = x_history.shape
        bases = []
        for m in range(seq_len):
            x_m = x_history[:, m, :]
            basis_m = self.basis_nets[m](x_m)  # 生成8个基函数
            bases.append(basis_m)
        bases = torch.stack(bases, dim=1)      # (batch_size, seq_len, 8)
        bases = bases.view(batch_size, -1)     # 展平
        y = self.combiner(bases)
        return y