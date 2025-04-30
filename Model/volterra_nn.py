import torch
import torch.nn as nn


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
        x_signal = x_signal.reshape(-1,1)
        # 计算预测值 y_pred = sum(coeffs * x_window_signals)
        y_pred = torch.sum(coeffs * x_signal, dim=1)  # [batch,]
        return [y_pred.real,y_pred.imag]


class Volterra_NN(nn.Module):
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