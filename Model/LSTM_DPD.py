import torch
import torch.nn as nn
import numpy as np

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

class LSTMDPD(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, output_size=2):
        super(LSTMDPD, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.5)  # 添加dropout层，p是丢弃概率

    def forward(self, x):
        # 添加序列维度（batch_size, sequence_length=10, input_size）
        # x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
        return out

    # 生成预失真信号
    def apply_dpd(self, signal, seq_length=10,M = 9):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将模型移动到设备上
        self.to(device)
        self.eval()

        with torch.no_grad():
            # 转换输入格式
            real = signal.real.reshape(-1, 1)
            imag = signal.imag.reshape(-1, 1)
            data = np.hstack((real, imag))  # (N, 2)

            # 创建输入序列（与训练时相同格式）
            # sequences = create_sequences(data, seq_length)[:, :-1, :]  # (N-seq_length+1, 9, 2)
            sequences = create_sequences_addmemory(data, seq_length, M=M)[:, :, :]  # (N-seq_length+1, 10, 20)
            # 预测
            with torch.no_grad():
                inputs = torch.FloatTensor(sequences).to(device)
                pred = self(inputs).cpu().numpy()  # (M, 2)

            # 重构完整信号（注意处理边界）
            full_pred = np.zeros((len(data), 2))
            full_pred[seq_length + M - 1:] = pred  # 对齐时间戳
        return full_pred[:, 0] + 1j * full_pred[:, 1]

class StreamingLSTMDPD(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 2)  # 预测实部和虚部

        # 初始化隐藏状态存储
        self.hidden = None

    def reset_hidden(self, batch_size=1):
        """重置隐藏状态"""
        device = next(self.parameters()).device
        self.hidden = (
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        )

    def forward(self, x):
        """
        输入格式:
        x - 当前时间步的输入 [batch_size, 1, 2] (实部+虚部)

        输出格式:
        pred - 当前时间步的预测 [batch_size, 2]
        """
        # LSTM前向传播
        out, self.hidden = self.lstm(x, self.hidden)  # out形状: [batch,1,hidden_size]

        # 全连接层
        pred = self.fc(out[:, -1, :])  # 形状: [batch,2]
        return pred


class ParallelLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x形状: [batch_size, seq_len, input_size]
        out, _ = self.lstm(x)  # [32, 512, 32]
        return self.fc(out)  # [32, 512, 2]