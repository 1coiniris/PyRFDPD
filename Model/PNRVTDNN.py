import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import Model.volterra_nn as volterra_nn
from function import Function_Lib as fun

class PNRVTDNN(nn.Module):
    def __init__(self, layer_dims, M, K=1, activation="ReLU"):
        super().__init__()
        # self.total_train = 1
        self.name = 'PNRVTDNN'
        self.M = M
        self.K = K
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert activation == "ReLU" or "Tanh" or "GELU" or "None"
        self.activation = activation
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

    def forward(self, x_window):
        """
        x_window: 当前窗口的记忆输入 [batch, (M+1)]
        """

        # 相位归一化
        X_norm, r, A = self.phase_normalization(x_window)
        # X_norm_real = torch.view_as_real(X_norm).view(len(X_norm), -1)

        # 2. 添加包络及其幂次项
        envelope_features = A
        for order in range(2,self.K+1):
            envelope_features = torch.concat((envelope_features, A ** order), dim=1)
            # envelope_features.append(A ** order)

        # X = x_window
        # for i in range(2,self.K+1):
        #     X_k = x_window.pow(i)
        #     X = torch.concat((X,X_k), dim=1)
        x_real = X_norm.real
        x_imag = X_norm.imag
        # 剪枝当前时刻的虚部 (因为归一化后它为0)
        x_imag = x_imag[:, :-1]  # 去掉最后一列（当前时刻）
        # x_real = x_real.view(len(x_real), -1)
        x = torch.cat((x_real, x_imag,envelope_features), dim=1)
        y_norm = self.layers(x)

        # 提取I和Q分量
        Y_I = y_norm[:, 0]
        Y_Q = y_norm[:, 1]

        # 相位反归一化 - 关键步骤！
        y_hat = torch.conj(r) * (Y_I + 1j * Y_Q)

        y_pred = torch.view_as_real(y_hat).view(len(y_hat), -1)
        return y_pred


    def model_train(self,x,y,model_path,logger=None,para = [0.001,150,512]):
        learning_rate = para[0]
        epochs = para[1]
        batch_size = para[2]
        device = self.device

        # optim
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        # fun.model_structure(self, logger)
        best_metric = float('inf')

        # dataset
        X_train, X_val, Y_train, Y_val = self.create_dataset(x, y)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = TensorDataset(X_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # train
        logger.info(f'------------------------Train Stage-------------------------------')
        train_loss_list = []
        val_loss_list = []
        # 训练循环
        start_time = time.time()  # 记录开始时间
        nosave_count = 0
        for epoch in range(epochs):
            self.train()
            i = 0
            train_loss = 0
            for inputs, targets in tqdm(train_loader):
                # 获取对应的复数信号窗口 [batch, M+1]
                optimizer.zero_grad()

                batch_x_signal = inputs.to(device)
                targets = torch.view_as_real(targets)
                targets = targets.to(device)
                # outputs = model(inputs)
                outputs = self(batch_x_signal)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader):
                    # 将数据移动到设备上
                    # 获取对应的复数信号窗口 [batch, M+1]
                    batch_x_signal = inputs.to(device)
                    targets = torch.view_as_real(targets)
                    targets = targets.to(device)
                    # outputs = model(inputs)
                    val_outputs = self(batch_x_signal)
                    val_loss += criterion(val_outputs, targets).item()

            train_loss_list.append(train_loss / len(train_loader))
            val_loss_list.append(val_loss / len(val_loader))
            # 记录指标
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }

            # # 保存当前模型（按周期命名）
            # torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
            save = 0
            # 更新最佳模型
            if val_loss_list[-1] < best_metric:
                best_metric = val_loss_list[-1]
                save = 1
                # 保存模型参数（推荐保存为 .pt 或 .pth 文件）
                best_model = self.state_dict()
            # print(f'Epoch {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f} | save:{save}')
            logger.info(
                f'Epoch {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.7f} | Val Loss: {val_loss / len(val_loader):.7f} | save:{save}')
            if save == 0:
                nosave_count = nosave_count + 1
            else:
                nosave_count = 0
            if nosave_count > 100:
                break

        torch.save(best_model, model_path)
        logger.info(f"model save: {model_path} ")
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model train time: {elapsed_time:.6f} s")


    # 数据预处理函数
    def create_dataset(self,x, y, test_size=0.2):
        M = self.M
        # 转换为实部虚部分离格式

        sequences = fun.create_memory_seq(x,M)  # [N, M+1]

        X_tensor = torch.from_numpy(sequences)
        # targets = volterra_nn.complex_to_real(y)    # [N, 1]
        targets = y

        Y_tensor = torch.from_numpy(y) # (N, 2)
        # Y_tensor = torch.view_as_complex(Y_tensor)
        X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=test_size, shuffle=False)
        return [X_train, X_val, Y_train, Y_val]

    def apply_dpd(self, signal, coef=None):
        M = self.M
        # threshold = self.threshold
        # K = self.K
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 将模型移动到设备上
        self.to(device)
        self.eval()
        sequences = fun.create_memory_seq(signal, M)  # [N, M+1]
        x = torch.from_numpy(sequences).to(device)
        out = self(x)
        out_complex = torch.view_as_complex(out).cpu()
        y_pred = out_complex.detach().numpy()

        return y_pred
    # def train(self):
    #     self.train()

    def phase_normalization(self, z):
        """
        相位归一化处理

        参数:
            z: 复数输入张量，形状为(batch_size, M+1)

        返回:
            X_norm: 归一化后的复数张量
            r: 归一化因子
            A: 包络张量
        """
        # 计算当前时刻的归一化因子 r(k) = z*(k)/|z(k)|
        z_current = z[:, -1]  # 当前时刻样本
        r = torch.conj(z_current) / (torch.abs(z_current) + 1e-10)

        # 对整个记忆窗口应用相位归一化
        X_norm = r.unsqueeze(1) * z

        # 计算包络向量 A(k) = [|z(k)|, |z(k-1)|, ..., |z(k-M)|]
        A = torch.abs(z)

        return X_norm, r, A
