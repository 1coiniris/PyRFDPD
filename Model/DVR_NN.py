import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time
from function import Function_Calculate as cal, Function_Lib as fun
from tqdm import tqdm
from matplotlib import pyplot as plt
import Model.volterra_nn as volterra_nn


class DVR_NN(nn.Module):
    def __init__(self, layer_dims,K, M, activation="ReLU",term = 6):
        super().__init__()
        # self.total_train = 1
        self.M = M
        self.K = K
        self.term = term
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.threshold = threshold
        assert activation == "ReLU" or "Tanh" or "GELU" or "Sigmoid" or "None"
        self.activation = activation
        self.DVR_layers = nn.Sequential()
        # self.DDR_layers = nn.Sequential()
        # layer_dims = [M+1,(M+1),(M+1),(M+1)*K]
        for index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            # self.layers.add_module(
            #     f"res_{index}",
            #     ResidualBlock(in_dim, out_dim, activation)
            # )
            self.DVR_layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim).double())
            # self.DDR_layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim).double())
            if activation == "ReLU":
                self.DVR_layers.add_module("actFunc " + str(index), nn.ReLU())
                # self.DDR_layers.add_module("actFunc " + str(index), nn.ReLU())
            elif activation == "Tanh":
                self.DVR_layers.add_module("actFunc " + str(index), nn.Tanh())
                # self.DDR_layers.add_module("actFunc " + str(index), nn.Tanh())
            elif activation == "Sigmoid":
                self.DVR_layers.add_module("actFunc " + str(index), nn.Sigmoid())
            elif activation == "GELU":
                self.DVR_layers.add_module("actFunc " + str(index), nn.GELU())
                # self.DDR_layers.add_module("actFunc " + str(index), nn.GELU())
            # self.layers.add_module("dropout" + str(index), nn.Dropout(0.2))
        self.DVR_layers = self.DVR_layers[:-1]  # remove the last activation layer
        self.linear = nn.Linear((K*(M+1)*term+M+1)*2, 2,bias=False).double() #+M+1
        if self.activation != "ABS":
            if self.activation == "ReLU":
                self.act = nn.ReLU()
            elif self.activation == "Tanh":
                self.act = nn.Tanh()
            elif self.activation == "GELU":
                self.act = nn.GELU()

        # self.linear_2 = nn.Linear(12, 2, bias=False).double()
        # self.main_layers = nn.Sequential()
        # self.para_layers = nn.Sequential()
        # self.main_nn = volterra_nn.GMP_NN(self.layer_dim1, L, M = self.M,activation=activation)
        # self.para_nn = volterra_nn.MCP_BASE_NN(self.layer_dim2, L, M = self.M, K = 5,activation=activation)
        # self.out_layer = nn.Linear(2*M+2, 2,bias=False)

    def forward(self, x_window,coef = None):
        """
        x_window: 当前窗口的记忆输入 [batch, (M+1)]
        """
        if coef is None:
            DVR_out_full = self.get_basis(x_window)
            DVR_out_full_real = torch.view_as_real(DVR_out_full)
            DVR = DVR_out_full_real.view(len(DVR_out_full_real), -1)

            y_pred = self.linear(DVR)
        else:
            y = self.DVR_NN_v(x_window, coef)
            y_pred = torch.view_as_real(y)
        # out = self.linear(DVR)
        # # out_act = self.act(out)
        # y_pred = self.linear_2(out)

        return y_pred



    def model_train(self,x,y,model_path,logger=None,total_train = 1,para = [0.001,150,512,1e-2]):
        learning_rate = para[0]
        epochs = para[1]
        batch_size = para[2]
        alpha = para[3]
        device = self.device

        # optim
        if total_train == 0:
            optimizer = optim.Adam(self.linear.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()


        # dataset
        X_train, X_val, Y_train, Y_val = self.create_dataset(x, y)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = TensorDataset(X_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # train
        logger.info(f'------------------------Joint Training with Orthogonal Loss-------------------------------')
        train_loss_list = []
        val_loss_list = []
        ortho_loss_list = []
        best_metric = float('inf')
        best_model_state = None

        # 训练循环
        start_time = time.time()  # 记录开始时间
        nosave_count = 0
        for epoch in range(epochs):
            self.train()
            i = 0
            train_loss_total = 0
            ortho_loss_total = 0
            train_loss = 0
            for inputs, targets in tqdm(train_loader):
                # 获取对应的复数信号窗口 [batch, M+1]
                optimizer.zero_grad()

                batch_x_signal = inputs.to(device)
                # 生成基函数 [batch, num_basis]
                # amp_basis = self.get_basis_amp(batch_x_signal)
                # 计算正交性损失
                # ortho_loss = self.orthogonal_loss(amp_basis)

                targets = targets.to(device)
                targets_real = torch.view_as_real(targets)
                targets_real = targets_real.to(device)
                if epochs-epoch<0:
                    coef = self.DVR_NN_e(batch_x_signal,targets,alpha)
                    outputs = self(batch_x_signal,coef)
                else:
                    outputs = self(batch_x_signal)
                # 计算预测损失
                mse_loss = criterion(outputs, targets_real)
                # 总损失 = 预测损失 + 正交性损失权重
                # 使用渐进式正交约束权重
                # if epoch <250:
                #     ortho_weight = min(1e-10 * (epoch / 5), 1e-9)  # 逐步增加权重
                # elif epoch < 400:
                #     ortho_weight = 5e-9 #min(5e-5 * (epoch-250), 1e-2)
                # else:
                    # ortho_weight = 10
                total_loss = mse_loss #+ ortho_weight * ortho_loss
                # total_loss = mse_loss + 1e-6 * ortho_loss
                total_loss.backward()
                optimizer.step()
                train_loss_total += mse_loss.item()
                # ortho_loss_total += ortho_loss.item()

            # 验证
            self.eval()
            val_loss_total = 0
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader):
                    # 将数据移动到设备上
                    # 获取对应的复数信号窗口 [batch, M+1]
                    batch_x_signal = inputs.to(device)
                    targets = targets.to(device)
                    targets_real = torch.view_as_real(targets)
                    targets_real = targets_real.to(device)
                    if epochs - epoch < 0:
                        # coef = self.DVR_NN_e(batch_x_signal, targets, alpha)
                        val_outputs = self(batch_x_signal, coef)
                    else:
                        val_outputs = self(batch_x_signal)
                        # coef = self.DVR_NN_e(batch_x_signal, targets,alpha)
                    # val_outputs = self(batch_x_signal, coef)
                    # val_outputs = self(batch_x_signal)
                    val_loss_total += criterion(val_outputs, targets_real).item()

            # 计算平均损失
            avg_train_loss = train_loss_total / len(train_loader)
            # avg_ortho_loss = ortho_loss_total / len(train_loader)
            avg_val_loss = val_loss_total / len(val_loader)

            train_loss_list.append(avg_train_loss)
            # ortho_loss_list.append(avg_ortho_loss)
            val_loss_list.append(avg_val_loss)
            # # 记录指标
            # metrics = {
            #     'epoch': epoch,
            #     'train_loss': train_loss,
            #     'val_loss': val_loss,
            # }

            # # 保存当前模型（按周期命名）
            # torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
            save = 0
            # 更新最佳模型
            # if val_loss_list[-1] < best_metric:
            #     best_metric = val_loss_list[-1]
            #     save = 1
            #     # 保存模型参数（推荐保存为 .pt 或 .pth 文件）
            #     best_model = self.state_dict()
            # 保存最佳模型
            if avg_val_loss < best_metric:
                best_metric = avg_val_loss
                best_model_state = self.state_dict()
                save = 1
                torch.save(best_model_state, model_path)
                # logger.info(f"Model improved, saved to {model_path}")
            # print(f'Epoch {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f} | save:{save}')
            logger.info(
                f'Epoch {epoch + 1:03} | Train Loss: {avg_train_loss:.7f} | Val Loss: {avg_val_loss:.7f} | save:{save}')
            if save == 0:
                nosave_count = nosave_count + 1
            else:
                nosave_count = 0
            # if nosave_count > 100:
            #     break

        # 保存最终模型
        # torch.save(self.state_dict(), model_path)
        logger.info(f"Final model saved to {model_path}")
        # logger.info(f"model save: {model_path} ")
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        logger.info(f"model train time: {elapsed_time:.6f} s")


        # if 1:
        #     y_train_loss = train_loss_list  # loss值，即y轴
        #     x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴
        #
        #     plt.figure()
        #
        #     # 去除顶部和右边框框
        #     ax = plt.axes()
        #     ax.spines['top'].set_visible(False)
        #     ax.spines['right'].set_visible(False)
        #
        #     plt.xlabel('iters')  # x轴标签
        #     plt.ylabel('loss')  # y轴标签
        #
        #     # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        #     # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        #     plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
        #     plt.legend()
        #     plt.title('Loss curve')
        #     plt.savefig(f'figures/MCP_NN/DVR_NN_loss_curve.png')
        #     plt.close()

    def get_basis(self,x_window):
        """
        x_window: 当前窗口的记忆输入 [batch, (M+1)]
        """
        M = self.M
        K = self.K
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # threshold
        # threshold = torch.from_numpy(self.threshold)
        # threshold_matrix = threshold.unsqueeze(1).t().expand(M + 1, -1)  # (M+1,K) K维度延拓 -1 表示保持K维度不变
        # threshold_matrix = threshold_matrix.to(device) # x_windows已移动到设备
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

        X = X_amp
        # for k in range(2,6):
        #     X_amp_k = X_amp.pow(k)
        #     X = torch.concat((X,X_amp_k), dim=1)

        # amp_matrix = X_amp.unsqueeze(2).repeat(1, 1, K)  # K维度延拓 |x(n-i)| (16384,M+1,K)
        # DVR_amp_matrix = amp_matrix - threshold_matrix  # (16384,M+1,K)
        # ABS_DVR_amp = torch.abs(DVR_amp_matrix)  # (16384,M+1,K)
        # DVR_core = ABS_DVR_amp.view(len(ABS_DVR_amp), -1)    # (16384,(M+1)*K)
        # DVR_amp = self.amp_linear(X_amp)    # (16384,(M+1)*K)
        DVR_amp = self.DVR_layers(X)
        DVR_core = torch.abs(DVR_amp)  # (16384,(M+1)*K)

        # if self.activation == "ABS":
        #     DVR_core = torch.abs(DVR_amp)  # (16384,(M+1)*K)
        # else:
        #     DVR_core = self.act(DVR_amp)


        # theta(n-i) DVR PHASE
        X_phase = torch.angle(X_tensor)  #theta(n-i) (16384,M+1)
        e_j_phase = torch.exp(1j * X_phase)  # (16384,M+1)
        e_j_phase_matrix = e_j_phase.unsqueeze(2).repeat(1, 1, K)  # (16384,M+1,K)
        # DVR_phase = torch.flatten(e_j_phase_matrix)  # (16384,(M+1)*K)
        DVR_phase = e_j_phase_matrix.view(len(e_j_phase_matrix), -1)

        # ABS(|x(n)|-beta) DDR CORE
        # DDR_amp_matrix = xn_amp_matrix - threshold_matrix #(16384,M+1,K)
        # ABS_DDR_amp = torch.abs(DDR_amp_matrix)     # (16384,M+1,K)
        # DDR_core = ABS_DDR_amp.view(len(ABS_DDR_amp), -1) # (16384,(M+1)*K)
        # DDR_amp = self.DDR_linear(xn_amp.unsqueeze(1)) # (16384,(M+1)*K)
        # # DDR_amp = self.DDR_layers(xn_amp.unsqueeze(1))
        # if self.activation == "ABS":
        #     DDR_core = torch.abs(DDR_amp)  # (16384,(M+1)*K)
        # else:
        #     DDR_core = self.act(DDR_amp)


        # e_Phase_m = DVR_phase
        e_Phase_n = DVR_xn/DVR_xn_amp #torch.exp(1j * Phase_n)
        # DDR31 = DVR_xn * DVR_xn * torch.conj(DVR_X)
        # e_Phase_DDR31 = DDR31/torch.abs(DDR31)
        # DDR32 = DVR_X * DVR_X * torch.conj(DVR_xn)
        # e_Phase_DDR32 = DDR32 / torch.abs(DDR32)
        # DDR51 = DVR_xn * DVR_xn * DVR_xn * torch.conj(DVR_X) * torch.conj(DVR_X)
        # e_Phase_DDR51 = DDR51 / torch.abs(DDR51)
        # DDR52 = DVR_X * DVR_X * DVR_X * torch.conj(DVR_xn) * torch.conj(DVR_xn)
        # e_Phase_DDR52 = DDR52 / torch.abs(DDR52)


        # DVR basis
        DVR_out_linear = X_tensor  # (16384,M+1)
        DVR_out_1 = DVR_core * DVR_phase  # (16384,(M+1)*K)
        DVR_out_21 = DVR_core * DVR_phase * DVR_xn_amp  # (16384,(M+1)*K)
        DVR_out_22 = DVR_core * DVR_xn
        DVR_out_23 = DVR_core * DVR_X
        DVR_out_24 = DVR_core * e_Phase_n * torch.abs(DVR_X)
        DVR_out_DDR_1 = DVR_core * DVR_X * DVR_X * torch.conj(DVR_xn)
        DVR_out_DDR_2 = DVR_core * DVR_xn * DVR_xn * torch.conj(DVR_X)
        DVR_out_DDR_51 = DVR_core * DVR_xn * DVR_xn * DVR_xn * torch.conj(DVR_X) * torch.conj(DVR_X)
        # DVR_core_p = DVR_core.view(len(DVR_core),(M+1)*K,1)
        # DVR_out_m = DVR_core_p[:,:,0].squeeze() * e_Phase_m
        # DVR_out_n = DVR_core_p[:,:,1].squeeze() * e_Phase_n
        # DVR_out_nm = DVR_core * e_Phase_n / e_Phase_nb
        # DVR_out_DDR31 = DVR_core_p[:,:,2].squeeze() * e_Phase_DDR31
        # DVR_out_DDR32 = DVR_core_p[:, :, 3].squeeze() * e_Phase_DDR32
        # DVR_out_DDR32 = DVR_core * e_Phase_DDR32
        # DVR_out_DDR51 = DVR_core_p[:,:,3].squeeze() * e_Phase_DDR51
        # DVR_out_DDR52 = DVR_core * e_Phase_DDR52

        # DVR_out_21,DVR_out_22,DVR_out_23,DVR_out_24,DVR_out_21,DVR_out_22,DVR_out_23,DVR_out_24,DVR_out_DDR_2
        if self.term == 7:
            DVR_out_full = torch.concatenate(( DVR_out_1,DVR_out_linear,DVR_out_21,DVR_out_22,DVR_out_23,DVR_out_24,DVR_out_DDR_2,DVR_out_DDR_51), dim=1) #  DVR_out_linear,DVR_out_1, DVR_out_21,DVR_out_22, DVR_out_23, DVR_out_DDR_1,DVR_out_DDR_2
        elif self.term == 6:
            DVR_out_full = torch.concatenate((DVR_out_1, DVR_out_linear, DVR_out_21, DVR_out_22, DVR_out_23, DVR_out_24,DVR_out_DDR_2), dim=1)
        elif self.term == 5:
            DVR_out_full = torch.concatenate((DVR_out_1, DVR_out_linear, DVR_out_21, DVR_out_22, DVR_out_23, DVR_out_24), dim=1)
        elif self.term == 4:
            DVR_out_full = torch.concatenate((DVR_out_1, DVR_out_linear, DVR_out_21, DVR_out_22, DVR_out_23), dim=1)
        elif self.term == 3:
            DVR_out_full = torch.concatenate((DVR_out_1, DVR_out_linear, DVR_out_21, DVR_out_22), dim=1)
        elif self.term == 2:
            DVR_out_full = torch.concatenate((DVR_out_1, DVR_out_linear, DVR_out_21), dim=1)
        elif self.term == 1:
            DVR_out_full = torch.concatenate((DVR_out_1, DVR_out_linear), dim=1)
        elif self.term == 0:
            DVR_out_full = DVR_out_1

        return DVR_out_full

    def DVR_NN_e(self,x_window,y,alpha = 1e-3,pri = 0):
        M = self.M
        # threshold = self.threshold
        K = self.K
        device = self.device

        basis = self.get_basis(x_window)

        X = basis
        np_X = X.cpu().detach().numpy()
        y_copy = y
        np_y = y_copy.cpu().detach().numpy()

        np_XH = np.conjugate(np_X.T)
        np_coef = np.linalg.pinv(np_XH.dot(np_X) + alpha * np.eye(np_X.shape[1])).dot(np_XH).dot(np_y)
        coef = torch.from_numpy(np_coef).to(device)

        self.coef = coef
        if pri==1:
            y_model = X @ coef
            y_model = y_model.cpu().detach().numpy()
            NMSE = cal.nmse(np_y, y_model)
            print(f"train NMSE: {NMSE:.3f} dB")
        return coef

        # basis = self.get_basis(x_window)  # 已在GPU上
        # X = basis
        # y = y  # 已在GPU上
        # # 计算伪逆（全部在GPU上）
        # # XH = X.conj().T
        # coef = torch.linalg.lstsq(X, y, rcond=alpha).solution
        # # Gram = XH @ X + alpha * torch.eye(X.shape[1], device=self.device)
        # # coef = torch.linalg.pinv(Gram) @ XH @ y  # 或使用 torch.linalg.lstsq
        # if pri:
        #     y_model = X @ coef
        #     NMSE = cal.nmse(y.cpu().numpy(), y_model.cpu().numpy())
        #     print(f"train NMSE: {NMSE:.3f} dB")
        # return coef



    # def DVR_NN_v(self, x, coef):
    #     M = self.M
    #     K = self.K
    #     device = self.device
    #     sequences = volterra_nn.create_memory_seq(x, M)  # [N, M+1]
    #     x_window = torch.from_numpy(sequences).to(device)
    #
    #     X = self.get_basis(x_window).cpu().detach().numpy()
    #     y = X.dot(coef)
    #     return y
    def DVR_NN_v(self, x_window, coef):
        """使用提取的系数进行预测（冻结基函数网络）"""
        # self.eval()  # 设置为评估模式
        M = self.M
        K = self.K
        device = self.device
        # sequences = volterra_nn.create_memory_seq(x, M)  # [N, M+1]
        # x_window = torch.from_numpy(sequences).to(device)

        # 生成正交基函数
        # with torch.no_grad():
        basis = self.get_basis(x_window)
        X = basis
        # X = basis.cpu().detach().numpy()
        # X = self.get_basis(x_window).cpu().detach().numpy()
        y = X @ coef
        return y

    # 数据预处理函数
    def create_dataset(self,x, y, test_size=0.2):
        M = self.M
        # 转换为实部虚部分离格式

        sequences = volterra_nn.create_memory_seq(x,M)  # [N, M+1]

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
        K = self.K
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 将模型移动到设备上
        self.to(device)
        self.eval()
        sequences = volterra_nn.create_memory_seq(signal, M)  # [N, M+1]
        x = torch.from_numpy(sequences).to(device)
        out = self(x,coef)
        out_complex = torch.view_as_complex(out).cpu()
        y_pred = out_complex.detach().numpy()

        # y_pred = np.stack()
        return y_pred
    # def train(self):
    #     self.train()

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

# import torch
# import torch.nn as nn
# import numpy as np
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# import time
# from function import Function_Calculate as cal, Function_Lib as fun
# from tqdm import tqdm
# from matplotlib import pyplot as plt
# import Model.volterra_nn as volterra_nn
#
# # class ADVR_NN(nn.Module):
# #     def __init__(self, K, M, activation="ReLU"):
# #         super().__init__()
# #         # self.total_train = 1
# #         self.M = M
# #         self.K = K
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         # self.threshold = threshold
# #         assert activation == "ReLU" or "Tanh" or "GELU" or "None"
# #         self.activation = activation
# #         self.DVR_layers = nn.Sequential()
# #         # self.DDR_layers = nn.Sequential()
# #         # layer_dims = [M+1,16,16,(M+1)*K]
# #         layer_dims = [M + 1,(M + 1) * K]
# #         for index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
# #             # self.layers.add_module(
# #             #     f"res_{index}",
# #             #     ResidualBlock(in_dim, out_dim, activation)
# #             # )
# #             self.DVR_layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim).double())
# #             # self.DDR_layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim).double())
# #             if activation == "ReLU":
# #                 self.DVR_layers.add_module("actFunc " + str(index), nn.ReLU())
# #                 # self.DDR_layers.add_module("actFunc " + str(index), nn.ReLU())
# #             elif activation == "Tanh":
# #                 self.DVR_layers.add_module("actFunc " + str(index), nn.Tanh())
# #                 # self.DDR_layers.add_module("actFunc " + str(index), nn.Tanh())
# #             elif activation == "GELU":
# #                 self.DVR_layers.add_module("actFunc " + str(index), nn.GELU())
# #                 # self.DDR_layers.add_module("actFunc " + str(index), nn.GELU())
# #             # self.layers.add_module("dropout" + str(index), nn.Dropout(0.2))
# #         if self.activation != "ABS":
# #             self.DVR_layers = self.DVR_layers[:-1]  # remove the last activation layer
# #         # self.amp_linear = nn.Linear(M+1, (M+1)*K).double()
# #         # self.DDR_linear = nn.Linear(1, (M+1)*K).double()
# #         # self.DDR_linear = nn.Linear(M + 1, (M + 1) * K).double()
# #         # self.amp_act = nn.()
# #         # self.linear = nn.Linear((K*(M+1)*6+M+1)*2, 2,bias=False).double()
# #         if self.activation != "ABS":
# #             if self.activation == "ReLU":
# #                 self.act = nn.ReLU()
# #             elif self.activation == "Tanh":
# #                 self.act = nn.Tanh()
# #             elif self.activation == "GELU":
# #                 self.act = nn.GELU()
# #         # self.linear_2 = nn.Linear(12, 2, bias=False).double()
# #         # self.main_layers = nn.Sequential()
# #         # self.para_layers = nn.Sequential()
# #         # self.main_nn = volterra_nn.GMP_NN(self.layer_dim1, L, M = self.M,activation=activation)
# #         # self.para_nn = volterra_nn.MCP_BASE_NN(self.layer_dim2, L, M = self.M, K = 5,activation=activation)
# #         # self.out_layer = nn.Linear(2*M+2, 2,bias=False)
# #
# #     def forward(self, x_window,coef = None):
# #         """
# #         x_window: 当前窗口的记忆输入 [batch, (M+1)]
# #         """
# #         # if coef is None:
# #         #     DVR_out_full = self.get_basis(x_window)
# #         #     DVR_out_full_real = torch.view_as_real(DVR_out_full)
# #         #     # DVR = DVR_out_full_real.view(len(DVR_out_full_real), -1)
# #         #     DVR = DVR_out_full_real.flatten(1)
# #         #     y_pred = self.linear(DVR)
# #         # else:
# #         y = self.DVR_NN_v(x_window,coef)
# #         y_pred = torch.view_as_real(y)
# #         # out = self.linear(DVR)
# #         # # out_act = self.act(out)
# #         # y_pred = self.linear_2(out)
# #
# #         return y_pred
# #
# #
# #     def model_train(self,x,y,model_path,logger=None,total_train = 1):
# #         learning_rate = 0.001
# #         epochs = 200
# #         batch_size = 1024
# #         device = self.device
# #
# #         # optim
# #         if total_train == 0:
# #             optimizer = optim.Adam(self.linear.parameters(), lr=learning_rate)
# #         else:
# #             optimizer = optim.Adam(self.parameters(), lr=learning_rate)
# #         criterion = nn.MSELoss()
# #
# #
# #         # dataset
# #         X_train, X_val, Y_train, Y_val = self.create_dataset(x, y)
# #         train_dataset = TensorDataset(X_train, Y_train)
# #         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# #         val_dataset = TensorDataset(X_val, Y_val)
# #         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# #
# #         # train
# #         logger.info(f'------------------------Joint Training with Orthogonal Loss-------------------------------')
# #         train_loss_list = []
# #         val_loss_list = []
# #         ortho_loss_list = []
# #         best_metric = float('inf')
# #         best_model_state = None
# #
# #         # 训练循环
# #         start_time = time.time()  # 记录开始时间
# #         nosave_count = 0
# #         for epoch in range(epochs):
# #             self.train()
# #             i = 0
# #             train_loss_total = 0
# #             ortho_loss_total = 0
# #             train_loss = 0
# #             for inputs, targets in tqdm(train_loader):
# #                 # 获取对应的复数信号窗口 [batch, M+1]
# #                 optimizer.zero_grad()
# #
# #                 batch_x_signal = inputs.to(device)
# #                 # 生成基函数 [batch, num_basis]
# #                 amp_basis = self.get_basis_amp(batch_x_signal)
# #                 # 计算正交性损失
# #                 ortho_loss = self.orthogonal_loss(amp_basis)
# #
# #                 targets = targets.to(device)
# #                 targets_real = torch.view_as_real(targets)
# #                 targets_real = targets_real.to(device)
# #                 coef = self.DVR_NN_e(batch_x_signal,targets)
# #
# #                 # outputs = model(inputs)
# #                 outputs = self(batch_x_signal,coef)
# #
# #                 # 计算预测损失
# #                 mse_loss = criterion(outputs, targets_real)
# #                 # 总损失 = 预测损失 + 正交性损失权重
# #                 # 使用渐进式正交约束权重
# #                 if epoch <250:
# #                     ortho_weight = min(1e-10 * (epoch / 5), 1e-9)  # 逐步增加权重
# #                 elif epoch < 400:
# #                     ortho_weight = 5e-9 #min(5e-5 * (epoch-250), 1e-2)
# #                 # else:
# #                     # ortho_weight = 10
# #                 total_loss = mse_loss #+ ortho_weight * ortho_loss
# #                 # total_loss = mse_loss + 1e-6 * ortho_loss
# #                 total_loss.backward()
# #                 optimizer.step()
# #                 train_loss_total += mse_loss.item()
# #                 ortho_loss_total += ortho_loss.item()
# #
# #             # 验证
# #             self.eval()
# #             val_loss_total = 0
# #             with torch.no_grad():
# #                 for inputs, targets in tqdm(val_loader):
# #                     # 将数据移动到设备上
# #                     # 获取对应的复数信号窗口 [batch, M+1]
# #                     batch_x_signal = inputs.to(device)
# #                     targets = targets.to(device)
# #                     targets_real = torch.view_as_real(targets)
# #                     targets_real = targets_real.to(device)
# #
# #                     coef = self.DVR_NN_e(batch_x_signal, targets)
# #                     # outputs = model(inputs)
# #                     val_outputs = self(batch_x_signal,coef)
# #                     val_loss_total += criterion(val_outputs, targets_real).item()
# #
# #             # 计算平均损失
# #             avg_train_loss = train_loss_total / len(train_loader)
# #             avg_ortho_loss = ortho_loss_total / len(train_loader)
# #             avg_val_loss = val_loss_total / len(val_loader)
# #
# #             train_loss_list.append(avg_train_loss)
# #             ortho_loss_list.append(avg_ortho_loss)
# #             val_loss_list.append(avg_val_loss)
# #             # # 记录指标
# #             # metrics = {
# #             #     'epoch': epoch,
# #             #     'train_loss': train_loss,
# #             #     'val_loss': val_loss,
# #             # }
# #
# #             # # 保存当前模型（按周期命名）
# #             # torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
# #             save = 0
# #             # 更新最佳模型
# #             # if val_loss_list[-1] < best_metric:
# #             #     best_metric = val_loss_list[-1]
# #             #     save = 1
# #             #     # 保存模型参数（推荐保存为 .pt 或 .pth 文件）
# #             #     best_model = self.state_dict()
# #             # 保存最佳模型
# #             if avg_val_loss < best_metric:
# #                 best_metric = avg_val_loss
# #                 best_model_state = self.state_dict()
# #                 save = 1
# #                 torch.save(best_model_state, model_path)
# #                 # logger.info(f"Model improved, saved to {model_path}")
# #             # print(f'Epoch {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f} | save:{save}')
# #             logger.info(
# #                 f'Epoch {epoch + 1:03} | Train Loss: {avg_train_loss:.6f} | Ortho Loss: {avg_ortho_loss:.6f} | Val Loss: {avg_val_loss:.6f} | save:{save}')
# #             if save == 0:
# #                 nosave_count = nosave_count + 1
# #             else:
# #                 nosave_count = 0
# #             # if nosave_count > 100:
# #             #     break
# #
# #         # 保存最终模型
# #         # torch.save(self.state_dict(), model_path)
# #         logger.info(f"Final model saved to {model_path}")
# #         # logger.info(f"model save: {model_path} ")
# #         end_time = time.time()  # 记录结束时间
# #         elapsed_time = end_time - start_time
# #         logger.info(f"model train time: {elapsed_time:.6f} s")
# #
# #
# #         if 1:
# #             y_train_loss = train_loss_list  # loss值，即y轴
# #             x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴
# #
# #             plt.figure()
# #
# #             # 去除顶部和右边框框
# #             ax = plt.axes()
# #             ax.spines['top'].set_visible(False)
# #             ax.spines['right'].set_visible(False)
# #
# #             plt.xlabel('iters')  # x轴标签
# #             plt.ylabel('loss')  # y轴标签
# #
# #             # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# #             # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
# #             plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
# #             plt.legend()
# #             plt.title('Loss curve')
# #             plt.savefig(f'figures/MCP_NN/DVR_NN_loss_curve.png')
# #             plt.close()
# #
# #     def get_basis(self,x_window):
# #         """
# #         x_window: 当前窗口的记忆输入 [batch, (M+1)]
# #         """
# #         M = self.M
# #         K = self.K
# #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         # threshold
# #         # threshold = torch.from_numpy(self.threshold)
# #         # threshold_matrix = threshold.unsqueeze(1).t().expand(M + 1, -1)  # (M+1,K) K维度延拓 -1 表示保持K维度不变
# #         # threshold_matrix = threshold_matrix.to(device) # x_windows已移动到设备
# #         # x(n)
# #         xn = x_window[:,-1] #当前信号x(n) (16384,1)
# #         xn_amp = torch.abs(xn)  #当前信号|x(n)| (16384,1)
# #         xn_amp_matrix = xn_amp.unsqueeze(1).repeat(1, M + 1).unsqueeze(2).repeat(1, 1, K)  # 当前信号|x(n)| (16384,M+1,K)
# #         xn_matrix = xn.unsqueeze(1).repeat(1, M + 1).unsqueeze(2).repeat(1, 1, K)  # (16384,(M+1)*K)
# #         DVR_xn = xn_matrix.view(len(xn), -1)
# #         DVR_xn_amp = xn_amp_matrix.view(len(xn), -1)
# #
# #         # x(n-i)
# #         X_tensor = x_window #记忆信号x(n-i) (16384,M+1)
# #         X_matrix = X_tensor.unsqueeze(2).repeat(1, 1, K)  # K维度延拓 x(n-i) (16384,M+1,K)
# #         DVR_X = X_matrix.view(len(X_matrix), -1)    # 展平 x(n-i) (16384,(M+1)*K)
# #
# #         # |x(n-i)| DVR CORE
# #         X_amp = torch.abs(X_tensor)  #|x(n-i)| (16384,M+1)
# #         # amp_matrix = X_amp.unsqueeze(2).repeat(1, 1, K)  # K维度延拓 |x(n-i)| (16384,M+1,K)
# #         # DVR_amp_matrix = amp_matrix - threshold_matrix  # (16384,M+1,K)
# #         # ABS_DVR_amp = torch.abs(DVR_amp_matrix)  # (16384,M+1,K)
# #         # DVR_core = ABS_DVR_amp.view(len(ABS_DVR_amp), -1)    # (16384,(M+1)*K)
# #         # DVR_amp = self.amp_linear(X_amp)    # (16384,(M+1)*K)
# #         DVR_amp = self.DVR_layers(X_amp)
# #         DVR_core = torch.abs(DVR_amp)  # (16384,(M+1)*K)
# #         # if self.activation == "ABS":
# #         #     DVR_core = torch.abs(DVR_amp)  # (16384,(M+1)*K)
# #         # else:
# #         #     DVR_core = self.act(DVR_amp)
# #
# #
# #         # theta(n-i) DVR PHASE
# #         X_phase = torch.angle(X_tensor)  #theta(n-i) (16384,M+1)
# #         e_j_phase = torch.exp(1j * X_phase)  # (16384,M+1)
# #         e_j_phase_matrix = e_j_phase.unsqueeze(2).repeat(1, 1, K)  # (16384,M+1,K)
# #         # DVR_phase = torch.flatten(e_j_phase_matrix)  # (16384,(M+1)*K)
# #         DVR_phase = e_j_phase_matrix.view(len(e_j_phase_matrix), -1)
# #
# #         # ABS(|x(n)|-beta) DDR CORE
# #         # DDR_amp_matrix = xn_amp_matrix - threshold_matrix #(16384,M+1,K)
# #         # ABS_DDR_amp = torch.abs(DDR_amp_matrix)     # (16384,M+1,K)
# #         # DDR_core = ABS_DDR_amp.view(len(ABS_DDR_amp), -1) # (16384,(M+1)*K)
# #         # DDR_amp = self.DDR_linear(X_amp)  # (16384,(M+1)*K)
# #         # # DDR_amp = self.DDR_linear(xn_amp.unsqueeze(1)) # (16384,(M+1)*K)
# #         # # DDR_amp = self.DDR_layers(xn_amp.unsqueeze(1))
# #         # if self.activation == "ABS":
# #         #     DDR_core = torch.abs(DDR_amp)  # (16384,(M+1)*K)
# #         # else:
# #         #     DDR_core = self.act(DDR_amp)
# #
# #         # DVR basis
# #         DVR_out_linear = X_tensor  # (16384,M+1)
# #         DVR_out_1 = DVR_core * DVR_phase  # (16384,(M+1)*K)
# #         DVR_out_21 = DVR_core * DVR_phase * DVR_xn_amp  # (16384,(M+1)*K)
# #         DVR_out_22 = DVR_core * DVR_xn
# #         DVR_out_23 = DVR_core * DVR_X
# #
# #         DVR_out_MP_3 = DVR_core * DVR_xn * DVR_xn_amp ** 2
# #         DVR_out_MP_5 = DVR_core * DVR_xn * DVR_xn_amp ** 4
# #
# #         # DVR_out_DDR_1 = DDR_core * DVR_X
# #         # DVR_out_DDR_2 = DDR_core * DVR_xn * DVR_xn * torch.conj(DVR_X)
# #         DVR_out_DDR_1 = DVR_core * DVR_X
# #         DVR_out_DDR_2 = DVR_core * DVR_xn * DVR_xn * torch.conj(DVR_X)
# #
# #         DVR_out_full = torch.concatenate((DVR_out_linear, DVR_out_1, DVR_out_21, DVR_out_22, DVR_out_23, DVR_out_DDR_1,DVR_out_DDR_2,DVR_out_MP_3,DVR_out_MP_5), dim=1)
# #         # 添加正交化层
# #         # 使用QR分解（更稳定）
# #         # Q, R = torch.linalg.qr(DVR_out_full,mode='reduced')
# #         # return Q
# #         return DVR_out_full
# #
# #     def DVR_NN_e(self,x_window,y,alpha=5e-2,pri = 0):
# #         """使用最小二乘法提取系数（冻结基函数网络）"""
# #         # self.eval()  # 设置为评估模式
# #         M = self.M
# #         # threshold = self.threshold
# #         K = self.K
# #         device = self.device
# #
# #         # sequences = volterra_nn.create_memory_seq(x, M)  # [N, M+1]
# #         # x_window = torch.from_numpy(sequences).to(device)
# #
# #         # 生成正交基函数
# #         # with torch.no_grad():
# #         basis = self.get_basis(x_window)
# #
# #         X = basis
# #         np_X = X.cpu().detach().numpy()
# #         y_copy = y
# #         np_y = y_copy.cpu().detach().numpy()
# #         # # 计算共轭转置 (对实数矩阵等价于转置)
# #         # XH = X.conj().T
# #         #
# #         # # 计算正则化矩阵
# #         # I = torch.eye(X.shape[1], device=device)
# #         # reg_matrix = alpha * I
# #         # # 核心计算: (X^H X + αI)^(-1) X^H y
# #         # XHX = XH @ X
# #         # XHX_reg = XHX + reg_matrix
# #         # # 使用 Cholesky 分解求解更稳定
# #         # try:
# #         #     L = torch.linalg.cholesky(XHX_reg)
# #         #     coef = torch.cholesky_solve(XH @ y, L)
# #         # except RuntimeError:  # 如果 Cholesky 失败，使用伪逆
# #         #     pinv = torch.linalg.pinv(XHX_reg)
# #         #     coef = pinv @ XH @ y
# #         # pinv = torch.linalg.pinv(XHX_reg)
# #         # coef = pinv @ XH @ y
# #         # X = basis.cpu().detach().numpy()
# #         # X = self.get_basis(x_window).cpu().detach().numpy()
# #         np_XH = np.conjugate(np_X.T)
# #         np_coef = np.linalg.pinv(np_XH.dot(np_X) + alpha * np.eye(np_X.shape[1])).dot(np_XH).dot(np_y)
# #         coef = torch.from_numpy(np_coef).to(device)
# #         y_model = X @ coef
# #
# #         y_model = y_model.cpu().detach().numpy()
# #         self.coef = coef
# #         NMSE = cal.nmse(np_y, y_model)
# #         if pri==1:
# #             print(f"NMSE: {NMSE:.3f} dB")
# #         return coef
# #
# #     def DVR_NN_v(self, x_window, coef):
# #         """使用提取的系数进行预测（冻结基函数网络）"""
# #         # self.eval()  # 设置为评估模式
# #         M = self.M
# #         K = self.K
# #         device = self.device
# #         # sequences = volterra_nn.create_memory_seq(x, M)  # [N, M+1]
# #         # x_window = torch.from_numpy(sequences).to(device)
# #
# #         # 生成正交基函数
# #         # with torch.no_grad():
# #         basis = self.get_basis(x_window)
# #         X = basis
# #         # X = basis.cpu().detach().numpy()
# #         # X = self.get_basis(x_window).cpu().detach().numpy()
# #         y = X @ coef
# #         return y
# #
# #     # 数据预处理函数
# #     def create_dataset(self,x, y, test_size=0.2):
# #         M = self.M
# #         # 转换为实部虚部分离格式
# #
# #         sequences = volterra_nn.create_memory_seq(x,M)  # [N, M+1]
# #
# #         X_tensor = torch.from_numpy(sequences)
# #         # targets = volterra_nn.complex_to_real(y)    # [N, 1]
# #         targets = y
# #
# #         Y_tensor = torch.from_numpy(y) # (N, 2)
# #         # Y_tensor = torch.view_as_complex(Y_tensor)
# #         X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=test_size, shuffle=False)
# #         return [X_train, X_val, Y_train, Y_val]
# #
# #     def apply_dpd(self, signal,coef = None):
# #         M = self.M
# #         # threshold = self.threshold
# #         K = self.K
# #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         # 将模型移动到设备上
# #         self.to(device)
# #         self.eval()
# #         sequences = volterra_nn.create_memory_seq(signal, M)  # [N, M+1]
# #         x = torch.from_numpy(sequences).to(device)
# #         # if coef is None:
# #         #     out = self(x)
# #         #     out_complex = torch.view_as_complex(out).cpu()
# #         #     y_pred = out_complex.detach().numpy()
# #         # else:
# #         out = self(x,coef)
# #         out_complex = torch.view_as_complex(out).cpu()
# #         y_pred = out_complex.detach().numpy()
# #         # y_pred = np.stack()
# #         return y_pred
# #     # def train(self):
# #     #     self.train()
# #
# #     def modified_gram_schmidt(self, V):
# #         """改进的Gram-Schmidt正交化"""
# #         U = V.clone()
# #         for k in range(V.shape[1]):
# #             # 归一化当前列
# #             norm = torch.norm(U[:, k])
# #             if norm > 1e-8:
# #                 U[:, k] = U[:, k] / norm
# #
# #             # 投影并减去其他方向的成分
# #             if k < V.shape[1] - 1:
# #                 proj = (U[:, k].conj() @ U[:, k + 1:]).real
# #                 U[:, k + 1:] = U[:, k + 1:] - U[:, k].unsqueeze(1) * proj
# #         return U
# #
# #     def orthogonal_loss(self, basis):
# #         """
# #         计算正交但不标准化基函数的损失
# #         目标：最小化非对角线元素的绝对值
# #         """
# #         # 计算Gram矩阵
# #         G = basis.conj().T @ basis
# #
# #         # 创建非对角线元素的掩码
# #         n = G.shape[0]
# #         mask = ~torch.eye(n, dtype=torch.bool, device=G.device)
# #
# #         # 提取非对角线元素
# #         off_diag = G[mask]
# #
# #         # 计算平均幅度（使用绝对值）
# #         # return torch.mean(torch.abs(off_diag))
# #         return torch.sum(torch.abs(off_diag))
# #
# #     def get_basis_amp(self,x_window):
# #         """
# #         x_window: 当前窗口的记忆输入 [batch, (M+1)]
# #         """
# #         M = self.M
# #         K = self.K
# #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         # threshold
# #         # threshold = torch.from_numpy(self.threshold)
# #         # threshold_matrix = threshold.unsqueeze(1).t().expand(M + 1, -1)  # (M+1,K) K维度延拓 -1 表示保持K维度不变
# #         # threshold_matrix = threshold_matrix.to(device) # x_windows已移动到设备
# #         # x(n)
# #         xn = x_window[:,-1] #当前信号x(n) (16384,1)
# #         xn_amp = torch.abs(xn)  #当前信号|x(n)| (16384,1)
# #         xn_amp_matrix = xn_amp.unsqueeze(1).repeat(1, M + 1).unsqueeze(2).repeat(1, 1, K)  # 当前信号|x(n)| (16384,M+1,K)
# #         xn_matrix = xn.unsqueeze(1).repeat(1, M + 1).unsqueeze(2).repeat(1, 1, K)  # (16384,(M+1)*K)
# #         DVR_xn = xn_matrix.view(len(xn), -1)
# #         DVR_xn_amp = xn_amp_matrix.view(len(xn), -1)
# #
# #         # x(n-i)
# #         X_tensor = x_window #记忆信号x(n-i) (16384,M+1)
# #         X_matrix = X_tensor.unsqueeze(2).repeat(1, 1, K)  # K维度延拓 x(n-i) (16384,M+1,K)
# #         DVR_X = X_matrix.view(len(X_matrix), -1)    # 展平 x(n-i) (16384,(M+1)*K)
# #
# #         # |x(n-i)| DVR CORE
# #         X_amp = torch.abs(X_tensor)  #|x(n-i)| (16384,M+1)
# #         # amp_matrix = X_amp.unsqueeze(2).repeat(1, 1, K)  # K维度延拓 |x(n-i)| (16384,M+1,K)
# #         # DVR_amp_matrix = amp_matrix - threshold_matrix  # (16384,M+1,K)
# #         # ABS_DVR_amp = torch.abs(DVR_amp_matrix)  # (16384,M+1,K)
# #         # DVR_core = ABS_DVR_amp.view(len(ABS_DVR_amp), -1)    # (16384,(M+1)*K)
# #         # DVR_amp = self.amp_linear(X_amp)    # (16384,(M+1)*K)
# #         DVR_amp = self.DVR_layers(X_amp)
# #         DVR_core = torch.abs(DVR_amp)  # (16384,(M+1)*K)
# #         # if self.activation == "ABS":
# #         #     DVR_core = torch.abs(DVR_amp)  # (16384,(M+1)*K)
# #         # else:
# #         #     DVR_core = self.act(DVR_amp)
# #
# #
# #         # theta(n-i) DVR PHASE
# #         X_phase = torch.angle(X_tensor)  #theta(n-i) (16384,M+1)
# #         e_j_phase = torch.exp(1j * X_phase)  # (16384,M+1)
# #         e_j_phase_matrix = e_j_phase.unsqueeze(2).repeat(1, 1, K)  # (16384,M+1,K)
# #         # DVR_phase = torch.flatten(e_j_phase_matrix)  # (16384,(M+1)*K)
# #         DVR_phase = e_j_phase_matrix.view(len(e_j_phase_matrix), -1)
# #
# #         # ABS(|x(n)|-beta) DDR CORE
# #         # DDR_amp_matrix = xn_amp_matrix - threshold_matrix #(16384,M+1,K)
# #         # ABS_DDR_amp = torch.abs(DDR_amp_matrix)     # (16384,M+1,K)
# #         # DDR_core = ABS_DDR_amp.view(len(ABS_DDR_amp), -1) # (16384,(M+1)*K)
# #         # DDR_amp = self.DDR_linear(X_amp)  # (16384,(M+1)*K)
# #         # # DDR_amp = self.DDR_linear(xn_amp.unsqueeze(1)) # (16384,(M+1)*K)
# #         # # DDR_amp = self.DDR_layers(xn_amp.unsqueeze(1))
# #         # if self.activation == "ABS":
# #         #     DDR_core = torch.abs(DDR_amp)  # (16384,(M+1)*K)
# #         # else:
# #         #     DDR_core = self.act(DDR_amp)
# #
# #         AMP_Core = DVR_core
# #
# #         return AMP_Core
#
# class DVR_NN(nn.Module):
#     def __init__(self, layer_dims,K, M, activation="ReLU",term = 6):
#         super().__init__()
#         # self.total_train = 1
#         self.M = M
#         self.K = K
#         self.term = term
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.threshold = threshold
#         # ---- meta-split（批内双流 LS）超参，默认值可由 model_train(para=...) 覆盖 ----
#         # batch 内用于 LS 提参的比例；<=0 表示关闭 meta-split（退回旧的"本批自拟合"，仅作基线）
#         self.split_ratio = 0.75
#         self.eval_every = 50      # 每隔多少个 epoch 记一次额外诊断（右端只有几次，开销可忽略）
#         self.meta_diag = False    # 是否额外计算 cond / rank 诊断
#         assert activation == "ReLU" or "Tanh" or "GELU" or "Sigmoid" or "None"
#         self.activation = activation
#         self.DVR_layers = nn.Sequential()
#         # self.DDR_layers = nn.Sequential()
#         # layer_dims = [M+1,(M+1),(M+1),(M+1)*K]
#         for index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
#             # self.layers.add_module(
#             #     f"res_{index}",
#             #     ResidualBlock(in_dim, out_dim, activation)
#             # )
#             self.DVR_layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim).double())
#             # self.DDR_layers.add_module("linear " + str(index), nn.Linear(in_dim, out_dim).double())
#             if activation == "ReLU":
#                 self.DVR_layers.add_module("actFunc " + str(index), nn.ReLU())
#                 # self.DDR_layers.add_module("actFunc " + str(index), nn.ReLU())
#             elif activation == "Tanh":
#                 self.DVR_layers.add_module("actFunc " + str(index), nn.Tanh())
#                 # self.DDR_layers.add_module("actFunc " + str(index), nn.Tanh())
#             elif activation == "Sigmoid":
#                 self.DVR_layers.add_module("actFunc " + str(index), nn.Sigmoid())
#             elif activation == "GELU":
#                 self.DVR_layers.add_module("actFunc " + str(index), nn.GELU())
#                 # self.DDR_layers.add_module("actFunc " + str(index), nn.GELU())
#             # self.layers.add_module("dropout" + str(index), nn.Dropout(0.2))
#         self.DVR_layers = self.DVR_layers[:-1]  # remove the last activation layer
#         self.linear = nn.Linear((K*(M+1)*term+M+1)*2, 2,bias=False).double() #+M+1
#         if self.activation != "ABS":
#             if self.activation == "ReLU":
#                 self.act = nn.ReLU()
#             elif self.activation == "Tanh":
#                 self.act = nn.Tanh()
#             elif self.activation == "GELU":
#                 self.act = nn.GELU()
#
#         # self.linear_2 = nn.Linear(12, 2, bias=False).double()
#         # self.main_layers = nn.Sequential()
#         # self.para_layers = nn.Sequential()
#         # self.main_nn = volterra_nn.GMP_NN(self.layer_dim1, L, M = self.M,activation=activation)
#         # self.para_nn = volterra_nn.MCP_BASE_NN(self.layer_dim2, L, M = self.M, K = 5,activation=activation)
#         # self.out_layer = nn.Linear(2*M+2, 2,bias=False)
#
#     def forward(self, x_window,coef = None):
#         """
#         x_window: 当前窗口的记忆输入 [batch, (M+1)]
#         """
#         if coef is None:
#             DVR_out_full = self.get_basis(x_window)
#             DVR_out_full_real = torch.view_as_real(DVR_out_full)
#             DVR = DVR_out_full_real.view(len(DVR_out_full_real), -1)
#
#             y_pred = self.linear(DVR)
#         else:
#             y = self.DVR_NN_v(x_window, coef)
#             y_pred = torch.view_as_real(y)
#         # out = self.linear(DVR)
#         # # out_act = self.act(out)
#         # y_pred = self.linear_2(out)
#
#         return y_pred
#
#
#     # def model_train(self,x,y,model_path,logger=None,total_train = 1,para = [0.001,150,512]):
#     #     learning_rate = para[0]
#     #     epochs = para[1]
#     #     batch_size = para[2]
#     #     device = self.device
#     #
#     #     # optim
#     #     if total_train == 0:
#     #         optimizer = optim.Adam(self.linear.parameters(), lr=learning_rate)
#     #     else:
#     #         optimizer = optim.Adam(self.parameters(), lr=learning_rate)
#     #     criterion = nn.MSELoss()
#     #     # fun.model_structure(self, logger)
#     #     best_metric = float('inf')
#     #
#     #     # dataset
#     #     X_train, X_val, Y_train, Y_val = self.create_dataset(x, y)
#     #     train_dataset = TensorDataset(X_train, Y_train)
#     #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#     #     val_dataset = TensorDataset(X_val, Y_val)
#     #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     #
#     #     # train
#     #     logger.info(f'------------------------Train Stage-------------------------------')
#     #     train_loss_list = []
#     #     val_loss_list = []
#     #     # 训练循环
#     #     start_time = time.time()  # 记录开始时间
#     #     nosave_count = 0
#     #     for epoch in range(epochs):
#     #         self.train()
#     #         i = 0
#     #         train_loss = 0
#     #         for inputs, targets in tqdm(train_loader):
#     #             # 获取对应的复数信号窗口 [batch, M+1]
#     #             optimizer.zero_grad()
#     #
#     #             batch_x_signal = inputs.to(device)
#     #             targets = torch.view_as_real(targets)
#     #             targets = targets.to(device)
#     #             # outputs = model(inputs)
#     #             coef = self.DVR_NN_e(batch_x_signal,targets)
#     #             outputs = self(batch_x_signal,coef)
#     #             # outputs = self(batch_x_signal)
#     #
#     #             loss = criterion(outputs, targets)
#     #             loss.backward()
#     #             optimizer.step()
#     #             train_loss += loss.item()
#     #
#     #         # 验证
#     #         self.eval()
#     #         val_loss = 0
#     #         with torch.no_grad():
#     #             for inputs, targets in tqdm(val_loader):
#     #                 # 将数据移动到设备上
#     #                 # 获取对应的复数信号窗口 [batch, M+1]
#     #                 batch_x_signal = inputs.to(device)
#     #                 targets = torch.view_as_real(targets)
#     #                 targets = targets.to(device)
#     #                 # outputs = model(inputs)
#     #                 coef = self.DVR_NN_e(batch_x_signal, targets,alpha=self.alpha)
#     #                 val_outputs = self(batch_x_signal, coef)
#     #                 # val_outputs = self(batch_x_signal)
#     #                 val_loss += criterion(val_outputs, targets).item()
#     #
#     #         train_loss_list.append(train_loss / len(train_loader))
#     #         val_loss_list.append(val_loss / len(val_loader))
#     #         # 记录指标
#     #         metrics = {
#     #             'epoch': epoch,
#     #             'train_loss': train_loss,
#     #             'val_loss': val_loss,
#     #         }
#     #
#     #         # # 保存当前模型（按周期命名）
#     #         # torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
#     #         save = 0
#     #         # 更新最佳模型
#     #         if val_loss_list[-1] < best_metric:
#     #             best_metric = val_loss_list[-1]
#     #             save = 1
#     #             # 保存模型参数（推荐保存为 .pt 或 .pth 文件）
#     #             best_model = self.state_dict()
#     #         # print(f'Epoch {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f} | save:{save}')
#     #         logger.info(
#     #             f'Epoch {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.7f} | Val Loss: {val_loss / len(val_loader):.7f} | save:{save}')
#     #         if save == 0:
#     #             nosave_count = nosave_count + 1
#     #         else:
#     #             nosave_count = 0
#     #         # if nosave_count > 100:
#     #         #     break
#     #
#     #     torch.save(best_model, model_path)
#     #     logger.info(f"model save: {model_path} ")
#     #     end_time = time.time()  # 记录结束时间
#     #     elapsed_time = end_time - start_time
#     #     logger.info(f"model train time: {elapsed_time:.6f} s")
#     #
#     #
#     #     # if 1:
#     #     #     y_train_loss = train_loss_list  # loss值，即y轴
#     #     #     x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴
#     #     #
#     #     #     plt.figure()
#     #     #
#     #     #     # 去除顶部和右边框框
#     #     #     ax = plt.axes()
#     #     #     ax.spines['top'].set_visible(False)
#     #     #     ax.spines['right'].set_visible(False)
#     #     #
#     #     #     plt.xlabel('iters')  # x轴标签
#     #     #     plt.ylabel('loss')  # y轴标签
#     #     #
#     #     #     # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
#     #     #     # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
#     #     #     plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
#     #     #     plt.legend()
#     #     #     plt.title('Loss curve')
#     #     #     plt.savefig(f'figures/MCP_NN/DVR_NN_loss_curve.png')
#     #     #     plt.close()
#
#     def model_train(self,x,y,model_path,logger=None,total_train = 1,para = [0.001,1000,4096,1e-5,0.75,50]):
#         """para = [learning_rate, epochs, batch_size, alpha, split_ratio, eval_every]
#
#         split_ratio > 0  -> meta-split（批内双流）：LS 只在 batch 的 fit 半段求解，
#                             loss 在 hold 半段计算，梯度只回传给基函数网络；
#         split_ratio <= 0 -> 关闭 meta-split，退回论文 Algorithm 1 的"每批自拟合"基线。
#         注意：开启 meta-split 后 batch_size 必须明显大于基函数列数 p=(6K+1)(M+1)，
#               建议 n_fit = batch_size*split_ratio >= 3p，否则 fit 半段欠定、投影压不下去。
#         """
#         learning_rate = para[0]
#         epochs = para[1]
#         batch_size = para[2]
#         alpha = para[3]
#         if len(para) > 4:
#             self.split_ratio = para[4]
#         if len(para) > 5:
#             self.eval_every = int(para[5])
#         device = self.device
#
#         basis_width = (6 * self.K + 1) * (self.M + 1)
#         n_fit_expected = int(round(batch_size * self.split_ratio)) if self.split_ratio and self.split_ratio > 0 else batch_size
#         if logger is not None:
#             logger.info(f'------------------------Training: meta-split dual-stream LS-------------------------------')
#             logger.info(f'[meta-split] batch={batch_size} split_ratio={self.split_ratio} '
#                         f'-> n_fit~{n_fit_expected} | basis width p~{basis_width} '
#                         f'(K={self.K},M={self.M},term={self.term}) | '
#                         f'fit/eval 样本比~{n_fit_expected / max(basis_width, 1):.2f}')
#             if self.split_ratio and self.split_ratio > 0 and n_fit_expected < 3 * basis_width:
#                 logger.info(f'[meta-split][WARN] n_fit({n_fit_expected}) < 3p({3 * basis_width})，'
#                             f'fit 半段可能欠定：请加大 batch_size 或减小 K/M/term，'
#                             f'否则残差压不下去、梯度会再次变小')
#
#         # optim
#         if total_train == 0:
#             optimizer = optim.Adam(self.linear.parameters(), lr=learning_rate)
#         else:
#             optimizer = optim.Adam(self.parameters(), lr=learning_rate)
#         criterion = nn.MSELoss()
#
#
#         # dataset
#         X_train, X_val, Y_train, Y_val = self.create_dataset(x, y)
#         train_dataset = TensorDataset(X_train, Y_train)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#         val_dataset = TensorDataset(X_val, Y_val)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
#         # train
#         train_loss_list = []
#         val_loss_list = []
#         ortho_loss_list = []
#         best_metric = float('inf')
#         best_model_state = None
#
#         # 训练循环
#         start_time = time.time()  # 记录开始时间
#         nosave_count = 0
#         last_info = {'n_fit': 0, 'n_eval': 0, 'coef_norm': float('nan'),
#                      'cond': float('nan'), 'rank': -1}
#         for epoch in range(epochs):
#             self.train()
#             train_loss_total = 0.0
#             batches = 0
#             for inputs, targets in tqdm(train_loader):
#                 optimizer.zero_grad()
#                 batch_x_signal = inputs.to(device)
#                 loss, info = self.train_step(batch_x_signal, targets, alpha)
#                 loss.backward()
#                 optimizer.step()
#                 train_loss_total += loss.item()
#                 batches += 1
#                 last_info = info
#
#             # 验证：同样走 meta-split，loss 只在 hold 半段计算（不再在 val 上自拟合）
#             self.eval()
#             val_loss_total = 0.0
#             val_batches = 0
#             with torch.no_grad():
#                 for inputs, targets in val_loader:
#                     batch_x_signal = inputs.to(device)
#                     val_loss, _ = self.train_step(batch_x_signal, targets, alpha)
#                     val_loss_total += val_loss.item()
#                     val_batches += 1
#
#             # 计算平均损失
#             avg_train_loss = train_loss_total / max(batches, 1)
#             avg_val_loss = val_loss_total / max(val_batches, 1)
#
#             train_loss_list.append(avg_train_loss)
#             val_loss_list.append(avg_val_loss)
#
#             save = 0
#             # 选模型只用"真实 hold-out 半段"的 loss（train 半段的拟合残差已不再计入 loss）
#             if avg_val_loss < best_metric:
#                 best_metric = avg_val_loss
#                 best_model_state = self.state_dict()
#                 save = 1
#                 torch.save(best_model_state, model_path)
#             logger.info(
#                 f'Epoch {epoch + 1:03} | Train Loss: {avg_train_loss:.7f} | Val Loss: {avg_val_loss:.7f} | save:{save}')
#             if (epoch + 1) % max(int(self.eval_every), 1) == 0 or epoch == 0:
#                 logger.info(
#                     f'    [meta] n_fit={last_info["n_fit"]} n_eval={last_info["n_eval"]} '
#                     f'rank={last_info["rank"]} cond={last_info["cond"]:.3e} '
#                     f'|coef|={last_info["coef_norm"]:.3e} '
#                     f'fit/eval={last_info["n_fit"] / max(last_info["n_eval"], 1):.2f}')
#             if save == 0:
#                 nosave_count = nosave_count + 1
#             else:
#                 nosave_count = 0
#
#         logger.info(f"Final model saved to {model_path}")
#         # logger.info(f"model save: {model_path} ")
#         end_time = time.time()  # 记录结束时间
#         elapsed_time = end_time - start_time
#         logger.info(f"model train time: {elapsed_time:.6f} s")
#
#
#         # if 1:
#         #     y_train_loss = train_loss_list  # loss值，即y轴
#         #     x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴
#         #
#         #     plt.figure()
#         #
#         #     # 去除顶部和右边框框
#         #     ax = plt.axes()
#         #     ax.spines['top'].set_visible(False)
#         #     ax.spines['right'].set_visible(False)
#         #
#         #     plt.xlabel('iters')  # x轴标签
#         #     plt.ylabel('loss')  # y轴标签
#         #
#         #     # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
#         #     # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
#         #     plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
#         #     plt.legend()
#         #     plt.title('Loss curve')
#         #     plt.savefig(f'figures/MCP_NN/DVR_NN_loss_curve.png')
#         #     plt.close()
#
#     def get_basis(self,x_window):
#         """
#         x_window: 当前窗口的记忆输入 [batch, (M+1)]
#         """
#         M = self.M
#         K = self.K
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # threshold
#         # threshold = torch.from_numpy(self.threshold)
#         # threshold_matrix = threshold.unsqueeze(1).t().expand(M + 1, -1)  # (M+1,K) K维度延拓 -1 表示保持K维度不变
#         # threshold_matrix = threshold_matrix.to(device) # x_windows已移动到设备
#         # x(n)
#         xn = x_window[:,-1] #当前信号x(n) (16384,1)
#         xn_amp = torch.abs(xn)  #当前信号|x(n)| (16384,1)
#         xn_amp_matrix = xn_amp.unsqueeze(1).repeat(1, M + 1).unsqueeze(2).repeat(1, 1, K)  # 当前信号|x(n)| (16384,M+1,K)
#         xn_matrix = xn.unsqueeze(1).repeat(1, M + 1).unsqueeze(2).repeat(1, 1, K)  # (16384,(M+1)*K)
#         DVR_xn = xn_matrix.view(len(xn), -1)
#         DVR_xn_amp = xn_amp_matrix.view(len(xn), -1)
#
#         # x(n-i)
#         X_tensor = x_window #记忆信号x(n-i) (16384,M+1)
#         X_matrix = X_tensor.unsqueeze(2).repeat(1, 1, K)  # K维度延拓 x(n-i) (16384,M+1,K)
#         DVR_X = X_matrix.view(len(X_matrix), -1)    # 展平 x(n-i) (16384,(M+1)*K)
#
#         # |x(n-i)| DVR CORE
#         X_amp = torch.abs(X_tensor)  #|x(n-i)| (16384,M+1)
#
#         X = X_amp
#         # for k in range(2,6):
#         #     X_amp_k = X_amp.pow(k)
#         #     X = torch.concat((X,X_amp_k), dim=1)
#
#         # amp_matrix = X_amp.unsqueeze(2).repeat(1, 1, K)  # K维度延拓 |x(n-i)| (16384,M+1,K)
#         # DVR_amp_matrix = amp_matrix - threshold_matrix  # (16384,M+1,K)
#         # ABS_DVR_amp = torch.abs(DVR_amp_matrix)  # (16384,M+1,K)
#         # DVR_core = ABS_DVR_amp.view(len(ABS_DVR_amp), -1)    # (16384,(M+1)*K)
#         # DVR_amp = self.amp_linear(X_amp)    # (16384,(M+1)*K)
#         DVR_amp = self.DVR_layers(X)
#         DVR_core = torch.abs(DVR_amp)  # (16384,(M+1)*K)
#
#         # if self.activation == "ABS":
#         #     DVR_core = torch.abs(DVR_amp)  # (16384,(M+1)*K)
#         # else:
#         #     DVR_core = self.act(DVR_amp)
#
#
#         # theta(n-i) DVR PHASE
#         X_phase = torch.angle(X_tensor)  #theta(n-i) (16384,M+1)
#         e_j_phase = torch.exp(1j * X_phase)  # (16384,M+1)
#         e_j_phase_matrix = e_j_phase.unsqueeze(2).repeat(1, 1, K)  # (16384,M+1,K)
#         # DVR_phase = torch.flatten(e_j_phase_matrix)  # (16384,(M+1)*K)
#         DVR_phase = e_j_phase_matrix.view(len(e_j_phase_matrix), -1)
#
#         # ABS(|x(n)|-beta) DDR CORE
#         # DDR_amp_matrix = xn_amp_matrix - threshold_matrix #(16384,M+1,K)
#         # ABS_DDR_amp = torch.abs(DDR_amp_matrix)     # (16384,M+1,K)
#         # DDR_core = ABS_DDR_amp.view(len(ABS_DDR_amp), -1) # (16384,(M+1)*K)
#         # DDR_amp = self.DDR_linear(xn_amp.unsqueeze(1)) # (16384,(M+1)*K)
#         # # DDR_amp = self.DDR_layers(xn_amp.unsqueeze(1))
#         # if self.activation == "ABS":
#         #     DDR_core = torch.abs(DDR_amp)  # (16384,(M+1)*K)
#         # else:
#         #     DDR_core = self.act(DDR_amp)
#
#
#         # e_Phase_m = DVR_phase
#         e_Phase_n = DVR_xn/DVR_xn_amp #torch.exp(1j * Phase_n)
#         # DDR31 = DVR_xn * DVR_xn * torch.conj(DVR_X)
#         # e_Phase_DDR31 = DDR31/torch.abs(DDR31)
#         # DDR32 = DVR_X * DVR_X * torch.conj(DVR_xn)
#         # e_Phase_DDR32 = DDR32 / torch.abs(DDR32)
#         # DDR51 = DVR_xn * DVR_xn * DVR_xn * torch.conj(DVR_X) * torch.conj(DVR_X)
#         # e_Phase_DDR51 = DDR51 / torch.abs(DDR51)
#         # DDR52 = DVR_X * DVR_X * DVR_X * torch.conj(DVR_xn) * torch.conj(DVR_xn)
#         # e_Phase_DDR52 = DDR52 / torch.abs(DDR52)
#
#
#         # DVR basis
#         DVR_out_linear = X_tensor  # (16384,M+1)
#         DVR_out_1 = DVR_core * DVR_phase  # (16384,(M+1)*K)
#         DVR_out_21 = DVR_core * DVR_phase * DVR_xn_amp  # (16384,(M+1)*K)
#         DVR_out_22 = DVR_core * DVR_xn
#         DVR_out_23 = DVR_core * DVR_X
#         DVR_out_24 = DVR_core * e_Phase_n * torch.abs(DVR_X)
#         DVR_out_DDR_1 = DVR_core * DVR_X * DVR_X * torch.conj(DVR_xn)
#         DVR_out_DDR_2 = DVR_core * DVR_xn * DVR_xn * torch.conj(DVR_X)
#         DVR_out_DDR_51 = DVR_core * DVR_xn * DVR_xn * DVR_xn * torch.conj(DVR_X) * torch.conj(DVR_X)
#         # DVR_core_p = DVR_core.view(len(DVR_core),(M+1)*K,1)
#         # DVR_out_m = DVR_core_p[:,:,0].squeeze() * e_Phase_m
#         # DVR_out_n = DVR_core_p[:,:,1].squeeze() * e_Phase_n
#         # DVR_out_nm = DVR_core * e_Phase_n / e_Phase_nb
#         # DVR_out_DDR31 = DVR_core_p[:,:,2].squeeze() * e_Phase_DDR31
#         # DVR_out_DDR32 = DVR_core_p[:, :, 3].squeeze() * e_Phase_DDR32
#         # DVR_out_DDR32 = DVR_core * e_Phase_DDR32
#         # DVR_out_DDR51 = DVR_core_p[:,:,3].squeeze() * e_Phase_DDR51
#         # DVR_out_DDR52 = DVR_core * e_Phase_DDR52
#
#         # DVR_out_21,DVR_out_22,DVR_out_23,DVR_out_24,DVR_out_21,DVR_out_22,DVR_out_23,DVR_out_24,DVR_out_DDR_2
#         if self.term == 7:
#             DVR_out_full = torch.concatenate(( DVR_out_1,DVR_out_linear,DVR_out_21,DVR_out_22,DVR_out_23,DVR_out_24,DVR_out_DDR_2,DVR_out_DDR_51), dim=1) #  DVR_out_linear,DVR_out_1, DVR_out_21,DVR_out_22, DVR_out_23, DVR_out_DDR_1,DVR_out_DDR_2
#         elif self.term == 6:
#             DVR_out_full = torch.concatenate((DVR_out_1, DVR_out_linear, DVR_out_21, DVR_out_22, DVR_out_23, DVR_out_24,DVR_out_DDR_2), dim=1)
#         elif self.term == 5:
#             DVR_out_full = torch.concatenate((DVR_out_1, DVR_out_linear, DVR_out_21, DVR_out_22, DVR_out_23, DVR_out_24), dim=1)
#         elif self.term == 4:
#             DVR_out_full = torch.concatenate((DVR_out_1, DVR_out_linear, DVR_out_21, DVR_out_22, DVR_out_23), dim=1)
#         elif self.term == 3:
#             DVR_out_full = torch.concatenate((DVR_out_1, DVR_out_linear, DVR_out_21, DVR_out_22), dim=1)
#         elif self.term == 2:
#             DVR_out_full = torch.concatenate((DVR_out_1, DVR_out_linear, DVR_out_21), dim=1)
#         elif self.term == 1:
#             DVR_out_full = torch.concatenate((DVR_out_1, DVR_out_linear), dim=1)
#         elif self.term == 0:
#             DVR_out_full = DVR_out_1
#
#         return DVR_out_full
#
#     def _split_batch(self, n):
#         """把一个 batch 随机切成"拟合半段 / 评估半段"两部分索引。
#         返回 (fit_idx, eval_idx)；关闭 meta-split 时 eval_idx 为空张量。"""
#         device = self.device
#         if self.split_ratio is None or self.split_ratio <= 0:
#             return (torch.arange(n, device=device), None)
#         r = float(self.split_ratio)
#         r = min(max(r, 1e-3), 1.0 - 1e-3)
#         n_fit = int(round(n * r))
#         n_fit = min(max(n_fit, 1), n - 1)
#         perm = torch.randperm(n, device=device)
#         return (perm[:n_fit], perm[n_fit:])
#
#     def _fit_basis(self, X_fit, y_fit, alpha):
#         """在拟合半段上做 LS 提参（detach，不走 autograd）。
#         X_fit: 基函数 (n_fit, p) complex；y_fit: (n_fit,) complex
#         返回 (coef_cpu, info)；coef 为复张量 (p,)，info 为诊断字典。
#         说明：在 batch_size >= 1024 且 n_fit >> p 时，特征基在 fit 半段上接近满秩/良态，
#         自实现的 LS 与 pinv 等价且更快；只有在 n_fit 相对量级很小（<=32）时才退回 pinv，
#         以防下游调用者传入极小的 fit 窗口。"""
#         Xc = X_fit.detach().cpu().numpy()
#         yc = y_fit.detach().cpu().numpy()
#         n_fit = Xc.shape[0]
#         p = Xc.shape[1]
#         if n_fit <= 32:
#             reg = max(float(alpha), 1e-12) * np.eye(p)
#             coef_np = np.linalg.pinv(Xc.conj().T.dot(Xc) + reg).dot(Xc.conj().T).dot(yc)
#             info = {'n_fit': n_fit, 'p': p, 'cond': float('nan'), 'rank': int(p)}
#             return torch.from_numpy(coef_np).to(self.device), info
#         G = Xc.conj().T.dot(Xc)                      # (p, p) Hermitian Gram
#         scale = np.trace(G).real / max(p, 1)         # 基函数列的平均能量
#         # LS 正则的绝对尺度与信号/基函数量纲绑定，避免手调 alpha 随 K、M、batch 漂移
#         ridge = max(float(alpha), 0.0) * scale
#         U, s, Vh = np.linalg.svd(G, full_matrices=False)
#         rank = int(self._numerical_rank(s))
#         w = s / (s ** 2 + ridge)                      # (G + ridge*I)^-1 的特征值
#         coef_np = (Vh.conj().T * w).dot(U.conj().T).dot(Xc.conj().T).dot(yc)
#         info = {'n_fit': n_fit, 'p': p,
#                 'cond': float(max(s) / max(s[-1], 1e-300)),
#                 'rank': rank, 'ridge': float(ridge)}
#         return torch.from_numpy(coef_np).to(self.device), info
#
#     @staticmethod
#     def _numerical_rank(s, rel_tol=1e-10):
#         """按最大奇异值的相对阈值估计有效秩（n_fit >> p 时为 p）。"""
#         s = np.asarray(s).ravel()
#         if s.size == 0:
#             return 0
#         smax = float(s[0]) if s[0] >= s[-1] else float(s[-1])
#         if not np.isfinite(smax) or smax <= 0:
#             return 0
#         return int((s > smax * rel_tol).sum())
#
#     def DVR_NN_e(self, x_window, y=None, alpha=1e-3, pri=0, fit_window=None, y_fit=None, fit_basis=None):
#         """LS 提参。
#
#         兼容三种调用方式（保旧调用方不受影响）：
#           DVR_NN_e(x, y)                  -> 在 x 上拟合，等价于旧行为
#           DVR_NN_e(x)                     -> 纯评估（不需要标签），只用基函数
#           DVR_NN_e(x_los, y_los, fit_window=x_fit, y_fit=y_fit)
#                                           -> meta-split：系数只在 fit 半段上解，评估用 x_los
#         fit_basis 可传入预先算好的 (detach) 基函数矩阵，避免重复前向。
#         返回 coef（complex，p 维）。info 及 self._meta_info / self._meta_coef 供调用方记录。
#         """
#         if fit_window is None:
#             fit_window = x_window
#         if y_fit is None:
#             y_fit = y
#         self._meta_info = None
#         self._meta_coef = None
#         basis = self.get_basis(x_window)
#         if y_fit is None:
#             # 纯评估：基函数值仍要走 autograd（meta-split 的外层需要梯度），但没有系数可解
#             self._meta_coef = basis
#             return basis
#         y_fit = torch.as_tensor(y_fit, device=self.device)
#         if y_fit.dim() != 1 or fit_window.shape[0] != y_fit.shape[0]:
#             raise ValueError(
#                 f"DVR_NN_e: fit_window 与 y_fit 行数不一致 "
#                 f"({tuple(fit_window.shape)} vs {tuple(y_fit.shape)})；"
#                 f"y 需为 1-D 复数向量 (n,)")
#         if fit_basis is None:
#             fit_basis = self.get_basis(fit_window) if fit_window is not x_window else basis
#         coef, info = self._fit_basis(fit_basis, y_fit, alpha)
#         self._meta_info = info
#         self._meta_coef = coef
#         self.coef = coef                      # 兼容旧属性访问
#         if pri == 1:
#             y_model = basis @ coef
#             NMSE = cal.nmse(y_fit.detach().cpu().numpy(),
#                             y_model.detach().cpu().numpy())
#             print(f"train NMSE: {NMSE:.3f} dB  (meta-split: LS on {info['n_fit']} samples, basis {info['p']})")
#         return coef
#
#     def train_step(self, batch_x_signal, targets, alpha):
#         """一个 meta-split 训练步：随机切分 → fit 半段 LS 提参(detach) → eval 半段算 loss → 反传。
#         返回 (loss, train_loss_dict)。关闭 meta-split 时退化为旧的"本批自拟合"。"""
#         n = batch_x_signal.shape[0]
#         fit_idx, eval_idx = self._split_batch(n)
#         device = self.device
#         targets = targets.to(device)
#         info = {'n_fit': int(fit_idx.numel()),
#                 'n_eval': int(eval_idx.numel()) if eval_idx is not None else 0,
#                 'coef_norm': float('nan'), 'cond': float('nan'), 'rank': -1}
#
#         if eval_idx is None or eval_idx.numel() == 0:
#             # 旧行为基线：同一批自拟合（保留以便对比）
#             x_fit = batch_x_signal
#             y_fit = targets
#             coef = self.DVR_NN_e(x_fit, y_fit, alpha)
#             x_los = batch_x_signal
#             y_los = targets
#         else:
#             x_fit = batch_x_signal[fit_idx]
#             y_fit = targets[fit_idx]
#             x_los = batch_x_signal[eval_idx]
#             y_los = targets[eval_idx]
#             # 注意：这里的 detach 只作用于 LS 提参路径；loss 路径用的 basis_los 是带梯度的
#             fit_basis = self.get_basis(x_fit).detach()
#             coef = self.DVR_NN_e(x_los, None, fit_window=x_fit, y_fit=y_fit,
#                                  alpha=alpha, fit_basis=fit_basis)
#             # y_los 用不到（loss 走复基函数直接算），保留索引以对齐语义
#
#         if self._meta_info is not None:
#             for k in ('cond', 'rank'):
#                 info[k] = self._meta_info[k] if self._meta_info[k] == self._meta_info[k] else float('nan')
#             if self._meta_coef is not None:
#                 info['coef_norm'] = float(self._meta_coef.detach().abs().norm())
#
#         basis_los = self.get_basis(x_los)                    # complex (n_eval, p)
#         y_pred = basis_los @ coef                            # coef 已 detach，梯度只走基函数
#         loss = self._complex_mse(y_pred, y_los)
#         return loss, info
#
#     @staticmethod
#     def _complex_mse(y_pred_complex, y_true_complex):
#         """复基函数输出的 MSE，用 view_as_real 保证与旧 criterion(outputs, targets_real) 同尺度。"""
#         pred = torch.view_as_real(y_pred_complex.contiguous()).reshape(-1)
#         true = torch.view_as_real(y_true_complex.contiguous()).reshape(-1)
#         return torch.mean((pred - true) ** 2)
#
#     # def DVR_NN_e(self, x_window, y, alpha = 1e-3, pri = 0):
#     #     ---- 旧的"本批自拟合"实现已由上方 meta-split 版本取代（split_ratio<=0 可复现旧行为）----
#     #     basis = self.get_basis(x_window)
#     #     X = basis
#     #     np_X = X.cpu().detach().numpy()
#     #     np_y = y.cpu().detach().numpy()
#     #     np_XH = np.conjugate(np_X.T)
#     #     np_coef = np.linalg.pinv(np_XH.dot(np_X) + alpha * np.eye(np_X.shape[1])).dot(np_XH).dot(np_y)
#     #     coef = torch.from_numpy(np_coef).to(device)
#     #     self.coef = coef
#     #     return coef
#
#
#     # def DVR_NN_v(self, x, coef):
#     #     M = self.M
#     #     K = self.K
#     #     device = self.device
#     #     sequences = volterra_nn.create_memory_seq(x, M)  # [N, M+1]
#     #     x_window = torch.from_numpy(sequences).to(device)
#     #
#     #     X = self.get_basis(x_window).cpu().detach().numpy()
#     #     y = X.dot(coef)
#     #     return y
#     def DVR_NN_v(self, x_window, coef):
#         """使用提取的系数进行预测（冻结基函数网络）"""
#         # self.eval()  # 设置为评估模式
#         M = self.M
#         K = self.K
#         device = self.device
#         # sequences = volterra_nn.create_memory_seq(x, M)  # [N, M+1]
#         # x_window = torch.from_numpy(sequences).to(device)
#
#         # 生成正交基函数
#         # with torch.no_grad():
#         basis = self.get_basis(x_window)
#         X = basis
#         # X = basis.cpu().detach().numpy()
#         # X = self.get_basis(x_window).cpu().detach().numpy()
#         y = X @ coef
#         return y
#
#     # 数据预处理函数
#     def create_dataset(self,x, y, test_size=0.2):
#         M = self.M
#         # 转换为实部虚部分离格式
#
#         sequences = volterra_nn.create_memory_seq(x,M)  # [N, M+1]
#
#         X_tensor = torch.from_numpy(sequences)
#         # targets = volterra_nn.complex_to_real(y)    # [N, 1]
#         targets = y
#
#         Y_tensor = torch.from_numpy(y) # (N, 2)
#         # Y_tensor = torch.view_as_complex(Y_tensor)
#         X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=test_size, shuffle=False)
#         return [X_train, X_val, Y_train, Y_val]
#
#     def apply_dpd(self, signal, coef=None):
#         M = self.M
#         # threshold = self.threshold
#         K = self.K
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # 将模型移动到设备上
#         self.to(device)
#         self.eval()
#         sequences = volterra_nn.create_memory_seq(signal, M)  # [N, M+1]
#         x = torch.from_numpy(sequences).to(device)
#         out = self(x,coef)
#         out_complex = torch.view_as_complex(out).cpu()
#         y_pred = out_complex.detach().numpy()
#
#         # y_pred = np.stack()
#         return y_pred
#     # def train(self):
#     #     self.train()
#
# class DVR_2(nn.Module):
#     def __init__(self, K, M, threshold, activation="ReLU"):
#         super().__init__()
#         self.M = M
#         self.K = K
#         self.threshold = threshold
#         assert activation == "ReLU" or "Tanh" or "ELU" or "None"
#         self.linear = nn.Linear((K*(M+1)*6+M+1)*2, 2,bias=False).double()
#         # self.act = nn.Tanh()
#         # self.linear_2 = nn.Linear(12, 2, bias=False).double()
#         # self.main_layers = nn.Sequential()
#         # self.para_layers = nn.Sequential()
#         # self.main_nn = volterra_nn.GMP_NN(self.layer_dim1, L, M = self.M,activation=activation)
#         # self.para_nn = volterra_nn.MCP_BASE_NN(self.layer_dim2, L, M = self.M, K = 5,activation=activation)
#         # self.out_layer = nn.Linear(2*M+2, 2,bias=False)
#
#     def forward(self, x_window):
#         """
#         x_window: 当前窗口的记忆输入 [batch, (M+1)]
#         """
#         M = self.M
#         K = self.K
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # threshold
#         threshold = torch.from_numpy(self.threshold)
#         threshold_matrix = threshold.unsqueeze(1).t().expand(M + 1, -1)  # (M+1,K) K维度延拓 -1 表示保持K维度不变
#         threshold_matrix = threshold_matrix.to(device) # x_windows已移动到设备
#         # x(n)
#         xn = x_window[:,-1] #当前信号x(n) (16384,1)
#         xn_amp = torch.abs(xn)  #当前信号|x(n)| (16384,1)
#         xn_amp_matrix = xn_amp.unsqueeze(1).repeat(1, M + 1).unsqueeze(2).repeat(1, 1, K)  # 当前信号|x(n)| (16384,M+1,K)
#         xn_matrix = xn.unsqueeze(1).repeat(1, M + 1).unsqueeze(2).repeat(1, 1, K)  # (16384,(M+1)*K)
#         DVR_xn = xn_matrix.view(len(xn), -1)
#         DVR_xn_amp = xn_amp_matrix.view(len(xn), -1)
#         # x(n-i)
#         X_tensor = x_window #记忆信号x(n-i) (16384,M+1)
#         X_matrix = X_tensor.unsqueeze(2).repeat(1, 1, K)  # K维度延拓 x(n-i) (16384,M+1,K)
#         DVR_X = X_matrix.view(len(X_matrix), -1)    # 展平 x(n-i) (16384,(M+1)*K)
#         # |x(n-i)| DVR CORE
#         X_amp = torch.abs(X_tensor)  #|x(n-i)| (16384,M+1)
#         amp_matrix = X_amp.unsqueeze(2).repeat(1, 1, K)  # K维度延拓 |x(n-i)| (16384,M+1,K)
#         DVR_amp_matrix = amp_matrix - threshold_matrix  # (16384,M+1,K)
#         ABS_DVR_amp = torch.abs(DVR_amp_matrix)  # (16384,M+1,K)
#         DVR_core = ABS_DVR_amp.view(len(ABS_DVR_amp), -1)    # (16384,(M+1)*K)
#         # theta(n-i) DVR PHASE
#         X_phase = torch.angle(X_tensor)  #theta(n-i) (16384,M+1)
#         e_j_phase = torch.exp(1j * X_phase)  # (16384,M+1)
#         e_j_phase_matrix = e_j_phase.unsqueeze(2).repeat(1, 1, K)  # (16384,M+1,K)
#         # DVR_phase = torch.flatten(e_j_phase_matrix)  # (16384,(M+1)*K)
#         DVR_phase = e_j_phase_matrix.view(len(e_j_phase_matrix), -1)
#         # ABS(|x(n)|-beta) DDR CORE
#         DDR_amp_matrix = xn_amp_matrix - threshold_matrix #(16384,M+1,K)
#         ABS_DDR_amp = torch.abs(DDR_amp_matrix)     # (16384,M+1,K)
#         DDR_core = ABS_DDR_amp.view(len(ABS_DDR_amp), -1) # (16384,(M+1)*K)
#
#         # DVR basis
#         DVR_out_linear = X_tensor  # (16384,M+1)
#         DVR_out_1 = DVR_core * DVR_phase  # (16384,(M+1)*K)
#         DVR_out_21 = DVR_core * DVR_phase * DVR_xn_amp  # (16384,(M+1)*K)
#         DVR_out_22 = DVR_core * DVR_xn
#         DVR_out_23 = DVR_core * DVR_X
#         DVR_out_DDR_1 = DDR_core * DVR_X
#         DVR_out_DDR_2 = DDR_core * DVR_xn * DVR_xn * torch.conj(DVR_X)
#
#         DVR_out_full = torch.concatenate((DVR_out_linear, DVR_out_1, DVR_out_21, DVR_out_22, DVR_out_23, DVR_out_DDR_1,DVR_out_DDR_2), dim=1)
#         DVR_out_full_real = torch.view_as_real(DVR_out_full)
#         DVR = DVR_out_full_real.view(len(DVR_out_full_real), -1)
#
#         y_pred = self.linear(DVR)
#         # out = self.linear(DVR)
#         # out_act = self.act(out)
#         # y_pred = self.linear_2(out_act)
#
#         return y_pred
#
#
#     # 数据预处理函数
#     def create_dataset(self,x, y, test_size=0.2):
#         M = self.M
#         # 转换为实部虚部分离格式
#         # X = volterra_nn.complex_to_real(x)
#         # Y = volterra_nn.complex_to_real(y)
#
#         sequences = volterra_nn.create_memory_seq(x,M)  # [N, M+1]
#
#         X_tensor = torch.from_numpy(sequences)
#         # targets = volterra_nn.complex_to_real(y)    # [N, 1]
#         targets = y
#
#         Y_tensor = torch.from_numpy(y) # (N, 2)
#         # Y_tensor = torch.view_as_complex(Y_tensor)
#         X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=test_size, shuffle=False)
#         return [X_train, X_val, Y_train, Y_val]
#
#     def apply_dpd(self, signal):
#         M = self.M
#         threshold = self.threshold
#         K = self.K
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # 将模型移动到设备上
#         self.to(device)
#         self.eval()
#         sequences = volterra_nn.create_memory_seq(signal, M)  # [N, M+1]
#         x = torch.from_numpy(sequences).to(device)
#
#         out = self(x)
#         out_complex = torch.view_as_complex(out).cpu()
#         y_pred = out_complex.detach().numpy()
#         # y_pred = np.stack()
#         return y_pred
#     # def train(self):
#     #     self.train()
