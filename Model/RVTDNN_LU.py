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


class RVTDNN_LU(nn.Module):
    def __init__(self, layer_dims, M, K=1, activation="LeakyReLU"):
        super().__init__()
        self.name = 'RVTDNN'
        self.M = M
        self.K = K
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 只使用可逆激活函数
        assert activation in ["LeakyReLU", "Tanh", "ELU", "Sigmoid"]
        self.activation = activation

        # 存储每一层的权重和偏置
        self.linear_layers = nn.ModuleList()
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.linear_layers.append(nn.Linear(in_dim, out_dim))

        # 定义可逆激活函数
        if activation == "LeakyReLU":
            self.act = nn.LeakyReLU(negative_slope=0.01)
        elif activation == "Tanh":
            self.act = nn.Tanh()
        elif activation == "ELU":
            self.act = nn.ELU(alpha=1.0)
        elif activation == "Sigmoid":
            self.act = nn.Sigmoid()

    def forward(self, x_window):
        """
        x_window: 当前窗口的记忆输入 [batch, (M+1)]
        """
        X = x_window
        x_real = X.real
        x_imag = X.imag
        X_abs = torch.abs(x_window)

        for i in range(2, self.K + 1):
            X_k = torch.abs(x_window).pow(i)
            X_abs = torch.concat((X_abs, X_k), dim=1)

        x = torch.cat((x_real, x_imag), dim=1)

        # 逐层前向传播
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            if i < len(self.linear_layers) - 1:  # 不是最后一层
                x = self.act(x)

        return x

    def activation_inverse(self, y):
        """逆激活函数"""
        if self.activation == "LeakyReLU":
            # LeakyReLU: f(x) = x if x >= 0 else alpha * x
            # 逆: f^{-1}(y) = y if y >= 0 else y / alpha
            alpha = 0.01
            return torch.where(y >= 0, y, y / alpha)

        elif self.activation == "Tanh":
            # Tanh: f(x) = tanh(x)
            # 逆: f^{-1}(y) = atanh(y)
            # 需要限制y的范围避免数值问题
            eps = 1e-7
            y_clamped = torch.clamp(y, -1 + eps, 1 - eps)
            return 0.5 * torch.log((1 + y_clamped) / (1 - y_clamped))

        elif self.activation == "ELU":
            # ELU: f(x) = x if x >= 0 else alpha * (exp(x) - 1)
            # 逆: f^{-1}(y) = y if y >= 0 else log(1 + y/alpha)
            alpha = 1.0
            return torch.where(y >= 0, y, torch.log(1 + y / alpha))

        elif self.activation == "Sigmoid":
            # Sigmoid: f(x) = 1 / (1 + exp(-x))
            # 逆: f^{-1}(y) = log(y / (1 - y))
            eps = 1e-7
            y_clamped = torch.clamp(y, eps, 1 - eps)
            return torch.log(y_clamped / (1 - y_clamped))

        else:
            return y  # 恒等函数

    def layerwise_linear_train(self, x_window, y_target, iterations=50, reg_lambda=1e-6):
        """
        逐层线性化训练方法
        x_window: 输入数据 [batch, M+1]
        y_target: 目标输出 [batch, 2] (实部和虚部)
        iterations: 交替优化迭代次数
        reg_lambda: 正则化参数
        """
        device = self.device

        # 准备输入特征
        X = x_window
        x_real = X.real
        x_imag = X.imag
        X_abs = torch.abs(x_window)

        # for i in range(2, self.K + 1):
        #     X_k = torch.abs(x_window).pow(i)
        #     X_abs = torch.concat((X_abs, X_k), dim=1)

        x_input = torch.cat((x_real, x_imag), dim=1).to(device)
        y_target = y_target.to(device)

        losses = []

        for iteration in range(iterations):
            # 前向传播，记录每一层的输入和输出
            current = x_input
            layer_linear_outputs = []
            layer_activation_outputs = []

            # 前向传播并记录中间结果
            for i, layer in enumerate(self.linear_layers):
                linear_out = layer(current)
                layer_linear_outputs.append(linear_out)

                if i < len(self.linear_layers) - 1:
                    activation_out = self.act(linear_out)
                    layer_activation_outputs.append(activation_out)
                    current = activation_out
                else:
                    layer_activation_outputs.append(linear_out)  # 最后一层没有激活函数
                    current = linear_out

            # 计算当前损失
            final_output = layer_activation_outputs[-1]
            loss = torch.mean((final_output - y_target) ** 2)
            losses.append(loss.item())

            print(f"Iteration {iteration}: Loss = {loss.item():.6f}")

            # 从后往前逐层优化
            for layer_idx in range(len(self.linear_layers) - 1, -1, -1):
                if layer_idx == len(self.linear_layers) - 1:
                    # 最后一层（输出层）：直接优化
                    # 目标：Y ≈ W * H + b
                    H = layer_activation_outputs[layer_idx - 1] if layer_idx > 0 else x_input

                    # 添加偏置项
                    ones = torch.ones(H.shape[0], 1, device=device)
                    H_aug = torch.cat([H, ones], dim=1)

                    # 岭回归解（正则化最小二乘）
                    W = torch.linalg.lstsq(
                        H_aug.T @ H_aug + reg_lambda * torch.eye(H_aug.shape[1], device=device),
                        H_aug.T @ y_target
                    ).solution.T

                    # 更新权重和偏置
                    self.linear_layers[layer_idx].weight.data = W[:, :-1]
                    self.linear_layers[layer_idx].bias.data = W[:, -1]

                else:
                    # 隐藏层：通过下一层反推目标值
                    next_layer = self.linear_layers[layer_idx + 1]

                    if layer_idx + 1 == len(self.linear_layers) - 1:
                        # 如果下一层是输出层，使用输出目标反推
                        # 解方程：Y = W_next * H_current + b_next
                        # 所以：H_current ≈ pinv(W_next) * (Y - b_next)

                        # 减去偏置
                        Y_target_next = y_target - next_layer.bias.unsqueeze(0)

                        # 使用伪逆求解 H_current
                        # 注意：我们需要求解 W_next @ H_current.T ≈ Y_target_next.T
                        # 所以 H_current.T = pinv(W_next) @ Y_target_next.T

                        # 计算 W_next 的伪逆
                        W_next_pinv = torch.linalg.pinv(next_layer.weight)

                        # 计算目标激活输出
                        H_target = (W_next_pinv @ Y_target_next.T).T

                    else:
                        # 如果下一层是隐藏层，使用下一层的线性输出作为目标
                        # 通过逆激活函数反推
                        Z_next_target = layer_linear_outputs[layer_idx + 1]
                        H_target = self.activation_inverse(Z_next_target)

                    # 通过逆激活函数得到当前层的目标线性输出
                    # 注意：H_target 应该是当前层的激活输出目标
                    # 所以当前层的线性输出目标应该是逆激活函数的输出
                    Z_target = self.activation_inverse(H_target)

                    # 当前层的输入
                    H_prev = layer_activation_outputs[layer_idx - 1] if layer_idx > 0 else x_input

                    # 添加偏置项
                    ones = torch.ones(H_prev.shape[0], 1, device=device)
                    H_prev_aug = torch.cat([H_prev, ones], dim=1)

                    # 岭回归解
                    W = torch.linalg.lstsq(
                        H_prev_aug.T @ H_prev_aug + reg_lambda * torch.eye(H_prev_aug.shape[1], device=device),
                        H_prev_aug.T @ Z_target
                    ).solution.T

                    # 更新权重和偏置
                    self.linear_layers[layer_idx].weight.data = W[:, :-1]
                    self.linear_layers[layer_idx].bias.data = W[:, -1]

        return losses

    def model_train(self, x, y, model_path, logger=None, para=[0.001, 150, 512], method="layerwise"):
        """
        训练函数，支持两种训练方法
        method: "gradient" - 梯度下降, "layerwise" - 逐层线性化
        """
        learning_rate = para[0]
        epochs = para[1]
        batch_size = para[2]

        if method == "gradient":
            return self._train_gradient_descent(x, y, model_path, logger, para)
        elif method == "layerwise":
            return self._train_layerwise(x, y, model_path, logger, para)
        else:
            raise ValueError("method must be 'gradient' or 'layerwise'")

    def _train_gradient_descent(self, x, y, model_path, logger=None, para=[0.001, 150, 512]):
        """传统的梯度下降训练"""
        learning_rate = para[0]
        epochs = para[1]
        batch_size = para[2]
        device = self.device

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        best_metric = float('inf')

        X_train, X_val, Y_train, Y_val = self.create_dataset(x, y)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = TensorDataset(X_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f'------------------------Gradient Descent Training-------------------------------')
        train_loss_list = []
        val_loss_list = []

        start_time = time.time()
        nosave_count = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0

            for inputs, targets in tqdm(train_loader):
                optimizer.zero_grad()
                batch_x_signal = inputs.to(device)
                targets = torch.view_as_real(targets).to(device)
                outputs = self(batch_x_signal)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader):
                    batch_x_signal = inputs.to(device)
                    targets = torch.view_as_real(targets).to(device)
                    val_outputs = self(batch_x_signal)
                    val_loss += criterion(val_outputs, targets).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            train_loss_list.append(avg_train_loss)
            val_loss_list.append(avg_val_loss)

            save = 0
            if avg_val_loss < best_metric:
                best_metric = avg_val_loss
                save = 1
                best_model = self.state_dict()

            logger.info(
                f'Epoch {epoch + 1:03} | Train Loss: {avg_train_loss:.7f} | Val Loss: {avg_val_loss:.7f} | save:{save}')

            if save == 0:
                nosave_count += 1
            else:
                nosave_count = 0

            if nosave_count > 100:
                break

        torch.save(best_model, model_path)
        logger.info(f"model save: {model_path}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"model train time: {elapsed_time:.6f} s")

        return best_metric

    def _train_layerwise(self, x, y, model_path, logger=None, para=[0.001, 150, 512]):
        """逐层线性化训练"""
        iterations = para[1]  # 使用epochs参数作为迭代次数
        device = self.device

        logger.info(f'------------------------Layerwise Linear Training-------------------------------')

        # 创建完整数据集（逐层优化通常在全数据集上进行）
        X_full, _, Y_full, _ = self.create_dataset(x, y)
        X_full = X_full.to(device)
        Y_full = Y_full.to(device)

        # 将目标转换为实数形式
        Y_full_real = torch.view_as_real(Y_full).view(Y_full.shape[0], -1)

        start_time = time.time()

        # 执行逐层线性化训练
        losses = self.layerwise_linear_train(X_full, Y_full_real, iterations=iterations)

        # 保存模型
        torch.save(self.state_dict(), model_path)
        logger.info(f"model save: {model_path}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"layerwise train time: {elapsed_time:.6f} s")
        logger.info(f"Final loss: {losses[-1]:.7f}")

        return losses[-1]

    # 数据预处理函数
    def create_dataset(self, x, y, test_size=0.2):
        M = self.M
        sequences = fun.create_memory_seq(x, M)  # [N, M+1]
        X_tensor = torch.from_numpy(sequences)
        Y_tensor = torch.from_numpy(y)
        X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=test_size, shuffle=False)
        return [X_train, X_val, Y_train, Y_val]

    def apply_dpd(self, signal, coef=None):
        M = self.M
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        sequences = fun.create_memory_seq(signal, M)
        x = torch.from_numpy(sequences).to(device)
        out = self(x)
        out_complex = torch.view_as_complex(out).cpu()
        y_pred = out_complex.detach().numpy()
        return y_pred