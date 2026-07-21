import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time
from function import Function_Calculate as cal
from tqdm import tqdm
from matplotlib import pyplot as plt
import Model.volterra_nn as volterra_nn


class AKPTDNN(nn.Module):
    """
    Advanced Kronecker Product-Based Time-Delay Neural Network (AKPTDNN)
    
    Reference:
        J. Shao, X. Hong, and W. Wang,
        "An Advanced Kronecker Product-Based Time-Delay Neural Network
         for Radio Frequency Power Amplifier Digital Predistortion,"
        IEEE Trans. Microw. Theory Techn., 2025.
    
    Architecture (Fig. 2 of paper):
        1. Input s = [Re[x(n)], Im[x(n)], ..., Re[x(n-M+1)], Im[x(n-M+1)]]  (2M)
        2. Linear Layer 1: 2M -> 2L (no activation)
        3. VDM: |z_hat_l| = sqrt(z_{2l-1}^2 + z_{2l}^2) for l=1..L
        4. For each p = 1..P:
           a. |z_hat|^p -> Linear 2.p (L->L) + ReLU -> h_p
           b. KP: y'_p = s (X) h_p
        5. Concatenate [s, y'_1, ..., y'_P] -> Linear Layer 3 -> [yi, yq]
    
    Complexity constraint: P * L = constant (default 24)
    """
    def __init__(self, M_taps, L, P, activation="ReLU"):
        """
        Args:
            M_taps: Number of memory taps (paper's M, input dim = 2*M_taps).
                    In codebase convention, M_taps = model_M + 1 from create_memory_seq.
            L: Output dimension of Linear Layer 1 (controls complexity)
            P: Maximum modulus order (number of parallel branches)
            activation: Activation for Linear Layers 2.p (default: ReLU per paper)
        """
        super().__init__()
        self.name = 'AKPTDNN'
        self.M_taps = M_taps
        self.L = L
        self.P = P
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert activation in ["ReLU", "Tanh", "GELU", "Sigmoid"], \
            f"Unsupported activation: {activation}"
        self.activation = activation

        # Input dimension: 2 * M_taps (real + imag per tap)
        input_dim = 2 * M_taps

        # Linear Layer 1: 2*M_taps -> 2*L, no activation
        self.linear1 = nn.Linear(input_dim, 2 * L).double()

        # Linear Layers 2.p: L -> L with activation, P parallel branches
        self.modulus_layers = nn.ModuleList()
        for _ in range(1, P + 1):
            layer_seq = nn.Sequential()
            layer_seq.add_module("linear", nn.Linear(L, L).double())
            if activation == "ReLU":
                layer_seq.add_module("act", nn.ReLU())
            elif activation == "Tanh":
                layer_seq.add_module("act", nn.Tanh())
            elif activation == "GELU":
                layer_seq.add_module("act", nn.GELU())
            elif activation == "Sigmoid":
                layer_seq.add_module("act", nn.Sigmoid())
            self.modulus_layers.append(layer_seq)

        # Final Linear Layer 3: (2*M_taps + P * 2*M_taps * L) -> 2
        final_input_dim = 2 * M_taps + P * (2 * M_taps * L)
        self.linear3 = nn.Linear(final_input_dim, 2).double()

    def forward(self, x_window):
        """
        Forward pass of AKPTDNN.
        
        Args:
            x_window: Complex memory windows [batch, M_taps] 
                      where x_window[:, -1] = x(n) (newest)
        
        Returns:
            output: [batch, 2] real-valued I/Q prediction
        """
        batch_size = x_window.shape[0]

        # Step 1: Convert complex windows to paper's s format
        # x_window[b, :] = [x(n-M_taps+1), ..., x(n-1), x(n)]
        # Paper: s = [Re[x(n)], Im[x(n)], ..., Re[x(n-M_taps+1)], Im[x(n-M_taps+1)]]
        # Reverse time dim so newest comes first
        x_rev = torch.flip(x_window, dims=[1])  # [batch, M_taps], [x(n), x(n-1), ..., x(n-M_taps+1)]

        # Interleave real/imag: [batch, M_taps, 2] -> [batch, 2*M_taps]
        x_real = torch.view_as_real(x_rev)  # [batch, M_taps, 2]
        s = x_real.reshape(batch_size, -1)  # [batch, 2*M_taps]

        # Step 2: Linear Layer 1 (no activation)
        z = self.linear1(s)  # [batch, 2*L]

        # Step 3: VDM - compute modulus for each consecutive pair
        # z contains 2L values, pair as (z_{2l-1}, z_{2l}) -> L modulus values
        z_pairs = z.reshape(batch_size, self.L, 2)  # [batch, L, 2]
        z_mod = torch.norm(z_pairs, dim=2)           # [batch, L], |z_hat_l|

        # Step 4: KP operations for each order p = 1..P
        kp_outputs = []

        for p in range(1, self.P + 1):
            # |z_hat|^p (element-wise power)
            z_mod_p = z_mod ** p  # [batch, L]

            # Linear Layer 2.p with activation
            h_p = self.modulus_layers[p - 1](z_mod_p)  # [batch, L]

            # KP: y'_p = s (X) h_p
            # s: [batch, 2*M_taps], h_p: [batch, L]
            # y'_p[i, j] = s[i] * h_p[j]  -> flattened: [batch, 2*M_taps * L]
            s_unsq = s.unsqueeze(2)                     # [batch, 2*M_taps, 1]
            h_unsq = h_p.unsqueeze(1)                   # [batch, 1, L]
            y_p_kp = (s_unsq * h_unsq).reshape(batch_size, -1)  # [batch, 2*M_taps*L]

            kp_outputs.append(y_p_kp)

        # Step 5: Concatenate s (y'_0) with all y'_p
        # s serves as y'_0 (zero-order term)
        combined = torch.cat([s] + kp_outputs, dim=1)
        # combined: [batch, 2*M_taps + P * 2*M_taps * L]

        # Step 6: Final Linear Layer 3 -> [yi, yq]
        output = self.linear3(combined)  # [batch, 2]

        return output

    def model_train(self, x, y, model_path, logger=None, total_train=1, para=None):
        """
        Train the AKPTDNN model.
        
        Args:
            x: Complex input signal array (numpy)
            y: Complex output signal array (numpy) 
            model_path: Path to save the best model
            logger: Logger for training info
            total_train: If 1, train all parameters; if 0, only final layer
            para: [learning_rate, epochs, batch_size]
        """
        if para is None:
            para = [1e-4, 2000, 2000]  # paper defaults: lr=1e-4, epochs=2000, batch=2000
        
        learning_rate = para[0]
        epochs = para[1]
        batch_size = para[2]
        device = self.device

        # Optimizer: Adam with fixed lr per paper
        if total_train == 0:
            optimizer = optim.Adam(self.linear3.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Dataset
        X_train, X_val, Y_train, Y_val = self.create_dataset(x, y)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training
        logger.info(f'------------------------AKPTDNN Train Stage-------------------------------')
        logger.info(f'M_taps={self.M_taps}, L={self.L}, P={self.P}, '
                     f'activation={self.activation}, lr={learning_rate}, '
                     f'epochs={epochs}, batch_size={batch_size}')

        train_loss_list = []
        val_loss_list = []
        best_metric = float('inf')
        best_model_state = None

        start_time = time.time()

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                inputs = inputs.to(device)
                targets = targets.to(device)
                targets_real = torch.view_as_real(targets)  # [batch, 2]

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets_real)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    targets_real = torch.view_as_real(targets)
                    val_outputs = self(inputs)
                    val_loss += criterion(val_outputs, targets_real).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            train_loss_list.append(avg_train_loss)
            val_loss_list.append(avg_val_loss)

            # Save best model
            save = 0
            if avg_val_loss < best_metric:
                best_metric = avg_val_loss
                best_model_state = self.state_dict()
                save = 1
                torch.save(best_model_state, model_path)

            logger.info(
                f'Epoch {epoch + 1:04} | Train Loss: {avg_train_loss:.7f} | '
                f'Val Loss: {avg_val_loss:.7f} | save:{save}'
            )

        # Save final model if no improvement
        if best_model_state is not None:
            torch.save(best_model_state, model_path)
            logger.info(f"Best model saved to {model_path}")
        else:
            torch.save(self.state_dict(), model_path)
            logger.info(f"Final model saved to {model_path}")

        end_time = time.time()
        logger.info(f"Training completed in {(end_time - start_time):.2f} s")

        # Plot loss curve
        if logger is not None:
            plt.figure()
            ax = plt.axes()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.plot(range(len(train_loss_list)), train_loss_list,
                     linewidth=1, linestyle="solid", label="train loss")
            plt.plot(range(len(val_loss_list)), val_loss_list,
                     linewidth=1, linestyle="dashed", label="val loss")
            plt.legend()
            plt.title(f'AKPTDNN (M={self.M_taps}, L={self.L}, P={self.P}) Loss Curve')
            plt.savefig(f'figures/MCP_NN/AKPTDNN_loss_curve.png')
            plt.close()

    def create_dataset(self, x, y, test_size=0.2):
        """
        Create train/val datasets from complex signal arrays.
        Uses memory sequence creation (existing project convention).
        
        Args:
            x: Complex input signal [N,]
            y: Complex output signal [N,]
            test_size: Validation split ratio
        
        Returns:
            [X_train, X_val, Y_train, Y_val]
        """
        M_taps = self.M_taps
        # create_memory_seq prepads with M_taps-1 samples so output has same length
        sequences = volterra_nn.create_memory_seq(x, M_taps - 1)  # [N, M_taps] complex

        X_tensor = torch.from_numpy(sequences)
        Y_tensor = torch.from_numpy(y)

        X_train, X_val, Y_train, Y_val = train_test_split(
            X_tensor, Y_tensor, test_size=test_size, shuffle=False
        )
        return [X_train, X_val, Y_train, Y_val]

    def apply_dpd(self, signal, coef=None):
        """
        Apply the trained model for DPD/prediction.
        
        Args:
            signal: Complex input signal [N,]
            coef: Not used (for compatibility with other models)
        
        Returns:
            y_pred: Complex predicted output [N-M_taps+1,]
        """
        M_taps = self.M_taps
        device = self.device
        self.to(device)
        self.eval()

        sequences = volterra_nn.create_memory_seq(signal, M_taps - 1)  # [N, M_taps] complex
        x = torch.from_numpy(sequences).to(device)

        with torch.no_grad():
            out = self(x)  # [N, 2] real-valued I/Q
        out_complex = torch.view_as_complex(out).cpu()
        y_pred = out_complex.detach().numpy()

        return y_pred

    def get_FLOPs_and_params(self):
        """
        Calculate FLOPs and parameter count per equation (19) in the paper.
        
        FLOPs = 2M * 2L + P * L * 2 + L * L * P + 2M * L * P + (2M + 2M * L * P) * 2 * 2
        Para. = (2M + 1) * 2L + (L + 1) * L * P + (2M + 2M * L * P + 1) * 2
        
        Returns:
            (flops, params)
        """
        M = self.M_taps
        L = self.L
        P = self.P

        flops = (2 * M * 2 * L
                 + P * L * 2
                 + L * L * P
                 + 2 * M * L * P
                 + (2 * M + 2 * M * L * P) * 2 * 2)

        params = ((2 * M + 1) * 2 * L
                  + (L + 1) * L * P
                  + (2 * M + 2 * M * L * P + 1) * 2)

        return flops, params
