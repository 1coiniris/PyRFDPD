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

class DDR:
    """
    Digital Predistortion model based on DDR (Digital Differential Regressor) structure.
    Supports three levels of complexity controlled by parameter r:
        r=0 : memoryless terms only (k=1..K)
        r=1 : first-order DDR (memory terms + cross-terms, k=1..K and k=3..K)
        r=2 : second-order DDR (includes double memory terms, k=3..K and k=5..K)
    """
    def __init__(self, M, K, r):
        """
        Initialize DDR model.

        Parameters:
        - M: int, memory depth (number of delayed taps)
        - K: int, nonlinearity order (polynomial order)
        - r: int, DDR order (0, 1, or 2)
        """
        self.M = M
        self.K = K
        self.r = r
        self.name = 'DDR'
        self.coef = None
        # self.x_norm_factor = 1.0   # will be set during training
        # self.y_norm_factor = 1.0

    def model_e(self, x, y, alpha=1e-5):
        """
        Train the DDR model using ridge regression.
        Input and output are normalized to maximum amplitude 1 for numerical stability.

        Parameters:
        - x: input signal (complex numpy array, shape (N,))
        - y: output signal (complex numpy array, shape (N,))
        - alpha: regularization parameter

        Returns:
        - coef: estimated coefficients (complex numpy array)
        """
        # Normalize signals to maximum amplitude 1
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        x_norm = x / np.max(np.abs(x))
        y_norm = y / np.max(np.abs(y))

        M, K, r = self.M, self.K, self.r
        # Build full basis matrix on normalized signals
        X = self.get_basis(x_norm, M, K, r)
        # Discard first M+10 samples (as in MATLAB)
        start = M + 1 + 10
        X1 = X[start:, :]
        y1 = y_norm[start:].reshape(-1, 1)

        # Ridge regression: (X1^H * X1 + alpha*I)^{-1} * X1^H * y1
        I = np.eye(X1.shape[1])
        coef = np.linalg.pinv(X1.conj().T @ X1 + alpha * I) @ X1.conj().T @ y1
        self.coef = coef.flatten()
        return self.coef

    def model_v(self, x):
        """
        Predict output using trained coefficients.
        Input is normalized using the factor stored during training,
        and the output is denormalized back to original scale.

        Parameters:
        - x: input signal (complex numpy array, shape (N,))

        Returns:
        - y: predicted output (complex numpy array, shape (N,))
        """
        if self.coef is None:
            raise ValueError("Model not trained. Call DDR_e first.")
        x = np.asarray(x).flatten()
        # Normalize input using training factor
        x_norm = x / np.max(np.abs(x))

        M, K, r = self.M, self.K, self.r
        X = self.get_basis(x_norm, M, K, r)
        y = X @ self.coef.reshape(-1, 1)
        y = y.flatten()

        start = M + 1 + 10

        y[:start] = x_norm[:start]
        # Clip samples whose magnitude exceeds 1 in the normalized domain
        y[np.abs(y) > 1] = x_norm[np.abs(y) > 1]

        return y

    def get_basis(self, x, r, K, M):
        """
        Construct the basis matrix (feature matrix) for the DDR model.
        Follows exactly the updated MATLAB code with proper k starting indices.

        Parameters:
        - x: normalized input signal (complex numpy array, shape (N,))
        - M: memory depth
        - K: nonlinearity order
        - r: DDR order (0,1,2)

        Returns:
        - X: basis matrix (complex numpy array, shape (N, n_features))
        """
        x = np.asarray(x).reshape(-1, 1)   # column vector
        N = len(x)
        X_list = []

        # ----- r >= 0 : memoryless terms -----
        # |x(n)|^{k-1} * x(n) , k = 1..K
        if r >= 0:
            for k in range(1, K + 1):
                X_buff = x * (np.abs(x) ** (k - 1))
                X_list.append(X_buff)

        # ----- r >= 1 : first-order DDR -----
        if r >= 1:
            # 1) |x(n)|^{k-1} * x(n-m) , m = 0..M , k = 1..K
            for m in range(M + 1):
                x_buff = np.zeros((N, 1), dtype=complex)
                if m < N:
                    x_buff[m:] = x[:N - m]
                for k in range(1, K + 1):
                    X_buff = x_buff * (np.abs(x) ** (k - 1))
                    X_list.append(X_buff)

            # 2) |x(n)|^{k-3} * x(n)^2 * conj{x(n-m)} , m = 1..M , k = 3..K
            for m in range(1, M + 1):
                x_buff = np.zeros((N, 1), dtype=complex)
                if m < N:
                    x_buff[m:] = np.conj(x[:N - m])
                for k in range(3, K + 1):
                    X_buff = (np.abs(x) ** (k - 3)) * (x ** 2) * x_buff
                    X_list.append(X_buff)

        # ----- r >= 2 : second-order DDR (double memory) -----
        if r >= 2:
            # a) |x(n)|^{k-3} * conj(x(n)) * x(n-m1) * x(n-m2)
            #    m1 = 1..M, m2 = 1..M, k = 3..K
            for m1 in range(1, M + 1):
                x_buff1 = np.zeros((N, 1), dtype=complex)
                if m1 < N:
                    x_buff1[m1:] = x[:N - m1]
                for m2 in range(1, M + 1):
                    x_buff2 = np.zeros((N, 1), dtype=complex)
                    if m2 < N:
                        x_buff2[m2:] = x[:N - m2]
                    for k in range(3, K + 1):
                        X_buff = (np.abs(x) ** (k - 3)) * np.conj(x) * x_buff1 * x_buff2
                        X_list.append(X_buff)

            # b) |x(n)|^{k-3} * x(n) * x(n-m1) * conj(x(n-m2))
            #    m1 = 1..M, m2 = 1..M, k = 3..K
            for m1 in range(1, M + 1):
                x_buff1 = np.zeros((N, 1), dtype=complex)
                if m1 < N:
                    x_buff1[m1:] = x[:N - m1]
                for m2 in range(1, M + 1):
                    x_buff2 = np.zeros((N, 1), dtype=complex)
                    if m2 < N:
                        x_buff2[m2:] = x[:N - m2]
                    for k in range(3, K + 1):
                        X_buff = (np.abs(x) ** (k - 3)) * x * x_buff1 * np.conj(x_buff2)
                        X_list.append(X_buff)

            # c) |x(n)|^{k-5} * x(n)^3 * conj(x(n-m1)) * conj(x(n-m2))
            #    m1 = 1..M, m2 = 1..M, k = 5..K
            for m1 in range(1, M + 1):
                x_buff1 = np.zeros((N, 1), dtype=complex)
                if m1 < N:
                    x_buff1[m1:] = x[:N - m1]
                for m2 in range(1, M + 1):
                    x_buff2 = np.zeros((N, 1), dtype=complex)
                    if m2 < N:
                        x_buff2[m2:] = x[:N - m2]
                    for k in range(5, K + 1):
                        X_buff = (np.abs(x) ** (k - 5)) * (x ** 3) * np.conj(x_buff1) * np.conj(x_buff2)
                        X_list.append(X_buff)

        # Concatenate all features horizontally
        X = np.hstack(X_list)
        return X