'''
Copyright 2023 Microwave System Lab or its affiliates. All Rights Reserved.
File: gmp.py
Authors:
Zhe Li, 904016301@qq.com

Description:
The Generalized Memory Polynomial (GMP) model extraction and evaluation functions. 

Revision histry:
Version   Date        Author      Changes
1.0    2024-1-18    Zhe Li      initial version
'''
import numpy as np
import scipy.io
import argparse
from function import Function_Calculate as cal


class GMP:
    def __init__(self,K: list, L: list, M: list):
        self.K = K
        self.L = L
        self.M = M
        self.name = 'GMP'
        self.coef = None


    def model_e(self,x_target: np.ndarray, y_target: np.ndarray, ratio: float=1)->np.ndarray:
        """
        This is the coefficient extraction file based on GMP DPD
        designed by Qianyun Lu, Oct 12, 2019, qianyun.lu@seu.edu.cn
        Refer to following paper for more information:
        [1] https://ieeexplore.ieee.org/document/1703853

        Args:
            x_target: the PA input signal,
            y_target: the PA output signal
            K: non-linearity order, three terms
            L: lagging depth, three terms
            M: memory depth, two terms
            ratio: ratio of samples for extraction

        Returns:
            coef: the extracted coefficients
        """
        K = self.K
        L = self.L
        M = self.M
        assert(len(x_target) == len(y_target)), "The length of x and y should be the same."
        x_target = x_target[:int(ratio * len(x_target))]
        y_target = y_target[:int(ratio * len(x_target))]
        x_target = np.ravel(x_target) # Change from 2D to 1D array
        y_target = np.ravel(y_target) # Change from 2D to 1D array
        # Ka, Kb, Kc = K
        # La, Lb, Lc = L
        # Mb, Mc = M
        # N = len(x_target)

        # X_align = np.empty((N, Ka*La), dtype='complex')
        # for k in range(Ka):
        #     for l in range(La):
        #         xd = np.roll(x_target, l)
        #         X_align[:, k*La+l] = xd * np.power(abs(xd), k)
        #
        # X_lag = np.empty((N, Kb*Lb*Mb), dtype='complex')
        # for k in range(Kb):
        #     for l in range(Lb):
        #         for m in range(Mb):
        #             xd = np.roll(x_target, l)
        #             xdm = np.roll(xd, m+1)
        #             X_lag[:, k*(Lb+Mb)+l*Mb+m] = xd * np.power(abs(xdm), k+1)
        #
        # X_lead = np.empty((N, Kc*Lb*Mb), dtype='complex')
        # for k in range(Kc):
        #     for l in range(Lc):
        #         for m in range(Mc):
        #             xd = np.roll(x_target, l)
        #             xdm = np.roll(xd, -(m+1))
        #             X_lead[:, k*(Lc+Mc)+l*Mc+m] = xd * np.power(abs(xdm), k+1)
        # X = np.hstack((X_align, X_lag, X_lead))
        # X[np.isnan(X)] = 0 # Remove NaN

        X = self.get_basis(x_target)
        XH = np.conjugate(X.T)
        coef = np.linalg.inv(XH.dot(X) + 1e-5*np.eye(X.shape[1])).dot(XH).dot(y_target)

        y_model = self.model_v(x_target, coef)
        NMSE = cal.nmse(y_target, y_model)
        # print(f'NMSE-with-model = {NMSE:.6f} dB')
        return coef

    def model_v(self,x_target: np.ndarray, coef)->np.ndarray:
        """
        This is the coefficient evaluation file based on MP DPD
        designed by Qianyun Lu, Oct. 12, 2018, qianyun.lu@seu.edu.cn

        Args:
            x_target: the PA input signal
            coef: the extracted coefficients
            K: non-linearity order
            L: lagging depth
            M: memory depth

        Returns:
            y: the calculated model output
        """
        K = self.K
        L = self.L
        M = self.M
        N = len(x_target)
        x_target = np.ravel(x_target) # Change from 2D to 1D array
        # Ka, Kb, Kc = K
        # La, Lb, Lc = L
        # Mb, Mc = M
        #
        # X_align = np.empty((N, Ka*La), dtype='complex')
        # for k in range(Ka):
        #     for l in range(La):
        #         xd = np.roll(x_target, l)
        #         X_align[:, k*La+l] = xd * np.power(abs(xd), k)
        #
        # X_lag = np.empty((N, Kb*Lb*Mb), dtype='complex')
        # for k in range(Kb):
        #     for l in range(Lb):
        #         for m in range(Mb):
        #             xd = np.roll(x_target, l)
        #             xdm = np.roll(xd, m+1)
        #             X_lag[:, k*(Lb+Mb)+l*Mb+m] = xd * np.power(abs(xdm), k+1)
        #
        # X_lead = np.empty((N, Kc*Lb*Mb), dtype='complex')
        # for k in range(Kc):
        #     for l in range(Lc):
        #         for m in range(Mc):
        #             xd = np.roll(x_target, l)
        #             xdm = np.roll(xd, -(m+1))
        #             X_lead[:, k*(Lc+Mc)+l*Mc+m] = xd * np.power(abs(xdm), k+1)
        #
        # X = np.hstack((X_align, X_lag, X_lead))
        # X[np.isnan(X)] = 0 # Remove NaN
        X = self.get_basis(x_target)
        y = X.dot(coef)
        return y

    def apply_dpd(self, signal,coef = None):
        y = self.model_v(signal,coef)
        return y

    def get_basis(self,x_target: np.ndarray)->np.ndarray:
        K = self.K
        L = self.L
        M = self.M

        N = len(x_target)
        x_target = np.ravel(x_target)  # Change from 2D to 1D array
        Ka, Kb, Kc = K
        La, Lb, Lc = L
        Mb, Mc = M

        X_align = np.empty((N, Ka * La), dtype='complex')
        for k in range(Ka):
            for l in range(La):
                xd = np.roll(x_target, l)
                X_align[:, k * La + l] = xd * np.power(abs(xd), k)

        X_lag = np.empty((N, Kb * Lb * Mb), dtype='complex')
        for k in range(Kb):
            for l in range(Lb):
                for m in range(Mb):
                    xd = np.roll(x_target, l)
                    xdm = np.roll(xd, m + 1)
                    X_lag[:, k * (Lb + Mb) + l * Mb + m] = xd * np.power(abs(xdm), k + 1)

        X_lead = np.empty((N, Kc * Lb * Mb), dtype='complex')
        for k in range(Kc):
            for l in range(Lc):
                for m in range(Mc):
                    xd = np.roll(x_target, l)
                    xdm = np.roll(xd, -(m + 1))
                    X_lead[:, k * (Lc + Mc) + l * Mc + m] = xd * np.power(abs(xdm), k + 1)

        X = np.hstack((X_align, X_lag, X_lead))
        X[np.isnan(X)] = 0  # Remove NaN
        return X



if __name__ == "__main__":
    print("This is the GMP extraction and evaluation functions.")
    print("To use these function, you need a PA_data.mat file")
    print("And two signal in that file, named 'xorg' and 'yorg', repectivly.")
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', "--file", type=str, default='PA_data.mat', help='input file name')
    parser.add_argument('-M', type=int, default=3, help='memory depth')
    parser.add_argument('-K', type=int, default=7, help='non-linearity order')

    args=parser.parse_args()

    PA_in = np.ravel(scipy.io.loadmat(args.file)['xorg'])
    PA_out = np.ravel(scipy.io.loadmat(args.file)['yorg'])

    # coef = GMP_e(PA_in, PA_out, args.M, args.K, len(PA_in))
    # PA_exp = GMP_v(PA_in, coef, args.M, args.K)

    # plt.plot(np.abs(PA_in[0:1000]), label="PA original input")
    # plt.plot(np.abs(PA_out[0:1000]), label="PA original output")
    # plt.plot(np.abs(PA_exp[0:1000]), label="PA expected output")
    # plt.legend()
    # plt.show()