import numpy as np
import warnings
from typing import Union, Dict, Any
from Instrument import VSA, VSG
from function import align
import argparse
import time

class SingleBandPA:
    """
    Python implementation of Single_Band_PA class for PA testbench.

    Attributes:
        pow (float): VSG output Power in dB
        VSG_IP (str): IP of Vector Signal Generator (VSG)
        VSA_IP (str): IP of Vector Signal Analyzer (VSA)
        VSA_type (str): Type of Vector Signal Analyzer (VSA) 'rs' or 'k'
        fs (float): sampling rate
        fc (float): carrier frequency
        att (float): attenuation level of VSA
        type (int): test type (0 for MATLAB-defined PA, 1 for real PA)
    """

    def __init__(self, params: Union[Dict[str, Any], None] = None):
        """
        Initialize SingleBandPA instance.

        Args:
            params: Dictionary containing initialization parameters.
                   If None, default values are used.
        """
        # Set default parameters if none provided
        if params is None:
            params = {
                'pow': -28,  # output power in dB
                'VSG_IP': '192.168.0.29',  # IP of Vector Signal Generator (VSG)
                'VSA_IP': '192.168.0.30',  # IP of Vector Signal Analyzer (VSA)
                'VSA_type': 'rs',  # Type of Vector Signal Analyzer (VSA) 'rs' or 'k'
                'waveformfile': 'waveform_crz',
                'fs': 160e6,  # sampling rate = 160 MHz
                'fc': 2.4e9,  # carrier frequency = 2.14 GHz
                'att': 20,  # attenuation level of (VSA) in dB
                'type': 1  # test type
            }

        # Initialize attributes
        self.pow = params.get('pow', -28)
        self.VSG_IP = params.get('VSG_IP', '192.168.0.29')
        self.VSA_IP = params.get('VSA_IP', '192.168.0.30')
        self.VSA_type = params.get('VSA_type', 'rs')
        self.waveformfile = params.get('waveformfile', 'waveform_crz')
        self.fs = params.get('fs', 160e6)
        self.fc = params.get('fc', 2.4e9)
        self.att = params.get('att', 20)
        self.type = params.get('type', 1)

        self.VSG_1 = VSG.VSG(self.VSG_IP)
        self.VSA_1 = VSA.VSA(self.VSA_IP)


    def transmit(self, x: np.ndarray, logger=None) -> np.ndarray:
        """
        Broadcast input signal {x} through real PA or MATLAB-defined PA.

        Args:
            x: Input signal as a column vector

        Returns:
            y: Output signal as a column vector, normalized to have the same norm as x
        """
        x = np.asarray(x).flatten()  # Ensure x is a 1D array

        if self.type == 1:  # Real PA mode
            if len(x) > 80001:
                warnings.warn("Too long for Single_Band_PA.", UserWarning)

            # Send signal to VSG (assuming SendVSG function is implemented)
            # SendVSG(x, self.fs, self.pow, self.fc, self.VSG_IP)

            # Collect signal from VSA (assuming CollectSignal function is implemented)
            # y0 = CollectSignal(self.fc, self.fs, self.att, self.VSA_IP, self.VSA_type)

            # For demonstration, we'll use a simple simulation
            # In practice, replace with actual hardware calls
            self.VSG_1.transmit(brand="rohde-schwarz", x=x, fc=self.fc,fs=self.fs, power=self.pow, logger=logger)

            y0 = self.VSA_1.collect_signal(name="keysight",fc=self.fc, fs=self.fs, att=self.att,logger=logger)
            # Time alignment and normalization (assuming align_coarse_norm and upsample_nrmse are implemented)
            # _, y = align_coarse_norm(x, y0)
            # _, y = upsample_nrmse(x, y, 16)

            # Simplified implementation for demonstration
            y = align.align(x, y0,'PCF')
            # y = align.coarse_align(x,y0)
            # y = y0
            y = y / np.max(np.abs(y))

            # Normalize to input norm (commented out in original)
            # y = y * np.linalg.norm(x) / np.linalg.norm(y)

            return y

        # elif self.type == 0:  # MATLAB-defined PA mode
        #     # Apply PA model (assuming PA function is implemented)
        #     # y0 = PA(x, self.pow)
        #
        #     # Simplified implementation for demonstration
        #     y0 = self._matlab_pa_model(x, self.pow)
        #
        #     # Time alignment and normalization
        #     # _, y = align_coarse_norm(x, y0.T)
        #     # _, y = upsample_nrmse(x, y, 16)
        #
        #     # Simplified implementation for demonstration
        #     y = self._align_and_normalize(x, y0)
        #     y = y / np.max(np.abs(y))
        #
        #     return y