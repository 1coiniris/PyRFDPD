import os
import math
import pyvisa
import numpy as np
import time
from enum import Enum
from time import sleep
from RsSmw import *

class VSA:
    def __init__(self, ip = '192.168.1.221',type = 'keysight'):
        # self.inst = pyvisa.ResourceManager().open_resource(f'TCPIP::{ip}::HISLIP')
        self.type = type
        self.IP = ip

    # def __del__(self):
    #     self.inst.close()

    def collect_signal(self,name, fc, fs, att, logger=None) -> np.ndarray:
        IP = self.IP
        assert name.lower() == "keysight" or "fsw" or "fpl"
        rm = pyvisa.ResourceManager()
        try:
            instr = rm.open_resource("TCPIP0::" + IP + "::hislip0::INSTR")
        except pyvisa.VisaIOError:
            print("Can't find the instrument with the IP address!")
            raise
        # else:
        #     if logger:
        #         logger.info("Sucessfully connected to spectrum analyzer!")

        time.sleep(2)
        # instr.write("*rst")
        # instr.query("*opc?")

        if name.lower() == "keysight":
            bandwidth = fs / 1.25
            measureTime = 20e-4
            instr.write("*SAV 8")
            # instr.write("*RST")
            instr.write(":INSTrument:SELect BASIC")
            instr.write(":SENSe:FREQuency:CENTer " + str(fc))
            # this command can't not found in manual but can be used
            instr.write(":SENSe:WAVEform:BANDwidth:RESolution " + str(bandwidth))
            instr.write(":SENSe:WAVeform:AVER OFF")
            instr.write(":INIT:CONT OFF")
            instr.write(":WAVeform:SWE:TIME " + str(measureTime))
            instr.write(":SENSe:POWer:RF:ATTenuation " + str(att))
            instr.write(":SENSe:VOLTage:IQ:RANGe:AUTO ON")
            instr.write(":FORMat:BORDer NORMal")
            instr.write(":FORMat:DATA REAL,32")
            instr.write(":INITiate:WAVeform")
            instr.write(":READ:WAV0?")
            data = instr.read_binary_values(
                datatype="f", is_big_endian=True
            )  # Attention! big endian. not like RS
            # data = instr.read_raw()
            instr.write(":INSTrument:SELect SA")
            # Go back to saved state 8
            # instr.write("*RCL 8")
            # instr.write("*TRG")
            I_data = data[0::2]
            Q_data = data[1::2]
            IQ_data = np.array([complex(I, Q) for I, Q in zip(I_data, Q_data)], dtype='complex')
            if logger:
                logger.info("Successfully captured IQ data!")
            return IQ_data / max(abs(IQ_data))

        if name.lower() == "fsw":
            # TODO: 问题待修复
            instr.write("*RST")
            instr.write(":INPut:TYPE INPUT2")
            instr.write(":FORM:DATA REAL, 32")  # set saved data format
            instr.write(
                "INSTrument:CREate IQ, 'IQANALYZER'"
            )  # new channel in IQ analysis mode
            instr.write("LAYout:REPLace:WINDow '1',RIMAG")
            instr.write("INIT:CONT OFF")  # select single sweep mode
            instr.write("TRAC:IQ:SRAT " + str(fs) + "Hz")
            instr.write("TRAC:IQ:RLEN 100000")  # set the number of samples to capture
            instr.write("FREQ:CENT " + str(fc) + "Hz")  # centre frequency
            instr.write("FREQ:SPAN " + str(fs) + "Hz")  # frequency span
            instr.write("DISP:TRAC1:Y:RLEV 0dBm")  # set the reference level to 0 dBm.
            instr.write(
                "INPut:ATTenuation " + str(att) + "dB"
            )  # set the input attenuation,
            # instr.write('SENSe:SWEep:TIME 500us') # Configure sweep time
            instr.write("SENSe:SWEep:POIN 100000")  # Configure sweep points
            instr.write(
                "TRAC:IQ:DATA:FORM IQP"
            )  # Lists all I values first, then all Q values in the trace results.
            # Perform the measurement
            instr.write("INIT;*WAI")
            # Retrieve the data
            instr.write("TRAC:DATA? TRACE1")
            data = instr.read_binary_values(datatype="f")  # return float
            # instr.query('TRAC1:X? TRACE1')
            # x = instr.read_binary_values(datatype = 'b')
            instr.write("INIT:CONT ON")  # select single sweep mode

            I_data = data[0: len(data) // 2]
            Q_data = data[len(data) // 2:]
            IQ_data = np.array([complex(I, Q) for I, Q in zip(I_data, Q_data)], dtype='complex_')
            # data = [int(binary_str, 2) for binary_str in raw_data]
            # IQ_data = complex(data[0:len(data)//2], data[len(data)//2+1:])
            return IQ_data

        if name.lower() == "fpl":
            # 未验证，因为FPL带宽太窄无法做DPD
            instr.write("*RST")
            instr.write(":INPut:TYPE INPUT2")
            instr.write(":FORM:DATA REAL, 32")  # set saved data format
            instr.write(
                "INSTrument:CREate SAN, 'Spectrum 1'"
            )  # new channel in Sepctrum mode
            instr.write("LAYout:REPLace:WINDow '1',RIMAG")
            instr.write("INIT:CONT OFF")  # select single sweep mode
            instr.write("TRAC:IQ:SRAT " + str(fs) + "Hz")
            instr.write("TRAC:IQ:RLEN 100000")  # set the number of samples to capture
            instr.write("FREQ:CENT " + str(fc) + "Hz")  # centre frequency
            instr.write("FREQ:SPAN " + str(fs) + "Hz")  # frequency span
            instr.write("DISP:TRAC1:Y:RLEV 0dBm")  # set the reference level to 0 dBm.
            instr.write(
                "INPut:ATTenuation " + str(att) + "dB"
            )  # set the input attenuation,
            # instr.write('SENSe:SWEep:TIME 500us') # Configure sweep time
            instr.write("SENSe:SWEep:POIN 100000")  # Configure sweep points
            instr.write(
                "TRAC:IQ:DATA:FORM IQP"
            )  # Lists all I values first, then all Q values in the trace results.
            # Perform the measurement
            instr.write("INIT;*WAI")
            # Retrieve the data
            instr.write("TRAC:DATA? TRACE1")
            raw_data = instr.read_binary_values(datatype="B")  # return unsigned char
            # instr.query('TRAC1:X? TRACE1')
            # x = instr.read_binary_values(datatype = 'b')
            instr.write("INIT:CONT ON")  # select single sweep mode

            data = [int(binary_str, 2) for binary_str in raw_data]
            IQ_data = complex(data[0: len(data) // 2], data[len(data) // 2:])
            return IQ_data
