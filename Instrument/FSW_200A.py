import os
import math
import pyvisa
import numpy as np
import time
from enum import Enum
from time import sleep


class FsvaTraceModeEnum():
    VIEW = 'VIEW'
    CLEAR_WRITE = 'WRIT'
    MAX_HOLD = 'MAXH'
    MIN_HOLD = 'MINH'
    AVERAGE = 'AVER'
    BLANK = 'BLAN'


class FsvaTraceTypeEnum():
    AUTO_PEAK = 'APE'
    POSITIVE_PEAK = 'POS'
    NEGATIVE_PEAK = 'NEG'
    RMS = 'RMS'
    AVERAGE = 'AVER'
    SAMPLE = 'SAMP'


class FsvaTraceEnum():
    mode = FsvaTraceModeEnum
    type = FsvaTraceTypeEnum


class FsvaEnum():
    trace = FsvaTraceEnum

class RS_FSW_200A:
    def __init__(self, ip = '192.168.1.221'):
        self.inst = pyvisa.ResourceManager().open_resource(f'TCPIP::{ip}::HISLIP')
        self.write(':SYST:DISP:UPD ON')
        self.write('INIT:CONT ON')
        self.enum = FsvaEnum

    def __del__(self):
        self.inst.close()

    def write(self, val:str):
        self.inst.write(val)

    def query(self, val:str):
        return self.inst.query(val)


    def generate_wv(self,x, fs, file_name):
        # https://www.rohde-schwarz.com/uk/faq/example-on-how-to-manually-generate-a-wv-file-with-python-faq_78704-1166336.html
        # validated with R&S arb toolbox
        I = [z.real for z in x]
        Q = [z.imag for z in x]
        named_tuple = time.localtime()
        time_string = time.strftime("%m-%d-%Y;%H:%M:%S", named_tuple)  # generate time stamp
        fobj_1 = open(file_name, "w")
        # write mandatory header/tag info to *.wv file
        fobj_1.write("{TYPE: SMU-WV,0}")
        fobj_1.write("{DATA:" + time_string + "}")
        fobj_1.write("{CLOCK:" + str(fs) + "}")
        # RMS, Peak
        fobj_1.write("{LEVEL OFFS:0.0, 0.0}")
        fobj_1.write("{SAMPLES:" + str(len(I)) + "}")
        fobj_1.write("{WAVEFORM-" + str((len(I) * 4) + 1) + ":#")
        fobj_1.close()

        fobj_2 = open(file_name, "ab")  # open created *.wv file to append I and Q bytes
        for i in range(len(I)):
            # 16-bit dac, hex, little endian
            little_end_hex_I = (math.floor(I[i] * 32767 + 0.5)).to_bytes(
                2, byteorder="little", signed=True
            )
            fobj_2.write(little_end_hex_I)
            little_end_hex_Q = (math.floor(Q[i] * 32767 + 0.5)).to_bytes(
                2, byteorder="little", signed=True
            )
            fobj_2.write(little_end_hex_Q)

        fobj_2.write(bytes("}".encode()))
        fobj_2.close()
        if os.path.isfile(file_name):
            print("File " + file_name + " successfully generated!")
