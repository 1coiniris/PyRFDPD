import os
import math
import pyvisa
import numpy as np
import time
from enum import Enum
from time import sleep
from RsSmw import *

# VSG: SMW200A

class VSG:
    def __init__(self, ip = '192.168.1.221',type = 'RS_FSW_200A'):
        # self.inst = pyvisa.ResourceManager().open_resource(f'TCPIP::{ip}::HISLIP')
        self.type = type
        self.IP = ip
        # self.enum = FsvaEnum

    # def __del__(self):
    #     self.inst.close()

    def transmit(self,brand, x, fc, fs, power=-30, file_name="Waveform_lz.wv", logger=None):
        if len(x) > 80001:
            print("Too long for VSG.")
        # self.SendVSG(x, fs, pow, fc)
        self.down_signal(brand,x,fc,fs,power,file_name,logger)
        return


    def SendVSG(self,x,fs,pow,fc):

        return

    def down_signal(self,brand, x, fc, fs, power=-30, file_name="Waveform_lz.wv", logger=None):
        x = x.copy()
        IP = self.IP
        if brand.lower() == "rohde-schwarz":
            # MATLAB移植时碰到很多官方的文件，难以移植，故直接采用官方Python包
            pc_wv_file = "./arbFile.wv"
            instr_wv_file_out = "/var/user/" + file_name
            smw = RsSmw("TCPIP::" + IP + "::HISLIP")
            # if logger:
            #     logger.info("Sucessfully Connected to signal generator!")
            # RsSmw.assert_minimum_version('5.0.44')
            # print(smw.utilities.idn_string)
            smw.utilities.reset()

            # I-component an Q-component data
            i_data = [data.real for data in x]
            q_data = [data.imag for data in x]
            smw.arb_files.create_waveform_file_from_samples(
                i_data,
                q_data,
                pc_wv_file,
                clock_freq=fs,
                auto_scale=True,
                comment="Created from I/Q vectors",
            )
            smw.arb_files.send_waveform_file_to_instrument(pc_wv_file, instr_wv_file_out)

            smw.source.bb.arbitrary.waveform.set_select(instr_wv_file_out)
            smw.source.frequency.set_frequency(fc)
            smw.source.power.level.immediate.set_amplitude(power)
            smw.source.bb.arbitrary.set_state(True)
            smw.output.state.set_value(True)

            smw.close()
            if logger:
                logger.info("SMW signal transmission finished!")
            time.sleep(2)

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