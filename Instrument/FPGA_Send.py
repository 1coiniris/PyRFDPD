import time
import socket
import struct
import numpy as np


def send_all(sock: socket.socket, data: bytes) -> None:
    """确保所有字节都发送完成"""
    total_sent = 0
    while total_sent < len(data):
        sent = sock.send(data[total_sent:])
        if sent == 0:
            raise ConnectionError("Socket connection broken")
        total_sent += sent


def create_tcp_client(ip_addr="192.168.0.22", port=5001, timeout=5):
    """主动连接板卡"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((ip_addr, port))
    print(f"Connected to {ip_addr}:{port}")
    return sock


def fpga_send_signal(sock: socket.socket, DAC1, DAC2) -> None:
    """
    对应 MATLAB 的 FPGA_SendSignal(server, DAC1, DAC2)

    DAC1, DAC2:
        长度必须为 524288 的复数向量
    """
    BD_length = 4194304
    BD_per_packet = 1
    dac_length = BD_per_packet * BD_length   # 单位: byte
    dac_length_int16 = dac_length // 2

    DAC1 = np.asarray(DAC1).reshape(-1)
    DAC2 = np.asarray(DAC2).reshape(-1)

    if len(DAC1) != 524288 or len(DAC2) != 524288:
        raise ValueError("DAC1 和 DAC2 长度必须都为 524288")

    # MATLAB:
    # dac0_data = real(DAC1); reshape(4, [])
    # dac1_data = imag(DAC1); reshape(4, [])
    # dac2_data = real(DAC2); reshape(4, [])
    # dac3_data = imag(DAC2); reshape(4, [])
    #
    # MATLAB reshape 是列优先，所以这里要用 order='F'
    dac0_data = np.real(DAC1).reshape(4, -1, order='F')
    dac1_data = np.imag(DAC1).reshape(4, -1, order='F')
    dac2_data = np.real(DAC2).reshape(4, -1, order='F')
    dac3_data = np.imag(DAC2).reshape(4, -1, order='F')

    # MATLAB 每次拼接:
    # [dac3(:,i)' dac2(:,i)' dac1(:,i)' dac0(:,i)']
    N = dac_length_int16 // 16
    dac_data = np.hstack([
        dac3_data[:, :N].T,
        dac2_data[:, :N].T,
        dac1_data[:, :N].T,
        dac0_data[:, :N].T
    ]).reshape(-1)

    # MATLAB: dac_data = dac_data * 16384;
    # 最终发送 int16
    dac_data = np.round(dac_data * 16384).astype(np.int16)

    command = 4

    # MATLAB: write(server, [command, dac_length], "uint32")
    # 通常这里按 little-endian 打包
    header = struct.pack("<II", command, dac_length)
    send_all(sock, header)

    time.sleep(0.5)

    # MATLAB: write(server, dac_data, 'int16')
    send_all(sock, dac_data.tobytes())

    time.sleep(30)


def build_dac_from_xorg(xorg, repeat_times=32, make_dac2_same=True):
    """
    根据 xorg 构造 DAC1 / DAC2

    参数:
        xorg : 复数向量，行/列都可以
        repeat_times : 默认 32，对应 MATLAB:
            for i = 1:32
                DAC1 = [DAC1; softwareDPDout(:)];
            end
        make_dac2_same : True -> DAC2 = DAC1
                         False -> DAC2 = 全零

    返回:
        DAC1, DAC2
    """
    xorg = np.asarray(xorg).reshape(-1)   # 不管原来是行还是列，都转成 1D

    softwareDPDout = xorg.copy()

    DAC1 = np.tile(softwareDPDout, repeat_times)

    if len(DAC1) != 524288:
        raise ValueError(
            f"构造后的 DAC1 长度为 {len(DAC1)}，不是 524288。"
            f" 当前 xorg 长度为 {len(xorg)}，repeat_times={repeat_times}。"
            f" 需要满足 len(xorg) * repeat_times == 524288。"
        )

    if make_dac2_same:
        DAC2 = DAC1.copy()
    else:
        DAC2 = np.zeros_like(DAC1, dtype=np.complex128)

    return DAC1, DAC2