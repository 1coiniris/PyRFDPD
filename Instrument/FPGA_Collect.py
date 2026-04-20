import socket
import struct
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


def recv_all(sock: socket.socket, num_bytes: int) -> bytes:
    chunks = []
    received = 0
    while received < num_bytes:
        chunk = sock.recv(num_bytes - received)
        if not chunk:
            raise ConnectionError("Socket connection broken while receiving data")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def bandpower(x: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    return float(np.mean(np.abs(x) ** 2))


def papr(x: np.ndarray) -> float:
    """
    返回 dB 形式的 PAPR
    MATLAB 里 papr(x) 通常也是 dB 量
    """
    x = np.asarray(x).reshape(-1)
    p_peak = np.max(np.abs(x) ** 2)
    p_avg = np.mean(np.abs(x) ** 2)
    if p_avg == 0:
        return 0.0
    return 10.0 * np.log10(p_peak / p_avg)


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    err = np.sqrt(np.mean(np.abs(a - b) ** 2))
    ref = np.sqrt(np.mean(np.abs(a) ** 2))
    if ref == 0:
        return 0.0
    return float(err / ref)

def fractional_circular_shift(x: np.ndarray, shift: float) -> np.ndarray:
    """
    对 x 做循环意义下的分数延时
    shift > 0 表示向右循环平移 shift 个采样
    """
    x = np.asarray(x).reshape(-1)
    N = len(x)
    X = np.fft.fft(x)
    k = np.fft.fftfreq(N) * N
    phase = np.exp(-1j * 2 * np.pi * k * shift / N)
    y = np.fft.ifft(X * phase)
    return y

def find_shift_am(x_abs: np.ndarray, y_abs: np.ndarray, max_lag=None) -> float:
    """
    用幅度包络互相关估计整数时延
    返回值含义：x 相对于 y 的最佳平移量
    """
    x_abs = np.asarray(x_abs).reshape(-1)
    y_abs = np.asarray(y_abs).reshape(-1)

    c = signal.correlate(x_abs, y_abs, mode="full")
    lags = signal.correlation_lags(len(x_abs), len(y_abs), mode="full")

    if max_lag is not None:
        mask = np.abs(lags) <= max_lag
        c = c[mask]
        lags = lags[mask]

    lag = lags[np.argmax(np.abs(c))]
    return float(lag)


def estimate_phase_offset(x: np.ndarray, y: np.ndarray, start=1000, stop=2000) -> float:
    """
    模拟 align_signals.m 中
    ph_shift = median(angle(y(1001:2000)) - angle(xI(1001:2000)))
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    start = max(0, start)
    stop = min(len(x), len(y), stop)
    if stop <= start:
        start = 0
        stop = min(len(x), len(y))

    phase_diff = np.angle(y[start:stop]) - np.angle(x[start:stop])
    return float(np.median(phase_diff))


def phase_divide_b2a(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    让 b 的整体相位尽量对齐到 a
    这里做一个单一复数比例补偿（LS意义）
    """
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    denom = np.vdot(b, b)
    if np.abs(denom) < 1e-12:
        return b.copy()

    alpha = np.vdot(b, a) / denom
    return b * alpha

def upsample_complex(x: np.ndarray, rate: int) -> np.ndarray:
    """
    复数信号上采样，分别对实部和虚部做线性插值
    """
    x = np.asarray(x).reshape(-1)
    N = len(x)
    old_idx = np.arange(N)
    new_idx = np.linspace(0, N - 1, N * rate)

    fr = interp1d(old_idx, np.real(x), kind="linear", fill_value="extrapolate")
    fi = interp1d(old_idx, np.imag(x), kind="linear", fill_value="extrapolate")
    return fr(new_idx) + 1j * fi(new_idx)


def upsample_nrmse_python(a: np.ndarray, b: np.ndarray, rate: int = 16, findlen: int = 100):
    """
    近似 MATLAB upsample_nrmse
    返回:
        new_nrmse, new_b
    """
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    a_up = upsample_complex(a, rate)
    b_up = upsample_complex(b, rate)

    a_up = a_up / np.max(np.abs(a_up))
    b_up = b_up / np.max(np.abs(b_up))

    a_env = np.abs(a_up)
    b_env = np.abs(b_up)

    half = findlen // 2
    shifts = np.arange(-half, half)
    cor_energy = []

    for s in shifts:
        b_env_shift = np.roll(b_env, s)
        cor_energy.append(np.sum(a_env * b_env_shift))

    best_shift = shifts[int(np.argmax(cor_energy))]
    b_up_sync = np.roll(b_up, best_shift)

    # 降采样
    a_new = a_up[::rate]
    b_new = b_up_sync[::rate]

    # 相位/复比例校正
    b_sync = phase_divide_b2a(a_new, b_new)
    return nrmse(a_new, b_sync), b_sync

def align_signals_python(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    让 x 对齐到 y，返回对齐后的 x_new
    模拟 MATLAB align_signals 的主要流程
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    # 第一次幅度相关找延时
    Lmax = -find_shift_am(np.abs(x), np.abs(y))
    xI = fractional_circular_shift(x, Lmax)

    # 第二次微调
    Lmax = -find_shift_am(np.abs(xI), np.abs(y))
    if abs(Lmax) > 1e-12:
        xI = fractional_circular_shift(xI, Lmax)

    # 再看一次
    Lmax = -find_shift_am(np.abs(xI), np.abs(y))

    # MATLAB 里有 findshiftPM，这里用整体复比例 + 中位相位差代替
    ph_shift = estimate_phase_offset(xI, y, start=1000, stop=2000)
    x_new = xI * np.exp(1j * ph_shift)

    return x_new

def align_coarse_norm_python(sig_o: np.ndarray, output: np.ndarray):
    """
    对应 MATLAB align_coarse_norm 的近似增强版
    返回:
        x_0, y_0
    """
    sig_o = np.asarray(sig_o).reshape(-1)
    output = np.asarray(output).reshape(-1)

    if len(output) < len(sig_o):
        raise ValueError("output 长度不能小于 sig_o")

    dd = 1
    start = dd - 1

    # 取与 sig_o 等长的一段 output
    y_00 = output[start:start + len(sig_o)].copy()
    y_00 = y_00 / np.max(np.abs(y_00))

    # 粗对齐：包络互相关
    ck = signal.correlate(sig_o, y_00, mode="full")
    lags = signal.correlation_lags(len(sig_o), len(y_00), mode="full")
    C_indx = lags[np.argmax(np.abs(ck))]
    y_00 = np.roll(y_00, int(C_indx))

    # AM / PM / 相位对齐
    y_0 = align_signals_python(y_00, sig_o)
    y_0 = y_0 / np.max(np.abs(y_0))

    # 上采样进一步微调
    _, y_0 = upsample_nrmse_python(sig_o, y_0, rate=16)

    x_0 = sig_o / np.max(np.abs(sig_o))
    return x_0, y_0

def fpga_collect_signal(sock: socket.socket, adc_length=524288, draw=False):
    """
    对应 MATLAB FPGA_CollectSignal
    保留原版 PAPR 归一化
    """
    if adc_length > 524288:
        raise ValueError("adc_length 不得超过 524288")

    byte_length = adc_length * 8
    command = 1

    header = struct.pack("<II", command, byte_length)
    sock.sendall(header)

    num_int16 = adc_length * 4
    raw = recv_all(sock, num_int16 * 2)
    addata = np.frombuffer(raw, dtype="<i2").copy()

    if len(addata) % 16 != 0:
        raise ValueError(f"接收到的 int16 数长度 {len(addata)} 不是 16 的整数倍")

    groups = addata.reshape(-1, 16)

    CH2_i = groups[:, 0:4].reshape(-1)
    CH2_r = groups[:, 4:8].reshape(-1)
    CH1_i = groups[:, 8:12].reshape(-1)
    CH1_r = groups[:, 12:16].reshape(-1)

    ADC1 = CH1_r.astype(np.float64) + 1j * CH1_i.astype(np.float64)
    ADC2 = CH2_r.astype(np.float64) + 1j * CH2_i.astype(np.float64)

    ADC1_max = float(np.max(np.abs(ADC1)))
    ADC2_max = float(np.max(np.abs(ADC2)))

    # 保留 MATLAB 中的 PAPR 归一化
    n1 = np.linalg.norm(ADC1)
    n2 = np.linalg.norm(ADC2)

    if n1 > 0:
        ADC1 = ADC1 / n1 * (len(ADC1) ** 0.5) / np.sqrt(10 ** (papr(ADC1) / 10))
    if n2 > 0:
        ADC2 = ADC2 / n2 * (len(ADC2) ** 0.5) / np.sqrt(10 ** (papr(ADC2) / 10))

    # 去掉前 1000 点
    start_point = 1000
    ADC1 = ADC1[start_point:]
    ADC2 = ADC2[start_point:]

    return ADC1, ADC2, ADC1_max, ADC2_max

def fpga_collect_align(sock: socket.socket, bandwidth, collect_length=524288):
    """
    对应 MATLAB FPGACollectAlign
    返回:
        ADC1, ADC2Aligned, ADC1Max
    """
    # 先丢掉一帧
    _ = fpga_collect_signal(sock, collect_length, draw=False)

    # 再正式采一帧
    ADC1Raw, ADC2Raw, ADC1Max, _ = fpga_collect_signal(sock, collect_length, draw=False)

    # 对齐
    _, ADC2AlignedRaw = align_coarse_norm_python(ADC1Raw, ADC2Raw)

    # 取前 262144 点
    out_len = 262144
    ADC1 = ADC1Raw[:out_len].copy()
    ADC2Aligned = ADC2AlignedRaw[:out_len].copy()

    return ADC1, ADC2Aligned, ADC1Max

def normalize_after_collect_align(ADC1: np.ndarray, ADC2: np.ndarray, ADC1Max: float):
    """
    对应你补充的新归一化：
        scale = ADC1Max / 32768;
        ADC1 = ADC1 / max(abs(ADC1)) * scale;
        ADC2 = ADC2 * sqrt(bandpower(ADC1) / bandpower(ADC2));
    """
    ADC1 = np.asarray(ADC1).reshape(-1).astype(np.complex128)
    ADC2 = np.asarray(ADC2).reshape(-1).astype(np.complex128)

    scale = ADC1Max / 32768.0

    peak1 = np.max(np.abs(ADC1))
    if peak1 == 0:
        raise ValueError("ADC1 全为 0，无法做幅度归一化")

    ADC1n = ADC1 / peak1 * scale

    p1 = bandpower(ADC1n)
    p2 = bandpower(ADC2)
    if p2 == 0:
        raise ValueError("ADC2 功率为 0，无法做功率归一化")

    ADC2n = ADC2 * np.sqrt(p1 / p2)
    return ADC1n, ADC2n

def create_tcp_client(ip_addr="192.168.0.22", port=5001, timeout=5):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((ip_addr, port))
    print(f"Connected to {ip_addr}:{port}")
    return sock
