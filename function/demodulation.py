import numpy as np
from scipy.signal import convolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def demodulate_16384qam(rx_signal, data_symbols, pilot_symbols,rrc_filt,
                        alpha=0.4, sps=7, plot_enable=True,savepath = 'figures'):
    """
    解调 16384-QAM 接收信号，绘制星座图并返回 EVM(%)

    参数:
        rx_signal : 复数基带信号 (1D array)
        data_symbols : 实际发送的数据符号 (1D complex array)
        pilot_symbols: 实际发送的导频符号 (1D complex array)
        alpha, sps   : 升余弦滤波器参数（需与发射相同）
        plot_enable  : 是否显示/保存星座图
    """

    pilot_symbols = np.asarray(pilot_symbols).ravel()   # 确保一维
    data_symbols = np.asarray(data_symbols).ravel()

    grp_delay = (len(rrc_filt) - 1) // 2

    # --- 匹配滤波 ---
    rx_filt = convolve(rx_signal, rrc_filt, mode='full')
    rx_filt = rx_filt[grp_delay: -grp_delay if grp_delay > 0 else None]

    num_pilot = len(pilot_symbols)
    # 导频相关搜索（可扩大范围）
    search_len = len(rx_filt) - num_pilot * sps
    corr = np.zeros(search_len)
    pconj = pilot_symbols.conj()
    for i in range(search_len):
        idx = i + np.arange(num_pilot) * sps
        corr[i] = np.abs(np.dot(pconj, rx_filt[idx]))
    start = np.argmax(corr)

    # 抽取所有符号
    total_sym = num_pilot + len(data_symbols)
    idx_sym = start + np.arange(total_sym) * sps
    if idx_sym[-1] >= len(rx_filt):
        valid = idx_sym < len(rx_filt)
        idx_sym = idx_sym[valid]
    rx_sym = rx_filt[idx_sym]

    rx_pilot = rx_sym[:num_pilot]
    rx_data = rx_sym[num_pilot:]

    # 信道估计与补偿
    gain = np.dot(pconj, rx_pilot) / np.dot(pconj, pilot_symbols)
    calibrated = rx_data / gain

    # EVM 计算
    N = min(len(calibrated), len(data_symbols))
    ref = data_symbols[:N]
    meas = calibrated[:N]
    error = meas - ref
    evm_rms = np.sqrt(np.mean(np.abs(error) ** 2)) / np.sqrt(np.mean(np.abs(ref) ** 2))
    evm_pct = evm_rms * 100

    # 星座图
    if plot_enable:
        plt.figure(figsize=(6, 6))
        plt.plot(meas.real, meas.imag, '.', markersize=2)
        plt.axis('square');
        plt.grid(True)
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.title('Received Constellation (16384-QAM)')
        plt.xlabel('I');
        plt.ylabel('Q')
        plt.tight_layout()
        # 自动处理无显示环境
        plt.savefig(f'{savepath}/constellation_16384qam.png', dpi=150)
        plt.close()

    return evm_pct


def _qam_slice_to_int(symbols, M):
    """
    QAM 硬判决 — 匹配 MATLAB 的 qamdemod(..., 'UnitAveragePower', true, 'OutputType', 'integer')。

    对方形 QAM (M = L^2)，UnitAveragePower=true 时：
      - 星座点位置: (2i-L+1 + 1j*(2q-L+1)) * sqrt(6/(M-1))，其中 i,q = 0..L-1
      - Gray 编码: 前 k/2 bits → I，后 k/2 bits → Q

    返回整数判决值 0..M-1。
    """
    # 防御性展平: 保留原始 shape 以便 reshape 回来
    orig_shape = symbols.shape
    symbols_flat = symbols.ravel(order='F')

    k = int(np.log2(M))
    L = int(np.sqrt(M))
    assert L * L == M, f"M={M} 不是方形 QAM"

    # 归一化因子 (MATLAB UnitAveragePower)
    d = np.sqrt(6.0 / (M - 1))

    # 缩放到整数网格并找最近电平
    I_int = np.round(np.real(symbols_flat) / d).astype(np.int64)
    Q_int = np.round(np.imag(symbols_flat) / d).astype(np.int64)

    # 裁剪到有效范围
    max_val = L - 1
    I_int = np.clip(I_int, -max_val, max_val)
    Q_int = np.clip(Q_int, -max_val, max_val)

    # 转换为索引 0..L-1: 电平 -127 → idx 0, -125 → 1, ..., 127 → 127
    idx_I = ((I_int + max_val) // 2).astype(np.int64)
    idx_Q = ((Q_int + max_val) // 2).astype(np.int64)

    # Gray 解码: binary = gray ^ (gray >> 1) ^ (gray >> 2) ^ ...
    # 最多 7 bits (L <= 128)，链式移位足够
    def gray_decode(x):
        x = x ^ (x >> 1)
        x = x ^ (x >> 2)
        x = x ^ (x >> 4)
        return x

    bin_I = gray_decode(idx_I)
    bin_Q = gray_decode(idx_Q)

    # 重组整数: I bits (MSB), Q bits (LSB)
    half_k = k // 2
    int_out = (bin_I << half_k) | bin_Q

    return int_out.reshape(orig_shape, order='F')


def demodulate_ofdm(rx_signal, N_fft, N_cp, data_idx, pilot_idx,
                    dataSymbols, pilotSymbols, M, dataBits=None,
                    plot_enable=True, savepath='figures',
                    labeling='OFDM'):
    """
    W-OFDM (CP-OFDM) 解调，计算 EVM(%)、BER 及星座图。

    完整流程:
        1. 从信号长度推断符号数 N_sym，截断到整数 OFDM 符号
        2. 重构成 (N_fft+N_cp) × N_sym 矩阵 (列优先 = 一个符号)
        3. 去除 CP → (N_fft, N_sym)
        4. FFT + fftshift → 频域 (N_fft, N_sym)
        5. 提取数据/导频子载波
        6. 导频信道估计 (线性插值)
        7. 迫零均衡
        8. 计算 EVM
        9. (可选) QAM 硬判决 + BER

    参数:
        rx_signal   : 复数基带接收信号 (1D array)
        N_fft       : FFT 点数
        N_cp        : CP 长度 (采样点)
        data_idx    : 数据子载波索引 (MATLAB 1-based, shape (N_data,))
        pilot_idx   : 导频子载波索引 (MATLAB 1-based, shape (N_pilot,))
        dataSymbols : 参考数据符号, shape (N_data, N_sym) 或 (N_data*N_sym,) 列优先
        pilotSymbols: 参考导频符号, shape (N_pilot, N_sym)
        M           : QAM 调制阶数
        dataBits    : (可选) 原始发送比特 (1D array)，用于 BER 计算
        plot_enable : 是否保存星座图
        savepath    : 图像保存路径
        labeling    : 图像标题标签

    返回:
        evm_pct     : EVM 百分比
        ber_val     : BER (若 dataBits 提供), 否则 None
        num_err     : 错误比特数 (若 dataBits 提供), 否则 None
    """
    rx_signal = np.asarray(rx_signal).ravel()
    spp = N_fft + N_cp
    total_len = len(rx_signal)

    # ---- 对齐符号数 ----
    N_sym = total_len // spp
    if N_sym * spp != total_len:
        print(f'[OFDM] 信号长度 {total_len} 不是 spp={spp} 的整数倍, '
              f'截断到 {N_sym * spp} ({N_sym} 符号)')
        rx_signal = rx_signal[:N_sym * spp]

    # ---- 确保 dataSymbols/pilotSymbols 是 2D ----
    dataSymbols = np.asarray(dataSymbols)
    pilotSymbols = np.asarray(pilotSymbols)
    N_data_ref = len(np.asarray(data_idx).ravel())

    if dataSymbols.ndim == 1:
        dataSymbols = dataSymbols.reshape((N_data_ref, -1), order='F')
    if pilotSymbols.ndim == 1:
        pilotSymbols = pilotSymbols.reshape((-1, dataSymbols.shape[1]), order='F')

    N_data, N_sym_ref = dataSymbols.shape
    N_pilot = pilotSymbols.shape[0]

    # 如果 ref 的符号数与信号不匹配，截取较短者
    N_sym_use = min(N_sym, N_sym_ref)
    if N_sym != N_sym_ref:
        print(f'[OFDM] N_sym 不匹配: 信号={N_sym}, 参考={N_sym_ref}, '
              f'使用 {N_sym_use}')
    dataSymbols = dataSymbols[:, :N_sym_use]
    pilotSymbols = pilotSymbols[:, :N_sym_use]

    # ---- 1. 重组为 OFDM 符号矩阵 (列优先, 每列一个符号) ----
    rxMat = np.reshape(rx_signal, (spp, N_sym_use), order='F')

    # ---- 2. 去 CP ----
    rxMat = rxMat[N_cp:, :]  # (N_fft, N_sym_use)

    # ---- 3. FFT + fftshift ----
    rxFreq = np.fft.fftshift(
        np.fft.fft(rxMat, n=N_fft, axis=0), axes=0
    )  # (N_fft, N_sym_use)

    # ---- 4. 提取子载波 ----
    data_idx_0 = np.asarray(data_idx).ravel().astype(int) - 1   # 1-based → 0-based
    pilot_idx_0 = np.asarray(pilot_idx).ravel().astype(int) - 1

    rxData = rxFreq[data_idx_0, :]    # (N_data, N_sym_use)
    rxPilots = rxFreq[pilot_idx_0, :]  # (N_pilot, N_sym_use)

    # ---- 5. 导频信道估计 ----
    H_pilot = rxPilots / pilotSymbols  # (N_pilot, N_sym_use)

    # 线性插值到数据子载波
    H_data = np.zeros((N_data, N_sym_use), dtype=complex)
    for n in range(N_sym_use):
        f_real = interp1d(pilot_idx_0, H_pilot[:, n].real, kind='linear',
                          fill_value='extrapolate', bounds_error=False)
        f_imag = interp1d(pilot_idx_0, H_pilot[:, n].imag, kind='linear',
                          fill_value='extrapolate', bounds_error=False)
        H_data[:, n] = f_real(data_idx_0) + 1j * f_imag(data_idx_0)

    # ---- 6. 迫零均衡 ----
    eqData = rxData / H_data  # (N_data, N_sym_use)

    # ---- 7. EVM ----
    eqSym = eqData.ravel(order='F')
    refSym = dataSymbols.ravel(order='F')
    mL = min(len(eqSym), len(refSym))
    err = eqSym[:mL] - refSym[:mL]
    evm_val = (np.sqrt(np.mean(np.abs(err) ** 2))
               / np.sqrt(np.mean(np.abs(refSym[:mL]) ** 2)) * 100)

    # ---- 8. QAM 硬判决 + BER (可选) ----
    ber_val = None
    num_err = None
    if dataBits is not None:
        dataBits = np.asarray(dataBits).ravel()
        k = int(np.log2(M))

        # 整数判决
        rxSym_int = _qam_slice_to_int(eqData, M)  # 保持 shape (N_data, N_sym_use)

        # 整数 → bits (MSB first, 列优先展平以匹配 MATLAB de2bi(...,'left-msb') 转置压平)
        n_total = rxSym_int.size
        rxBits = np.zeros(n_total * k, dtype=np.uint8)
        for i, val in enumerate(rxSym_int.ravel(order='F')):
            for b in range(k):
                rxBits[i * k + b] = (val >> (k - 1 - b)) & 1

        # 对齐长度
        min_len = min(len(rxBits), len(dataBits))
        rxBits = rxBits[:min_len]
        txBits = dataBits[:min_len]

        num_err = int(np.sum(rxBits != txBits))
        ber_val = num_err / min_len

    # ---- 9. 星座图 ----
    if plot_enable:
        plt.figure(figsize=(6, 6))
        plot_n = min(5000, mL)
        plt.plot(eqSym[:plot_n].real, eqSym[:plot_n].imag, 'b.', markersize=2)
        plt.axis('square')
        plt.grid(True)
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.title(f'OFDM {M}-QAM ({labeling})  EVM={evm_val:.2f}%')
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.tight_layout()
        fname = f'{savepath}/constellation_ofdm_{M}qam_{labeling}.png' \
                .replace(' ', '_').replace('=', '_')
        plt.savefig(fname, dpi=150)
        plt.close()

    print(f'[OFDM] EVM = {evm_val:.4f}%', end='')
    if ber_val is not None:
        print(f', BER = {ber_val:.6g} ({num_err}/{min_len})', end='')
    print()

    return evm_val, ber_val, num_err