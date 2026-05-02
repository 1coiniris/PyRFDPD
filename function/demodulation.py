import numpy as np
from scipy.signal import convolve
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