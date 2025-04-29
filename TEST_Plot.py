import numpy as np
import matplotlib.pyplot as plt
from pyrfdpd.utils.metrics import rms
import matplotlib.pyplot as plt


def plot_power_spectrum(signals: dict, fs, freq_range=None, nfft=1024):
    """
    绘制信号的功率谱密度图
    
    参数：
    xin   : 输入信号（复数或实数）
    fs    : 采样率（Hz）
    freq_range : 横轴显示范围 [min, max]（MHz），默认显示全带宽
    ax    : 可选的matplotlib轴对象
    nfft  : FFT点数（默认使用信号长度）
    
    返回：
    fig, ax : matplotlib的figure和axes对象
    """
    # plt.figure(dpi=300, figsize=(16, 10))
    # 改变文字大小参数-fontsize
    plt.xticks(fontsize=20)
    fig, ax = plt.subplots(figsize=(16, 10))
    for name, signal in signals.items():
        nfft = nfft or len(signal)
        signal = signal / rms(signal)
        X = np.fft.fft(signal, n=nfft)
        power = np.abs(X) ** 2
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs)) / 1e6  # 转换为MHz
        power_shifted = np.fft.fftshift(power)
        # 转换为dB
        power_db = 10 * np.log10(power_shifted + 1e-12)  # 避免log(0)
        # 应用频率范围限制
        if freq_range is not None:
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            freqs = freqs[mask]
            power_db = power_db[mask]
        # 绘制曲线
        ax.plot(freqs, power_db, linewidth=1)

        # plt.psd(signal, NFFT=1024, Fs=fs, scale_by_freq=True, linewidth=3, label=name)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power (dB)')
    ax.set_title('Power Spectrum Density')
    ax.grid(True)
    # ax.set_xlim(freqs[0], freqs[-1])
    # plt.ylim(-100, 0)
    # 设置默认参数
    # nfft = nfft or len(xin)
    
    # 计算FFT
    # X = np.fft.fft(xin, n=nfft)
    # power = np.abs(X) ** 2
    
    # 创建频率轴（MHz单位）
    # freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1/fs)) / 1e6  # 转换为MHz
    # power_shifted = np.fft.fftshift(power)
    
    # 转换为dB
    # power_db = 10 * np.log10(power_shifted + 1e-12)  # 避免log(0)
    #
    # # 应用频率范围限制
    # if freq_range is not None:
    #     mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    #     freqs = freqs[mask]
    #     power_db = power_db[mask]
    #
    # # 创建图形
    # if ax is None:
    #     fig, ax = plt.subplots(figsize=(10, 5))
    # else:
    #     fig = ax.figure
    #
    # # 绘制曲线
    # ax.plot(freqs, power_db, linewidth=1)
    # ax.set_xlabel('Frequency (MHz)')
    # ax.set_ylabel('Power (dB)')
    # ax.set_title('Power Spectrum Density')
    # ax.grid(True)
    
    # # 设置坐标轴范围
    # if freq_range is not None:
    #     plt.set_xlim(freq_range)
    # else:
    #     ax.set_xlim(freqs[0], freqs[-1])
    
    # return fig, ax


def plot_amam(x, y, filename="amam.png", lang="en"):
    x = x / max(abs(x))
    y = y / max(abs(y))
    for x_i,y_i in zip(x,y):
        # signal = signal / abs(signal)
        plt.scatter(abs(x_i), abs(y_i), marker=".", s = 10)
    if (lang == "en"):
        plt.xlabel("Normalized Input Amplitude")
        plt.ylabel("Normalized Output Amplitude")
    elif (lang == "zh"):
        plt.xlabel("归一化输入幅度")
        plt.ylabel("归一化输出幅度")
    else:
        raise ValueError("Language not supported")
    plt.grid(True, linestyle='--')
    plt.savefig(filename)
    # plt.close()