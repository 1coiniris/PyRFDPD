import numpy as np
import scipy.signal as sp_signal

def PA_DVR_v1(In):
    coeff = np.array([
        1.04827998843773 + 0.241152160005451j,
        0.222655182509850 - 0.628143402090827j,
        0.254610619689032 + 0.579125552277299j,
        -0.160807022077543 - 0.222114493803607j,
        0.00226768003956492 + 0.00240245750400001j,
        0.00365782006208160 - 0.00306586858812232j,
        -0.0208333584046602 - 0.000136752049613179j,
        -0.149544209286585 - 0.00931340519001411j,
        0.125260330856312 + 0.00602274336764408j,
        -1.62691139658291e-05 - 0.00148863971972834j,
        -0.00148891822833049 - 0.000836330965187277j,
        0.00717633564604881 - 0.00278863613893021j,
        -0.00607339814113250 + 0.00301216497901613j,
        -0.00145495586562350 + 0.00306162834247278j,
        -0.000166227916275796 + 0.000969076980558328j,
        -0.000669644133145646 + 0.00106525767912307j,
        -0.00142239170544400 + 0.00283228174742312j,
        -0.0309061313694030 - 0.00440437207630190j,
        0.0335553630913401 - 0.00177786938844450j,
        -0.000664299147976805 - 0.00224462795018614j,
        0.00160724996531741 - 0.00212040289336071j,
        -0.00228233666416248 + 9.97440465101973e-05j,
        0.000564479658432926 - 0.00962956839248841j,
        -0.000155683685069546 + 0.0105352952114264j,
        0.00158003967001719 - 0.00567586143921495j,
        -0.000717455997970184 + 0.00811775085930741j,
        0.0308594705537271 - 0.00296087589546351j,
        0.152051784105977 + 0.0105222386866957j,
        -0.172831973326899 - 0.0116015831363012j,
        -0.00164499482689805 + 0.00253371269853459j,
        0.00294868142745101 - 0.000966276506393152j,
        -0.0148347672068408 + 0.00547745309314884j,
        0.00560677094260170 - 0.00149557506260412j,
        0.0180702850418202 - 0.0100129508973224j,
        0.000936682120900950 + 0.00194787314189510j,
        0.00470320757486850 - 0.00430661210166675j,
        0.00115342774459121 - 0.00817264299030479j,
        0.0466415819535346 + 0.000714955829102526j,
        -0.0594899125559211 + 0.0152574488939750j,
        -0.000970011896414713 + 0.00553849404345896j,
        -0.00233853884315438 + 0.00598274933547057j,
        0.00386100550493254 + 0.00117927548780261j,
        0.00421580483211491 + 0.0216860763272922j,
        -0.00278865510939859 - 0.0244271395047192j,
        -0.124391318359899 + 0.0431199833123972j,
        -0.0848253595048931 + 0.119101936700711j,
        0.104700629590704 + 0.106030960736097j,
        0.0281657326547973 + 0.0166031142924679j,
        0.0657572706344267 - 0.275860193845160j,
        0.00229280533029867 - 0.0256788082994137j,
        -0.0120242246191023 - 0.0160155026410203j,
        -0.00119740752690101 - 0.0213278122197783j,
        -0.0216251327939806 - 0.0213637318696761j,
        0.0101077714068642 + 0.120161201757151j,
        -0.00514309863801844 + 0.00926364330484501j,
        0.00703419543712422 - 0.00355867729732138j,
        0.0108705960137818 + 0.00689686255531308j,
        0.0104839794734259 - 0.0192190236510283j,
        -0.0152697862389132 - 0.0177058449868349j,
        -0.0926165373225203 + 0.00581803255207042j,
        0.0508770169646676 - 0.00932056088662063j,
        -0.114557421821780 + 0.00738869149682109j,
        -0.00673455704994730 + 0.0292246942796973j,
        -0.213036458708642 - 0.00770370369360973j,
        -0.0309839980857128 + 0.0180615138276955j,
        -0.0706599653839356 + 0.0610788277390500j,
        -0.0218321224433113 - 0.0129177949485039j,
        0.0172672194462344 - 0.00106675350388774j,
        0.0125503785951798 - 0.0575739864994883j,
        0.0164875796861194 + 0.0338866321618026j,
        0.0159535122471492 - 0.0100601904139929j,
        0.00674339590602674 - 0.00740444279240729j,
        -0.00312790840456839 + 0.000971420218482177j,
        0.00404923897066846 + 0.0285365877757073j,
        -1.05490179485727 + 0.0126977736529565j,
        0.393140426990073 - 0.288375701123869j,
        -0.125115967185505 + 0.219587772567834j,
        -0.0677089545008514 + 0.284825829082659j,
        -0.440845947942112 + 0.308993452428475j,
        1.02977110532445 - 0.0986261734802154j,
        -0.434405703905316 + 0.286885440015464j,
        0.0913842033918055 - 0.206800900957566j,
        0.0242880303950841 - 0.409267416248616j,
        0.344184851652518 + 0.0165653259872958j,
        -0.317508114765206 + 0.0668907623469855j,
        0.200822121430407 - 0.132117414562065j,
        -0.0540311790944986 + 0.0315965487334272j,
        -0.0104682920336954 + 0.185730516961994j,
        -0.0986423632017992 - 0.0529551707073728j,
        1.34588176286796 - 2.29696344827252j,
        -2.02149506477924 + 2.60959599051737j,
        0.301440221557532 + 0.0428884018044129j,
        0.243067251086811 + 0.0207488361270185j,
        0.325475319652135 - 0.282775893375259j,
        -1.49715865330550 + 2.42144803243846j,
        2.43527392863058 - 2.82677549300479j,
        -0.308064556317515 - 0.106030186624632j,
        -0.277213688124964 - 0.270152096663641j,
        0.0478642692023366 + 0.180149249278690j,
        0.604600860502983 - 0.858996270870974j,
        -0.920648513858947 + 1.00589291024699j,
        0.101384421124108 + 0.00242800486140871j,
        0.115112513024480 + 0.218296421731768j,
        -0.0233089661754429 - 0.0299707271982128j
        ], dtype=np.complex128)

    # 输入预处理
    In = np.asarray(In).flatten()  # 确保列向量
    N = len(In)
    In = In / np.max(np.abs(In))  # 归一化
    
    # 生成阈值数组
    thold = np.arange(0.2, 1.2, 0.2)  # 等效MATLAB的0.2:0.2:1
    M = 3
    K = len(thold)
    
    # 初始化角度计算 (注意：MATLAB原代码中的/180可能是错误)
    xi = In.copy()
    xip = np.angle(In) * np.pi / 180  # 保持与MATLAB一致
    
    # 初始化特征矩阵
    X_lin = np.zeros((N, 0), dtype=np.complex128)
    X_1 = np.zeros((N, 0), dtype=np.complex128)
    X_21 = np.zeros((N, 0), dtype=np.complex128)
    X_22 = np.zeros((N, 0), dtype=np.complex128)
    X_23 = np.zeros((N, 0), dtype=np.complex128)
    X_ddr_1 = np.zeros((N, 0), dtype=np.complex128)
    X_ddr_2 = np.zeros((N, 0), dtype=np.complex128)

    # 主处理循环
    for m in range(M+1):  # 等效MATLAB的m=0:M
        # 循环移位
        xi_shift = np.roll(xi, m)
        xip_shift = np.roll(xip, m)
        
        # 线性项
        X_lin = np.hstack((X_lin, xi_shift.reshape(-1, 1)))
        
        # 阈值处理循环
        for k in range(K):
            # 核心计算
            xi_core = np.abs(np.abs(xi_shift) - thold[k])
            
            # X_1项
            xi_1_shift = xi_core * np.exp(1j * xip_shift)
            X_1 = np.hstack((X_1, xi_1_shift.reshape(-1, 1)))
            
            # X_21项
            xi_21_shift = xi_core * np.exp(1j * xip_shift) * np.abs(xi)
            X_21 = np.hstack((X_21, xi_21_shift.reshape(-1, 1)))
            
            if m > 0:  # 当m>0时处理附加项
                # X_22项
                xi_22_shift = xi_core * xi
                X_22 = np.hstack((X_22, xi_22_shift.reshape(-1, 1)))
                
                # X_23项
                xi_23_shift = xi_core * xi_shift
                X_23 = np.hstack((X_23, xi_23_shift.reshape(-1, 1)))
                
                # DDR项
                xi_ddr_core = np.abs(np.abs(xi) - thold[k])
                
                # X_ddr_1项
                xi_ddr_1 = xi_ddr_core * xi_shift
                X_ddr_1 = np.hstack((X_ddr_1, xi_ddr_1.reshape(-1, 1)))
                
                # X_ddr_2项
                xi_ddr_2 = xi_ddr_core * xi * xi * np.conj(xi_shift)
                X_ddr_2 = np.hstack((X_ddr_2, xi_ddr_2.reshape(-1, 1)))

    # 合并所有特征矩阵
    X = np.hstack((X_lin, X_1, X_21, X_22, X_23, X_ddr_1, X_ddr_2))
    
    # 矩阵乘法
    Out = X.dot(coeff).reshape(-1,1)
    Out = Out.squeeze()
    # Out1 = Out.reshape(-1,1)
    return Out


# ======================================================================
# 基础指标: rms / papr / NMSE_ZTE / ACLR_ZTE
# (对应 MATLAB Function-Essential/Caculating_function/)
# ======================================================================
def rms(x):
    return np.sqrt(np.mean(np.abs(np.asarray(x)) ** 2))


def papr(x):
    """PAPR, 返回 (p1_dB, p2_倍数), 对应 papr.m"""
    x = np.asarray(x)
    p2 = np.max(np.abs(x)) ** 2 / rms(x) ** 2
    p1 = 10 * np.log10(p2)
    return p1, p2


def NMSE_ZTE(x, y, a=None, b=None):
    """按功率归一化 NMSE (dB), 对应 NMSE_ZTE.m
    a/b 可选 (1 基): 取 x(a:b), y(a:b)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if a is not None:
        if b is None:
            x = x[a - 1:]
            y = y[a - 1:]
        else:
            x = x[a - 1:b]
            y = y[a - 1:b]
    x = x / np.sqrt(np.mean(x * np.conj(x)))
    y = y / np.sqrt(np.mean(y * np.conj(y)))
    error = x - y
    nmse = 10 * np.log10(np.mean(np.abs(error) ** 2) / np.mean(np.abs(x) ** 2))
    return nmse


def ACLR_ZTE(signal, bandwith, offset, fs):
    """ACPR/ACLR 指标 (dBc), 对应 ACLR_ZTE.m
    signal: 信号; bandwith: 积分带宽 (Hz); offset: 积分间隔 (Hz); fs: 采样率
    返回 (aclr1, aclr2)
    """
    from scipy import signal as sp_signal
    signal = np.asarray(signal).ravel()
    N = 4096
    win = np.hanning(4096)                          # MATLAB hanning(4096)
    f, psd_input = sp_signal.welch(signal, fs=fs, window=win, nperseg=N,
                                   noverlap=N // 2, nfft=N,
                                   return_onesided=False)
    psd_input = np.fft.fftshift(psd_input)          # 'centered'

    M = int(round(bandwith / fs * N))
    M = int(round(M / 2) * 2)
    M1 = int(round(offset / fs * N))
    M1 = int(round(M1 / 2) * 2)

    mid = N // 2
    middle_power = np.sum(np.abs(psd_input[mid - M // 2: mid + M // 2 + 1]))
    adj1_power = np.sum(np.abs(psd_input[mid - M1 - M // 2: mid - M1 + M // 2 + 1]))
    adj2_power = np.sum(np.abs(psd_input[mid + M1 - M // 2: mid + M1 + M // 2 + 1]))
    aclr1 = 10 * np.log10(adj1_power / middle_power)
    aclr2 = 10 * np.log10(adj2_power / middle_power)
    return aclr1, aclr2


# ======================================================================
# CAF_new.m 转换 (削峰: Clipping And Filtering)
# ======================================================================
def CAF_new(x, thr_dB, oversample, T, hard_clipping, thr_new):
    """削峰, 对应 CAF_new.m
    返回 c_caf; 削峰后信号 = x - c_caf
    """
    x = np.asarray(x).ravel()
    L = x.size
    c_caf = np.zeros(L, dtype=complex)
    c = np.zeros(L, dtype=complex)
    thr = rms(x) * 10 ** (thr_dB / 20)
    papr_caf1 = [papr(x)[0]]

    # --- 识别原始信号频带 ---
    X_f = np.fft.fftshift(np.fft.fft(x))
    magnitude_threshold = np.max(np.abs(X_f)) * 1e-2
    freq_mask = np.abs(X_f) > magnitude_threshold

    # --- 削峰 + 滤波迭代 ---
    for t in range(T):
        if papr(x)[0] <= thr_new:
            break
        # --- 削峰 ---
        for i in range(L):
            if abs(x[i]) >= thr:
                c[i] = x[i] * (1 - thr / abs(x[i]))
            else:
                c[i] = 0
        # --- 频带限制滤波 ---
        c_f = np.fft.fftshift(np.fft.fft(c))
        c_f_new = np.zeros(L, dtype=complex)
        c_f_new[freq_mask] = c_f[freq_mask]
        c_new = np.fft.ifft(np.fft.ifftshift(c_f_new))
        x = x - c_new
        c_caf = c_caf + c_new
        papr_caf1.append(papr(x)[0])

    # --- 硬限幅 ---
    if hard_clipping == 1:
        c_hc = np.zeros(L, dtype=complex)
        if np.max(np.abs(x)) >= thr:
            for i in range(L):
                if abs(x[i]) >= thr:
                    c_hc[i] = x[i] * (1 - thr / abs(x[i]))
            c_caf = c_caf + c_hc
        papr_caf2 = papr(x - c_caf)
        _ = papr_caf2
    return c_caf


# ======================================================================
# DVR_getbasis.m / DVR_e.m / DVR_v.m 转换
# ======================================================================
def DVR_getbasis(x, M, threshold):
    """DVR 基函数构造, 对应 DVR_getbasis.m
    x: 输入复信号; M: 记忆深度; threshold: 阈值向量
    返回 X: (N, 列数) 基函数矩阵
    """
    x = np.asarray(x).ravel()
    x = x / np.max(np.abs(x))
    thold = np.asarray(threshold).ravel()
    K = thold.size
    xi = x
    xip = np.angle(x)

    X_lin = []
    X_1 = []
    X_21 = []
    X_22 = []
    X_23 = []
    X_ddr_1 = []
    X_ddr_2 = []
    for m in range(M + 1):
        xi_shift = np.roll(xi, m)                   # circshift(xi, m)
        xip_shift = np.roll(xip, m)                 # circshift(xip, m)
        X_lin.append(xi_shift)
        for kth in range(K):
            xi_core = np.abs(np.abs(xi_shift) - thold[kth])
            X_1.append(xi_core * np.exp(1j * xip_shift))
            X_21.append(xi_core * np.exp(1j * xip_shift) * np.abs(xi))
            if m > 0:
                X_22.append(xi_core * xi)
                X_23.append(xi_core * xi_shift)
                xi_ddr_core = np.abs(np.abs(xi) - thold[kth])
                X_ddr_1.append(xi_ddr_core * xi_shift)
                X_ddr_2.append(xi_ddr_core * xi * xi * np.conj(xi_shift))
    X = np.column_stack(X_lin + X_1 + X_21 + X_22 + X_23 + X_ddr_1 + X_ddr_2)
    return X


def DVR_e(x, y, M, threshold, alpha, print_flag):
    """DVR 模型辨识, 对应 DVR_e.m
    x: 输入; y: 期望输出; M: 记忆深度; threshold: 阈值向量
    alpha: 正则化系数; print_flag: 是否打印训练 NMSE
    返回 coef (系数向量)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x / np.max(np.abs(x))
    y = y / np.max(np.abs(y))
    X = DVR_getbasis(x, M, threshold)
    start = M + 1 + 10
    X1 = X[start:, :]
    y1 = y[start:]
    I = np.eye(X1.shape[1], dtype=complex)
    beta = np.linalg.pinv(X1.conj().T @ X1 + alpha * I) @ X1.conj().T @ y1
    coef = beta
    if print_flag == 1:
        y_model = X @ beta
        y_model[:start] = y[:start]
        A = np.nonzero(np.abs(y_model) > 1)[0]
        y_model[A] = y[A]
        nmse_val = NMSE_ZTE(y, y_model)
        print(f'NMSE-train-model = {nmse_val:.2f} dB')
    return coef


def DVR_v(x, coef, M, threshold):
    """DVR 预失真输出, 对应 DVR_v.m
    x: 输入; coef: DVR_e 得到的系数; M: 记忆深度; threshold: 阈值向量
    返回 y
    """
    x = np.asarray(x).ravel()
    x = x / np.max(np.abs(x))
    X = DVR_getbasis(x, M, threshold)
    start = M + 1 + 10
    y = X @ np.asarray(coef)
    y[:start] = x[:start]
    A = np.nonzero(np.abs(y) > 1)[0]
    y[A] = x[A]
    return y


# ======================================================================
# psd_crz.m / psd_b.m / amam.m / ampm.m 转换 (绘图)
# ======================================================================
def _psd_core(x, Fs, N):
    """PSD 计算与绘图核心 (psd_crz / psd_b 共用)"""
    import matplotlib.pyplot as plt
    x = np.asarray(x).ravel()
    x = x / np.linalg.norm(x) * 300
    win = np.hanning(N)
    f, Pxx = sp_signal.welch(x, fs=Fs, window=win, nperseg=N,
                             noverlap=N // 2, nfft=N, return_onesided=False)
    Pxx = 10 * np.log10(np.fft.fftshift(Pxx))
    f = Fs * (np.arange(Pxx.size) / Pxx.size - 0.5) / 1e6
    return f, Pxx


def psd_crz(x, Fs, N, color=None, lineform='-', linewidth=2, ax=None, **kwargs):
    """频谱绘图, 对应 psd_crz.m; 返回 (f, Pxx)"""
    import matplotlib.pyplot as plt
    f, Pxx = _psd_core(x, Fs, N)
    if ax is None:
        ax = plt.gca()
    if color is None:
        ax.plot(f, Pxx, '-', linewidth=linewidth)
    else:
        ax.plot(f, Pxx, lineform, linewidth=linewidth, color=color, **kwargs)
    ax.set_xlabel('Frequency Offset (MHZ)')
    ax.set_ylabel('Normalized Spectrum Density (dBm)')
    return f, Pxx


def psd_b(x, Fs, N, displayname=None, ax=None, **kwargs):
    """频谱绘图 (带 DisplayName), 对应 psd_b.m; 返回 (f, Pxx)"""
    import matplotlib.pyplot as plt
    f, Pxx = _psd_core(x, Fs, N)
    if ax is None:
        ax = plt.gca()
    if displayname is None:
        ax.plot(f, Pxx, '-', linewidth=2, **kwargs)
    else:
        ax.plot(f, Pxx, '-', linewidth=2, label=displayname, **kwargs)
    ax.set_xlabel('Frequency Offset (MHZ)')
    ax.set_ylabel('Normalized Spectrum Density (dBm)')
    return f, Pxx


def amam(x, y, color='r', norm=1, a=None, b=None, ax=None):
    """AM/AM 图, 对应 amam.m"""
    import matplotlib.pyplot as plt
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if a is not None:
        if b is None:
            x = x[a - 1:]
            y = y[a - 1:]
        else:
            x = x[a - 1:b]
            y = y[a - 1:b]
    if norm == 0:
        y1 = np.abs(y)
        x1 = np.abs(x)
    else:
        y1 = np.abs(y) / np.max(np.abs(y))
        x1 = np.abs(x) / np.max(np.abs(x))
    if ax is None:
        ax = plt.gca()
    if color == 0:
        ax.plot(x1, y1, '.')
    else:
        ax.plot(x1, y1, '.', color=color, markersize=5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])


def ampm(x, y, color='r', norm=1, a=None, b=None, ax=None):
    """AM/PM 图, 对应 ampm.m"""
    import matplotlib.pyplot as plt
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if a is not None:
        if b is None:
            x = x[a - 1:]
            y = y[a - 1:]
        else:
            x = x[a - 1:b]
            y = y[a - 1:b]
    if norm == 0:
        y1 = np.abs(y)
        x1 = np.abs(x)
    else:
        y1 = np.abs(y) / np.max(np.abs(y))
        x1 = np.abs(x) / np.max(np.abs(x))
    z1 = np.angle(y / x) / np.pi * 180
    if ax is None:
        ax = plt.gca()
    ax.plot(x1, z1, '.', color=color, markersize=5)
    ax.set_xlim([0, 1])


# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    # 生成测试信号
    N = 1000
    In = np.random.randn(N) + 1j*np.random.randn(N)
    
    # 调用函数
    out = PA_DVR_v1(In)
    
    print("输出信号示例：")
    print(out[:5])  # 打印前5个输出样本

