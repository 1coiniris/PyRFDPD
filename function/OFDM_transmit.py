# -*- coding: utf-8 -*-
"""
OFDM_transmit.py — MATLAB OFDM_transmit 文件夹的 Python 转换版
=============================================================
对应 CODE_MATLAB/OFDM_transmit 下的:
    ofdm_params.m / ofdm_modulate.m / ofdm_demodulate.m / ofdm_tx_filter.m
    bytes_to_bits.m / bits_to_bytes.m / qam_soft_llr.m
    ldpc_pcm.m / fec_ldpc_encode.m / fec_ldpc_decode.m
    fec_rs_encode.m / fec_rs_decode.m
    zip_ofdm_tx.m / zip_ofdm_rx.m / zip_ofdm_tx_pkt.m / zip_ofdm_rx_pkt.m
    zip_ofdm_stream_txrx.m
外加 MATLAB Communications Toolbox 的 qammod/qamdemod 的等价实现
(Gray 映射 + UnitAveragePower)。

依赖: 仅 numpy / scipy (自包含, 无需 Communications Toolbox)。
注意: LDPC 使用确定性种子构造的规则 LDPC 码 (功能等价, 码型与
      802.11n 标准码不同但收发自洽, 支持任意码率);
      RS 使用 GF(256) (本原多项式 285) 的 (255, k) 系统码, 与 MATLAB 一致。
"""

import os
import numpy as np
from scipy.signal import lfilter
from scipy.special import i0 as besseli0

__all__ = [
    'ofdm_params', 'ofdm_modulate', 'ofdm_demodulate', 'ofdm_tx_filter',
    'bytes_to_bits', 'bits_to_bytes', 'qammod', 'qamdemod', 'qam_soft_llr',
    'scramble_bits', 'ldpc_pcm', 'fec_ldpc_encode', 'fec_ldpc_decode',
    'fec_rs_encode', 'fec_rs_decode',
    'zip_ofdm_tx', 'zip_ofdm_rx', 'zip_ofdm_tx_pkt', 'zip_ofdm_rx_pkt',
    'zip_ofdm_stream_txrx',
    'ofdm_tx_packet', 'ofdm_rx_packet', 'ofdm_tx_file', 'ofdm_rx_file',
]


# ======================================================================
# 基础工具: 字节 <-> 比特 (MSB 在前, 与 MATLAB 一致)
# ======================================================================
def bytes_to_bits(bytes_in):
    """uint8 字节数组 -> 0/1 比特向量 (每字节 8 比特, MSB 在前)"""
    b = np.asarray(bytes_in, dtype=np.uint8).ravel()
    return np.unpackbits(b, bitorder='big').astype(np.uint8)


def bits_to_bytes(bits):
    """0/1 比特向量 -> uint8 字节数组 (与 bytes_to_bits 互逆)"""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    assert bits.size % 8 == 0, '比特数必须是 8 的整数倍'
    return np.packbits(bits, bitorder='big')


# ======================================================================
# QAM 调制 / 解调 (MATLAB qammod/qamdemod 等价, Gray 映射 + UnitAveragePower)
# ======================================================================
def _gray_encode(idx, k):
    """整数索引 -> k 位二进制反射格雷码 (返回 k 位整数)"""
    return idx ^ (idx >> 1)


def _gray_decode(g, k):
    """k 位格雷码整数 -> 普通二进制整数"""
    b = g
    for _ in range(k):
        g >>= 1
        b ^= g
    return b


_qam_tbl_cache = {}


def _qam_table(M):
    """构建 (const, bit_table, inv_idx) 星座表
    const:    M 个星座点 (UnitAveragePower), 与 MATLAB qammod 一致:
             索引 i 的二进制 (MSB first) 前 m 位经 Gray 逆变换 -> I 电平索引,
             后 m 位经 Gray 逆变换 -> Q 电平索引 (标准 Gray 方形 QAM)。
    bit_table: (M, k) 每符号的比特 (MSB first, 即 de2bi(i,k,'left-msb'))
    inv_idx:  (2^m, 2^m) 电平索引 -> 符号索引
    """
    if M in _qam_tbl_cache:
        return _qam_tbl_cache[M]
    k = int(round(np.log2(M)))
    m = k // 2
    assert 2 ** k == M and 2 * m == k, f'M 必须是 2 的偶数次幂, got {M}'
    L = 2 ** m - 1
    levels = np.arange(-L, L + 1, 2)            # 升序电平
    idx = np.arange(M)
    p_i = _gray_decode_vec(idx >> m, m)          # 高 m 位 Gray 逆变换 -> I 电平索引
    p_q = _gray_decode_vec(idx & (2 ** m - 1), m)  # 低 m 位 Gray 逆变换 -> Q 电平索引
    const = (levels[p_i] + 1j * levels[p_q]).astype(complex)
    scale = np.sqrt(2.0 * (M - 1) / 3.0)         # UnitAveragePower 缩放
    const = const / scale
    # 比特表: 符号索引 i 的比特 = de2bi(i, k, 'left-msb')
    bit_table = ((idx[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8)
    # 逆映射: 电平索引 (p_i, p_q) -> 符号索引
    inv_idx = np.full((2 ** m, 2 ** m), -1, dtype=np.int64)
    inv_idx[p_i, p_q] = idx
    tbl = (const, bit_table, inv_idx, k, m, levels)
    _qam_tbl_cache[M] = tbl
    return tbl


def _gray_decode_vec(g, m):
    """向量化 m 位格雷码 -> 二进制 (g: 整数数组, 值域 [0, 2^m))"""
    b = g.copy()
    for _ in range(m):
        g = g >> 1
        b = b ^ g
    return b


def qammod(bits, M, input_type='bit', unit_average_power=True):
    """QAM 调制: bits (0/1, MSB first, 长度 k 的整数倍) -> 星座符号列向量
    等价 MATLAB: qammod(bits, M, 'InputType','bit','UnitAveragePower',true)
    """
    assert input_type == 'bit'
    assert unit_average_power
    const, _, _, k, _, _ = _qam_table(M)
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    nSym = bits.size // k
    bmat = bits[:nSym * k].reshape(nSym, k)
    idx = (bmat * (1 << np.arange(k - 1, -1, -1))).sum(axis=1).astype(np.int64)
    return const[idx]


def qamdemod(sym, M, output_type='bit', unit_average_power=True):
    """QAM 硬判决解调 -> 比特 (MSB first), 等价 MATLAB qamdemod 'bit'"""
    assert output_type == 'bit'
    assert unit_average_power
    const, bit_table, inv_idx, k, m, levels = _qam_table(M)
    sym = np.asarray(sym).ravel()
    # 逐维最近电平 (方形 QAM 的最近邻 = 每维最近电平)
    scale = np.sqrt(2.0 * (M - 1) / 3.0)
    x = np.real(sym) * scale
    y = np.imag(sym) * scale
    nL = 2 ** m
    lo, hi = levels[0], levels[-1]
    p_i = np.clip(np.round((x - lo) / 2.0), 0, nL - 1).astype(np.int64)
    p_q = np.clip(np.round((y - lo) / 2.0), 0, nL - 1).astype(np.int64)
    idx = inv_idx[p_i, p_q]
    # idx -> bits (MSB first)
    bits = ((idx[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8)
    return bits.ravel()


# ======================================================================
# qam_soft_llr.m 转换: 快速 max-log 软判决 LLR (O(k)/符号, 无 ±Inf)
# ======================================================================
_llr_tbl_cache = {}


def _llr_table(M):
    """构建 nearIdxI/nearIdxQ 表 (与 qam_soft_llr.m 的 persistent tbl 相同)"""
    if M in _llr_tbl_cache:
        return _llr_tbl_cache[M]
    const, bit_table, _, k, m, _ = _qam_table(M)
    syms = const
    lv = np.sort(np.unique(np.real(syms)))          # PAM 电平 (升序)
    nL = lv.size
    bits_all = bit_table                              # (M, k) MSB first
    # 每电平的 I/Q 维 m 位 (实测, 与 MATLAB 一致)
    i_bits = np.zeros((nL, m), dtype=np.int64)
    q_bits = np.zeros((nL, m), dtype=np.int64)
    for j in range(nL):
        idxI = np.nonzero(np.abs(np.real(syms) - lv[j]) < 1e-9)[0][0]
        idxQ = np.nonzero(np.abs(np.imag(syms) - lv[j]) < 1e-9)[0][0]
        i_bits[j] = bits_all[idxI, :m]
        q_bits[j] = bits_all[idxQ, m:k]
    # nearIdxI/nearIdxQ (j, b, v): 距电平 j 最近、第 b 位为 v 的电平索引
    near_idx_i = np.zeros((nL, m, 2), dtype=np.int64)
    near_idx_q = np.zeros((nL, m, 2), dtype=np.int64)
    for b in range(m):
        for v in (0, 1):
            setI = np.nonzero(i_bits[:, b] == v)[0]
            setQ = np.nonzero(q_bits[:, b] == v)[0]
            for j in range(nL):
                near_idx_i[j, b, v] = setI[np.argmin(np.abs(lv[setI] - lv[j]))]
                near_idx_q[j, b, v] = setQ[np.argmin(np.abs(lv[setQ] - lv[j]))]
    tbl = (lv, near_idx_i, near_idx_q, m, k)
    _llr_tbl_cache[M] = tbl
    return tbl


def qam_soft_llr(rxSym, M, sigma2):
    """快速 max-log 软判决 LLR (与 qam_soft_llr.m 一致)
    LLR > 0 表示比特 1; 长度 numel(rxSym)*log2(M), 逐符号 MSB 在前。
    """
    lv, near_idx_i, near_idx_q, m, k = _llr_table(M)
    rxSym = np.asarray(rxSym).ravel()
    nL = lv.size
    spacing = lv[1] - lv[0]
    x = np.real(rxSym)
    y = np.imag(rxSym)
    jx = np.clip(np.round((x - lv[0]) / spacing).astype(np.int64), 0, nL - 1)
    jy = np.clip(np.round((y - lv[0]) / spacing).astype(np.int64), 0, nL - 1)
    nS = x.size
    llr = np.zeros(nS * k)
    for b in range(m):
        d0x = (x - lv[near_idx_i[jx, b, 0]]) ** 2
        d1x = (x - lv[near_idx_i[jx, b, 1]]) ** 2
        d0y = (y - lv[near_idx_q[jy, b, 0]]) ** 2
        d1y = (y - lv[near_idx_q[jy, b, 1]]) ** 2
        llr[np.arange(nS) * k + b] = (d0x - d1x) / sigma2      # I 维位
        llr[np.arange(nS) * k + m + b] = (d0y - d1y) / sigma2  # Q 维位
    return llr


# ======================================================================
# ofdm_params.m 转换
# ======================================================================
def ofdm_params(BW, M, fs=None, cpRatio=1 / 8, pilotRatio=0.08):
    """生成 OFDM 系统参数字典 (字段与 ofdm_params.m 的 p 对应, 索引为 0 基)"""
    if fs is None:
        fs = 2 * BW
    assert BW > 0 and fs > 0, 'BW 与 fs 必须为正'
    assert fs >= 2 * BW, '采样率需满足 fs >= 2*BW (Nyquist)'
    k = int(round(np.log2(M)))
    assert 2 ** k == M and k % 2 == 0 and M >= 4, \
        'M 必须是 2 的偶数次幂 (如 256/1024/4096/16384)'

    N_fft = int(2 ** np.ceil(np.log2(fs / 100e3)))   # 2^nextpow2
    N_fft = max(N_fft, 64)
    N_used = int(round(BW / fs * N_fft))
    N_used = min(N_used, int(round(0.92 * N_fft)))
    N_used = N_used - (N_used % 2)                   # 取偶数
    N_used = max(N_used, 4)

    if pilotRatio > 0:
        nPilot = max(2, int(round(N_used * pilotRatio)))
        nPilot = min(nPilot, N_used // 2)
        nPilot = nPilot - (nPilot % 2)
        nPilot = max(nPilot, 2)
        spacing = int(np.floor(N_used / nPilot))
        # MATLAB: round(spacing/2):spacing:N_used 取前 nPilot 个 (1 基位置)
        pilotPos = np.arange(round(spacing / 2), N_used + 1, spacing,
                             dtype=np.int64)
        pilotPos = pilotPos[:nPilot]
    else:
        nPilot = 0
        pilotPos = np.array([], dtype=np.int64)
    dataPos = np.setdiff1d(np.arange(1, N_used + 1), pilotPos)  # 1 基打包位置
    N_data = N_used - nPilot

    p = {
        'BW': BW, 'fs': fs, 'M': M, 'k': k,
        'N_fft': N_fft, 'fsc': fs / N_fft, 'N_used': N_used,
        'N_data': N_data, 'nPilot': nPilot,
        'N_cp': int(round(N_fft * cpRatio)), 'cpRatio': cpRatio,
        'occBW': N_used * fs / N_fft,
        'Tsym': (N_fft + int(round(N_fft * cpRatio))) / fs,
        'rateMbps': N_data * k * fs / (N_fft + int(round(N_fft * cpRatio))) / 1e6,
    }
    nHalf = N_used // 2
    posBins = np.arange(1, nHalf + 1)                # 0 基: 正频率子载波
    negBins = np.arange(N_fft - nHalf, N_fft)        # 0 基: 负频率子载波
    allDataBins = np.concatenate([posBins, negBins])  # 打包顺序: 正频在前
    p['posBins'] = posBins
    p['negBins'] = negBins
    p['allDataBins'] = allDataBins
    p['pilotPos'] = pilotPos - 1                     # 1 基 -> 0 基
    p['dataPos'] = dataPos - 1
    p['pilotBins'] = allDataBins[pilotPos - 1]
    p['dataBins'] = allDataBins[dataPos - 1]

    # 已知导频序列 (ZC 风格, 单位功率)
    kk = np.arange(nPilot)
    if nPilot > 0:
        pilotSeq = np.exp(1j * np.pi * 29 * kk * (kk + 1) / max(nPilot, 1))
        pilotSeq = pilotSeq / np.sqrt(np.mean(np.abs(pilotSeq) ** 2))
    else:
        pilotSeq = np.array([], dtype=complex)
    p['pilotSeq'] = pilotSeq
    return p


# ======================================================================
# ofdm_modulate.m / ofdm_demodulate.m / ofdm_tx_filter.m 转换
# ======================================================================
def ofdm_modulate(sym, p):
    """QAM 符号向量 -> OFDM 复基带信号 (导频插入 + 子载波映射 + IFFT + 加 CP)"""
    sym = np.asarray(sym).ravel()
    nSym = sym.size
    nPadSym = (-nSym) % p['N_data']
    sym = np.concatenate([sym, np.zeros(nPadSym, dtype=sym.dtype)])
    symMat = sym.reshape(p['N_data'], -1, order='F')   # 每列一个 OFDM 符号
    nOfdm = symMat.shape[1]
    X = np.zeros((p['N_fft'], nOfdm), dtype=complex)
    X[p['dataBins'], :] = symMat
    if p['nPilot'] > 0:
        X[p['pilotBins'], :] = p['pilotSeq'][:, None] * np.ones(
            (1, nOfdm), dtype=complex)
    x = np.fft.ifft(X, axis=0)                         # N_fft x nOfdm (无缩放)
    xcp = np.vstack([x[-p['N_cp']:, :], x])            # 加循环前缀
    sig = xcp.ravel(order='F')
    return sig, nPadSym


def ofdm_demodulate(sig, p, nSym=None):
    """OFDM 复基带信号 -> 导频增益估计/均衡 -> QAM 符号向量
    返回 (sym, gHat); gHat: 每 OFDM 符号估计的复增益 (nOfdm,)
    """
    sig = np.asarray(sig).ravel()
    L = p['N_fft'] + p['N_cp']
    nOfdm = sig.size // L
    y = sig[:nOfdm * L].reshape(L, nOfdm, order='F')
    y = y[p['N_cp']:, :]                               # 去循环前缀
    Y = np.fft.fft(y, axis=0)

    # 发射端频谱整形滤波器群时延补偿 (可选)
    tau = p.get('txFilterTau')
    if tau:
        kAll = (np.arange(p['N_fft']) + p['N_fft'] // 2) % p['N_fft'] - p['N_fft'] // 2
        Y = Y * np.exp(1j * 2 * np.pi * kAll * tau / p['N_fft'])[:, None]

    # 导频 LS 增益估计 (每 OFDM 符号一个复增益)
    if p['nPilot'] > 0:
        yp = Y[p['pilotBins'], :]
        gHat = np.mean(yp / p['pilotSeq'][:, None], axis=0)
    else:
        gHat = np.ones(nOfdm)
    Y = Y / gHat[None, :]                              # 均衡 (含幅度与相位)

    symAll = np.concatenate([Y[p['posBins'], :], Y[p['negBins'], :]])
    sym = symAll[p['dataPos'], :].ravel(order='F')
    if nSym is not None:
        sym = sym[:min(nSym, sym.size)]
    return sym, gHat


def ofdm_tx_filter(sig, p):
    """发射端频谱整形滤波器: 线性相位 Kaiser 窗 FIR, ACPR -> -70 dB 以下
    返回 (sigF, tau); tau 为群时延 (采样数), 接收端需频域补偿。
    """
    fs = p['fs']
    fOcc = p['occBW'] / 2
    fAdj = fOcc + 0.035 * p['BW']
    fPass = fOcc
    fStop = fAdj
    A = 80.0
    df = fStop - fPass
    assert df > 0, '带宽配置过紧: 无过渡带空间 (需要 fs > 2*occBW)'

    beta = 0.1102 * (A - 8.7)
    N = int(np.ceil((A - 8) / (2.285 * 2 * np.pi * df / fs)))
    N = max(N, 32)
    Nmax = p['N_cp'] + 1
    if N > Nmax:
        print(f'[ofdm_tx_filter] 滤波器阶数 {N} 超过 CP 限制 {Nmax}, 已截断')
        N = Nmax
    if N % 2 == 0:
        N += 1

    fc = (fPass + fStop) / 2
    n = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1)
    h = 2 * fc / fs * np.sinc(2 * fc / fs * n)         # 理想低通冲激响应
    w = besseli0(beta * np.sqrt(1 - (2 * n / (N - 1)) ** 2)) / besseli0(beta)
    h = h * w
    h = h / h.sum()                                    # 直流增益 = 1
    tau = (N - 1) / 2

    sigF = lfilter(h, [1.0], np.asarray(sig).ravel())
    return sigF, tau


# ======================================================================
# 扰码: 确定性 PN 序列异或 (收发同种子, 白化编码比特)
# ======================================================================
def scramble_bits(bits, seed=20240517):
    """与确定性 PN 序列异或 (等价 MATLAB rand(RandStream('mt19937ar',
    'Seed', seed), N, 1) > 0.5)"""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    rng = np.random.RandomState(seed)
    pn = (rng.rand(bits.size) > 0.5).astype(np.uint8)
    return bits ^ pn


# ======================================================================
# LDPC: 确定性规则 LDPC 码构造 + 编码 + BP 译码
# (ldpc_pcm.m / fec_ldpc_encode.m / fec_ldpc_decode.m 的等价实现)
# ======================================================================
_ldpc_cache = {}


def _make_regular_ldpc(n, k, dc=3, seed=0):
    """PEG 风格贪心构造规则 LDPC 校验矩阵 H (m x n, m=n-k, 列重 dc),
    无 4-cycle (girth >= 6), 并求系统编码矩阵 P (m x k) 使码字
    c=[u, P@u] 满足 Hc=0。返回 (H, P) 或 None (重试)。
    """
    rng = np.random.RandomState(seed)
    for _attempt in range(50):
        m = n - k
        adj_var = [set() for _ in range(n)]          # 变量节点连接的校验节点
        adj_chk = [set() for _ in range(m)]          # 校验节点连接的变量节点
        deg_chk = np.zeros(m, dtype=np.int64)

        def _no_4cycle(v, j):
            """添加边 (v,j) 是否引入 4-cycle:
            存在已连校验 j0, 使 j 与 j0 共享非 v 的变量邻居"""
            for j0 in adj_var[v]:
                if adj_chk[j0] & (adj_chk[j] - {v}):
                    return False
            return True

        for v in range(n):
            for _e in range(dc):
                # 候选: 度数最小的且无 4-cycle 的校验节点
                best = None
                best_deg = None
                # 按度数升序扫描 (限制候选数, 加速)
                order = np.argsort(deg_chk, kind='stable')
                for j in order:
                    if j in adj_var[v]:
                        continue
                    if _no_4cycle(v, int(j)):
                        best = int(j)
                        break
                    if best is None:
                        best = int(j)                # 退路 (允许 4-cycle)
                if best is None:
                    raise RuntimeError('LDPC 构造失败: 无可用校验节点')
                adj_var[v].add(best)
                adj_chk[best].add(v)
                deg_chk[best] += 1

        H = np.zeros((m, n), dtype=np.uint8)
        for v in range(n):
            for j in adj_var[v]:
                H[j, v] = 1

        # GF(2) RREF 求系统形式 (行操作保持零空间)
        M = H.copy()
        pivots = []
        col = 0
        ok = True
        for row in range(m):
            while col < n and not np.any(M[row:, col]):
                col += 1
            if col >= n:
                ok = False
                break
            r = row + int(np.nonzero(M[row:, col])[0][0])
            if r != row:
                M[[row, r]] = M[[r, row]]
            nz = np.nonzero(M[:, col])[0]
            nz = nz[nz != row]
            M[nz] = M[nz] ^ M[row]
            pivots.append(col)
            col += 1
        if not ok or len(pivots) < m:
            continue
        nonpiv = np.array([c for c in range(n) if c not in pivots],
                          dtype=np.int64)
        P = M[:, nonpiv]                               # m x k
        # 校验: M[:, pivots] 应为单位阵
        if not np.array_equal(M[:, pivots], np.eye(m, dtype=np.uint8)):
            continue
        perm = np.concatenate([nonpiv, pivots])        # 码字重排: [u | p]
        return H, P, perm
    raise RuntimeError(f'无法构造 LDPC (n={n}, k={k}) 校验矩阵')


def _ldpc_encode_batch(u, P, perm):
    """批量 LDPC 编码: u (k, nBlk) -> c (n, nBlk) 原始列序码字
    码字在 perm 顺序下为 [u; P@u] (GF2)
    """
    u = np.asarray(u, dtype=np.uint8)
    p = (P.astype(np.int32) @ u.astype(np.int32)) % 2
    c_perm = np.vstack([u, p.astype(np.uint8)])       # (n, nBlk) 按 perm 序
    c = np.empty_like(c_perm)
    c[perm] = c_perm                                  # 重排回原始列序
    return c


def _ldpc_code_name(code):
    """规范化 LDPC 码名: None/''/dvb -> 'wifi'"""
    if not code:
        return 'wifi'
    code = str(code).lower()
    return 'wifi' if code == 'dvb' else code


def ldpc_pcm(code=None, rate=None):
    """获取 LDPC 码参数 (等价 ldpc_pcm.m)
    code: 'wifi' (默认) / 'dvb' (回退 wifi 并警告)
    rate: 可选, 指定码率 (0,1); 不指定时用 'wifi' 的 rate 1/2
    返回 (n, k, codeUsed); 注: MATLAB 的 rate 用 DVB-S2 64800 长码,
    本实现用 n=1944 的规则 LDPC 码近似 (码率一致, 码长更短, 收发自洽)。
    """
    if code is None or code == '':
        code = 'wifi'
    code = _ldpc_code_name(code)
    if code == 'dvb':
        print('[ldpc_pcm] 警告: dvb 码 (DVB-S2X) 在 Python 版中不可用, '
              '自动回退到 wifi 码')
        code = 'wifi'
    if rate is None:
        n, k = 1944, 972                      # wifi: 802.11n rate 1/2 尺寸
        codeUsed = 'wifi'
    else:
        assert 0 < rate < 1, '码率 rate 必须在 (0,1) 内'
        n = 1944
        k = int(round(n * rate))
        codeUsed = f'wifi-r{rate:g}'
    key = (code, k)
    if key not in _ldpc_cache:
        H, P, perm = _make_regular_ldpc(n, k, dc=3, seed=hash(key) & 0xffff)
        _ldpc_cache[key] = (H, P, perm)
    return n, k, codeUsed


def fec_ldpc_encode(bits, fec):
    """LDPC 信道编码 + 交织 (比特级), 等价 fec_ldpc_encode.m
    fec: dict, 含 'code' / 'rate' (可选)
    返回 (bitsOut, stats)
    """
    if fec is None:
        fec = {}
    fec = dict(fec)
    code = _ldpc_code_name(fec.get('code'))
    rate = fec.get('rate')
    n, k, codeUsed = ldpc_pcm(code, rate)
    H, P, perm = _ldpc_cache[(code, k)]

    bits = np.asarray(bits, dtype=np.uint8).ravel()
    nBits = bits.size
    nPadBits = (-nBits) % k
    if nBits == 0:
        stats = {'type': 'ldpc', 'code': codeUsed, 'fecRate': rate,
                 'n': n, 'k': k, 'rate': k / n, 'nPadBits': 0, 'nBlk': 0}
        return np.zeros(0, dtype=np.uint8), stats

    msg = np.concatenate([bits, np.random.randint(0, 2, nPadBits,
                                                  dtype=np.uint8)])
    nBlk = msg.size // k
    msgMat = msg.reshape(k, nBlk, order='F')       # 每列一个信息字
    code_mat = _ldpc_encode_batch(msgMat, P, perm)  # n x nBlk, 每列一个码字
    bitsOut = code_mat.T.ravel(order='F')          # 码字间交织 (列优先读出)

    stats = {'type': 'ldpc', 'code': codeUsed, 'fecRate': rate,
             'n': n, 'k': k, 'rate': k / n, 'nPadBits': nPadBits,
             'nBlk': nBlk}
    return bitsOut, stats


def _ldpc_bp_decode(llr, H, max_iter=30):
    """批量 sum-product 置信传播译码 (与 MATLAB ldpcDecode 同款 BP)
    llr: (n, nBlk), LLR>0 -> 比特 1 的约定
    返回 (hard_bits (k, nBlk), nIter)
    """
    m, n = H.shape
    k = n - m
    nBlk = llr.shape[1]
    # 边表: 每条边 (校验节点 j, 变量节点 i)
    rows, cols = np.nonzero(H)                       # rows: 校验, cols: 变量
    nE = rows.size
    # 按校验节点分组的边索引
    chk_edges = [[] for _ in range(m)]
    var_edges = [[] for _ in range(n)]
    for e in range(nE):
        chk_edges[rows[e]].append(e)
        var_edges[cols[e]].append(e)
    chk_edges = [np.array(v, dtype=np.int64) for v in chk_edges]
    var_edges = [np.array(v, dtype=np.int64) for v in var_edges]

    # 消息: v2c / c2v 按边组织 (nE, nBlk)
    v2c = llr[cols]                                   # 变量 -> 校验 (初始化 = 信道 LLR)
    c2v = np.zeros((nE, nBlk))
    hard = np.zeros((n, nBlk), dtype=np.uint8)
    it = 0
    for it in range(max_iter):
        # ---- 校验节点更新 (sum-product, 等价 2*atanh(Π tanh(v2c/2))) ----
        # LLR 约定: LLR>0 -> bit 1; 校验约束 XOR=0 时消息符号因子为 (-1)^d
        for e_list in chk_edges:
            d = e_list.size
            if d == 0:
                continue
            msgs = v2c[e_list]                        # (d, nBlk)
            t = np.tanh(msgs / 2.0)
            prod = np.prod(t, axis=0)                 # (nBlk,)
            parity = 1.0 if (d % 2 == 0) else -1.0    # (-1)^d
            out = np.zeros((d, nBlk))
            for t_idx in range(d):
                # 排除自身: prod / t[t_idx]
                with np.errstate(divide='ignore', invalid='ignore'):
                    p = prod / t[t_idx]
                p = np.clip(p, -0.999999999, 0.999999999)
                out[t_idx] = parity * 2.0 * np.arctanh(p)
            c2v[e_list] = out
        # ---- 变量节点更新 ----
        for i in range(n):
            e_list = var_edges[i]
            d = e_list.size
            if d == 0:
                continue
            tot = llr[i] + c2v[e_list].sum(axis=0)    # (nBlk,)
            for t in range(d):
                v2c[e_list[t]] = tot - c2v[e_list[t]]
        # ---- 判决 + 提前终止 ----
        for i in range(n):
            e_list = var_edges[i]
            if e_list.size == 0:
                hard[i] = (llr[i] > 0).astype(np.uint8)
            else:
                Ltot = llr[i] + c2v[e_list].sum(axis=0)
                hard[i] = (Ltot > 0).astype(np.uint8)
        synd = (H.astype(np.int32) @ hard.astype(np.int32)) % 2
        if not np.any(synd):
            break
    return hard, it + 1


def fec_ldpc_decode(bits, fec):
    """LDPC 置信传播译码 + 去交织 (支持软判决 LLR 输入), 等价 fec_ldpc_decode.m
    bits: 接收编码比特流 (硬判决 0/1) 或 fec['llr']=True 时的软 LLR 向量
    fec: dict, 含 'code' / 'fecRate' / 'n' / 'k' / 'nPadBits' / 'llr'
    返回 (bitsOut, stats)
    """
    k = fec['k']
    n = fec['n']
    if bits is None or (hasattr(bits, 'size') and bits.size == 0):
        stats = {'nBlk': 0, 'nErrCW': 0, 'nIter': []}
        return np.zeros(0, dtype=np.uint8), stats
    code = _ldpc_code_name(fec.get('code'))
    rate = fec.get('fecRate')
    n2, k2, codeUsed = ldpc_pcm(code, rate)
    assert n2 == n and k2 == k, '收发端 LDPC 参数不一致'
    H, P, perm = _ldpc_cache[(code, k)]

    softLLR = bool(fec.get('llr'))
    bits = np.asarray(bits).ravel()
    nBlk = bits.size // n
    assert bits.size % n == 0, '接收比特数不是码字长度的整数倍'
    codeMat = bits.reshape(nBlk, n, order='F').T    # n x nBlk (去交织)
    if softLLR:
        llr_in = codeMat                              # LLR>0 -> 比特 1
    else:
        llr_in = np.where(codeMat > 0, 1.0, -1.0) * 20.0
    hard, n_iter = _ldpc_bp_decode(llr_in, H, max_iter=30)
    # 校验方程检测: 带误码到达的码字数 (用于统计)
    synd = (H.astype(np.int32) @ (llr_in < 0).astype(np.int32)) % 2
    nErrCW = int(np.any(synd, axis=0).sum())
    # 信息位位于 perm[:k] (nonpiv 列)
    info = hard[perm[:k]]                             # (k, nBlk)
    bitsOut = info.ravel(order='F')                   # 码字连续排列
    nPadBits = fec.get('nPadBits', 0)
    if nPadBits > 0:
        bitsOut = bitsOut[:-nPadBits]
    stats = {'nBlk': nBlk, 'nErrCW': nErrCW, 'nIter': n_iter}
    return bitsOut, stats


# ======================================================================
# RS(255,k): GF(256) 编解码 (与 MATLAB gf(8) 一致, 本原多项式 285)
# ======================================================================
_GF_EXP = None
_GF_LOG = None
_GF_GENPOLY_CACHE = {}


def _gf_init():
    """GF(256) 指数/对数表, 本原多项式 0x11D (285, D^8+D^4+D^3+D^2+1)"""
    global _GF_EXP, _GF_LOG
    if _GF_EXP is not None:
        return
    exp = np.zeros(512, dtype=np.int64)
    log = np.zeros(256, dtype=np.int64)
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11D
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    _GF_EXP = exp
    _GF_LOG = log


def _gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _gf_pow(a, p):
    if a == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] * p) % 255]


def _gf_inv(a):
    if a == 0:
        return 0
    return _GF_EXP[255 - _GF_LOG[a]]


def _rs_genpoly(n, k, b=1):
    """RS 生成多项式 g(x) = prod_{i=b}^{b+n-k-1} (x - alpha^i), 系数 MSB 在前"""
    key = (n, k, b)
    if key in _GF_GENPOLY_CACHE:
        return _GF_GENPOLY_CACHE[key]
    _gf_init()
    deg = n - k
    g = [1]                                          # 系数 (低次在前)
    for i in range(deg):
        root = _gf_pow(2, b + i)
        # g = g * (x - root)
        new = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            new[j] ^= c                              # c*x
            new[j + 1] ^= _gf_mul(c, root)           # -c*root (GF2^8 中 - = +)
        g = new
    _GF_GENPOLY_CACHE[key] = g                       # 低次在前, g[deg]=1
    return g


def _rs_encode_msg(msg, genpoly):
    """系统 RS 编码: msg (k 字节) -> codeword (n 字节), 信息在前"""
    n = len(msg) + len(genpoly) - 1
    reg = [0] * (len(genpoly) - 1)
    for byte in msg:
        factor = byte ^ reg[0]
        reg = reg[1:] + [0]
        if factor:
            for j in range(len(genpoly) - 1):
                reg[j] ^= _gf_mul(genpoly[j + 1], factor)
    return list(msg) + reg


def _rs_syndrome(recv, n, k, b=1):
    """计算伴随式 S_i = recv(alpha^(b+i)), i=0..n-k-1
    码字数组高次在前 (recv[0] 为 x^(n-1) 系数), 用霍纳法求值
    """
    deg = n - k
    S = []
    for i in range(deg):
        x = _gf_pow(2, b + i)
        s = 0
        for j in range(n):
            s = _gf_mul(s, x) ^ recv[j]              # 霍纳: 从最高次开始
        S.append(s)
    return S


def _rs_bm(S):
    """Berlekamp-Massey: 输入伴随式, 输出错误位置多项式 sigma (低次在前)
    以及 deg; 若 deg > len(S)/2 返回 None (无法纠正)
    """
    _gf_init()
    C = [1]
    B = [1]
    L = 0
    m = 1
    b = 1
    for i in range(len(S)):
        # 计算不匹配 delta
        d = S[i]
        for j in range(1, L + 1):
            if j < len(C):
                d ^= _gf_mul(C[j], S[i - j])
        if d == 0:
            m += 1
        elif 2 * L <= i:
            T = C[:]
            scale = _gf_mul(d, _gf_inv(b))
            C = C + [0] * (len(B) + m - len(C))
            for j in range(len(B)):
                C[j + m] ^= _gf_mul(scale, B[j])
            L = i + 1 - L
            B = T
            b = d
            m = 1
        else:
            scale = _gf_mul(d, _gf_inv(b))
            C = C + [0] * (len(B) + m - len(C))
            for j in range(len(B)):
                C[j + m] ^= _gf_mul(scale, B[j])
            m += 1
    if L > len(S) // 2:
        return None
    return C, L


def _rs_chien_forney(recv, n, k, sigma, S, b=1):
    """Chien 搜索 + Forney 求错误值, 返回纠正后的码字或 None
    码字数组高次在前: recv[j] 是 x^(n-1-j) 的系数, 故数组位置 j 的
    错误定位子 X_j = alpha^(n-1-j), X_j^-1 = alpha^(j+1) (n=255 时)。
    sigma: 错误位置多项式, 低次在前, sigma[0]=1
    """
    _gf_init()
    deg = len(sigma) - 1
    t = (n - k) // 2
    # ---- Chien 搜索: 求 sigma(X_j^-1) = 0 的错误位置 j ----
    err_pos = []
    for j in range(n):
        x = _gf_pow(2, (j + 1) % 255)                # X_j^-1 = alpha^(j+1)
        val = 0
        for c_idx, c in enumerate(sigma):
            val ^= _gf_mul(c, _gf_pow(x, c_idx))
        if val == 0:
            err_pos.append(j)                        # 数组位置 j 出错
    if len(err_pos) != deg or deg > t:
        return None
    # ---- 错误多项式 Omega(x) = (S(x) * sigma(x)) mod x^(2t) ----
    # S(x) = sum_{i=0}^{2t-1} S_i x^i
    omega = [0] * (2 * t)
    for i in range(2 * t):
        for jj in range(min(i + 1, len(sigma))):
            if i - jj < len(S):
                omega[i] ^= _gf_mul(S[i - jj], sigma[jj])
    # ---- sigma'(x) (GF(2^8) 中偶数次项导数为 0) ----
    sigma_der = [0] * max(len(sigma) - 1, 1)
    for i in range(1, len(sigma), 2):
        sigma_der[i - 1] = sigma[i]
    # ---- Forney: e_j = Xj^(1-b) * Omega(Xj^-1) / sigma'(Xj^-1) ----
    out = list(recv)
    for pos in err_pos:
        Xj = _gf_pow(2, (n - 1 - pos) % 255)         # 定位子 alpha^(n-1-pos)
        Xj_inv = _gf_pow(2, (pos + 1) % 255)
        om = 0
        for i in range(len(omega)):
            om ^= _gf_mul(omega[i], _gf_pow(Xj_inv, i))
        sd = 0
        for i in range(len(sigma_der)):
            sd ^= _gf_mul(sigma_der[i], _gf_pow(Xj_inv, i))
        if sd == 0:
            return None
        e = _gf_mul(om, _gf_inv(sd))
        if b != 1:
            e = _gf_mul(e, _gf_pow(Xj, (1 - b) % 255))
        out[pos] ^= e
    return out


def fec_rs_encode(bytes_in, fec):
    """RS(255,k) 信道编码 + 交织 (字节级), 等价 fec_rs_encode.m
    fec: dict, 含 'k' (每码字信息字节数, 默认 223)
    返回 (bytes_out, stats)
    """
    if fec is None:
        fec = {}
    fec = dict(fec)
    k = fec.get('k', 223)
    n = 255
    assert (n - k) % 2 == 0 and 1 <= k < n, 'RS(255,k) 要求 n-k 为偶数'
    bytes_in = np.asarray(bytes_in, dtype=np.uint8).ravel()
    nBytes = bytes_in.size
    nPadBytes = (-nBytes) % k
    if nBytes == 0:
        stats = {'n': n, 'k': k, 'nPadBytes': 0, 'nCW': 0, 'rate': k / n}
        return np.zeros(0, dtype=np.uint8), stats
    msg = np.concatenate([bytes_in, np.random.randint(0, 256, nPadBytes,
                                                      dtype=np.uint8)])
    nCW = msg.size // k
    msgMat = msg.reshape(k, nCW, order='F').T        # nCW x k, 每行一个码字
    genpoly = _rs_genpoly(n, k, b=1)
    code = np.zeros((nCW, n), dtype=np.uint8)
    for i in range(nCW):
        code[i] = np.array(_rs_encode_msg(list(msgMat[i]), genpoly),
                           dtype=np.uint8)
    bytes_out = code.ravel(order='F')                # 列优先读出 => 码字间交织
    stats = {'n': n, 'k': k, 'nPadBytes': nPadBytes, 'nCW': nCW, 'rate': k / n}
    return bytes_out, stats


def fec_rs_decode(bytes_in, fec):
    """RS(255,k) 译码 + 去交织, 等价 fec_rs_decode.m
    fec: dict, 含 'n' / 'k' / 'nPadBytes'
    返回 (bytes_out, stats)
    """
    n = fec['n']
    k = fec['k']
    bytes_in = np.asarray(bytes_in, dtype=np.uint8).ravel()
    if bytes_in.size == 0:
        stats = {'cnumerr': [], 'nFail': 0, 'nBytesIn': 0}
        return np.zeros(0, dtype=np.uint8), stats
    nCW = bytes_in.size // n
    assert bytes_in.size % n == 0, '编码字节流长度必须是 255 的整数倍'
    code = bytes_in.reshape(nCW, n, order='F')       # 行 = 码字 (去交织)
    genpoly = _rs_genpoly(n, k, b=1)
    dec = np.zeros((nCW, k), dtype=np.uint8)
    cnumerr = np.zeros(nCW, dtype=np.int64)
    nFail = 0
    for i in range(nCW):
        recv = list(code[i])
        S = _rs_syndrome(recv, n, k, b=1)
        if not any(S):
            dec[i] = recv[:k]
            cnumerr[i] = 0
            continue
        res = _rs_bm(S)
        if res is None:
            dec[i] = recv[:k]
            cnumerr[i] = -1
            nFail += 1
            continue
        sigma, L = res
        fixed = _rs_chien_forney(recv, n, k, sigma, S, b=1)
        if fixed is None:
            dec[i] = recv[:k]
            cnumerr[i] = -1
            nFail += 1
        else:
            dec[i] = fixed[:k]
            cnumerr[i] = L
    bytes_out = dec.T.ravel(order='F')               # 恢复原始字节顺序
    nPadBytes = fec.get('nPadBytes', 0)
    if nPadBytes > 0:
        bytes_out = bytes_out[:-nPadBytes]
    stats = {'cnumerr': cnumerr, 'nFail': nFail, 'nBytesIn': bytes_out.size}
    return bytes_out, stats


# ======================================================================
# zip_ofdm_tx.m / zip_ofdm_rx.m 转换
# ======================================================================
def zip_ofdm_tx(zipFile, M, BW, fs=None, cpRatio=1 / 8, fec=None,
                normRms=None):
    """读取文件比特流并调制为 OFDM 复基带信号, 返回 (sig, meta)
    等价 zip_ofdm_tx.m; meta 为 dict (原样传给 zip_ofdm_rx)
    """
    if fs is None:
        fs = 2 * BW
    if fec is None:
        fec = {}
    if normRms is not None:
        assert normRms > 0, 'normRms 必须为正'
    p = ofdm_params(BW, M, fs, cpRatio)

    with open(zipFile, 'rb') as f:
        bytes_in = np.frombuffer(f.read(), dtype=np.uint8)
    nBytes = bytes_in.size

    fec = dict(fec)
    useLDPC = bool(fec.get('type')) and str(fec.get('type')).lower() == 'ldpc'
    if useLDPC:
        bits = bytes_to_bits(bytes_in)
        bits, fecStats = fec_ldpc_encode(bits, fec)
    else:
        if fec:                                      # RS 信道编码
            bytes_in, fecStats = fec_rs_encode(bytes_in, fec)
        bits = bytes_to_bits(bytes_in)

    # 加扰 (收发同种子)
    bits = scramble_bits(bits, seed=20240517)

    # 比特 -> QAM 符号 (末尾补零到 log2(M) 整数倍)
    k = p['k']
    nBitsOrig = bits.size
    nPadBits = (-nBitsOrig) % k
    bits = np.concatenate([bits, np.zeros(nPadBits, dtype=np.uint8)])
    sym = qammod(bits, M)

    # 符号 -> OFDM 基带信号
    sig, nPadSym = ofdm_modulate(sym, p)

    # 发射端归一化 (DPD/PA 驱动电平)
    scale = 1.0
    if normRms is not None:
        scale = normRms / np.sqrt(np.mean(np.abs(sig) ** 2))
        sig = sig * scale

    meta = dict(p)
    meta['zipFile'] = zipFile
    meta['nBytes'] = nBytes
    meta['nBitsOrig'] = nBitsOrig
    meta['nPadBits'] = nPadBits
    meta['nPadSym'] = nPadSym
    meta['nOfdm'] = sig.size / (p['N_fft'] + p['N_cp'])
    if fec:
        meta['fec'] = fecStats
    if normRms is not None:
        meta['normRms'] = normRms
        meta['rxScale'] = 1 / scale
    return sig, meta


def zip_ofdm_rx(sig, meta, outFile=None):
    """对 OFDM 复基带信号解调, 还原出文件并校验一致性
    返回 (ok, outFile, stats); 等价 zip_ofdm_rx.m
    """
    p = meta
    if outFile is None:
        import os
        d = os.path.dirname(meta['zipFile'])
        name = os.path.splitext(os.path.basename(meta['zipFile']))[0]
        outFile = os.path.join(d, f'{name}_restored.zip')

    # 增益补偿: 无导频时回退到 rxScale
    if meta['nPilot'] == 0 and meta.get('rxScale'):
        sig = np.asarray(sig) * meta['rxScale']

    symHat, gHat = ofdm_demodulate(np.asarray(sig), p)
    bits = qamdemod(symHat, p['M'])
    bits = bits[:meta['nBitsOrig']]
    bits = scramble_bits(bits, seed=20240517)        # 解扰 (同种子 XOR)

    fec = meta.get('fec')
    stats = {}
    useLDPC = bool(fec and fec.get('type')) and \
        str(fec.get('type')).lower() == 'ldpc'
    if useLDPC:
        bits, statsFEC = fec_ldpc_decode(bits, fec)
        bytes_out = bits_to_bytes(bits)
        stats['fec'] = statsFEC
    else:
        bytes_out = bits_to_bytes(bits)
        if fec:
            bytes_out, statsFEC = fec_rs_decode(bytes_out, fec)
            stats['fec'] = statsFEC
    assert bytes_out.size == meta['nBytes'], '还原字节数与原始文件不符'
    with open(outFile, 'wb') as f:
        f.write(bytes_out.tobytes())

    # 与原始文件逐字节比对
    import os
    ok = False
    nErr = np.nan
    if os.path.isfile(meta['zipFile']):
        with open(meta['zipFile'], 'rb') as f:
            bytes0 = np.frombuffer(f.read(), dtype=np.uint8)
        nErr = int(np.count_nonzero(bytes_to_bits(bytes_out) !=
                                    bytes_to_bits(bytes0)))
        ok = bool(np.array_equal(bytes_out, bytes0))
    else:
        print(f'[zip_ofdm_rx] 找不到原始文件 {meta["zipFile"]}, 无法校验')
    stats['nBytes'] = bytes_out.size
    stats['nBitErrors'] = nErr
    stats['ber'] = nErr / (bytes_out.size * 8)
    stats['symHat'] = symHat
    stats['gainEst'] = gHat
    return ok, outFile, stats


# ======================================================================
# zip_ofdm_tx_pkt.m / zip_ofdm_rx_pkt.m / zip_ofdm_stream_txrx.m 转换
# ======================================================================
def _fec_block_sizes(fec, p):
    """FEC 块尺寸 (n=码字长, k=每码字信息位), 等价 fec_block_sizes"""
    fec = dict(fec or {})
    useLDPC = bool(fec.get('type')) and str(fec.get('type')).lower() == 'ldpc'
    if useLDPC:
        nFEC, kFEC, _ = ldpc_pcm(fec.get('code'), fec.get('rate'))
    elif fec:
        nFEC = 255
        kFEC = fec.get('k', 223)
    else:
        nFEC = p['k']
        kFEC = p['k']
    return useLDPC, nFEC, kFEC


def zip_ofdm_tx_pkt(zipFile, M, BW, fs=None, cpRatio=1 / 8, fec=None,
                    normRms=None, maxLen=80000):
    """分包发射机: 大文件拆成多个小数据包, 每包生成独立 OFDM 信号
    返回 (sigPkt, metaPkt); 等价 zip_ofdm_tx_pkt.m
    """
    if fs is None:
        fs = 2 * BW
    if fec is None:
        fec = {}
    if normRms is not None:
        assert normRms > 0, 'normRms 必须为正'
    scrSeed = 20240517
    p = ofdm_params(BW, M, fs, cpRatio)
    L = p['N_fft'] + p['N_cp']
    nSymPkt = max(1, int(np.floor(maxLen / L)))
    assert nSymPkt >= 1, 'maxLen 太小: 至少容纳 1 个 OFDM 符号'
    lenPkt = nSymPkt * L

    fec = dict(fec)
    useLDPC, nFEC, kFEC = _fec_block_sizes(fec, p)
    availBits = nSymPkt * p['N_data'] * p['k']
    nBlkPkt = int(np.floor(availBits / nFEC))
    infoBitsPkt = nBlkPkt * kFEC
    bytesPkt = int(np.floor(infoBitsPkt / 8))

    with open(zipFile, 'rb') as f:
        bytes_in = np.frombuffer(f.read(), dtype=np.uint8)
    nBytes = bytes_in.size
    nPkt = max(1, int(np.ceil(nBytes / bytesPkt)))

    sigPkt = []
    pktBytes = np.zeros(nPkt, dtype=np.int64)
    encBits = np.zeros(nPkt, dtype=np.int64)
    for i in range(nPkt):
        b0 = i * bytesPkt
        b1 = min((i + 1) * bytesPkt, nBytes)
        pktBytes[i] = b1 - b0
        bitsPkt = bytes_to_bits(bytes_in[b0:b1])
        if useLDPC:
            txBitsPkt, _ = fec_ldpc_encode(bitsPkt, fec)
        elif fec:
            bytesP, _ = fec_rs_encode(bytes_in[b0:b1], fec)
            txBitsPkt = bytes_to_bits(bytesP)
        else:
            txBitsPkt = bitsPkt
        encBits[i] = txBitsPkt.size
        txBitsPkt = scramble_bits(txBitsPkt, seed=scrSeed)

        symPkt = qammod(np.concatenate([txBitsPkt,
                                        np.zeros((-txBitsPkt.size) % p['k'],
                                                 dtype=np.uint8)]), M)
        nDummy = (-symPkt.size) % p['N_data']
        if nDummy > 0:
            dum = qammod(np.random.randint(0, 2, nDummy * p['k'],
                                           dtype=np.uint8), M)
            symPkt = np.concatenate([symPkt, dum])
        sigP, _ = ofdm_modulate(symPkt, p)
        sigP, tauF = ofdm_tx_filter(sigP, p)
        p['txFilterTau'] = tauF
        if normRms is not None:
            sigP = sigP * (normRms / np.sqrt(np.mean(np.abs(sigP) ** 2)))
        sigPkt.append(sigP)

    metaPkt = dict(p)
    metaPkt['zipFile'] = zipFile
    metaPkt['fec'] = fec
    metaPkt['nBytes'] = nBytes
    metaPkt['nPkt'] = nPkt
    metaPkt['nSymPkt'] = nSymPkt
    metaPkt['lenPkt'] = lenPkt
    metaPkt['pktBytes'] = pktBytes
    metaPkt['encBits'] = encBits
    metaPkt['normRms'] = normRms
    metaPkt['maxLen'] = maxLen
    return sigPkt, metaPkt


def zip_ofdm_rx_pkt(sigPkt, metaPkt, outFile=None):
    """分包接收机: 逐包解调 -> 译码 -> 拼接还原文件
    返回 (ok, outFile, statsPkt); 等价 zip_ofdm_rx_pkt.m
    """
    import os
    p = metaPkt
    if outFile is None:
        d = os.path.dirname(metaPkt['zipFile'])
        name = os.path.splitext(os.path.basename(metaPkt['zipFile']))[0]
        outFile = os.path.join(d, f'{name}_restored.zip')
    fec = dict(metaPkt.get('fec') or {})
    useLDPC = bool(fec.get('type')) and str(fec.get('type')).lower() == 'ldpc'
    if useLDPC:
        nFEC, kFEC, codeUsed = ldpc_pcm(fec.get('code'), fec.get('rate'))
    else:
        nFEC = kFEC = codeUsed = None

    with open(metaPkt['zipFile'], 'rb') as f:
        bytes0 = np.frombuffer(f.read(), dtype=np.uint8)

    bytesCell = []
    nErrPkt = np.zeros(metaPkt['nPkt'], dtype=np.int64)
    lenPkt = np.zeros(metaPkt['nPkt'], dtype=np.int64)
    for i in range(metaPkt['nPkt']):
        symHat, _ = ofdm_demodulate(sigPkt[i], p)
        bitsRxPad = qamdemod(symHat, p['M'])
        rxBits = bitsRxPad[:metaPkt['encBits'][i]]
        rxBits = scramble_bits(rxBits, seed=20240517)   # 解扰 (硬, 统计用)

        if useLDPC:
            varEst = np.mean(np.abs(symHat - qammod(bitsRxPad, p['M'])) ** 2) / 2
            llrP = qam_soft_llr(symHat, p['M'], varEst)
            llrP = llrP[:metaPkt['encBits'][i]]
            # 软解扰 (LLR 乘 ±1)
            rng = np.random.RandomState(20240517)
            pn = (rng.rand(llrP.size) > 0.5).astype(np.float64)
            llrP = llrP * (1 - 2 * pn)
            fecDec = {'code': codeUsed, 'fecRate': fec.get('rate'),
                      'n': nFEC, 'k': kFEC,
                      'nPadBits': (-metaPkt['pktBytes'][i] * 8) % kFEC,
                      'llr': True}
            bitsDec, _ = fec_ldpc_decode(llrP, fecDec)
            bytesP = bits_to_bytes(bitsDec)
        elif fec:
            fecDec = {'n': 255, 'k': fec.get('k', 223),
                      'nPadBytes': (-metaPkt['pktBytes'][i]) % fec.get('k', 223)}
            bytesP, _ = fec_rs_decode(bits_to_bytes(rxBits), fecDec)
        else:
            bytesP = bits_to_bytes(rxBits)
        assert bytesP.size == metaPkt['pktBytes'][i], \
            f'包 {i+1} 还原字节数不符 ({bytesP.size} != {metaPkt["pktBytes"][i]})'
        bytesCell.append(bytesP)
        lenPkt[i] = len(sigPkt[i])
        b0 = int(metaPkt['pktBytes'][:i].sum())
        b1 = b0 + metaPkt['pktBytes'][i]
        nErrPkt[i] = int(np.count_nonzero(
            bytes_to_bits(bytesP) != bytes_to_bits(bytes0[b0:b1])))

    bytesRx = np.concatenate(bytesCell)
    with open(outFile, 'wb') as f:
        f.write(bytesRx.tobytes())
    ok = bool(np.array_equal(bytesRx, bytes0))
    statsPkt = {
        'nPkt': metaPkt['nPkt'],
        'nErrPkt': nErrPkt,
        'berPkt': nErrPkt / (metaPkt['pktBytes'] * 8),
        'lenPkt': lenPkt,
        'berMean': float(np.mean(nErrPkt / (metaPkt['pktBytes'] * 8))),
        'berMax': float(np.max(nErrPkt / (metaPkt['pktBytes'] * 8))),
        'ok': ok,
    }
    return ok, outFile, statsPkt


def zip_ofdm_stream_txrx(zipFile, M, BW, fs=None, cpRatio=1 / 8, fec=None,
                         normRms=None, SNR_dB=np.inf, maxLen=80000,
                         outFile=None):
    """逐包流式 OFDM 收发 (内存安全), 等价 zip_ofdm_stream_txrx.m
    SNR_dB: inf = 无噪声; >0 = Es/N0(dB) 加 AWGN; <=0 = 过 PA 模型
    """
    from function.PA_DVR import PA_DVR_v1
    import os, time
    if fs is None:
        fs = 2 * BW
    if fec is None:
        fec = {}
    if normRms is not None:
        assert normRms > 0, 'normRms 必须为正'
    if outFile is None:
        d = os.path.dirname(zipFile)
        name = os.path.splitext(os.path.basename(zipFile))[0]
        outFile = os.path.join(d, f'{name}_restored.zip')
    scrSeed = 20240517

    p = ofdm_params(BW, M, fs, cpRatio)
    L = p['N_fft'] + p['N_cp']
    nSymPkt = max(1, int(np.floor(maxLen / L)))
    assert nSymPkt >= 1, 'maxLen 太小: 至少容纳 1 个 OFDM 符号'
    lenPkt = nSymPkt * L

    fec = dict(fec)
    useLDPC, nFEC, kFEC = _fec_block_sizes(fec, p)
    if useLDPC:
        _, _, codeUsed = ldpc_pcm(fec.get('code'), fec.get('rate'))
    elif fec:
        codeUsed = 'rs'
    else:
        codeUsed = 'none'
    bytesPkt = int(np.floor(np.floor(nSymPkt * p['N_data'] * p['k'] / nFEC)
                            * kFEC / 8))

    with open(zipFile, 'rb') as f:
        bytes_in = np.frombuffer(f.read(), dtype=np.uint8)
    nBytes = bytes_in.size
    nPkt = max(1, int(np.ceil(nBytes / bytesPkt)))
    print(f'文件 {nBytes/1e6:.2f} MB -> {nPkt} 个数据包, 每包 {bytesPkt} 字节'
          f' / OFDM 信号 {lenPkt} 点 (上限 {maxLen})')

    bytesCell = []
    nErrPkt = np.zeros(nPkt, dtype=np.int64)
    evmPkt = np.zeros(nPkt)
    lenPktA = np.zeros(nPkt, dtype=np.int64)
    pktBytesA = np.zeros(nPkt, dtype=np.int64)
    t0 = time.time()
    for i in range(nPkt):
        b0 = i * bytesPkt
        b1 = min((i + 1) * bytesPkt, nBytes)
        nB = b1 - b0
        pktBytesA[i] = nB
        bitsPkt = bytes_to_bits(bytes_in[b0:b1])

        # ---- 发射 ----
        if useLDPC:
            txBitsPkt, fecStatsP = fec_ldpc_encode(bitsPkt, fec)
            encBits = txBitsPkt.size
        elif fec:
            bytesP, _ = fec_rs_encode(bytes_in[b0:b1], fec)
            txBitsPkt = bytes_to_bits(bytesP)
            encBits = txBitsPkt.size
        else:
            txBitsPkt = bitsPkt
            encBits = txBitsPkt.size
        txBitsPkt = scramble_bits(txBitsPkt, seed=scrSeed)
        symPkt = qammod(np.concatenate([txBitsPkt,
                                        np.zeros((-encBits) % p['k'],
                                                 dtype=np.uint8)]), M)
        nDummy = (-symPkt.size) % p['N_data']
        if nDummy > 0:
            symPkt = np.concatenate([symPkt, qammod(
                np.random.randint(0, 2, nDummy * p['k'], dtype=np.uint8), M)])
        sigP, _ = ofdm_modulate(symPkt, p)
        sigP, tauF = ofdm_tx_filter(sigP, p)
        p['txFilterTau'] = tauF
        if normRms is not None:
            sigP = sigP * (normRms / np.sqrt(np.mean(np.abs(sigP) ** 2)))
        gNorm = 1 / np.max(np.abs(sigP))
        xP = sigP * gNorm

        # ---- 信道 ----
        if np.isinf(SNR_dB):
            rxP = xP
        elif SNR_dB > 0:
            N0 = 1 / 10 ** (SNR_dB / 10)
            sigma = np.sqrt(N0 / (2 * p['N_fft'])) * gNorm
            rxP = xP + sigma * (np.random.randn(xP.size) +
                                1j * np.random.randn(xP.size))
        else:
            rxP = PA_DVR_v1(xP)
        yP = rxP / np.max(np.abs(rxP))

        # ---- 接收 ----
        rxSymP, _ = ofdm_demodulate(yP, p)
        bitsRxPad = qamdemod(rxSymP, p['M'])
        rxBits = bitsRxPad[:encBits]
        rxBits = scramble_bits(rxBits, seed=scrSeed)

        if useLDPC:
            varEst = np.mean(np.abs(rxSymP - qammod(bitsRxPad, p['M'])) ** 2) / 2
            llrP = qam_soft_llr(rxSymP, p['M'], varEst)
            llrP = llrP[:encBits]
            rng = np.random.RandomState(scrSeed)
            pn = (rng.rand(llrP.size) > 0.5).astype(np.float64)
            llrP = llrP * (1 - 2 * pn)
            fecDec = {'code': codeUsed, 'fecRate': fec.get('rate'),
                      'n': nFEC, 'k': kFEC, 'nPadBits': (-nB * 8) % kFEC,
                      'llr': True}
            bitsDec, _ = fec_ldpc_decode(llrP, fecDec)
            bytesP = bits_to_bytes(bitsDec)
        elif fec:
            fecDec = {'n': 255, 'k': fec.get('k', 223),
                      'nPadBytes': (-nB) % fec.get('k', 223)}
            bytesP, _ = fec_rs_decode(bits_to_bytes(rxBits), fecDec)
        else:
            bytesP = bits_to_bytes(rxBits)
        assert bytesP.size == nB, f'包 {i+1} 还原字节数不符'
        bytesCell.append(bytesP)

        nErrPkt[i] = int(np.count_nonzero(
            bytes_to_bits(bytesP) != bytes_to_bits(bytes_in[b0:b1])))
        nRef = symPkt.size
        evmPkt[i] = np.sqrt(np.mean(np.abs(rxSymP[:nRef] - symPkt) ** 2) /
                            np.mean(np.abs(symPkt) ** 2)) * 100
        lenPktA[i] = sigP.size
        if (i + 1) % 50 == 0 or (i + 1) == nPkt:
            print(f'包 {i+1:4d}/{nPkt}  信号 {lenPktA[i]:7d} 点  '
                  f'BER {nErrPkt[i]/(nB*8):.2e}  EVM {evmPkt[i]:.3f}%  '
                  f'已用 {time.time()-t0:.1f} s')

    bytesRx = np.concatenate(bytesCell)
    with open(outFile, 'wb') as f:
        f.write(bytesRx.tobytes())
    ok = bool(np.array_equal(bytesRx, bytes_in))
    stats = {
        'nPkt': nPkt, 'lenPkt': lenPktA, 'nErrPkt': nErrPkt,
        'berPkt': nErrPkt / (pktBytesA * 8), 'evmPkt': evmPkt,
        'berMean': float(np.mean(nErrPkt / (pktBytesA * 8))),
        'evmMean': float(np.mean(evmPkt)),
        'elapsed': time.time() - t0, 'ok': ok,
    }
    print(f'还原: {outFile} | 逐字节一致: {ok} | 平均BER {stats["berMean"]:.2e}'
          f' | 平均EVM {stats["evmMean"]:.3f}% | 总耗时 {stats["elapsed"]:.1f} s')
    return ok, outFile, stats


# ======================================================================
# 高层封装: 单包/整文件 OFDM 发射与解调评估
# (test_OFDM_main.py 使用; 输入 zip 文件 + OFDM/削峰配置, 输出信号;
#  解调输出 EVM / 纠错前后误码率 / 星座图)
# ======================================================================
def ofdm_tx_packet(bytes_pkt, p, fec=None, scr_seed=20240517, caf=None,
                   norm_rms=None):
    """单包发射链路: 字节 -> FEC -> 加扰 -> QAM(+哑元) -> OFDM ->
    频谱整形 -> 峰值归一化 -> (可选 CAF 削峰) -> 再次峰值归一化

    参数:
        bytes_pkt : 本包原始字节 (uint8 数组)
        p         : ofdm_params 生成的参数字典
        fec       : FEC 配置 dict, None/{} = 不加 FEC;
                    {'type':'ldpc','rate':9/10} 或 {'k':223}
        scr_seed  : 扰码种子 (收发必须一致)
        caf       : 削峰配置 dict 或 None:
                    {'thr_dB': 9.5, 'T': 10, 'hard_clipping': 0,
                     'thr_new': 9.8, 'oversample': 6}
        norm_rms  : 目标 RMS 归一化值 (None = 峰值归一化)

    返回:
        x    : 归一化后的 OFDM 复基带信号 (可直接送 DPD/PA/信道)
        info : 解调所需参考信息 dict:
               {'enc_bits', 'tx_bits_scrambled', 'tx_sym', 'n_dummy',
                'g_norm', 'tau', 'fec_stats', 'n_bytes'}
    """
    M = p['M']
    bytes_pkt = np.asarray(bytes_pkt, dtype=np.uint8).ravel()
    n_bytes = bytes_pkt.size

    # ---- 1. 字节 -> 比特 + FEC 编码 ----
    bits_pkt = bytes_to_bits(bytes_pkt)
    fec = dict(fec or {})
    useLDPC = bool(fec.get('type')) and str(fec.get('type')).lower() == 'ldpc'
    if useLDPC:
        tx_bits, fec_stats = fec_ldpc_encode(bits_pkt, fec)
    elif fec:
        enc_bytes, fec_stats = fec_rs_encode(bytes_pkt, fec)
        tx_bits = bytes_to_bits(enc_bytes)
    else:
        tx_bits = bits_pkt
        fec_stats = None
    enc_bits = tx_bits.size

    # ---- 2. 加扰 (确定性 PN 序列异或) ----
    tx_bits_scrambled = scramble_bits(tx_bits, seed=scr_seed)

    # ---- 3. QAM 映射 + 补零 + 哑元补齐 ----
    sym = qammod(np.concatenate([tx_bits_scrambled,
                                 np.zeros((-enc_bits) % p['k'],
                                          dtype=np.uint8)]), M)
    n_dummy = (-sym.size) % p['N_data']
    if n_dummy > 0:
        sym = np.concatenate([sym, qammod(np.random.randint(
            0, 2, n_dummy * p['k'], dtype=np.uint8), M)])

    # ---- 4. OFDM 调制 + 频谱整形 ----
    sig, _ = ofdm_modulate(sym, p)
    sig, tau = ofdm_tx_filter(sig, p)

    # ---- 5. 峰值归一化 + CAF 削峰 + 再次归一化 ----
    g_norm = 1 / np.max(np.abs(sig))
    x = sig * g_norm
    if caf:
        from function.PA_DVR import CAF_new
        caf = dict(caf)
        thr_dB = caf.get('thr_dB', 9.5)
        c_caf = CAF_new(x, thr_dB,
                        caf.get('oversample', 6),
                        caf.get('T', 10),
                        caf.get('hard_clipping', 0),
                        caf.get('thr_new', thr_dB + 0.3))
        x = x - c_caf
    x = x / np.max(np.abs(x))
    if norm_rms is not None:
        assert norm_rms > 0, 'norm_rms 必须为正'
        x = x * (norm_rms / np.sqrt(np.mean(np.abs(x) ** 2)))

    info = {
        'enc_bits': enc_bits,
        'tx_bits_scrambled': tx_bits_scrambled,
        'tx_sym': sym,
        'n_dummy': n_dummy,
        'g_norm': g_norm,
        'tau': tau,
        'fec_stats': fec_stats,
        'n_bytes': n_bytes,
    }
    return x, info


def ofdm_rx_packet(rx_sig, p, info, fec=None, scr_seed=20240517,
                   plot=False, savepath=None, label='RX'):
    """单包解调链路: 去CP/FFT -> 导频增益估计 -> QAM硬判决 -> 解扰
    -> EVM / 纠错前误码率统计 -> 软LLR -> FEC 译码 -> 字节还原

    参数:
        rx_sig   : 接收到的 OFDM 复基带信号
        p        : ofdm_params 参数字典 (发射端同款, 含 txFilterTau)
        info     : ofdm_tx_packet 返回的参考信息
        fec      : FEC 配置 dict (与发射端一致)
        scr_seed : 扰码种子
        plot     : 是否绘制星座图 (TX 蓝 / RX 红)
        savepath : 星座图保存路径 (PNG), None = 不保存
        label    : 图标题标签 (如 'woDPD' / 'withDPD')

    返回:
        res : dict
              {'evm', 'ber_raw', 'n_err_raw', 'ber_after', 'n_err_after',
               'rx_sym', 'g_hat', 'bytes_dec', 'n_iter'}
    """
    M = p['M']
    rx_sig = np.asarray(rx_sig)
    fec = dict(fec or {})

    # 频谱整形滤波器群时延补偿参数 (发射端 ofdm_tx_filter 的 tau)
    p = dict(p)
    if info.get('tau'):
        p['txFilterTau'] = info['tau']

    # ---- 1. 解调 (导频增益估计 + 均衡) ----
    rx_sym, g_hat = ofdm_demodulate(rx_sig, p)

    # ---- 2. QAM 硬判决 + 解扰 ----
    bits_rx_pad = qamdemod(rx_sym, M)
    enc_bits = info['enc_bits']
    rx_bits_scrambled = bits_rx_pad[:enc_bits]
    rx_bits = scramble_bits(rx_bits_scrambled, seed=scr_seed)   # 解扰

    # ---- 3. EVM (发射符号 vs 接收星座) ----
    tx_sym = info['tx_sym']
    n_ref = min(tx_sym.size, rx_sym.size)
    evm = np.sqrt(np.mean(np.abs(rx_sym[:n_ref] - tx_sym[:n_ref]) ** 2) /
                  np.mean(np.abs(tx_sym[:n_ref]) ** 2)) * 100

    # ---- 4. 纠错前误码率 (加扰域对比) ----
    n_err_raw = int(np.count_nonzero(rx_bits_scrambled !=
                                     info['tx_bits_scrambled']))
    ber_raw = n_err_raw / enc_bits

    # ---- 5. 软判决 LLR + 软解扰 ----
    var_est = np.mean(np.abs(rx_sym - qammod(bits_rx_pad, M)) ** 2) / 2
    llr = qam_soft_llr(rx_sym, M, var_est)[:enc_bits]
    rng = np.random.RandomState(scr_seed)
    pn = (rng.rand(llr.size) > 0.5).astype(float)
    llr = llr * (1 - 2 * pn)

    # ---- 6. FEC 译码 ----
    useLDPC = bool(fec.get('type')) and str(fec.get('type')).lower() == 'ldpc'
    n_iter = None
    if useLDPC:
        nFEC, kFEC, codeUsed = ldpc_pcm(fec.get('code'), fec.get('rate'))
        fecDec = {'code': codeUsed, 'fecRate': fec.get('rate'),
                  'n': nFEC, 'k': kFEC,
                  'nPadBits': (-info['n_bytes'] * 8) % kFEC, 'llr': True}
        bits_dec, st = fec_ldpc_decode(llr, fecDec)
        bytes_dec = bits_to_bytes(bits_dec)
        n_iter = st['nIter']
    elif fec:
        fecDec = {'n': 255, 'k': fec.get('k', 223),
                  'nPadBytes': (-info['n_bytes']) % fec.get('k', 223)}
        bytes_dec, _ = fec_rs_decode(bits_to_bytes(rx_bits), fecDec)
    else:
        bytes_dec = bits_to_bytes(rx_bits)
    assert bytes_dec.size == info['n_bytes'], '还原字节数与发射包不符'

    # ---- 7. 纠错后误码率 (字节 -> 比特级) ----
    tx_bytes = info.get('tx_bytes')
    if tx_bytes is not None and tx_bytes.size == info['n_bytes']:
        n_err_after = int(np.count_nonzero(
            bytes_to_bits(bytes_dec) != bytes_to_bits(tx_bytes)))
    else:
        n_err_after = 0   # 未提供原始字节时无法统计 (仅打印用)
    ber_after = n_err_after / (bytes_dec.size * 8) if bytes_dec.size else 0.0

    # ---- 8. 星座图 (TX 蓝 / RX 红) ----
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n_show = min(500000, tx_sym.size)
        idx_p = np.random.choice(tx_sym.size, n_show, replace=False)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(np.real(tx_sym[idx_p]), np.imag(tx_sym[idx_p]), '.',
                color=[196 / 255, 215 / 255, 248 / 255], markersize=12)
        ax.plot(np.real(rx_sym[idx_p]), np.imag(rx_sym[idx_p]), '.',
                color=[0.7, 0.13, 0.13], markersize=12)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal')
        ax.set_title(f'{M}-QAM Constellation (blue=TX red={label})')
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=150)
        plt.close(fig)

    res = {
        'evm': evm,
        'ber_raw': ber_raw,
        'n_err_raw': n_err_raw,
        'ber_after': ber_after,
        'n_err_after': n_err_after,
        'rx_sym': rx_sym,
        'g_hat': g_hat,
        'bytes_dec': bytes_dec,
        'n_iter': n_iter,
    }
    return res


def ofdm_tx_file(zipFile, M, BW, fs=None, cpRatio=1 / 8, fec=None,
                 max_len_pkt=80000, caf=None, norm_rms=None,
                 scr_seed=20240517, verbose=True):
    """整文件 OFDM 发射: 读取文件 -> 分包 -> 逐包 ofdm_tx_packet

    参数:
        zipFile     : 待传输文件路径 (任意文件)
        M / BW / fs / cpRatio / fec / caf / norm_rms : 见 ofdm_tx_packet
        max_len_pkt : 每包 OFDM 信号长度上限 (采样点)

    返回:
        sig_pkts : list, 每项为一个数据包的复基带信号
        meta     : dict, 解调所需全部参数:
                   {'p', 'zipFile', 'nBytes', 'nPkt', 'bytesPkt',
                    'pktBytes', 'encBits', 'fec', 'info_list'}
    """
    if fs is None:
        fs = 2 * BW
    p = ofdm_params(BW, M, fs, cpRatio)
    L = p['N_fft'] + p['N_cp']
    n_sym_pkt = max(1, int(np.floor(max_len_pkt / L)))
    assert n_sym_pkt >= 1, 'maxLen 太小: 至少容纳 1 个 OFDM 符号'

    fec = dict(fec or {})
    useLDPC, nFEC, kFEC = _fec_block_sizes(fec, p)
    avail_bits = n_sym_pkt * p['N_data'] * p['k']
    n_blk_pkt = int(np.floor(avail_bits / nFEC))
    bytes_pkt = int(np.floor(n_blk_pkt * kFEC / 8))

    with open(zipFile, 'rb') as f:
        bytes_all = np.frombuffer(f.read(), dtype=np.uint8)
    n_bytes = bytes_all.size
    n_pkt = max(1, int(np.ceil(n_bytes / bytes_pkt)))

    sig_pkts = []
    info_list = []
    pkt_bytes = np.zeros(n_pkt, dtype=np.int64)
    enc_bits = np.zeros(n_pkt, dtype=np.int64)
    for i in range(n_pkt):
        b0 = i * bytes_pkt
        b1 = min((i + 1) * bytes_pkt, n_bytes)
        pkt_bytes[i] = b1 - b0
        x_pkt, info_i = ofdm_tx_packet(bytes_all[b0:b1], p, fec,
                                       scr_seed=scr_seed, caf=caf,
                                       norm_rms=norm_rms)
        info_i['tx_bytes'] = bytes_all[b0:b1]
        enc_bits[i] = info_i['enc_bits']
        sig_pkts.append(x_pkt)
        info_list.append(info_i)

    meta = dict(p)
    meta['zipFile'] = zipFile
    meta['nBytes'] = n_bytes
    meta['nPkt'] = n_pkt
    meta['bytesPkt'] = bytes_pkt
    meta['pktBytes'] = pkt_bytes
    meta['encBits'] = enc_bits
    meta['fec'] = fec
    meta['info_list'] = info_list
    if verbose:
        print(f'[ofdm_tx_file] 文件 {n_bytes/1e6:.2f} MB -> {n_pkt} 包, '
              f'每包 {bytes_pkt} 字节 / 信号 {n_sym_pkt*L} 点')
    return sig_pkts, meta


def ofdm_rx_file(rx_sig_pkts, meta, fec=None, scr_seed=20240517,
                 plot=False, savepath=None, label='RX', verbose=True):
    """整文件 OFDM 解调评估: 逐包 ofdm_rx_packet -> 拼接还原

    参数:
        rx_sig_pkts : 接收信号包列表 (list of 复基带信号)
        meta        : ofdm_tx_file 返回的 meta
        fec / scr_seed / plot / savepath / label : 见 ofdm_rx_packet

    返回:
        res : dict
              {'ok', 'bytes_rx', 'evm_pkt', 'ber_raw_pkt', 'ber_after_pkt',
               'evm_mean', 'ber_raw_mean', 'ber_after_mean',
               'pkt_results'}
    """
    fec = dict(fec or meta.get('fec') or {})
    p = meta
    n_pkt = meta['nPkt']
    evm_pkt = np.zeros(n_pkt)
    ber_raw_pkt = np.zeros(n_pkt)
    ber_after_pkt = np.zeros(n_pkt)
    bytes_cell = []
    pkt_results = []
    for i in range(n_pkt):
        sp = None
        if savepath:
            base, ext = os.path.splitext(savepath)
            sp = f'{base}_pkt{i+1}{ext}'
        res_i = ofdm_rx_packet(rx_sig_pkts[i], p, meta['info_list'][i],
                               fec=fec, scr_seed=scr_seed,
                               plot=plot, savepath=sp, label=label)
        evm_pkt[i] = res_i['evm']
        ber_raw_pkt[i] = res_i['ber_raw']
        ber_after_pkt[i] = res_i['ber_after']
        bytes_cell.append(res_i['bytes_dec'])
        pkt_results.append(res_i)
        if verbose:
            print(f'  包 {i+1:3d}/{n_pkt}  EVM {res_i["evm"]:6.3f}%  '
                  f'BER(纠错前) {res_i["ber_raw"]:.2e}  '
                  f'BER(纠错后) {res_i["ber_after"]:.2e}')

    bytes_rx = np.concatenate(bytes_cell)
    # 与原始文件对比 (若存在)
    ok = False
    import os as _os
    if _os.path.isfile(meta['zipFile']):
        with open(meta['zipFile'], 'rb') as f:
            bytes0 = np.frombuffer(f.read(), dtype=np.uint8)
        ok = bool(np.array_equal(bytes_rx, bytes0))
    res = {
        'ok': ok,
        'bytes_rx': bytes_rx,
        'evm_pkt': evm_pkt,
        'ber_raw_pkt': ber_raw_pkt,
        'ber_after_pkt': ber_after_pkt,
        'evm_mean': float(np.mean(evm_pkt)),
        'ber_raw_mean': float(np.mean(ber_raw_pkt)),
        'ber_after_mean': float(np.mean(ber_after_pkt)),
        'pkt_results': pkt_results,
    }
    return res
