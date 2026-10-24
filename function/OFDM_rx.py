# -*- coding: utf-8 -*-
"""
OFDM_rx.py — Python 解调 MATLAB main_ofdm_zip_transmit* 生成信号所需的补充工具
==============================================================================
与 function/OFDM_transmit.py 配合使用。新增内容:

1. MATLAB 兼容 QAM 星座映射
   MATLAB qammod/qamdemod(...,'UnitAveragePower',true) 在大调制阶数
   (M>=256) 下的内部电平/比特排布与 16/64-QAM 的简单公式不一致, 因此
   直接使用由 MATLAB 导出的参考星座表 (data/OFDM_16384QAM/qam16384_ref.npz,
   下标 = 符号索引, 比特按 MSB 在前 de2bi(idx,k,'left-msb')):
     qammod_bits  / qamdemod_bits / qam_soft_llr

2. MATLAB p 参数结构 -> Python 参数字典
   main_ofdm_zip_transmit*.mlx 保存的 .mat 中 p 是 1 基 FFT bin 索引,
   function/OFDM_transmit.py 内部用 0 基索引:
     matlab_p_to_python(pm)

3. DVB-S2 LDPC (n=64800, 常规帧) 置信传播译码
   MATLAB 端 ldpc_pcm([], rate) 返回 dvbs2ldpc 稀疏校验矩阵 H (系统码,
   信息位在前 k 位)。Python 端用向量化 sum-product BP 译码:
     build_B_csc / dvb_bp_decode / dvb_soft_decode_stream
   (函数只新增, 不改动 OFDM_transmit.py 现有行为。)
"""

import os
import numpy as np

# ----------------------------------------------------------------------
# 1) MATLAB 兼容 QAM (参考星座表驱动)
# ----------------------------------------------------------------------
def _axis_tables_from_const(const, m):
    """由参考星座 const(下标=符号索引) 提取两轴升序电平及电平的比特标签。

    方形 QAM 结构: 符号索引 idx 的 MSB 前 m 位决定实轴, 后 m 位决定虚轴。
    返回:
      rl / il   : 实/虚轴升序电平 (长度 L)
      r_h / i_h : 每个轴电平(按升序电平索引)对应的半整数 h (0..L-1)
      r_bits / i_bits : (L, m) 0/1 比特表 (MSB 在前)
    """
    k = 2 * m
    L = 1 << m
    re = np.real(const)
    im = np.imag(const)
    rl = np.sort(np.unique(np.round(re, 12)))
    il = np.sort(np.unique(np.round(im, 12)))
    assert rl.size == L and il.size == L, '参考星座不是可分离方形 QAM'
    r_h = np.zeros(L, dtype=np.int64)
    i_h = np.zeros(L, dtype=np.int64)
    # 每个实轴电平对应的第一个 idx -> idx>>m = 半整数 h
    for j, rv in enumerate(rl):
        idx0 = int(np.nonzero(np.round(re, 12) == rv)[0][0])
        r_h[j] = idx0 >> m
    for j, iv in enumerate(il):
        idx0 = int(np.nonzero(np.round(im, 12) == iv)[0][0])
        i_h[j] = idx0 & (L - 1)
    r_bits = ((r_h[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(np.uint8)
    i_bits = ((i_h[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(np.uint8)
    return rl, il, r_h, i_h, r_bits, i_bits


def load_qam_table(M, const_npz=None, const=None):
    """载入 MATLAB 兼容 QAM 表。const_npz: npz 路径含字段 const (长度 M);
    const 也可直接给 ndarray。"""
    if const is None:
        if const_npz is None:
            const_npz = os.path.join(os.path.dirname(__file__), '..', 'data',
                                     'OFDM_16384QAM', f'qam{M}_ref.npz')
        const = np.load(const_npz)['const']
    const = np.asarray(const).ravel()
    assert const.size == M
    k = int(round(np.log2(M)))
    m = k // 2
    assert 2 ** k == M and 2 * m == k
    rl, il, r_h, i_h, r_bits, i_bits = _axis_tables_from_const(const, m)
    return {
        'M': M, 'k': k, 'm': m, 'L': 1 << m,
        'const': const.astype(complex),
        'rl': rl, 'il': il, 'r_h': r_h, 'i_h': i_h,
        'r_bits': r_bits, 'i_bits': i_bits,
    }


def qammod_bits(bits, tbl):
    """0/1 比特 (MSB 在前, 长度 k 整数倍) -> MATLAB 兼容星座符号列向量"""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    k = tbl['k']
    n = bits.size // k
    idx = (bits[:n * k].reshape(n, k)
           * (1 << np.arange(k - 1, -1, -1))).sum(axis=1).astype(np.int64)
    return tbl['const'][idx]


def qamdemod_bits(sym, tbl):
    """MATLAB 兼容 QAM 硬判决 -> 0/1 比特 (MSB 在前)"""
    sym = np.asarray(sym).ravel()
    L, m, k = tbl['L'], tbl['m'], tbl['k']
    rl, il = tbl['rl'], tbl['il']
    x = np.clip(np.round((np.real(sym) - rl[0]) / (rl[1] - rl[0])).astype(
        np.int64), 0, L - 1)
    y = np.clip(np.round((np.imag(sym) - il[0]) / (il[1] - il[0])).astype(
        np.int64), 0, L - 1)
    hi = tbl['r_h'][x]
    lo = tbl['i_h'][y]
    idx = (hi << m) | lo
    bits = ((idx[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8)
    return bits.ravel()


def qam_soft_llr(sym, tbl, sigma2):
    """快速 max-log 软判决 LLR (LLR>0 -> 比特 1; 逐符号 MSB 在前),
    与 MATLAB qam_soft_llr.m / OFDM_transmit.qam_soft_llr 同口径。"""
    sym = np.asarray(sym).ravel()
    n = sym.size
    m, k, L = tbl['m'], tbl['k'], tbl['L']
    rl, il = tbl['rl'], tbl['il']
    r_bits, i_bits = tbl['r_bits'], tbl['i_bits']
    llr = np.zeros(n * k)
    x = np.real(sym)[:, None]
    y = np.imag(sym)[:, None]
    # 逐位最近相反比特电平的欧氏距离 (两轴分别处理)
    dr = (x - rl[None, :]) ** 2        # (n, L)
    di = (y - il[None, :]) ** 2
    for b in range(m):
        d0r = np.where(r_bits[:, b][None, :] == 0, dr, np.inf).min(axis=1)
        d1r = np.where(r_bits[:, b][None, :] == 1, dr, np.inf).min(axis=1)
        d0i = np.where(i_bits[:, b][None, :] == 0, di, np.inf).min(axis=1)
        d1i = np.where(i_bits[:, b][None, :] == 1, di, np.inf).min(axis=1)
        llr[np.arange(n) * k + b] = (d0r - d1r) / sigma2
        llr[np.arange(n) * k + m + b] = (d0i - d1i) / sigma2
    return llr


# ----------------------------------------------------------------------
# 2) MATLAB p 结构转换
# ----------------------------------------------------------------------
def matlab_p_to_python(pm):
    """MATLAB save 下来的 p 结构 (scipy np.void, 字段可能带 object 包装,
    电平/Fft bin 为 1 基) -> function/OFDM_transmit.py 使用的 0 基字典。"""
    def getf(key):
        v = pm[key]
        if isinstance(v, np.ndarray) and v.dtype == object:
            v = v.item()
        return np.asarray(v)

    def num(key):
        return float(getf(key).ravel()[0])

    p = {
        'BW': num('BW'), 'fs': num('fs'), 'M': int(num('M')),
        'k': int(num('k')), 'N_fft': int(num('N_fft')),
        'fsc': num('fsc'), 'N_used': int(num('N_used')),
        'N_data': int(num('N_data')), 'nPilot': int(num('nPilot')),
        'N_cp': int(num('N_cp')), 'cpRatio': num('cpRatio'),
        'occBW': num('occBW'), 'Tsym': num('Tsym'),
        'rateMbps': num('rateMbps'), 'txFilterTau': num('txFilterTau'),
    }
    for key in ('posBins', 'negBins', 'allDataBins', 'pilotBins',
                'dataBins', 'pilotPos', 'dataPos'):
        p[key] = getf(key).ravel().astype(np.int64) - 1
    p['pilotSeq'] = getf('pilotSeq').ravel()
    return p


# ----------------------------------------------------------------------
# 3) DVB-S2 LDPC (64800 常规帧) 向量化 sum-product BP 译码
# ----------------------------------------------------------------------
def _build_B_csc(col_by_row):
    """(n x E) 变量-边 关联矩阵 (行=变量节点, 列=边, 1 表示关联)"""
    from scipy.sparse import csc_matrix
    n = int(col_by_row.max()) + 1
    E = len(col_by_row)
    rows = col_by_row.astype(np.int64)
    cols = np.arange(E, dtype=np.int64)
    return csc_matrix((np.ones(E, dtype=np.float64), (rows, cols)),
                      shape=(n, E))


def _dvb_bp_prep(H):
    """H: scipy sparse (M x n)。返回复用数组: Hcsr / Bcsc / row 元数据。"""
    Hcsr = H.tocsr().astype(np.float64) if not H.dtype == np.float64 else H.tocsr()
    B = _build_B_csc(Hcsr.indices)
    row_ptr = Hcsr.indptr
    Mrows = Hcsr.shape[0]
    rows = np.repeat(np.arange(Mrows, dtype=np.int64), np.diff(row_ptr))
    seg_starts = row_ptr[:-1]
    deg_odd = (np.diff(row_ptr) & 1).astype(np.int8)[:, None]
    return Hcsr, B, rows, seg_starts, deg_odd


def dvb_bp_decode(llr, Hcsr, prep, max_iter=30, llr_sat=25.0):
    """对 (n, nBlk) LLR 矩阵做 flooding sum-product BP 译码 (LLR>0 -> 比特 1)。
    prep 由 _dvb_bp_prep 返回。返回 (hard, n_iter)。
    校验节点更新带 (-1)^deg 符号因子 (与 MATLAB/python LDPC 约定一致)。"""
    Hcsr, B, rows, seg_starts, deg_odd = prep
    n, nBlk = llr.shape
    chan = np.clip(llr, -llr_sat, llr_sat).astype(np.float64)
    col_by_row = Hcsr.indices
    v2c = chan[col_by_row].copy()
    hard = np.zeros((n, nBlk), dtype=np.uint8)
    n_iter = 0
    for it in range(1, max_iter + 1):
        half = v2c * 0.5
        t = np.tanh(half)
        with np.errstate(divide='ignore', invalid='ignore'):
            lt = np.log(np.abs(t))
        lt = np.maximum(lt, -700.0)
        seg_log = np.add.reduceat(lt, seg_starts, axis=0)
        excl_log = seg_log[rows] - lt
        with np.errstate(over='ignore', invalid='ignore'):
            c2v_mag = 2.0 * np.arctanh(np.minimum(np.exp(excl_log),
                                                  1.0 - 1e-15))
        neg = v2c < 0
        parity = np.add.reduceat(neg.astype(np.int8), seg_starts, axis=0) & 1
        parity_eff = (parity ^ deg_odd)
        sgn = np.where(parity_eff[rows] ^ neg, -1.0, 1.0)
        c2v = sgn * c2v_mag
        var_total = B.dot(c2v)
        v2c = chan[col_by_row] + var_total[col_by_row] - c2v
        v2c = np.clip(v2c, -llr_sat, llr_sat)
        posterior = chan + var_total
        hard_new = (posterior > 0).astype(np.uint8)
        hard = hard_new
        n_iter = it
        synd = (Hcsr.dot(hard_new.astype(np.float64)) % 2).astype(np.int8)
        if not np.any(synd):
            break
    return hard, n_iter


def dvb_soft_decode_stream(llr_stream, H, n, k, nPadBits=0, max_iter=30,
                           info_axis=0, prep=None):
    """对交织后的软 LLR 流 (长度 = n*nBlk, 码字间比特交织) 译码并去交织。
    返回 (info_bits, stats)；info_bits 长度为 nBlk*k - nPadBits。
    info_axis: dvbs2ldpc 为系统码, 信息位在每码字前 k 位 (info_axis=0)。
    prep: 可选, _dvb_bp_prep(H) 的预计算结果 (重复译码时复用)。
    """
    n = int(n); k = int(k)
    llr_stream = np.asarray(llr_stream).ravel()
    nBlk = llr_stream.size // n
    assert nBlk * n == llr_stream.size, '软 LLR 流长度不是码字长度整数倍'
    codeMat = llr_stream.reshape(nBlk, n, order='F').T      # (n, nBlk)
    if prep is None:
        prep = _dvb_bp_prep(H)
    hard, n_iter = dvb_bp_decode(codeMat, H.tocsr(), prep, max_iter=max_iter)
    if info_axis == 0:
        info = hard[:k]
    else:
        info = hard[n - k:]
    infoStream = info.ravel(order='F')
    if nPadBits > 0:
        infoStream = infoStream[:-nPadBits]
    stats = {'nBlk': nBlk, 'nIter': n_iter,
             'nErrCW': int(np.any(
                 (H.tocsr().dot(hard.astype(np.float64)) % 2) != 0, axis=0).sum())}
    return infoStream, stats
