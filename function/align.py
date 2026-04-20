import numpy as np
from scipy import signal
from scipy.special import sinc
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev
from function.Function_Calculate import rms


def align(x, y, method="Spline"):
    """
    The align function is used to align the original baseband signal to the
    output of power amplifier. The align process consist of four process:
    1. Coarse align: use correlate function to find the delay between the two signals
    2. Fine align: use Spline (Cubic function) or PCF (quadratic function) or
        LS (Least Square) to find the finer delay between the two signals.
        Then, use sinc interpolation to interpolate and align the two signals.
    3. Phase synchronization: compensate the phase difference between the two signals
    4. RMS normalization: normalize the RMS of the two signals

    Args:
    x: input signal
    y: output signal, waited to be aligned

    Returns:
    ycir: aligned output signal

    Reference:
    [1] PCF: http://ieeexplore.ieee.org/document/7017459/
    """
    assert method == "Spline" or "PCF" or "LS"
    y = y[: len(x)]
    x = x.copy()
    y = y.copy()
    ycir = coarse_align(x, y)
    if method == "Spline":
        for i in range(3):
            Lmax = fine_align(np.abs(x), np.abs(ycir))
            ycir = sinccircular(ycir, Lmax, 200)
    elif method == "PCF":
        # The index of sinc interpolation sequence, ideally
        # should be -lnf to lnf, we use -128:128 here
        nsamp = np.arange(-128, 129)
        iteration = 6
        for i in range(iteration):
            # Compute the maximum correlation value and the two side values
            xcorrmaxval = abs(np.dot(x.conj(), ycir)) ** 2
            xcorrmaxvalleft = abs(np.dot(x.conj(), np.roll(ycir, 1))) ** 2
            xcorrmaxvalright = abs(np.dot(x.conj(), np.roll(ycir, -1))) ** 2
            # Use the extremum of the quadratic function to
            # estimate the fractional delay
            fracdelay = (
                0.5
                * (xcorrmaxvalleft - xcorrmaxvalright)
                / (xcorrmaxvalleft + xcorrmaxvalright - 2 * xcorrmaxval)
            )
            # Use the sinc interpolation to interpolate the signal,
            # the sinc interpolation is performed in the range of -0.5 to 0.5
            # of the sample period, if it is greater than 0.5, the sequence
            # is first circularly shifted, and then the fractional delay is
            # converted to this time interval
            if abs(fracdelay) > 0.5:
                ycir = np.roll(ycir, -round(fracdelay))
                if fracdelay > 0.5:
                    fracdelay = 1 - fracdelay
                else:
                    fracdelay = 1 + fracdelay
            # Genearte the sinc interpolation sequence
            sincvec = sinc(fracdelay - nsamp)
            ycircyc = np.concatenate((ycir[-128:], ycir, ycir[:128]))
            for i in range(len(x)):
                ycir[i] = np.dot(sincvec, ycircyc[i : i + 257])
    elif method == "LS":
        yshift = np.concatenate(([0], ycir[:-1]))
        # LS algorithm uses the previous point and current point as the interpolation
        Y = np.column_stack((ycir, yshift))
        coeffs = np.linalg.lstsq(
            np.dot(Y.conj().T, Y), np.dot(Y.conj().T, x), rcond=None
        )
        ycir = np.dot(Y, coeffs[0])

    # Phase Synchronization
    pd = x / ycir
    pd_avg = np.mean(pd)
    ycir = ycir * pd_avg

    # RMS normalizaiton
    ycir = ycir * rms(x) / rms(ycir)

    return ycir


def coarse_align(x, y):
    """
    Find the largets correlation value, in the precesion
    of a single sample, to eliminate the integer delay.

    Args:
    x: input signal
    y: output signal, waited to be aligned

    Returns:
    ycir: coarse aligned output signal
    """
    x = np.concatenate((x, x))
    corr = signal.correlate(x, y, mode="valid")
    delay = np.argmax(abs(corr))
    ycir = np.roll(y, delay)
    return ycir


def sinccircular(x, n, ns=100):
    """
    Compensate for the integer and fractional delay using sinc interpolation.

    Args:
    x: input signal
    n: interger plus fractional delay
    ns: number of samples in the sinc interpolation filter

    Returns:
    y: sinc interpolated signal
    """
    assert int(ns) == ns, "ns must be an integer"

    # Split delay into integer part ni which is performed by sample rotating
    # and fractional part n which is performed by sinc interpolation
    ni = int(np.floor(n))
    n = n - ni

    y = np.concatenate([x[-ni:], x[:-ni]])

    if n > 0:
        # Setup interpolation filter
        Bsinc = sinc(np.arange(-ns, ns + 1) - n)  # Interpolation filter

        # Pad data before and after data to allow the filter to "fill up"
        # and "empty out"
        ni = int(np.floor(ns / len(y)))  # How many times vector needs to be repeated
        nr = ns - ni * len(y)  # Remainder
        y1 = np.concatenate([np.tile(y, ni), y, np.tile(y, ni)])
        if nr > 0:
            y1 = np.concatenate([y[-nr:], y1, y[:nr]])

        # Do sinc interpolation
        y1 = signal.lfilter(Bsinc, 1, y1)
        y1 = y1[-len(y) :]
        y = y1

    return y


def fine_align(x, y):
    """
    This function searches the total search space very fast, taking advantage of
    that correlation is 32 samples wide in a pseudo way, since np.correlate
    don't support maxlag.

    Args:
    x: input signal
    y: output signal, waited to be aligned

    Returns:
    Lmax: the finer delay between the two signals
    """

    Dfact = 16
    C_finer = np.correlate(x, y, mode="same")
    L_finer = np.arange(len(x)) - len(x) / 2
    low_index = len(C_finer) // 2 - Dfact
    high_index = len(C_finer) // 2 + Dfact
    C_finer = C_finer[low_index:high_index]
    L_finer = L_finer[low_index:high_index]

    LI = np.linspace(-32, 32, 1000)
    spl = splrep(L_finer, C_finer)
    CI = splev(LI, spl)

    Lmax = LI[np.argmax(CI)]
    Lmax = Lmax

    return Lmax
