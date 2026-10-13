"""
rrc.py -- Root-raised-cosine pulse-shaping filter.

Two equivalent forms:
  - rrc_time():  the classic truncated time-domain impulse response (needs a
                 'delay' = half-length in symbols), kept for reference.
  - rrc_freq():  the analytic closed-form frequency response.  Zero-phase, real,
                 NO 'delay' parameter, no truncation.  This is what the pipeline
                 should use: the matched filtering is a single per-frequency
                 multiply, so building the filter directly in frequency removes
                 the awkward time-domain truncation/roll.

Verified: rrc_time (zero-padded + rolled + FFT) and rrc_freq agree to ~3e-3 in
shape and give identical filtered signals (NMSE ~5e-9) for delay=510, rolloff
0.05.  The tiny shape difference is outside the signal band, hence irrelevant.
"""

import numpy as np


def rrc_time(rolloff, delay, OS):
    """Truncated time-domain RRC, length 2*round(delay*OS)+1, energy-normalized.

    rolloff : roll-off in [0, 1]
    delay   : half-length in symbols (truncation point of the ideal RRC)
    OS      : samples per symbol
    """
    n = round(delay * OS)
    h = np.zeros(2 * n + 1)
    h[n] = 1 + rolloff * (4 / np.pi - 1)
    for i in range(1, n + 1):
        t = i / OS
        if abs(t - 1 / (4 * rolloff)) < 1e-12:
            v = rolloff / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff)))
        else:
            v = (np.sin(np.pi * t * (1 - rolloff)) + 4 * rolloff * t * np.cos(np.pi * t * (1 + rolloff))) \
                / (np.pi * t * (1 - (4 * rolloff * t)**2))
        h[n + i] = v
        h[n - i] = v
    return h / np.sqrt(np.sum(h**2))


def rrc_freq(N, OS, rolloff):
    """Analytic RRC frequency response of length N (FFT order), peak = 1.

    No 'delay' parameter.  Apply once at TX and once at RX (root each side).

    N       : FFT length (number of samples in the block)
    OS      : samples per symbol
    rolloff : roll-off in [0, 1]
    """
    f = np.concatenate((np.arange(0, N // 2), np.arange(-N // 2, 0))) * OS / N  # units of fsym
    af = np.abs(f)
    H = np.zeros(N)
    H[af <= (1 - rolloff) / 2] = 1.0
    mid = (af > (1 - rolloff) / 2) & (af <= (1 + rolloff) / 2)
    H[mid] = np.sqrt(0.5 * (1 + np.cos(np.pi / rolloff * (af[mid] - (1 - rolloff) / 2))))
    return H


if __name__ == "__main__":
    # cross-check the two forms on a random signal
    N, OS, ro, delay = 8192, 1.125, 0.05, 510
    h = rrc_time(ro, delay, OS)
    pad = np.roll(np.concatenate((h, np.zeros(N - len(h)))), -(len(h) // 2))
    Ht = np.fft.fft(pad)
    Hf = rrc_freq(N, OS, ro)
    Hf = Hf / np.sqrt(np.mean(Hf**2)) * np.sqrt(np.mean(np.abs(Ht)**2))
    x = np.random.RandomState(0).randn(N) + 1j * np.random.RandomState(1).randn(N)
    yt = np.fft.ifft(np.fft.fft(x) * Ht)
    yf = np.fft.ifft(np.fft.fft(x) * Hf)
    nmse = np.mean(np.abs(yt - yf)**2) / np.mean(np.abs(yt)**2)
    print(f"time vs freq RRC, filtered-signal NMSE = {nmse:.2e}  (zero-phase: max|Im(Ht)|/max = {np.max(np.abs(Ht.imag))/np.max(np.abs(Ht)):.1e})")
