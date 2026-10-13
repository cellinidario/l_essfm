"""ldbp_eval.py -- numpy evaluation of an LDBP (Hager neural-DBP) model.

Mirrors the reference eval that produced the published LDBP curve points
(repro_ofc/test_ldbp.py): dual-pol, complex CD-FIR per step applied in frequency,
SCALAR Kerr phase per step using TOTAL dual-pol power, then matched filter +
overlap-add downsample (same tail as L-ESSFM). LDBP is a TIME-domain method so it
runs at n>=2 (digital oversampling 2).

parameters.csv layout per step (3 lines): CD-real, CD-imag, nl_param scalar.
The saved nl scalar already folds in bw.nl_param[NN] (ldbp.py save convention), so
the Kerr phase is exp(1j * |y|^2 * nl_scalar) with NO extra nl_param multiply.
"""
import numpy as np
try:
    import cupy as _cp
    _HAVE_CUPY = True
except Exception:          # no CuPy: the numpy path below is used
    _cp, _HAVE_CUPY = None, False


def load_ldbp_params(path, model_steps):
    """per step: (cd_filter complex [Lcd], nl_scalar float). 3 lines/step."""
    lines = open(path).read().strip().split('\n')
    if len(lines) != 3*model_steps:
        raise ValueError(f'LDBP parameter count does not match geometry: {path}')
    cd, nl = {}, {}
    for NN in range(model_steps):
        cr = np.array([float(v) for v in lines[3 * NN].split(',')])
        ci = np.array([float(v) for v in lines[3 * NN + 1].split(',')])
        cd[NN] = cr + 1j * ci
        nl[NN] = float(lines[3 * NN + 2].split(',')[0])
    return cd, nl


def ldbp_backprop(S, bw, cd, nl, y, x, P, gpu=None, single=True):
    """Dual-pol LDBP backprop. y:[Npol,Nsamp_a], x:[Npol,Nsym]. Returns effSNR [dB].
    Exact mirror of repro_ofc/test_ldbp.py backprop (the reference for 18.857).

    gpu: None -> CuPy if available. The step loop then runs on the GPU in double
    precision (input upcast to complex128 first, as numpy's FFT does): same values to
    ~1e-12 dB, 5-6x faster (checked 2026-10-07 on 15x80 km, 300 steps).
    single=True (default, GPU only): step loop in complex64, tail in double; agreed
    tolerance 0.01 dB (Dario, 2026-10-07)."""
    Nsamp_d, OS_a, OS_d, Nsym, Npol = S['Nsamp_d'], S['OS_a'], S['OS_d'], S['Nsym'], S['Npol']
    ps_rx_freq = S['ps_rx_freq']
    use_gpu = (_HAVE_CUPY if gpu is None else gpu) and _HAVE_CUPY
    xp = _cp if use_gpu else np
    single = single and use_gpu
    cdt = xp.complex64 if single else xp.complex128
    y = xp.asarray(y).astype(cdt)
    Y = xp.fft.fft(y); Y = Y[:, :Nsamp_d] + Y[:, -Nsamp_d:]; y = xp.fft.ifft(Y) / OS_a * OS_d
    for NN in range(bw.model_steps):
        # linear: complex CD-FIR, zero-padded and rolled by -cd_len//2+1, applied in freq
        cd_len = cd[NN].shape[0]
        cd_time = xp.zeros(Nsamp_d, dtype=cdt)
        cd_time[:cd_len] = xp.asarray(cd[NN])
        cd_time = xp.roll(cd_time, -cd_len // 2 + 1)
        y = xp.fft.ifft(xp.fft.fft(y) * xp.fft.fft(cd_time))
        # nonlinear: scalar Kerr phase using TOTAL dual-pol power
        ysq = xp.abs(y[0, :]) ** 2 + xp.abs(y[1, :]) ** 2
        y = y * xp.exp(1j * (ysq * nl[NN]).astype(ysq.dtype))
    y = _cp.asnumpy(y).astype(np.complex128) if use_gpu else y
    # matched filter + overlap-add downsample (same tail as L-ESSFM backprop)
    Y = np.fft.fft(y) * ps_rx_freq
    Yal = Y[:, :Nsym].copy()
    Yal[:, Nsym - Nsamp_d // 2:Nsym // 2] += Y[:, Nsamp_d // 2:Nsamp_d - Nsym // 2]
    Yal[:, Nsym // 2:Nsamp_d // 2] += Y[:, Nsamp_d - Nsym // 2:3 * Nsamp_d // 2 - Nsym]
    Yal[:, Nsamp_d // 2:Nsym] = Y[:, 3 * Nsamp_d // 2 - Nsym:]
    y = np.fft.ifft(Yal) / OS_d / np.sqrt(P / Npol) / np.sqrt(OS_d)
    x_hat = np.zeros([Npol, Nsym], dtype=np.complex64)
    for pp in range(Npol):
        phi = np.angle(np.dot(np.conj(x[pp, :]), y[pp, :]))
        x_hat[pp, :] = y[pp, :] * np.exp(-1j * phi)
    return -10.0 * np.log10(np.mean(np.abs(x - x_hat) ** 2))
