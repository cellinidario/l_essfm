"""forward_sim.py -- full-length forward propagation on the GPU (CuPy), to GENERATE
the simulation forward (e.g. 262144 symbols) when a saved forward_results*.npy is
not available for a scenario.

This is a faithful port of Dario's original cupy forward_propagation
(l-essfm_test.ipynb cell 11): pulse shaping, WDM comb, per-span ASE noise + SSFM
(split-step, model_steps), low-pass. Returns y at the ANALOG rate [Npol,Nsamp_a]
and the center-channel symbols x [Npol,Nsym] -- exactly the format of the saved
forward dicts ({P_W: [(y,x), ...]}), so backprop()/curve() consume it unchanged.

CuPy (not torch) on purpose: simulation needs no autograd, and cupy is a drop-in
numpy that matches the original bit-for-bit and is fast on the GPU.
"""
import numpy as np
from collections import OrderedDict
from system import ase_polarizations, ase_injection_location
try:
    import cupy as cp
    _HAVE_CUPY = True
except Exception:
    cp = np
    _HAVE_CUPY = False


def forward_propagation(S, P, seed=None):
    """One forward realization at launch power P [W]. Returns (y[Npol,Nsamp_a],
    x_center[Npol,Nsym]). Mirrors Dario's cupy forward_propagation."""
    rng = np.random.RandomState(seed)
    noise_rng = cp.random.RandomState(seed)
    Npol, Nch, Nsym = S['Npol'], S['Nch'], S['Nsym']
    OS_a, Nsamp_a = S['OS_a'], S['Nsamp_a']
    fsamp_a, spacing = S['fsamp_a'], S['spacing']
    fw = S['fw']
    location = ase_injection_location(S)

    # [SOURCE] constellation / Gaussian symbols
    if S['modulation'] == 'QAM' or S['modulation'] == '64-QAM':
        const = S['const']
        x = const[rng.randint(const.shape[0], size=[Npol, Nch, Nsym])]
    else:  # Gaussian
        x = (rng.normal(0, 1, [Npol, Nch, Nsym])
             + 1j * rng.normal(0, 1, [Npol, Nch, Nsym])) / np.sqrt(2)
    # [MODULATION] upsample + RRC pulse shaping (TX), scale to launch power
    x_up = np.zeros([Npol, Nch, Nsamp_a], dtype=np.complex64)
    x_up[:, :, ::OS_a] = x * np.sqrt(OS_a)
    u = np.fft.ifft(np.fft.fft(x_up) * S['ps_tx_freq']) * np.sqrt(P / Npol)
    # WDM comb
    u_wdm = np.zeros([Npol, Nsamp_a], dtype=np.complex64)
    nvec = np.arange(Nsamp_a)
    for NN in range(Nch):
        fs = (NN - Nch // 2) * spacing
        u_wdm += u[:, NN, :] * np.exp(1j * 2 * np.pi * fs * nvec / fsamp_a)

    # [CHANNEL] SSFM on the GPU
    u_wdm = cp.asarray(u_wdm)
    # A full 262144-symbol forward has 1.8 million analog samples. Keeping one
    # complex128 transfer function per step would use >11 GB at 400 steps, even
    # before the signal/FFT buffers. Cache only a few distinct lengths, preserving
    # the same float64 phase and complex128 transfer functions.
    cd_base = cp.asarray((fw.beta2 / 2) * (2*np.pi*fw.fvec)**2 * fw.direction)
    cd_cache = OrderedDict()
    def cd_filter(index):
        length = float(fw.cd_length[index])
        if length not in cd_cache:
            if len(cd_cache) >= 2:
                cd_cache.popitem(last=False)
            cd_cache[length] = cp.exp(1j * cd_base * length)
        cd_cache.move_to_end(length)
        return cd_cache[length]
    nlp = [float(fw.nl_param[MM]) for MM in range(fw.model_steps)]
    sig = np.sqrt(S['sigma2'] / 2)
    for _ in range(S['Nsp']):                                  # per span: ASE noise + SSFM
        shape = (ase_polarizations(S), Nsamp_a)
        noise = sig * (noise_rng.standard_normal(shape)
                       + 1j * noise_rng.standard_normal(shape)).astype(cp.complex64)
        if location == 'span_input':
            u_wdm = u_wdm + noise
        for MM in range(fw.model_steps):
            u_wdm = cp.fft.ifft(cp.fft.fft(u_wdm) * cd_filter(MM))
            u_wdm = u_wdm * cp.exp(1j * nlp[MM] * (cp.abs(u_wdm) ** 2).sum(axis=0))
        if location == 'span_output':
            u_wdm = u_wdm + noise
    # [RECEIVER] low-pass
    u_wdm = cp.asnumpy(u_wdm) if _HAVE_CUPY else u_wdm
    y = np.fft.ifft(np.fft.fft(u_wdm) * S['lp_freq'])
    return y, x[:, Nch // 2, :]


def generate_forward(S, Pgrid_dbm, n_real=1, seed=0):
    """Generate a full forward dict {P_W: [(y,x), ...]} like the saved files, for a
    grid of launch powers [dBm]. n_real realizations per power."""
    out = {}
    k = 0
    for pdbm in Pgrid_dbm:
        Pw = 10 ** (pdbm / 10) * 1e-3
        out[Pw] = [forward_propagation(S, Pw, seed=seed + k + i) for i in range(n_real)]
        k += n_real
    return out
