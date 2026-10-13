"""
system.py -- System setup, forward propagation, and backpropagation for the
L-ESSFM / ESSFM / OSSFM / EDC / Ideal-DBP study.

Self-contained (no exec of legacy files). Numpy/scipy only; no TensorFlow.
Used by:
  - forward.py        to generate + cache the received signal once
  - the notebooks     to evaluate any DBP method on a cached forward
  - validation        to cross-check trained models (params.csv) on a forward

Conventions
-----------
- effSNR = -10*log10(mean|x - x_hat|^2) after phase alignment, peak over power.
- A "method" applies a per-step GVD multiplier cd_mult[NN] and a per-step NLPR
  filter nlf[NN] (one-sided, mirrored to 2N-1 taps at apply time).
- RRC matched filtering is done in frequency (rrc_freq), no time-domain 'delay'.
"""

import numpy as np
import scipy as sp
import configparser

from rrc import rrc_freq

# physical constants
CO_H = 6.6260657e-34
CO_C0 = 299792458.0
CO_LAMBDA = 1550.0e-9
NU = CO_C0 / CO_LAMBDA
DB = 10.0 * np.log10(np.e)   # 4.3429...


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def ase_injection_location(S):
    """Where each span's ASE is added: 'span_input' (default) or 'span_output'
    (config option 'ase injection location')."""
    location = S.get('ase_injection_location', 'span_input')
    if location not in ('span_input', 'span_output'):
        raise ValueError(f'Unknown ASE injection location: {location}')
    return location


def ase_polarizations(S):
    """Independent ASE per polarization (default); 'legacy_shared' reproduces the old
    results, where both polarizations got the same noise."""
    mode = S.get('ase_mode', 'independent')
    if mode not in ('independent', 'legacy_shared'):
        raise ValueError(f'Unknown ASE mode: {mode}')
    return S['Npol'] if mode == 'independent' else 1


def effective_length(length, alpha_lin):
    if alpha_lin == 0:
        return length
    return (1 - np.exp(-alpha_lin * length)) / alpha_lin


def get_fvec(N, fs):
    return np.concatenate((np.arange(0, N // 2), np.arange(-N // 2, 0))) * fs / N


def qam_constellation(M):
    Msqrt = int(round(np.sqrt(M)))
    if Msqrt**2 != M:
        raise ValueError("M must be a perfect square (M = 4^m)")
    pam = -(Msqrt - 2 * np.arange(1, Msqrt + 1) + 1)
    re, im = np.meshgrid(pam, pam)
    const = (re + 1j * im).ravel()
    return const / np.sqrt(np.mean(np.abs(const)**2))


# --------------------------------------------------------------------------- #
# SSFM step geometry (faithful port of the validated ssfm_parameters class)
# --------------------------------------------------------------------------- #
class SSFMParameters:
    """Step sizes, per-step GVD lengths and nonlinear strengths for (back)prop.

    Required opts: alpha[dB/km->/m as 'alpha'], beta2, gamma, Nsp, Lsp, fsamp,
    Nsamp, step_size_method ('linear'|'logarithmic'), ssfm_method
    ('symmetric'|'asymmetric'), StPS, direction (+1 fwd, -1 bwd).
    Optional: combine_half_steps (default True), adjusting_factor (log, def 0.4).
    """

    def __init__(self, opts):
        self.__dict__.update(opts)
        alpha_lin = self.alpha / DB
        Nsp, Lsp, direction, StPS = self.Nsp, self.Lsp, self.direction, self.StPS
        if direction == +1 and Nsp > 1:
            raise ValueError("forward propagation valid only for 1 span")
        if getattr(self, 'combine_half_steps', None) is None:
            self.combine_half_steps = True
        if self.step_size_method == 'logarithmic' and 'adjusting_factor' not in opts:
            self.adjusting_factor = 0.4

        # step sizes for one span
        if self.step_size_method == 'logarithmic':
            a = self.adjusting_factor * alpha_lin
            delta = (1 - np.exp(-a * Lsp)) / StPS
            nn = (np.arange(StPS) + 1) if direction == -1 else (StPS - np.arange(StPS))
            step = -1 / a * np.log((1 - (StPS - nn + 1) * delta) / (1 - (StPS - nn) * delta))
        elif self.step_size_method == 'linear':
            step = Lsp / StPS * np.ones(StPS)
        else:
            raise ValueError("step_size_method must be 'linear' or 'logarithmic'")

        # cd_length, nl_length, amplifier_location
        if self.ssfm_method == 'symmetric' and self.combine_half_steps:
            M = Nsp * StPS + 1
            cd = np.zeros(M); nl = np.zeros(M)
            for s in range(Nsp):
                for m in range(StPS):
                    cd[s * StPS + m] = step[m] / 2 + step[(m + StPS - 1) % StPS] / 2
                    nl[s * StPS + m] = step[m]
            cd[0] = step[0] / 2
            cd[-1] = step[StPS - 1] / 2
            amp = np.zeros(M); amp[:-1:StPS] = 1
        elif self.ssfm_method == 'symmetric':
            M = Nsp * (StPS + 1)
            cd = np.concatenate([[step[0] / 2], (step[:-1] + step[1:]) / 2, [step[-1] / 2]])
            cd = np.tile(cd, Nsp)
            nl = np.tile(np.concatenate([step, [0]]), Nsp)
            amp = np.zeros(M); amp[::StPS + 1] = 1
        elif self.ssfm_method == 'asymmetric':
            M = Nsp * StPS
            cd = np.zeros(M); nl = np.zeros(M)
            for s in range(Nsp):
                for m in range(StPS):
                    cd[s * StPS + m] = step[m]
                    nl[s * StPS + m] = effective_length(step[m], abs(alpha_lin))
            amp = np.zeros(M); amp[::StPS] = 1
        else:
            raise ValueError("ssfm_method must be 'symmetric' or 'asymmetric'")

        # nonlinear strength per step, weighted by the (back)propagated power profile
        nl_param = direction * self.gamma * nl
        att = np.exp(-direction * alpha_lin * cd / 2)
        for NN in range(M):
            if direction == -1 and amp[NN] == 1:
                att[NN] *= np.exp(direction * alpha_lin * Lsp / 2)
        for NN in range(M):
            nl_param[NN] *= np.prod(att[:NN + 1])**2

        self.model_steps = M
        self.cd_length = cd
        self.nl_length = nl
        self.nl_param = nl_param
        N = self.Nsamp
        self.fvec = get_fvec(N, self.fsamp)

    def get_cd_filter_freq(self, NN):
        return (self.beta2 / 2) * (2 * np.pi * self.fvec)**2 * (self.direction * self.cd_length[NN])


# --------------------------------------------------------------------------- #
# system build from .ini
# --------------------------------------------------------------------------- #
def build_system(cfg_path):
    """Parse the .ini and build fixed system objects (filters, forward SSFM)."""
    d = {"number of polarizations": "2", "modulation": "64-QAM",
         "combine half-steps": "yes", "cd alpha": "1", "nl alpha": "1",
         'forward step size method': 'linear', 'forward split step method': 'symmetric'}
    cfg = configparser.ConfigParser(d); cfg.read(cfg_path)
    cs, ce, cg = cfg['system parameters'], cfg['L-ESSFM parameters'], cfg['data generation']
    S = {}
    S['Lsp'] = cs.getfloat('span length [km]') * 1e3
    S['alpha'] = cs.getfloat('alpha [dB/km]') * 1e-3
    S['gamma'] = cs.getfloat('gamma [1/W/km]') * 1e-3
    S['noise_figure'] = cs.getfloat('amplifier noise figure [dB]')
    S['ase_injection_location'] = cs.get('ase injection location', 'span_input')
    ase_injection_location(S)
    S['Npol'] = cs.getint('number of polarizations')
    S['Nsp'] = cs.getint('number of spans')
    S['fsym'] = cs.getfloat('symbol rate [Gbaud]') * 1e9
    mod = cs['modulation']
    S['rolloff'] = cs.getfloat('RRC roll-off')
    S['lp_bw'] = cs.getfloat('low-pass filter bandwidth [GHz]') * 1e9
    S['Nsym'] = cs.getint('data symbols per block')
    S['OS_a'] = cs.getint('analog oversampling'); S['OS_d'] = cs.getfloat('digital oversampling')
    S['Nch'] = cs.getint('number of channels'); S['spacing'] = cs.getfloat('channel spacing [GHz]') * 1e9
    Dps = cs.getfloat('D [ps/nm/km]') * 1e-6
    S['beta2'] = -Dps * CO_LAMBDA**2 / (2 * np.pi * CO_C0)
    S['cd_alpha'] = ce.getfloat('cd alpha'); S['nl_alpha'] = ce.getfloat('nl alpha')
    S['nl_filter_length'] = ce.getint('nl filter length')
    S['step_method_bw'] = ce['step size method']; S['ssfm_method_bw'] = ce['split step method']
    S['combine'] = ce.getboolean('combine half-steps')
    S['StPS_fw'] = cg.getint('forward steps per span')
    S['step_method_fw'] = cg['forward step size method']; S['ssfm_method_fw'] = cg['forward split step method']
    # training launch power [dBm] for this scenario (near its operating peak): the
    # PyTorch trainer trains at this power. Searched across all sections; default 9.0.
    S['train_power_dbm'] = next((float(cfg[sec]['training power [dbm]'])
                                 for sec in cfg.sections()
                                 if cfg.has_option(sec, 'training power [dbm]')), 9.0)

    alpha_lin = S['alpha'] / DB
    sef = 10.0**(S['noise_figure'] / 10.0) / 2.0
    N0 = (np.exp(alpha_lin * S['Lsp']) - 1.0) * CO_H * NU * sef
    S['sigma2'] = N0 * S['fsym'] * S['OS_a']
    S['Nsamp_a'] = S['Nsym'] * S['OS_a']; S['Nsamp_d'] = round(S['Nsym'] * S['OS_d'])
    S['fsamp_a'] = S['fsym'] * S['OS_a']; S['fsamp_d'] = S['fsym'] * S['OS_d']
    f_a = get_fvec(S['Nsamp_a'], S['fsamp_a'])
    S['modulation'] = 'QAM' if 'QAM' in mod else 'Gaussian'
    if S['modulation'] == 'QAM':
        S['const'] = qam_constellation(int(mod.split('-')[0]))

    # TX/RX matched filters in frequency (analytic RRC, no 'delay').
    # Scale by sqrt(OS): this reproduces exactly the normalization of the legacy
    # energy-normalized time-domain RRC after FFT (verified: in-band shape
    # correlation 1.0, scale = sqrt(OS_a) for TX, sqrt(OS_d) for RX).
    S['ps_tx_freq'] = rrc_freq(S['Nsamp_a'], S['OS_a'], S['rolloff']) * np.sqrt(S['OS_a'])
    S['ps_rx_freq'] = rrc_freq(S['Nsamp_d'], S['OS_d'], S['rolloff']) * np.sqrt(S['OS_d'])
    S['lp_freq'] = (np.abs(f_a) <= S['lp_bw'] / 2).astype(float)

    S['fw'] = SSFMParameters({"alpha": S['alpha'], "beta2": S['beta2'], "gamma": S['gamma'],
        "Nsp": 1, "Lsp": S['Lsp'], "fsamp": S['fsamp_a'], "Nsamp": S['Nsamp_a'],
        "step_size_method": S['step_method_fw'], "ssfm_method": S['ssfm_method_fw'],
        "StPS": S['StPS_fw'], "direction": 1})
    S['cfg_path'] = cfg_path           # so trainers can rebuild with a larger block
    return S


def forward(S, P, rng=None):
    """One forward realization at launch power P [W]: returns (y_rx, x_center)."""
    Npol, Nch, Nsym, OS_a, Nsamp_a = S['Npol'], S['Nch'], S['Nsym'], S['OS_a'], S['Nsamp_a']
    location = ase_injection_location(S)
    rng = np.random if rng is None else rng
    if S['modulation'] == 'QAM':
        x = S['const'][rng.randint(S['const'].shape[0], size=[Npol, Nch, Nsym])]
    else:
        x = (rng.normal(0, 1, [Npol, Nch, Nsym]) + 1j * rng.normal(0, 1, [Npol, Nch, Nsym])) / np.sqrt(2)
    x_up = np.zeros([Npol, Nch, Nsamp_a], dtype=np.complex64); x_up[:, :, ::OS_a] = x * np.sqrt(OS_a)
    u = np.fft.ifft(np.fft.fft(x_up) * S['ps_tx_freq']) * np.sqrt(P / Npol)
    u_wdm = np.zeros([Npol, Nsamp_a], dtype=np.complex64)
    for NN in range(Nch):
        fs = (NN - Nch // 2) * S['spacing']
        u_wdm += u[:, NN, :] * np.exp(1j * 2 * np.pi * fs * np.arange(Nsamp_a) / S['fsamp_a'])
    for _ in range(S['Nsp']):
        # one INDEPENDENT ASE field per polarisation. Drawing (1, Nsamp_a) and
        # letting it broadcast gave both polarisations the identical noise: the
        # per-pol power was right but the two were fully correlated, which the
        # Manakov Kerr term (|u_x|^2 + |u_y|^2) does see.
        shape = (ase_polarizations(S), Nsamp_a)
        noise = np.sqrt(S['sigma2'] / 2) * (rng.standard_normal(shape)
                                          + 1j * rng.standard_normal(shape))
        if location == 'span_input':
            u_wdm += noise
        for MM in range(S['fw'].model_steps):
            u_wdm = np.fft.ifft(np.fft.fft(u_wdm) * np.exp(1j * S['fw'].get_cd_filter_freq(MM)))
            u_wdm *= np.exp(1j * S['fw'].nl_param[MM] * (np.abs(u_wdm)**2).sum(axis=0))
        if location == 'span_output':
            u_wdm += noise
    y = np.fft.ifft(np.fft.fft(u_wdm) * S['lp_freq'])
    return y, x[:, Nch // 2, :]


def make_bw(S, ns, total_steps=None):
    """Backprop SSFMParameters. `ns` is always the TOTAL number of DBP steps over the
    whole link (the Fig.1b complexity axis). Internally we follow Stella's convention
    and think in steps-per-span (StPS = ns / Nsp):

      - StPS >= 1 (ns is a multiple of Nsp): keep the Nsp REAL spans with StPS=ns/Nsp
        integer steps each. This preserves the intra-span exp(alpha*z) power profile
        (each span has one real EDFA), so nl_param carries Stella's xi profile. For
        single-span scenarios (Nsp=1) this is just StPS=ns -- 01/02 unchanged.

      - StPS < 1 (ns < Nsp, "less steps than spans") OR ns not a multiple of Nsp:
        cannot place an integer number of steps per real span, so fall back to the
        'one long effective span' mapping (Nsp_eff=ns, Lsp_eff=Ltot/ns, StPS=1),
        keeping alpha (attenuation in the power profile, NOT zeroed). model_steps=ns+1.

    total_steps overrides the auto choice (True -> long-span mapping; False -> per-span
    with StPS=ns treated as steps PER SPAN, the raw SSFMParameters convention). Default
    None -> auto per the StPS rule above.

    PHYSICS: the long-span mapping spreads the Kerr UNIFORMLY across steps (fictitious
    amplifiers every Ltot/ns km), which is correct only when ns<=Nsp. For ns>Nsp it
    breaks the intra-span profile and costs ESSFM ~0.2 dB at high Ns -- hence the
    per-span branch. See memory essfm-highns-plateau-bug.
    """
    Nsp = S['Nsp']
    if total_steps is None:
        # per-span mode when an integer number of steps fits each real span; otherwise
        # the long-span 'less steps than spans' mapping.
        use_per_span = (ns >= Nsp) and (ns % Nsp == 0)
    elif total_steps is False:
        use_per_span = True      # ns interpreted as steps PER SPAN (raw convention)
    else:
        use_per_span = False     # forced long-span mapping
    common = {'beta2': S['beta2'], 'gamma': S['gamma'], 'fsamp': S['fsamp_d'],
        'Nsamp': S['Nsamp_d'], 'step_size_method': S['step_method_bw'],
        'ssfm_method': S['ssfm_method_bw'], 'combine_half_steps': S['combine'],
        'direction': -1, 'alpha': S['alpha']}
    if use_per_span:
        # StPS per real span. total_steps=False keeps the legacy meaning (ns=StPS);
        # auto/None passes ns total -> StPS=ns//Nsp.
        StPS = ns if total_steps is False else ns // Nsp
        return SSFMParameters({**common, 'Nsp': Nsp, 'Lsp': S['Lsp'], 'StPS': StPS})
    # long-span mapping (ns < Nsp or non-divisible): Nsp_eff=ns spans of Ltot/ns, StPS=1
    return SSFMParameters({**common, 'Nsp': ns, 'Lsp': S['Lsp'] * Nsp / ns, 'StPS': 1})


# Optional GPU backend (CuPy) for the heavy backprop loop, like the original
# l-essfm_test.ipynb. Falls back to numpy if CuPy is unavailable. The signal lives
# on the GPU through the O(model_steps) CD+NLPR loop; only the light matched-filter
# tail runs on the CPU (exactly the original's cp.asnumpy() handoff).
try:
    import cupy as _cp
    _HAVE_CUPY = True
except Exception:
    _cp = np
    _HAVE_CUPY = False


def backprop_many(S, bw, cd_mult, nlf, ys, xs, Ps, edc=False, chunk=8, single=False):
    """backprop() for many signals at once on the GPU (CuPy), effSNR [dB] per signal.

    Same operations and precision as backprop(); the (power, realization) signals of
    a curve are stacked, and the per-step GVD and NLPR factors are built on the GPU
    once per step for the whole stack instead of on the CPU once per step per signal.
    single=False (default): same precision as backprop() (values to ~1e-14 dB).
    single=True: step loop in complex64 (GVD phase reduced modulo 2*pi in double first),
    tail in double; changes the effSNR by < 1e-4 dB but was not faster here: these
    curves are bound by FFT work on large signals, not by precision (2026-10-07)."""
    cp = _cp
    Nsamp_d, OS_a, OS_d, Nsym, Npol = S['Nsamp_d'], S['OS_a'], S['OS_d'], S['Nsym'], S['Npol']
    cd_alpha, nl_alpha = S['cd_alpha'], S['nl_alpha']
    M = bw.model_steps
    # get_cd_filter_freq(NN) = (beta2/2)*(2*pi*f)**2 * (direction*cd_length[NN]), same order
    base = cp.asarray((bw.beta2 / 2) * (2 * np.pi * bw.fvec) ** 2)
    ps_rx = cp.asarray(S['ps_rx_freq'])
    out = []
    for a in range(0, len(ys), chunk):
        y = cp.asarray(np.stack(ys[a:a + chunk]))
        Y = cp.fft.fft(y); Y = Y[..., :Nsamp_d] + Y[..., -Nsamp_d:]
        y = cp.fft.ifft(Y) / OS_a * OS_d
        del Y
        for NN in range(M):
            cdf = base * (bw.direction * bw.cd_length[NN]) * cd_mult[NN] / cd_alpha
            if single:
                E = cp.exp(1j * cp.fmod(cdf, 2 * np.pi).astype(cp.float32))
            else:
                E = cp.exp(1j * cdf)
            y = cp.fft.ifft(cp.fft.fft(y) * E)
            if (NN < M - 1) and not edc:
                ysq = cp.abs(y[:, 0, :]) ** 2 + cp.abs(y[:, 1, :]) ** 2
                # NLPR filter built on the GPU from the few taps (was built on the CPU
                # and uploaded at full length, once per step per signal)
                nfl = len(nlf[NN])
                taps = cp.asarray(-np.asarray(nlf[NN], dtype=np.float64) / nl_alpha)
                nl_time = cp.zeros(Nsamp_d, dtype=cp.float64)
                nl_time[:nfl - 1] = taps[1:][::-1]
                nl_time[nfl - 1:2 * nfl - 1] = taps
                nl_time = cp.roll(nl_time, -nfl + 1).astype(cp.float32 if single else cp.float64)
                ysqf = cp.fft.irfft(cp.fft.rfft(ysq) * cp.fft.rfft(nl_time))
                y = y * cp.exp(1j * ysqf)[:, None, :]
        Y = cp.fft.fft(y.astype(cp.complex128)) * ps_rx
        Yal = Y[..., :Nsym].copy()
        Yal[..., Nsym - Nsamp_d // 2:Nsym // 2] += Y[..., Nsamp_d // 2:Nsamp_d - Nsym // 2]
        Yal[..., Nsym // 2:Nsamp_d // 2] += Y[..., Nsamp_d - Nsym // 2:3 * Nsamp_d // 2 - Nsym]
        Yal[..., Nsamp_d // 2:Nsym] = Y[..., 3 * Nsamp_d // 2 - Nsym:]
        P = cp.asarray(np.asarray(Ps[a:a + chunk], dtype=float))[:, None, None]
        yo = cp.fft.ifft(Yal) / OS_d / cp.sqrt(P / Npol) / np.sqrt(OS_d)
        x = cp.asarray(np.stack(xs[a:a + chunk]))
        phi = cp.angle((cp.conj(x) * yo).sum(-1, keepdims=True))
        xh = (yo * cp.exp(-1j * phi)).astype(np.complex64)
        out.append(cp.asnumpy(-10.0 * cp.log10(cp.mean(cp.abs(x - xh) ** 2, axis=(1, 2)))))
    return np.concatenate(out)


def backprop(S, bw, cd_mult, nlf, y, x, P, edc=False, gpu=None):
    """Apply DBP and return effSNR [dB]. cd_mult/nlf: per-step dicts.

    gpu: True -> run the CD+NLPR loop on the GPU via CuPy (fast for the large
    262144-symbol signals); None -> auto (GPU if CuPy present); False -> numpy.
    """
    use_gpu = (_HAVE_CUPY if gpu is None else gpu) and _HAVE_CUPY
    xp = _cp if use_gpu else np
    Nsamp_d, OS_a, OS_d, Nsym, Npol = S['Nsamp_d'], S['OS_a'], S['OS_d'], S['Nsym'], S['Npol']
    cd_alpha, nl_alpha = S['cd_alpha'], S['nl_alpha']
    y = xp.asarray(y)
    Y = xp.fft.fft(y); Y = Y[:, :Nsamp_d] + Y[:, -Nsamp_d:]; y = xp.fft.ifft(Y) / OS_a * OS_d
    for NN in range(bw.model_steps):
        cdf = xp.asarray(bw.get_cd_filter_freq(NN) * cd_mult[NN] / cd_alpha)
        y = xp.fft.ifft(xp.fft.fft(y) * xp.exp(1j * cdf))
        if (NN < bw.model_steps - 1) and not edc:
            ysq = xp.abs(y[0, :])**2 + xp.abs(y[1, :])**2
            nfl = len(nlf[NN])
            nl_time = -np.asarray(nlf[NN]) / nl_alpha
            nl_time = np.concatenate([np.flip(nl_time[1:]), nl_time, np.zeros(Nsamp_d - 2 * nfl + 1)])
            nl_time = np.roll(nl_time, -nfl + 1)
            ysqf = xp.fft.irfft(xp.fft.rfft(ysq) * xp.fft.rfft(xp.asarray(nl_time)))
            y = y * xp.exp(1j * ysqf)
    # matched-filter tail + CPE on the CPU (light), as in the original
    y = _cp.asnumpy(y) if use_gpu else y
    Y = np.fft.fft(y) * S['ps_rx_freq']
    Yal = Y[:, :Nsym].copy()
    Yal[:, Nsym - Nsamp_d // 2:Nsym // 2] += Y[:, Nsamp_d // 2:Nsamp_d - Nsym // 2]
    Yal[:, Nsym // 2:Nsamp_d // 2] += Y[:, Nsamp_d - Nsym // 2:3 * Nsamp_d // 2 - Nsym]
    Yal[:, Nsamp_d // 2:Nsym] = Y[:, 3 * Nsamp_d // 2 - Nsym:]
    y = np.fft.ifft(Yal) / OS_d / np.sqrt(P / Npol) / np.sqrt(OS_d)
    x_hat = np.zeros([Npol, Nsym], dtype=np.complex64)
    for pp in range(Npol):
        phi = np.angle(np.dot(np.conj(x[pp, :]), y[pp, :]))
        x_hat[pp, :] = y[pp, :] * np.exp(-1j * phi)
    return -10.0 * np.log10(np.mean(np.abs(x - x_hat)**2))
