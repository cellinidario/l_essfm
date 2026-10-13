"""
backprop.py -- Apply DBP methods to a cached forward and build SNR curves.

A "method" is a string:
  edc                 electronic dispersion compensation (linear only)
  ideal:<steps>       ideal DBP = SSFM with many steps per span (e.g. ideal:100)
  ossfm:<Ns>          SSFM with one optimized nonlinear scale (grid-searched)
  lessfm:<params>:<Ns>  trained L-ESSFM / ESSFM (parameters.csv)

For lessfm, the saved parameters.csv already folds the per-step nl_param scaling
into each step's NLPR filter, so loading is uniform for both L-ESSFM (per-step
filters) and ESSFM (tied filter, pre-scaled).
"""
import os, glob, configparser
import numpy as np

from system import build_system, make_bw, backprop, backprop_many, _HAVE_CUPY


def load_params(S, bw, kind, params_path):
    """Return (cd_mult, nlf) per-step dicts for a method."""
    cd_mult, nlf = {}, {}
    nfl = S['nl_filter_length']
    if kind in ('edc', 'ideal', 'ossfm'):
        scale = S.get('_ossfm_scale', 1.0) if kind == 'ossfm' else 1.0
        for NN in range(bw.model_steps):
            cd_mult[NN] = 1.0
            if NN < bw.model_steps - 1:
                d = np.zeros(nfl, dtype=np.float32)
                d[0] = -bw.nl_param[NN] * S['nl_alpha'] * scale
                nlf[NN] = d
    else:  # lessfm / essfm: load trained params.csv
        nfl_model = nfl
        d = os.path.dirname(params_path)
        inis = glob.glob(os.path.join(d, '*.ini'))
        if inis:
            c = configparser.ConfigParser(); c.read(inis[0])
            try:
                nfl_model = c['L-ESSFM parameters'].getint('nl filter length')
            except Exception:
                pass
        lines = open(params_path).read().strip().split('\n')
        if bw.model_steps <= 1 or (len(lines)-bw.model_steps) % (bw.model_steps-1):
            raise ValueError(f'Parameter count does not match Ns: {params_path}')
        expected = bw.model_steps + (bw.model_steps - 1) * nfl_model
        if len(lines) != expected:
            nfl_model = (len(lines) - bw.model_steps) // (bw.model_steps - 1)
        if nfl_model < 1 or not np.isfinite(np.asarray(lines,dtype=float)).all():
            raise ValueError(f'Invalid/non-finite parameters: {params_path}')
        idx = 0
        for NN in range(bw.model_steps):
            cd_mult[NN] = float(lines[idx]); idx += 1
            if NN == 0 or NN == bw.model_steps - 1:
                cd_mult[NN] *= 2
            if NN < bw.model_steps - 1:
                nlf[NN] = np.array([float(lines[idx + k]) for k in range(nfl_model)]); idx += nfl_model
    return cd_mult, nlf


def curve(S, data, Pgrid, kind, ns, params=None, edc=False, total_steps=None):
    """effSNR(P) averaged over realizations for one method.

    total_steps: passed to make_bw. None=auto (total-steps for multi-span). For the
    IDEAL DBP / EDC reference on a multi-span link, pass total_steps=False so `ns`
    means steps PER SPAN (Ideal needs an accurate per-span backprop, e.g. 50/span =
    750 steps over 15 spans). Trained methods (L-ESSFM/ESSFM) keep total-steps (the
    Fig.1b complexity convention: ns = TOTAL DBP steps)."""
    Pgrid = np.asarray(Pgrid, dtype=float)
    if Pgrid.ndim != 1 or len(set(Pgrid)) != len(Pgrid):
        raise ValueError('Pgrid must be a one-dimensional grid of distinct powers')
    missing = [float(p) for p in Pgrid if p not in data]
    if missing:
        raise ValueError(f'Forward data missing requested powers: {missing}')
    bw = make_bw(S, ns, total_steps=total_steps)
    if kind == 'ossfm':
        # Optimize the single nonlinear scale. The optimum is power-insensitive, so
        # search at ONE central power (1 realization) -> ~11 backprops instead of
        # ~70. Coarse 0.4..1.4 step 0.1, then a fine pass ±0.1 step 0.025.
        Pmid = sorted(Pgrid)[len(Pgrid) // 2]
        ymid, xmid = data[Pmid][0]
        Pw = 10 ** (Pmid / 10) * 1e-3

        def scale_snr(sc):
            S['_ossfm_scale'] = sc
            cd_mult, nlf = load_params(S, bw, 'ossfm', None)
            return backprop(S, bw, cd_mult, nlf, ymid, xmid, Pw)
        coarse = np.arange(0.0, 1.41, 0.1)
        best_scale = max(coarse, key=scale_snr)
        fine = np.arange(best_scale - 0.1, best_scale + 0.101, 0.025)
        best_scale = max(fine, key=scale_snr)
        S['_ossfm_scale'] = best_scale
    cd_mult, nlf = load_params(S, bw, kind, params)
    snr = np.zeros(len(Pgrid))
    if _HAVE_CUPY:      # all (power, realization) signals together on the GPU, same values
        owner = [ip for ip, Pdb in enumerate(Pgrid) for _ in data[Pdb]]
        sig = [(y, x, 10**(Pdb / 10) * 1e-3) for Pdb in Pgrid for (y, x) in data[Pdb]]
        vals = backprop_many(S, bw, cd_mult, nlf, [s[0] for s in sig], [s[1] for s in sig],
                             [s[2] for s in sig], edc=edc)
        for ip in range(len(Pgrid)):
            snr[ip] = np.mean([v for o, v in zip(owner, vals) if o == ip])
        return snr
    for ip, Pdb in enumerate(Pgrid):
        P = 10**(Pdb / 10) * 1e-3
        snr[ip] = np.mean([backprop(S, bw, cd_mult, nlf, y, x, P, edc=edc)
                           for (y, x) in data[Pdb]])
    return snr


def run_methods(S, data, Pgrid, methods):
    """methods: list of strings; returns {label: snr_array}.

    'edc' | 'ideal:<steps PER SPAN>' | 'ossfm:<total steps>' | '<kind>:<params.csv>:<total steps>'.
    Ideal DBP counts steps per span, like the notebooks' peak(): 'ideal:50' on 15 x 80 km
    runs 750 steps. The learned methods keep Ns = total steps (the complexity axis)."""
    out = {}
    for m in methods:
        parts = m.split(':'); kind = parts[0]
        total = None
        if kind == 'edc':
            ns, params, edc = 1, None, True
        elif kind == 'ideal':
            ns, params, edc, total = int(parts[1]), None, False, False
        elif kind == 'ossfm':
            ns, params, edc = int(parts[1]), None, False
        else:
            params, ns, edc = parts[1], int(parts[2]), False
        snr = curve(S, data, Pgrid, kind, ns, params, edc, total_steps=total)
        out[m] = snr
        unit = 'steps/span' if total is False else 'steps'
        print(f"{kind:8s} {ns} {unit}: peak {snr.max():.4f} dB @ {Pgrid[snr.argmax()]:.2f} dBm")
    return out

