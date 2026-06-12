"""
backprop.py -- Apply DBP methods to a cached forward and build SNR curves.

A "method" is a string:
  edc                 electronic dispersion compensation (linear only)
  ideal:<Nsteps>      ideal DBP = SSFM with many steps (e.g. ideal:800)
  ossfm:<Ns>          SSFM with one optimized nonlinear scale (grid-searched)
  lessfm:<params>:<Ns>  trained L-ESSFM / ESSFM (params.csv from lessfm.py)

For lessfm, the saved params.csv already folds the per-step nl_param scaling into
each step's NLPR filter (done by lessfm.py at save time), so loading is uniform
for both L-ESSFM (per-step filters) and ESSFM (tied filter, pre-scaled).

CLI:
  python3 backprop.py --forward ../results/<sc>/forward.npz \
      --methods edc ideal:800 lessfm:../results/<sc>/lessfm_Ns10/parameters.csv:10 \
      --out ../results/<sc>/curve.npz
"""
import os, argparse, glob, configparser
import numpy as np

from system import build_system, make_bw, backprop
from forward import load


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
        expected = bw.model_steps + (bw.model_steps - 1) * nfl_model
        if len(lines) != expected:
            nfl_model = (len(lines) - bw.model_steps) // (bw.model_steps - 1)
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
    bw = make_bw(S, ns, total_steps=total_steps)
    if kind == 'ossfm':
        # Optimize the single nonlinear scale. The optimum is power-insensitive, so
        # search at ONE central power (1 realization) -> ~11 backprops instead of
        # ~70. Coarse 0.4..1.4 step 0.1, then a fine pass ±0.1 step 0.025.
        Pmid = sorted(data)[len(data) // 2]
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
    for ip, Pdb in enumerate(sorted(data)):
        P = 10**(Pdb / 10) * 1e-3
        snr[ip] = np.mean([backprop(S, bw, cd_mult, nlf, y, x, P, edc=edc)
                           for (y, x) in data[Pdb]])
    return snr


def run_methods(S, data, Pgrid, methods):
    """methods: list of strings; returns {label: snr_array}."""
    out = {}
    for m in methods:
        parts = m.split(':'); kind = parts[0]
        if kind == 'edc':
            ns, params, edc = 1, None, True
        elif kind in ('ideal', 'ossfm'):
            ns, params, edc = int(parts[1]), None, False
        else:
            params, ns, edc = parts[1], int(parts[2]), False
        snr = curve(S, data, Pgrid, kind, ns, params, edc)
        out[m] = snr
        print(f"{kind:8s} Ns={ns}: peak {snr.max():.4f} dB @ {Pgrid[snr.argmax()]:.2f} dBm")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--forward', required=True, help='cached forward .npz')
    ap.add_argument('--config', help='config .ini (default: stored in forward)')
    ap.add_argument('--methods', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    data, Pgrid, _ = load(args.forward)
    cfg = args.config or str(np.load(args.forward, allow_pickle=True)['config'])
    S = build_system(cfg)
    res = run_methods(S, data, Pgrid, args.methods)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, Pgrid=Pgrid, **{m.replace('/', '_'): v for m, v in res.items()})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
