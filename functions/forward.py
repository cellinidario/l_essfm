"""forward.py -- the received signal of a scenario, simulated once and cached.

All methods are evaluated on the same cached signal, so method-to-method
differences are not blurred by different noise realizations. The forward (the
full WDM band over thousands of SSFM steps) is the slow part: it runs only when
the cache is missing.

Cache folder, forward/independent_ase_v1/<scenario>/:
    manifest.json        what was simulated: physics hash, powers, seed
    p000_r000.npz ...    one file per (power, realization), written atomically,
                         so an interrupted generation resumes where it stopped
A cache made with different physics, powers or seed is refused, never reused.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np

from system import ase_polarizations, ase_injection_location
from storage import exclusive

ROOT = Path(__file__).resolve().parents[1]
# Part of every model and evaluation hash: changing it invalidates all of them.
FORWARD_VERSION = 'independent-ase-v1'


def _plain(value):
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        return {'shape': a.shape, 'dtype': str(a.dtype),
                'sha256': hashlib.sha256(a.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def forward_identity(S):
    """Hash of every physical input of the forward; backward/training settings excluded."""
    keys = ('Lsp', 'Nsp', 'alpha', 'gamma', 'beta2', 'noise_figure', 'sigma2',
            'Npol', 'Nch', 'Nsym', 'Nsamp_a', 'Nsamp_d', 'OS_a', 'OS_d',
            'fsym', 'fsamp_a', 'fsamp_d', 'spacing', 'modulation', 'const',
            'rolloff', 'lp_bw', 'ps_tx_freq', 'ps_rx_freq', 'lp_freq')
    values = {k: S[k] for k in keys if k in S}
    values.update(version=FORWARD_VERSION, ase_mode=S.get('ase_mode', 'independent'),
                  fw=vars(S['fw']))
    ase_polarizations(S)  # validate before either generating or reusing data
    location = ase_injection_location(S)
    # ASE at the span input keeps the hashes of the existing caches unchanged;
    # ASE at the span output gets its own hashes, so it can never reuse them.
    if location != 'span_input':
        values['ase_injection_location'] = location
    raw = json.dumps(_plain(values), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def load_or_generate(S, powers_dbm, directory, nreal=1, seed=0):
    """{P_dBm: [(y_analog, x), ...]}: loaded from `directory`, generated where missing."""
    root = Path(directory)
    with exclusive(root.parent / (root.name + '.lock')):
        return _load_or_generate(S, powers_dbm, root, nreal, seed)


def _load_or_generate(S, powers_dbm, root, nreal, seed):
    powers = [float(p) for p in powers_dbm]
    if not powers or not np.isfinite(powers).all() or len(set(powers)) != len(powers):
        raise ValueError('Power grid must contain distinct finite values')
    if nreal < 1:
        raise ValueError('nreal must be positive')
    root.mkdir(parents=True, exist_ok=True)
    expected = dict(schema=1, forward_version=FORWARD_VERSION,
                    forward_identity=forward_identity(S), ase_mode=S.get('ase_mode', 'independent'),
                    powers_dbm=powers, nreal=int(nreal), seed=int(seed), analog_output=True)
    meta = root / 'manifest.json'
    if meta.exists():
        if json.loads(meta.read_text()) != expected:
            raise ValueError(f'{root} holds a forward with different physics, powers or seed; '
                             'use a new folder')
    else:
        if any(root.iterdir()):
            raise ValueError(f'{root} is not empty but has no manifest.json')
        with tempfile.NamedTemporaryFile(mode='w', dir=root, delete=False) as f:
            json.dump(expected, f, indent=2)
        os.replace(f.name, meta)
    from forward_sim import forward_propagation
    data = {}
    for ip, p in enumerate(powers):
        rows = []
        for real in range(nreal):
            path = root / f'p{ip:03d}_r{real:03d}.npz'
            if not path.exists():
                print(f'Generating {p:g} dBm, realization {real+1}/{nreal}', flush=True)
                y, x = forward_propagation(S, 1e-3 * 10**(p / 10), seed=seed + ip * nreal + real)
                with tempfile.NamedTemporaryFile(dir=root, delete=False) as f:
                    np.savez(f, y=y, x=x, power_dbm=p, realization=real)
                os.replace(f.name, path)
            with np.load(path, allow_pickle=False) as d:
                y, x = d['y'], d['x']
                if float(d['power_dbm']) != p or int(d['realization']) != real:
                    raise ValueError(f'Power/realization mismatch: {path}')
                if y.shape != (S['Npol'], S['Nsamp_a']) or x.shape != (S['Npol'], S['Nsym']):
                    raise ValueError(f'Forward cache shape mismatch: {path}')
                if not np.isfinite(y).all() or not np.isfinite(x).all():
                    raise ValueError(f'Non-finite forward cache: {path}')
                rows.append((y, x))
        data[p] = rows
    return data


def load_legacy(scenario):
    """Forward of the old result set (ASE identical on the two polarizations):
    forward/forward_<scenario>.npy, as {P_dBm: [(y, x), ...]}."""
    fwd = np.load(ROOT / 'forward' / f'forward_{scenario}.npy', allow_pickle=True).item()
    return {round(10 * np.log10(Pw / 1e-3), 4): r for Pw, r in fwd.items()}
