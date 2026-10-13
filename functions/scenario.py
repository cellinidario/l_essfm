"""scenario.py -- one link + signal setup, as the scenario notebooks use it.

    scenario = Scenario('scenario_170km_93GBd', retrain=False)
    data, Pgrid = scenario.load_forward()            # cached forward (simulated if missing)
    scenario.peak(n, 'edc' | 'ideal' | 'ossfm', ns)  # Ideal DBP and EDC: steps per span
    scenario.model_peak(n, 'lessfm' | 'essfm_optrho' | 'essfm', ns)   # trained if missing
    scenario.ldbp_point(tag, target_taps, steps_per_span, init_taps, power_dbm, iters)

n is the digital oversampling (1.125 or 2). Trained methods, as named on disk:
lessfm, essfm_optrho (ESSFM with optimized splitting ratio) and essfm (rho = 0.5).

Everything is saved in results/independent_ase_v1/<scenario>/:
    lessfm_n2_Ns10/parameters.csv            trained coefficients
                  /parameters.manifest.json  what they were trained on
    evaluations/lessfm_n2_Ns10.json          SNR vs power, and what it was computed on
A model is trained only if it is missing or retrain=True. A saved model trained on
a different setup is refused, never silently reused. A saved SNR curve is
recomputed when the forward, the receiver or the coefficients change.
"""
import configparser
import json
from pathlib import Path
import numpy as np

from system import build_system, make_bw
from backprop import curve
from forward import forward_identity, load_or_generate, FORWARD_VERSION
from storage import digest, write_json, exclusive

ROOT = Path(__file__).resolve().parents[1]
RESULT_SET = 'independent_ase_v1'
# Part of every model manifest: changing it marks all models as trained differently.
TRAINING_VERSION = 'seeded-low-lr-quantile-v2'
# Forward launch powers [dBm], 0.1 dB apart, around each scenario's optimum.
POWER_GRIDS = {
    'scenario_170km_93GBd': (8.5, 10.5),
    'scenario_170km_186GBd': (11.5, 13.5),
    'scenario_15x80km_93GBd': (1.5, 4.5),
    'scenario_170km_10GBd': (0., 2.),
    'scenario_15x80km_10GBd': (-6., -4.),
    'scenario_15x80km_32GBd': (-2., 0.),
}
TRAINED_METHODS = ('lessfm', 'essfm', 'essfm_optrho')
# LDBP split step. 'asymmetric' (Hager's 25 x 80 km case): N_st filters for N_st Kerr
# steps, models in ldbp_asym_<tag>/. 'symmetric': one more filter, a final linear
# half-step without Kerr; models in ldbp_<tag>/ (and results/<scenario>/ldbp_<tag>/).
LDBP_SPLIT_STEP = 'asymmetric'


def _suffix(nos):
    return '' if nos == 1.125 else '_n2'


class Scenario:

    def __init__(self, name, retrain=False, nfl_mode='formula', training_power_dbm=None):
        """name: e.g. 'scenario_170km_93GBd' (config/<name>.ini and <name>_n2.ini).
        nfl_mode: L-ESSFM NLPR length, 'formula' (shrinks with Ns, as ESSFM) or
        'ns1' (the Ns = 1 length for every Ns).
        training_power_dbm: training power, if not the config's 'training power [dBm]'."""
        self.name = name
        self.retrain = retrain
        self.nfl_mode = nfl_mode
        self.training_power_dbm = training_power_dbm
        self.configs = {nos: ROOT / 'config' / f'{name}{_suffix(nos)}.ini' for nos in (1.125, 2.)}
        self.systems = {nos: build_system(str(p)) for nos, p in self.configs.items()}
        self.resdir = ROOT / 'results' / RESULT_SET / name
        self.resdir.mkdir(parents=True, exist_ok=True)
        self.forward_dir = ROOT / 'forward' / RESULT_SET / name
        lo, hi = POWER_GRIDS[name]
        self.powers = np.round(np.arange(lo, hi, .1), 4)
        self.data = None

    def load_forward(self):
        """(data, powers): the cached forward, generated where missing."""
        self.data = load_or_generate(self.systems[1.125], self.powers, self.forward_dir,
                                     nreal=1, seed=0)
        print(f'{self.name}: {len(self.powers)} powers; independent ASE; '
              f'forward cache {self.forward_dir}', flush=True)
        return self.data, self.powers

    # --------------------------------------------------------- trained models
    def training_config(self, nos, ldbp=False):
        """The scenario config with the training settings (Gaussian symbols, short
        blocks, fewer forward steps); saved in training_configs/ for the record."""
        p = configparser.ConfigParser()
        p.read(self.configs[nos])
        p['system parameters']['modulation'] = 'Gaussian'
        symbols = 1024
        if ldbp:
            # the FIR training blocks must cover the link's dispersive memory
            symbols = 8192 if self.systems[nos]['Nsp'] > 1 else 1024
            p['L-ESSFM parameters']['step size method'] = 'logarithmic'
            p['L-ESSFM parameters']['split step method'] = LDBP_SPLIT_STEP
        p['system parameters']['data symbols per block'] = str(symbols)
        if self.systems[nos]['Nsp'] > 1:
            fast = self.systems[nos]['fsym'] >= 93e9
            p['data generation']['forward steps per span'] = '100' if fast else '50'
        if self.training_power_dbm is not None:
            p['data generation']['training power [dBm]'] = str(self.training_power_dbm)
        tag = ('_ldbp_asym' if LDBP_SPLIT_STEP == 'asymmetric' else '_ldbp') if ldbp else ''
        path = self.resdir / 'training_configs' / f'n{nos:g}{tag}.ini'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            p.write(f)
        return str(path)

    def model_path(self, nos, tag, ns):
        return self.resdir / f'{tag}{_suffix(nos)}_Ns{ns}' / 'parameters.csv'

    def model(self, nos, tag, ns, iters=12000, seed=0):
        """Path of the trained coefficients; trains them if missing or retrain=True."""
        if tag not in TRAINED_METHODS:
            raise ValueError(f'{tag}: expected one of {TRAINED_METHODS}')
        path = self.model_path(nos, tag, ns)
        with exclusive(self.resdir / '.locks' / f'{path.parent.name}.lock'):
            return self._model(nos, tag, ns, path, iters, seed)

    def _model(self, nos, tag, ns, path, iters, seed):
        import train
        S = build_system(self.training_config(nos))
        manifest = path.with_suffix('.manifest.json')
        identity = dict(schema=1, scenario=self.name, method=tag, ns=int(ns),
                        oversampling=nos, forward_identity=forward_identity(S),
                        forward_version=FORWARD_VERSION, ase_mode='independent',
                        nfl_mode=self.nfl_mode, seed=seed, training_version=TRAINING_VERSION,
                        training_power_dbm=S['train_power_dbm'])
        if path.exists() and not self.retrain:
            if not manifest.exists():
                raise ValueError(f'{path} has no manifest: it was not trained by this code. '
                                 'Move it away, or set RETRAIN = True.')
            saved = json.loads(manifest.read_text())
            if saved.get('identity') != identity or saved.get('sha256') != digest(path):
                raise ValueError(f'{path} was trained on a different setup (see its manifest). '
                                 'Set RETRAIN = True to retrain it.')
            return str(path)
        print(f'Training {tag}, n={nos:g}, Ns={ns}', flush=True)
        if tag == 'lessfm':
            nfl = train.nlpr_length(S, 1) if self.nfl_mode == 'ns1' else None
            train.train_lessfm(S, ns, str(path), iters=iters, seed=seed, nfl=nfl)
        else:
            train.train_essfm(S, ns, str(path), iters=iters, seed=seed,
                              opt_rho=(tag == 'essfm_optrho'))
        write_json(manifest, dict(identity=identity, sha256=digest(path)))
        return str(path)

    def diagnostics(self, nos, ns, tag='lessfm'):
        """Geometry report of a trained L-ESSFM, saved next to it (.geometry.json).
        A geometric flag is not a proof of a dead nonlinear stage."""
        from geometry import convergence_report
        S = self.systems[nos]
        path = self.model_path(nos, tag, ns)
        report = convergence_report(S, make_bw(S, ns), ns, str(path))
        plain = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in report.items()}
        plain.update(parameter_sha256=digest(path), reference_used_in_loss=False)
        write_json(path.with_suffix('.geometry.json'), plain)
        return plain

    # ------------------------------------------------------------ evaluation
    def peak(self, nos, kind, ns, params=None, edc=False):
        """Peak effective SNR. kind: 'edc', 'ideal' (ns = steps PER SPAN), 'ossfm', or
        'lessfm' with params = a parameters.csv (L-ESSFM and ESSFM alike)."""
        total_steps = False if kind in ('ideal', 'edc') else None
        return float(self.snr_curve(nos, kind, ns, params, edc, total_steps).max())

    def model_peak(self, nos, tag, ns):
        return self.peak(nos, 'lessfm', ns, self.model(nos, tag, ns))

    def snr_curve(self, nos, kind, ns, params=None, edc=False, total_steps=None):
        """SNR vs launch power, saved as evaluations/<name>.json: edc, ideal_n2_50_per_span,
        ossfm_n2_Ns5, or the model folder name (e.g. lessfm_n2_Ns10)."""
        S = self.systems[nos]
        sfx = _suffix(nos)
        name = {'edc': f'edc{sfx}', 'ideal': f'ideal{sfx}_{ns}_per_span',
                'ossfm': f'ossfm{sfx}_Ns{ns}'}.get(kind) or Path(params).parent.name
        identity = dict(forward_manifest=digest(self.forward_dir / 'manifest.json'),
                        receiver=forward_identity(S), kind=kind, ns=int(ns),
                        backward={k: S[k] for k in ('step_method_bw', 'ssfm_method_bw', 'combine',
                                                    'cd_alpha', 'nl_alpha', 'nl_filter_length')},
                        params_sha256=digest(params) if params else None,
                        edc=bool(edc), total_steps=total_steps,
                        powers_dbm=self.powers.tolist())
        record = self._saved(name, identity)
        if record:
            return np.array(record['snr_db'])
        self._need_forward()
        snr = curve(dict(S), self.data, self.powers, kind, ns, params,
                    edc=edc, total_steps=total_steps)
        if not np.isfinite(snr).all():
            raise FloatingPointError(f'{name}: non-finite SNR, not saved')
        write_json(self._evaluation_path(name), dict(
            identity=identity, snr_db=snr.tolist(), effective_snr_db=float(snr.max()),
            peak_power_dbm=float(self.powers[snr.argmax()])))
        return snr

    # ------------------------------------------------------------------- LDBP
    def ldbp_point(self, tag, targets, steps_per_span, init_taps, power_dbm, iters=15000, lr=None):
        """(complexity, effSNR) of one LDBP at n = 2 (logarithmic steps, fixed Kerr,
        split step LDBP_SPLIT_STEP).

        Training starts from init_taps per step and prunes gradually to targets
        (e.g. [11], or [11, 9] repeated along the link), at power_dbm. A finite symmetric
        model of the old result set in results/<scenario>/ is re-evaluated, not retrained.
        The complexity counts N_st filters, one per Kerr step, with the taps actually
        saved; the extra half-step filter of the symmetric SSM is not counted (Dario,
        2026-10-08). A diverged model gives nan.
        lr: Adam learning rate; None keeps 5e-4 (1e-3 at 20000 iterations)."""
        from ldbp_eval import load_ldbp_params, ldbp_backprop
        from complexity import complexity_ldbp
        import train
        asymmetric = LDBP_SPLIT_STEP == 'asymmetric'
        S = dict(self.systems[2.], step_method_bw='logarithmic', ssfm_method_bw=LDBP_SPLIT_STEP)
        bw = make_bw(S, steps_per_span, total_steps=False)
        name = f'ldbp_asym_{tag}' if asymmetric else f'ldbp_{tag}'
        path = self.resdir / name / 'parameters.csv'
        legacy = ROOT / 'results' / self.name / f'ldbp_{tag}' / 'parameters.csv'
        if not asymmetric and not path.exists() and legacy.exists() and not self.retrain:
            cd_old, nl_old = load_ldbp_params(str(legacy), bw.model_steps)
            if all(np.isfinite(cd_old[i]).all() and np.isfinite(nl_old[i]) for i in cd_old):
                path = legacy
            else:
                print(f'LDBP {tag}: the old model is not finite; training a new one', flush=True)
        if self.retrain or not path.exists():
            with exclusive(self.resdir / '.locks' / f'{name}.lock'):
                St = build_system(self.training_config(2., ldbp=True))
                M = make_bw(St, steps_per_span, total_steps=False).model_steps
                train.train_ldbp(St, steps_per_span, str(path), [init_taps] * M, iters=iters,
                                 P_dbm=power_dbm, lr=lr or (1e-3 if iters == 20000 else 5e-4),
                                 batch=8 if St['Nsp'] > 1 else 100, n_blocks=200,
                                 target_taps=targets, total_steps=False)
        cd, nl = load_ldbp_params(str(path), bw.model_steps)
        taps = [len(cd[i]) for i in range(bw.model_steps)]
        steps = bw.model_steps if asymmetric else bw.model_steps - 1   # Kerr steps, N_st
        cost = float(complexity_ldbp(steps, (float(np.mean(taps)) - 1) / 2, 2))
        identity = dict(forward_manifest=digest(self.forward_dir / 'manifest.json'),
                        params_sha256=digest(path), kind='ldbp',
                        config_sha256=digest(self.configs[2.]),
                        backward_steps_per_span=steps_per_span, step_method='logarithmic',
                        powers_dbm=self.powers.tolist())
        if asymmetric:   # symmetric records keep their original identity
            identity['split_step'] = 'asymmetric'
        record = self._saved(name, identity)
        if record:
            values = [np.nan if v is None else v for v in record['snr_db']]
        else:
            self._need_forward()
            values = [float(np.mean([ldbp_backprop(S, bw, cd, nl, y, x, 1e-3 * 10**(p / 10))
                                     for y, x in self.data[p]])) for p in self.powers]
            # a non-finite SNR (diverged model) is stored as null and reported, not hidden
            write_json(self._evaluation_path(name), dict(
                identity=identity, snr_db=[v if np.isfinite(v) else None for v in values],
                taps=taps, complexity_rm2d=cost, parameter_path=str(path)))
        best = float(np.nanmax(values)) if np.isfinite(values).any() else float('nan')
        print(f'LDBP {tag} ({LDBP_SPLIT_STEP}): Ns={steps}, saved mean taps={np.mean(taps):.3f}, '
              f'effSNR={best:.4f} dB' + ('' if np.isfinite(best) else ' (FAILED: non-finite output)'),
              flush=True)
        return cost, best

    # ---------------------------------------------------------------- helpers
    def _evaluation_path(self, name):
        return self.resdir / 'evaluations' / f'{name}.json'

    def _saved(self, name, identity):
        """The saved record if it was computed on exactly this identity, else None."""
        path = self._evaluation_path(name)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        return record if record['identity'] == identity else None

    def _need_forward(self):
        if self.data is None:
            raise RuntimeError('Call load_forward() first: this curve is not saved yet.')
