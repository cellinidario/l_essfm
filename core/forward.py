"""
forward.py -- Generate the received signal ONCE and cache it to disk.

The forward SSFM (thousands of steps over ~1.8M samples) is the slow, "legitimate"
cost.  By caching (y, x) for every (power, realization), every backprop method is
then applied cheaply (<=50 steps) and -- crucially -- sees the EXACT SAME received
signal, so statistical fluctuations cancel in method-to-method comparisons.

For notebooks: use `load_or_generate(...)` with a saved path; it loads if present,
otherwise generates and saves (the RETRAIN-style "load or compute" pattern).

CLI:
  python3 forward.py --config CONF.ini --nreal 3 --pmin 9 --pmax 10.5 --pstep 0.3 \
      --out ../results/<scenario>/forward.npz
"""
import os, argparse
import numpy as np

from system import build_system, forward


def generate(S, Pgrid, nreal, seed=12345):
    """Return dict {Pdbm: [(y, x), ...nreal]} for the power grid."""
    np.random.seed(seed)
    data = {}
    for Pdb in Pgrid:
        P = 10**(Pdb / 10) * 1e-3
        reals = []
        for _ in range(nreal):
            y, x = forward(S, P)
            reals.append((y.astype(np.complex64), x.astype(np.complex64)))
        data[float(Pdb)] = reals
        print(f"  P={Pdb:.2f} dBm: {nreal} realizations", flush=True)
    return data


def save(path, data, Pgrid, nreal, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ys, xs, meta = [], [], []
    for ip, Pdb in enumerate(sorted(data)):
        for r, (y, x) in enumerate(data[Pdb]):
            ys.append(y); xs.append(x); meta.append((ip, r, Pdb))
    np.savez_compressed(path, ys=np.array(ys), xs=np.array(xs),
                        meta=np.array(meta, dtype=np.float32),
                        Pgrid=np.array(Pgrid), nreal=nreal, config=config)
    print(f"saved {path}  ({len(ys)} signals, {os.path.getsize(path)/1e6:.0f} MB)")


def load(path):
    """Load a saved forward into {Pdbm: [(y, x), ...]}."""
    d = np.load(path, allow_pickle=True)
    ys, xs, meta = d['ys'], d['xs'], d['meta']
    data = {}
    for (ip, r, Pdb), y, x in zip(meta, ys, xs):
        data.setdefault(float(Pdb), []).append((y, x))
    return data, d['Pgrid'], int(d['nreal'])


def load_or_generate(path, S, Pgrid, nreal, seed=12345):
    """Notebook helper: load the cached forward if present, else generate+save."""
    if os.path.exists(path):
        print(f"loading cached forward: {path}")
        data, _, _ = load(path)
        return data
    print(f"no cache at {path}; generating forward ...")
    data = generate(S, Pgrid, nreal, seed)
    save(path, data, Pgrid, nreal, '')
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--nreal', type=int, default=3)
    ap.add_argument('--pmin', type=float, required=True)
    ap.add_argument('--pmax', type=float, required=True)
    ap.add_argument('--pstep', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    S = build_system(args.config)
    Pgrid = np.arange(args.pmin, args.pmax + 1e-9, args.pstep)
    print(f"forward: {S['fw'].model_steps-1} steps/span, {S['Nsym']} symbols, "
          f"{len(Pgrid)} powers x {args.nreal} realizations")
    data = generate(S, Pgrid, args.nreal, args.seed)
    save(args.out, data, Pgrid, args.nreal, args.config)


if __name__ == "__main__":
    main()
