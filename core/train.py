"""
train.py -- Unified PyTorch trainer for L-ESSFM / ESSFM / LDBP (dual-pol).

Replaces the old TensorFlow trainers (lessfm.py, ldbp.py). Key properties, all
validated to reproduce/beat the published exercise:
  - GPU forward data-gen (forward_torch.TorchForward): the split-step forward runs
    batched on the GPU, ~30x faster than the numpy loop -> data-gen is ~free.
  - DUAL-POL objective: the NLPR/Kerr phase uses the TOTAL power over both
    polarizations (the physically-correct condition). This beats the single-pol
    training the exercise used, by +0.05..+0.14 dB across methods.
  - per-group learning rates (no nl_alpha trick), cosine LR decay, early stop on a
    held-out validation set.
  - saves parameters.csv in the eval convention -> the numpy backprop reproduces
    the model exactly.

Methods:
  lessfm : free GVD lengths (constrained sum) + per-step NLPR filters
  essfm  : uniform GVD + one rho + one tied NLPR filter (scaled per step)
  ldbp   : per-step complex symmetric CD-FIR (pruned taps) + scalar Kerr; n=2 only

CLI:
  python train.py --method lessfm --config CONF.ini --ns 10 --out OUT/parameters.csv
"""
import os, argparse, time
import numpy as np
import torch

from system import build_system, make_bw
from forward_torch import TorchForward
from lessfm_torch import LessfmModel


def save_params(path, model):
    """Write parameters.csv in the eval convention (borders halved, physical filter,
    per-step nl_param scaling FOLDED IN -- so the numpy backprop, which applies
    nlf[NN] as-is, reproduces the model exactly for both L-ESSFM (free per-step
    filters) and ESSFM (one tied filter pre-scaled per step by nl_param[NN]/nl_ref))."""
    lm = model.length_mult().detach().cpu().numpy()
    nlf = model.nlf.detach().cpu().numpy()
    M = model.M
    scaled = bool(model.tied_kerr and model.opt_rho)     # ESSFM: tied filter scaled per step
    nl_param = model.nl_param.detach().cpu().numpy() if scaled else None
    nl_ref = model.nl_ref if scaled else None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for NN in range(M):
            v = lm[NN]
            if NN == 0 or NN == M - 1:
                v = v / 2                                # halved borders
            f.write(f"{v:.18e}\n")
            if NN < M - 1:
                if model.tied_kerr:
                    row = nlf * (nl_param[NN] / nl_ref) if scaled else nlf
                else:
                    row = nlf[NN]
                for c in row:
                    f.write(f"{c:.18e}\n")


# --------------------------------------------------------------------------- #
# dual-pol data generation (GPU forward) and loss
# --------------------------------------------------------------------------- #
def gen_dualpol(tf, S, n_blocks, P_dbm, seed):
    """GPU forward -> dual-pol blocks kept paired: y[B,Npol,Nd,2], x[B,Npol,Nsym]."""
    Npol, Nsamp_d, Nsym = S['Npol'], S['Nsamp_d'], S['Nsym']
    torch.manual_seed(seed)
    P = 10 ** (P_dbm / 10) * 1e-3
    yb, xb = tf(n_blocks, P)
    return yb.reshape(n_blocks, Npol, Nsamp_d, 2), xb.reshape(n_blocks, Npol, Nsym), P


def equalize_loss(model, y, x, P, S, ldbp=False):
    """Dual-pol MSE loss. model(...) returns [B,Npol,N] complex; matched filter +
    overlap-add downsample + per-example phase align. y:[B,Npol,N,2], x:[B,Npol,Nsym]."""
    Nsamp_d, OS_d, Nsym, Npol = S['Nsamp_d'], S['OS_d'], S['Nsym'], S['Npol']
    yc = model(y) if ldbp else model(y, pol_dim=1)
    Y = torch.fft.fft(yc) * model.ps_rx
    Yal = Y[..., :Nsym].clone()
    Yal[..., Nsym - Nsamp_d // 2:Nsym // 2] += Y[..., Nsamp_d // 2:Nsamp_d - Nsym // 2]
    Yal[..., Nsym // 2:Nsamp_d // 2] += Y[..., Nsamp_d - Nsym // 2:3 * Nsamp_d // 2 - Nsym]
    Yal[..., Nsamp_d // 2:Nsym] = Y[..., 3 * Nsamp_d // 2 - Nsym:]
    yo = torch.fft.ifft(Yal) / OS_d / np.sqrt(P / Npol) / np.sqrt(OS_d)
    phi = torch.angle((x.conj() * yo).sum(-1, keepdim=True))
    xh = yo * torch.exp(-1j * phi)
    return ((x - xh).abs() ** 2).mean()


# --------------------------------------------------------------------------- #
# generic training loop
# --------------------------------------------------------------------------- #
def _train_loop(model, S, tf, param_groups, iters, batch, P_dbm, n_blocks, seed,
                ldbp=False, device='cuda', prune_events=None, patience=3000):
    """prune_events: optional list of (iteration, step_index) -- at the given iteration,
    remove ONE outermost tap from that step (model.prune_one). Mirrors ldbp_diag.py:
    gradual one-tap-at-a-time pruning so the model adapts between prunes (pruning many
    taps at once shocks the model to 0 dB).
    patience: EARLY STOP -- end training if the best val has not improved by more than
    0.005 dB in the last `patience` iterations (never while prunes are pending, so a
    post-prune recovery is not cut short). None disables. Motivation: flat tails were
    burning hours (lessfm n2 Ns=300: 16602s with best frozen since iter 1000)."""
    ys, xs, P = gen_dualpol(tf, S, n_blocks, P_dbm, seed)
    yv, xv, _ = gen_dualpol(tf, S, max(8, n_blocks // 8), P_dbm, seed + 10000)
    Nex = ys.shape[0]
    opt = torch.optim.Adam(param_groups)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    pend = list(prune_events) if prune_events else []
    best, bstate = 1e9, None
    last_improve = 0
    t0 = time.time()
    for it in range(iters):
        while pend and it >= pend[0][0]:
            _, NN = pend.pop(0)
            model.prune_one(NN)
        idx = torch.randint(0, Nex, (batch,), device=device)
        opt.zero_grad()
        loss = equalize_loss(model, ys[idx], xs[idx], P, S, ldbp=ldbp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if it % 200 == 0 or it == iters - 1:
            with torch.no_grad():
                vl = equalize_loss(model, yv, xv, P, S, ldbp=ldbp).item()
            # only checkpoint the best AFTER pruning is complete -- otherwise the best
            # is the pre-pruned (full-tap) model, which doesn't match the final masks.
            if not pend and vl < best:
                if 10 * (np.log10(best) - np.log10(vl)) > 0.005 or bstate is None:
                    last_improve = it                      # meaningful improvement
                best = vl
                bstate = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if it % 1000 == 0:
                tag = f" (pruning, {len(pend)} left)" if pend else ""
                print(f"  iter {it}: val SNR {-10*np.log10(vl):.3f}  best "
                      f"{-10*np.log10(best) if best < 1e8 else float('nan'):.3f}{tag}", flush=True)
            if (patience is not None and not pend and bstate is not None
                    and it - last_improve >= patience):
                print(f"  early stop at iter {it} (no val improvement in {patience} iters)",
                      flush=True)
                break
    if bstate is not None:
        model.load_state_dict(bstate)
    print(f"  trained in {time.time()-t0:.0f}s, best val SNR {-10*np.log10(best):.3f} dB", flush=True)
    return model


# --------------------------------------------------------------------------- #
# per-method training
# --------------------------------------------------------------------------- #
def _power(S, P_dbm):
    """Training launch power: explicit arg, else the scenario's training power [dBm]
    (near its operating peak), read from the config by build_system. Default 9.0."""
    return S.get('train_power_dbm', 9.0) if P_dbm is None else P_dbm


def nlpr_length(S, ns):
    """One-sided NLPR filter length from the MATLAB essfmNc formula (Civelli,
    test_ccessfm.m / DSP.m): Nc = round(pi*(L_tot/ns)*|b2|*(R*n)^2/2)+2, length=Nc+1.
    Sized on the PER-STEP memory L_tot/ns -- shrinks as Ns grows. Used for BOTH
    ESSFM and L-ESSFM (Dario, June 2026). Examples: 01 Ns=1 -> 66 (historical),
    03 Ns=1 -> 450 (+0.06 vs fixed 66), 03 Ns=10 -> 48.
    Measured caveat (03 Ns=10): the longest LEARNED step can exceed L_tot/ns (e.g.
    1.34x with nonneg) -> per-step sizing loses ~0.05 vs 66 there; pass nfl= to
    override in experiments."""
    LL = S['Lsp'] * S['Nsp']
    nc = max(round(np.pi * LL * abs(S['beta2']) / max(int(ns), 1)
                   * (S['fsym'] * S['OS_d']) ** 2 / 2), 1) + 2
    return int(nc) + 1


def _ensure_block(S, nfl):
    """Training blocks must HOLD the symmetric NLPR filter with margin: measured on
    03/n=2/Ns=1, a 2047-tap filter in a 2048-sample circular block self-wraps and
    LOSES 0.15 dB (17.486 vs 17.633 with 66 taps), while the same filter with a
    2x block wins (17.760). Rule (Dario): keep 1024-sym blocks unless the full
    filter (2*nfl-1) exceeds HALF the block; then rebuild S from its config with
    the block doubled (x2, x4, ...) until it fits. Block size itself is otherwise
    irrelevant (1024 vs 2048 at nfl=66: 17.633 vs 17.627)."""
    import re
    from system import build_system
    while 2 * nfl - 1 > S['Nsamp_d'] // 2:
        new_sym = 2 * S['Nsym']
        s = open(S['cfg_path']).read()
        s = re.sub(r'(?m)^data symbols per block = .*',
                   f'data symbols per block = {new_sym}', s)
        tmp = 'config/_autoblock.ini'
        open(tmp, 'w').write(s)
        S = build_system(tmp)
        print(f"  [block doubled to {new_sym} sym to fit nfl={nfl}]", flush=True)
    return S


def train_lessfm(S, ns, out, device='cuda', iters=12000, batch=200, P_dbm=None,
                 lr_len=0.2, lr_nlf=0.002, n_blocks=250, seed=0, nonneg=True, nfl=None):
    S = dict(S)
    S['nl_filter_length'] = nlpr_length(S, ns) if nfl is None else nfl
    S = dict(_ensure_block(S, S['nl_filter_length']), nl_filter_length=S['nl_filter_length'])
    tf = TorchForward(S, device)
    m = LessfmModel(S, ns, device, nonneg=nonneg).to(device)
    groups = [{'params': [m.length], 'lr': lr_len}, {'params': [m.nlf], 'lr': lr_nlf}]
    _train_loop(m, S, tf, groups, iters, batch, _power(S, P_dbm), n_blocks, seed)
    save_params(out, m); print(f"  saved {out} (nfl={S['nl_filter_length']})")
    return m


def train_essfm(S, ns, out, device='cuda', iters=12000, batch=200, P_dbm=None,
                lr_rho=0.01, lr_nlf=0.002, n_blocks=250, seed=0, nfl=None):
    S = dict(S)
    S['nl_filter_length'] = nlpr_length(S, ns) if nfl is None else nfl
    S = dict(_ensure_block(S, S['nl_filter_length']), nl_filter_length=S['nl_filter_length'])
    tf = TorchForward(S, device)
    m = LessfmModel(S, ns, device, tied_kerr=True, opt_rho=True).to(device)
    groups = [{'params': [m.rho], 'lr': lr_rho}, {'params': [m.nlf], 'lr': lr_nlf}]
    _train_loop(m, S, tf, groups, iters, batch, _power(S, P_dbm), n_blocks, seed)
    save_params(out, m); print(f"  saved {out} (nfl={S['nl_filter_length']})")
    return m


def _tile_targets(pattern, M):
    """Tile a per-step target tap pattern (e.g. [11,9]) to M steps: 11 9 11 9 ..."""
    return [int(pattern[i % len(pattern)]) for i in range(M)]


def train_ldbp(S, ns, out, cd_lengths, device='cuda', iters=15000, batch=100, P_dbm=None,
               lr=5e-4, n_blocks=2000, seed=0, init_cd=None, init_nl=None, total_steps=False,
               target_taps=None, n_prune_stages=8, single_pol=False):
    """LDBP trainer with GRADUAL PRUNING (the working recipe, see memory
    ldbp-pruning-variable-taps): init with abundant cd_lengths taps, then prune the
    CD-FIRs progressively DURING training down to target_taps. target_taps is a
    per-step PATTERN (e.g. [11,9]) tiled across steps -- a single uniform value
    degenerates at 93 GBaud; a variable pattern reaches low average complexity.
    Use a logarithmic-step config for best results."""
    from ldbp_torch import LdbpModel
    S_train = dict(S)
    S_train['Npol'] = 1  # LDBP is single-pol; train on single-pol to avoid XPM noise floor
    tf = TorchForward(S_train, device)
    m = LdbpModel(S, ns, device, cd_lengths, init_cd=init_cd, init_nl=init_nl,
                  total_steps=total_steps).to(device)
    m.single_pol = single_pol      # match TF1 single-pol Kerr during training if set
    # Hager's LDBP trains ONLY the CD-FIR filters; the Kerr (m.nl) is a fixed buffer
    # (= physical bw.nl_param), not optimized.
    groups = [{'params': list(m.cd_re.parameters()) + list(m.cd_im.parameters()), 'lr': lr}]
    prune_events = None
    if target_taps is not None:
        import random as _random
        tgt = _tile_targets(target_taps, m.M)
        init0 = list(cd_lengths)
        # build the per-(step) one-tap prune queue: for each step, it must lose
        # (init-target)/2 right-half taps. Mirror ldbp_diag.py: outer taps first,
        # shuffled across steps, on an exponential schedule (dense near the end).
        order = []
        for NN in range(m.M):
            n_remove = max(0, (init0[NN] - tgt[NN]) // 2)
            order += [NN] * n_remove
        _random.Random(seed).shuffle(order)
        K = len(order)
        if K > 0:
            pruning_schedule = np.ceil(np.ceil(2.0**(-np.arange(K, 0, -1)) * iters))
            pruning_schedule = np.ceil(pruning_schedule + np.arange(K) * iters / 8 / K)
            sched = np.clip(pruning_schedule, 0, iters - 1).astype(int)
            prune_events = list(zip(sched.tolist(), order))
    _train_loop(m, S_train, tf, groups, iters, batch, _power(S, P_dbm), n_blocks, seed,
                ldbp=True, prune_events=prune_events)
    _save_ldbp(out, m); print(f"  saved {out}")
    return m


def _save_ldbp(path, model):
    """LDBP params.csv: per step CD-real, CD-imag, nl scalar (mirror right-half).
    Applies the pruning mask and drops the trailing zeroed taps so the saved filter
    has the PRUNED length (the eval reads the actual length per row)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for NN in range(model.M):
            mask = model.masks[NN].detach().cpu().numpy()
            keep = int(mask.sum())                                   # active right-half taps
            rr = (model.cd_re[NN].detach().cpu().numpy() * mask)[:keep]
            ri = (model.cd_im[NN].detach().cpu().numpy() * mask)[:keep]
            fr = np.concatenate([rr[1:][::-1], rr]); fi = np.concatenate([ri[1:][::-1], ri])
            f.write(','.join(f'{v:.18e}' for v in fr) + '\n')
            f.write(','.join(f'{v:.18e}' for v in fi) + '\n')
            f.write(f'{model.nl[NN].item():.18e}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['lessfm', 'essfm', 'ldbp'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--ns', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=12000)
    ap.add_argument('--cd-taps', type=int, default=11, help='LDBP: CD-FIR taps per step')
    args = ap.parse_args()
    S = build_system(args.config)
    if args.method == 'lessfm':
        train_lessfm(S, args.ns, args.out, iters=args.iters)
    elif args.method == 'essfm':
        train_essfm(S, args.ns, args.out, iters=args.iters)
    else:
        bw = make_bw(S, args.ns)
        cd_lengths = [args.cd_taps] * bw.model_steps
        train_ldbp(S, args.ns, args.out, cd_lengths, iters=args.iters)


if __name__ == '__main__':
    main()
