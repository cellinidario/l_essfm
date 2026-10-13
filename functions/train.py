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
import os, sys, argparse, time, json
import numpy as np
import torch

from system import build_system, make_bw
from forward_torch import TorchForward
from lessfm_torch import LessfmModel
from forward import forward_identity, FORWARD_VERSION


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
    if hasattr(model, 'training_record'):
        with open(path + '.json', 'w') as f:
            json.dump(model.training_record, f, indent=2)


# --------------------------------------------------------------------------- #
# dual-pol data generation (GPU forward) and loss
# --------------------------------------------------------------------------- #
def gen_dualpol(tf, S, n_blocks, P_dbm, seed):
    """GPU forward -> dual-pol blocks kept paired: y[B,Npol,Nd,2], x[B,Npol,Nsym]."""
    Npol, Nsamp_d, Nsym = S['Npol'], S['Nsamp_d'], S['Nsym']
    generator = torch.Generator(device=tf.dev).manual_seed(seed)
    P = 10 ** (P_dbm / 10) * 1e-3
    yb, xb = tf(n_blocks, P, generator=generator)
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
def release_gpu_memory():
    """Give back to the device the memory cached by earlier work in this process.
    Besides PyTorch's allocator and cuFFT plans, CuPy (used by the evaluations) keeps
    its own memory pool and FFT plans, which torch.cuda.empty_cache() does not touch:
    measured 2026-10-09, one evaluation of 8 powers left 3.2 GiB unavailable to
    PyTorch, and in a notebook kernel this shrank the LDBP microbatch from 100 to 78
    (two passes per iteration instead of one, twice the training time)."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.cufft_plan_cache.clear()
    torch.cuda.empty_cache()
    cp = sys.modules.get('cupy')
    if cp is not None:
        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def _other_gpu_processes():
    """(pid, command) of the other processes that have the GPU open: /dev/dxg on WSL,
    /dev/nvidia<N> on Linux. Every process with a CUDA context keeps it open."""
    found = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            targets = [os.readlink(f'/proc/{pid}/fd/{fd}') for fd in os.listdir(f'/proc/{pid}/fd')]
            if any(t == '/dev/dxg' or (t.startswith('/dev/nvidia') and t[11:].isdigit())
                   for t in targets):
                with open(f'/proc/{pid}/cmdline', 'rb') as f:
                    cmd = f.read().replace(b'\0', b' ').decode(errors='replace').strip()
                found.append((int(pid), cmd))
        except OSError:                     # process gone, or not ours
            continue
    return found


def warn_if_gpu_shared():
    """Print a warning when other processes (e.g. notebook kernels) have the GPU open.
    On WSL every process sees almost the whole GPU as free, so two jobs can together
    ask for more memory than the card has; the driver then moves memory to the host
    instead of failing, and everything slows down. Measured 2026-10-09: the LDBP
    training-data generation of 15x80 km took 18 s alone and more than 20 min while a
    notebook kernel was training on the same GPU.
    Until 2026-10-12 this estimated the others' memory as nvidia-smi's total minus
    this process's PyTorch and CuPy pools plus 0.5 GiB of context. It reported 1.5 GiB
    of "other processes" when this was the only process with the GPU open: the
    context and the libraries loaded on the GPU (two cuFFT, PyTorch's and CuPy's) are
    larger than 0.5 GiB. Hence the check on the processes themselves."""
    others = _other_gpu_processes()
    if others:
        names = '; '.join(f'PID {pid}: {cmd[:80]}' for pid, cmd in others)
        print(f'  WARNING: other processes have the GPU open ({names}). Running GPU jobs side '
              'by side can slow training down many times.', flush=True)


# --------------------------------------------------------------------------- #
# training-forward cache: the forward data-gen for a multi-span scenario (15 spans
# x ~100 fwd steps = 1500 SSFM steps) is ~15x heavier than single-span and was being
# regenerated for EVERY Ns. Cache the complete forward physics, dimensions,
# train power, n_blocks, seed and device -- NOT Ns or the backward model -- and reuse
# across all Ns sharing the same block size. Cuts sc03 data-gen from ~30 regens to
# ~2 (one per distinct block size: 1024 and the doubled 2048).
_FWD_CACHE = {}
def train_forward(tf, S, n_blocks, P_dbm, seed):
    """Cache by complete forward identity, exact power, seeds and device."""
    key = (forward_identity(S), float(P_dbm), n_blocks, seed, str(tf.dev))
    if key not in _FWD_CACHE:
        ys, xs, P = gen_dualpol(tf, S, n_blocks, P_dbm, seed)
        yv, xv, _ = gen_dualpol(tf, S, max(8, n_blocks // 8), P_dbm, seed + 10000)
        _FWD_CACHE[key] = (ys, xs, yv, xv, P)
    return _FWD_CACHE[key]


def clear_forward_cache():
    """Free the cached training forwards (e.g. before switching scenario/oversampling)."""
    _FWD_CACHE.clear()


EARLY_STOP_DB = 0.002        # see _train_loop docstring
EARLY_STOP_CHECKS = 1


def _train_loop(model, S, tf, param_groups, iters, batch, P_dbm, n_blocks, seed,
                ldbp=False, device='cuda', prune_events=None, early_stop=True, data=None):
    """prune_events: optional list of (iteration, step_index) -- at the given iteration,
    remove ONE outermost tap from that step (model.prune_one). Mirrors ldbp_diag.py:
    gradual one-tap-at-a-time pruning so the model adapts between prunes (pruning many
    taps at once shocks the model to 0 dB).
    data: optional pre-generated (ys, xs, yv, xv, P) -- skips the forward data-gen
    (reused across Ns with the same block size; see train_forward).
    Early stop: stop at the first 1000-iteration checkpoint whose validation SNR moved
    by less than EARLY_STOP_DB = 0.002 dB (in either direction). Chosen 2026-10-07 on
    the 129 trainings of the independent_ase_v1 set: 41% of the iterations of the
    previous rule (three consecutive changes below 0.001 dB), SNR lost vs the full run
    median 0.0001, 95th percentile 0.0022, worst 0.0048 dB (tolerance agreed with
    Dario: 0.01 dB). This is a validation plateau, not proof of
    geometric or global convergence. NOT armed while pruning is pending: `best` is recorded
    once the masks are final, so an early stop mid-pruning would leave bstate None
    and save a model with more taps than the target. On the
    measured logs: flat ESSFM stops early (was burning hours, e.g. 16602s with best
    frozen since iter 1000), while the slow steady LDBP tail (+0.007..0.03 dB/1000)
    is never cut."""
    if iters < 1:
        raise ValueError('iters must be positive')
    if data is not None:
        ys, xs, yv, xv, P = data
    else:
        ys, xs, P = gen_dualpol(tf, S, n_blocks, P_dbm, seed)
        yv, xv, _ = gen_dualpol(tf, S, max(8, n_blocks // 8), P_dbm, seed + 10000)
    Nex = ys.shape[0]
    initial_learning_rates = [float(g['lr']) for g in param_groups]
    # Bound activation memory, not the effective minibatch. Accumulate gradients
    # for the same sampled batch before one Adam update at high step counts.
    samples_per_example = S['Npol']*S['Nsamp_d']*getattr(model,'M',1)
    # ~28 bytes of activations per sample and step (measured); use up to 40% of the
    # free GPU memory, never less than the previous fixed bound of 32 M samples.
    limit = 32_000_000
    if str(device).startswith('cuda'):
        release_gpu_memory()               # memory cached by earlier work counts as free
        warn_if_gpu_shared()
        limit = max(limit, int(0.3 * torch.cuda.mem_get_info(device)[0] / 30))
    microbatch = min(batch, max(1, limit // samples_per_example))
    pend = list(prune_events) if prune_events else []
    # The training step is launch-bound: at 200 steps one iteration is ~10^5 tiny GPU
    # kernels and the GPU was busy ~13% of the time. Record the whole step (all
    # microbatches, clipping, Adam) once in a CUDA graph and replay it: 4-7x faster,
    # parameters bit-identical to the same step run eagerly (checked 2026-10-07).
    # Also used for LDBP with pruning: the masks are modified in place, so the
    # recorded step sees every pruning event. Disable with LESSFM_CUDA_GRAPH=0.
    use_graph = (str(device).startswith('cuda')
                 and os.environ.get('LESSFM_CUDA_GRAPH', '1') != '0')
    if use_graph:          # capturable Adam reads the learning rate from device memory
        for g in param_groups:
            g['lr'] = torch.tensor(float(g['lr']), device=device)
    opt = torch.optim.Adam(param_groups, capturable=use_graph)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    static_idx = torch.zeros(batch, dtype=torch.long, device=device)
    static_loss = torch.zeros((), device=device)

    def train_step():
        opt.zero_grad(set_to_none=not use_graph)
        static_loss.zero_()
        for start in range(0, batch, microbatch):
            subidx = static_idx[start:start + microbatch]
            weight = subidx.numel() / batch
            part = equalize_loss(model, ys[subidx], xs[subidx], P, S, ldbp=ldbp)
            static_loss.add_(part.detach() * weight)
            (part * weight).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    graph = None
    if use_graph:
        snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
        while graph is None:
            try:
                # warm-up allocates .grad and the Adam state; its updates are then undone
                side = torch.cuda.Stream(device)
                side.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(side):
                    for _ in range(2):
                        train_step()
                torch.cuda.current_stream(device).wait_stream(side)
                with torch.no_grad():
                    model.load_state_dict(snap)
                    for state in opt.state.values():
                        for v in state.values():
                            if torch.is_tensor(v):
                                v.zero_()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    train_step()
            except Exception as e:
                graph = None
                torch.cuda.synchronize(device)
                with torch.no_grad():
                    model.load_state_dict(snap)
                    for state in opt.state.values():
                        for v in state.values():
                            if torch.is_tensor(v):
                                v.zero_()
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if 'out of memory' in str(e) and microbatch > 8:
                    # the eager fallback is several times slower: retry with less memory
                    microbatch = max(8, microbatch // 2)
                    print(f'  CUDA graph capture out of memory; retrying with microbatch {microbatch}', flush=True)
                    continue
                print(f'  CUDA graph capture failed ({type(e).__name__}: {e}); training without it', flush=True)
                use_graph = False
                for g, lr in zip(param_groups, initial_learning_rates):
                    g['lr'] = lr
                opt = torch.optim.Adam(param_groups)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
                break
    best, bstate = 1e9, None
    prev_ckpt = None                           # val SNR [dB] at the previous checkpoint
    stable_checkpoints = 0
    # Data generation, cache hits and model comparisons must not change minibatches.
    batch_rng = torch.Generator(device=device).manual_seed(seed + 20000)
    history = []
    with torch.no_grad():
        initial_loss = equalize_loss(model, yv, xv, P, S, ldbp=ldbp).item()
    if not np.isfinite(initial_loss):
        raise FloatingPointError('Non-finite initial validation loss')
    history.append({'iteration': -1, 'validation_mse': initial_loss,
                    'pruning_pending': len(pend)})
    if not pend:
        best = initial_loss
        bstate = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    for it in range(iters):
        while pend and it >= pend[0][0]:
            _, NN = pend.pop(0)
            model.prune_one(NN)
        static_idx.copy_(torch.randint(0, Nex, (batch,), device=device, generator=batch_rng))
        if graph is not None:
            graph.replay()
        else:
            train_step()
        sch.step()
        loss = static_loss
        if it % 200 == 0 or it == iters - 1:
            with torch.no_grad():
                vl = equalize_loss(model, yv, xv, P, S, ldbp=ldbp).item()
            if not np.isfinite(vl):
                raise FloatingPointError(f'Non-finite validation loss at iteration {it}')
            history.append({'iteration': it, 'validation_mse': vl,
                            'training_mse': float(loss.detach()), 'pruning_pending': len(pend)})
            # only checkpoint the best AFTER pruning is complete -- otherwise the best
            # is the pre-pruned (full-tap) model, which doesn't match the final masks.
            if not pend and vl < best:
                best = vl
                bstate = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if it % 1000 == 0:
                tag = f" (pruning, {len(pend)} left)" if pend else ""
                snr_db = -10 * np.log10(vl)
                print(f"  iter {it}: val SNR {snr_db:.3f}{tag}", flush=True)
                # never stop while taps are still pending: `best` is only recorded
                # once pruning is complete, so stopping here would save an
                # unfinished model and report best = -90 dB. Seen on 10 GBaud,
                # where the val SNR is flat enough to trip the 0.001 dB criterion
                # mid-pruning (ldbp_p7 stopped with 2 taps left to remove).
                if not pend and prev_ckpt is not None and abs(snr_db - prev_ckpt) < EARLY_STOP_DB:
                    stable_checkpoints += 1
                else:
                    stable_checkpoints = 0
                if early_stop and stable_checkpoints >= EARLY_STOP_CHECKS:
                    print(f"  early stop at iter {it} (val moved "
                          f"less than {EARLY_STOP_DB} dB since the previous checkpoint; not a global-convergence certificate)", flush=True)
                    break
                prev_ckpt = snr_db
    if bstate is not None:
        model.load_state_dict(bstate)
    model.training_record = {'forward_version': FORWARD_VERSION,
        'forward_identity': forward_identity(S), 'ase_mode': S.get('ase_mode', 'independent'),
        'seed': seed, 'minibatch_seed': seed + 20000, 'power_dbm': P_dbm,
        'n_blocks': n_blocks, 'batch': batch, 'requested_iterations': iters,
        'microbatch': microbatch, 'cuda_graph': graph is not None,
        'initial_learning_rates': initial_learning_rates,
        'completed_iterations': it + 1, 'best_validation_mse': best,
        'stop_reason': 'validation_plateau' if it + 1 < iters else 'iteration_budget',
        'history': history}
    passes = -(-batch // microbatch)
    print(f"  trained in {time.time()-t0:.0f}s ({1e3*(time.time()-t0)/(it+1):.0f} ms/iteration, "
          f"{passes} pass{'es' if passes > 1 else ''} per iteration, CUDA graph {'on' if graph is not None else 'off'}), "
          f"best val SNR {-10*np.log10(best):.3f} dB", flush=True)
    # release the graph's private memory pool and the cache, or it accumulates over
    # the many trainings of one notebook (03 ran out of memory on 2026-10-07)
    graph = None
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return model


# --------------------------------------------------------------------------- #
# per-method training
# --------------------------------------------------------------------------- #
def _power(S, P_dbm):
    """Training launch power: explicit arg, else the scenario's training power [dBm]
    (near its operating peak), read from the config by build_system. Default 9.0."""
    return S.get('train_power_dbm', 9.0) if P_dbm is None else P_dbm


def _essfm_nc(S, ns):
    """MATLAB essfmNc formula (Civelli, test_ccessfm.m / DSP.m), one-sided filter
    length Nc+1 = round(pi*(L_tot/ns)*|b2|*(R*n)^2/2)+3. Per-step (~L_tot/ns)."""
    LL = S['Lsp'] * S['Nsp']
    nc = max(round(np.pi * LL * abs(S['beta2']) / max(int(ns), 1)
                   * (S['fsym'] * S['OS_d']) ** 2 / 2), 1) + 2
    return int(nc) + 1


def nlpr_length(S, ns, method='lessfm'):
    """One-sided NLPR filter length.
    ESSFM (uniform steps): pure per-step formula `_essfm_nc(Ns)` (shrinks with Ns).
    L-ESSFM (Dario's OFC convention): the GVD step lengths are LEARNED and the first
    step stays ~the whole fiber even at moderate StPS, so the NLPR kernel must stay
    long. The freeze must be on STEPS-PER-SPAN (StPS = ns/Nsp), NOT total steps:
    keep the StPS=1 value (= _essfm_nc at one step per span) FIXED while 1 <= StPS
    <= 10, then follow the per-step formula. Below StPS=1 ("less steps than spans")
    each step spans MORE than one span -> longer kernel -> use the formula directly.
    For single-span scenarios (Nsp=1) StPS==ns, so this reduces to "freeze ns<=10",
    matching 01/02 (66 / 257) unchanged. On 03 (Nsp=15) the frozen value is
    _essfm_nc(StPS=1)=_essfm_nc(15)=33 (n=1.125) / 97 (n=2), held for Ns=15..150,
    then the formula for Ns=300,450 -- NOT the old buggy ns<=10 (which froze the
    full-link 450 and under-sized everything from Ns=30 on)."""
    if method == 'essfm':
        return _essfm_nc(S, ns)
    Nsp = S['Nsp']
    if ns < Nsp:                       # StPS < 1: step spans > one span -> formula (long)
        return _essfm_nc(S, ns)
    if ns <= 10 * Nsp:                 # 1 <= StPS <= 10: freeze the StPS=1 value
        return _essfm_nc(S, Nsp)
    return _essfm_nc(S, ns)            # StPS > 10: per-step formula


def _ensure_block(S, nfl):
    """Training blocks must HOLD the symmetric NLPR filter with margin: measured on
    03/n=2/Ns=1, a 2047-tap filter in a 2048-sample circular block self-wraps and
    LOSES 0.15 dB (17.486 vs 17.633 with 66 taps), while the same filter with a
    2x block wins (17.760). Rule (Dario): keep 1024-sym blocks unless the full
    filter (2*nfl-1) exceeds HALF the block; then rebuild S from its config with
    the block doubled (x2, x4, ...) until it fits. Block size itself is otherwise
    irrelevant (1024 vs 2048 at nfl=66: 17.633 vs 17.627)."""
    import re
    import tempfile
    from system import build_system
    while 2 * nfl - 1 > S['Nsamp_d'] // 2:
        new_sym = 2 * S['Nsym']
        s = open(S['cfg_path']).read()
        s = re.sub(r'(?m)^data symbols per block = .*',
                   f'data symbols per block = {new_sym}', s)
        original_path = S['cfg_path']
        mode = S.get('ase_mode', 'independent')
        with tempfile.TemporaryDirectory(prefix='lessfm-block-') as tmp:
            cfg = os.path.join(tmp, 'block.ini')
            with open(cfg, 'w') as f:
                f.write(s)
            S = build_system(cfg)
        S['cfg_path'] = original_path
        S['ase_mode'] = mode
        print(f"  [block doubled to {new_sym} sym to fit nfl={nfl}]", flush=True)
    return S


def train_lessfm(S, ns, out, device='cuda', iters=12000, batch=200, P_dbm=None,
                 lr_len=0.02, lr_nlf=0.002, n_blocks=250, seed=0, nonneg=True, nfl=None):
    # Keep the reduced length learning rate introduced on 2026-10-06.
    S = dict(S)
    S['nl_filter_length'] = nlpr_length(S, ns) if nfl is None else nfl
    S = dict(_ensure_block(S, S['nl_filter_length']), nl_filter_length=S['nl_filter_length'])
    tf = TorchForward(S, device)
    m = LessfmModel(S, ns, device, nonneg=nonneg).to(device)
    groups = [{'params': [m.length], 'lr': lr_len}, {'params': [m.nlf], 'lr': lr_nlf}]
    data = train_forward(tf, S, n_blocks, _power(S, P_dbm), seed)
    _train_loop(m, S, tf, groups, iters, batch, _power(S, P_dbm), n_blocks, seed, data=data, device=device)
    m.training_record.update(method='lessfm',ns=ns,nfl=S['nl_filter_length'],
        nonnegative_lengths=bool(nonneg),
        initialization='equal_area_quantile',
        geometry_penalty_in_loss=False)
    save_params(out, m); print(f"  saved {out} (nfl={S['nl_filter_length']})")
    return m


def train_essfm(S, ns, out, device='cuda', iters=12000, batch=200, P_dbm=None,
                lr_rho=0.01, lr_nlf=0.002, n_blocks=250, seed=0, nfl=None, opt_rho=False):
    """Train tied-filter ESSFM. Set opt_rho=True for the optimized benchmark.

    The historical default retains the separately labelled fixed-rho=0.5 curve.
    Do not conflate that curve with optimized ESSFM when assessing learned gains.
    """
    S = dict(S)
    S['nl_filter_length'] = nlpr_length(S, ns, method='essfm') if nfl is None else nfl
    S = dict(_ensure_block(S, S['nl_filter_length']), nl_filter_length=S['nl_filter_length'])
    tf = TorchForward(S, device)
    # Both variants retain uniform nominal steps and a tied, physically scaled
    # filter. Omitting rho from the optimizer gives the fixed-rho control only.
    m = LessfmModel(S, ns, device, tied_kerr=True, opt_rho=True).to(device)
    groups = [{'params': [m.nlf], 'lr': lr_nlf}]
    if opt_rho:
        groups.insert(0, {'params': [m.rho], 'lr': lr_rho})
    data = train_forward(tf, S, n_blocks, _power(S, P_dbm), seed)
    _train_loop(m, S, tf, groups, iters, batch, _power(S, P_dbm), n_blocks, seed, data=data, device=device)
    m.training_record.update(method='essfm',ns=ns,nfl=S['nl_filter_length'],
                            optimized_rho=bool(opt_rho),rho=float(m.rho.detach().cpu()))
    save_params(out, m); print(f"  saved {out} (nfl={S['nl_filter_length']}, "
                               f"rho={'opt' if opt_rho else '0.5 fixed'})")
    return m


def _tile_targets(pattern, M):
    """Tile a per-step target tap pattern (e.g. [11,9]) to M steps: 11 9 11 9 ..."""
    return [int(pattern[i % len(pattern)]) for i in range(M)]


def train_ldbp(S, ns, out, cd_lengths, device='cuda', iters=15000, batch=100, P_dbm=None,
               lr=5e-4, n_blocks=2000, seed=0, init_cd=None, init_nl=None, total_steps=False,
               target_taps=None, n_prune_stages=8, single_pol=False, fir_bandwidth=1.0):
    """LDBP trainer with GRADUAL PRUNING (the working recipe, see memory
    ldbp-pruning-variable-taps): init with abundant cd_lengths taps, then prune the
    CD-FIRs progressively DURING training down to target_taps. target_taps is a
    per-step PATTERN (e.g. [11,9]) tiled across steps -- a single uniform value
    degenerates at 93 GBaud; a variable pattern reaches low average complexity.
    Use a logarithmic-step config for best results.
    fir_bandwidth: share of the band the LS-CO init is fitted on (ldbp_torch._fir_obj)."""
    from ldbp_torch import LdbpModel
    S_train = dict(S)
    S_train['Npol'] = 1  # LDBP is single-pol; train on single-pol to avoid XPM noise floor
    tf = TorchForward(S_train, device)
    m = LdbpModel(S, ns, device, cd_lengths, init_cd=init_cd, init_nl=init_nl,
                  total_steps=total_steps, fir_bandwidth=fir_bandwidth).to(device)
    m.single_pol = single_pol      # match TF1 single-pol Kerr during training if set
    if init_cd is None:
        band, oob = m.init_cascade_db()
        memory = (2 * np.pi * abs(S['beta2']) * max(m.bw.cd_length[:m.M])
                  * (1 + S['rolloff']) * S['fsym'] * m.fsamp)
        print(f"  init: {m.M} filters of {max(cd_lengths)} taps (LS-CO on {fir_bandwidth:.0%} of the "
              f"band); alone they invert CD to {band:.1f} dB in band, gain up to {oob:+.1f} dB out "
              f"of band; CD memory of the longest step {memory:.0f} taps", flush=True)
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
                ldbp=True, prune_events=prune_events, device=device)
    m.training_record['fir_bandwidth'] = m.fir_bandwidth
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
    if hasattr(model, 'training_record'):
        with open(path + '.json', 'w') as f:
            json.dump(model.training_record, f, indent=2)


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
