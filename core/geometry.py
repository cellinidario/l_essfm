"""geometry.py -- the physical reference for L-ESSFM GVD step lengths.

Marco, Stella and Dario expect the trained GVD lengths to approach a specific
distribution tied to the link power profile. This module builds that reference so
it can be used to JUDGE a training run, and lessfm_torch uses it as the starting
point of the GVD lengths. Nothing here is imposed on the optimiser: the training
stays free to find its own solution.

The construction (Dario's sketch, option B)
-------------------------------------------
Split the link into Ns intervals carrying EQUAL ACCUMULATED NONLINEARITY, i.e.
equal area under the real power profile. On a link of Nsp spans of length Lsp,
each amplifier restores the launch power, so the profile is a sawtooth and the
accumulated nonlinearity over a full span is the effective length

    Leff(L) = (1 - exp(-alpha L)) / alpha.

Walking from the transmitter, the i-th boundary z_i solves

    NL(z_i) = (i / Ns) * Nsp * Leff(Lsp),    NL(z) = k*Leff(Lsp) + Leff(z - k*Lsp)

with k the number of complete spans before z. High-power stretches give short
intervals, low-power stretches long ones.

Two things this module is careful about, both of which the old inline
initialiser got wrong:

1. **alpha units.** `build_system` already stores `S['alpha']` in dB/m, so the
   linear coefficient is `S['alpha']/DB` and nothing more. Dividing by 1000 again
   makes the fibre look 1000x more transparent, the attenuation over a span drops
   from 7.8 to 0.008 nepers, and the "exponential" split degenerates to uniform.

2. **The real profile, not the remapped one.** When Ns is not a multiple of Nsp
   (15 spans, 10 steps), `make_bw` maps the link onto Ns fictitious spans with one
   step each. Building the split on THAT geometry invents amplifiers that do not
   exist and again gives uniform steps. The split must be taken on the true
   sawtooth, even where an interval straddles a real amplifier.

Backprop runs from the receiver, so the returned arrays are reversed: index 0 is
the stretch at the far end of the link, which carries the least power and is
therefore the longest.
"""
import numpy as np

DB = 10.0 * np.log10(np.e)


def alpha_lin(S):
    """Attenuation in 1/m from the system dict (S['alpha'] is dB/m)."""
    return S['alpha'] / DB


def _leff(a, L):
    return (1.0 - np.exp(-a * L)) / a if a > 0 else L


def equal_nl_intervals(a, Lsp, Nsp, ns):
    """Interval lengths [m], transmitter -> receiver, of equal accumulated
    nonlinearity on the real sawtooth power profile. Returns ns values summing to
    Nsp*Lsp."""
    leff_span = _leff(a, Lsp)
    total_nl = Nsp * leff_span
    z = [0.0]
    for i in range(1, ns):
        target = (i / ns) * total_nl
        k = min(int(target // leff_span), Nsp - 1)   # complete spans before z
        rem = target - k * leff_span
        # invert Leff within the span; clip guards the floating-point edge
        inner = -np.log(max(1.0 - a * rem, 1e-300)) / a if a > 0 else rem
        z.append(k * Lsp + min(inner, Lsp))
    z.append(Nsp * Lsp)
    return np.diff(np.array(z))


def reference_lengths(S, bw, ns):
    """Reference GVD segment lengths [m] for a backprop model with `bw.model_steps`
    segments, in BACKPROP order (index 0 = far end of the link).

    The GVD segments are the stretches between consecutive nonlinear operators, plus
    the two borders, with the operators placed by `nl_positions`. The first version
    put each operator at the LENGTH midpoint of its equal-nonlinearity interval,
    which on a 170 km span with one step gave 0.50 / 0.50 instead of 0.91 / 0.09,
    and on 15 x 80 km put every operator mid-span, between two power peaks.
    """
    pos = nl_positions(S, ns)
    edges = np.concatenate([[0.0], pos, [S['Nsp'] * S['Lsp']]])
    cd = np.diff(edges)[::-1]                                       # receiver -> tx
    assert cd.size == bw.model_steps, (cd.size, bw.model_steps)
    return cd


def reference_multipliers(S, bw, ns):
    """`reference_lengths` expressed as multipliers of bw.cd_length, which is the
    quantity the model actually trains and parameters.csv actually stores."""
    return (reference_lengths(S, bw, ns) / bw.cd_length).astype(np.float32)


def trained_lengths(params_csv, bw):
    """Physical GVD segment lengths [m] of a saved L-ESSFM model.

    parameters.csv stores the multiplier per segment with the two border values
    already halved (see train.save_params), so the halving is undone here before
    multiplying by the nominal geometry."""
    M = bw.model_steps
    lines = open(params_csv).read().strip().split('\n')
    nfl = (len(lines) - M) // (M - 1)
    mult, idx = np.zeros(M), 0
    for NN in range(M):
        mult[NN] = float(lines[idx]); idx += 1
        if NN == 0 or NN == M - 1:
            mult[NN] *= 2
        if NN < M - 1:
            idx += nfl
    return mult * bw.cd_length


def convergence_report(S, bw, ns, params_csv):
    """Compare a trained model against the reference. Returns the two length
    profiles plus four numbers worth looking at:

      rel_rms      RMS of (trained - reference) over the mean segment length.
      max_rel_dev  worst single segment, same normalisation.
      n_collapsed  segments under 20% of the reference at that index, i.e. steps
                   the optimiser effectively threw away.
      n_sign_flips sign changes of (trained - reference) along the link. The
                   reference is smooth, so a converged run should track it; many
                   flips mean the profile is oscillating, which is what pairwise
                   step merging looks like.

    Note the reference is NOT always decreasing, and it is not the optimum either:
    on 15 x 80 km at Ns = 15 it puts each operator 14.5 km after its amplifier
    (18.379 dB with the lengths frozen there), while the trained optimum sits near
    7 km (18.460). Judge against the reference, never against an assumed shape.
    """
    ref = reference_lengths(S, bw, ns)
    got = trained_lengths(params_csv, bw)
    d = got - ref
    scale = float(ref.mean())
    sign = np.sign(d[np.abs(d) > 0.01 * scale])
    return dict(
        reference_km=ref / 1e3,
        trained_km=got / 1e3,
        rel_rms=float(np.sqrt(np.mean(d ** 2)) / scale),
        max_rel_dev=float(np.max(np.abs(d)) / scale),
        n_collapsed=int((got < 0.2 * ref).sum()),
        n_sign_flips=int((np.diff(sign) != 0).sum()) if sign.size > 1 else 0,
        total_ref_km=float(ref.sum()) / 1e3,
        total_trained_km=float(got.sum()) / 1e3,
    )


def nl_positions(S, ns):
    """Positions [m] of the nonlinear operators, from the transmitter. The k-th sits
    where the accumulated nonlinearity reaches (k - 1/2)/ns of the total, i.e. at the
    point of its equal-nonlinearity interval with equal areas on both sides. This is
    the quantity Dario's sketch draws on the power profile."""
    a = alpha_lin(S)
    Lsp, Nsp = S['Lsp'], S['Nsp']
    leff_span = _leff(a, Lsp)
    pos = []
    for k in range(1, ns + 1):
        target = (k - 0.5) / ns * Nsp * leff_span
        j = min(int(target // leff_span), Nsp - 1)   # complete spans before the operator
        rem = target - j * leff_span
        inner = -np.log(max(1.0 - a * rem, 1e-300)) / a if a > 0 else rem
        pos.append(j * Lsp + min(inner, Lsp))
    return np.array(pos)
