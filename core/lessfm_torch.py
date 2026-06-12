"""
lessfm_torch.py -- L-ESSFM / ESSFM training in PyTorch (migration from TF1).

Goals of the migration:
  - 6-12x faster (large batch saturates the GPU; torch.compile fuses kernels).
  - per-group learning rates instead of the nl_alpha trick (cd-length params and
    NLPR-filter params get their own lr) -> robust convergence, no magic nl_alpha.
  - early stopping on the best iterate.
  - clean eager code.

The PHYSICS (per-step GVD lengths, NLPR strengths) reuses the validated
SSFMParameters from core/system.py, so geometry matches the TF1 / MATLAB exactly.

Backprop model (Ns steps, combine_half_steps -> Ns+1 GVD segments):
  for NN in range(M):
     y = ifft( fft(y) * exp(j * cd_filter_freq[NN] * length_mult[NN]) )    # GVD
     if NN < M-1:                                                          # NLPR
        ysq = |y|^2 (summed over pol)
        theta = irfft( rfft(ysq) * rfft(nlf_padded[NN]) )                  # FIR in freq
        y *= exp(j*theta)

length_mult[NN] are the trainable GVD multipliers (L-ESSFM: all free, exp-init;
ESSFM: uniform + one rho). nlf[NN] are the trainable NLPR filters (one-sided,
mirrored to 2*nfl-1 taps at apply). Loss = MSE between equalized symbols and tx.
"""
import numpy as np
import torch

from system import build_system, make_bw, SSFMParameters, qam_constellation


def _exp_length_mult(S, bw, ns):
    """Exponential GVD-length init (backward equal-NL power profile), as in TF1.
    Returns multipliers of the nominal cd_length (sum -> Ns, halved borders).

    Uses the BACKPROP geometry from `bw` (bw.Nsp, bw.Lsp, bw.StPS) so it is correct
    for BOTH per-span and total-steps conventions. With total-steps, bw maps the
    link to bw.Nsp 'spans' of length bw.Lsp with StPS=1 -> the exp profile is built
    on that geometry (NOT the original S['Nsp']/S['Lsp'], which would mismatch)."""
    Lsp = bw.Lsp; Nsp = bw.Nsp; sps = bw.StPS
    alpha_l = S['alpha'] / (10 * np.log10(np.e)) / 1000.0   # 1/m

    def span_steps(Nstep):
        z = [0.0]
        for i in range(1, Nstep):
            z.append(-np.log(1 - (i / Nstep) * (1 - np.exp(-alpha_l * Lsp))) / alpha_l)
        z.append(Lsp)
        return np.diff(z)[::-1]
    steps_phys = np.concatenate([span_steps(sps) for _ in range(Nsp)])
    M = bw.model_steps
    cd_phys = np.zeros(M)
    cd_phys[0] = steps_phys[0] / 2
    for i in range(1, M - 1):
        cd_phys[i] = (steps_phys[i - 1] + steps_phys[i]) / 2
    cd_phys[M - 1] = steps_phys[-1] / 2
    return (cd_phys / bw.cd_length).astype(np.float32)


class LessfmModel(torch.nn.Module):
    """L-ESSFM (free lengths + per-step NLPR) or ESSFM (uniform + rho + tied NLPR)."""

    def __init__(self, S, ns, device, tied_kerr=False, opt_rho=False, nonneg=False):
        super().__init__()
        self.S = S
        self.bw = make_bw(S, ns)
        self.M = self.bw.model_steps
        self.nfl = S['nl_filter_length']
        self.ns = ns
        self.tied_kerr = tied_kerr
        self.opt_rho = opt_rho
        self.nonneg = nonneg          # constrain GVD lengths >=0 (softmax param)
        self.dev = device

        # fixed per-step GVD phase = get_cd_filter_freq(NN) (includes cd_length[NN]).
        # The trainable length multiplier then scales this. Matches numpy backprop:
        #   exp(j * get_cd_filter_freq(NN) * length_mult[NN] / cd_alpha)
        cdf = [self.bw.get_cd_filter_freq(NN) for NN in range(self.M)]
        self.cd_base = torch.tensor(np.array(cdf), dtype=torch.float32, device=device)  # [M, N]

        # trainable GVD length multipliers
        if opt_rho:                                   # ESSFM: one shared rho
            self.rho = torch.nn.Parameter(torch.tensor([0.5], device=device))
        else:
            # L-ESSFM: the FIRST M-1 steps are free trainable multipliers; the LAST
            # step is DERIVED so the total dispersion (sum with halved borders) stays
            # fixed at target_sum. This matches TF1 (lessfm.py L691-703) and is the
            # key to a well-conditioned problem: the lengths can redistribute (front-
            # load step 0) while the total CD budget is preserved. Training all M
            # freely (no constraint) lets the sum drift -> stuck at init or diverges.
            init = _exp_length_mult(S, self.bw, ns)   # length [M] (exp-init, halved borders)
            self.target_sum = float(init[0] / 2 + init[1:-1].sum() + init[-1] / 2)
            if nonneg:
                # softmax over M params (init from the exp profile via log) -> the
                # length_mult() softmax reproduces the exp init and stays >=0.
                self.length = torch.nn.Parameter(torch.log(torch.tensor(init, device=device).clamp_min(1e-3)))
            else:
                self.length = torch.nn.Parameter(torch.tensor(init[:-1], device=device))  # free: [M-1]

        # trainable NLPR filters (one-sided), init UNIT DELTA in time -- matches TF1.
        # TF1 stores no_filter[0] = nl_alpha and applies -nlf*nl_step_scale/nl_alpha,
        # so for L-ESSFM (nl_step_scale=1) the PHYSICAL initial filter is a unit delta.
        # We carry no nl_alpha (per-group lr replaces it), so we store the physical
        # delta directly: no[0] = 1.0.
        no = np.zeros(self.nfl, dtype=np.float32); no[0] = 1.0
        if tied_kerr:
            self.nlf = torch.nn.Parameter(torch.tensor(no, device=device))         # [nfl]
        else:
            self.nlf = torch.nn.Parameter(torch.tensor(np.tile(no, (self.M - 1, 1)), device=device))  # [M-1, nfl]

        # per-step nonlinear strength (for tied ESSFM scaling), constant
        self.nl_param = torch.tensor(self.bw.nl_param.astype(np.float32), device=device)
        self.nl_ref = float(np.mean(self.bw.nl_param[:-1])) if self.M > 1 else 1.0

        self.Nsamp_d = S['Nsamp_d']
        # RX matched filter (frequency), real-valued
        self.ps_rx = torch.tensor(S['ps_rx_freq'].astype(np.complex64), device=device)
        self.cd_alpha = S['cd_alpha']

    def length_mult(self):
        """Per-step GVD multiplier vector [M]."""
        if self.opt_rho:
            r = self.rho
            mult = torch.ones(self.M, device=self.dev)
            mult = mult.clone()
            mult[0] = 2.0 * r.squeeze()
            mult[-1] = 2.0 * (1.0 - r.squeeze())
            return mult
        if getattr(self, 'nonneg', False):
            # Non-negative GVD lengths with fixed total dispersion: softmax over all
            # M params -> positive fractions summing to 1, scaled so the halved-border
            # sum equals target_sum. Border weight 1/2 folded into the budget. Keeps
            # lengths >=0 (physical: GVD steps are distances) and the CD budget fixed.
            w = torch.softmax(self.length, dim=0)        # [M], >0, sums to 1
            # halved-border effective sum = w[0]/2 + w[1:-1].sum() + w[-1]/2 = 1 - (w[0]+w[-1])/2
            eff = 1.0 - (w[0] + w[-1]) / 2
            return w * (self.target_sum / eff)
        # free first M-1 multipliers; derive the last to keep the dispersion sum
        # (halved borders) fixed at target_sum (matches TF1 L694-703):
        #   cd_sum = length[0]/2 + sum(length[1:M-1]);  last = 2*(target_sum - cd_sum)
        free = self.length                              # [M-1]
        cd_sum = free[0] / 2 + free[1:].sum()
        last = 2.0 * (self.target_sum - cd_sum)
        return torch.cat([free, last.reshape(1)])

    def _nlf_padded_fft(self, NN):
        """rfft of the padded/mirrored NLPR filter for step NN."""
        f = self.nlf if self.tied_kerr else self.nlf[NN]
        f = -f   # matches numpy backprop: nl_time = -nlf (filter applied with minus)
        if self.tied_kerr and self.opt_rho:
            f = f * (self.nl_param[NN] / self.nl_ref)
        # EXACT numpy mirror: concat[ flip(f[1:]), f, zeros ], then roll by -(nfl-1).
        # f[1:].flip = the taps from index 1.. reversed (drops the CENTER/first tap once).
        mirror = f[1:].flip(0)
        pad = torch.zeros(self.Nsamp_d, dtype=f.dtype, device=self.dev)
        pad[:self.nfl - 1] = mirror
        pad[self.nfl - 1:2 * self.nfl - 1] = f
        pad = torch.roll(pad, -self.nfl + 1)
        return torch.fft.rfft(pad)

    def forward(self, y, pol_dim=None):
        """Backpropagate y and return the equalized complex signal.

        y: real/imag tensor with last dim = 2.
           - training: y is [B, N, 2] (each sample is one polarization; dual-pol
             is fed as two separate batch examples). pol_dim=None.
           - test/eval (dual-pol jointly): y is [Npol, N, 2] and the NLPR uses the
             TOTAL power over polarizations -> pass pol_dim=0 so ysq sums over pol.
        """
        # Cache nonlinear filter FFTs during evaluation to save huge CPU/GPU time
        if not self.training:
            if getattr(self, '_cached_nlf_fft', None) is None:
                self._cached_nlf_fft = [self._nlf_padded_fft(nn) for nn in range(self.M - 1)]
        else:
            self._cached_nlf_fft = None

        lm = self.length_mult()
        yc = torch.complex(y[..., 0], y[..., 1])
        for NN in range(self.M):
            phase = self.cd_base[NN] * lm[NN] / self.cd_alpha
            yc = torch.fft.ifft(torch.fft.fft(yc) * torch.exp(1j * phase))
            if NN < self.M - 1:
                ysq = yc.abs() ** 2                          # |y|^2 per signal
                if pol_dim is not None:
                    ysq = ysq.sum(pol_dim, keepdim=True)     # total power over pols
                
                nlf_fft = self._cached_nlf_fft[NN] if not self.training else self._nlf_padded_fft(NN)
                theta = torch.fft.irfft(torch.fft.rfft(ysq) * nlf_fft, n=self.Nsamp_d)
                yc = yc * torch.exp(1j * theta)
        return yc
