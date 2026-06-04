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
    Returns multipliers of the nominal cd_length (sum -> Ns, halved borders)."""
    Lsp = S['Lsp']; Nsp = S['Nsp']
    alpha_l = S['alpha'] / (10 * np.log10(np.e)) / 1000.0   # 1/m

    def span_steps(Nstep):
        z = [0.0]
        for i in range(1, Nstep):
            z.append(-np.log(1 - (i / Nstep) * (1 - np.exp(-alpha_l * Lsp))) / alpha_l)
        z.append(Lsp)
        return np.diff(z)[::-1]
    steps_phys = np.concatenate([span_steps(ns) for _ in range(Nsp)])
    M = bw.model_steps
    cd_phys = np.zeros(M)
    cd_phys[0] = steps_phys[0] / 2
    for i in range(1, M - 1):
        cd_phys[i] = (steps_phys[i - 1] + steps_phys[i]) / 2
    cd_phys[M - 1] = steps_phys[-1] / 2
    return (cd_phys / bw.cd_length).astype(np.float32)


class LessfmModel(torch.nn.Module):
    """L-ESSFM (free lengths + per-step NLPR) or ESSFM (uniform + rho + tied NLPR)."""

    def __init__(self, S, ns, device, tied_kerr=False, opt_rho=False):
        super().__init__()
        self.S = S
        self.bw = make_bw(S, ns)
        self.M = self.bw.model_steps
        self.nfl = S['nl_filter_length']
        self.ns = ns
        self.tied_kerr = tied_kerr
        self.opt_rho = opt_rho
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
            init = _exp_length_mult(S, self.bw, ns)   # L-ESSFM: free, exp-init
            self.length = torch.nn.Parameter(torch.tensor(init, device=device))

        # trainable NLPR filters (one-sided), init delta
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
        return self.length

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
        lm = self.length_mult()
        for NN in range(self.M):
            yc = torch.complex(y[..., 0], y[..., 1])
            phase = self.cd_base[NN] * lm[NN] / self.cd_alpha
            yc = torch.fft.ifft(torch.fft.fft(yc) * torch.exp(1j * phase))
            y = torch.stack([yc.real, yc.imag], -1)
            if NN < self.M - 1:
                ysq = (y ** 2).sum(-1)                       # |y|^2 per signal
                if pol_dim is not None:
                    ysq = ysq.sum(pol_dim, keepdim=True)     # total power over pols
                theta = torch.fft.irfft(torch.fft.rfft(ysq) * self._nlf_padded_fft(NN),
                                        n=self.Nsamp_d)
                c, s = torch.cos(theta), torch.sin(theta)
                y = torch.stack([y[..., 0] * c - y[..., 1] * s,
                                 y[..., 0] * s + y[..., 1] * c], -1)
        return torch.complex(y[..., 0], y[..., 1])
