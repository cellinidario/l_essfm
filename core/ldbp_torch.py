"""ldbp_torch.py -- LDBP (Hager neural-DBP) trainable model in PyTorch.

Per step: complex symmetric CD-FIR (truncated, the "pruning" tap count) applied in
frequency, then a SCALAR Kerr phase using TOTAL dual-pol power. Matches the numpy
reference (ldbp_eval.ldbp_backprop) which reproduces the published curve exactly.

Trainable params per step:
  - CD-FIR: complex symmetric -> only the right half (L+1) real + (L+1) imag taps are
    free; the model mirrors them to the full odd length 2L+1 (matches TF1
    tf_complex_symmetric_filter).
  - nl: one scalar Kerr coefficient.

LDBP is a TIME-domain method -> runs at n>=2 (digital oversampling 2).
The model forward expects the dual-pol signal jointly ([B,Npol,N,2]) so the Kerr
phase sees total power (the physically-correct, dual-pol objective).
"""
import numpy as np
import torch


_FIR_OBJ = {}


def _fir_obj(S, method='LS-CO', bandwidth=1.0, oob_gain=1.05):
    """Cached lib.fir.cd_fir_filter for the CD-FIR INIT design. Uses FULL bandwidth
    (1.0): the LS-CO 0.6-bandwidth init optimizes only the central 60% of the band,
    so the filter is wrong outside it -> with many steps (multi-span) the error
    accumulates and training diverges (-385 dB). bandwidth=1.0 gives the ideal CD
    filter (matches the truncated ideal IDFT), which inverts CD correctly and lets
    training converge. (Dario's ldbp2.py used 0.6 + training to recover; on long
    links the init must already be good.)"""
    key = (S['Nsamp_d'], method, bandwidth, oob_gain)
    if key not in _FIR_OBJ:
        from lib import fir
        _FIR_OBJ[key] = fir.cd_fir_filter({
            'beta2': S['beta2'], 'fsamp': S['fsamp_d'], 'Nsamp': S['Nsamp_d'],
            'method': method, 'bandwidth': bandwidth, 'max_out_of_band_gain': oob_gain})
    return _FIR_OBJ[key]


def cd_fir_init(S, bw, NN, cd_len):
    """Initial complex CD-FIR coefficients for step NN, length cd_len (odd), via the
    least-squares (LS-CO) design from lib.fir -- the EXACT method Dario's ldbp2.py
    uses (fir_obj.get_filter(bw.cd_length[NN], cd_len)). The earlier truncated-IDFT
    init was wrong (didn't invert CD on few taps -> diverged). Returns complex [cd_len]."""
    return _fir_obj(S).get_filter(bw.cd_length[NN], int(cd_len)).astype(np.complex64)


class LdbpModel(torch.nn.Module):
    """Trainable LDBP. cd_lengths: list of per-step CD-FIR tap counts (the pruning
    pattern, e.g. all 11 for 'p11', or a variable pattern for 'aligned')."""

    def __init__(self, S, ns, device, cd_lengths, init_cd=None, init_nl=None, total_steps=None):
        super().__init__()
        from system import make_bw
        self.S = S
        self.dev = device
        # LDBP step convention: ns = steps PER SPAN (total_steps=False) by default,
        # like Hager's LDBP. model_steps = StPS*Nsp+1.
        self.M = ns + 1 if total_steps else S['Nsp'] * ns + 1
        self.Nsamp_d = S['Nsamp_d']
        self.OS = S['OS_d']  # TorchForward always returns downsampled signal at OS_d
        self.fsamp = self.OS * S['fsym']
        S_bw = dict(S); S_bw['fsamp'] = self.fsamp; S_bw['Nsamp'] = self.Nsamp_d
        self.bw = make_bw(S_bw, ns, total_steps=total_steps)
        self.cd_lengths = cd_lengths if cd_lengths else [21] * self.M
        self.delays = [(L - 1) // 2 for L in self.cd_lengths]

        # trainable CD-FIR: store the RIGHT HALF (delay+1) real & imag taps per step;
        # mirror to full length at apply (symmetric complex filter).
        self.cd_re = torch.nn.ParameterList()
        self.cd_im = torch.nn.ParameterList()
        # Kerr is NOT trained in Hager's LDBP -- it is FIXED to the physical per-step
        # nonlinear phase bw.nl_param (only the CD-FIR filters are learned). Register
        # as a non-trainable buffer. (Training it from 0 left it ~noise -> ~EDC.)
        self.register_buffer('nl', torch.tensor(np.asarray(self.bw.nl_param[:self.M]),
                                                 dtype=torch.float32, device=device))
        # create trainable parameters for the complex right-half CD-FIR
        # pruning masks on the right-half taps (1=active, 0=pruned). Start all-1 (full
        # init length cd_lengths[NN]); training shrinks them toward target_taps via
        # prune_to(). Like ldbp_diag.py: init abundant -> gradual prune to a sawtooth.
        self.masks = []
        for NN in range(self.M):
            if init_cd is not None:
                c = init_cd[NN]
            else:
                c = cd_fir_init(S, self.bw, NN, self.cd_lengths[NN])
            d = self.delays[NN]
            rh_re = torch.tensor(np.real(c)[d:].copy(), dtype=torch.float32, device=device)
            rh_im = torch.tensor(np.imag(c)[d:].copy(), dtype=torch.float32, device=device)
            self.cd_re.append(torch.nn.Parameter(rh_re))
            self.cd_im.append(torch.nn.Parameter(rh_im))
            self.masks.append(torch.ones(d + 1, device=device))   # [delay+1], non-trainable
        if init_nl is not None:                                    # eval override only
            with torch.no_grad():
                self.nl.copy_(torch.tensor(np.asarray(init_nl), dtype=torch.float32, device=device))

        self.ps_rx = torch.tensor(S['ps_rx_freq'].astype(np.complex64), device=device)

    def prune_to(self, target_taps):
        """Set each step's mask so the active filter length is target_taps[NN] (odd).
        Keeps the central target//2+1 right-half taps, zeros the rest -> the filter is
        effectively a shorter symmetric FIR. target_taps: list (sawtooth) per step."""
        for NN in range(self.M):
            keep = (int(target_taps[NN]) - 1) // 2 + 1     # right-half active count
            m = torch.zeros_like(self.masks[NN])
            m[:keep] = 1.0
            self.masks[NN] = m

    def prune_one(self, NN):
        """Remove ONE outermost active tap from step NN (right-half mask). Mirrors
        ldbp_diag.py get_prune_op: gradual, one tap per event -> the model adapts
        between prunes (pruning the whole model at once shocks it to 0)."""
        m = self.masks[NN]
        active = int(m.sum().item())
        if active > 1:
            m[active - 1] = 0.0

    def current_lengths(self):
        """Active filter length per step (odd): 2*active_right_half - 1."""
        return [2 * int(self.masks[NN].sum().item()) - 1 for NN in range(self.M)]

    def _cd_freq(self, NN):
        """Full symmetric complex CD-FIR for step NN, zero-padded + rolled, as rfft? no:
        return the full-length complex frequency response (fft of the padded time filter)."""
        mask = self.masks[NN]                                    # [delay+1], pruning
        rh_re = self.cd_re[NN] * mask
        rh_im = self.cd_im[NN] * mask
        # mirror right-half -> full symmetric [2*delay+1]
        full_re = torch.cat([rh_re[1:].flip(0), rh_re])
        full_im = torch.cat([rh_im[1:].flip(0), rh_im])
        cd_full = torch.complex(full_re, full_im)               # [cd_len]
        cd_len = cd_full.shape[0]
        pad = torch.zeros(self.Nsamp_d, dtype=cd_full.dtype, device=self.dev)
        pad[:cd_len] = cd_full
        pad = torch.roll(pad, -(cd_len // 2))
        return torch.fft.fft(pad)                               # [N] complex

    def forward(self, y):
        """y: [B, Npol, N, 2] (dual-pol jointly). Returns equalized complex [B,Npol,N].
        single_pol=True -> Kerr uses per-polarization power (NOT the dual-pol sum), to
        match the TF1 single-pol training; the eval (ldbp_eval) uses total power."""
        yc = torch.complex(y[..., 0], y[..., 1])                # [B,Npol,N]
        for NN in range(self.M):
            yc = torch.fft.ifft(torch.fft.fft(yc) * self._cd_freq(NN))   # CD-FIR
            ysq = yc.real ** 2 + yc.imag ** 2
            yc = yc * torch.exp(1j * (ysq * self.nl[NN]))               # scalar Kerr
        return yc
