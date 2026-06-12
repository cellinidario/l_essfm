"""forward_torch.py -- GPU batched forward propagation (split-step) in PyTorch.

Ports system.forward to torch so training data is generated ON the GPU, in large
batches, instead of the CPU numpy loop (the shared data-gen bottleneck the
benchmark exposed). Same physics: pulse shaping, WDM comb, per-step CD + Kerr,
ASE noise, low-pass, downsample to the digital rate, return (y[B,Nd,2], x[B,Nsym]).

Validated to match system.forward statistically (same SSFM, same RNG-free physics;
the only difference is the random symbols/noise draw, which is what we want for
fresh training data each call)."""
import numpy as np
import torch


class TorchForward:
    """Batched GPU forward. Precomputes all fixed filters/phases as torch tensors."""

    def __init__(self, S, device='cuda'):
        self.S = S
        self.dev = device
        self.Npol = S['Npol']; self.Nch = S['Nch']; self.Nsym = S['Nsym']
        self.OS_a = S['OS_a']; self.OS_d = S['OS_d']
        self.Nsamp_a = S['Nsamp_a']; self.Nsamp_d = S['Nsamp_d']
        self.spacing = S['spacing']; self.fsamp_a = S['fsamp_a']
        self.Nsp = S['Nsp']
        self.sigma2 = S['sigma2']
        fw = S['fw']; self.M = fw.model_steps
        t = lambda a, dt=torch.complex64: torch.tensor(np.asarray(a), dtype=dt, device=device)
        self.ps_tx = t(S['ps_tx_freq'])                       # [Nsamp_a]
        self.lp = t(S['lp_freq'])
        # per-step CD phase exp(j*cd) and Kerr strength
        self.cd_exp = torch.stack([t(np.exp(1j * fw.get_cd_filter_freq(MM)))
                                   for MM in range(self.M)])    # [M, Nsamp_a]
        self.nlp = torch.tensor([float(fw.nl_param[MM]) for MM in range(self.M)],
                                dtype=torch.float32, device=device)  # [M]
        # WDM carrier phases per channel
        nvec = torch.arange(self.Nsamp_a, device=device)
        self.carrier = torch.stack([
            torch.exp(1j * 2 * np.pi * ((NN - self.Nch // 2) * self.spacing)
                      * nvec / self.fsamp_a).to(torch.complex64)
            for NN in range(self.Nch)])                         # [Nch, Nsamp_a]
        self.const = t(S['const']) if S['modulation'] == 'QAM' else None
        self.modulation = S['modulation']

    @torch.no_grad()
    def __call__(self, B, P_W):
        """Generate B independent blocks at launch power(s) P_W (scalar or [B] tensor).
        Returns y[B, Nsamp_d, 2] (real/imag) and x[B, Nsym] complex (center channel),
        flattened so each polarization is one example (matches the trainer)."""
        dev = self.dev
        Npol, Nch, Nsym, OS_a = self.Npol, self.Nch, self.Nsym, self.OS_a
        if not torch.is_tensor(P_W):
            P_W = torch.full((B,), float(P_W), device=dev)
        P_W = P_W.reshape(B, 1, 1, 1)
        # symbols [B, Npol, Nch, Nsym]
        if self.modulation == 'QAM':
            idx = torch.randint(0, self.const.shape[0], (B, Npol, Nch, Nsym), device=dev)
            x = self.const[idx]
        else:
            x = (torch.randn(B, Npol, Nch, Nsym, device=dev)
                 + 1j * torch.randn(B, Npol, Nch, Nsym, device=dev)) / np.sqrt(2)
            x = x.to(torch.complex64)
        x_up = torch.zeros(B, Npol, Nch, self.Nsamp_a, dtype=torch.complex64, device=dev)
        x_up[:, :, :, ::OS_a] = x * np.sqrt(OS_a)
        u = torch.fft.ifft(torch.fft.fft(x_up) * self.ps_tx) * torch.sqrt(P_W / Npol)
        # WDM comb -> [B, Npol, Nsamp_a]
        u_wdm = (u * self.carrier).sum(2)
        for _ in range(self.Nsp):
            u_wdm = u_wdm + np.sqrt(self.sigma2 / 2) * (
                torch.randn(B, 1, self.Nsamp_a, device=dev)
                + 1j * torch.randn(B, 1, self.Nsamp_a, device=dev)).to(torch.complex64)
            for MM in range(self.M):
                u_wdm = torch.fft.ifft(torch.fft.fft(u_wdm) * self.cd_exp[MM])
                pw = (u_wdm.abs() ** 2).sum(1, keepdim=True)        # total power over pols
                u_wdm = u_wdm * torch.exp(1j * (self.nlp[MM] * pw))
        y = torch.fft.ifft(torch.fft.fft(u_wdm) * self.lp)         # [B, Npol, Nsamp_a]
        # downsample to digital rate (overlap-add of the two band halves)
        Y = torch.fft.fft(y)
        Y = Y[:, :, :self.Nsamp_d] + Y[:, :, -self.Nsamp_d:]
        y = torch.fft.ifft(Y) / OS_a * self.OS_d                    # [B, Npol, Nsamp_d]
        xc = x[:, :, Nch // 2, :]                                   # center channel [B,Npol,Nsym]
        # flatten pols into batch (each pol = one single-pol training example)
        yb = y.permute(0, 1, 2, ).reshape(B * Npol, self.Nsamp_d)
        yb = torch.stack([yb.real, yb.imag], -1)                   # [B*Npol, Nd, 2]
        xb = xc.reshape(B * Npol, Nsym)
        return yb, xb
