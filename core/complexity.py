"""
complexity.py -- Computational complexity of DBP algorithms in real
multiplications per 2D symbol (RM/2D).

Formulas from S. Civelli et al., "A New Twist on Low-Complexity Digital
Backpropagation," JLT 2025 (Sec. III-D, Eqs. 29-32), and the MATLAB reference
`complessitaRM_CBESSFM_dario.m`.

Key facts (verified against the published OFC arrays, digit-for-digit):
- ESSFM / L-ESSFM are computed in the FREQUENCY domain, so the NLPR filter
  length Nc does NOT enter the complexity (an FFT + per-frequency multiply costs
  the same regardless of how many filter taps it represents).  Use `eval_compl`.
- The TIME-domain variant (`eval_compl_time`, with the 11+Nc term) is only for
  OSSFM-in-time / ESSFM-in-time comparisons, NOT for the freq-domain curves.
- Same band rate R for both oversamplings: n is samples/symbol, so n=2 costs
  ~2x n=1.125 at equal Ns (sanity check: 170km/93GBd Ns=1 -> 61.4 RM at
  n=1.125, 125.4 RM at n=2).
- N (overlap-and-save block length) is optimized: the minimum over a small grid
  of powers of two is taken.  With a long enough block, overlap-and-save is
  equivalent to plain block processing (which is what the code actually does).

B2 = |beta2| ~ 2.17e-26 s^2/m for D = 17 ps/nm/km.
"""

import numpy as np

B2 = 2.17e-26  # |beta2| [s^2/m], D = 17 ps/nm/km


def _block_grid(L, R, nos):
    """Overlap (Nov) and the optimized-N grid for overlap-and-save."""
    Nh = 2 * np.pi * B2 * L * nos**2 * R**2
    No = np.ceil(Nh) + 10                       # overlap Nov
    N = 2.0**(np.ceil(np.log2(No)) + np.arange(0, 11))
    eta = N / (N - No)
    return N, eta


def eval_compl(Nst, nos, L, R, Nsb=1):
    """ESSFM / L-ESSFM / CB-ESSFM complexity in RM/2D (frequency domain).

    Nst : number of DBP steps
    nos : oversampling (samples/symbol), e.g. 1.125 or 2
    L   : total link length [m]
    R   : symbol (band) rate [Hz]
    Nsb : number of subbands (1 = ESSFM/L-ESSFM, >1 = CB-ESSFM).  Nc absent.
    """
    N, eta = _block_grid(L, R, nos)
    s1 = (5 * Nst + 4) * np.log2(N / Nsb)
    s2 = Nst * (3 * Nsb + 1) / 2
    s3 = 4 * np.log2(Nsb) - 6
    s4 = (20 * Nsb * Nst + 16) / N
    return float(np.min(nos / 2 * eta * (s1 + s2 + s3 + s4)))


def eval_compl_time(Nst, nos, L, R, Nc=0):
    """Time-domain complexity (OSSFM-time, ESSFM-time).  Here Nc DOES enter."""
    N, eta = _block_grid(L, R, nos)
    s1 = (Nst + 1) * (4 * np.log2(N) - 6 + 16 / N)
    s2 = Nst * (11 + Nc)
    return float(np.min(nos / 2 * eta * (s1 + s2)))


def eval_compl_edc(nos, L, R):
    """EDC (GVD only): one linear CD filter, no NL steps (Nst=0)."""
    return eval_compl(0, nos, L, R, Nsb=1)


def complexity_ldbp(Nst, Nc, nos=2):
    """LDBP (Haeger, time domain): C = nos/2 * ((6*Nc+11)*Nst + 6) RM/2D.

    Nc = (FIR taps - 1)/2 per step.  The paper uses nos=2.  For variable
    per-step taps, pass the AVERAGE Nc over the (periodic) tap pattern.
    """
    return nos / 2 * ((6 * Nc + 11) * Nst + 6)


if __name__ == "__main__":
    # self-check vs published OFC arrays (170 km, 93 GBaud)
    L, R = 170e3, 93e9
    pub_n1125 = {1: 61.4, 50: 1865.2}
    pub_n2 = {1: 125.4, 50: 3769.8}
    for ns, v in pub_n1125.items():
        c = eval_compl(ns, 1.125, L, R)
        print(f"n=1.125 Ns={ns}: {c:.1f} (published {v})  {'OK' if abs(c-v) < 0.5 else 'MISMATCH'}")
    for ns, v in pub_n2.items():
        c = eval_compl(ns, 2.0, L, R)
        print(f"n=2     Ns={ns}: {c:.1f} (published {v})  {'OK' if abs(c-v) < 0.5 else 'MISMATCH'}")
