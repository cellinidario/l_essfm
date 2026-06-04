# l_essfm — Learned-Enhanced Split-Step Fourier Method for Digital Backpropagation

Clean, reorganized codebase for the L-ESSFM / ESSFM / LDBP study (replaces the
`ldbp/` and `ldbp2/` sprawl). All code in English. Native ext4 (fast I/O).

## What it does
Trains and evaluates digital-backpropagation (DBP) algorithms for fiber
nonlinearity compensation, and reproduces the performance-vs-complexity figures:

- **L-ESSFM**: learns every GVD step length + a per-step NLPR (Kerr) filter.
- **ESSFM**: L-ESSFM constrained to uniform step lengths + one shared NLPR filter
  + a single optimized splitting ratio rho.
- **OSSFM / SSFM**: split-step with one (optionally optimized) nonlinear coeff.
- **LDBP** (Haeger): time-domain FIR DBP.
- **EDC**, **Ideal DBP** (SSFM with many steps): references.

Performance = effective SNR (effSNR, peak over transmit power).
Complexity = real multiplications per 2D symbol (RM/2D), frequency-domain.

## Scenarios
1. `170km_93GBd`   — single span, 93 GBaud (the OFC/CLEO paper scenario)
2. `170km_186GBd`  — single span, 186 GBaud
3. `15x80km_93GBd` — 15-span long-haul, 93 GBaud

## Layout
```
core/        reusable modules (English, self-tested)
  complexity.py   RM/2D formulas (eval_compl, freq domain; reproduces published arrays)
  rrc.py          root-raised-cosine filter (analytic frequency form, no 'delay')
  system.py       build_system / forward / backprop  (TODO)
  lessfm.py       L-ESSFM & ESSFM training (one parametric file)  (TODO)
  ldbp.py         LDBP training (FFT-conv)  (TODO)
  forward.py      generate + cache the received signal once  (TODO)
config/      one .ini per scenario
notebooks/   human-readable, runnable end-to-end (RETRAIN flag: load or train)
  00_tutorial_snr_vs_power.ipynb   SNR vs power + GVD-length / NLPR-filter plots
  01_170km_93GBd.ipynb             reproduce paper figures (this scenario)
  02_170km_186GBd.ipynb
  03_15x80km_93GBd.ipynb
results/     saved forwards + trained coefficients (so RETRAIN=False is instant)
```

## Key implementation notes (lessons from the reproduction)
- **Complexity is frequency-domain**: the NLPR filter length Nc does NOT enter
  ESSFM/L-ESSFM complexity (`core/complexity.py`).  Same band rate R for both
  oversamplings; n=2 costs ~2x n=1.125 at equal Ns.
- **ESSFM = L-ESSFM with**: tied NLPR filter (shared across steps) + uniform
  lengths + one optimized splitting ratio rho.  At Ns=1 the two MUST coincide.
- **ESSFM tied-filter scaling**: a single shared NLPR filter must be scaled per
  step by the step's nonlinear strength nl_param[NN] (signed mean as reference),
  otherwise it cannot match steps whose nl_param differ by up to ~1700x.
- **rho convention** (MATLAB/practice): first GVD border = rho*L, last =
  (1-rho)*L; rho ~ 0.9 for a single step.  (The paper writes it mirrored.)
- **Nc per Ns**: ESSFM uses essfmNc(Ns)+1 coeffs (shrinks with Ns); L-ESSFM keeps
  Nc=65 up to Ns=10 then follows the same formula (first step ~ full fiber).
- **RRC** in frequency (analytic), no time-domain truncation / 'delay'.
- **Forward steps** must be verified sufficient PER scenario (the only
  "legitimate" slowness; everything else is to be optimized away).

## Status
Reorganization in progress.  Core numeric results validated against the MATLAB
reference (effSNR within ~0.02-0.05 dB across Ns and both oversamplings).

## Running the notebooks
Open them from `notebooks/` (Jupyter) or anywhere — the first cell auto-detects
the project root and adds `core/` to the path, so imports work regardless of the
launch directory. `RETRAIN=False` loads the shipped coefficients (instant);
`RETRAIN=True` retrains from scratch.
