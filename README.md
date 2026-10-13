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
One notebook each; configs in `config/scenario_<name>.ini` (n = 1.125) and `_n2.ini`.
1. `170km_93GBd`   — single span, 93 GBaud (the OFC/CLEO paper scenario)
2. `170km_186GBd`  — single span, 186 GBaud
3. `15x80km_93GBd` — 15-span long-haul, 93 GBaud
5. `170km_10GBd`   — single span, 10 GBaud
6. `15x80km_10GBd` — 15-span long-haul, 10 GBaud
7. `15x80km_32GBd` — 15-span long-haul, 32 GBaud

## Layout
```
functions/
  scenario.py     Scenario: forward, references, trained models, LDBP, saved results
  result_plots.py the performance figures (fig1b, fig_vs_steps)
  system.py       build_system, forward and backward propagation (numpy / CuPy)
  forward.py      cached forward of a scenario (forward/independent_ase_v1/)
  forward_sim.py, forward_torch.py   forward on the GPU (generation / training)
  backprop.py     SNR vs power of EDC, Ideal DBP, OSSFM, L-ESSFM, ESSFM
  train.py        training of L-ESSFM, ESSFM (PyTorch) and LDBP
  lessfm_torch.py, ldbp_torch.py, ldbp_eval.py   the trainable models / LDBP evaluation
  geometry.py     initial L-ESSFM step lengths (NLPR at the nonlinearity quantiles)
  complexity.py   RM/2D formulas (frequency domain; reproduces the published arrays)
  rrc.py, storage.py, lib/
config/      one .ini per scenario and oversampling
notebooks/   00 tutorial (SNR vs power), 01-07 one per scenario, 04 GVD-length figures
results/independent_ase_v1/<scenario>/   current results (independent ASE per polarization)
  <method>[_n2]_Ns<Ns>/parameters.csv     trained coefficients (+ manifest: what they were trained on)
  evaluations/<method>[_n2]_Ns<Ns>.json   SNR vs power
  summary.json, fig1b.*, fig_vs_steps.*
results/<scenario>/                       old results (same ASE on both polarizations)
```

## Key implementation notes (lessons from the reproduction)
- **Complexity is frequency-domain**: the NLPR filter length Nc does NOT enter
  ESSFM/L-ESSFM complexity (`functions/complexity.py`).  Same band rate R for both
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
Use the `l_essfm_torch` kernel. Saved models and SNR curves are loaded; missing ones
are trained or computed. `RETRAIN = True` retrains every model.
