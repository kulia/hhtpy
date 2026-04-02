# hhtpy Roadmap

Features to implement to make hhtpy the most complete HHT library available.

## Done

- [x] Parallelize EEMD/CEEMDAN (`concurrent.futures`, configurable `n_jobs`)
- [x] Wu-Huang (2004) statistical significance test for IMFs
- [x] Configurable spline methods (PChip, Akima) and boundary padding (mirror, none)
- [x] Cycle analysis — detect cycles, characterize amplitude/frequency/duration/waveform shape, phase alignment

## Planned

- [ ] Holospectrum / second-layer sift — decompose amplitude envelopes to build 3D energy distribution (carrier freq × AM freq × time)
- [ ] Iterated mask sift — automated mask frequency refinement (cf. Quinn's `emd`)
- [ ] VMD (Variational Mode Decomposition) — optimization-based alternative to EMD
- [ ] Energy-based stopping criterion
- [ ] Parabolic extrema detection — sub-sample precision for extrema locations

## Context

No other Python library has all of these. Current landscape:

| Feature | hhtpy | PyEMD | emd (Quinn) | libeemd |
|---------|-------|-------|-------------|---------|
| EMD / EEMD / CEEMDAN | Yes | Yes | Yes | Yes |
| MEMD | **Yes** | No | No | No |
| Masked EMD (3 init strategies) | **Yes** | No | Partial | No |
| 7 IF methods | **Yes** | 1 | 1 | 0 |
| Frequency despiking | **Yes** | No | No | No |
| Index of orthogonality | **Yes** | No | No | No |
| Parallel ensemble methods | **Yes** | Yes | Yes | Yes (OpenMP) |
| Wu-Huang significance test | **Yes** | No | No | No |
| Configurable splines/padding | **Yes** | Yes | Yes | Partial |
| Holospectrum | Planned | No | Yes | No |
| Cycle analysis | **Yes** | No | Yes | No |
| VMD | Planned | No | No | No |
