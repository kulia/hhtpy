# hhtpy

A Python implementation of the Hilbert-Huang Transform (HHT), including Empirical Mode Decomposition (EMD), instantaneous frequency estimation, and Hilbert spectral analysis for nonlinear and non-stationary time series.

This library was written by **Lars Havstad** and **Geir Kulia**.

## Quick Start

```python
import numpy as np
from hhtpy import hilbert_huang_transform
from hhtpy.plot import plot_imfs, plot_hilbert_spectrum, plot_marginal_hilbert_spectrum

T = 5  # sec
f_s = 15000  # Hz
t = np.arange(T * f_s) / f_s

y = np.cos(2 * np.pi * 50 * t + 20 * np.sin(2 * np.pi * 0.5 * t)) + 2 * np.cos(
    2 * np.pi * 20 * t
)

imfs, residue = hilbert_huang_transform(y, f_s)
```

## Plotting

### IMFs

```python
fig, axs = plot_imfs(imfs, y, residue, t, max_number_of_imfs=2)
```

![Plot of IMFs](figs/imfs.png)

### Hilbert Spectrum

```python
from hhtpy.plot import HilbertSpectrumConfig

fig, ax, clb = plot_hilbert_spectrum(
    imfs,
    config=HilbertSpectrumConfig(max_number_of_imfs=2),
)
```

![Plot Hilbert Spectrum](figs/hilbert_spectrum.png)

### Marginal Hilbert Spectrum

```python
fig, ax = plot_marginal_hilbert_spectrum(imfs)
```

![Plot marginal Hilbert spectrum](figs/marginal_hilbert_spectrum.png)

## EMD Decomposition

The `decompose()` function extracts Intrinsic Mode Functions (IMFs) from a signal using the sifting process.

```python
from hhtpy import decompose

imfs, residue = decompose(signal)

# Reconstruction is always exact:
# np.sum(imfs, axis=0) + residue == signal
```

### Parameters

```python
imfs, residue = decompose(
    signal,                   # 1D numpy array
    stopping_criterion=...,   # Controls when sifting stops (see below)
    max_imfs=None,            # Limit number of IMFs (None = automatic)
    max_sifts=100,            # Safety limit per IMF to prevent non-convergence
)
```

- **max_imfs**: Set this to extract only the first N IMFs. Useful when you know
  how many components your signal has, or for faster computation.
- **max_sifts**: Safety valve that stops sifting after 100 iterations even if the
  stopping criterion hasn't been met. Only relevant for adaptive criteria.

## Stopping Criteria

The sifting process needs a rule to decide when an IMF is "good enough". hhtpy provides four built-in criteria. All are passed to `decompose()` via the `stopping_criterion` parameter.

### Fixed Number of Sifts (Default)

The simplest approach: sift exactly N times. Huang (2015) recommended 10-15 sifts as a practical default.

```python
from hhtpy import get_stopping_criterion_fixed_number_of_sifts

# Default in decompose() is 15 sifts
criterion = get_stopping_criterion_fixed_number_of_sifts(10)
imfs, residue = decompose(signal, stopping_criterion=criterion)
```

### S-Number (Huang et al., 2003)

Counts consecutive sifts where the number of extrema and zero-crossings stays the same. Stops when this count reaches S. This detects when the sifting has converged in terms of the signal's oscillatory structure.

```python
from hhtpy import get_stopping_criterion_s_number

criterion = get_stopping_criterion_s_number(s_number=5)
imfs, residue = decompose(signal, stopping_criterion=criterion)
```

### Cauchy Convergence

Stops when the relative energy change between consecutive sifts falls below a threshold. Measures how much the sifting is still modifying the signal.

```python
from hhtpy import get_stopping_criterion_cauchy

criterion = get_stopping_criterion_cauchy(threshold=0.3)
imfs, residue = decompose(signal, stopping_criterion=criterion)
```

### Rilling–Flandrin–Gonçalves (2003)

Evaluates IMF quality by comparing the mean envelope to the amplitude envelope at each sample. Unlike Cauchy (which measures convergence rate), this directly measures whether the current mode satisfies the IMF property of having a near-zero mean envelope.

Two conditions must both be met:
1. At most `alpha` fraction of samples have `|mean_envelope| / amplitude > threshold_1`
2. No sample has `|mean_envelope| / amplitude > threshold_2`

```python
from hhtpy import get_stopping_criterion_rilling

# Default parameters from the original paper
criterion = get_stopping_criterion_rilling(
    threshold_1=0.05,   # 5% tolerance for most samples
    threshold_2=0.5,    # 50% hard ceiling for any sample
    alpha=0.05,         # Allow 5% of samples to exceed threshold_1
)
imfs, residue = decompose(signal, stopping_criterion=criterion)
```

### Custom Stopping Criterion

Any function matching the signature `(mode: np.ndarray, total_sifts_performed: int) -> bool` works:

```python
def my_criterion(mode, total_sifts_performed):
    if total_sifts_performed == 0:
        return False  # Always do at least one sift
    # Your logic here
    return total_sifts_performed >= 20

imfs, residue = decompose(signal, stopping_criterion=my_criterion)
```

## Instantaneous Frequency Methods

After decomposition, hhtpy computes instantaneous frequency and amplitude for each IMF. Two methods are available:

### Quadrature Method (Default)

The direct quadrature method normalizes the IMF and computes the analytic signal as `z(t) = x(t) + i·q(t)` where `q(t) = sign(dx/dt) · sqrt(1 - x²)`. This avoids limitations of the Hilbert transform (Bedrosian theorem) for wideband signals.

```python
from hhtpy import hilbert_huang_transform

imfs, residue = hilbert_huang_transform(signal, sampling_frequency)
# Each IMF object has: .signal, .instantaneous_frequency, .instantaneous_amplitude
```

### Hilbert Transform

The standard approach using `scipy.signal.hilbert` to compute the analytic signal, then deriving instantaneous frequency from the unwrapped phase gradient.

```python
from hhtpy import hilbert_huang_transform, calculate_instantaneous_frequency_hilbert

imfs, residue = hilbert_huang_transform(
    signal,
    sampling_frequency,
    frequency_calculation_method=calculate_instantaneous_frequency_hilbert,
)
```

## Quality Diagnostics

### Index of Orthogonality

Measures how orthogonal the extracted IMFs are to each other. A good decomposition produces nearly orthogonal IMFs (IO close to 0). High values suggest mode mixing or energy leakage between IMFs.

```python
from hhtpy import decompose, index_of_orthogonality

imfs, residue = decompose(signal)
io = index_of_orthogonality(imfs)
print(f"Index of orthogonality: {io:.4f}")  # Lower is better
```

## Ensemble EMD (EEMD)

Standard EMD can suffer from *mode mixing* — when oscillatory components of different scales end up in the same IMF. EEMD (Wu & Huang, 2009) mitigates this by adding white Gaussian noise over multiple trials and averaging the resulting IMFs. The noise populates the time-frequency space uniformly, guiding the sifting process to separate scales consistently.

```python
from hhtpy import eemd

imfs, residue = eemd(
    signal,
    num_trials=100,        # Number of noise-perturbed decompositions
    noise_amplitude=0.2,   # Noise std as fraction of signal std (20%)
    seed=42,               # For reproducibility
)
```

**Note:** EEMD does not guarantee exact reconstruction. The residual noise decreases as `noise_amplitude / sqrt(num_trials)` but never reaches zero. For exact reconstruction, use CEEMDAN.

## CEEMDAN

Complete Ensemble EMD with Adaptive Noise (Torres et al., 2011) improves on EEMD in two ways:

1. **Exact reconstruction** — `sum(imfs) + residue == signal` is guaranteed by construction.
2. **Adaptive noise** — noise is added at each decomposition stage (not just to the original signal), keeping the signal-to-noise ratio constant across all stages.

At each stage *k*, the noise contribution is the *k*-th IMF of the original noise realization, scaled to the current residue's standard deviation.

```python
from hhtpy import ceemdan

imfs, residue = ceemdan(
    signal,
    num_trials=100,        # Number of ensemble trials
    noise_amplitude=0.2,   # Noise scale factor (fraction of residue std)
    seed=42,               # For reproducibility
)

# Exact reconstruction is guaranteed:
# np.sum(imfs, axis=0) + residue == signal
```

See `example_eemd.py` for a side-by-side comparison of EMD, EEMD, and CEEMDAN on a signal with intermittent high-frequency bursts.

## Multivariate EMD (MEMD)

Multivariate EMD (Rehman & Mandic, 2010) extends EMD to multi-channel signals. It computes envelopes by projecting the signal onto uniformly distributed direction vectors on the unit hypersphere (generated via the Hammersley quasi-random sequence), then averages the back-projected mean envelopes.

The key advantage over applying standard EMD to each channel independently: MEMD **aligns** common oscillatory scales across all channels, ensuring shared modes appear at the same IMF index.

```python
from hhtpy import memd

# signal shape: (n_channels, n_samples)
signal = np.array([ch1, ch2, ch3])

imfs, residue = memd(
    signal,
    num_directions=64,     # Direction vectors on the unit hypersphere
    max_imfs=None,         # None = automatic
    max_sifts=100,         # Safety limit per IMF
    stop_threshold=0.075,  # Normalized mean envelope threshold
)

# imfs shape: (n_imfs, n_channels, n_samples)
# Exact reconstruction: np.sum(imfs, axis=0) + residue == signal
```

- **num_directions** must be >= 2 × n_channels. Higher values give better envelope estimates at the cost of computation. Default is 64.
- The input shape is `(n_channels, n_samples)` — channels first.

See `example_memd.py` for a complete example with a two-channel signal.

## Mode Mixing / Separation Analysis

EMD can suffer from *mode mixing* — when two frequency components end up in the same IMF instead of being separated. Whether the EMD resolves two tones or treats them as a single modulated component depends on their amplitude and frequency ratios, as analyzed by [Rilling & Flandrin (2008)](https://doi.org/10.1109/TSP.2007.906771).

The plot below maps the separation boundary: dark regions indicate successful separation, light regions indicate mode mixing.

![EMD separation performance](figs/figure3_rilling_flandrin.png)

See `emd_separation_analysis.py` to reproduce this analysis.

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `decompose(signal, ...)` | EMD decomposition into IMFs + residue |
| `hilbert_huang_transform(signal, fs, ...)` | Full HHT: decompose + instantaneous frequency/amplitude |
| `marginal_hilbert_spectrum(imfs)` | Frequency-domain amplitude integration |
| `index_of_orthogonality(imfs)` | Decomposition quality metric |
| `eemd(signal, ...)` | Ensemble EMD — noise-assisted decomposition |
| `ceemdan(signal, ...)` | Complete EEMD with Adaptive Noise |
| `memd(signal, ...)` | Multivariate EMD for multi-channel signals |

### Stopping Criteria

| Function | Description |
|----------|-------------|
| `get_stopping_criterion_fixed_number_of_sifts(n)` | Stop after exactly n sifts |
| `get_stopping_criterion_s_number(s)` | Stop when extrema/zero-crossings stabilize |
| `get_stopping_criterion_cauchy(threshold)` | Stop when energy change is small |
| `get_stopping_criterion_rilling(t1, t2, alpha)` | Stop when envelope symmetry is good |

### Frequency Methods

| Function | Description |
|----------|-------------|
| `calculate_instantaneous_frequency_quadrature` | Direct quadrature (default) |
| `calculate_instantaneous_frequency_hilbert` | Via scipy Hilbert transform |

### Plotting

| Function | Description |
|----------|-------------|
| `plot_imfs(imfs, signal, residue, x_axis)` | Time-domain IMF subplots |
| `plot_hilbert_spectrum(imfs, config)` | Time-frequency Hilbert spectrum |
| `plot_marginal_hilbert_spectrum(imfs)` | Frequency-domain marginal spectrum |

## Acknowledgements

We want to express our sincere gratitude to the following individuals for their invaluable contributions and support throughout this project:

- **Professor Norden Huang**: For his extensive one-on-one lectures over ten days, during which he taught us the Hilbert-Huang Transform (HHT) and guided us through the nuances of implementing it. Many of the insights and implementation techniques used in this project directly result from these invaluable sessions.

- **Professor Marta Molinas**: To introduce us to the HHT methodology, provide foundational knowledge, and engage in valuable discussions about the implementation. Her guidance has been instrumental in shaping our understanding and approach.

- **Professor Olav B. Fosso**: For his numerous fruitful dialogues on improving and optimizing the algorithm. His insights have greatly influenced the refinement of our implementation.

- **Sumit Kumar Ram (@sumitram)**: For explaining the HHT algorithm to me for the first time. His clear and concise explanation provided the initial spark that fueled our deeper exploration of the method.

Thank you all for your expertise, time, and mentorship, which made this work possible.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please open an issue or submit a pull request on the GitHub repository.
