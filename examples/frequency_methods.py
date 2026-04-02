"""
Instantaneous frequency estimation methods comparison.

Compares the 7 instantaneous frequency methods available in hhtpy
on a frequency-modulated signal, showing how each method handles
rapid frequency variations.

Methods:
  - Hilbert transform (analytic signal)
  - Quadrature (Huang's direct quadrature)
  - Zero crossing
  - Generalized zero crossing
  - Teager energy operator (TEO)
  - Hou (arccos-based)
  - Wu (quadrature + smoothing)
"""

import numpy as np
import matplotlib.pyplot as plt
from hhtpy import (
    decompose,
    calculate_instantaneous_frequency_hilbert,
    calculate_instantaneous_frequency_quadrature,
    calculate_instantaneous_frequency_zero_crossing,
    calculate_instantaneous_frequency_generalized_zero_crossing,
    calculate_instantaneous_frequency_teo,
    calculate_instantaneous_frequency_hou,
    calculate_instantaneous_frequency_wu,
)

plt.style.use("seaborn-v0_8")

# ── FM signal ────────────────────────────────────────────────────────
f_s = 1000  # Hz
T = 2  # seconds
t = np.arange(T * f_s) / f_s

# Frequency-modulated signal: carrier 50 Hz, FM deviation ±20 Hz at 2 Hz
f_mod = 2  # modulation frequency
f_carrier = 50
f_deviation = 20
true_freq = f_carrier + f_deviation * np.sin(2 * np.pi * f_mod * t)
phase = 2 * np.pi * np.cumsum(true_freq) / f_s
signal = np.cos(phase)

# ── Decompose to get a clean IMF ────────────────────────────────────
imfs, _ = decompose(signal, max_imfs=1)
imf = imfs[0]

# ── Compute IF with all methods ─────────────────────────────────────
methods = {
    "Hilbert": calculate_instantaneous_frequency_hilbert,
    "Quadrature": calculate_instantaneous_frequency_quadrature,
    "Zero crossing": calculate_instantaneous_frequency_zero_crossing,
    "Generalized ZC": calculate_instantaneous_frequency_generalized_zero_crossing,
    "TEO": calculate_instantaneous_frequency_teo,
    "Hou": calculate_instantaneous_frequency_hou,
    "Wu": calculate_instantaneous_frequency_wu,
}

results = {}
for name, func in methods.items():
    try:
        results[name] = func(imf, f_s)
    except Exception as e:
        print(f"  {name}: failed ({e})")

# ── Plot ──────────────────────────────────────────────────────────
n_methods = len(results)
fig, axes = plt.subplots(n_methods + 1, 1, figsize=(12, 2.5 * (n_methods + 1)),
                         sharex=True)

# Signal
axes[0].plot(t, imf, "k", lw=0.5)
axes[0].set_ylabel("IMF")
axes[0].set_title("FM Signal (carrier 50 Hz, deviation ±20 Hz)")

# Each method
for i, (name, freq) in enumerate(results.items()):
    ax = axes[i + 1]
    ax.plot(t, true_freq, "k--", lw=1, alpha=0.5, label="True frequency")
    ax.plot(t, freq, lw=1, label=name)
    ax.set_ylabel("Freq (Hz)")
    ax.set_ylim([0, 100])
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Instantaneous Frequency Methods Comparison",
             fontsize=14, fontweight="bold")
fig.tight_layout()
plt.show()
