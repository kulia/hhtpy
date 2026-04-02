"""
Masked EMD example: targeted mode separation.

Demonstrates how masked EMD uses a known sinusoidal mask to force
separation at a specific frequency, avoiding the mode mixing that
occurs with standard EMD on closely-spaced components.

Reference:
    Deering, R. & Kaiser, J.F. (2005). "The use of a masking signal
    to improve empirical mode decomposition." ICASSP 2005.
"""

import numpy as np
import matplotlib.pyplot as plt
from hhtpy import decompose, masked_decompose, adaptive_masked_decompose

plt.style.use("seaborn-v0_8")

# ── Signal: two close tones (hard for EMD) ──────────────────────────
f_s = 1000  # Hz
T = 3  # seconds
t = np.arange(T * f_s) / f_s

f1, f2 = 30, 40  # close frequencies → mode mixing in standard EMD
signal = np.cos(2 * np.pi * f1 * t) + 0.8 * np.cos(2 * np.pi * f2 * t)

# ── Standard EMD ────────────────────────────────────────────────────
imfs_emd, res_emd = decompose(signal)

# ── Masked EMD (explicit mask at 40 Hz) ─────────────────────────────
imfs_masked, res_masked = masked_decompose(
    signal,
    mask_frequency=45,
    mask_amplitude=2.0,
    sampling_frequency=f_s,
    max_imfs=3,
)

# ── Adaptive masked EMD ────────────────────────────────────────────
imfs_adaptive, res_adaptive = adaptive_masked_decompose(
    signal,
    sampling_frequency=f_s,
    max_imfs=3,
)

# ── Plot comparison ─────────────────────────────────────────────────
methods = {
    "EMD": (imfs_emd, res_emd),
    "Masked EMD\n(mask @ 45 Hz)": (imfs_masked, res_masked),
    "Adaptive\nMasked EMD": (imfs_adaptive, res_adaptive),
}

n_rows = 4  # signal + 3 IMFs
fig, axes = plt.subplots(n_rows, len(methods), figsize=(14, 8),
                         sharex=True, sharey="row")

for col, (name, (imfs, res)) in enumerate(methods.items()):
    # Plot signal
    axes[0, col].plot(t, signal, "k", lw=0.5)
    axes[0, col].set_title(name, fontsize=11, fontweight="bold")
    if col == 0:
        axes[0, col].set_ylabel("Signal")

    # Plot IMFs
    for row in range(1, n_rows):
        idx = row - 1
        if idx < len(imfs):
            axes[row, col].plot(t, imfs[idx], lw=0.5)
        if col == 0:
            axes[row, col].set_ylabel(f"IMF {idx + 1}")

axes[-1, 1].set_xlabel("Time (s)")

# Zoom to show separation quality
for ax_row in axes:
    for ax in ax_row:
        ax.set_xlim([1.0, 1.3])

fig.suptitle(
    f"Mode Separation: {f1} Hz + {f2} Hz tones",
    fontsize=14, fontweight="bold",
)
fig.tight_layout()

# Print component correlations
print("Correlation of IMF 1 with 40 Hz component:")
ref_40 = 0.8 * np.cos(2 * np.pi * f2 * t)
for name, (imfs, _) in methods.items():
    if len(imfs) > 0:
        corr = np.abs(np.corrcoef(imfs[0], ref_40)[0, 1])
        print(f"  {name.replace(chr(10), ' '):30s}: {corr:.3f}")

plt.show()
