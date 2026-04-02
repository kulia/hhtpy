"""
Multivariate EMD (MEMD) example: aligned decomposition of multi-channel signals.

Demonstrates how MEMD produces aligned IMFs across channels, ensuring
shared oscillatory modes are captured at the same IMF index — unlike
applying standard EMD to each channel independently.
"""

import numpy as np
import matplotlib.pyplot as plt
from hhtpy import memd, decompose

plt.style.use("seaborn-v0_8")

# ── Two-channel signal with shared modes ─────────────────────────────────
f_s = 500  # Hz
T = 2  # seconds
t = np.arange(T * f_s) / f_s

# Both channels share 10 Hz and 40 Hz oscillatory modes
ch1 = np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 40 * t)
ch2 = np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 40 * t)
signal = np.array([ch1, ch2])

# ── MEMD decomposition ──────────────────────────────────────────────────
imfs_memd, residue_memd = memd(signal, num_directions=32)

# ── Channel-wise EMD for comparison ──────────────────────────────────────
imfs_ch1, _ = decompose(ch1)
imfs_ch2, _ = decompose(ch2)

# ── Plot MEMD results ───────────────────────────────────────────────────
n_imfs = min(len(imfs_memd), 4)
fig, axes = plt.subplots(n_imfs + 1, 2, figsize=(12, 2 * (n_imfs + 1)), sharex=True)

for ch in range(2):
    axes[0, ch].plot(t, signal[ch], "k", linewidth=0.5)
    axes[0, ch].set_title(f"Channel {ch + 1}")
    axes[0, ch].set_ylabel("Signal")

    for k in range(n_imfs):
        axes[k + 1, ch].plot(t, imfs_memd[k, ch], linewidth=0.5)
        axes[k + 1, ch].set_ylabel(f"IMF {k + 1}")

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")
fig.suptitle(
    "Multivariate EMD — Aligned IMFs Across Channels",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout()

# Verify reconstruction
reconstructed = np.sum(imfs_memd, axis=0) + residue_memd
print(f"MEMD reconstruction error: {np.max(np.abs(reconstructed - signal)):.2e}")
print(f"Number of aligned IMFs: {len(imfs_memd)}")
print(f"IMF shape: {imfs_memd.shape}  (n_imfs, n_channels, n_samples)")

plt.show()
