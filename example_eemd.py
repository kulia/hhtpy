"""
EEMD and CEEMDAN example: mode mixing mitigation.

Demonstrates how EEMD and CEEMDAN handle a signal with intermittent
high-frequency bursts — a classic mode mixing scenario where standard
EMD fails to cleanly separate components.
"""

import numpy as np
import matplotlib.pyplot as plt
from hhtpy import decompose, eemd, ceemdan

plt.style.use("seaborn-v0_8")

# ── Signal with intermittent burst ───────────────────────────────────────
f_s = 1000  # Hz
T = 3  # seconds
t = np.arange(T * f_s) / f_s

# 10 Hz tone + intermittent 80 Hz burst in the middle second
signal = np.cos(2 * np.pi * 10 * t)
signal[1000:2000] += 0.5 * np.cos(2 * np.pi * 80 * t[1000:2000])

# ── Decompose with all three methods ────────────────────────────────────
imfs_emd, res_emd = decompose(signal)
imfs_eemd, res_eemd = eemd(signal, num_trials=50, noise_amplitude=0.2, seed=42)
imfs_ceemdan, res_ceemdan = ceemdan(signal, num_trials=50, noise_amplitude=0.2, seed=42)

# ── Plot comparison ─────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 3, figsize=(14, 8), sharex=True, sharey="row")

titles = ["EMD", "EEMD", "CEEMDAN"]
all_imfs = [imfs_emd, imfs_eemd, imfs_ceemdan]

for col, (title, imfs) in enumerate(zip(titles, all_imfs)):
    axes[0, col].plot(t, signal, "k", linewidth=0.5)
    axes[0, col].set_title(title)
    if col == 0:
        axes[0, col].set_ylabel("Signal")

    for row in range(1, 4):
        idx = row - 1
        if idx < len(imfs):
            axes[row, col].plot(t, imfs[idx], linewidth=0.5)
        if col == 0:
            axes[row, col].set_ylabel(f"IMF {idx + 1}")

axes[-1, 1].set_xlabel("Time (s)")
fig.suptitle(
    "Mode Mixing Comparison: EMD vs EEMD vs CEEMDAN",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout()

# Print reconstruction errors
print("Reconstruction errors:")
print(f"  EMD:     {np.max(np.abs(np.sum(imfs_emd, axis=0) + res_emd - signal)):.2e}")
print(f"  EEMD:    {np.max(np.abs(np.sum(imfs_eemd, axis=0) + res_eemd - signal)):.2e}")
print(f"  CEEMDAN: {np.max(np.abs(np.sum(imfs_ceemdan, axis=0) + res_ceemdan - signal)):.2e}")

plt.savefig("figs/eemd_ceemdan_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
