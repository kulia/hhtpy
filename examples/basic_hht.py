"""
Basic Hilbert-Huang Transform: EMD, Hilbert spectrum, marginal spectrum.

Decomposes a two-component FM+AM signal, plots the IMFs, Hilbert spectrum,
and marginal Hilbert spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
from hhtpy import hilbert_huang_transform
from hhtpy.plot import (
    plot_imfs,
    plot_hilbert_spectrum,
    HilbertSpectrumConfig,
    plot_marginal_hilbert_spectrum,
)

plt.style.use("seaborn-v0_8")
plt.rcParams["image.cmap"] = "viridis"

# ── Signal: FM component + pure tone ────────────────────────────────
T = 5  # sec
f_s = 15000  # Hz
n = np.arange(T * f_s)
t = n / f_s

y = 1 * np.cos(2 * np.pi * 50 * t + 20 * np.sin(2 * np.pi * 0.5 * t)) + 2 * np.cos(
    2 * np.pi * 20 * t
)

# ── Decompose ───────────────────────────────────────────────────────
imfs, residue = hilbert_huang_transform(y, f_s)

# ── Plot IMFs ───────────────────────────────────────────────────────
fig, axs = plot_imfs(imfs, y, residue, t, max_number_of_imfs=2)
axs[1].set_ylim([-1.1, 1.1])
[ax.set_xlim([2.4, 2.5]) for ax in axs]
[ax.set_xticks([]) for ax in axs[:-1]]
axs[-1].set_xlabel("Time (s)")
axs[0].set_ylabel("Original\nSignal")
fig.tight_layout()

# ── Hilbert spectrum ────────────────────────────────────────────────
fig2, ax2, clb = plot_hilbert_spectrum(
    imfs,
    config=HilbertSpectrumConfig(max_number_of_imfs=2),
)
fig2.tight_layout()
ax2.set_xlim([0.5, 4.5])

# ── Marginal Hilbert spectrum ───────────────────────────────────────
fig3, ax3 = plot_marginal_hilbert_spectrum(imfs)
ax3.set_yscale("log")

plt.show()
