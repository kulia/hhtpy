import numpy as np
import matplotlib.pyplot as plt
from hhtpy import decompose, hilbert_huang_transform
from hhtpy.plot import (
    plot_imfs,
    plot_hilbert_spectrum,
    HilbertSpectrumConfig,
    plot_marginal_hilbert_spectrum,
)

plt.style.use("seaborn-v0_8")
plt.rcParams["image.cmap"] = "viridis"


T = 5  # sec
f_s = 15000  # Hz
n = np.arange(T * f_s)
t = n / f_s  # sec

# y = 0.3 * np.cos(2 * np.pi * 5 * t**2) + 2 * np.cos(2 * np.pi * 1 * t) + 1 * t

y = 1 * np.cos(2 * np.pi * 50 * t + 20 * np.sin(2 * np.pi * 0.5 * t)) + 2 * np.cos(
    2 * np.pi * 20 * t
)

imfs, residue = hilbert_huang_transform(y, f_s)

fig, axs = plot_imfs(imfs, y, residue, t, max_number_of_imfs=2)
axs[1].set_ylim([-1.1, 1.1])
[ax.set_xlim([2.4, 2.5]) for ax in axs]
[ax.set_xticks([]) for ax in axs[:-1]]
axs[-1].set_xlabel("Time (s)")
axs[0].set_ylabel("Original\nSignal")

fig.tight_layout()

fig.savefig("figs/imfs.png", dpi=300)

fig, ax, clb = plot_hilbert_spectrum(
    imfs,
    config=HilbertSpectrumConfig(max_number_of_imfs=2),
)

fig.tight_layout()
ax.set_xlim([0.5, 4.5])

fig.savefig("figs/hilbert_spectrum.png", dpi=300)

fig, ax = plot_marginal_hilbert_spectrum(imfs)
ax.set_yscale("log")

fig.savefig("figs/marginal_hilbert_spectrum.png", dpi=300)

plt.show()
