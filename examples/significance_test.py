"""
Wu-Huang significance test for IMFs.

Tests whether each IMF is statistically distinguishable from white
noise using the energy-period relationship established in:

    Wu, Z. & Huang, N.E. (2004). "A study of the characteristics of
    white noise using the empirical mode decomposition method."
    Proc. R. Soc. A, 460, 1597-1611.

IMFs whose energy falls within the expected noise band are flagged
as non-significant.
"""

import numpy as np
import matplotlib.pyplot as plt
from hhtpy import decompose, significance_test

plt.style.use("seaborn-v0_8")

# ── Signal: two tones + white noise ─────────────────────────────────
rng = np.random.default_rng(42)
f_s = 1000
T = 5
t = np.arange(T * f_s) / f_s

clean = 2.0 * np.cos(2 * np.pi * 5 * t) + 0.8 * np.cos(2 * np.pi * 25 * t)
noise = 0.5 * rng.normal(size=len(t))
signal = clean + noise

# ── Decompose ──────────────────────────────────────────────────────
imfs, residue = decompose(signal)

# ── Significance test (a priori) ───────────────────────────────────
results = significance_test(imfs, alpha=0.95, method="apriori")

print(f"{'IMF':>4}  {'log(Period)':>12}  {'log(Energy)':>12}  {'Significant':>12}")
print("-" * 46)
for r in results:
    print(f"{r.index+1:>4}  {r.log_period:>12.3f}  {r.log_energy:>12.3f}  "
          f"{'YES' if r.is_significant else 'no':>12}")

# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: IMFs colored by significance
ax = axes[0]
n_show = min(len(results), 6)
for r in results[:n_show]:
    offset = -r.index * 3
    color = "C0" if r.is_significant else "gray"
    label = f"IMF {r.index+1}" + (" *" if r.is_significant else "")
    ax.plot(t, imfs[r.index] + offset, color=color, lw=0.5, label=label)
ax.set_xlabel("Time (s)")
ax.set_ylabel("IMFs (offset for clarity)")
ax.set_title("Significant IMFs (color) vs noise-like (gray)")
ax.legend(fontsize=8, loc="upper right")

# Right: energy-period scatter with confidence bounds
ax = axes[1]

log_periods = np.array([r.log_period for r in results])
log_energies = np.array([r.log_energy for r in results])
is_sig = [r.is_significant for r in results]
upper_bounds = np.array([r.upper_bound for r in results])
lower_bounds = np.array([r.lower_bound for r in results
                          if r.lower_bound is not None])

ax.scatter(
    log_periods, log_energies,
    c=["C0" if s else "gray" for s in is_sig],
    s=60, zorder=5, edgecolors="k",
)

# Theoretical noise line and confidence band
sort_idx = np.argsort(log_periods)
lp_sorted = log_periods[sort_idx]

# Noise model: ln(E) = -ln(T) + const
ax.plot(lp_sorted, upper_bounds[sort_idx], "r-", lw=1, alpha=0.7, label="Upper bound")
if len(lower_bounds) == len(results):
    ax.plot(lp_sorted, lower_bounds[sort_idx], "r-", lw=1, alpha=0.7, label="Lower bound")
    ax.fill_between(
        lp_sorted, lower_bounds[sort_idx], upper_bounds[sort_idx],
        alpha=0.15, color="red", label="95% confidence band",
    )

# Noise line: E*T = const → ln(E) = -ln(T)
noise_line = -lp_sorted
ax.plot(lp_sorted, noise_line, "k--", lw=1, alpha=0.5, label="Noise model")

for r in results:
    ax.annotate(f"{r.index+1}", (r.log_period, r.log_energy),
                textcoords="offset points", xytext=(5, 5), fontsize=8)

ax.set_xlabel("ln(Period)")
ax.set_ylabel("ln(Energy)")
ax.set_title("Wu-Huang Significance Test")
ax.legend(fontsize=8)

fig.suptitle("IMF Significance Testing", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.show()
