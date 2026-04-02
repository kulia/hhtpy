"""
Cycle-by-cycle analysis of an IMF.

Demonstrates detecting individual oscillatory cycles, characterizing
their amplitude, frequency, and waveform shape, and phase-aligning
them for visual comparison.

Reference:
    Cole, S. & Voytek, B. (2019). "Cycle-by-cycle analysis of neural
    oscillations." J. Neurophysiology, 122(2), 849-861.
"""

import numpy as np
import matplotlib.pyplot as plt
from hhtpy import decompose, detect_cycles, phase_align, get_cycle_vector, cycle_summary_table

plt.style.use("seaborn-v0_8")

# ── Signal: chirp with amplitude modulation ─────────────────────────
f_s = 1000  # Hz
T = 2  # seconds
t = np.arange(T * f_s) / f_s

# 10-20 Hz chirp with slow AM envelope
freq = 10 + 10 * t / T  # linear chirp from 10 to 20 Hz
phase = 2 * np.pi * np.cumsum(freq) / f_s
amplitude = 1.0 + 0.5 * np.sin(2 * np.pi * 1 * t)  # 1 Hz AM
signal = amplitude * np.cos(phase) + 0.3 * np.cos(2 * np.pi * 60 * t)

# ── EMD decomposition ──────────────────────────────────────────────
imfs, residue = decompose(signal)
imf = imfs[0]  # first IMF (highest frequency)

# ── Detect cycles ──────────────────────────────────────────────────
cycles = detect_cycles(imf, sampling_frequency=f_s)
table = cycle_summary_table(cycles)

print(f"Detected {len(cycles)} cycles in IMF 1")
print(f"Frequency range: {table['frequency'].min():.1f} – {table['frequency'].max():.1f} Hz")
print(f"Amplitude range: {table['amplitude'].min():.3f} – {table['amplitude'].max():.3f}")

# ── Phase-align cycles ─────────────────────────────────────────────
aligned = phase_align(imf, n_points=100)
phase_grid = np.linspace(0, 360, 100, endpoint=False)

# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Top-left: IMF with cycles color-coded
cv = get_cycle_vector(imf)
ax = axes[0, 0]
ax.plot(t, imf, "k", lw=0.5, alpha=0.3)
for c in cycles[:20]:  # color first 20 cycles
    mask = slice(c.start_sample, c.end_sample)
    ax.plot(t[mask], imf[mask], lw=1.2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("IMF 1 with detected cycles")

# Top-right: frequency and amplitude over cycles
ax = axes[0, 1]
ax.scatter(table["frequency"], table["amplitude"], c=table["index"],
           cmap="viridis", s=20, alpha=0.7)
ax.set_xlabel("Cycle frequency (Hz)")
ax.set_ylabel("Cycle amplitude")
ax.set_title("Cycle-by-cycle frequency vs amplitude")

# Bottom-left: phase-aligned waveforms
ax = axes[1, 0]
for i in range(min(len(aligned), 30)):
    ax.plot(phase_grid, aligned[i], alpha=0.3, lw=0.8)
ax.plot(phase_grid, np.mean(aligned, axis=0), "k", lw=2, label="Mean")
ax.set_xlabel("Phase (degrees)")
ax.set_ylabel("Amplitude")
ax.set_title("Phase-aligned cycles")
ax.legend()

# Bottom-right: waveform shape metrics
ax = axes[1, 1]
complete = [c for c in cycles if c.is_complete]
if complete:
    rise_fracs = [c.rise_fraction for c in complete]
    symmetries = [c.peak_trough_symmetry for c in complete]
    ax.hist(rise_fracs, bins=20, alpha=0.7, label="Rise fraction")
    ax.axvline(0.25, color="r", ls="--", label="Symmetric cosine (0.25)")
    ax.set_xlabel("Rise fraction")
    ax.set_ylabel("Count")
    ax.set_title("Waveform shape distribution")
    ax.legend()

fig.suptitle("Cycle-by-Cycle Analysis", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.show()
