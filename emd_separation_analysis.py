"""
EMD mode mixing / separation analysis.

Reproduces the resolution analysis from:

    Rilling & Flandrin (2008), "One or Two Frequencies? The Empirical
    Mode Decomposition Answers", IEEE Trans. Signal Processing.
    https://doi.org/10.1109/TSP.2007.906771

For a two-tone signal  x(t) = cos(2πt) + a·cos(2πft + φ),
the EMD either separates the tones into distinct IMFs or treats
them as a single modulated component (mode mixing), depending
on the amplitude ratio (a) and frequency ratio (f).

The criterion c₁ quantifies this:

    c₁ = ‖IMF₁ − cos(2πt)‖ / ‖a·cos(2πft + φ)‖

    c₁ ≈ 0  →  tones separated (no mode mixing)
    c₁ ≈ 1  →  tones merged (mode mixing)

This script sweeps (a, f) and maps the separation boundary,
with theoretical curves from the paper overlaid.
"""

import time
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt

from hhtpy._emd_utils import sift

# ── Configuration ────────────────────────────────────────────────────
# Increase GRID_SIZE and N_PHASES for publication quality.
# At defaults below, expect ~2-5 minutes runtime.

GRID_SIZE = 50  # points per axis (paper uses ~100)
N_PHASES = 10  # phase values φ to average over
N_SIFTS = 10  # sifting iterations (n = 10 in the paper)

SAMPLES_PER_PERIOD = 100  # temporal resolution (fs in units of HF frequency)
SIGNAL_PERIODS = 10  # signal duration in HF periods

LOG_A_RANGE = (-2.0, 2.0)  # log₁₀(a) range
F_RANGE = (0.02, 0.98)  # frequency ratio f ∈ (0, 1)

# ── Helpers ──────────────────────────────────────────────────────────


def sift_n_times(signal, n):
    """Apply the sifting operator n times to extract the first IMF."""
    mode = signal.copy()
    for _ in range(n):
        mode = sift(mode)
    return mode


def compute_row(args):
    """Compute one row of the criterion grid (fixed f, all a values).

    Designed for use with multiprocessing.Pool.
    """
    i, f, log_a_vals, t, hf, phi_vals, n_sifts = args
    row = np.zeros(len(log_a_vals))

    for j, log_a in enumerate(log_a_vals):
        a = 10**log_a
        c_sum = 0.0

        for phi in phi_vals:
            lf = a * np.cos(2 * np.pi * f * t + phi)
            signal = hf + lf

            try:
                imf1 = sift_n_times(signal, n_sifts)
            except Exception:
                # If sifting fails (degenerate extrema), treat as no separation
                c_sum += 1.0
                continue

            numerator = np.linalg.norm(imf1 - hf)
            denominator = np.linalg.norm(lf)
            c_sum += numerator / denominator if denominator > 0 else 0.0

        row[j] = c_sum / len(phi_vals)

    return i, row


# ── Main ─────────────────────────────────────────────────────────────


def main():
    N = SIGNAL_PERIODS * SAMPLES_PER_PERIOD
    t = np.arange(N) / SAMPLES_PER_PERIOD
    hf = np.cos(2 * np.pi * t)

    log_a_vals = np.linspace(*LOG_A_RANGE, GRID_SIZE)
    f_vals = np.linspace(*F_RANGE, GRID_SIZE)
    phi_vals = np.linspace(0, 2 * np.pi, N_PHASES, endpoint=False)

    total_runs = GRID_SIZE * GRID_SIZE * N_PHASES
    n_workers = max(1, cpu_count() - 1)

    print(f"Grid: {GRID_SIZE}×{GRID_SIZE}, {N_PHASES} phases, {N_SIFTS} sifts")
    print(f"Signal: {N} samples ({SIGNAL_PERIODS} periods @ {SAMPLES_PER_PERIOD} samp/period)")
    print(f"Total: {total_runs:,} sifting runs on {n_workers} workers")

    # Build argument list for parallel map
    tasks = [
        (i, f, log_a_vals, t, hf, phi_vals, N_SIFTS) for i, f in enumerate(f_vals)
    ]

    criterion = np.zeros((GRID_SIZE, GRID_SIZE))
    t0 = time.time()

    with Pool(n_workers) as pool:
        for i, row in pool.imap_unordered(compute_row, tasks):
            criterion[i, :] = row
            done = np.count_nonzero(criterion.any(axis=1))
            elapsed = time.time() - t0
            eta = elapsed / max(done, 1) * (GRID_SIZE - done)
            print(
                f"\r  {done}/{GRID_SIZE} rows — "
                f"elapsed {elapsed:.0f}s, ETA ~{eta:.0f}s",
                end="",
                flush=True,
            )

    print(f"\nDone in {time.time() - t0:.1f}s")

    # ── Plot ─────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.pcolormesh(
        log_a_vals,
        f_vals,
        criterion,
        cmap="gray",
        vmin=0,
        vmax=1,
        shading="auto",
    )
    fig.colorbar(
        im, ax=ax, label=r"$\langle c_1^{(10)}(a,f,\varphi) \rangle_\varphi$"
    )

    # ── Theoretical boundary curves ──────────────────────────────────
    f_th = np.linspace(0.01, 0.99, 500)

    # af = 1
    log_a1 = -np.log10(f_th)
    m1 = (log_a1 >= LOG_A_RANGE[0]) & (log_a1 <= LOG_A_RANGE[1])
    ax.plot(log_a1[m1], f_th[m1], "r--", lw=1.5, label=r"$af = 1$")

    # af² = 1
    log_a2 = -2 * np.log10(f_th)
    m2 = (log_a2 >= LOG_A_RANGE[0]) & (log_a2 <= LOG_A_RANGE[1])
    ax.plot(log_a2[m2], f_th[m2], "r-.", lw=1.5, label=r"$af^2 = 1$")

    # af sin(3πf/2) = 1 — refined boundary, only valid for f < 1/3
    # (Section IV-C-2: tighter than af=1 in this range)
    f_th3 = f_th[f_th < 1 / 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_term = f_th3 * np.sin(3 * np.pi * f_th3 / 2)
        log_a3 = np.where(sin_term > 0, -np.log10(sin_term), np.inf)
    m3 = (
        (log_a3 >= LOG_A_RANGE[0])
        & (log_a3 <= LOG_A_RANGE[1])
        & np.isfinite(log_a3)
    )
    ax.plot(log_a3[m3], f_th3[m3], "r:", lw=1.5, label=r"$af\sin(3\pi f/2) = 1$")

    # c = 0.5 contour (thick black line as in the paper)
    ax.contour(
        log_a_vals, f_vals, criterion, levels=[0.5], colors="k", linewidths=2
    )

    # Region labels
    ax.text(
        -1.0, 0.3, r"$\approx 0$", fontsize=16, color="gray", ha="center",
        fontweight="bold",
    )
    ax.text(
        1.5, 0.8, r"$\approx 1$", fontsize=16, color="black", ha="center",
        fontweight="bold",
    )

    ax.set_xlabel(r"$\log_{10}\, a$", fontsize=12)
    ax.set_ylabel(r"$f$", fontsize=12)
    ax.set_title(
        r"EMD separation: $\langle c_1^{(10)}(a,f,\varphi) \rangle_\varphi$"
        "\n(Rilling & Flandrin, 2008, Fig. 3)",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig("figs/figure3_rilling_flandrin.png", dpi=200)
    print("Saved to figs/figure3_rilling_flandrin.png")
    plt.show()


if __name__ == "__main__":
    main()
