"""
Flandrin separation analysis across EMD methods.

Extends the Rilling & Flandrin (2008) separation analysis to compare
how different decomposition methods handle mode mixing.

For a two-tone signal x(t) = cos(2πt) + a·cos(2πft + φ), the criterion

    c₁ = ‖IMF₁ − cos(2πt)‖ / ‖a·cos(2πft + φ)‖

quantifies separation quality: c₁ ≈ 0 means clean separation,
c₁ ≈ 1 means mode mixing.

This script computes the c₁ criterion for:
  - EMD (direct sifting, matching Rilling & Flandrin)
  - EEMD (noise-assisted ensemble)
  - CEEMDAN (complete EEMD with adaptive noise)
  - Masked EMD (mask-signal-assisted sifting)

and plots them side by side to show how each method expands or
contracts the separation region.

Note on EMD: We use direct sifting (sift_n_times) instead of
decompose() because decompose() has an is_imf() early-exit check
that short-circuits sifting for signals that "look like" IMFs.
For the two-tone Flandrin signal at small a, this incorrectly
returns the whole signal as IMF₁ without separating the tones.
EEMD/CEEMDAN are not affected because added noise breaks the
is_imf condition.

Note: Computationally expensive. Expect 15-45 minutes with
default settings. Reduce GRID_SIZE and N_PHASES for faster previews.
"""

import time
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt

from hhtpy import eemd, ceemdan, masked_decompose
from hhtpy._emd_utils import sift

# ── Configuration ────────────────────────────────────────────────────

GRID_SIZE = 30  # points per axis (increase for smoother plots)
N_PHASES = 8  # phase values φ to average over
N_SIFTS = 10  # sifting iterations for EMD (n = 10 in the paper)
SAMPLES_PER_PERIOD = 100  # temporal resolution
SIGNAL_PERIODS = 10  # duration in periods of the high-frequency tone

LOG_A_RANGE = (-2.0, 2.0)  # log₁₀(a), matching Rilling & Flandrin Fig. 3
F_RANGE = (0.02, 0.98)  # frequency ratio f ∈ (0, 1)

# EEMD/CEEMDAN parameters
ENSEMBLE_TRIALS = 100
NOISE_AMPLITUDE = 0.2

# ── Helpers ──────────────────────────────────────────────────────────


def _criterion(imf1, hf, lf):
    """Compute the separation criterion c₁."""
    num = np.linalg.norm(imf1 - hf)
    den = np.linalg.norm(lf)
    return num / den if den > 0 else 0.0


def _sift_n_times(signal, n):
    """Apply the sifting operator n times (no stopping criterion, no
    normalization, no is_imf check — matching Rilling & Flandrin)."""
    mode = signal.copy()
    for _ in range(n):
        mode = sift(mode)
    return mode


def _extract_first_imf_emd(signal):
    """First IMF via direct sifting (matching Rilling & Flandrin)."""
    return _sift_n_times(signal, N_SIFTS)


def _extract_first_imf_eemd(signal):
    """First IMF via EEMD."""
    imfs, _ = eemd(
        signal, num_trials=ENSEMBLE_TRIALS,
        noise_amplitude=NOISE_AMPLITUDE, max_imfs=2, seed=42,
    )
    return imfs[0] if len(imfs) > 0 else signal


def _extract_first_imf_ceemdan(signal):
    """First IMF via CEEMDAN."""
    imfs, _ = ceemdan(
        signal, num_trials=ENSEMBLE_TRIALS,
        noise_amplitude=NOISE_AMPLITUDE, max_imfs=2, seed=42,
    )
    return imfs[0] if len(imfs) > 0 else signal


def _extract_first_imf_masked(signal, f_s, mask_freq):
    """First IMF via masked EMD.

    Mask amplitude scales with max(|signal|) so the mask always
    dominates the sifting — this removes amplitude-ratio dependence,
    leaving only frequency-ratio dependence (horizontal structure
    in the Flandrin plot is expected for masked EMD).
    """
    mask_amp = 2.0 * np.max(np.abs(signal))
    imfs, _ = masked_decompose(
        signal,
        mask_frequency=mask_freq,
        mask_amplitude=mask_amp,
        sampling_frequency=f_s,
        max_imfs=1,
    )
    return imfs[0] if len(imfs) > 0 else signal


def compute_row_all_methods(args):
    """Compute one row (fixed f) of the criterion grid for all methods."""
    i, f, log_a_vals, t, hf, phi_vals, f_s = args
    n_methods = 4
    rows = np.zeros((n_methods, len(log_a_vals)))

    hf_freq_hz = f_s / SAMPLES_PER_PERIOD  # = 1.0 Hz

    for j, log_a in enumerate(log_a_vals):
        a = 10.0 ** log_a  # log₁₀(a) → linear amplitude
        c_sums = np.zeros(n_methods)

        for phi in phi_vals:
            lf = a * np.cos(2 * np.pi * f * t + phi)
            signal = hf + lf

            # EMD (direct sifting, no normalization)
            try:
                imf1 = _extract_first_imf_emd(signal)
                c_sums[0] += _criterion(imf1, hf, lf)
            except Exception:
                c_sums[0] += 1.0

            # EEMD
            try:
                imf1 = _extract_first_imf_eemd(signal)
                c_sums[1] += _criterion(imf1, hf, lf)
            except Exception:
                c_sums[1] += 1.0

            # CEEMDAN
            try:
                imf1 = _extract_first_imf_ceemdan(signal)
                c_sums[2] += _criterion(imf1, hf, lf)
            except Exception:
                c_sums[2] += 1.0

            # Masked EMD
            try:
                imf1 = _extract_first_imf_masked(signal, f_s, hf_freq_hz)
                c_sums[3] += _criterion(imf1, hf, lf)
            except Exception:
                c_sums[3] += 1.0

        rows[:, j] = c_sums / len(phi_vals)

    return i, rows


# ── Main ─────────────────────────────────────────────────────────────


def main():
    N = SIGNAL_PERIODS * SAMPLES_PER_PERIOD
    f_s = float(SAMPLES_PER_PERIOD)  # 1 period = 1 "second" → HF at 1 Hz
    t = np.arange(N) / f_s
    hf = np.cos(2 * np.pi * t)  # 1 Hz

    log_a_vals = np.linspace(*LOG_A_RANGE, GRID_SIZE)
    f_vals = np.linspace(*F_RANGE, GRID_SIZE)
    phi_vals = np.linspace(0, 2 * np.pi, N_PHASES, endpoint=False)

    n_workers = max(1, cpu_count() - 1)
    method_names = ["EMD", "EEMD", "CEEMDAN", "Masked EMD"]

    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, {N_PHASES} phases, {len(method_names)} methods")
    print(f"Signal: {N} samples ({SIGNAL_PERIODS} periods)")
    print(f"Amplitude range: a = {10**LOG_A_RANGE[0]:.3g} to {10**LOG_A_RANGE[1]:.3g}")
    print(f"Workers: {n_workers}")

    tasks = [
        (i, f, log_a_vals, t, hf, phi_vals, f_s)
        for i, f in enumerate(f_vals)
    ]

    criteria = np.zeros((len(method_names), GRID_SIZE, GRID_SIZE))
    t0 = time.time()
    completed = 0

    with Pool(n_workers) as pool:
        for i, rows in pool.imap_unordered(compute_row_all_methods, tasks):
            criteria[:, i, :] = rows
            completed += 1
            elapsed = time.time() - t0
            eta = elapsed / completed * (GRID_SIZE - completed)
            print(
                f"\r  {completed}/{GRID_SIZE} rows — "
                f"elapsed {elapsed:.0f}s, ETA ~{eta:.0f}s",
                end="", flush=True,
            )

    print(f"\nDone in {time.time() - t0:.1f}s")

    # ── Plot 2x2 comparison ──────────────────────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    # Theoretical boundary curves in log₁₀(a) space
    f_th = np.linspace(0.01, 0.99, 500)
    # af = 1 → a = 1/f → log₁₀(a) = -log₁₀(f)
    log_a_af1 = -np.log10(f_th)
    # af² = 1 → a = 1/f² → log₁₀(a) = -2·log₁₀(f)
    log_a_af2 = -2 * np.log10(f_th)
    # af·sin(3πf/2) = 1 (refined boundary, valid for f < 1/3)
    f_th3 = f_th[f_th < 1 / 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_term = f_th3 * np.sin(3 * np.pi * f_th3 / 2)
        log_a_af3 = np.where(sin_term > 0, -np.log10(sin_term), np.inf)

    for idx, (ax, name) in enumerate(zip(axes.flat, method_names)):
        im = ax.pcolormesh(
            log_a_vals, f_vals, criteria[idx],
            cmap="gray", vmin=0, vmax=1, shading="auto",
        )

        # c₁ = 0.5 contour
        try:
            ax.contour(
                log_a_vals, f_vals, criteria[idx],
                levels=[0.5], colors="cyan", linewidths=2,
            )
        except Exception:
            pass

        # Theoretical curves (EMD theory, for reference on all panels)
        m1 = (log_a_af1 >= LOG_A_RANGE[0]) & (log_a_af1 <= LOG_A_RANGE[1])
        ax.plot(log_a_af1[m1], f_th[m1], "r--", lw=1, alpha=0.7,
                label=r"$af = 1$")

        m2 = (log_a_af2 >= LOG_A_RANGE[0]) & (log_a_af2 <= LOG_A_RANGE[1])
        ax.plot(log_a_af2[m2], f_th[m2], "r-.", lw=1, alpha=0.7,
                label=r"$af^2 = 1$")

        # Refined boundary (only on EMD panel)
        if idx == 0:
            m3 = (
                (log_a_af3 >= LOG_A_RANGE[0])
                & (log_a_af3 <= LOG_A_RANGE[1])
                & np.isfinite(log_a_af3)
            )
            ax.plot(log_a_af3[m3], f_th3[m3], "r:", lw=1, alpha=0.7,
                    label=r"$af\sin(3\pi f/2) = 1$")

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel(r"$\log_{10}\, a$")
        ax.set_ylabel(r"$f$")

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=r"$\langle c_1 \rangle_\varphi$")

    axes[0, 0].legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Flandrin Separation Analysis: EMD vs EEMD vs CEEMDAN vs Masked EMD",
        fontsize=14, fontweight="bold", y=0.98,
    )

    fig.savefig("figs/flandrin_separation_comparison.png", dpi=200, bbox_inches="tight")
    print("Saved to figs/flandrin_separation_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
