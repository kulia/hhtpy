"""
Noise-assisted EMD variants for mitigating mode mixing.

- **EEMD** (Wu & Huang, 2009): Ensemble EMD — averages IMFs across
  multiple noise-perturbed decompositions.
- **CEEMDAN** (Torres et al., 2011): Complete EEMD with Adaptive Noise —
  adds noise adaptively at each stage, guaranteeing exact reconstruction.
"""

from typing import Optional
import numpy as np
from .emd import decompose
from .sift_stopping_criteria import (
    SiftStoppingCriterion,
    get_stopping_criterion_fixed_number_of_sifts,
)
from ._emd_utils import is_monotonic, find_local_extrema


def eemd(
    signal: np.ndarray,
    num_trials: int = 100,
    noise_amplitude: float = 0.2,
    stopping_criterion: SiftStoppingCriterion = get_stopping_criterion_fixed_number_of_sifts(
        15
    ),
    max_imfs: Optional[int] = None,
    max_sifts: int = 100,
    seed: Optional[int] = None,
):
    """
    Ensemble Empirical Mode Decomposition (Wu & Huang, 2009).

    Adds white Gaussian noise to the signal over multiple trials and
    averages the resulting IMFs. This mitigates mode mixing by populating
    the time-frequency space uniformly, ensuring that components of
    similar scale are assigned to the same IMF.

    Note: EEMD does **not** guarantee exact reconstruction. The residual
    noise decreases as ``noise_amplitude / sqrt(num_trials)`` but never
    reaches zero for finite ensembles. For exact reconstruction, use
    :func:`ceemdan` instead.

    Args:
        signal: 1D input signal.
        num_trials: Number of noise-perturbed decomposition trials.
            Default is 100.
        noise_amplitude: Standard deviation of added noise, expressed as
            a fraction of the signal's standard deviation. Default is 0.2
            (20%), as recommended by Wu & Huang.
        stopping_criterion: Sifting stopping criterion for each trial's
            EMD. Default is 15 fixed sifts.
        max_imfs: Maximum number of IMFs to extract per trial.
        max_sifts: Safety limit on sifting iterations per IMF.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (imfs, residue):
            - imfs (np.ndarray): Array of shape ``(n_imfs, n_samples)``.
            - residue (np.ndarray): ``signal - sum(imfs)``.
    """
    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional.")
    if signal.size == 0:
        raise ValueError("Input signal must not be empty.")

    rng = np.random.default_rng(seed)
    noise_std = noise_amplitude * np.std(signal)
    n_samples = len(signal)

    # Collect IMFs from all trials
    all_trial_imfs = []
    max_n_imfs = 0

    for _ in range(num_trials):
        noise = rng.normal(0, noise_std, n_samples)
        noisy_signal = signal + noise

        imfs_trial, _ = decompose(
            noisy_signal,
            stopping_criterion=stopping_criterion,
            max_imfs=max_imfs,
            max_sifts=max_sifts,
        )

        all_trial_imfs.append(imfs_trial)
        max_n_imfs = max(max_n_imfs, len(imfs_trial))

    # Pad trials with fewer IMFs (treat missing IMFs as zero)
    padded = np.zeros((num_trials, max_n_imfs, n_samples))
    for i, trial_imfs in enumerate(all_trial_imfs):
        padded[i, : len(trial_imfs), :] = trial_imfs

    # Ensemble average
    imfs = np.mean(padded, axis=0)

    # Residue: exact difference so that sum(imfs) + residue = signal
    residue = signal - np.sum(imfs, axis=0)

    return imfs, residue


def ceemdan(
    signal: np.ndarray,
    num_trials: int = 100,
    noise_amplitude: float = 0.2,
    stopping_criterion: SiftStoppingCriterion = get_stopping_criterion_fixed_number_of_sifts(
        15
    ),
    max_imfs: Optional[int] = None,
    max_sifts: int = 100,
    seed: Optional[int] = None,
):
    """
    Complete Ensemble EMD with Adaptive Noise (Torres et al., 2011).

    Like EEMD, adds white noise to reduce mode mixing, but does so
    adaptively at each decomposition stage. This guarantees **exact
    reconstruction**: ``sum(imfs) + residue == signal``.

    At each stage k, the noise contribution is the k-th IMF of the
    original noise realization, scaled to the current residue level.
    This keeps the signal-to-noise ratio approximately constant across
    all stages.

    Args:
        signal: 1D input signal.
        num_trials: Number of ensemble trials. Default is 100.
        noise_amplitude: Noise scale factor (fraction of the current
            residue's standard deviation). Default is 0.2 (20%).
        stopping_criterion: Sifting stopping criterion used internally.
        max_imfs: Maximum number of IMFs to extract.
        max_sifts: Safety limit on sifting iterations per IMF.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (imfs, residue):
            - imfs (np.ndarray): Array of shape ``(n_imfs, n_samples)``.
            - residue (np.ndarray): The final residue, satisfying
              ``sum(imfs) + residue == signal`` exactly.
    """
    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional.")
    if signal.size == 0:
        raise ValueError("Input signal must not be empty.")

    rng = np.random.default_rng(seed)
    n_samples = len(signal)

    theoretical_max = int(np.log2(n_samples) - 1)
    n_imfs_limit = (
        min(max_imfs, theoretical_max) if max_imfs is not None else theoretical_max
    )

    # Pre-generate noise realizations and decompose them
    noise_realizations = [
        rng.normal(0, 1, n_samples) for _ in range(num_trials)
    ]

    # Decompose each noise realization to get noise IMFs for each stage
    noise_imfs_all = []
    for noise in noise_realizations:
        noise_imfs, _ = decompose(
            noise,
            stopping_criterion=stopping_criterion,
            max_imfs=n_imfs_limit,
            max_sifts=max_sifts,
        )
        noise_imfs_all.append(noise_imfs)

    # Stage 1: Extract first IMF
    beta = noise_amplitude * np.std(signal)
    first_imfs = []
    for i in range(num_trials):
        noisy = signal + beta * noise_realizations[i]
        trial_imfs, _ = decompose(
            noisy,
            stopping_criterion=stopping_criterion,
            max_imfs=1,
            max_sifts=max_sifts,
        )
        if len(trial_imfs) > 0:
            first_imfs.append(trial_imfs[0])
        else:
            first_imfs.append(np.zeros(n_samples))

    c1 = np.mean(first_imfs, axis=0)
    residue = signal - c1
    imfs_list = [c1]

    # Subsequent stages
    for k in range(1, n_imfs_limit):
        if _should_stop_ceemdan(residue):
            break

        beta_k = noise_amplitude * np.std(residue)
        if beta_k < np.finfo(float).eps:
            break

        stage_imfs = []
        for i in range(num_trials):
            # Use the (k+1)-th IMF of the i-th noise realization
            if k < len(noise_imfs_all[i]):
                noise_k = noise_imfs_all[i][k]
            else:
                noise_k = np.zeros(n_samples)

            perturbed = residue + beta_k * noise_k
            trial_imfs, _ = decompose(
                perturbed,
                stopping_criterion=stopping_criterion,
                max_imfs=1,
                max_sifts=max_sifts,
            )
            if len(trial_imfs) > 0:
                stage_imfs.append(trial_imfs[0])
            else:
                stage_imfs.append(np.zeros(n_samples))

        ck = np.mean(stage_imfs, axis=0)
        residue = residue - ck
        imfs_list.append(ck)

    return np.array(imfs_list), residue


def _should_stop_ceemdan(residue: np.ndarray) -> bool:
    """Check if the residue can no longer be decomposed."""
    if is_monotonic(residue):
        return True

    maxima, minima = find_local_extrema(residue)
    if len(maxima) + len(minima) < 2:
        return True

    return False
