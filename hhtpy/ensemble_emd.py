"""
Noise-assisted EMD variants for mitigating mode mixing.

- **EEMD** (Wu & Huang, 2009): Ensemble EMD — averages IMFs across
  multiple noise-perturbed decompositions.
- **CEEMDAN** (Torres et al., 2011): Complete EEMD with Adaptive Noise —
  adds noise adaptively at each stage, guaranteeing exact reconstruction.
"""

from typing import Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from .emd import decompose
from .sift_stopping_criteria import (
    SiftStoppingCriterion,
    get_stopping_criterion_fixed_number_of_sifts,
)
from ._emd_utils import is_monotonic, find_local_extrema


# ── Top-level worker functions (picklable for multiprocessing) ───────


def _eemd_single_trial(args):
    """Decompose one noise-perturbed signal for EEMD."""
    signal, noise_seed, noise_std, stopping_criterion, max_imfs, max_sifts = args
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(0, noise_std, len(signal))
    imfs_trial, _ = decompose(
        signal + noise,
        stopping_criterion=stopping_criterion,
        max_imfs=max_imfs,
        max_sifts=max_sifts,
    )
    return imfs_trial


def _ceemdan_decompose_noise(args):
    """Decompose one noise realization for CEEMDAN pre-computation."""
    noise, stopping_criterion, max_imfs, max_sifts = args
    noise_imfs, _ = decompose(
        noise,
        stopping_criterion=stopping_criterion,
        max_imfs=max_imfs,
        max_sifts=max_sifts,
    )
    return noise_imfs


def _ceemdan_stage_trial(args):
    """Extract one IMF from a perturbed residue for a CEEMDAN stage."""
    perturbed, stopping_criterion, max_sifts = args
    trial_imfs, _ = decompose(
        perturbed,
        stopping_criterion=stopping_criterion,
        max_imfs=1,
        max_sifts=max_sifts,
    )
    if len(trial_imfs) > 0:
        return trial_imfs[0]
    return np.zeros(len(perturbed))


# ── Public API ───────────────────────────────────────────────────────


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
    n_jobs: Optional[int] = None,
):
    """
    Ensemble Empirical Mode Decomposition (EEMD).

    Mitigates mode mixing by decomposing the signal multiple times with
    added white Gaussian noise, then averaging the IMFs across trials:

    .. math::
        \\bar{c}_k(t) = \\frac{1}{N_e} \\sum_{i=1}^{N_e} c_k^{(i)}(t)

    where :math:`c_k^{(i)}` is the *k*-th IMF from the *i*-th
    noise-perturbed trial and :math:`N_e` is the number of trials.

    The noise populates the time-frequency space uniformly, forcing the
    sifting process to separate scales consistently across trials. The
    residual noise in the averaged IMFs decreases as:

    .. math::
        \\epsilon \\sim \\frac{\\sigma_n}{\\sqrt{N_e}}

    where :math:`\\sigma_n` is the noise standard deviation.

    **Note:** EEMD does **not** guarantee exact reconstruction. For
    exact reconstruction, use :func:`ceemdan`.

    Args:
        signal: 1D input signal.
        num_trials: Number of noise-perturbed decomposition trials.
            Default is 100.
        noise_amplitude: Standard deviation of added noise as a fraction
            of the signal's standard deviation. Default is 0.2 (20%).
        stopping_criterion: Sifting stopping criterion for each trial.
        max_imfs: Maximum number of IMFs to extract per trial.
        max_sifts: Safety limit on sifting iterations per IMF.
        seed: Random seed for reproducibility.
        n_jobs: Number of parallel workers. ``None`` or ``1`` runs
            serially. ``-1`` uses all available CPU cores.

    Returns:
        Tuple of ``(imfs, residue)``:
            - **imfs**: shape ``(n_imfs, n_samples)``
            - **residue**: ``signal - sum(imfs)``

    Reference:
        Wu, Z. & Huang, N.E. (2009). "Ensemble empirical mode
        decomposition: a noise-assisted data analysis method."
        *Advances in Adaptive Data Analysis*, 1(1), 1-41.
    """
    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional.")
    if signal.size == 0:
        raise ValueError("Input signal must not be empty.")

    n_samples = len(signal)
    noise_std = noise_amplitude * np.std(signal)

    # Generate deterministic per-trial seeds from the master seed
    master_rng = np.random.default_rng(seed)
    trial_seeds = master_rng.integers(0, 2**63, size=num_trials)

    args_list = [
        (signal, int(trial_seeds[i]), noise_std, stopping_criterion, max_imfs, max_sifts)
        for i in range(num_trials)
    ]

    all_trial_imfs = _parallel_map(_eemd_single_trial, args_list, n_jobs)

    # Pad trials with fewer IMFs (treat missing IMFs as zero)
    max_n_imfs = max(len(t) for t in all_trial_imfs)
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
    n_jobs: Optional[int] = None,
):
    """
    Complete Ensemble EMD with Adaptive Noise (CEEMDAN).

    Improves on EEMD in two ways:

    1. **Exact reconstruction** — ``sum(imfs) + residue == signal``
       by construction.
    2. **Adaptive noise** — at each stage *k*, the noise contribution
       is the *k*-th IMF of the original noise realization, scaled to
       the current residue:

    .. math::
        \\tilde{r}_k^{(i)} = r_k + \\beta_k \\, E_k(w^{(i)})

    where :math:`r_k` is the residue after extracting *k* IMFs,
    :math:`E_k(w^{(i)})` is the *k*-th IMF of the *i*-th noise
    realization, and :math:`\\beta_k = \\epsilon \\cdot \\sigma(r_k)`.

    The *k*-th CEEMDAN IMF is then:

    .. math::
        \\tilde{c}_{k+1} = \\frac{1}{N_e}
            \\sum_{i=1}^{N_e} E_1\\!\\left(\\tilde{r}_k^{(i)}\\right)

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
        n_jobs: Number of parallel workers. ``None`` or ``1`` runs
            serially. ``-1`` uses all available CPU cores.

    Returns:
        Tuple of ``(imfs, residue)``:
            - **imfs**: shape ``(n_imfs, n_samples)``
            - **residue**: satisfies ``sum(imfs) + residue == signal``

    Reference:
        Torres, M.E. et al. (2011). "A complete ensemble empirical mode
        decomposition with adaptive noise." *IEEE International
        Conference on Acoustics, Speech and Signal Processing
        (ICASSP)*, 4144-4147.
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

    # Pre-generate noise realizations
    noise_realizations = [
        rng.normal(0, 1, n_samples) for _ in range(num_trials)
    ]

    # Decompose each noise realization (parallelizable)
    noise_args = [
        (noise, stopping_criterion, n_imfs_limit, max_sifts)
        for noise in noise_realizations
    ]
    noise_imfs_all = _parallel_map(_ceemdan_decompose_noise, noise_args, n_jobs)

    # Stage 1: Extract first IMF
    beta = noise_amplitude * np.std(signal)
    stage1_args = [
        (signal + beta * noise_realizations[i], stopping_criterion, max_sifts)
        for i in range(num_trials)
    ]
    first_imfs = _parallel_map(_ceemdan_stage_trial, stage1_args, n_jobs)

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

        # Build perturbed signals for this stage
        perturbed_list = []
        for i in range(num_trials):
            if k < len(noise_imfs_all[i]):
                noise_k = noise_imfs_all[i][k]
            else:
                noise_k = np.zeros(n_samples)
            perturbed_list.append(residue + beta_k * noise_k)

        stage_args = [
            (p, stopping_criterion, max_sifts) for p in perturbed_list
        ]
        stage_imfs = _parallel_map(_ceemdan_stage_trial, stage_args, n_jobs)

        ck = np.mean(stage_imfs, axis=0)
        residue = residue - ck
        imfs_list.append(ck)

    return np.array(imfs_list), residue


# ── Internal helpers ─────────────────────────────────────────────────


def _should_stop_ceemdan(residue: np.ndarray) -> bool:
    """Check if the residue can no longer be decomposed."""
    if is_monotonic(residue):
        return True

    maxima, minima = find_local_extrema(residue)
    if len(maxima) + len(minima) < 2:
        return True

    return False


def _resolve_n_jobs(n_jobs: Optional[int]) -> Optional[int]:
    """Resolve n_jobs to a concrete worker count, or None for serial."""
    if n_jobs is None or n_jobs == 1:
        return None
    if n_jobs == -1:
        import os
        return os.cpu_count()
    return n_jobs


def _parallel_map(func, args_list, n_jobs):
    """Run func over args_list, serially or in parallel."""
    workers = _resolve_n_jobs(n_jobs)

    if workers is None:
        return [func(args) for args in args_list]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(func, args) for args in args_list]
        # Preserve order
        return [f.result() for f in futures]
