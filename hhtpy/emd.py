from typing import List, Optional
import numpy as np
from hhtpy._emd_utils import is_monotonic, is_imf, sift, EnvelopeOptions
from .sift_stopping_criteria import (
    get_stopping_criterion_fixed_number_of_sifts,
    SiftStoppingCriterion,
)


def decompose(
    signal: np.ndarray,
    stopping_criterion: SiftStoppingCriterion = get_stopping_criterion_fixed_number_of_sifts(
        15
    ),
    max_imfs: Optional[int] = None,
    max_sifts: int = 100,
    envelope_opts: Optional[EnvelopeOptions] = None,
):
    """
    Perform the Empirical Mode Decomposition on a given signal.

    Args:
        signal: The input signal to decompose.
        stopping_criterion: The stopping criterion to use for the sifting
            process. See ``sift_stopping_criteria`` for available options.
        max_imfs: Maximum number of IMFs to extract. If ``None`` (default),
            uses the theoretical maximum ``floor(log2(N)) - 1``.
        max_sifts: Safety limit on sifting iterations per IMF. Prevents
            non-convergence with adaptive criteria. Default is 100.
        envelope_opts: Configuration for envelope interpolation. Controls
            the spline method (``"cubic"``, ``"pchip"``, ``"akima"``) and
            boundary handling (``"linear"``, ``"mirror"``, ``"none"``).
            Default is cubic spline with linear boundary extrapolation.

    Returns:
        Tuple of (imfs, residue):
            - imfs (np.ndarray): Array of shape ``(n_imfs, n_samples)``.
            - residue (np.ndarray): The residual signal after decomposition.
    """
    if signal.size == 0:
        raise ValueError("Input signal must not be empty.")
    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional.")

    signal_std = np.std(signal)
    signal_mean = np.mean(signal)

    if signal_std == 0:
        raise ValueError(
            "Input signal is constant (zero variance). EMD requires a non-constant signal."
        )

    signal_normalized = (signal - signal_mean) / signal_std

    theoretical_max = int(np.log2(len(signal)) - 1)
    n_imfs_limit = min(max_imfs, theoretical_max) if max_imfs is not None else theoretical_max

    residue = signal_normalized
    imfs: List[np.ndarray] = []

    for i in range(n_imfs_limit):
        if is_monotonic(residue):
            break

        if is_imf(residue):
            imfs.append(residue)
            residue = np.zeros_like(residue)
            break

        mode = residue
        total_sifts_performed = 0

        while not stopping_criterion(mode, total_sifts_performed):
            mode = sift(mode, envelope_opts=envelope_opts)
            total_sifts_performed += 1
            if total_sifts_performed >= max_sifts:
                break

        residue -= mode
        imfs.append(mode)

    return np.array(imfs) * signal_std, residue * signal_std + signal_mean
