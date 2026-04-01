"""
Masked EMD for mitigating mode mixing.

Masked (or masking signal) EMD adds a known sinusoidal mask to the signal
before sifting, then averages the results from opposite-phase masks to
cancel the mask contribution. This forces the sifting process to separate
components according to the mask frequency rather than relying on the
signal's own scale structure.

Two approaches are provided:

- **``masked_decompose``**: Masked EMD with explicit mask parameters.
  You specify the mask frequency and amplitude directly.

- **``adaptive_masked_decompose``**: Automatically estimates mask
  frequency and amplitude using one of several initialization
  strategies, then applies masked sifting for each IMF.

Mask initialization strategies:

- **Huang** (US Patent 20170116155): ``f_0 = max(zero-crossing frequency)``,
  ``a_0 = mean(amplitude)``. Simple and general.

- **Deering–Kaiser** (ICASSP 2005): ``f_0 = sum(a*f^2) / sum(a*f)``,
  ``a_0 = 1.6 * max(amplitude)``. Weighted frequency centroid — good
  for well-separated tones.

- **Spectral** (DFT-based): ``f_0`` is the highest-energy frequency
  from the signal's periodogram. Robust for broadband signals.
"""

from typing import Optional, Callable
import numpy as np
from scipy.signal import find_peaks, periodogram

from .emd import decompose
from ._emd_utils import sift, is_monotonic, find_local_extrema
from .sift_stopping_criteria import (
    SiftStoppingCriterion,
    get_stopping_criterion_fixed_number_of_sifts,
)


MaskInitMethod = Callable[[np.ndarray, float], tuple[float, float]]


def masked_decompose(
    signal: np.ndarray,
    mask_frequency: float,
    mask_amplitude: float,
    sampling_frequency: float,
    num_phase_shifts: int = 8,
    stopping_criterion: SiftStoppingCriterion = get_stopping_criterion_fixed_number_of_sifts(
        15
    ),
    max_imfs: Optional[int] = None,
    max_sifts: int = 100,
):
    """
    Masked EMD with explicit mask parameters.

    For each IMF level *j*, a sinusoidal mask is added to and subtracted
    from the signal at multiple phase shifts before sifting:

    .. math::
        m_j(t) = a \\cos(2\\pi f_j t + \\phi_k)

    where :math:`f_j = f_0 / 2^j` (dyadic frequency halving) and
    :math:`\\phi_k = k\\pi / K` for :math:`k = 0, \\ldots, K-1` phase
    shifts. The final IMF is the average over all masks:

    .. math::
        c_j(t) = \\frac{1}{K} \\sum_{k=1}^{K}
        \\frac{\\text{sift}(r + m_k) + \\text{sift}(r - m_k)}{2}

    The opposite-sign averaging cancels the mask contribution while
    preserving the signal component near :math:`f_j`.

    Args:
        signal: 1D input signal.
        mask_frequency: Base mask frequency :math:`f_0` in Hz.
        mask_amplitude: Mask amplitude *a* (absolute, not relative).
        sampling_frequency: Sampling frequency of the signal in Hz.
        num_phase_shifts: Number of phase shifts *K* to average over.
            Default is 8.
        stopping_criterion: Sifting stopping criterion.
        max_imfs: Maximum number of IMFs to extract.
        max_sifts: Safety limit on sifting iterations per IMF.

    Returns:
        Tuple of ``(imfs, residue)``:
            - **imfs**: shape ``(n_imfs, n_samples)``
            - **residue**: ``signal - sum(imfs)``

    Reference:
        Deering, R. & Kaiser, J.F. (2005). "The use of a masking signal
        to improve empirical mode decomposition." *IEEE ICASSP*,
        IV-485-488.
    """
    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional.")
    if signal.size == 0:
        raise ValueError("Input signal must not be empty.")

    n = len(signal)
    theoretical_max = int(np.log2(n) - 1)
    n_imfs_limit = (
        min(max_imfs, theoretical_max) if max_imfs is not None else theoretical_max
    )

    t = np.arange(n) / sampling_frequency
    residue = signal.copy()
    imfs = []

    for j in range(n_imfs_limit):
        if is_monotonic(residue):
            break

        maxima, minima = find_local_extrema(residue)
        if len(maxima) + len(minima) < 4:
            break

        # Mask frequency halves per IMF level
        f_mask = mask_frequency / (2 ** j)

        # Average over multiple phase-shifted masked sifts
        imf_sum = np.zeros(n)
        for k in range(num_phase_shifts):
            phase = k * np.pi / num_phase_shifts
            mask = mask_amplitude * np.cos(2 * np.pi * f_mask * t + phase)

            imf_plus = _sift_to_imf(
                residue + mask, stopping_criterion, max_sifts
            )
            imf_minus = _sift_to_imf(
                residue - mask, stopping_criterion, max_sifts
            )
            imf_sum += (imf_plus + imf_minus) / 2

        imf_avg = imf_sum / num_phase_shifts
        residue = residue - imf_avg
        imfs.append(imf_avg)

    if len(imfs) == 0:
        return np.empty((0, n)), signal.copy()

    return np.array(imfs), residue


def adaptive_masked_decompose(
    signal: np.ndarray,
    sampling_frequency: float,
    mask_init_method: MaskInitMethod = None,
    num_phase_shifts: int = 8,
    stopping_criterion: SiftStoppingCriterion = get_stopping_criterion_fixed_number_of_sifts(
        15
    ),
    max_imfs: Optional[int] = None,
    max_sifts: int = 100,
):
    """
    Adaptive masked EMD with automatic mask parameter estimation.

    Estimates the mask frequency :math:`f_0` and amplitude :math:`a_0`
    from the signal using the specified initialization strategy, then
    delegates to :func:`masked_decompose`.

    Three built-in strategies are provided:

    - :func:`mask_init_huang` — zero-crossing max frequency (default)
    - :func:`mask_init_deering_kaiser` — amplitude-weighted centroid
    - :func:`mask_init_spectral` — DFT peak frequency

    Args:
        signal: 1D input signal.
        sampling_frequency: Sampling frequency of the signal in Hz.
        mask_init_method: Strategy for estimating mask frequency and
            amplitude. Must accept ``(signal, sampling_frequency)`` and
            return ``(frequency, amplitude)``. Default is
            :func:`mask_init_huang`.
        num_phase_shifts: Number of phase shifts to average over.
        stopping_criterion: Sifting stopping criterion.
        max_imfs: Maximum number of IMFs to extract.
        max_sifts: Safety limit on sifting iterations per IMF.

    Returns:
        Tuple of ``(imfs, residue)``:
            - **imfs**: shape ``(n_imfs, n_samples)``
            - **residue**: ``signal - sum(imfs)``
    """
    if mask_init_method is None:
        mask_init_method = mask_init_huang

    f_mask, a_mask = mask_init_method(signal, sampling_frequency)

    return masked_decompose(
        signal=signal,
        mask_frequency=f_mask,
        mask_amplitude=a_mask,
        sampling_frequency=sampling_frequency,
        num_phase_shifts=num_phase_shifts,
        stopping_criterion=stopping_criterion,
        max_imfs=max_imfs,
        max_sifts=max_sifts,
    )


# ── Mask initialization strategies ────────────────────────────────────


def mask_init_huang(
    signal: np.ndarray, sampling_frequency: float
) -> tuple[float, float]:
    """
    Huang's mask initialization.

    .. math::
        f_0 = \\max(f_{zc}), \\quad a_0 = \\overline{|\\hat{x}|}

    where :math:`f_{zc}` is the piecewise zero-crossing frequency and
    :math:`\\hat{x}` is the peak envelope. Simple and general-purpose;
    works well when the highest frequency component dominates.

    Reference:
        Huang, N.E. (2017). US Patent Application 20170116155.
    """
    f_zc = _zero_crossing_frequency(signal, sampling_frequency)
    amplitude = _estimate_amplitude(signal)

    return np.max(f_zc) if len(f_zc) > 0 else sampling_frequency / 4, amplitude


def mask_init_deering_kaiser(
    signal: np.ndarray, sampling_frequency: float
) -> tuple[float, float]:
    """
    Deering & Kaiser mask initialization.

    .. math::
        f_0 = \\frac{\\sum a_i f_i^2}{\\sum a_i f_i}, \\quad
        a_0 = 1.6 \\cdot \\max(a_i)

    where :math:`a_i` and :math:`f_i` are the local amplitude and
    zero-crossing frequency at each half-cycle. The weighted centroid
    gives a robust frequency estimate for well-separated tones.

    Reference:
        Deering, R. & Kaiser, J.F. (2005). "The use of a masking signal
        to improve empirical mode decomposition." *IEEE ICASSP*,
        IV-485-488.
    """
    # Get a rough first IMF via a quick sift
    first_imf = signal.copy()
    for _ in range(5):
        first_imf = sift(first_imf)

    # Zero-crossing frequency and amplitude
    f_zc = _zero_crossing_frequency(first_imf, sampling_frequency)
    amp_env = _amplitude_at_zero_crossings(first_imf)

    if len(f_zc) == 0 or len(amp_env) == 0:
        return sampling_frequency / 4, 1.6 * np.std(signal)

    # Match lengths
    min_len = min(len(f_zc), len(amp_env))
    f_zc = f_zc[:min_len]
    amp_env = amp_env[:min_len]

    denom = np.sum(amp_env * f_zc)
    if denom < np.finfo(float).eps:
        return np.mean(f_zc), 1.6 * np.max(amp_env)

    f_0 = np.sum(amp_env * f_zc ** 2) / denom
    a_0 = 1.6 * np.max(amp_env)

    return f_0, a_0


def mask_init_spectral(
    signal: np.ndarray, sampling_frequency: float
) -> tuple[float, float]:
    """
    Spectral (DFT-based) mask initialization.

    .. math::
        f_0 = \\arg\\max_f S(f), \\quad
        a_0 = 2\\sqrt{S(f_0) \\cdot \\Delta f}

    where :math:`S(f)` is the Welch periodogram. Robust for broadband
    or multi-component signals where zero-crossing methods may be
    ambiguous.
    """
    freqs, psd = periodogram(signal, fs=sampling_frequency)

    # Ignore DC
    if len(freqs) > 1:
        freqs = freqs[1:]
        psd = psd[1:]

    if len(psd) == 0:
        return sampling_frequency / 4, np.std(signal)

    peak_idx = np.argmax(psd)
    f_0 = freqs[peak_idx]
    a_0 = 2 * np.sqrt(psd[peak_idx] * (freqs[1] - freqs[0]) if len(freqs) > 1 else psd[peak_idx])

    # Ensure amplitude is reasonable
    a_0 = max(a_0, 0.1 * np.std(signal))

    return f_0, a_0


# ── Internal helpers ──────────────────────────────────────────────────


def _sift_to_imf(
    signal: np.ndarray,
    stopping_criterion: SiftStoppingCriterion,
    max_sifts: int,
) -> np.ndarray:
    """Sift a signal to extract one IMF."""
    mode = signal.copy()
    total_sifts = 0

    while not stopping_criterion(mode, total_sifts):
        mode = sift(mode)
        total_sifts += 1
        if total_sifts >= max_sifts:
            break

    return mode


def _zero_crossing_frequency(
    signal: np.ndarray, sampling_frequency: float
) -> np.ndarray:
    """Compute piecewise zero-crossing frequency."""
    sign_changes = np.where(np.diff(np.sign(signal)))[0]

    if len(sign_changes) < 2:
        return np.array([])

    # Sub-sample zero crossings
    zc = np.empty(len(sign_changes))
    for i, idx in enumerate(sign_changes):
        x0, x1 = signal[idx], signal[idx + 1]
        denom = x1 - x0
        zc[i] = idx - x0 / denom if abs(denom) > np.finfo(float).eps else idx + 0.5

    half_periods = np.diff(zc)
    return sampling_frequency / (2 * half_periods)


def _estimate_amplitude(signal: np.ndarray) -> float:
    """Estimate mean amplitude from peak envelope."""
    maxima, _ = find_peaks(np.abs(signal))
    if len(maxima) > 0:
        return np.mean(np.abs(signal[maxima]))
    return np.std(signal)


def _amplitude_at_zero_crossings(signal: np.ndarray) -> np.ndarray:
    """Estimate local amplitude at each zero-crossing interval."""
    sign_changes = np.where(np.diff(np.sign(signal)))[0]
    if len(sign_changes) < 2:
        return np.array([])

    amplitudes = []
    for i in range(len(sign_changes) - 1):
        segment = signal[sign_changes[i]:sign_changes[i + 1] + 1]
        amplitudes.append(np.max(np.abs(segment)))

    return np.array(amplitudes)
