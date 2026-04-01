"""
Instantaneous frequency estimation methods.

All functions follow the ``FrequencyCalculationMethod`` signature::

    (imf: np.ndarray, sampling_frequency: float) -> np.ndarray

and can be passed directly to
``hilbert_huang_transform(frequency_calculation_method=...)``.

The two primary methods (quadrature and Hilbert transform) are in
:mod:`hhtpy.hht`. This module provides five additional methods from the
literature, each with different trade-offs between locality, robustness,
and normalization requirements.
"""

import numpy as np
from scipy.signal import find_peaks, hilbert


def calculate_instantaneous_frequency_zero_crossing(
    imf: np.ndarray,
    sampling_frequency: float,
) -> np.ndarray:
    """
    Instantaneous frequency from half-period zero-crossing intervals.

    Locates consecutive zero crossings (with sub-sample linear
    interpolation) and estimates frequency from the half-period:

    .. math::
        f(t) = \\frac{f_s}{2 \\, \\Delta t_{zc}}

    where :math:`\\Delta t_{zc}` is the interval between adjacent zero
    crossings. The result is a piecewise-constant frequency trace.

    Simple and robust — no normalization required.

    Args:
        imf: Input IMF signal.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        Instantaneous frequency array (Hz), same length as ``imf``.
    """
    n = len(imf)
    sign_changes = np.where(np.diff(np.sign(imf)))[0]

    if len(sign_changes) < 2:
        return np.full(n, np.nan)

    # Sub-sample zero-crossing positions via linear interpolation
    zc = np.empty(len(sign_changes))
    for i, idx in enumerate(sign_changes):
        x0, x1 = imf[idx], imf[idx + 1]
        zc[i] = idx - x0 / (x1 - x0)

    # Half-period between consecutive zero crossings
    half_periods = np.diff(zc)
    freq_at_zc = sampling_frequency / (2 * half_periods)

    # Assign piecewise-constant frequency
    frequency = np.full(n, np.nan)
    for i in range(len(freq_at_zc)):
        start = int(np.round(zc[i]))
        end = int(np.round(zc[i + 1])) + 1
        frequency[start:min(end, n)] = freq_at_zc[i]

    # Extend to edges
    frequency[:int(np.round(zc[0])) + 1] = freq_at_zc[0]
    frequency[int(np.round(zc[-1])):] = freq_at_zc[-1]

    return frequency


def calculate_instantaneous_frequency_generalized_zero_crossing(
    imf: np.ndarray,
    sampling_frequency: float,
) -> np.ndarray:
    """
    Generalized Zero-Crossing (GZC) method.

    Estimates instantaneous frequency by computing multiple period
    estimates at each reference point and averaging them. This produces
    a locally-computed frequency that avoids the phase-unwrapping issues
    of Hilbert-based methods.

    Four period types are used at each reference point:

    - :math:`T_1`: between successive maxima
    - :math:`T_2`: between successive minima
    - :math:`T_3`: between successive upward zero crossings
    - :math:`T_4`: between successive downward zero crossings

    The local frequency at each reference point is:

    .. math::
        f = \\frac{1}{4} \\sum_{j=1}^{4} \\frac{f_s}{T_j}

    The result is interpolated to the full sample grid. The GZC method
    is the most robust to noise among all IF estimation methods.

    Args:
        imf: Input IMF signal.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        Instantaneous frequency array (Hz), same length as ``imf``.

    Reference:
        Huang, N.E. et al. (2009). "On instantaneous frequency."
        *Advances in Adaptive Data Analysis*, 1(2), 177-229.
    """
    n = len(imf)

    # Find extrema
    maxima, _ = find_peaks(imf)
    minima, _ = find_peaks(-imf)

    # Find zero crossings with sub-sample precision
    sign = np.sign(imf)
    sign_diff = np.diff(sign)
    up_crossings_idx = np.where(sign_diff > 0)[0]
    down_crossings_idx = np.where(sign_diff < 0)[0]

    # Sub-sample interpolation for zero crossings
    up_crossings = _subsample_zero_crossings(imf, up_crossings_idx)
    down_crossings = _subsample_zero_crossings(imf, down_crossings_idx)

    # Collect (position, frequency) pairs from all period types
    positions = []
    frequencies = []

    # T1: successive maxima periods
    _add_successive_periods(maxima, maxima, sampling_frequency, positions, frequencies)
    # T2: successive minima periods
    _add_successive_periods(minima, minima, sampling_frequency, positions, frequencies)
    # T3: successive upward zero crossings
    _add_successive_periods(up_crossings, up_crossings, sampling_frequency, positions, frequencies)
    # T4: successive downward zero crossings
    _add_successive_periods(down_crossings, down_crossings, sampling_frequency, positions, frequencies)

    if len(positions) == 0:
        return np.full(n, np.nan)

    positions = np.array(positions)
    frequencies = np.array(frequencies)

    # Sort by position and interpolate to full sample grid
    order = np.argsort(positions)
    positions = positions[order]
    frequencies = frequencies[order]

    # Average duplicate positions
    unique_pos, inverse = np.unique(np.round(positions).astype(int), return_inverse=True)
    avg_freq = np.zeros(len(unique_pos))
    counts = np.zeros(len(unique_pos))
    for i, freq in enumerate(frequencies):
        avg_freq[inverse[i]] += freq
        counts[inverse[i]] += 1
    avg_freq /= counts

    # Interpolate to full grid
    unique_pos = np.clip(unique_pos, 0, n - 1)
    result = np.interp(np.arange(n), unique_pos, avg_freq)

    return result


def calculate_instantaneous_frequency_teo(
    imf: np.ndarray,
    sampling_frequency: float,
) -> np.ndarray:
    """
    Teager Energy Operator (TEO) method.

    Estimates instantaneous frequency using the Teager-Kaiser energy
    operator, which is extremely local (uses only 3 samples):

    .. math::
        \\psi[x(n)] = x(n)^2 - x(n-1) \\cdot x(n+1)

    .. math::
        f(n) = \\frac{f_s}{2\\pi} \\arccos\\left(1 - \\frac{\\psi[\\dot{x}(n)]}{2 \\psi[x(n)]}\\right)

    Very responsive to rapid frequency changes but sensitive to noise.
    Works best on narrowband, mono-component signals.

    Args:
        imf: Input IMF signal.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        Instantaneous frequency array (Hz), same length as ``imf``.

    References:
        Kaiser, J.F. (1990). "On a simple algorithm to calculate the
        'energy' of a signal." *IEEE ICASSP*, 381-384.

        Maragos, P., Kaiser, J.F. & Quatieri, T.F. (1993). "Energy
        separation in signal modulations with application to speech
        analysis." *IEEE Trans. Signal Processing*, 41(10), 3024-3051.
    """
    n = len(imf)

    # Backward difference (single-step, sample-based)
    dx = np.zeros(n)
    dx[1:] = imf[1:] - imf[:-1]
    dx[0] = dx[1]

    # Teager energy of x and dx
    psi_x = np.zeros(n)
    psi_x[1:-1] = imf[1:-1] ** 2 - imf[:-2] * imf[2:]

    psi_dx = np.zeros(n)
    psi_dx[1:-1] = dx[1:-1] ** 2 - dx[:-2] * dx[2:]

    # Avoid division by zero
    valid = psi_x > np.finfo(float).eps
    frequency = np.full(n, np.nan)

    ratio = np.zeros(n)
    ratio[valid] = psi_dx[valid] / (2 * psi_x[valid])
    # Clamp to valid arccos range
    ratio = np.clip(ratio, 0, 2)

    frequency[valid] = sampling_frequency / (2 * np.pi) * np.arccos(1 - ratio[valid])

    return frequency


def calculate_instantaneous_frequency_hou(
    imf: np.ndarray,
    sampling_frequency: float,
) -> np.ndarray:
    """
    Hou's arccos method.

    A direct, normalization-free frequency estimator derived from the
    cosine identity for sampled sinusoids:

    .. math::
        f(n) = \\frac{f_s}{2\\pi} \\arccos\\left(\\frac{x(n+1) + x(n-1)}{2\\,x(n)}\\right)

    Simple and efficient. Works well when the IMF amplitude varies
    slowly relative to the oscillation period. No normalization or
    Hilbert transform required.

    Args:
        imf: Input IMF signal.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        Instantaneous frequency array (Hz), same length as ``imf``.

    Reference:
        Hou, T.Y. & Shi, Z. (2013). "Data-driven time-frequency
        analysis." *Applied and Computational Harmonic Analysis*,
        35(2), 284-308.
    """
    n = len(imf)
    frequency = np.full(n, np.nan)

    # Interior points only
    x = imf
    valid = np.abs(x[1:-1]) > np.finfo(float).eps
    ratio = np.zeros(n - 2)
    ratio[valid] = (x[2:][valid] + x[:-2][valid]) / (2 * x[1:-1][valid])

    # Clamp to valid arccos range [-1, 1]
    ratio = np.clip(ratio, -1, 1)

    frequency[1:-1] = sampling_frequency / (2 * np.pi) * np.arccos(ratio)

    # Extend edges
    frequency[0] = frequency[1]
    frequency[-1] = frequency[-2]

    return frequency


def calculate_instantaneous_frequency_wu(
    imf: np.ndarray,
    sampling_frequency: float,
    normalize: bool = True,
) -> np.ndarray:
    """
    Wu's method.

    Derives instantaneous frequency from the normalized IMF and its
    quadrature (the companion signal 90° out of phase):

    .. math::
        f(n) = \\frac{f_s}{4\\pi} \\cdot \\frac{|x(n+1) - x(n-1)|}{|q(n)|}

    where the quadrature is:

    .. math::
        q(n) = \\operatorname{sign}\\!\\left(-\\frac{dx}{dt}\\right)
               \\sqrt{1 - x(n)^2}

    Requires a normalized IMF (amplitude in [-1, 1]). If ``normalize``
    is True (default), :func:`hhtpy.hht.normalize_imf` is called
    automatically.

    Args:
        imf: Input IMF signal.
        sampling_frequency: Sampling frequency in Hz.
        normalize: Whether to normalize the IMF first. Default True.

    Returns:
        Instantaneous frequency array (Hz), same length as ``imf``.

    Reference:
        Huang, N.E. et al. (2009). "On instantaneous frequency."
        *Advances in Adaptive Data Analysis*, 1(2), 177-229.
    """
    from .hht import normalize_imf

    x = imf.copy()
    if normalize:
        x = normalize_imf(x, max_attempts=150)

    n = len(x)

    # Quadrature
    sign = np.zeros(n)
    sign[:-1] = -np.sign(np.diff(x))
    sign[-1] = sign[-2]
    q = sign * np.sqrt(np.maximum(0, 1 - x ** 2))

    # Central difference
    dx = np.zeros(n)
    dx[1:-1] = x[2:] - x[:-2]

    # Frequency
    valid = np.abs(q) > np.finfo(float).eps
    frequency = np.full(n, np.nan)
    frequency[valid] = sampling_frequency / (4 * np.pi) * np.abs(dx[valid]) / np.abs(q[valid])

    return frequency


def despike_frequency(
    frequency: np.ndarray,
    imf: np.ndarray,
) -> np.ndarray:
    """
    Remove quadrature-induced frequency spikes at IMF extrema.

    The quadrature and Wu methods produce artificial frequency spikes
    at IMF extrema where :math:`q(t)` passes through zero (division
    by a near-zero denominator). This function detects extrema
    positions via :func:`scipy.signal.find_peaks` and replaces the
    frequency values there by linear interpolation from valid
    neighbors.

    Typically applied as a post-processing step::

        freq = calculate_instantaneous_frequency_quadrature(imf, fs)
        freq = despike_frequency(freq, imf)

    Args:
        frequency: Instantaneous frequency array (may contain NaN).
        imf: The IMF signal (used to locate extrema).

    Returns:
        Despiked frequency array (same length, copy of input).
    """
    frequency = frequency.copy()
    maxima, _ = find_peaks(imf)
    minima, _ = find_peaks(-imf)
    extrema = np.sort(np.concatenate([maxima, minima]))

    if len(extrema) == 0:
        return frequency

    # For each extremum, replace with interpolation from neighbors
    n = len(frequency)
    for idx in extrema:
        # Find valid neighbors (not NaN, not another extremum)
        left = idx - 1
        while left >= 0 and (np.isnan(frequency[left]) or left in extrema):
            left -= 1
        right = idx + 1
        while right < n and (np.isnan(frequency[right]) or right in extrema):
            right += 1

        if left >= 0 and right < n:
            # Linear interpolation
            t = (idx - left) / (right - left)
            frequency[idx] = frequency[left] + t * (frequency[right] - frequency[left])
        elif left >= 0:
            frequency[idx] = frequency[left]
        elif right < n:
            frequency[idx] = frequency[right]

    return frequency


def _subsample_zero_crossings(signal: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Sub-sample zero-crossing positions via linear interpolation."""
    if len(indices) == 0:
        return np.array([])
    x0 = signal[indices]
    x1 = signal[indices + 1]
    denom = x1 - x0
    safe = np.abs(denom) > np.finfo(float).eps
    result = indices.astype(float)
    result[safe] = indices[safe] - x0[safe] / denom[safe]
    return result


def _add_successive_periods(
    ref_points, period_markers, sampling_frequency, positions, frequencies
):
    """Add frequency estimates from successive period markers at midpoints."""
    if len(period_markers) < 2:
        return
    periods = np.diff(period_markers)
    midpoints = (period_markers[:-1] + period_markers[1:]) / 2
    freqs = sampling_frequency / periods

    for pos, freq in zip(midpoints, freqs):
        positions.append(pos)
        frequencies.append(freq)
