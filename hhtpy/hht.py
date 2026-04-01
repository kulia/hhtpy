from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
from scipy.signal import hilbert
from ._emd_utils import find_local_extrema, get_freq_lim, EnvelopeOptions
from .emd import decompose


@dataclass
class IntrinsicModeFunction:
    """
    Dataclass to store the intrinsic mode function (IMF) and its instantaneous frequency.
    """

    signal: np.ndarray
    instantaneous_frequency: np.ndarray
    instantaneous_amplitude: np.ndarray
    sampling_frequency: float


def calculate_instantaneous_frequency_quadrature(
    imf: np.ndarray,
    sampling_frequency: float,
    normalize: bool = True,
    median_filter_window_pct: float = 0.05,
) -> np.ndarray:
    """
    Instantaneous frequency via the direct quadrature method.

    Constructs the analytic signal as :math:`z(t) = x(t) + i\\,q(t)` where
    the quadrature is computed directly from the normalized IMF:

    .. math::
        q(t) = \\operatorname{sign}\\!\\left(-\\frac{dx}{dt}\\right)
               \\sqrt{1 - x^2(t)}

    This avoids the Hilbert transform entirely, sidestepping the
    Bedrosian theorem limitation that causes errors when the amplitude
    and frequency spectra overlap (common in wideband IMFs).

    The IMF is first normalized to [-1, 1] by iteratively dividing by
    its spline-interpolated amplitude envelope, then the instantaneous
    frequency is extracted from the phase gradient and smoothed with a
    median filter.

    Args:
        imf: Input IMF signal.
        sampling_frequency: Sampling frequency in Hz.
        normalize: Whether to normalize the IMF first. Default True.
        median_filter_window_pct: Median filter window as a fraction of
            the sampling frequency. Default 0.05 (5%).

    Returns:
        Instantaneous frequency array (Hz), same length as ``imf``.

    Reference:
        Huang, N.E. et al. (2009). "On instantaneous frequency."
        *Advances in Adaptive Data Analysis*, 1(2), 177-229.
    """
    imf = imf.copy()

    if normalize:
        imf = normalize_imf(imf, max_attempts=150)

    frequency = quadrature_method(imf, sampling_frequency)
    frequency = median_filter(
        frequency, size=int(sampling_frequency * median_filter_window_pct)
    )

    return frequency


def normalize_imf(
    imf: np.ndarray, max_attempts: int = 150, crop_edges: float = 0.01
) -> np.ndarray:
    """
    Normalize an IMF to the range [-1, 1] by iterative amplitude division.

    At each iteration, the instantaneous amplitude envelope :math:`a(t)` is
    estimated via cubic spline interpolation through the peaks of
    :math:`|x(t)|`, and the IMF is divided pointwise:

    .. math::
        x^{(k+1)}(t) = \\frac{x^{(k)}(t)}{a^{(k)}(t)}

    Iteration stops when :math:`\\max|x(t)| \\leq 1`, or after
    ``max_attempts``. Edge samples are set to NaN to suppress boundary
    artifacts from the spline interpolation.

    Args:
        imf: Input IMF signal.
        max_attempts: Maximum normalization iterations. Default 150.
        crop_edges: Fraction of the signal to NaN at each end.
            Default 0.01 (1%). Set to 0 to disable.

    Returns:
        Normalized IMF with values in [-1, 1] (edges may be NaN).

    Raises:
        ValueError: If normalization does not converge.
    """
    for _ in np.arange(max_attempts):
        if np.nanmax(np.abs(imf)) <= 1:
            break

        imf /= calculate_instantaneous_amplitude_spline(imf)

    if crop_edges > 0:
        if crop_edges >= 0.5:
            raise ValueError(
                "Cannot crop whole signal. Must be less than 0.5, i.e., 50%."
            )

        crop_size = int(len(imf) * crop_edges)
        imf[:crop_size] = np.nan
        imf[-crop_size:] = np.nan

    if np.nanmax(np.abs(imf)) > 1:
        raise ValueError(
            "Normalization failed. Maximum absolute value is still greater than 1."
        )

    return imf


def calculate_instantaneous_amplitude_spline(imf: np.ndarray) -> np.ndarray:
    """
    Instantaneous amplitude via cubic spline through absolute-value peaks.

    Finds local maxima of :math:`|x(t)|`, adds the signal endpoints,
    and fits a cubic spline through these points to produce a smooth
    amplitude envelope :math:`a(t)`.

    Args:
        imf: Input IMF signal.

    Returns:
        Instantaneous amplitude array, same length as ``imf``.
    """
    x_max, _ = find_local_extrema(np.abs(imf))
    x_max = np.concatenate(([0], x_max, [len(imf) - 1]))

    n = np.arange(len(imf))
    return CubicSpline(x_max, abs(imf[x_max]))(n)


def _quadrature_phase(monocomponent_normalized: np.ndarray) -> np.ndarray:
    """
    Calculates the quadrature phase of the normalized IMF signal.

    The quadrature phase is given by:

    .. math::
        \\theta(t) = \\arctan{\\frac{q(t)}{x(t)}}

    where `q(t)` is the quadrature of the signal.

    Args:
        monocomponent_normalized (np.ndarray): IMF normalized between -1 and 1.

    Returns:
        np.ndarray: Quadrature phase :math:`\\theta(t)`.
    """
    if not isinstance(monocomponent_normalized, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    if not np.all(
        np.abs(monocomponent_normalized[~np.isnan(monocomponent_normalized)]) <= 1
    ):
        raise ValueError("Input values must be normalized between -1 and 1.")

    quadrature = _calculate_quadrature(monocomponent_normalized)
    z = monocomponent_normalized + 1j * quadrature
    return np.angle(z)


def quadrature_method(
    monocomponent_normalized: np.ndarray, sampling_frequency: float
) -> np.ndarray:
    """
    Compute instantaneous frequency from a normalized IMF via quadrature.

    Given a normalized IMF :math:`x(t) \\in [-1, 1]`, constructs the
    analytic signal :math:`z(t) = x(t) + i\\,q(t)` where :math:`q(t)` is
    the direct quadrature (see :func:`_calculate_quadrature`), then
    extracts frequency from the phase derivative:

    .. math::
        f(t) = \\frac{f_s}{2\\pi} \\left|\\frac{d}{dt}
               \\arg z(t)\\right|

    Args:
        monocomponent_normalized: IMF normalized to [-1, 1].
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        Instantaneous frequency array (Hz).

    Reference:
        Huang, N.E. et al. (2009). "On instantaneous frequency."
        *Advances in Adaptive Data Analysis*, 1(2), 177-229.
    """
    if not isinstance(monocomponent_normalized, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if not isinstance(sampling_frequency, (int, float)):
        raise ValueError("Sampling frequency must be a float or integer.")

    if not np.all(
        np.abs(monocomponent_normalized[~np.isnan(monocomponent_normalized)]) <= 1
    ):
        raise ValueError("Input values must be normalized between -1 and 1.")

    phase = _quadrature_phase(monocomponent_normalized)
    frequency = sampling_frequency / (2 * np.pi) * np.abs(np.gradient(phase))

    return frequency


def _calculate_quadrature(monocomponent: np.ndarray) -> np.ndarray:
    """
    Calculates the quadrature of the normalized IMF signal.

    The quadrature is calculated as:

    .. math::
        q(t) = \\text{sign}\\left(\\frac{dx(t)}{dt}\\right) \\cdot \\sqrt{1 - x^2(t)}

    Args:
        monocomponent (np.ndarray): IMF normalized between -1 and 1.

    Returns:
        np.ndarray: Quadrature :math:`q(t)`.
    """
    if not isinstance(monocomponent, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if not np.all(np.abs(monocomponent[~np.isnan(monocomponent)]) <= 1):
        raise ValueError("Input values must be normalized between -1 and 1.")

    # Calculate the sign based on the derivative of the signal
    sign = np.zeros_like(monocomponent)
    sign[:-1] = -np.sign(np.diff(monocomponent))
    sign[-1] = sign[-2]  # Handle the last element by copying the second last

    # Calculate the quadrature with numerical stability (handling small values)
    quadrature = sign * np.sqrt(np.maximum(0, 1 - monocomponent**2))

    return quadrature


def calculate_instantaneous_frequency_hilbert(
    imf: np.ndarray,
    sampling_frequency: float,
) -> np.ndarray:
    """
    Instantaneous frequency via the Hilbert transform.

    Constructs the analytic signal using ``scipy.signal.hilbert``:

    .. math::
        z(t) = x(t) + i\\,\\mathcal{H}[x(t)]

    then extracts instantaneous frequency from the unwrapped phase:

    .. math::
        f(t) = \\frac{f_s}{2\\pi} \\left|\\frac{d}{dt}
               \\arg z(t)\\right|

    This is the standard approach in signal processing. It works well
    when the Bedrosian condition is satisfied (amplitude spectrum and
    frequency spectrum do not overlap). For wideband IMFs, consider the
    quadrature method instead.

    Args:
        imf: Input IMF signal.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        Instantaneous frequency array (Hz), same length as ``imf``.

    Reference:
        Huang, N.E. et al. (1998). "The empirical mode decomposition
        and the Hilbert spectrum for nonlinear and non-stationary time
        series analysis." *Proceedings of the Royal Society A*, 454,
        903-995.
    """
    analytic_signal = hilbert(imf)
    phase = np.unwrap(np.angle(analytic_signal))
    frequency = sampling_frequency / (2 * np.pi) * np.abs(np.gradient(phase))
    return frequency


FrequencyCalculationMethod = Callable[[np.ndarray, float], np.ndarray]
AmplitudeCalculationMethod = Callable[[np.ndarray], np.ndarray]


def hilbert_huang_transform(
    signal: np.ndarray,
    sampling_frequency: float,
    frequency_calculation_method: FrequencyCalculationMethod = calculate_instantaneous_frequency_quadrature,
    amplitude_calculation_method: AmplitudeCalculationMethod = calculate_instantaneous_amplitude_spline,
    envelope_opts: Optional[EnvelopeOptions] = None,
    decompose_fn: Optional[Callable] = None,
) -> (list[IntrinsicModeFunction], np.ndarray):
    """
    Hilbert-Huang Transform: decompose a signal and compute time-frequency representation.

    Performs EMD (or a user-specified variant) to extract IMFs, then
    computes instantaneous frequency and amplitude for each IMF. The
    result can be used directly for Hilbert spectral analysis.

    The HHT was introduced by Huang et al. (1998) as a method for
    analyzing nonlinear and non-stationary signals. Unlike Fourier or
    wavelet methods, it makes no assumptions about linearity or
    stationarity.

    Args:
        signal: 1D input signal.
        sampling_frequency: Sampling frequency in Hz.
        frequency_calculation_method: Function to compute instantaneous
            frequency. Signature: ``(imf, fs) -> frequency``. Default
            is the direct quadrature method.
        amplitude_calculation_method: Function to compute instantaneous
            amplitude. Signature: ``(imf,) -> amplitude``. Default is
            cubic spline through absolute-value peaks.
        envelope_opts: Configuration for envelope interpolation during
            sifting. See :class:`EnvelopeOptions`. Ignored when
            ``decompose_fn`` is provided.
        decompose_fn: Custom decomposition function. Must accept
            ``signal`` as first argument and return ``(imfs, residue)``.
            Use ``functools.partial`` to bind extra arguments::

                from functools import partial
                decompose_fn=partial(eemd, num_trials=100, seed=42)

            Default is :func:`decompose` (standard EMD).

    Returns:
        Tuple of ``(imf_objects, residue)`` where ``imf_objects`` is a
        list of :class:`IntrinsicModeFunction` and ``residue`` is the
        decomposition residual.

    Reference:
        Huang, N.E. et al. (1998). "The empirical mode decomposition
        and the Hilbert spectrum for nonlinear and non-stationary time
        series analysis." *Proceedings of the Royal Society A*, 454,
        903-995.
    """
    if decompose_fn is not None:
        imfs, residue = decompose_fn(signal)
    else:
        imfs, residue = decompose(signal, envelope_opts=envelope_opts)

    return [
        IntrinsicModeFunction(
            signal=imf,
            instantaneous_frequency=frequency_calculation_method(
                imf, sampling_frequency
            ),
            instantaneous_amplitude=amplitude_calculation_method(imf),
            sampling_frequency=sampling_frequency,
        )
        for imf in imfs
    ], residue


def marginal_hilbert_spectrum(
    imfs: list[IntrinsicModeFunction], frequency_bin_size=None
):
    """
    Marginal Hilbert spectrum: time-integrated frequency-amplitude distribution.

    Integrates the instantaneous amplitude over time for each frequency
    bin, producing a one-dimensional spectrum analogous to the Fourier
    power spectrum but derived from the Hilbert spectral analysis:

    .. math::
        h(\\omega) = \\frac{1}{T} \\int_0^T H(\\omega, t)\\, dt

    where :math:`H(\\omega, t)` is the Hilbert spectrum (instantaneous
    amplitude as a function of time and frequency).

    Unlike the Fourier spectrum, the marginal Hilbert spectrum
    represents the total amplitude (energy) contribution from each
    frequency, accumulated over the entire signal duration.

    Args:
        imfs: List of :class:`IntrinsicModeFunction` objects (output of
            :func:`hilbert_huang_transform`).
        frequency_bin_size: Width of frequency bins in Hz. Default is
            ``sampling_frequency / signal_length`` (one bin per DFT
            frequency).

    Returns:
        Tuple of ``(frequencies, amplitudes)`` — both 1D arrays.

    Reference:
        Huang, N.E. et al. (1998). "The empirical mode decomposition
        and the Hilbert spectrum for nonlinear and non-stationary time
        series analysis." *Proceedings of the Royal Society A*, 454,
        903-995.
    """
    sampling_frequency = imfs[0].sampling_frequency

    if not frequency_bin_size:
        frequency_bin_size = sampling_frequency / len(imfs[0].signal)

    min_freq, max_freq = get_freq_lim(imfs)

    frequencies = np.arange(int(max_freq / frequency_bin_size)) * frequency_bin_size
    amplitudes = np.zeros(len(frequencies))

    for imf in imfs:
        freq = imf.instantaneous_frequency
        amp = imf.instantaneous_amplitude

        if len(freq) == 0:
            continue

        freq_lt = freq < max_freq
        freq_gt = freq > min_freq
        freq_cond_index = np.where(np.bitwise_and(freq_gt, freq_lt))[0]

        freq = freq[freq_cond_index]
        amp = amp[freq_cond_index]

        freq_intervals_imf = np.floor(freq / frequency_bin_size).astype(int)
        freq_intervals_imf = np.clip(freq_intervals_imf, 0, len(frequencies) - 1)
        amplitudes += np.bincount(
            freq_intervals_imf, weights=amp, minlength=len(frequencies)
        )

    amplitudes = amplitudes / len(imfs[0].signal)

    return frequencies, amplitudes


def index_of_orthogonality(imfs: np.ndarray) -> float:
    """
    Index of orthogonality for a set of IMFs.

    Measures how orthogonal the extracted IMFs are to each other. A value
    close to zero indicates a good decomposition where IMFs capture
    independent oscillatory modes with minimal energy leakage.

    Defined as the sum of all pairwise cross-energies, normalized by
    the total signal energy:

    .. math::
        IO = \\frac{\\sum_{i \\neq j} |\\langle c_i, c_j \\rangle|}
             {2 \\, \\sum_t x(t)^2}

    where :math:`c_i` are the IMFs and :math:`x(t) = \\sum_i c_i(t)`.

    Args:
        imfs: Array of IMF signals, shape ``(n_imfs, n_samples)``.
            This is the first return value of ``decompose()``.

    Returns:
        Index of orthogonality (float). Lower is better; 0 means
        perfectly orthogonal IMFs.

    Reference:
        Huang, N.E. et al. (1998). "The empirical mode decomposition
        and the Hilbert spectrum for nonlinear and non-stationary time
        series analysis." *Proceedings of the Royal Society A*, 454,
        903-995. (Section 3.4)
    """
    if len(imfs) < 2:
        return 0.0

    dot_products = np.dot(imfs, imfs.T)
    mask = ~np.eye(len(imfs), dtype=bool)
    cross_energy = np.abs(dot_products[mask]).sum()
    total_energy = np.sum(imfs.sum(axis=0) ** 2)

    if total_energy == 0:
        return 0.0

    return cross_energy / (2 * total_energy)
