from scipy.signal import find_peaks
from numpy.typing import ArrayLike
from typing import Tuple, Optional
from typing import Union, List
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator


@dataclass
class EnvelopeOptions:
    """Configuration for envelope interpolation in the sifting process.

    Attributes:
        spline_method: Interpolation method for envelopes.
            ``"cubic"`` (default), ``"pchip"`` (monotone, no overshoot),
            or ``"akima"`` (smooth, reduced overshoot).
        boundary_mode: How to handle signal boundaries.
            ``"linear"`` (default): extrapolate from nearest 2 extrema.
            ``"mirror"``: reflect the nearest extremum across the boundary.
            ``"none"``: use the actual endpoint values directly.
    """

    spline_method: str = "cubic"
    boundary_mode: str = "linear"


# Module-level default for backwards compatibility
_DEFAULT_ENVELOPE_OPTS = EnvelopeOptions()


_SPLINE_CONSTRUCTORS = {
    "cubic": CubicSpline,
    "pchip": PchipInterpolator,
    "akima": Akima1DInterpolator,
}


def sift(
    mode: Union[List[float], np.ndarray],
    envelope_opts: Optional[EnvelopeOptions] = None,
):
    """
    One iteration of the sifting process: subtract the mean envelope.

    Computes upper and lower envelopes by interpolating through the
    local maxima and minima, then subtracts their mean:

    .. math::
        h(t) = x(t) - \\frac{e_{\\max}(t) + e_{\\min}(t)}{2}

    This is the fundamental operation of EMD — repeated application
    progressively removes the local mean until an IMF is obtained.

    Args:
        mode: Current signal/mode to sift.
        envelope_opts: Envelope interpolation options. If ``None``,
            uses cubic spline with linear boundary extrapolation.

    Returns:
        The sifted signal (same length as input).
    """
    if envelope_opts is None:
        envelope_opts = _DEFAULT_ENVELOPE_OPTS

    maxima_indices, minima_indices = find_local_extrema(mode)
    x_max, y_max = include_endpoints_in_extrema(
        maxima_indices, mode, extrema_type="maxima",
        boundary_mode=envelope_opts.boundary_mode,
    )
    x_min, y_min = include_endpoints_in_extrema(
        minima_indices, mode, extrema_type="minima",
        boundary_mode=envelope_opts.boundary_mode,
    )

    spline_cls = _SPLINE_CONSTRUCTORS.get(envelope_opts.spline_method)
    if spline_cls is None:
        raise ValueError(
            f"Unknown spline method '{envelope_opts.spline_method}'. "
            f"Choose from: {list(_SPLINE_CONSTRUCTORS.keys())}"
        )

    n = np.arange(len(mode))
    upper_envelope = spline_cls(x_max, y_max)(n)
    lower_envelope = spline_cls(x_min, y_min)(n)

    mean_envelope = 0.5 * (upper_envelope + lower_envelope)

    return mode - mean_envelope


def include_endpoints_in_extrema(
    x_extrema: np.ndarray,
    data: np.ndarray,
    extrema_type: str,
    boundary_mode: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Include the start and end points in the extrema indices and values,
    correcting for end effects in cubic spline interpolation.

    Args:
        x_extrema: Indices of the extrema used for interpolation.
        data: The input signal array.
        extrema_type: Type of extrema, either 'maxima' or 'minima'.
        boundary_mode: Boundary handling mode ('linear', 'mirror', 'none').
    """
    if extrema_type not in ["maxima", "minima"]:
        raise ValueError("extrema_type must be 'maxima' or 'minima'.")

    y_extrema = data[x_extrema]
    num_extrema = len(x_extrema)
    data_length = len(data)

    if boundary_mode == "mirror":
        return _endpoints_mirror(x_extrema, y_extrema, data, data_length)
    elif boundary_mode == "none":
        return _endpoints_none(x_extrema, y_extrema, data, data_length)
    else:
        # Default: linear extrapolation (original behavior)
        return _endpoints_linear(x_extrema, y_extrema, data, data_length, num_extrema)


def _endpoints_linear(x_extrema, y_extrema, data, data_length, num_extrema):
    """Original linear extrapolation boundary handling."""
    if num_extrema > 4:
        # Handle the start point
        if x_extrema[0] != 0:
            predicted_value = linear_interpolation_at_x(
                x_value=0,
                point_1=(x_extrema[0], y_extrema[0]),
                point_2=(x_extrema[1], y_extrema[1]),
            )
            x_extrema = np.insert(x_extrema, 0, 0)
            y_extrema = np.insert(y_extrema, 0, predicted_value)

        # Handle the end point
        if x_extrema[-1] != data_length - 1:
            predicted_value = linear_interpolation_at_x(
                x_value=data_length - 1,
                point_1=(x_extrema[-2], y_extrema[-2]),
                point_2=(x_extrema[-1], y_extrema[-1]),
            )
            x_extrema = np.append(x_extrema, data_length - 1)
            y_extrema = np.append(y_extrema, predicted_value)

    elif num_extrema >= 1:
        if x_extrema[0] != 0:
            x_extrema = np.insert(x_extrema, 0, 0)
            y_extrema = np.insert(y_extrema, 0, data[0])

        if x_extrema[-1] != data_length - 1:
            x_extrema = np.append(x_extrema, data_length - 1)
            y_extrema = np.append(y_extrema, data[-1])

    else:
        x_extrema = np.array([0, data_length - 1])
        y_extrema = np.array([data[0], data[-1]])

    return x_extrema, y_extrema


def _endpoints_mirror(x_extrema, y_extrema, data, data_length):
    """Mirror/reflect boundary handling.

    Reflects the nearest extremum across the boundary, creating a
    virtual extremum at ``-x_extrema[0]`` with the same value, and
    similarly at the end.
    """
    if len(x_extrema) == 0:
        return np.array([0, data_length - 1]), np.array([data[0], data[-1]])

    # Mirror at start: reflect first extremum across index 0
    if x_extrema[0] != 0:
        mirror_x = -x_extrema[0]
        mirror_y = y_extrema[0]
        x_extrema = np.insert(x_extrema, 0, mirror_x)
        y_extrema = np.insert(y_extrema, 0, mirror_y)

    # Mirror at end: reflect last extremum across index (data_length - 1)
    if x_extrema[-1] != data_length - 1:
        mirror_x = 2 * (data_length - 1) - x_extrema[-1]
        mirror_y = y_extrema[-1]
        x_extrema = np.append(x_extrema, mirror_x)
        y_extrema = np.append(y_extrema, mirror_y)

    return x_extrema, y_extrema


def _endpoints_none(x_extrema, y_extrema, data, data_length):
    """No special boundary handling — use actual endpoint values."""
    if len(x_extrema) == 0:
        return np.array([0, data_length - 1]), np.array([data[0], data[-1]])

    if x_extrema[0] != 0:
        x_extrema = np.insert(x_extrema, 0, 0)
        y_extrema = np.insert(y_extrema, 0, data[0])

    if x_extrema[-1] != data_length - 1:
        x_extrema = np.append(x_extrema, data_length - 1)
        y_extrema = np.append(y_extrema, data[-1])

    return x_extrema, y_extrema


def linear_interpolation_at_x(
    x_value: float, point_1: Tuple[float, float], point_2: Tuple[float, float]
) -> float:
    """
    Perform linear interpolation between two points.

    Parameters:
    - x_value (float): The x-coordinate at which to evaluate the interpolated y-value.
    - point_1 (_Tuple[float, float]): The first point as (x, y).
    - point_2 (_Tuple[float, float]): The second point as (x, y).

    Returns:
    - float: Interpolated y-value at x_value.
    """

    # Extract coordinates
    x1, y1 = point_1
    x2, y2 = point_2

    # Ensure that x1 != x2 to avoid division by zero
    if x1 == x2:
        raise ValueError(
            "x-coordinates of point_1 and point_2 must be different for interpolation."
        )

    # Perform linear interpolation using numpy
    return np.interp(x_value, [x1, x2], [y1, y2])


def get_extrema_indices(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the indices of local maxima and minima, handling saddle points.

    Args:
        data: The input signal array.

    Returns:
        Tuple of (maxima_indices, minima_indices).
    """
    # Identify where the slope changes for maxima and minima
    maxima_indices = (data[:-2] < data[1:-1]) & (data[2:] < data[1:-1])
    minima_indices = (data[:-2] > data[1:-1]) & (data[2:] > data[1:-1])

    # Find indices of the maxima and minima
    maxima_indices = np.where(maxima_indices)[0] + 1
    minima_indices = np.where(minima_indices)[0] + 1

    # Handle equal points (saddle points)
    maxima_indices, minima_indices = handle_saddle_points(
        data, maxima_indices, minima_indices
    )

    return np.sort(maxima_indices), np.sort(minima_indices)


def handle_saddle_points(
    data: np.ndarray, maxima_indices: np.ndarray, minima_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle saddle points where adjacent signal values are equal.

    Scans for plateaus (consecutive equal values) and classifies them
    as local maxima or minima by comparing the surrounding values.

    Args:
        data: The input signal array.
        maxima_indices: Detected maxima indices.
        minima_indices: Detected minima indices.

    Returns:
        Tuple of updated (maxima_indices, minima_indices).
    """
    equal_indices = np.where(np.diff(data) == 0)[0]

    for eq_index in equal_indices:
        # Check the surrounding values to determine if it's part of a saddle point
        left_value = data[eq_index - 1] if eq_index - 1 >= 0 else np.inf
        right_value = data[eq_index + 2] if eq_index + 2 < len(data) else np.inf

        if data[eq_index] > left_value and data[eq_index + 1] > right_value:
            maxima_indices = np.append(maxima_indices, eq_index + 1)
        elif data[eq_index] < left_value and data[eq_index + 1] < right_value:
            minima_indices = np.append(minima_indices, eq_index + 1)

    return maxima_indices, minima_indices


def is_monotonic(signal: ArrayLike) -> bool:
    """Return True if the signal is monotonically non-decreasing or non-increasing."""
    diff = np.diff(signal)
    return np.all(diff >= 0) or np.all(diff <= 0)


def is_imf(
    signal: ArrayLike, tolerance: float = 0.01, strict_mode: bool = False
) -> bool:
    """
    Check if the input array satisfies the criteria for an Intrinsic Mode Function (IMF).

    An IMF must satisfy two conditions:
    1. The number of extrema and the number of zero crossings must differ at most by one.
    2. The mean of the envelopes defined by the local maxima and minima is approximately zero.

    Args:
        signal (ArrayLike): Input array to check.
        tolerance (float): Tolerance for the mean envelope to be considered zero.

    Returns:
        bool: True if the array is an IMF, False otherwise.
    """
    # Calculate the differences between consecutive elements
    signal = np.array(signal, dtype=float)
    n = len(signal)

    if n < 3:
        return False  # An IMF requires at least 3 points

    signal -= np.mean(signal)
    diff = np.diff(signal)

    # Find where the first derivative changes sign: maxima and minima
    maxima = (diff[:-1] > 0) & (diff[1:] <= 0)  # Maxima points
    minima = (diff[:-1] < 0) & (diff[1:] >= 0)  # Minima points

    # Get the actual maxima and minima values
    maxima_values = signal[1:-1][maxima]
    minima_values = signal[1:-1][minima]

    # Check if all maxima are positive and all minima are negative
    if not (np.all(maxima_values > 0) and np.all(minima_values < 0)):
        return False

    if strict_mode:
        # Interpolate upper and lower envelopes
        x = np.arange(n)
        from scipy.interpolate import CubicSpline

        maxima_indices, minima_indices = find_local_extrema(signal)
        x_max, y_max = include_endpoints_in_extrema(
            maxima_indices, signal, extrema_type="maxima"
        )
        x_min, y_min = include_endpoints_in_extrema(
            minima_indices, signal, extrema_type="minima"
        )

        n = np.arange(len(signal))
        upper_envelope = CubicSpline(x_max, y_max)(n)
        lower_envelope = CubicSpline(x_min, y_min)(n)

        mean_envelope = 0.5 * (upper_envelope + lower_envelope)

        # Check if the mean envelope is approximately zero
        mean_abs = np.abs(mean_envelope)
        signal_amplitude = np.abs(signal)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized_mean = np.where(
                signal_amplitude != 0, mean_abs / signal_amplitude, 0
            )

        if np.max(normalized_mean) > tolerance:
            return False

    return True


def find_local_extrema(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the indices of local maxima and minima in the signal,
    handling saddle points (flat regions).

    Parameters:
        signal (np.ndarray): The input signal array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Indices of local maxima and minima.
    """
    # Find maxima, including plateaus
    maxima_indices, _ = find_peaks(signal, plateau_size=1)

    # Find minima by inverting the signal
    minima_indices, _ = find_peaks(-signal, plateau_size=1)

    return maxima_indices, minima_indices


def get_freq_lim(imfs, padding=0.1, percentile=None):
    """Compute frequency axis limits from a list of IMFs for plotting."""
    all_freqs = np.concatenate(
        [imf.instantaneous_frequency[~np.isnan(imf.instantaneous_frequency)] for imf in imfs]
    )
    if percentile is not None:
        min_freq = np.percentile(all_freqs, percentile)
        max_freq = np.percentile(all_freqs, 100 - percentile)
    else:
        min_freq = np.min(all_freqs)
        max_freq = np.max(all_freqs)
    return min_freq * (1 - padding), max_freq * (1 + padding)
