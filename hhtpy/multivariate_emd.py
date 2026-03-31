"""
Multivariate Empirical Mode Decomposition (Rehman & Mandic, 2010).

Extends EMD to multi-channel signals by computing envelopes via
projections onto uniformly distributed direction vectors on the
unit hypersphere, generated using the Hammersley quasi-random sequence.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from ._emd_utils import is_monotonic


def memd(
    signal: np.ndarray,
    num_directions: int = 64,
    max_imfs: int = None,
    max_sifts: int = 100,
    stop_threshold: float = 0.075,
):
    """
    Multivariate Empirical Mode Decomposition (Rehman & Mandic, 2010).

    Decomposes a multi-channel signal into multivariate Intrinsic Mode
    Functions (IMFs). Each IMF is a matrix of the same shape as the input
    signal, with aligned oscillatory modes across all channels.

    Unlike applying univariate EMD to each channel separately, MEMD
    ensures that common oscillatory scales across channels are captured
    in the same IMF index (the *alignment property*).

    Args:
        signal: Input signal of shape ``(n_channels, n_samples)``.
        num_directions: Number of direction vectors on the unit
            hypersphere. Must be >= 2 * n_channels. Default is 64.
        max_imfs: Maximum number of IMFs to extract. If ``None``,
            uses ``floor(log2(n_samples)) - 1``.
        max_sifts: Maximum sifting iterations per IMF. Default is 100.
        stop_threshold: Normalized mean envelope threshold for the
            sifting stopping criterion. Default is 0.075.

    Returns:
        Tuple of (imfs, residue):
            - imfs (np.ndarray): Shape ``(n_imfs, n_channels, n_samples)``.
            - residue (np.ndarray): Shape ``(n_channels, n_samples)``.
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 2:
        raise ValueError(
            f"Input must be 2D (n_channels, n_samples), got {signal.ndim}D."
        )

    n_channels, n_samples = signal.shape

    if n_channels >= n_samples:
        raise ValueError(
            f"First axis should be channels ({n_channels}) and second axis "
            f"samples ({n_samples}), but n_channels >= n_samples. "
            f"Did you mean to transpose the input?"
        )

    if num_directions < 2 * n_channels:
        raise ValueError(
            f"num_directions ({num_directions}) must be >= "
            f"2 * n_channels ({2 * n_channels})."
        )

    theoretical_max = int(np.log2(n_samples) - 1)
    n_imfs_limit = (
        min(max_imfs, theoretical_max) if max_imfs is not None else theoretical_max
    )

    # Generate direction vectors on the unit hypersphere
    directions = _hammersley_directions(num_directions, n_channels)

    residue = signal.copy()
    imfs = []

    for _ in range(n_imfs_limit):
        # Check if residue is monotonic in all channels
        if all(is_monotonic(residue[ch]) for ch in range(n_channels)):
            break

        # Check if residue has enough extrema for meaningful decomposition
        if not _has_enough_extrema(residue):
            break

        # Sifting process
        proto_imf = residue.copy()

        for sift_iter in range(max_sifts):
            mean_envelope = _multivariate_mean_envelope(
                proto_imf, directions, n_samples
            )

            # Stopping criterion: normalized mean envelope power
            mean_power = np.sum(mean_envelope ** 2)
            signal_power = np.sum(proto_imf ** 2)

            if signal_power > 0:
                ratio = np.sqrt(mean_power / signal_power)
            else:
                ratio = 0.0

            proto_imf = proto_imf - mean_envelope

            if ratio < stop_threshold and sift_iter > 0:
                break

        residue = residue - proto_imf
        imfs.append(proto_imf)

    if len(imfs) == 0:
        return np.empty((0, n_channels, n_samples)), residue

    return np.array(imfs), residue


def _multivariate_mean_envelope(
    signal: np.ndarray, directions: np.ndarray, n_samples: int
) -> np.ndarray:
    """
    Compute the multivariate mean envelope by projecting the signal
    onto each direction, computing upper/lower envelopes of the
    projection, and averaging the back-projected mean envelopes.
    """
    n_channels = signal.shape[0]
    n_directions = len(directions)
    mean_env = np.zeros((n_channels, n_samples))
    t = np.arange(n_samples)

    for direction in directions:
        # Project multivariate signal onto this direction (scalar time series)
        projection = direction @ signal  # shape (n_samples,)

        # Find extrema of the projection
        upper = _envelope_spline(projection, t, kind="upper")
        lower = _envelope_spline(projection, t, kind="lower")

        # Mean of projection envelopes
        proj_mean = (upper + lower) / 2.0

        # Back-project: multiply scalar mean by direction vector
        mean_env += np.outer(direction, proj_mean)

    mean_env /= n_directions
    return mean_env


def _envelope_spline(
    signal: np.ndarray, t: np.ndarray, kind: str
) -> np.ndarray:
    """Compute the upper or lower envelope via cubic spline interpolation."""
    from scipy.signal import find_peaks

    if kind == "upper":
        peaks, _ = find_peaks(signal)
    else:
        peaks, _ = find_peaks(-signal)

    if len(peaks) < 2:
        # Not enough extrema — return the signal itself
        return signal.copy()

    # Include endpoints using linear extrapolation
    x_pts = np.concatenate(([0], peaks, [len(signal) - 1]))
    y_pts = signal[x_pts].copy()

    # Extrapolate endpoints
    if peaks[0] != 0:
        slope = (signal[peaks[1]] - signal[peaks[0]]) / (peaks[1] - peaks[0])
        y_pts[0] = signal[peaks[0]] + slope * (0 - peaks[0])
    if peaks[-1] != len(signal) - 1:
        slope = (signal[peaks[-1]] - signal[peaks[-2]]) / (peaks[-1] - peaks[-2])
        y_pts[-1] = signal[peaks[-1]] + slope * (len(signal) - 1 - peaks[-1])

    return CubicSpline(x_pts, y_pts)(t)


def _has_enough_extrema(signal: np.ndarray) -> bool:
    """Check that at least one channel has enough extrema for sifting."""
    from scipy.signal import find_peaks

    for ch in range(signal.shape[0]):
        maxima, _ = find_peaks(signal[ch])
        minima, _ = find_peaks(-signal[ch])
        if len(maxima) >= 2 and len(minima) >= 2:
            return True
    return False


def _hammersley_directions(n_points: int, n_dims: int) -> np.ndarray:
    """
    Generate quasi-uniform direction vectors on the unit hypersphere
    using the Hammersley low-discrepancy sequence.

    Args:
        n_points: Number of direction vectors to generate.
        n_dims: Dimensionality (number of channels).

    Returns:
        Array of shape ``(n_points, n_dims)`` with unit vectors.
    """
    if n_dims == 1:
        return np.array([[1.0]] * n_points)

    # Generate Hammersley sequence in [0, 1)^(n_dims-1)
    seq = np.zeros((n_points, n_dims - 1))

    # First coordinate: uniform spacing
    seq[:, 0] = np.arange(n_points) / n_points

    # Remaining coordinates: Van der Corput sequences in prime bases
    primes = _first_primes(n_dims - 2)
    for j, p in enumerate(primes):
        seq[:, j + 1] = _van_der_corput(n_points, p)

    # Map [0,1)^(n_dims-1) to unit sphere S^(n_dims-1)
    if n_dims == 2:
        # Simple case: circle
        angles = seq[:, 0] * 2 * np.pi
        directions = np.column_stack([np.cos(angles), np.sin(angles)])
    elif n_dims == 3:
        # Sphere S^2
        cos_theta = 2 * seq[:, 0] - 1  # [-1, 1]
        phi = seq[:, 1] * 2 * np.pi  # [0, 2*pi)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        directions = np.column_stack([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        ])
    else:
        # General n_dims: generalized spherical coordinates
        directions = _hammersley_to_sphere(seq, n_dims)

    return directions


def _hammersley_to_sphere(seq: np.ndarray, n_dims: int) -> np.ndarray:
    """Map Hammersley points in [0,1)^(n_dims-1) to S^(n_dims-1)."""
    n_points = len(seq)
    # Map to [-1, 1]
    b = 2 * seq - 1

    # Compute angles using arctan2
    directions = np.zeros((n_points, n_dims))

    for i in range(n_points):
        angles = np.zeros(n_dims - 1)

        for j in range(n_dims - 2):
            tail_sum = np.sqrt(np.sum(b[i, j + 1:] ** 2))
            angles[j] = np.arctan2(tail_sum, b[i, j])

        angles[-1] = np.arctan2(b[i, -1], np.sqrt(np.sum(b[i, :-1] ** 2)))
        angles[-1] = (angles[-1] + np.pi)  # shift to [0, 2*pi)

        # Convert spherical angles to Cartesian coordinates
        directions[i, 0] = np.cos(angles[0])
        sin_prod = 1.0
        for j in range(1, n_dims - 1):
            sin_prod *= np.sin(angles[j - 1])
            directions[i, j] = sin_prod * np.cos(angles[j])
        sin_prod *= np.sin(angles[-1])
        directions[i, -1] = sin_prod

    return directions


def _van_der_corput(n: int, base: int) -> np.ndarray:
    """Generate the first n elements of the Van der Corput sequence."""
    result = np.zeros(n)
    for i in range(n):
        f = 1.0
        r = 0.0
        val = i
        while val > 0:
            f /= base
            r += f * (val % base)
            val //= base
        result[i] = r
    return result


def _first_primes(count: int) -> list:
    """Return the first `count` prime numbers."""
    if count <= 0:
        return []
    primes = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes
