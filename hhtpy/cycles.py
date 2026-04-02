"""
Cycle analysis for Intrinsic Mode Functions.

Detects individual oscillatory cycles in IMFs and characterizes them
by amplitude, frequency, duration, and waveform shape. This enables
cycle-by-cycle analysis of non-stationary oscillations — essential for
neuroscience, ocean wave analysis, and vibration monitoring where
cycle-to-cycle variability carries meaningful information.

A cycle is delimited by consecutive ascending zero crossings of the
IMF. Within each cycle, four **control points** are identified:

1. Ascending zero crossing (cycle start)
2. Peak (local maximum)
3. Descending zero crossing
4. Trough (local minimum)

These control points define shape metrics:

- **Peak-trough asymmetry**: ratio of rise time to fall time
- **Ascending-descending ratio**: fraction of cycle spent ascending

Reference:
    Cole, S. & Voytek, B. (2019). "Cycle-by-cycle analysis of neural
    oscillations." *Journal of Neurophysiology*, 122(2), 849-861.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.signal import find_peaks


@dataclass
class Cycle:
    """One oscillatory cycle from an IMF.

    Attributes:
        index: Cycle index (0-based).
        start_sample: Sample index of the ascending zero crossing
            (cycle start).
        end_sample: Sample index of the next ascending zero crossing
            (cycle end, exclusive).
        peak_sample: Sample index of the cycle's maximum.
        trough_sample: Sample index of the cycle's minimum.
        descending_zero_sample: Sample index of the descending zero
            crossing (between peak and trough). ``-1`` if not found.
        duration_samples: Number of samples in the cycle.
        duration_seconds: Duration in seconds.
        frequency: Cycle frequency in Hz (``1 / duration_seconds``).
        amplitude: Half the peak-to-trough range:
            ``(peak_value - trough_value) / 2``.
        peak_value: Signal value at the peak.
        trough_value: Signal value at the trough.
        rise_fraction: Fraction of the cycle from start to peak
            (0 to 1). Values > 0.5 indicate a slow rise / fast fall.
        decay_fraction: Fraction of the cycle from peak to descending
            zero crossing (0 to 1).
        peak_trough_symmetry: ``rise_fraction / (1 - rise_fraction)``.
            Values > 1 mean the rise is slower than the fall.
        is_complete: True if all four control points were found.
    """

    index: int
    start_sample: int
    end_sample: int
    peak_sample: int
    trough_sample: int
    descending_zero_sample: int
    duration_samples: int
    duration_seconds: float
    frequency: float
    amplitude: float
    peak_value: float
    trough_value: float
    rise_fraction: float
    decay_fraction: float
    peak_trough_symmetry: float
    is_complete: bool


def detect_cycles(
    imf: np.ndarray,
    sampling_frequency: float,
    min_samples: int = 4,
) -> list[Cycle]:
    """
    Detect and characterize individual oscillatory cycles in an IMF.

    Cycles are delimited by ascending zero crossings. Within each
    cycle, the four control points (ascending zero, peak, descending
    zero, trough) are located and used to compute per-cycle metrics.

    Args:
        imf: 1D IMF signal.
        sampling_frequency: Sampling frequency in Hz.
        min_samples: Minimum number of samples for a valid cycle.
            Cycles shorter than this are discarded. Default is 4.

    Returns:
        List of :class:`Cycle` objects, one per detected cycle.

    Example:
        >>> from hhtpy import decompose
        >>> from hhtpy.cycles import detect_cycles
        >>> imfs, _ = decompose(signal)
        >>> cycles = detect_cycles(imfs[0], sampling_frequency=1000)
        >>> for c in cycles[:5]:
        ...     print(f"Cycle {c.index}: freq={c.frequency:.1f} Hz, "
        ...           f"amp={c.amplitude:.3f}")
    """
    imf = np.asarray(imf, dtype=float)
    if imf.ndim != 1:
        raise ValueError("IMF must be one-dimensional.")

    # Find ascending zero crossings (negative → positive)
    sign = np.sign(imf)
    sign_diff = np.diff(sign)
    ascending_zc_idx = np.where(sign_diff > 0)[0]

    if len(ascending_zc_idx) < 2:
        return []

    # Sub-sample zero-crossing positions
    ascending_zc = _subsample_zc(imf, ascending_zc_idx)

    cycles = []
    cycle_idx = 0

    for i in range(len(ascending_zc) - 1):
        start = ascending_zc_idx[i]
        end = ascending_zc_idx[i + 1]
        duration_samples = end - start

        if duration_samples < min_samples:
            continue

        segment = imf[start:end + 1]

        # Find peak (largest maximum in the segment)
        local_maxima, _ = find_peaks(segment)
        if len(local_maxima) == 0:
            peak_local = np.argmax(segment)
        else:
            peak_local = local_maxima[np.argmax(segment[local_maxima])]
        peak_sample = start + peak_local
        peak_value = imf[peak_sample]

        # Find trough (largest minimum in the segment)
        local_minima, _ = find_peaks(-segment)
        if len(local_minima) == 0:
            trough_local = np.argmin(segment)
        else:
            trough_local = local_minima[np.argmin(segment[local_minima])]
        trough_sample = start + trough_local
        trough_value = imf[trough_sample]

        # Find descending zero crossing (between peak and trough)
        desc_zc_sample = _find_descending_zc(imf, peak_sample, end)

        # Metrics
        duration_seconds = duration_samples / sampling_frequency
        frequency = 1.0 / duration_seconds if duration_seconds > 0 else np.nan
        amplitude = (peak_value - trough_value) / 2.0

        rise_fraction = (peak_sample - start) / duration_samples
        if desc_zc_sample >= 0:
            decay_fraction = (desc_zc_sample - peak_sample) / duration_samples
        else:
            decay_fraction = np.nan

        fall_fraction = 1.0 - rise_fraction
        if fall_fraction > 0:
            peak_trough_symmetry = rise_fraction / fall_fraction
        else:
            peak_trough_symmetry = np.nan

        is_complete = (
            peak_value > 0
            and trough_value < 0
            and desc_zc_sample >= 0
            and peak_sample < trough_sample
        )

        cycles.append(Cycle(
            index=cycle_idx,
            start_sample=start,
            end_sample=end,
            peak_sample=peak_sample,
            trough_sample=trough_sample,
            descending_zero_sample=desc_zc_sample,
            duration_samples=duration_samples,
            duration_seconds=duration_seconds,
            frequency=frequency,
            amplitude=amplitude,
            peak_value=peak_value,
            trough_value=trough_value,
            rise_fraction=rise_fraction,
            decay_fraction=decay_fraction,
            peak_trough_symmetry=peak_trough_symmetry,
            is_complete=is_complete,
        ))
        cycle_idx += 1

    return cycles


def get_cycle_vector(
    imf: np.ndarray,
    min_samples: int = 4,
) -> np.ndarray:
    """
    Label each sample with its cycle index.

    Returns an integer array where sample ``n`` has value ``k`` if it
    belongs to cycle ``k`` (1-based). Samples not belonging to any
    cycle have value 0.

    Args:
        imf: 1D IMF signal.
        min_samples: Minimum cycle length in samples.

    Returns:
        Integer array, same length as ``imf``.
    """
    imf = np.asarray(imf, dtype=float)
    cycle_vec = np.zeros(len(imf), dtype=int)

    sign = np.sign(imf)
    sign_diff = np.diff(sign)
    ascending_zc_idx = np.where(sign_diff > 0)[0]

    if len(ascending_zc_idx) < 2:
        return cycle_vec

    label = 1
    for i in range(len(ascending_zc_idx) - 1):
        start = ascending_zc_idx[i]
        end = ascending_zc_idx[i + 1]
        if end - start >= min_samples:
            cycle_vec[start:end] = label
            label += 1

    return cycle_vec


def get_cycle_stat(
    cycle_vector: np.ndarray,
    values: np.ndarray,
    func: Callable = np.mean,
) -> np.ndarray:
    """
    Compute a per-cycle aggregate statistic.

    Applies ``func`` to the values within each cycle and returns one
    result per cycle.

    Args:
        cycle_vector: Integer cycle labels from :func:`get_cycle_vector`.
        values: Array of values to aggregate (same length as
            ``cycle_vector``). Typically instantaneous frequency or
            amplitude.
        func: Aggregation function. Default is ``np.mean``.

    Returns:
        1D array of length ``n_cycles``.

    Example:
        >>> cv = get_cycle_vector(imf)
        >>> mean_freq = get_cycle_stat(cv, inst_freq, func=np.nanmean)
    """
    cycle_vector = np.asarray(cycle_vector, dtype=int)
    values = np.asarray(values, dtype=float)

    labels = np.unique(cycle_vector)
    labels = labels[labels > 0]  # skip 0 (no cycle)

    result = np.empty(len(labels))
    for i, label in enumerate(labels):
        mask = cycle_vector == label
        result[i] = func(values[mask])

    return result


def phase_align(
    imf: np.ndarray,
    cycle_vector: Optional[np.ndarray] = None,
    n_points: int = 48,
) -> np.ndarray:
    """
    Phase-align detected cycles to a common grid.

    Resamples each cycle to ``n_points`` equally spaced samples,
    enabling direct comparison and averaging of cycles with different
    durations.

    Args:
        imf: 1D IMF signal.
        cycle_vector: Integer cycle labels from :func:`get_cycle_vector`.
            If ``None``, computed automatically.
        n_points: Number of points in the aligned output per cycle.
            Default is 48 (corresponding to 7.5° phase resolution).

    Returns:
        Array of shape ``(n_cycles, n_points)`` with phase-aligned
        waveforms.
    """
    imf = np.asarray(imf, dtype=float)

    if cycle_vector is None:
        cycle_vector = get_cycle_vector(imf)

    labels = np.unique(cycle_vector)
    labels = labels[labels > 0]

    if len(labels) == 0:
        return np.empty((0, n_points))

    aligned = np.empty((len(labels), n_points))
    target_phase = np.linspace(0, 1, n_points, endpoint=False)

    for i, label in enumerate(labels):
        mask = cycle_vector == label
        segment = imf[mask]
        n = len(segment)
        source_phase = np.linspace(0, 1, n, endpoint=False)
        aligned[i] = np.interp(target_phase, source_phase, segment)

    return aligned


def cycle_summary_table(cycles: list[Cycle]) -> dict[str, np.ndarray]:
    """
    Convert a list of Cycle objects to a dict of arrays (table format).

    Useful for quick analysis with numpy or conversion to a DataFrame::

        import pandas as pd
        df = pd.DataFrame(cycle_summary_table(cycles))

    Args:
        cycles: List of :class:`Cycle` objects from :func:`detect_cycles`.

    Returns:
        Dict mapping field names to 1D arrays.
    """
    if len(cycles) == 0:
        return {
            "index": np.array([], dtype=int),
            "start_sample": np.array([], dtype=int),
            "end_sample": np.array([], dtype=int),
            "duration_seconds": np.array([]),
            "frequency": np.array([]),
            "amplitude": np.array([]),
            "peak_value": np.array([]),
            "trough_value": np.array([]),
            "rise_fraction": np.array([]),
            "decay_fraction": np.array([]),
            "peak_trough_symmetry": np.array([]),
            "is_complete": np.array([], dtype=bool),
        }

    return {
        "index": np.array([c.index for c in cycles]),
        "start_sample": np.array([c.start_sample for c in cycles]),
        "end_sample": np.array([c.end_sample for c in cycles]),
        "duration_seconds": np.array([c.duration_seconds for c in cycles]),
        "frequency": np.array([c.frequency for c in cycles]),
        "amplitude": np.array([c.amplitude for c in cycles]),
        "peak_value": np.array([c.peak_value for c in cycles]),
        "trough_value": np.array([c.trough_value for c in cycles]),
        "rise_fraction": np.array([c.rise_fraction for c in cycles]),
        "decay_fraction": np.array([c.decay_fraction for c in cycles]),
        "peak_trough_symmetry": np.array([c.peak_trough_symmetry for c in cycles]),
        "is_complete": np.array([c.is_complete for c in cycles]),
    }


# ── Internal helpers ─────────────────────────────────────────────────


def _subsample_zc(signal: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Sub-sample zero-crossing positions via linear interpolation."""
    x0 = signal[indices]
    x1 = signal[indices + 1]
    denom = x1 - x0
    safe = np.abs(denom) > np.finfo(float).eps
    result = indices.astype(float)
    result[safe] = indices[safe] - x0[safe] / denom[safe]
    return result


def _find_descending_zc(
    signal: np.ndarray, peak_idx: int, end_idx: int
) -> int:
    """Find the first descending zero crossing between peak_idx and end_idx."""
    for i in range(peak_idx, min(end_idx, len(signal) - 1)):
        if signal[i] >= 0 and signal[i + 1] < 0:
            return i
    return -1
