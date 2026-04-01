import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection
from dataclasses import dataclass
from . import IntrinsicModeFunction, marginal_hilbert_spectrum
from ._emd_utils import get_freq_lim

from typing import Union, Optional, Any


@dataclass
class HilbertSpectrumConfig:
    time_variable: Optional[np.ndarray] = None
    log_color: bool = True
    log_freq: bool = False
    min_amplitude_lim: float = 1e-2
    amplitude_unit: Optional[str] = None
    max_number_of_imfs: Optional[int] = None
    fig: Optional[Any] = None
    ax: Optional[Any] = None


def plot_hilbert_spectrum(
    imfs: list[IntrinsicModeFunction],
    config: HilbertSpectrumConfig = HilbertSpectrumConfig(),
):
    """
    Plot the Hilbert spectrum: time-frequency colored by amplitude.

    Each IMF's instantaneous frequency trace is drawn as a colored line
    where the color represents instantaneous amplitude. This is the
    standard Hilbert spectrum visualization from Huang et al. (1998).

    Args:
        imfs: List of :class:`IntrinsicModeFunction` objects from
            :func:`hilbert_huang_transform`.
        config: Plot configuration (color scale, axis limits, etc.).

    Returns:
        Tuple of ``(fig, ax, colorbar)``.

    Reference:
        Huang, N.E. et al. (1998). "The empirical mode decomposition
        and the Hilbert spectrum for nonlinear and non-stationary time
        series analysis." *Proc. R. Soc. A*, 454, 903-995.
    """
    if len(imfs) == 0:
        raise ValueError("No IMFs to plot.")

    if config.max_number_of_imfs:
        num_imfs = np.min([len(imfs), config.max_number_of_imfs])
    else:
        num_imfs = len(imfs)

    fig = plt.figure() if config.fig is None else config.fig
    ax = plt.subplot2grid((1, 1), (0, 0)) if config.ax is None else config.ax

    time_variable = (
        np.arange(len(imfs[0].signal)) / imfs[0].sampling_frequency
        if config.time_variable is None
        else config.time_variable
    )

    x_lim = [0, time_variable[-1]]
    y_lim = get_freq_lim(imfs[:num_imfs], percentile=1)
    c_lim = get_clim(imfs[:num_imfs], config.min_amplitude_lim)

    for imf in imfs[:num_imfs]:
        frequency = imf.instantaneous_frequency.copy()

        if frequency is None:
            continue

        if config.log_freq:
            frequency[frequency <= 0] = np.nan
            frequency = np.log10(frequency)

        points = np.array([time_variable, frequency]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if config.log_color:
            norm = colors.LogNorm(c_lim[0], c_lim[1], clip=True)
        else:
            norm = plt.Normalize(c_lim[0], c_lim[1], clip=True)

        lc = LineCollection(
            segments=segments, array=imf.instantaneous_amplitude, norm=norm
        )
        ax.add_collection(lc)

    clb = fig.colorbar(ax.collections[-1], aspect=20, fraction=0.1, pad=0.02)

    clb.set_label(
        f"Amplitude {f'[{config.amplitude_unit}]' if config.amplitude_unit else ''}"
    )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if config.log_freq:
        ax.set_yscale("log")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    return fig, ax, clb


def get_clim(imfs, min_amplitude):
    amps = []
    for imf in imfs:
        amp = imf.instantaneous_amplitude[~np.isnan(imf.instantaneous_amplitude)]
        amps.extend([np.min(amp), np.max(amp)])

    amps = np.array(amps)
    return np.max((np.min(amps), min_amplitude)), np.max(amps)


def plot_imfs(
    imfs: Union[np.ndarray, list[IntrinsicModeFunction]],
    signal: np.ndarray = None,
    residue: np.ndarray = None,
    x_axis: np.ndarray = None,
    max_number_of_imfs: int = None,
):
    """
    Plot the IMFs (and optionally the signal and residue) as subplots.

    Args:
        imfs: IMFs as a 2D array or list of :class:`IntrinsicModeFunction`.
        signal: Original signal to plot above the IMFs.
        residue: Decomposition residue to plot below the IMFs.
        x_axis: Custom x-axis values (e.g., time vector).
        max_number_of_imfs: Limit the number of IMFs shown.

    Returns:
        Tuple of ``(fig, axs)``.
    """
    imfs = (
        [imf.signal for imf in imfs]
        if isinstance(imfs[0], IntrinsicModeFunction)
        else imfs
    )

    num_imfs = (
        np.min([len(imfs), max_number_of_imfs]) if max_number_of_imfs else len(imfs)
    )

    number_of_subplots = num_imfs
    number_of_subplots += 1 if signal is not None else 0
    number_of_subplots += 1 if residue is not None else 0

    fig, axs = plt.subplots(number_of_subplots, 1)
    if number_of_subplots == 1:
        axs = np.array([axs])

    idx = 0
    if signal is not None:
        if x_axis is None:
            axs[idx].plot(signal)
        else:
            axs[idx].plot(x_axis, signal)
        axs[idx].set_ylabel("Signal")
        idx += 1
    for i in range(num_imfs):
        if x_axis is None:
            axs[idx].plot(imfs[i])
        else:
            axs[idx].plot(x_axis, imfs[i])
        axs[idx].set_ylabel(f"IMF {i + 1}")
        idx += 1
    if residue is not None:
        if x_axis is None:
            axs[idx].plot(residue)
        else:
            axs[idx].plot(x_axis, residue)
        axs[idx].set_ylabel("Residue")

    return fig, axs


def plot_hilbert_spectrum_contour(
    imfs: list[IntrinsicModeFunction],
    num_frequency_bins: int = 200,
    num_time_bins: int = 200,
    max_number_of_imfs: Optional[int] = None,
    log_color: bool = True,
    fig=None,
    ax=None,
):
    """
    Contour-style Hilbert spectrum.

    Accumulates instantaneous amplitude into a 2D time-frequency grid
    and renders as a filled contour plot. This gives a smoother, more
    readable visualization than the line-collection Hilbert spectrum,
    especially for dense or noisy signals.

    Args:
        imfs: List of IntrinsicModeFunction objects.
        num_frequency_bins: Number of frequency bins. Default 200.
        num_time_bins: Number of time bins. Default 200.
        max_number_of_imfs: Limit the number of IMFs to plot.
        log_color: Use logarithmic color scale. Default True.
        fig: Existing figure (optional).
        ax: Existing axes (optional).

    Returns:
        Tuple of (fig, ax, contour_set).
    """
    if len(imfs) == 0:
        raise ValueError("No IMFs to plot.")

    num_imfs = min(len(imfs), max_number_of_imfs) if max_number_of_imfs else len(imfs)

    fig = plt.figure() if fig is None else fig
    ax = fig.add_subplot(111) if ax is None else ax

    f_s = imfs[0].sampling_frequency
    n_samples = len(imfs[0].signal)
    t_max = n_samples / f_s

    f_min, f_max = get_freq_lim(imfs[:num_imfs], percentile=1)
    f_min = max(f_min, 0)

    t_edges = np.linspace(0, t_max, num_time_bins + 1)
    f_edges = np.linspace(f_min, f_max, num_frequency_bins + 1)
    spectrum = np.zeros((num_frequency_bins, num_time_bins))

    time_axis = np.arange(n_samples) / f_s

    for imf in imfs[:num_imfs]:
        freq = imf.instantaneous_frequency
        amp = imf.instantaneous_amplitude

        valid = ~np.isnan(freq) & ~np.isnan(amp) & (freq >= f_min) & (freq <= f_max)
        if not np.any(valid):
            continue

        t_idx = np.digitize(time_axis[valid], t_edges) - 1
        f_idx = np.digitize(freq[valid], f_edges) - 1

        t_idx = np.clip(t_idx, 0, num_time_bins - 1)
        f_idx = np.clip(f_idx, 0, num_frequency_bins - 1)

        np.add.at(spectrum, (f_idx, t_idx), amp[valid])

    # Replace zeros for log scale
    spectrum_plot = spectrum.copy()
    if log_color:
        spectrum_plot[spectrum_plot <= 0] = np.nan

    t_centers = (t_edges[:-1] + t_edges[1:]) / 2
    f_centers = (f_edges[:-1] + f_edges[1:]) / 2

    if log_color:
        valid_vals = spectrum_plot[~np.isnan(spectrum_plot)]
        if len(valid_vals) > 0:
            vmin = np.percentile(valid_vals[valid_vals > 0], 5) if np.any(valid_vals > 0) else 1e-10
            vmax = np.max(valid_vals)
            cs = ax.contourf(
                t_centers, f_centers, spectrum_plot,
                levels=np.logspace(np.log10(vmin), np.log10(vmax), 30),
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            )
        else:
            cs = ax.contourf(t_centers, f_centers, spectrum_plot, levels=30)
    else:
        cs = ax.contourf(t_centers, f_centers, spectrum_plot, levels=30)

    clb = fig.colorbar(cs, ax=ax, aspect=20, fraction=0.1, pad=0.02)
    clb.set_label("Amplitude")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    return fig, ax, cs


def plot_marginal_hilbert_spectrum(
    imfs: list[IntrinsicModeFunction], fig=None, ax=None
):
    """
    Plot the marginal Hilbert spectrum (time-integrated amplitude vs frequency).

    Args:
        imfs: List of :class:`IntrinsicModeFunction` objects.
        fig: Existing figure (optional).
        ax: Existing axes (optional).

    Returns:
        Tuple of ``(fig, ax)``.
    """
    fig = plt.figure() if fig is None else fig
    ax = fig.add_subplot(111) if ax is None else ax

    frequencies, amplitudes = marginal_hilbert_spectrum(imfs)
    ax.plot(frequencies, amplitudes)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    return fig, ax
