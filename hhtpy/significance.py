"""
Wu-Huang (2004) statistical significance test for IMFs.

Tests whether each IMF contains statistically significant information
beyond what would be expected from white noise, using the theoretical
energy-spread function of white noise IMFs.

Reference:
    Wu, Z. & Huang, N.E. (2004). "A study of the characteristics of
    white noise using the empirical mode decomposition method."
    Proceedings of the Royal Society A, 460, 1597-1611.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.signal import find_peaks


@dataclass
class SignificanceResult:
    """Result of the Wu-Huang significance test for one IMF.

    Attributes:
        index: IMF index (0-based).
        is_significant: Whether the IMF is statistically significant.
        log_energy: Natural log of the IMF's energy density.
        log_period: Natural log of the IMF's mean period.
        upper_bound: Upper confidence bound in log-energy space.
        lower_bound: Lower confidence bound (None for a posteriori test).
    """

    index: int
    is_significant: bool
    log_energy: float
    log_period: float
    upper_bound: float
    lower_bound: float | None


def significance_test(
    imfs: np.ndarray,
    alpha: float = 0.95,
    method: str = "aposteriori",
    rescaling_imf: int = 0,
) -> list[SignificanceResult]:
    """
    Wu-Huang statistical significance test for EMD IMFs.

    Tests each IMF against the null hypothesis that the signal is white
    noise. EMD applied to white noise behaves as a dyadic filter bank
    where ``E_n * T_n = const`` (energy density times mean period is
    constant across IMF indices). This test checks whether each IMF's
    energy is consistent with this null model.

    Two test variants are available:

    - **a priori**: Two-sided test against the theoretical white noise
      energy-period line. Tests whether the IMF's energy falls within
      the confidence interval ``[-ln(T) ± z * spread]``.

    - **a posteriori**: Uses the first IMF (or a specified IMF) as a
      noise reference to estimate the noise level, then applies a
      one-sided test (upper bound only) to detect IMFs with more energy
      than expected from noise.

    Args:
        imfs: Array of IMFs, shape ``(n_imfs, n_samples)``. These should
            be the raw signal arrays (not IntrinsicModeFunction objects).
        alpha: Confidence level, between 0 and 1. Default is 0.95 (95%).
        method: ``"apriori"`` or ``"aposteriori"``. Default is
            ``"aposteriori"``.
        rescaling_imf: Index of the IMF to use as noise reference in the
            a posteriori test (0-based). Default is 0 (first IMF).

    Returns:
        List of :class:`SignificanceResult`, one per IMF.

    Raises:
        ValueError: If inputs are invalid.

    Example:
        >>> from hhtpy import decompose, significance_test
        >>> imfs, residue = decompose(signal)
        >>> results = significance_test(imfs)
        >>> for r in results:
        ...     status = "significant" if r.is_significant else "noise"
        ...     print(f"IMF {r.index}: {status}")
    """
    if not isinstance(imfs, np.ndarray) or imfs.ndim != 2:
        raise ValueError("imfs must be a 2D numpy array of shape (n_imfs, n_samples).")
    if imfs.shape[0] == 0:
        return []
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if method not in ("apriori", "aposteriori"):
        raise ValueError('method must be "apriori" or "aposteriori".')

    n_imfs, n_samples = imfs.shape
    z_alpha = abs(stats.norm.ppf((1 - alpha) / 2))

    # Compute log-energy and log-period for each IMF
    log_energies = []
    log_periods = []
    for imf in imfs:
        e = np.sum(imf ** 2) / n_samples
        t = _mean_period(imf)
        log_energies.append(math.log(max(e, np.finfo(float).tiny)))
        log_periods.append(math.log(max(t, 1.0)))

    if method == "apriori":
        return _test_apriori(log_energies, log_periods, n_samples, z_alpha)
    else:
        return _test_aposteriori(
            log_energies, log_periods, n_samples, z_alpha, rescaling_imf
        )


def _test_apriori(log_energies, log_periods, n_samples, z_alpha):
    """
    Two-sided a priori significance test.

    The null model states that the energy-period relationship of white
    noise IMFs follows:

    .. math::
        \\ln(\\bar{E}_k) = -\\ln(\\bar{T}_k) + \\text{const}

    The two-sided confidence band at significance level :math:`\\alpha` is:

    .. math::
        -\\ln(\\bar{T}_k) \\pm z_{\\alpha/2} \\,
        \\sqrt{\\frac{2}{N}} \\, e^{\\ln(\\bar{T}_k)/2}

    An IMF is declared significant if its log-energy falls outside this
    band.
    """
    results = []
    for i, (ln_e, ln_t) in enumerate(zip(log_energies, log_periods)):
        spread = z_alpha * math.sqrt(2 / n_samples) * math.exp(ln_t / 2)
        upper = -ln_t + spread
        lower = -ln_t - spread
        is_sig = not (lower <= ln_e <= upper)
        results.append(
            SignificanceResult(
                index=i,
                is_significant=is_sig,
                log_energy=ln_e,
                log_period=ln_t,
                upper_bound=upper,
                lower_bound=lower,
            )
        )
    return results


def _test_aposteriori(log_energies, log_periods, n_samples, z_alpha, ref_idx):
    """
    One-sided a posteriori significance test.

    Uses the reference IMF (typically the first) to estimate the noise
    energy level, then tests each IMF with a one-sided upper bound.
    The reference IMF's energy is rescaled to the upper confidence
    limit, and all other IMFs are compared against this calibrated
    noise floor. An IMF is significant if its rescaled energy exceeds
    the upper bound.
    """
    # Use reference IMF to estimate noise level
    ref_ln_t = log_periods[ref_idx]
    ref_ln_e = log_energies[ref_idx]

    # Upper limit of the reference IMF defines the scaling factor
    ref_spread = z_alpha * math.sqrt(2 / n_samples) * math.exp(ref_ln_t / 2)
    ref_upper = -ref_ln_t + ref_spread
    scaling_factor = math.exp(ref_upper)

    results = []
    for i, (ln_e, ln_t) in enumerate(zip(log_energies, log_periods)):
        spread = z_alpha * math.sqrt(2 / n_samples) * math.exp(ln_t / 2)
        upper = -ln_t + spread

        if i == ref_idx:
            scaled_ln_e = math.log(scaling_factor)
        else:
            e_density = math.exp(ln_e)
            scaled_ln_e = math.log(max(e_density / scaling_factor, np.finfo(float).tiny))

        is_sig = scaled_ln_e > upper

        results.append(
            SignificanceResult(
                index=i,
                is_significant=is_sig,
                log_energy=ln_e,
                log_period=ln_t,
                upper_bound=upper,
                lower_bound=None,
            )
        )
    return results


def _mean_period(imf: np.ndarray) -> float:
    """Compute mean period as ``N / n_peaks`` from positive peaks."""
    peaks, _ = find_peaks(imf, height=0)
    if len(peaks) == 0:
        return float(len(imf))
    return len(imf) / len(peaks)
