from typing import Callable
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline

# mode, total_sifts_performed -> True if stopping criterion is met
SiftStoppingCriterion = Callable[[np.ndarray, int], bool]


class _FixedNumberOfSifts:
    """Picklable callable for fixed-count stopping criterion."""

    def __init__(self, fixed_number_of_sifts: int):
        self.fixed_number_of_sifts = fixed_number_of_sifts

    def __call__(self, _: np.ndarray, total_sifts_performed: int) -> bool:
        return total_sifts_performed >= self.fixed_number_of_sifts


class _SNumberCriterion:
    """Picklable callable for S-number stopping criterion."""

    def __init__(self, s_number: int):
        self.s_number = s_number
        self._prev_num_extrema = None
        self._prev_num_zero_crossings = None
        self._consecutive = 0

    def __call__(self, mode: np.ndarray, total_sifts_performed: int) -> bool:
        if total_sifts_performed == 0:
            self._prev_num_extrema = None
            self._prev_num_zero_crossings = None
            self._consecutive = 0
            return False

        num_maxima = len(find_peaks(mode)[0])
        num_minima = len(find_peaks(-mode)[0])
        num_extrema = num_maxima + num_minima

        zero_crossings = np.sum(np.diff(np.sign(mode)) != 0)

        if (
            num_extrema == self._prev_num_extrema
            and zero_crossings == self._prev_num_zero_crossings
        ):
            self._consecutive += 1
        else:
            self._consecutive = 0

        self._prev_num_extrema = num_extrema
        self._prev_num_zero_crossings = zero_crossings

        return self._consecutive >= self.s_number


class _CauchyCriterion:
    """Picklable callable for Cauchy convergence stopping criterion."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._prev_mode = None

    def __call__(self, mode: np.ndarray, total_sifts_performed: int) -> bool:
        if total_sifts_performed == 0:
            self._prev_mode = mode.copy()
            return False

        prev_energy = np.sum(self._prev_mode ** 2)
        if prev_energy == 0:
            self._prev_mode = mode.copy()
            return True

        sd = np.sum((self._prev_mode - mode) ** 2) / prev_energy
        self._prev_mode = mode.copy()

        return sd < self.threshold


class _RillingCriterion:
    """Picklable callable for Rilling–Flandrin–Gonçalves stopping criterion."""

    def __init__(self, threshold_1: float, threshold_2: float, alpha: float):
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.alpha = alpha

    def __call__(self, mode: np.ndarray, total_sifts_performed: int) -> bool:
        if total_sifts_performed == 0:
            return False

        maxima_idx, _ = find_peaks(mode)
        minima_idx, _ = find_peaks(-mode)

        if len(maxima_idx) < 2 or len(minima_idx) < 2:
            return True

        n = np.arange(len(mode))
        upper = CubicSpline(
            np.concatenate(([0], maxima_idx, [len(mode) - 1])),
            np.concatenate(([mode[0]], mode[maxima_idx], [mode[-1]])),
        )(n)
        lower = CubicSpline(
            np.concatenate(([0], minima_idx, [len(mode) - 1])),
            np.concatenate(([mode[0]], mode[minima_idx], [mode[-1]])),
        )(n)

        mean_env = (upper + lower) / 2
        amp = (upper - lower) / 2

        # Avoid division by zero where amplitude is negligible
        valid = amp > np.finfo(float).eps
        if not np.any(valid):
            return True

        sigma = np.zeros_like(mode)
        sigma[valid] = np.abs(mean_env[valid]) / amp[valid]

        condition_1 = np.mean(sigma > self.threshold_1) < self.alpha
        condition_2 = np.all(sigma < self.threshold_2)

        return condition_1 and condition_2


# ── Public factory functions (preserve existing API) ─────────────────


def get_stopping_criterion_fixed_number_of_sifts(
    fixed_number_of_sifts: int,
) -> SiftStoppingCriterion:
    """
    Fixed-count stopping criterion.

    Stops after exactly ``fixed_number_of_sifts`` iterations. This is the
    simplest criterion and was recommended by Huang (2015) with values of
    10–15 as a practical default.

    Args:
        fixed_number_of_sifts: Number of sifts to perform. Default in
            ``decompose()`` is 15.
    """
    return _FixedNumberOfSifts(fixed_number_of_sifts)


def get_stopping_criterion_s_number(s_number: int = 5) -> SiftStoppingCriterion:
    """
    S-number stopping criterion (Huang et al., 2003).

    Counts consecutive sifts where the number of extrema and
    zero-crossings remains unchanged. Stops when this count reaches
    ``s_number``. This ensures the IMF has stabilized in terms of its
    oscillatory structure without over-sifting.

    Args:
        s_number: Number of consecutive unchanged sifts before
            stopping. Default is 5.

    Reference:
        Huang, N.E., Wu, M.-L.C., Long, S.R. et al. (2003). "A
        confidence limit for the empirical mode decomposition and
        Hilbert spectral analysis." *Proceedings of the Royal Society
        A*, 459, 2317-2345.
    """
    return _SNumberCriterion(s_number)


def get_stopping_criterion_cauchy(threshold: float = 0.3) -> SiftStoppingCriterion:
    """
    Cauchy convergence stopping criterion.

    Stops sifting when the relative energy change between consecutive
    sifts falls below a threshold:

    .. math::
        SD = \\frac{\\sum_t (h_{k-1}(t) - h_k(t))^2}{\\sum_t h_{k-1}(t)^2}
        < \\text{threshold}

    This is the original criterion proposed in Huang et al. (1998).
    Typical threshold values range from 0.2 to 0.3.

    Args:
        threshold: Convergence threshold. Default is 0.3.

    Reference:
        Huang, N.E. et al. (1998). "The empirical mode decomposition
        and the Hilbert spectrum for nonlinear and non-stationary time
        series analysis." *Proceedings of the Royal Society A*, 454,
        903-995.
    """
    return _CauchyCriterion(threshold)


def get_stopping_criterion_rilling(
    threshold_1: float = 0.05,
    threshold_2: float = 0.5,
    alpha: float = 0.05,
) -> SiftStoppingCriterion:
    """
    Rilling–Flandrin–Gonçalves stopping criterion (2003).

    Evaluates IMF quality by comparing the mean envelope to the amplitude
    envelope at each sample. Sifting stops when both conditions are met:

    1. The fraction of samples where ``|mean| / amplitude > threshold_1``
       is less than ``alpha``.
    2. No sample has ``|mean| / amplitude > threshold_2``.

    This directly measures how symmetric the envelopes are (i.e., how
    close the mean envelope is to zero relative to the signal amplitude),
    rather than measuring convergence rate like Cauchy.

    Args:
        threshold_1: Tolerance for the mean/amplitude ratio at most samples.
            Default is 0.05 (5%).
        threshold_2: Hard ceiling — no sample may exceed this ratio.
            Default is 0.5 (50%).
        alpha: Maximum fraction of samples allowed to exceed threshold_1.
            Default is 0.05 (5%).

    Reference:
        Rilling, Flandrin & Gonçalves, "On Empirical Mode Decomposition
        and its algorithms", IEEE-EURASIP Workshop on Nonlinear Signal
        and Image Processing (NSIP), 2003.
    """
    return _RillingCriterion(threshold_1, threshold_2, alpha)
