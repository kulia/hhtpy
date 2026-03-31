from typing import Callable
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline

# mode, total_sifts_performed -> True if stopping criterion is met
SiftStoppingCriterion = Callable[[np.ndarray, int], bool]


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

    def _fixed_number_of_sifts(_, total_sifts_performed: int):
        return total_sifts_performed >= fixed_number_of_sifts

    return _fixed_number_of_sifts


def get_stopping_criterion_s_number(s_number: int = 5) -> SiftStoppingCriterion:
    """
    S-number stopping criterion (Huang et al., 2003).

    Counts consecutive sifts where the number of extrema and zero-crossings
    remains unchanged. Stops when this count reaches ``s_number``.

    Args:
        s_number: Number of consecutive unchanged sifts before stopping. Default is 5.
    """
    state = {"prev_num_extrema": None, "prev_num_zero_crossings": None, "consecutive": 0}

    def _s_number_criterion(mode: np.ndarray, total_sifts_performed: int) -> bool:
        if total_sifts_performed == 0:
            state["prev_num_extrema"] = None
            state["prev_num_zero_crossings"] = None
            state["consecutive"] = 0
            return False

        num_maxima = len(find_peaks(mode)[0])
        num_minima = len(find_peaks(-mode)[0])
        num_extrema = num_maxima + num_minima

        zero_crossings = np.sum(np.diff(np.sign(mode)) != 0)

        if (
            num_extrema == state["prev_num_extrema"]
            and zero_crossings == state["prev_num_zero_crossings"]
        ):
            state["consecutive"] += 1
        else:
            state["consecutive"] = 0

        state["prev_num_extrema"] = num_extrema
        state["prev_num_zero_crossings"] = zero_crossings

        return state["consecutive"] >= s_number

    return _s_number_criterion


def get_stopping_criterion_cauchy(threshold: float = 0.3) -> SiftStoppingCriterion:
    """
    Cauchy convergence stopping criterion.

    Stops sifting when the relative energy change between consecutive sifts
    falls below a threshold:

    .. math::
        SD = \\frac{\\sum (h_{k-1}(t) - h_k(t))^2}{\\sum h_{k-1}(t)^2} < \\text{threshold}

    Args:
        threshold: Convergence threshold. Default is 0.3.
    """
    state = {"prev_mode": None}

    def _cauchy_criterion(mode: np.ndarray, total_sifts_performed: int) -> bool:
        if total_sifts_performed == 0:
            state["prev_mode"] = mode.copy()
            return False

        prev_energy = np.sum(state["prev_mode"] ** 2)
        if prev_energy == 0:
            state["prev_mode"] = mode.copy()
            return True

        sd = np.sum((state["prev_mode"] - mode) ** 2) / prev_energy
        state["prev_mode"] = mode.copy()

        return sd < threshold

    return _cauchy_criterion


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

    def _rilling_criterion(mode: np.ndarray, total_sifts_performed: int) -> bool:
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

        condition_1 = np.mean(sigma > threshold_1) < alpha
        condition_2 = np.all(sigma < threshold_2)

        return condition_1 and condition_2

    return _rilling_criterion
