from typing import Callable
import numpy as np
from scipy.signal import find_peaks

# mode, total_sifts_performed -> True if stopping criterion is met
SiftStoppingCriterion = Callable[[np.ndarray, int], bool]


def get_stopping_criterion_fixed_number_of_sifts(
    fixed_number_of_sifts: int,
) -> SiftStoppingCriterion:
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
