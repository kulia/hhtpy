from .emd import decompose
from ._emd_utils import EnvelopeOptions
from hhtpy import _emd_utils
from .hht import (
    IntrinsicModeFunction,
    hilbert_huang_transform,
    marginal_hilbert_spectrum,
    index_of_orthogonality,
    calculate_instantaneous_frequency_hilbert,
    calculate_instantaneous_frequency_quadrature,
)
from .sift_stopping_criteria import (
    SiftStoppingCriterion,
    get_stopping_criterion_fixed_number_of_sifts,
    get_stopping_criterion_s_number,
    get_stopping_criterion_cauchy,
    get_stopping_criterion_rilling,
)
from .ensemble_emd import eemd, ceemdan
from .multivariate_emd import memd
from .masked_emd import (
    masked_decompose,
    adaptive_masked_decompose,
    mask_init_huang,
    mask_init_deering_kaiser,
    mask_init_spectral,
)
from .significance import significance_test, SignificanceResult
from .frequency_methods import (
    calculate_instantaneous_frequency_zero_crossing,
    calculate_instantaneous_frequency_generalized_zero_crossing,
    calculate_instantaneous_frequency_teo,
    calculate_instantaneous_frequency_hou,
    calculate_instantaneous_frequency_wu,
    despike_frequency,
)
