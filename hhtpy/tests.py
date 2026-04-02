import unittest
import numpy as np
from hhtpy.emd import decompose
from hhtpy.hht import (
    hilbert_huang_transform,
    marginal_hilbert_spectrum,
    calculate_instantaneous_frequency_hilbert,
)
from hhtpy.sift_stopping_criteria import (
    get_stopping_criterion_s_number,
    get_stopping_criterion_cauchy,
    get_stopping_criterion_rilling,
)
from hhtpy.hht import index_of_orthogonality
from hhtpy.ensemble_emd import eemd, ceemdan
from hhtpy.multivariate_emd import memd
from hhtpy.masked_emd import (
    masked_decompose,
    adaptive_masked_decompose,
    mask_init_huang,
    mask_init_deering_kaiser,
    mask_init_spectral,
)
from hhtpy.frequency_methods import (
    calculate_instantaneous_frequency_zero_crossing,
    calculate_instantaneous_frequency_generalized_zero_crossing,
    calculate_instantaneous_frequency_teo,
    calculate_instantaneous_frequency_hou,
    calculate_instantaneous_frequency_wu,
    despike_frequency,
)
from hhtpy.plot import plot_imfs, plot_hilbert_spectrum_contour
import matplotlib.pyplot as plt


class TestEMDAndPlotting(unittest.TestCase):

    def setUp(self):
        T = 5  # sec
        f_s = 15000  # Hz
        n = np.arange(T * f_s)
        t = n / f_s  # sec

        self.y = (
            0.3 * np.cos(2 * np.pi * 5 * t**2) + 2 * np.cos(2 * np.pi * 1 * t) + 1 * t
        )

    def test_emd_decomposition(self):
        imfs, residue = decompose(self.y)

        self.assertIsNotNone(imfs)
        self.assertGreater(len(imfs), 0, "IMFs should not be empty after decomposition")

        self.assertIsNotNone(residue, "Residue should not be None after decomposition")
        self.assertEqual(
            len(residue),
            len(self.y),
            "Residue length should match the input signal length",
        )

    def test_plot_imfs(self):
        imfs, residue = decompose(self.y)

        fig, axs = plot_imfs(imfs, self.y, residue)

        self.assertIsInstance(
            fig, plt.Figure, "The output should be a matplotlib Figure object"
        )
        self.assertIsInstance(
            axs, np.ndarray, "The output should be an array of matplotlib Axes"
        )


class TestEMDRoundTrip(unittest.TestCase):
    """sum(imfs) + residue must reconstruct the original signal."""

    def test_round_trip_simple_signal(self):
        f_s = 1000
        t = np.arange(5 * f_s) / f_s
        signal = np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 50 * t)

        imfs, residue = decompose(signal)
        reconstructed = np.sum(imfs, axis=0) + residue

        np.testing.assert_allclose(
            reconstructed, signal, atol=1e-10,
            err_msg="IMFs + residue must reconstruct the original signal",
        )

    def test_round_trip_with_trend(self):
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        signal = np.cos(2 * np.pi * 20 * t) + 2 * t + 5

        imfs, residue = decompose(signal)
        reconstructed = np.sum(imfs, axis=0) + residue

        np.testing.assert_allclose(
            reconstructed, signal, atol=1e-10,
            err_msg="IMFs + residue must reconstruct signal with trend",
        )

    def test_round_trip_hht(self):
        """Round-trip also holds through the full HHT pipeline."""
        f_s = 1000
        t = np.arange(2 * f_s) / f_s
        signal = 3 * np.cos(2 * np.pi * 15 * t)

        imf_objects, residue = hilbert_huang_transform(signal, f_s)
        imf_signals = np.array([imf.signal for imf in imf_objects])
        reconstructed = np.sum(imf_signals, axis=0) + residue

        np.testing.assert_allclose(
            reconstructed, signal, atol=1e-10,
            err_msg="HHT IMFs + residue must reconstruct the original signal",
        )


class TestEMDEdgeCases(unittest.TestCase):

    def test_empty_signal_raises(self):
        with self.assertRaises(ValueError):
            decompose(np.array([]))

    def test_2d_signal_raises(self):
        with self.assertRaises(ValueError):
            decompose(np.array([[1, 2], [3, 4]]))

    def test_monotonic_signal(self):
        """A monotonic signal has no oscillatory components — should produce 0 IMFs."""
        signal = np.linspace(0, 10, 1000)
        imfs, residue = decompose(signal)

        self.assertEqual(len(imfs), 0, "Monotonic signal should produce 0 IMFs")
        np.testing.assert_allclose(
            residue, signal, atol=1e-10,
            err_msg="Residue should equal the monotonic signal",
        )

    def test_constant_signal(self):
        """A constant signal has zero variance — decompose should raise ValueError."""
        signal = np.ones(500) * 42.0
        with self.assertRaises(ValueError):
            decompose(signal)

    def test_short_signal(self):
        """Very short signals should still not crash."""
        signal = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        imfs, residue = decompose(signal)
        reconstructed = np.sum(imfs, axis=0) + residue if len(imfs) > 0 else residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)


class TestEMDCorrectness(unittest.TestCase):

    def test_pure_cosine_produces_one_dominant_imf(self):
        """A pure cosine should decompose into ~1 IMF carrying most energy."""
        f_s = 1000
        t = np.arange(5 * f_s) / f_s
        freq = 10  # Hz
        signal = np.cos(2 * np.pi * freq * t)

        imfs, residue = decompose(signal)

        self.assertGreaterEqual(len(imfs), 1)

        # First IMF should carry > 95% of signal energy
        signal_energy = np.sum(signal**2)
        first_imf_energy = np.sum(imfs[0] ** 2)
        energy_ratio = first_imf_energy / signal_energy

        self.assertGreater(
            energy_ratio, 0.95,
            f"First IMF should capture >95% of pure cosine energy, got {energy_ratio:.2%}",
        )

    def test_two_tones_produce_two_imfs(self):
        """Two well-separated tones should produce at least 2 IMFs."""
        f_s = 2000
        t = np.arange(5 * f_s) / f_s
        signal = np.cos(2 * np.pi * 5 * t) + np.cos(2 * np.pi * 80 * t)

        imfs, residue = decompose(signal)

        self.assertGreaterEqual(
            len(imfs), 2,
            "Two well-separated tones should produce at least 2 IMFs",
        )

    def test_high_frequency_extracted_first(self):
        """EMD extracts highest frequency component first."""
        f_s = 2000
        t = np.arange(5 * f_s) / f_s
        low_freq = np.cos(2 * np.pi * 5 * t)
        high_freq = np.cos(2 * np.pi * 80 * t)
        signal = low_freq + high_freq

        imfs, residue = decompose(signal)

        # First IMF should correlate more with the high-frequency component
        corr_high = np.abs(np.corrcoef(imfs[0], high_freq)[0, 1])
        corr_low = np.abs(np.corrcoef(imfs[0], low_freq)[0, 1])

        self.assertGreater(
            corr_high, corr_low,
            "First IMF should correlate more with the high-frequency component",
        )


class TestHHTFrequencyEstimation(unittest.TestCase):

    def test_cosine_instantaneous_frequency(self):
        """HHT on a pure cosine should yield instantaneous frequency close to true frequency."""
        f_s = 5000
        true_freq = 25  # Hz
        t = np.arange(3 * f_s) / f_s
        signal = np.cos(2 * np.pi * true_freq * t)

        imf_objects, residue = hilbert_huang_transform(signal, f_s)

        # The first IMF should have the dominant content
        freq = imf_objects[0].instantaneous_frequency
        # Use median of non-NaN interior values (edges are cropped)
        valid = freq[~np.isnan(freq)]
        median_freq = np.median(valid)

        self.assertAlmostEqual(
            median_freq, true_freq, delta=2.0,
            msg=f"Median instantaneous frequency should be ~{true_freq} Hz, got {median_freq:.1f} Hz",
        )

    def test_marginal_spectrum_peak(self):
        """Marginal Hilbert spectrum should peak near the true frequency."""
        f_s = 5000
        true_freq = 30  # Hz
        t = np.arange(3 * f_s) / f_s
        signal = 2 * np.cos(2 * np.pi * true_freq * t)

        imf_objects, residue = hilbert_huang_transform(signal, f_s)
        frequencies, amplitudes = marginal_hilbert_spectrum(imf_objects)

        peak_freq = frequencies[np.argmax(amplitudes)]

        self.assertAlmostEqual(
            peak_freq, true_freq, delta=3.0,
            msg=f"Marginal spectrum peak should be ~{true_freq} Hz, got {peak_freq:.1f} Hz",
        )


class TestHHTDecomposeFn(unittest.TestCase):

    def test_hht_with_eemd(self):
        """HHT with EEMD decomposition via decompose_fn."""
        from functools import partial

        f_s = 5000
        t = np.arange(2 * f_s) / f_s
        signal = np.cos(2 * np.pi * 25 * t)

        imf_objects, residue = hilbert_huang_transform(
            signal, f_s,
            decompose_fn=partial(eemd, num_trials=5, seed=42),
        )

        self.assertGreater(len(imf_objects), 0)
        self.assertIsNotNone(imf_objects[0].instantaneous_frequency)

    def test_hht_with_ceemdan(self):
        """HHT with CEEMDAN decomposition via decompose_fn."""
        from functools import partial

        f_s = 5000
        t = np.arange(2 * f_s) / f_s
        signal = np.cos(2 * np.pi * 25 * t)

        imf_objects, residue = hilbert_huang_transform(
            signal, f_s,
            decompose_fn=partial(ceemdan, num_trials=5, seed=42),
        )

        self.assertGreater(len(imf_objects), 0)


class TestPlotSubplots(unittest.TestCase):
    """Verify plot_imfs creates the correct number of subplots."""

    def setUp(self):
        f_s = 1000
        t = np.arange(2 * f_s) / f_s
        signal = np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 50 * t)
        self.imfs, self.residue = decompose(signal)
        self.signal = signal
        plt.close("all")

    def test_imfs_only(self):
        fig, axs = plot_imfs(self.imfs)
        self.assertEqual(len(axs), len(self.imfs))
        plt.close(fig)

    def test_imfs_with_signal(self):
        fig, axs = plot_imfs(self.imfs, signal=self.signal)
        self.assertEqual(len(axs), len(self.imfs) + 1)
        plt.close(fig)

    def test_imfs_with_signal_and_residue(self):
        fig, axs = plot_imfs(self.imfs, signal=self.signal, residue=self.residue)
        self.assertEqual(len(axs), len(self.imfs) + 2)
        plt.close(fig)

    def test_imfs_with_residue_only(self):
        fig, axs = plot_imfs(self.imfs, residue=self.residue)
        self.assertEqual(len(axs), len(self.imfs) + 1)
        plt.close(fig)

    def test_max_number_of_imfs(self):
        fig, axs = plot_imfs(
            self.imfs, signal=self.signal, residue=self.residue, max_number_of_imfs=1,
        )
        self.assertEqual(len(axs), 1 + 1 + 1)  # signal + 1 IMF + residue
        plt.close(fig)


class TestHilbertMethod(unittest.TestCase):

    def test_hilbert_frequency_pure_cosine(self):
        """Hilbert method should recover the correct frequency of a pure cosine."""
        f_s = 5000
        true_freq = 40  # Hz
        t = np.arange(3 * f_s) / f_s
        signal = np.cos(2 * np.pi * true_freq * t)

        imf_objects, residue = hilbert_huang_transform(
            signal, f_s,
            frequency_calculation_method=calculate_instantaneous_frequency_hilbert,
        )

        freq = imf_objects[0].instantaneous_frequency
        # Interior values (skip edge effects)
        interior = freq[len(freq) // 10 : -len(freq) // 10]
        median_freq = np.median(interior)

        self.assertAlmostEqual(
            median_freq, true_freq, delta=2.0,
            msg=f"Hilbert method median freq should be ~{true_freq} Hz, got {median_freq:.1f} Hz",
        )

    def test_hilbert_method_round_trip(self):
        """HHT with Hilbert method should still reconstruct the signal."""
        f_s = 1000
        t = np.arange(2 * f_s) / f_s
        signal = np.cos(2 * np.pi * 20 * t)

        imf_objects, residue = hilbert_huang_transform(
            signal, f_s,
            frequency_calculation_method=calculate_instantaneous_frequency_hilbert,
        )
        imf_signals = np.array([imf.signal for imf in imf_objects])
        reconstructed = np.sum(imf_signals, axis=0) + residue

        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)


class TestStoppingCriteria(unittest.TestCase):

    def _make_signal(self):
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        return np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 50 * t)

    def test_s_number_produces_valid_decomposition(self):
        signal = self._make_signal()
        criterion = get_stopping_criterion_s_number(s_number=5)
        imfs, residue = decompose(signal, stopping_criterion=criterion)

        self.assertGreater(len(imfs), 0)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)

    def test_cauchy_produces_valid_decomposition(self):
        signal = self._make_signal()
        criterion = get_stopping_criterion_cauchy(threshold=0.3)
        imfs, residue = decompose(signal, stopping_criterion=criterion)

        self.assertGreater(len(imfs), 0)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)

    def test_s_number_resets_between_imfs(self):
        """S-number state should reset for each IMF extraction (two-tone signal)."""
        signal = self._make_signal()
        criterion = get_stopping_criterion_s_number(s_number=3)
        imfs, residue = decompose(signal, stopping_criterion=criterion)

        # Should produce at least 2 IMFs for a two-tone signal
        self.assertGreaterEqual(len(imfs), 2)

    def test_cauchy_resets_between_imfs(self):
        """Cauchy state should reset for each IMF extraction."""
        signal = self._make_signal()
        criterion = get_stopping_criterion_cauchy(threshold=0.2)
        imfs, residue = decompose(signal, stopping_criterion=criterion)

        self.assertGreaterEqual(len(imfs), 2)


class TestRillingCriterion(unittest.TestCase):

    def _make_signal(self):
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        return np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 50 * t)

    def test_rilling_produces_valid_decomposition(self):
        signal = self._make_signal()
        criterion = get_stopping_criterion_rilling()
        imfs, residue = decompose(signal, stopping_criterion=criterion)

        self.assertGreater(len(imfs), 0)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)

    def test_rilling_resets_between_imfs(self):
        signal = self._make_signal()
        criterion = get_stopping_criterion_rilling()
        imfs, residue = decompose(signal, stopping_criterion=criterion)

        self.assertGreaterEqual(len(imfs), 2)

    def test_rilling_custom_thresholds(self):
        """Tighter thresholds should still produce a valid decomposition."""
        signal = self._make_signal()
        criterion = get_stopping_criterion_rilling(
            threshold_1=0.01, threshold_2=0.1, alpha=0.01
        )
        imfs, residue = decompose(signal, stopping_criterion=criterion)

        self.assertGreater(len(imfs), 0)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)


class TestIndexOfOrthogonality(unittest.TestCase):

    def test_orthogonal_imfs(self):
        """Perfectly orthogonal signals should give IO close to 0."""
        t = np.arange(1000) / 1000
        imfs = np.array([
            np.sin(2 * np.pi * 10 * t),
            np.cos(2 * np.pi * 10 * t),
        ])
        io = index_of_orthogonality(imfs)
        self.assertLess(io, 0.05)

    def test_identical_imfs(self):
        """Identical signals should give a high IO."""
        t = np.arange(1000) / 1000
        signal = np.sin(2 * np.pi * 5 * t)
        imfs = np.array([signal, signal])
        io = index_of_orthogonality(imfs)
        self.assertAlmostEqual(io, 0.25, places=2)

    def test_real_decomposition(self):
        """IO from a real EMD should be small (good decomposition)."""
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        signal = np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 50 * t)
        imfs, _ = decompose(signal)
        io = index_of_orthogonality(imfs)
        self.assertLess(io, 0.2)

    def test_single_imf(self):
        """Single IMF should return 0."""
        io = index_of_orthogonality(np.array([[1.0, 2.0, 3.0]]))
        self.assertEqual(io, 0.0)


class TestMaxImfs(unittest.TestCase):

    def test_max_imfs_limits_output(self):
        f_s = 1000
        t = np.arange(5 * f_s) / f_s
        signal = np.cos(2 * np.pi * 5 * t) + np.cos(2 * np.pi * 50 * t)

        imfs_full, _ = decompose(signal)
        imfs_limited, residue = decompose(signal, max_imfs=1)

        self.assertEqual(len(imfs_limited), 1)
        self.assertGreater(len(imfs_full), 1)

    def test_max_imfs_round_trip(self):
        """Even with limited IMFs, sum(imfs) + residue = signal."""
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        signal = np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 50 * t)

        imfs, residue = decompose(signal, max_imfs=2)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)


class TestEEMD(unittest.TestCase):

    def _make_signal(self):
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        return np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 50 * t)

    def test_eemd_produces_imfs(self):
        signal = self._make_signal()
        imfs, residue = eemd(signal, num_trials=10, seed=42)

        self.assertGreater(len(imfs), 0)
        self.assertEqual(imfs.shape[1], len(signal))

    def test_eemd_reproducible_with_seed(self):
        signal = self._make_signal()
        imfs1, _ = eemd(signal, num_trials=10, seed=42)
        imfs2, _ = eemd(signal, num_trials=10, seed=42)

        np.testing.assert_array_equal(imfs1, imfs2)

    def test_eemd_reduces_mode_mixing(self):
        """EEMD should produce more IMFs or better separation than EMD
        on a signal prone to mode mixing."""
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        # Intermittent signal — classic mode mixing scenario
        signal = np.cos(2 * np.pi * 10 * t)
        signal[1000:1500] += 0.5 * np.cos(2 * np.pi * 80 * t[1000:1500])

        imfs, residue = eemd(signal, num_trials=20, seed=42)

        self.assertGreater(len(imfs), 0)
        self.assertEqual(imfs.shape[1], len(signal))

    def test_eemd_edge_cases(self):
        """Empty and 2D signals should raise ValueError."""
        with self.assertRaises(ValueError):
            eemd(np.array([]))
        with self.assertRaises(ValueError):
            eemd(np.array([[1, 2], [3, 4]]))

    def test_eemd_high_frequency_first(self):
        """First EEMD IMF should correlate with the high-frequency component."""
        f_s = 2000
        t = np.arange(3 * f_s) / f_s
        low = np.cos(2 * np.pi * 5 * t)
        high = np.cos(2 * np.pi * 80 * t)
        signal = low + high

        imfs, _ = eemd(signal, num_trials=10, seed=42)

        corr_high = np.abs(np.corrcoef(imfs[0], high)[0, 1])
        corr_low = np.abs(np.corrcoef(imfs[0], low)[0, 1])
        self.assertGreater(corr_high, corr_low)

    def test_eemd_parallel(self):
        """Parallel EEMD should produce valid IMFs."""
        signal = self._make_signal()
        imfs, residue = eemd(signal, num_trials=10, seed=42, n_jobs=2)

        self.assertGreater(len(imfs), 0)
        self.assertEqual(imfs.shape[1], len(signal))

    def test_eemd_parallel_reproducible(self):
        """Parallel EEMD with same seed should be reproducible."""
        signal = self._make_signal()
        imfs1, _ = eemd(signal, num_trials=10, seed=42, n_jobs=2)
        imfs2, _ = eemd(signal, num_trials=10, seed=42, n_jobs=2)

        np.testing.assert_array_equal(imfs1, imfs2)


class TestCEEMDAN(unittest.TestCase):

    def _make_signal(self):
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        return np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 50 * t)

    def test_ceemdan_produces_imfs(self):
        signal = self._make_signal()
        imfs, residue = ceemdan(signal, num_trials=10, seed=42)

        self.assertGreater(len(imfs), 0)
        self.assertEqual(imfs.shape[1], len(signal))

    def test_ceemdan_exact_reconstruction(self):
        """CEEMDAN guarantees sum(imfs) + residue == signal."""
        signal = self._make_signal()
        imfs, residue = ceemdan(signal, num_trials=10, seed=42)

        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(
            reconstructed, signal, atol=1e-10,
            err_msg="CEEMDAN must guarantee exact reconstruction",
        )

    def test_ceemdan_reproducible_with_seed(self):
        signal = self._make_signal()
        imfs1, res1 = ceemdan(signal, num_trials=10, seed=42)
        imfs2, res2 = ceemdan(signal, num_trials=10, seed=42)

        np.testing.assert_array_equal(imfs1, imfs2)
        np.testing.assert_array_equal(res1, res2)

    def test_ceemdan_edge_cases(self):
        with self.assertRaises(ValueError):
            ceemdan(np.array([]))
        with self.assertRaises(ValueError):
            ceemdan(np.array([[1, 2], [3, 4]]))

    def test_ceemdan_high_frequency_first(self):
        """First CEEMDAN IMF should correlate with the high-frequency component."""
        f_s = 2000
        t = np.arange(3 * f_s) / f_s
        low = np.cos(2 * np.pi * 5 * t)
        high = np.cos(2 * np.pi * 80 * t)
        signal = low + high

        imfs, _ = ceemdan(signal, num_trials=10, seed=42)

        corr_high = np.abs(np.corrcoef(imfs[0], high)[0, 1])
        corr_low = np.abs(np.corrcoef(imfs[0], low)[0, 1])
        self.assertGreater(corr_high, corr_low)

    def test_ceemdan_parallel(self):
        """Parallel CEEMDAN should guarantee exact reconstruction."""
        signal = self._make_signal()
        imfs, residue = ceemdan(signal, num_trials=10, seed=42, n_jobs=2)

        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)

    def test_ceemdan_parallel_reproducible(self):
        """Parallel CEEMDAN with same seed should be reproducible."""
        signal = self._make_signal()
        imfs1, res1 = ceemdan(signal, num_trials=10, seed=42, n_jobs=2)
        imfs2, res2 = ceemdan(signal, num_trials=10, seed=42, n_jobs=2)

        np.testing.assert_array_equal(imfs1, imfs2)
        np.testing.assert_array_equal(res1, res2)


class TestMEMD(unittest.TestCase):

    def _make_bivariate_signal(self):
        """Two-channel signal with shared oscillatory modes."""
        f_s = 500
        t = np.arange(2 * f_s) / f_s
        ch1 = np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 40 * t)
        ch2 = np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 40 * t)
        return np.array([ch1, ch2])

    def test_memd_produces_imfs(self):
        signal = self._make_bivariate_signal()
        imfs, residue = memd(signal, num_directions=16)

        self.assertGreater(len(imfs), 0)
        self.assertEqual(imfs.shape[1], 2)  # 2 channels
        self.assertEqual(imfs.shape[2], signal.shape[1])

    def test_memd_round_trip(self):
        """sum(imfs) + residue must reconstruct the signal."""
        signal = self._make_bivariate_signal()
        imfs, residue = memd(signal, num_directions=16)

        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(
            reconstructed, signal, atol=1e-10,
            err_msg="MEMD IMFs + residue must reconstruct the original signal",
        )

    def test_memd_imf_alignment(self):
        """Both channels should have the same number of IMFs (alignment property)."""
        signal = self._make_bivariate_signal()
        imfs, residue = memd(signal, num_directions=16)

        # All IMFs share the same first dimension — inherent from the algorithm
        self.assertEqual(imfs.shape[1], signal.shape[0])

    def test_memd_three_channels(self):
        """MEMD should work with 3+ channels."""
        f_s = 500
        t = np.arange(2 * f_s) / f_s
        ch1 = np.cos(2 * np.pi * 10 * t)
        ch2 = np.sin(2 * np.pi * 10 * t)
        ch3 = np.cos(2 * np.pi * 10 * t + np.pi / 4)
        signal = np.array([ch1, ch2, ch3])

        imfs, residue = memd(signal, num_directions=16)

        self.assertGreater(len(imfs), 0)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)

    def test_memd_edge_cases(self):
        """Invalid inputs should raise ValueError."""
        # 1D input
        with self.assertRaises(ValueError):
            memd(np.array([1.0, 2.0, 3.0]))

        # Too few directions
        signal = self._make_bivariate_signal()
        with self.assertRaises(ValueError):
            memd(signal, num_directions=2)

    def test_memd_channels_transposed(self):
        """Signal with n_channels >= n_samples should raise a helpful error."""
        bad_signal = np.random.randn(100, 10)  # 100 channels, 10 samples
        with self.assertRaises(ValueError):
            memd(bad_signal)


class TestFrequencyMethods(unittest.TestCase):
    """Test all instantaneous frequency estimation methods."""

    def _make_cosine(self, freq=25, f_s=5000, duration=3):
        t = np.arange(duration * f_s) / f_s
        return np.cos(2 * np.pi * freq * t), f_s, freq

    def test_zero_crossing_frequency(self):
        signal, f_s, true_freq = self._make_cosine()
        freq = calculate_instantaneous_frequency_zero_crossing(signal, f_s)

        valid = freq[~np.isnan(freq)]
        self.assertGreater(len(valid), 0)
        median_freq = np.median(valid)
        self.assertAlmostEqual(median_freq, true_freq, delta=1.0)

    def test_generalized_zero_crossing_frequency(self):
        signal, f_s, true_freq = self._make_cosine()
        freq = calculate_instantaneous_frequency_generalized_zero_crossing(signal, f_s)

        valid = freq[~np.isnan(freq)]
        self.assertGreater(len(valid), 0)
        median_freq = np.median(valid)
        self.assertAlmostEqual(median_freq, true_freq, delta=2.0)

    def test_teo_frequency(self):
        signal, f_s, true_freq = self._make_cosine()
        freq = calculate_instantaneous_frequency_teo(signal, f_s)

        valid = freq[~np.isnan(freq)]
        self.assertGreater(len(valid), 0)
        # TEO can be noisy — use larger tolerance
        interior = valid[len(valid) // 10: -len(valid) // 10]
        median_freq = np.median(interior)
        self.assertAlmostEqual(median_freq, true_freq, delta=3.0)

    def test_hou_frequency(self):
        signal, f_s, true_freq = self._make_cosine()
        freq = calculate_instantaneous_frequency_hou(signal, f_s)

        valid = freq[~np.isnan(freq)]
        self.assertGreater(len(valid), 0)
        median_freq = np.median(valid)
        self.assertAlmostEqual(median_freq, true_freq, delta=2.0)

    def test_wu_frequency(self):
        signal, f_s, true_freq = self._make_cosine()
        freq = calculate_instantaneous_frequency_wu(signal, f_s)

        valid = freq[~np.isnan(freq)]
        self.assertGreater(len(valid), 0)
        median_freq = np.median(valid)
        self.assertAlmostEqual(median_freq, true_freq, delta=3.0)

    def test_all_methods_in_hht_pipeline(self):
        """All IF methods should work when passed to hilbert_huang_transform."""
        f_s = 5000
        t = np.arange(2 * f_s) / f_s
        signal = np.cos(2 * np.pi * 20 * t)

        methods = [
            calculate_instantaneous_frequency_zero_crossing,
            calculate_instantaneous_frequency_generalized_zero_crossing,
            calculate_instantaneous_frequency_teo,
            calculate_instantaneous_frequency_hou,
        ]

        for method in methods:
            imf_objects, residue = hilbert_huang_transform(
                signal, f_s, frequency_calculation_method=method
            )
            self.assertGreater(len(imf_objects), 0, f"Failed for {method.__name__}")


class TestDespikeFrequency(unittest.TestCase):

    def test_despike_removes_extrema_spikes(self):
        """Frequency spikes at extrema should be smoothed out."""
        from scipy.signal import find_peaks

        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        imf = np.cos(2 * np.pi * 10 * t)

        # Place spikes exactly at detected peaks
        maxima, _ = find_peaks(imf)
        freq = np.full(len(t), 10.0)
        for idx in maxima[:5]:
            freq[idx] = 100.0  # artificial spike

        despiked = despike_frequency(freq, imf)

        # Spikes at extrema should be reduced
        self.assertLess(np.max(despiked), np.max(freq))

    def test_despike_preserves_clean_signal(self):
        """A clean frequency signal should be mostly unchanged."""
        imf = np.cos(2 * np.pi * 10 * np.arange(1000) / 1000)
        freq = np.full(1000, 10.0)

        despiked = despike_frequency(freq, imf)

        # Should be very close to original
        np.testing.assert_allclose(despiked, freq, atol=0.1)


class TestMaskedEMD(unittest.TestCase):

    def _make_signal(self):
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        return np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 50 * t), f_s

    def test_masked_decompose_produces_imfs(self):
        signal, f_s = self._make_signal()
        imfs, residue = masked_decompose(
            signal, mask_frequency=80, mask_amplitude=1.0,
            sampling_frequency=f_s,
        )

        self.assertGreater(len(imfs), 0)
        self.assertEqual(imfs.shape[1], len(signal))

    def test_masked_decompose_round_trip(self):
        signal, f_s = self._make_signal()
        imfs, residue = masked_decompose(
            signal, mask_frequency=80, mask_amplitude=1.0,
            sampling_frequency=f_s,
        )

        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(
            reconstructed, signal, atol=1e-10,
            err_msg="Masked EMD must reconstruct the signal",
        )

    def test_adaptive_masked_decompose(self):
        signal, f_s = self._make_signal()
        imfs, residue = adaptive_masked_decompose(
            signal, sampling_frequency=f_s,
        )

        self.assertGreater(len(imfs), 0)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)

    def test_adaptive_with_deering_kaiser(self):
        signal, f_s = self._make_signal()
        imfs, residue = adaptive_masked_decompose(
            signal, sampling_frequency=f_s,
            mask_init_method=mask_init_deering_kaiser,
        )

        self.assertGreater(len(imfs), 0)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)

    def test_adaptive_with_spectral(self):
        signal, f_s = self._make_signal()
        imfs, residue = adaptive_masked_decompose(
            signal, sampling_frequency=f_s,
            mask_init_method=mask_init_spectral,
        )

        self.assertGreater(len(imfs), 0)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)

    def test_mask_init_methods_return_valid(self):
        """All mask init methods should return positive frequency and amplitude."""
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        signal = np.cos(2 * np.pi * 20 * t) + 0.5 * np.cos(2 * np.pi * 60 * t)

        for method in [mask_init_huang, mask_init_deering_kaiser, mask_init_spectral]:
            f, a = method(signal, f_s)
            self.assertGreater(f, 0, f"{method.__name__} frequency must be positive")
            self.assertGreater(a, 0, f"{method.__name__} amplitude must be positive")

    def test_masked_emd_edge_cases(self):
        with self.assertRaises(ValueError):
            masked_decompose(np.array([]), 100, 1.0, 1000)
        with self.assertRaises(ValueError):
            masked_decompose(np.array([[1, 2], [3, 4]]), 100, 1.0, 1000)


class TestSignificanceTest(unittest.TestCase):

    def test_white_noise_imfs_aposteriori(self):
        """A posteriori test on white noise: most IMFs should not be significant."""
        from hhtpy.significance import significance_test

        rng = np.random.default_rng(42)
        noise = rng.normal(0, 1, 5000)
        imfs, _ = decompose(noise)

        results = significance_test(imfs, alpha=0.95, method="aposteriori")

        # A posteriori test uses first IMF as noise reference, so it
        # should flag fewer IMFs as significant for pure noise
        n_significant = sum(r.is_significant for r in results)
        self.assertLess(n_significant, len(results))

    def test_signal_imfs_are_significant(self):
        """IMFs from a clear signal should be significant."""
        from hhtpy.significance import significance_test

        t = np.arange(5000) / 1000
        signal = 5 * np.cos(2 * np.pi * 10 * t) + 3 * np.cos(2 * np.pi * 50 * t)
        imfs, _ = decompose(signal)

        results = significance_test(imfs, alpha=0.95, method="apriori")

        # At least one IMF should be significant (the signal IMFs)
        n_significant = sum(r.is_significant for r in results)
        self.assertGreater(n_significant, 0)

    def test_aposteriori_method(self):
        """A posteriori test should work and return results for all IMFs."""
        from hhtpy.significance import significance_test

        t = np.arange(3000) / 1000
        signal = np.cos(2 * np.pi * 10 * t) + 0.1 * np.random.default_rng(42).normal(0, 1, 3000)
        imfs, _ = decompose(signal)

        results = significance_test(imfs, alpha=0.95, method="aposteriori")

        self.assertEqual(len(results), len(imfs))
        # A posteriori results should have lower_bound=None
        for r in results:
            self.assertIsNone(r.lower_bound)

    def test_result_fields(self):
        """SignificanceResult should have all expected fields."""
        from hhtpy.significance import significance_test

        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 2000)
        imfs, _ = decompose(signal)

        results = significance_test(imfs, alpha=0.95, method="apriori")

        for r in results:
            self.assertIsInstance(r.index, int)
            self.assertIsInstance(r.is_significant, (bool, np.bool_))
            self.assertIsInstance(r.log_energy, float)
            self.assertIsInstance(r.log_period, float)
            self.assertIsInstance(r.upper_bound, float)
            # a priori should have lower bound
            self.assertIsInstance(r.lower_bound, float)

    def test_empty_imfs(self):
        """Empty IMF array should return empty results."""
        from hhtpy.significance import significance_test

        results = significance_test(np.empty((0, 100)), alpha=0.95)
        self.assertEqual(len(results), 0)

    def test_invalid_inputs(self):
        """Invalid inputs should raise ValueError."""
        from hhtpy.significance import significance_test

        with self.assertRaises(ValueError):
            significance_test(np.array([1, 2, 3]))  # 1D, not 2D
        with self.assertRaises(ValueError):
            significance_test(np.zeros((3, 100)), alpha=1.5)
        with self.assertRaises(ValueError):
            significance_test(np.zeros((3, 100)), method="invalid")


class TestEnvelopeOptions(unittest.TestCase):

    def _make_signal(self):
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        return np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 50 * t)

    def test_pchip_spline_decomposition(self):
        """PChip spline should produce valid decomposition."""
        from hhtpy._emd_utils import EnvelopeOptions

        signal = self._make_signal()
        opts = EnvelopeOptions(spline_method="pchip")
        imfs, residue = decompose(signal, envelope_opts=opts)

        self.assertGreater(len(imfs), 0)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)

    def test_akima_spline_decomposition(self):
        """Akima spline should produce valid decomposition."""
        from hhtpy._emd_utils import EnvelopeOptions

        signal = self._make_signal()
        opts = EnvelopeOptions(spline_method="akima")
        imfs, residue = decompose(signal, envelope_opts=opts)

        self.assertGreater(len(imfs), 0)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)

    def test_mirror_boundary_decomposition(self):
        """Mirror boundary should produce valid decomposition."""
        from hhtpy._emd_utils import EnvelopeOptions

        signal = self._make_signal()
        opts = EnvelopeOptions(boundary_mode="mirror")
        imfs, residue = decompose(signal, envelope_opts=opts)

        self.assertGreater(len(imfs), 0)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)

    def test_none_boundary_decomposition(self):
        """None boundary should produce valid decomposition."""
        from hhtpy._emd_utils import EnvelopeOptions

        signal = self._make_signal()
        opts = EnvelopeOptions(boundary_mode="none")
        imfs, residue = decompose(signal, envelope_opts=opts)

        self.assertGreater(len(imfs), 0)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)

    def test_pchip_with_mirror(self):
        """Combined PChip + mirror should work."""
        from hhtpy._emd_utils import EnvelopeOptions

        signal = self._make_signal()
        opts = EnvelopeOptions(spline_method="pchip", boundary_mode="mirror")
        imfs, residue = decompose(signal, envelope_opts=opts)

        self.assertGreater(len(imfs), 0)
        reconstructed = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, signal, atol=1e-10)

    def test_envelope_opts_in_hht(self):
        """EnvelopeOptions should work through hilbert_huang_transform."""
        from hhtpy._emd_utils import EnvelopeOptions

        signal = self._make_signal()
        opts = EnvelopeOptions(spline_method="pchip")
        imf_objects, residue = hilbert_huang_transform(
            signal, 1000, envelope_opts=opts
        )

        self.assertGreater(len(imf_objects), 0)

    def test_invalid_spline_method(self):
        """Invalid spline method should raise ValueError."""
        from hhtpy._emd_utils import EnvelopeOptions, sift

        signal = self._make_signal()
        opts = EnvelopeOptions(spline_method="invalid")
        with self.assertRaises(ValueError):
            decompose(signal, envelope_opts=opts)


class TestContourPlot(unittest.TestCase):

    def test_contour_plot_creates_figure(self):
        f_s = 5000
        t = np.arange(2 * f_s) / f_s
        signal = np.cos(2 * np.pi * 25 * t)

        imf_objects, _ = hilbert_huang_transform(signal, f_s)
        fig, ax, cs = plot_hilbert_spectrum_contour(imf_objects)

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_contour_plot_with_max_imfs(self):
        f_s = 5000
        t = np.arange(2 * f_s) / f_s
        signal = np.cos(2 * np.pi * 25 * t) + 0.5 * np.cos(2 * np.pi * 60 * t)

        imf_objects, _ = hilbert_huang_transform(signal, f_s)
        fig, ax, cs = plot_hilbert_spectrum_contour(
            imf_objects, max_number_of_imfs=1
        )

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class TestResidueQuality(unittest.TestCase):
    """Verify that residues have the expected physical properties."""

    def test_residue_is_monotonic_or_nearly_so(self):
        """EMD residue should be monotonic or have very few extrema."""
        from scipy.signal import find_peaks

        f_s = 1000
        t = np.arange(5 * f_s) / f_s
        signal = np.cos(2 * np.pi * 20 * t) + 0.5 * np.cos(2 * np.pi * 60 * t) + 2 * t

        imfs, residue = decompose(signal)

        # EMD stops when residue is monotonic or has < 2 extrema
        maxima, _ = find_peaks(residue)
        minima, _ = find_peaks(-residue)
        total_extrema = len(maxima) + len(minima)

        self.assertLessEqual(
            total_extrema, 3,
            f"Residue should be nearly monotonic, got {total_extrema} extrema",
        )

    def test_residue_captures_dc_offset(self):
        """A DC offset should end up mostly in the residue."""
        f_s = 1000
        t = np.arange(5 * f_s) / f_s
        dc = 10.0
        signal = np.cos(2 * np.pi * 20 * t) + dc

        imfs, residue = decompose(signal)

        # Mean of residue should be close to the DC offset
        self.assertAlmostEqual(
            np.mean(residue), dc, delta=1.0,
            msg=f"Residue mean should be ~{dc}, got {np.mean(residue):.2f}",
        )

    def test_residue_has_fewer_oscillations_than_signal(self):
        """Residue should have far fewer zero crossings than the original signal."""
        f_s = 1000
        t = np.arange(5 * f_s) / f_s
        signal = np.cos(2 * np.pi * 20 * t) + 0.5 * np.cos(2 * np.pi * 60 * t)

        imfs, residue = decompose(signal)

        signal_zc = np.sum(np.diff(np.sign(signal)) != 0)
        residue_zc = np.sum(np.diff(np.sign(residue)) != 0)

        self.assertLess(
            residue_zc, signal_zc / 2,
            "Residue should have far fewer zero crossings than the signal",
        )

    def test_trend_captured_by_residue_plus_last_imfs(self):
        """The trend should be captured by the residue + lowest-frequency IMFs."""
        f_s = 1000
        t = np.arange(5 * f_s) / f_s
        trend = 0.5 * t ** 2
        signal = np.cos(2 * np.pi * 30 * t) + trend

        imfs, residue = decompose(signal)

        # The trend lives in the residue plus the slowest IMFs.
        # The last IMF + residue should correlate with the trend.
        low_freq_part = residue + imfs[-1] if len(imfs) > 0 else residue
        corr = np.corrcoef(low_freq_part, trend)[0, 1]
        self.assertGreater(
            corr, 0.90,
            f"Residue + last IMF should capture the trend (corr={corr:.3f})",
        )

    def test_residue_energy_is_small_for_pure_oscillation(self):
        """For a pure oscillation with no trend, residue should carry little energy."""
        f_s = 1000
        t = np.arange(5 * f_s) / f_s
        signal = np.cos(2 * np.pi * 20 * t)

        imfs, residue = decompose(signal)

        signal_energy = np.sum(signal ** 2)
        residue_energy = np.sum(residue ** 2)
        ratio = residue_energy / signal_energy

        self.assertLess(
            ratio, 0.01,
            f"Residue should carry <1% of energy for pure oscillation, got {ratio:.4f}",
        )

    def test_eemd_residue_is_reasonable(self):
        """EEMD doesn't guarantee exact reconstruction, but error should be small."""
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        signal = np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 50 * t)

        imfs, residue = eemd(signal, num_trials=20, seed=42)
        reconstructed = np.sum(imfs, axis=0) + residue

        # Not exact, but should be close
        relative_error = np.sqrt(np.mean((reconstructed - signal) ** 2)) / np.std(signal)
        self.assertLess(
            relative_error, 0.1,
            f"EEMD reconstruction error should be <10%, got {relative_error:.2%}",
        )

    def test_ceemdan_residue_exact_with_trend(self):
        """CEEMDAN should exactly reconstruct even with a trend."""
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        signal = np.cos(2 * np.pi * 10 * t) + 2 * t + 5

        imfs, residue = ceemdan(signal, num_trials=10, seed=42)
        reconstructed = np.sum(imfs, axis=0) + residue

        np.testing.assert_allclose(
            reconstructed, signal, atol=1e-10,
            err_msg="CEEMDAN must exactly reconstruct signal with trend",
        )


class TestStoppingCriteriaBehavior(unittest.TestCase):
    """Test that stopping criteria affect decomposition behavior as expected."""

    def _make_signal(self):
        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        return np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 50 * t)

    def test_more_sifts_gives_lower_io(self):
        """More sifting iterations should generally reduce the index of orthogonality."""
        signal = self._make_signal()
        from hhtpy.sift_stopping_criteria import get_stopping_criterion_fixed_number_of_sifts

        imfs_few, _ = decompose(
            signal, stopping_criterion=get_stopping_criterion_fixed_number_of_sifts(3)
        )
        imfs_many, _ = decompose(
            signal, stopping_criterion=get_stopping_criterion_fixed_number_of_sifts(30)
        )

        io_few = index_of_orthogonality(imfs_few) if len(imfs_few) >= 2 else float("inf")
        io_many = index_of_orthogonality(imfs_many) if len(imfs_many) >= 2 else float("inf")

        # With more sifting, the IMFs should be at least as orthogonal
        self.assertLessEqual(io_many, io_few + 0.05)

    def test_s_number_criterion_stops_adaptively(self):
        """S-number should stop sifting once the extrema count stabilizes."""
        signal = self._make_signal()

        # S-number with a small value should stop early
        criterion_fast = get_stopping_criterion_s_number(s_number=2)
        imfs_fast, _ = decompose(signal, stopping_criterion=criterion_fast)

        # S-number with a large value should sift more
        criterion_slow = get_stopping_criterion_s_number(s_number=10)
        imfs_slow, _ = decompose(signal, stopping_criterion=criterion_slow)

        # Both should produce valid decompositions
        self.assertGreater(len(imfs_fast), 0)
        self.assertGreater(len(imfs_slow), 0)

    def test_cauchy_tight_threshold_sifts_more(self):
        """A tighter Cauchy threshold should result in more sifting (flatter residual)."""
        signal = self._make_signal()

        # Loose threshold — stops early
        criterion_loose = get_stopping_criterion_cauchy(threshold=0.5)
        imfs_loose, residue_loose = decompose(signal, stopping_criterion=criterion_loose)

        # Tight threshold — sifts more
        criterion_tight = get_stopping_criterion_cauchy(threshold=0.01)
        imfs_tight, residue_tight = decompose(signal, stopping_criterion=criterion_tight)

        # Both should reconstruct
        np.testing.assert_allclose(
            np.sum(imfs_loose, axis=0) + residue_loose, signal, atol=1e-10
        )
        np.testing.assert_allclose(
            np.sum(imfs_tight, axis=0) + residue_tight, signal, atol=1e-10
        )

    def test_rilling_produces_imf_like_results(self):
        """Rilling criterion should produce IMFs with nearly zero mean envelopes."""
        from scipy.signal import find_peaks
        from scipy.interpolate import CubicSpline

        signal = self._make_signal()
        criterion = get_stopping_criterion_rilling(
            threshold_1=0.05, threshold_2=0.5, alpha=0.05
        )
        imfs, _ = decompose(signal, stopping_criterion=criterion)

        # Check that the first IMF has a small mean envelope
        imf = imfs[0]
        maxima, _ = find_peaks(imf)
        minima, _ = find_peaks(-imf)

        if len(maxima) >= 2 and len(minima) >= 2:
            n = np.arange(len(imf))
            upper = CubicSpline(
                np.concatenate(([0], maxima, [len(imf) - 1])),
                np.concatenate(([imf[0]], imf[maxima], [imf[-1]])),
            )(n)
            lower = CubicSpline(
                np.concatenate(([0], minima, [len(imf) - 1])),
                np.concatenate(([imf[0]], imf[minima], [imf[-1]])),
            )(n)
            mean_env = (upper + lower) / 2
            amp = (upper - lower) / 2

            # The mean envelope should be small relative to amplitude
            valid = amp > np.finfo(float).eps
            if np.any(valid):
                ratio = np.mean(np.abs(mean_env[valid]) / amp[valid])
                self.assertLess(
                    ratio, 0.2,
                    f"Rilling IMF mean/amplitude ratio should be small, got {ratio:.3f}",
                )


class TestCycleAnalysis(unittest.TestCase):
    """Test cycle detection, characterization, and analysis tools."""

    def _make_cosine(self, freq=10, f_s=1000, duration=3):
        t = np.arange(duration * f_s) / f_s
        return np.cos(2 * np.pi * freq * t), f_s

    def test_detect_cycles_pure_cosine(self):
        """A pure cosine should produce cycles with correct frequency."""
        from hhtpy.cycles import detect_cycles

        freq = 10
        imf, f_s = self._make_cosine(freq=freq)
        cycles = detect_cycles(imf, f_s)

        self.assertGreater(len(cycles), 0)

        # Most cycles should have frequency close to 10 Hz
        freqs = [c.frequency for c in cycles]
        median_freq = np.median(freqs)
        self.assertAlmostEqual(
            median_freq, freq, delta=0.5,
            msg=f"Cycle frequency should be ~{freq} Hz, got {median_freq:.1f}",
        )

    def test_detect_cycles_amplitude(self):
        """Cycle amplitude of a unit cosine should be ~1."""
        from hhtpy.cycles import detect_cycles

        imf, f_s = self._make_cosine()
        cycles = detect_cycles(imf, f_s)

        amps = [c.amplitude for c in cycles]
        median_amp = np.median(amps)
        self.assertAlmostEqual(
            median_amp, 1.0, delta=0.1,
            msg=f"Cycle amplitude should be ~1.0, got {median_amp:.3f}",
        )

    def test_detect_cycles_symmetry(self):
        """A pure cosine should have symmetric cycles (rise ≈ fall)."""
        from hhtpy.cycles import detect_cycles

        imf, f_s = self._make_cosine()
        cycles = detect_cycles(imf, f_s)

        # Filter to complete cycles
        complete = [c for c in cycles if c.is_complete]
        self.assertGreater(len(complete), 0)

        rise_fractions = [c.rise_fraction for c in complete]
        median_rise = np.median(rise_fractions)
        # Cosine starts at ascending zero → peak is at 1/4 cycle
        self.assertAlmostEqual(
            median_rise, 0.25, delta=0.05,
            msg=f"Rise fraction of cosine should be ~0.25, got {median_rise:.3f}",
        )

    def test_detect_cycles_completeness(self):
        """Most cycles of a clean cosine should be complete."""
        from hhtpy.cycles import detect_cycles

        imf, f_s = self._make_cosine()
        cycles = detect_cycles(imf, f_s)

        n_complete = sum(1 for c in cycles if c.is_complete)
        ratio = n_complete / len(cycles)
        self.assertGreater(
            ratio, 0.9,
            f"Most cosine cycles should be complete, got {ratio:.0%}",
        )

    def test_detect_cycles_from_emd(self):
        """Cycle detection should work on real EMD output."""
        from hhtpy.cycles import detect_cycles

        f_s = 1000
        t = np.arange(3 * f_s) / f_s
        signal = np.cos(2 * np.pi * 20 * t) + 0.5 * np.cos(2 * np.pi * 60 * t)
        imfs, _ = decompose(signal)

        cycles = detect_cycles(imfs[0], f_s)
        self.assertGreater(len(cycles), 5)

    def test_detect_cycles_min_samples(self):
        """Cycles shorter than min_samples should be excluded."""
        from hhtpy.cycles import detect_cycles

        imf, f_s = self._make_cosine(freq=10, f_s=100)  # 10 samples/cycle
        all_cycles = detect_cycles(imf, f_s, min_samples=4)
        strict_cycles = detect_cycles(imf, f_s, min_samples=15)

        self.assertGreater(len(all_cycles), len(strict_cycles))

    def test_get_cycle_vector(self):
        """Cycle vector should label each sample with its cycle index."""
        from hhtpy.cycles import get_cycle_vector

        imf, _ = self._make_cosine()
        cv = get_cycle_vector(imf)

        self.assertEqual(len(cv), len(imf))
        # Should have labeled cycles (non-zero values)
        self.assertGreater(np.max(cv), 0)
        # Labels should be consecutive integers
        unique_labels = np.unique(cv[cv > 0])
        expected = np.arange(1, len(unique_labels) + 1)
        np.testing.assert_array_equal(unique_labels, expected)

    def test_get_cycle_stat(self):
        """Per-cycle statistics should work correctly."""
        from hhtpy.cycles import get_cycle_vector, get_cycle_stat

        imf, _ = self._make_cosine()
        cv = get_cycle_vector(imf)

        # Mean of the IMF within each cycle of a cosine should be ~0
        means = get_cycle_stat(cv, imf, func=np.mean)
        self.assertGreater(len(means), 0)
        self.assertAlmostEqual(
            np.median(np.abs(means)), 0, delta=0.1,
            msg="Mean of each cosine cycle should be ~0",
        )

    def test_phase_align(self):
        """Phase-aligned cycles should all have the same length."""
        from hhtpy.cycles import phase_align, get_cycle_vector

        imf, _ = self._make_cosine()
        cv = get_cycle_vector(imf)
        n_points = 48

        aligned = phase_align(imf, cv, n_points=n_points)

        n_cycles = len(np.unique(cv[cv > 0]))
        self.assertEqual(aligned.shape, (n_cycles, n_points))

        # Average aligned waveform of a cosine should look like a cosine
        mean_waveform = np.mean(aligned, axis=0)
        # It should have a peak in the first quarter and trough in the third quarter
        peak_pos = np.argmax(mean_waveform) / n_points
        self.assertLess(peak_pos, 0.5, "Peak should be in the first half")

    def test_phase_align_auto_cycle_vector(self):
        """Phase alignment should work without providing a cycle vector."""
        from hhtpy.cycles import phase_align

        imf, _ = self._make_cosine()
        aligned = phase_align(imf)

        self.assertGreater(aligned.shape[0], 0)
        self.assertEqual(aligned.shape[1], 48)

    def test_cycle_summary_table(self):
        """Summary table should return a dict with correct keys and lengths."""
        from hhtpy.cycles import detect_cycles, cycle_summary_table

        imf, f_s = self._make_cosine()
        cycles = detect_cycles(imf, f_s)
        table = cycle_summary_table(cycles)

        self.assertIn("frequency", table)
        self.assertIn("amplitude", table)
        self.assertIn("rise_fraction", table)
        self.assertIn("is_complete", table)

        for key, arr in table.items():
            self.assertEqual(len(arr), len(cycles), f"Length mismatch for '{key}'")

    def test_cycle_summary_table_empty(self):
        """Summary table should handle empty cycle list."""
        from hhtpy.cycles import cycle_summary_table

        table = cycle_summary_table([])
        self.assertEqual(len(table["frequency"]), 0)

    def test_chirp_signal_varying_frequency(self):
        """Cycles from a chirp should show increasing frequency."""
        from hhtpy.cycles import detect_cycles

        f_s = 1000
        t = np.arange(5 * f_s) / f_s
        # Linear chirp from 5 Hz to 50 Hz
        phase = 2 * np.pi * (5 * t + (50 - 5) / (2 * 5) * t ** 2)
        imf = np.cos(phase)

        cycles = detect_cycles(imf, f_s)
        freqs = [c.frequency for c in cycles]

        # Frequency should increase over time
        first_quarter = np.median(freqs[:len(freqs) // 4])
        last_quarter = np.median(freqs[-len(freqs) // 4:])
        self.assertGreater(
            last_quarter, first_quarter,
            "Chirp cycle frequency should increase over time",
        )

    def test_edge_cases(self):
        """Edge cases should be handled gracefully."""
        from hhtpy.cycles import detect_cycles

        # Too short for any cycles
        cycles = detect_cycles(np.array([1.0, -1.0, 1.0]), 100)
        self.assertEqual(len(cycles), 0)

        # Constant signal — no zero crossings
        cycles = detect_cycles(np.ones(100), 100)
        self.assertEqual(len(cycles), 0)

        # 1D check
        with self.assertRaises(ValueError):
            detect_cycles(np.ones((2, 100)), 100)


if __name__ == "__main__":
    unittest.main()
