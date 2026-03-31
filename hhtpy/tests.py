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
)
from hhtpy.plot import plot_imfs
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


if __name__ == "__main__":
    unittest.main()
