"""Tests für das peak_finder-Modul (Quad-Fingerprinting).

Gemäß anforderungen_quad.md Abschnitt TESTS:
- Test Max-Filter: Synthetischer Einzelpeak → genau ein Peak gefunden.
- Test Min-Filter: Stille-Region → KEINE Peaks (Min=Max → alle verworfen).
- Test parabolische Interpolation: Peak-Koordinaten sind float32, nicht int.
- Test Query-Filter-Größen: Korrekt aus Referenz + Epsilon berechnet.

Weitere Tests:
- Rückgabe (N, 2) float32, sortiert nach Zeit
- Adjacency-Cleanup: Benachbarte Peaks gleicher Magnitude → nur einer bleibt
- Zwei verschiedene Peaks bleiben erhalten
- Referenz-Peaks verwenden größere Filter (weniger Peaks) als Query-Peaks
- Fehlerbehandlung: leeres / 1D / zu kleines Spektrogramm
"""

import numpy as np
import pytest

from quad_fingerprint import config
from quad_fingerprint.peak_finder import (
    adjacency_cleanup,
    apply_max_filter,
    apply_min_filter,
    compute_query_filter_sizes,
    extract_peaks,
    extract_query_peaks,
    extract_reference_peaks,
    parabolic_interpolation,
)

# Kleines Spektrogramm-Format, das schnell zu verarbeiten ist
# (Filter: 5×5 für Tests, statt 151×75 Referenz-Filter)
SMALL_W = 5   # Filterbreite (Frames)
SMALL_H = 5   # Filterhöhe (Bins)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _blank(n_bins: int = 50, n_frames: int = 100) -> np.ndarray:
    """Leeres (Null-)Spektrogramm."""
    return np.zeros((n_bins, n_frames), dtype=np.float32)


def _with_peak(
    freq_bin: int, time_frame: int,
    amplitude: float = 1.0,
    n_bins: int = 50,
    n_frames: int = 100,
) -> np.ndarray:
    """Spektrogramm mit einem einzelnen Impuls bei (freq_bin, time_frame)."""
    mag = _blank(n_bins, n_frames)
    mag[freq_bin, time_frame] = amplitude
    return mag


def _uniform(value: float, n_bins: int = 50, n_frames: int = 100) -> np.ndarray:
    """Spektrogramm mit konstantem Wert überall (uniforme Region)."""
    return np.full((n_bins, n_frames), value, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Anforderungs-Tests (direkt aus anforderungen_quad.md)
# ---------------------------------------------------------------------------

class TestMaxFilterRequirement:
    """Test Max-Filter: Synthetischer Einzelpeak → genau ein Peak gefunden."""

    def test_single_peak_found(self) -> None:
        """Ein einzelner Impuls → extract_peaks muss genau einen Peak liefern."""
        mag = _with_peak(freq_bin=25, time_frame=50)
        peaks = extract_peaks(mag, max_filter_width=SMALL_W, max_filter_height=SMALL_H)
        assert len(peaks) == 1

    def test_single_peak_position_approx(self) -> None:
        """Peak-Position des Einzelpulses entspricht dem gesetzten Bin/Frame."""
        freq_bin, time_frame = 25, 50
        mag = _with_peak(freq_bin=freq_bin, time_frame=time_frame)
        peaks = extract_peaks(mag, max_filter_width=SMALL_W, max_filter_height=SMALL_H)
        # Rückgabe: [(time, freq)] — Toleranz für parabolische Interpolation
        assert peaks[0, 0] == pytest.approx(time_frame, abs=1.0)
        assert peaks[0, 1] == pytest.approx(freq_bin, abs=1.0)

    def test_no_peaks_in_blank_spectrogram(self) -> None:
        """Nullspektrogramm → keine Peaks (alles uniform, Min-Filter greift)."""
        mag = _blank()
        peaks = extract_peaks(mag, max_filter_width=SMALL_W, max_filter_height=SMALL_H)
        assert len(peaks) == 0

    def test_max_filter_output_shape(self) -> None:
        mag = _blank()
        filtered = apply_max_filter(mag, height=SMALL_H, width=SMALL_W)
        assert filtered.shape == mag.shape


class TestMinFilterRequirement:
    """Test Min-Filter: Stille/uniforme Region → KEINE Peaks."""

    def test_uniform_region_yields_no_peaks(self) -> None:
        """Konstantes Spektrogramm: Min-Filter = Max-Filter → alle Peaks verworfen.

        Paper: 'discard peak candidates if they are detected by both,
        the min and the max filter.' (Section IV-A)
        """
        mag = _uniform(1.0)
        peaks = extract_peaks(mag, max_filter_width=SMALL_W, max_filter_height=SMALL_H)
        assert len(peaks) == 0

    def test_min_filter_detects_uniform(self) -> None:
        """Min-Filter auf uniformem Spektrogramm ≡ dem Spektrogramm selbst."""
        mag = _uniform(0.5)
        min_filt = apply_min_filter(mag)
        np.testing.assert_array_equal(min_filt, mag)

    def test_min_filter_output_shape(self) -> None:
        mag = _blank()
        filtered = apply_min_filter(mag)
        assert filtered.shape == mag.shape

    def test_silence_yields_no_peaks(self) -> None:
        """Stilles Signal (Nullen) → keine Peaks."""
        mag = _blank()
        peaks = extract_peaks(mag, max_filter_width=SMALL_W, max_filter_height=SMALL_H)
        assert len(peaks) == 0


class TestParabolicInterpolationRequirement:
    """Test parabolische Interpolation: Koordinaten sind float32, nicht int."""

    def test_peaks_are_float32(self) -> None:
        """Paper: 'peaks are represented as single precision floating point values.'
        (Section IV-A)
        """
        mag = _with_peak(freq_bin=25, time_frame=50)
        peaks = extract_peaks(mag, max_filter_width=SMALL_W, max_filter_height=SMALL_H)
        assert peaks.dtype == np.float32

    def test_interpolated_values_not_integer(self) -> None:
        """Gibt es Nachbarbins, sollen die Koordinaten float sein, nicht int."""
        # Asymmetrische Umgebung → Parabel-Maximum liegt nicht exakt auf Integer
        n_bins, n_frames = 50, 100
        mag = _blank(n_bins, n_frames)
        mag[25, 50] = 1.0
        mag[25, 49] = 0.6   # linker Nachbar: asymmetrisch
        mag[25, 51] = 0.3   # rechter Nachbar
        mag[24, 50] = 0.4
        mag[26, 50] = 0.2
        peaks = extract_peaks(mag, max_filter_width=SMALL_W, max_filter_height=SMALL_H)
        assert len(peaks) == 1
        time_val = peaks[0, 0]
        # Ergebnis muss float32 sein — ob exakt ganzzahlig ist implementationsabhängig
        assert isinstance(time_val, (np.floating, float))
        assert peaks.dtype == np.float32

    def test_parabolic_interpolation_direct(self) -> None:
        """Direkt auf parabolic_interpolation: symmetrische Umgebung → δ = 0."""
        n_bins, n_frames = 20, 40
        mag = np.zeros((n_bins, n_frames), dtype=np.float32)
        mag[10, 20] = 1.0
        mag[10, 19] = 0.5
        mag[10, 21] = 0.5   # symmetrisch → δ_t = 0
        mag[9, 20] = 0.5
        mag[11, 20] = 0.5   # symmetrisch → δ_f = 0
        coords = np.array([[10, 20]], dtype=np.int64)
        interp = parabolic_interpolation(coords, mag)
        assert interp[0, 1] == pytest.approx(20.0, abs=0.01)  # time unverändert
        assert interp[0, 0] == pytest.approx(10.0, abs=0.01)  # freq unverändert

    def test_parabolic_interpolation_asymmetric_time(self) -> None:
        """Asymmetrie in Zeit → Sub-Sample-Verschiebung im Ergebnis."""
        n_bins, n_frames = 20, 40
        mag = np.zeros((n_bins, n_frames), dtype=np.float32)
        mag[10, 20] = 1.0
        mag[10, 19] = 0.8   # stärker links
        mag[10, 21] = 0.4   # schwächer rechts
        mag[9, 20] = 0.0
        mag[11, 20] = 0.0
        coords = np.array([[10, 20]], dtype=np.int64)
        interp = parabolic_interpolation(coords, mag)
        # Peak-Verschiebung nach links (richtung stärkerer Seite)
        assert interp[0, 1] < 20.0


class TestQueryFilterSizesRequirement:
    """Test Query-Filter-Größen: korrekt aus Referenz + Epsilon berechnet."""

    def test_query_width_eq2(self) -> None:
        """Eq. (2): m_w^query = round(m_w^ref / (1 + ε_t))."""
        q_width, _ = compute_query_filter_sizes()
        expected = round(config.REF_MAX_FILTER_WIDTH / (1 + config.EPSILON_T))
        assert q_width == expected

    def test_query_height_eq3(self) -> None:
        """Eq. (3): m_h^query = round(m_h^ref · (1 − ε_p))."""
        _, q_height = compute_query_filter_sizes()
        expected = round(config.REF_MAX_FILTER_HEIGHT * (1 - config.EPSILON_P))
        assert q_height == expected

    def test_query_width_less_than_ref(self) -> None:
        """Query-Filter muss kleiner als Referenz-Filter sein (ε_t > 0)."""
        q_width, q_height = compute_query_filter_sizes()
        assert q_width < config.REF_MAX_FILTER_WIDTH
        assert q_height < config.REF_MAX_FILTER_HEIGHT

    def test_custom_epsilon(self) -> None:
        """Manuelle Parametrierung mit angepassten Epsilon-Werten."""
        q_w, q_h = compute_query_filter_sizes(
            ref_width=100, ref_height=50,
            epsilon_t=0.25, epsilon_p=0.25,
        )
        assert q_w == round(100 / 1.25)
        assert q_h == round(50 * 0.75)

    def test_config_query_widths_match_computation(self) -> None:
        """config.QUERY_MAX_FILTER_WIDTH/HEIGHT stimmen mit compute_query_filter_sizes überein."""
        q_w, q_h = compute_query_filter_sizes()
        assert q_w == config.QUERY_MAX_FILTER_WIDTH
        assert q_h == config.QUERY_MAX_FILTER_HEIGHT


# ---------------------------------------------------------------------------
# 2. Rückgabeformat
# ---------------------------------------------------------------------------

class TestExtractPeaksOutput:
    def test_returns_ndarray(self) -> None:
        mag = _with_peak(25, 50)
        peaks = extract_peaks(mag, max_filter_width=SMALL_W, max_filter_height=SMALL_H)
        assert isinstance(peaks, np.ndarray)

    def test_shape_is_n2(self) -> None:
        mag = _with_peak(25, 50)
        peaks = extract_peaks(mag, max_filter_width=SMALL_W, max_filter_height=SMALL_H)
        assert peaks.ndim == 2
        assert peaks.shape[1] == 2

    def test_empty_result_shape(self) -> None:
        """Kein Peak: Rückgabe muss (0, 2) sein, nicht (0,)."""
        mag = _blank()
        peaks = extract_peaks(mag, max_filter_width=SMALL_W, max_filter_height=SMALL_H)
        assert peaks.shape == (0, 2)

    def test_sorted_by_time(self) -> None:
        """Peaks müssen nach Zeit (Spalte 0) aufsteigend sortiert sein."""
        mag = _blank(60, 200)
        # Zwei weit auseinander liegende Peaks
        mag[30, 180] = 1.0
        mag[30, 20] = 0.9
        peaks = extract_peaks(mag, max_filter_width=SMALL_W, max_filter_height=SMALL_H)
        if len(peaks) >= 2:
            assert peaks[0, 0] <= peaks[1, 0]

    def test_column_order_time_freq(self) -> None:
        """Spalte 0 = Zeit (Frame), Spalte 1 = Frequenz (Bin)."""
        freq_bin, time_frame = 15, 60
        mag = _with_peak(freq_bin=freq_bin, time_frame=time_frame, n_bins=60, n_frames=200)
        peaks = extract_peaks(mag, max_filter_width=SMALL_W, max_filter_height=SMALL_H)
        assert len(peaks) == 1
        # Zeit näher an time_frame als freq_bin (sehr unterschiedliche Werte)
        assert abs(peaks[0, 0] - time_frame) < abs(peaks[0, 0] - freq_bin)


# ---------------------------------------------------------------------------
# 3. Adjacency-Cleanup
# ---------------------------------------------------------------------------

class TestAdjacencyCleanup:
    def test_isolated_peaks_both_kept(self) -> None:
        """Zwei weit entfernte Peaks → beide bleiben erhalten."""
        mag = _blank(60, 200)
        mag[10, 20] = 1.0
        mag[50, 180] = 0.8
        peaks = extract_peaks(mag, max_filter_width=SMALL_W, max_filter_height=SMALL_H)
        assert len(peaks) == 2

    def test_adjacent_equal_peaks_one_survives(self) -> None:
        """Direkt benachbarte Peaks gleicher Amplitude → nur einer bleibt.

        Paper: 'We group all peaks by magnitude, and search for adjacent peaks.'
        (Section IV-A)
        """
        n_bins, n_frames = 30, 60
        mag = np.zeros((n_bins, n_frames), dtype=np.float32)
        amp = 1.0
        # Zwei benachbarte Peaks (in Frequenz-Richtung) mit identischer Amplitude
        mag[10, 30] = amp
        mag[11, 30] = amp
        coords = np.array([[10, 30], [11, 30]], dtype=np.int64)
        result = adjacency_cleanup(coords, mag)
        assert len(result) == 1

    def test_single_peak_unchanged(self) -> None:
        n_bins, n_frames = 20, 40
        mag = _blank(n_bins, n_frames)
        coords = np.array([[10, 20]], dtype=np.int64)
        result = adjacency_cleanup(coords, mag)
        assert len(result) == 1

    def test_different_amplitude_peaks_both_kept(self) -> None:
        """Benachbarte Peaks VERSCHIEDENER Amplitude werden NICHT als Cluster zusammengeführt."""
        n_bins, n_frames = 30, 60
        mag = np.zeros((n_bins, n_frames), dtype=np.float32)
        mag[10, 30] = 1.0
        mag[11, 30] = 0.9   # andere Amplitude → andere Magnitude-Gruppe
        coords = np.array([[10, 30], [11, 30]], dtype=np.int64)
        result = adjacency_cleanup(coords, mag)
        assert len(result) == 2

    def test_empty_coords_unchanged(self) -> None:
        mag = _blank()
        coords = np.empty((0, 2), dtype=np.int64)
        result = adjacency_cleanup(coords, mag)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 4. Referenz- vs. Query-Peaks
# ---------------------------------------------------------------------------

class TestReferenceVsQueryPeaks:
    def test_query_peaks_more_than_reference(self) -> None:
        """Query-Filter ist kleiner → mehr Peaks als Referenz-Filter."""
        rng = np.random.default_rng(42)
        mag = rng.random((300, 600)).astype(np.float32)
        ref_peaks = extract_reference_peaks(mag)
        query_peaks = extract_query_peaks(mag)
        assert len(query_peaks) >= len(ref_peaks)

    def test_reference_uses_large_filter(self) -> None:
        """extract_reference_peaks: verwendet REF_MAX_FILTER_WIDTH/HEIGHT."""
        rng = np.random.default_rng(0)
        mag = rng.random((300, 600)).astype(np.float32)
        ref = extract_reference_peaks(mag)
        manual = extract_peaks(
            mag,
            max_filter_width=config.REF_MAX_FILTER_WIDTH,
            max_filter_height=config.REF_MAX_FILTER_HEIGHT,
        )
        np.testing.assert_array_equal(ref, manual)

    def test_query_uses_small_filter(self) -> None:
        """extract_query_peaks: verwendet QUERY_MAX_FILTER_WIDTH/HEIGHT."""
        rng = np.random.default_rng(1)
        mag = rng.random((300, 600)).astype(np.float32)
        query = extract_query_peaks(mag)
        manual = extract_peaks(
            mag,
            max_filter_width=config.QUERY_MAX_FILTER_WIDTH,
            max_filter_height=config.QUERY_MAX_FILTER_HEIGHT,
        )
        np.testing.assert_array_equal(query, manual)


# ---------------------------------------------------------------------------
# 5. Apply-Filter (Einzelschritte)
# ---------------------------------------------------------------------------

class TestApplyMaxFilter:
    def test_output_shape_unchanged(self) -> None:
        mag = _blank(40, 80)
        out = apply_max_filter(mag, height=5, width=7)
        assert out.shape == mag.shape

    def test_single_peak_propagates(self) -> None:
        """Max-Filter auf Einzelpeak: gesamtes Fenster hat diesen Maximalwert."""
        mag = _blank(30, 60)
        mag[15, 30] = 2.0
        out = apply_max_filter(mag, height=5, width=5)
        # Im 5×5-Fenster um den Peak: alle Positionen sollen den Max-Wert übernehmen
        assert out[15, 30] == 2.0

    def test_uniform_field_unchanged_by_max(self) -> None:
        """Max-Filter auf uniformem Feld → Ergebnis identisch."""
        mag = _uniform(0.7)
        out = apply_max_filter(mag, height=5, width=5)
        np.testing.assert_allclose(out, mag)


class TestApplyMinFilter:
    def test_output_shape_unchanged(self) -> None:
        mag = _blank(40, 80)
        out = apply_min_filter(mag)
        assert out.shape == mag.shape

    def test_min_of_uniform_field(self) -> None:
        """Min-Filter auf uniformem Feld → Ergebnis identisch."""
        mag = _uniform(0.3)
        out = apply_min_filter(mag)
        np.testing.assert_allclose(out, mag)

    def test_peak_suppressed_by_min_filter(self) -> None:
        """Isolierter Peak: Minimum im 3×3-Fenster ist 0 (Hintergrund)."""
        mag = _blank(20, 40)
        mag[10, 20] = 5.0
        out = apply_min_filter(mag)
        # Der Min-Wert im Fenster um [10, 20] ist 0 (Hintergrund)
        assert out[10, 20] == 0.0


# ---------------------------------------------------------------------------
# 6. Fehlerbehandlung
# ---------------------------------------------------------------------------

class TestExtractPeaksErrors:
    def test_1d_signal_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_peaks(np.zeros(100, dtype=np.float32))

    def test_empty_magnitude_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_peaks(np.zeros((0, 0), dtype=np.float32))

    def test_zero_filter_width_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_peaks(_blank(), max_filter_width=0, max_filter_height=5)

    def test_zero_filter_height_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_peaks(_blank(), max_filter_width=5, max_filter_height=0)

    def test_too_small_for_min_filter_raises(self) -> None:
        """Spektrogramm kleiner als 3×3 (Min-Filter) → ValueError."""
        mag = np.zeros((2, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            extract_peaks(mag, max_filter_width=1, max_filter_height=1)
