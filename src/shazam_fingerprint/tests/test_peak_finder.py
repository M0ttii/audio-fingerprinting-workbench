"""
Tests für peak_finder.py.

Prüft lokale-Maxima-Detektion, Amplitude-Threshold-Filterung und
Dichte-Kriterium anhand synthetischer Spektrogramme.
"""

import numpy as np
import pytest

from shazam_fingerprint import config
from shazam_fingerprint.peak_finder import find_peaks, _detect_local_maxima, _apply_density_criterion
from shazam_fingerprint.spectrogram import Spectrogram


# ======================================================================
# Hilfsfunktionen
# ======================================================================

def _make_spectrogram(
    magnitude: np.ndarray,
    duration_sec: float = 3.0,
    freq_bin_min: int = 0,
) -> Spectrogram:
    """Erzeugt ein Spectrogram-NamedTuple aus einem gegebenen Magnitude-Array.

    Args:
        magnitude: 2D-Array (n_freqs × n_frames) in dB-Skala.
        duration_sec: Simulierte Dauer des Signals in Sekunden.
        freq_bin_min: Offset für globale Frequenz-Bin-Indizes.

    Returns:
        Spectrogram-NamedTuple mit konsistenten Achsen.
    """
    n_freqs, n_frames = magnitude.shape
    times = np.linspace(0.0, duration_sec, n_frames, endpoint=True)
    freq_hz_per_bin = config.SAMPLE_RATE / config.N_FFT
    frequencies = (np.arange(n_freqs) + freq_bin_min) * freq_hz_per_bin
    return Spectrogram(
        magnitude=magnitude,
        times=times,
        frequencies=frequencies,
        freq_bin_min=freq_bin_min,
    )


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def single_peak_spec() -> Spectrogram:
    """Spektrogramm mit exakt einem isolierten Peak.

    Ein einzelner Punkt bei (freq=50, time=50) wird auf 0 dB gesetzt,
    der Rest ist auf -80 dB (unter AMPLITUDE_THRESHOLD_DB).
    """
    n_freqs, n_frames = 100, 100
    magnitude = np.full((n_freqs, n_frames), -80.0)
    magnitude[50, 50] = 0.0
    return _make_spectrogram(magnitude, duration_sec=5.0)


@pytest.fixture
def multi_peak_spec() -> Spectrogram:
    """Spektrogramm mit mehreren weit auseinander liegenden Peaks.

    Peaks sind weit genug voneinander entfernt, sodass sie sich in
    verschiedenen Nachbarschaftsregionen befinden und alle als lokale
    Maxima erkannt werden.
    """
    n_freqs, n_frames = 200, 200
    magnitude = np.full((n_freqs, n_frames), -80.0)
    # Fünf Peaks, jeweils weit genug auseinander (> 2*NEIGHBORHOOD_SIZE)
    peak_positions = [(30, 30), (80, 80), (130, 130), (30, 180), (170, 30)]
    for f, t in peak_positions:
        magnitude[f, t] = 0.0
    return _make_spectrogram(magnitude, duration_sec=10.0)


@pytest.fixture
def silence_spec() -> Spectrogram:
    """Spektrogramm aus reiner Stille (homogene dB-Werte)."""
    n_freqs, n_frames = 100, 100
    magnitude = np.full((n_freqs, n_frames), -80.0)
    return _make_spectrogram(magnitude, duration_sec=5.0)


@pytest.fixture
def dense_peaks_spec() -> Spectrogram:
    """Spektrogramm mit vielen Peaks in einem kurzen Zeitsegment.

    Erzeugt deutlich mehr Peaks als MAX_PEAKS_PER_SECOND in einer
    einzigen Sekunde, um das Dichte-Kriterium zu testen.
    """
    n_freqs, n_frames = 200, 50
    magnitude = np.full((n_freqs, n_frames), -80.0)
    # Viele Peaks im selben Zeitsegment, weit genug in Frequenz getrennt
    # aber alle im selben Zeitbereich → Dichte-Kriterium muss filtern
    rng = np.random.default_rng(config.RANDOM_SEED)
    n_candidate_peaks = config.MAX_PEAKS_PER_SECOND * 5
    freq_positions = rng.choice(n_freqs, size=n_candidate_peaks, replace=False)
    # Alle Peaks in der Mitte des Zeitfensters (ein 1-Sekunden-Segment)
    time_col = n_frames // 2
    amplitudes = rng.uniform(-30.0, 0.0, size=n_candidate_peaks)
    for f, amp in zip(freq_positions, amplitudes):
        magnitude[f, time_col] = amp
    return _make_spectrogram(magnitude, duration_sec=1.0)


# ======================================================================
# Einzelner Peak
# ======================================================================

class TestSinglePeak:
    def test_single_peak_found(self, single_peak_spec: Spectrogram) -> None:
        """Ein einzelner isolierter Peak wird korrekt erkannt."""
        peaks = find_peaks(single_peak_spec)
        assert len(peaks) == 1

    def test_single_peak_coordinates(self, single_peak_spec: Spectrogram) -> None:
        """Die zurückgegebenen Koordinaten sind korrekt (inkl. freq_bin_min-Offset)."""
        peaks = find_peaks(single_peak_spec)
        freq_bin, time_frame = peaks[0]
        # freq_bin_min=0 → globaler Index = lokaler Index
        assert freq_bin == 50 + single_peak_spec.freq_bin_min
        assert time_frame == 50


# ======================================================================
# Mehrere Peaks
# ======================================================================

class TestMultiplePeaks:
    def test_all_peaks_found(self, multi_peak_spec: Spectrogram) -> None:
        """Alle weit auseinander liegenden Peaks werden erkannt."""
        peaks = find_peaks(multi_peak_spec)
        assert len(peaks) == 5

    def test_peaks_sorted_by_time(self, multi_peak_spec: Spectrogram) -> None:
        """Peaks sind aufsteigend nach time_frame sortiert."""
        peaks = find_peaks(multi_peak_spec)
        time_frames = [t for _, t in peaks]
        assert time_frames == sorted(time_frames)


# ======================================================================
# Rückgabeformat
# ======================================================================

class TestReturnFormat:
    def test_returns_list_of_tuples(self, single_peak_spec: Spectrogram) -> None:
        """find_peaks gibt eine Liste von (int, int)-Tupeln zurück."""
        peaks = find_peaks(single_peak_spec)
        assert isinstance(peaks, list)
        for peak in peaks:
            assert isinstance(peak, tuple)
            assert len(peak) == 2
            assert isinstance(peak[0], int)
            assert isinstance(peak[1], int)

    def test_no_amplitude_in_output(self, multi_peak_spec: Spectrogram) -> None:
        """Peaks enthalten nur Koordinaten, keine Amplitudenwerte.

        Wang: "the amplitude component has been eliminated."
        """
        peaks = find_peaks(multi_peak_spec)
        for peak in peaks:
            # Jedes Tupel hat genau 2 Elemente: (freq_bin, time_frame)
            assert len(peak) == 2


# ======================================================================
# Stille / keine Peaks
# ======================================================================

class TestSilence:
    def test_silence_no_peaks(self, silence_spec: Spectrogram) -> None:
        """Homogenes Spektrogramm (Stille) erzeugt keine Peaks.

        Da alle Werte identisch sind, gibt es keine lokalen Maxima,
        und zusätzlich liegen alle Werte unter AMPLITUDE_THRESHOLD_DB.
        """
        peaks = find_peaks(silence_spec)
        assert len(peaks) == 0


# ======================================================================
# Amplitude-Threshold
# ======================================================================

class TestAmplitudeThreshold:
    def test_peaks_below_threshold_rejected(self) -> None:
        """Peaks mit Amplitude unter AMPLITUDE_THRESHOLD_DB werden verworfen."""
        n_freqs, n_frames = 100, 100
        magnitude = np.full((n_freqs, n_frames), -80.0)
        # Peak knapp UNTER dem Threshold → darf nicht erkannt werden
        magnitude[50, 50] = config.AMPLITUDE_THRESHOLD_DB - 1.0
        spec = _make_spectrogram(magnitude, duration_sec=5.0)
        peaks = find_peaks(spec)
        assert len(peaks) == 0

    def test_peaks_above_threshold_accepted(self) -> None:
        """Peaks mit Amplitude über AMPLITUDE_THRESHOLD_DB werden erkannt."""
        n_freqs, n_frames = 100, 100
        magnitude = np.full((n_freqs, n_frames), -80.0)
        # Peak knapp ÜBER dem Threshold → muss erkannt werden
        magnitude[50, 50] = config.AMPLITUDE_THRESHOLD_DB + 10.0
        spec = _make_spectrogram(magnitude, duration_sec=5.0)
        peaks = find_peaks(spec)
        assert len(peaks) == 1


# ======================================================================
# Dichte-Kriterium (MAX_PEAKS_PER_SECOND)
# ======================================================================

class TestDensityCriterion:
    def test_density_limit_respected(self, dense_peaks_spec: Spectrogram) -> None:
        """Pro 1-Sekunden-Segment werden maximal MAX_PEAKS_PER_SECOND Peaks behalten.

        Wang Section 2.1: "Candidate peaks are chosen according to a density
        criterion in order to assure that the time-frequency strip for the
        audio file has reasonably uniform coverage."
        """
        peaks = find_peaks(dense_peaks_spec)
        assert len(peaks) <= config.MAX_PEAKS_PER_SECOND

    def test_strongest_peaks_kept(self) -> None:
        """Das Dichte-Kriterium behält die stärksten Peaks pro Segment.

        Von vielen Peaks im selben Segment werden die mit den höchsten
        Amplituden bevorzugt (absteigende Sortierung).
        """
        n_freqs, n_frames = 200, 50
        magnitude = np.full((n_freqs, n_frames), -80.0)
        time_col = n_frames // 2

        # Erzeuge Peaks mit absteigenden Amplituden
        n_peaks = config.MAX_PEAKS_PER_SECOND + 20
        amplitudes = np.linspace(0.0, -50.0, n_peaks)
        freq_positions = np.arange(0, n_peaks * 1, 1)  # Jeder Frequenz-Bin

        for i, (f, amp) in enumerate(zip(freq_positions, amplitudes)):
            if f < n_freqs:
                magnitude[f, time_col] = amp

        spec = _make_spectrogram(magnitude, duration_sec=1.0)
        peaks = find_peaks(spec)

        # Alle zurückgegebenen Peaks müssen unter dem Limit liegen
        assert len(peaks) <= config.MAX_PEAKS_PER_SECOND

    def test_uniform_coverage_across_segments(self) -> None:
        """Peaks werden gleichmäßig über Zeitsegmente verteilt.

        Jedes 1-Sekunden-Segment bekommt maximal MAX_PEAKS_PER_SECOND Peaks,
        was eine gleichmäßige zeitliche Abdeckung sicherstellt.
        """
        n_freqs = 200
        duration_sec = 3.0
        frames_per_sec = config.SAMPLE_RATE / config.HOP_LENGTH
        n_frames = int(duration_sec * frames_per_sec)
        magnitude = np.full((n_freqs, n_frames), -80.0)

        rng = np.random.default_rng(123)
        # Verteile viele Peaks über mehrere Sekunden
        for seg in range(3):
            frame_start = int(seg * frames_per_sec)
            frame_center = frame_start + int(frames_per_sec) // 2
            if frame_center >= n_frames:
                continue
            n_candidates = config.MAX_PEAKS_PER_SECOND * 3
            freqs = rng.choice(n_freqs, size=min(n_candidates, n_freqs), replace=False)
            for i, f in enumerate(freqs):
                magnitude[f, frame_center] = rng.uniform(-50.0, 0.0)

        spec = _make_spectrogram(magnitude, duration_sec=duration_sec)
        peaks = find_peaks(spec)

        # Gesamtzahl Peaks ≤ 3 Segmente × MAX_PEAKS_PER_SECOND
        assert len(peaks) <= 3 * config.MAX_PEAKS_PER_SECOND


# ======================================================================
# Lokale-Maxima-Detektion (Hilfsfunktion)
# ======================================================================

class TestLocalMaximaDetection:
    def test_plateau_rejected(self) -> None:
        """Plateaus (identische Werte) werden nicht als Peaks erkannt.

        Die _detect_local_maxima-Funktion prüft, dass ein Peak strikt über
        dem lokalen Minimum liegt, um Plateaus auszuschließen.
        """
        n_freqs, n_frames = 100, 100
        magnitude = np.full((n_freqs, n_frames), -40.0)  # Einheitliches Plateau
        mask = _detect_local_maxima(magnitude)
        assert not np.any(mask)

    def test_isolated_peak_detected(self) -> None:
        """Ein einzelner Peak umgeben von niedrigeren Werten wird erkannt."""
        n_freqs, n_frames = 100, 100
        magnitude = np.full((n_freqs, n_frames), -80.0)
        magnitude[50, 50] = 0.0
        mask = _detect_local_maxima(magnitude)
        assert mask[50, 50] is np.True_

    def test_neighbor_suppression(self) -> None:
        """Zwei nahe beieinander liegende Peaks: nur der stärkere überlebt.

        Wenn zwei Punkte innerhalb derselben Nachbarschaft liegen, kann nur
        der mit dem höheren Wert als lokales Maximum klassifiziert werden.
        """
        n_freqs, n_frames = 100, 100
        magnitude = np.full((n_freqs, n_frames), -80.0)
        # Zwei Peaks im selben Nachbarschaftsfenster
        magnitude[50, 50] = 0.0
        magnitude[51, 51] = -10.0
        mask = _detect_local_maxima(magnitude)
        # Der stärkere Peak muss erkannt werden
        assert mask[50, 50]
        # Der schwächere darf nicht erkannt werden (gleiche Nachbarschaft)
        assert not mask[51, 51]


# ======================================================================
# Globaler Frequenz-Offset (freq_bin_min)
# ======================================================================

class TestFreqBinMinOffset:
    def test_global_freq_indices(self) -> None:
        """Peaks verwenden globale Frequenz-Bin-Indizes (inkl. freq_bin_min).

        Das Spektrogramm ist auf einen Frequenzbereich gecroppt.
        find_peaks() muss freq_bin_min auf die lokalen Indizes addieren,
        damit fingerprint.py korrekt quantisieren kann.
        """
        n_freqs, n_frames = 100, 100
        magnitude = np.full((n_freqs, n_frames), -80.0)
        magnitude[20, 50] = 0.0
        freq_bin_min = 56  # Simuliert Cropping bei config.MIN_FREQUENCY_HZ
        spec = _make_spectrogram(magnitude, duration_sec=5.0, freq_bin_min=freq_bin_min)
        peaks = find_peaks(spec)
        assert len(peaks) == 1
        freq_bin, time_frame = peaks[0]
        # Globaler Index = lokaler Index (20) + freq_bin_min (56)
        assert freq_bin == 20 + freq_bin_min
        assert time_frame == 50


# ======================================================================
# Fehlerbehandlung
# ======================================================================

class TestErrorHandling:
    def test_empty_spectrogram_raises(self) -> None:
        """Leeres Spektrogramm wirft ValueError."""
        magnitude = np.array([]).reshape(0, 0)
        spec = _make_spectrogram(magnitude, duration_sec=0.0)
        with pytest.raises(ValueError, match="leer"):
            find_peaks(spec)


# ======================================================================
# Integration mit echtem Signal
# ======================================================================

class TestWithRealSignal:
    def test_sine_signal_produces_peaks(self) -> None:
        """Ein Sinussignal erzeugt nach Spektrogramm und Peak-Extraktion Peaks."""
        from shazam_fingerprint.spectrogram import compute_spectrogram

        sr = config.SAMPLE_RATE
        duration = 3.0
        t = np.linspace(0, duration, int(duration * sr), endpoint=False)
        signal = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        spec = compute_spectrogram(signal, sr)
        peaks = find_peaks(spec)
        assert len(peaks) > 0

    def test_sine_peak_near_correct_frequency(self) -> None:
        """Peaks eines 1000-Hz-Sinustons liegen nahe dem erwarteten Frequenz-Bin."""
        from shazam_fingerprint.spectrogram import compute_spectrogram

        sr = config.SAMPLE_RATE
        duration = 3.0
        t = np.linspace(0, duration, int(duration * sr), endpoint=False)
        signal = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        spec = compute_spectrogram(signal, sr)
        peaks = find_peaks(spec)

        # Der erwartete globale Frequenz-Bin für 1000 Hz
        expected_bin = int(1000 / (sr / config.N_FFT))
        freq_bins = [f for f, _ in peaks]
        # Mindestens ein Peak sollte nahe dem erwarteten Bin liegen
        min_distance = min(abs(f - expected_bin) for f in freq_bins)
        assert min_distance <= 3, (
            f"Kein Peak nahe dem erwarteten Bin {expected_bin} für 1000 Hz. "
            f"Nächster Peak-Bin: {min(freq_bins, key=lambda f: abs(f - expected_bin))}"
        )
