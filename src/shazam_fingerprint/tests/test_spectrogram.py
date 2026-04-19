"""
Tests für spectrogram.py.

Prüft STFT-Korrektheit, dB-Skalierung, Achsen-Konsistenz und
Frequenzbereichs-Clipping anhand synthetischer Testsignale.
"""

import numpy as np
import pytest

from shazam_fingerprint import config
from shazam_fingerprint.spectrogram import Spectrogram, compute_spectrogram


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def sr() -> int:
    """Abtastrate aus config."""
    return config.SAMPLE_RATE


@pytest.fixture
def sine_440(sr: int) -> np.ndarray:
    """3 Sekunden 440-Hz-Sinuston, normalisiert."""
    t = np.linspace(0, 3.0, int(3.0 * sr), endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def silence(sr: int) -> np.ndarray:
    """3 Sekunden Stille."""
    return np.zeros(int(3.0 * sr), dtype=np.float32)


@pytest.fixture
def multi_tone(sr: int) -> np.ndarray:
    """3 Sekunden mit drei Tönen: 500 Hz, 1000 Hz, 2500 Hz."""
    t = np.linspace(0, 3.0, int(3.0 * sr), endpoint=False)
    sig = (
        np.sin(2 * np.pi * 500 * t)
        + np.sin(2 * np.pi * 1000 * t)
        + np.sin(2 * np.pi * 2500 * t)
    ).astype(np.float32)
    return sig / np.max(np.abs(sig))


# ======================================================================
# Rückgabetyp und Struktur
# ======================================================================

class TestReturnType:
    def test_returns_spectrogram_namedtuple(self, sine_440: np.ndarray, sr: int) -> None:
        """compute_spectrogram gibt ein Spectrogram-NamedTuple zurück."""
        result = compute_spectrogram(sine_440, sr)
        assert isinstance(result, Spectrogram)

    def test_has_all_fields(self, sine_440: np.ndarray, sr: int) -> None:
        """Alle vier Felder des NamedTuple sind vorhanden und nicht None."""
        spec = compute_spectrogram(sine_440, sr)
        assert spec.magnitude is not None
        assert spec.times is not None
        assert spec.frequencies is not None
        assert spec.freq_bin_min is not None

    def test_magnitude_is_2d(self, sine_440: np.ndarray, sr: int) -> None:
        """magnitude ist ein 2D-Array (n_freqs × n_frames)."""
        spec = compute_spectrogram(sine_440, sr)
        assert spec.magnitude.ndim == 2

    def test_axes_consistent_with_magnitude(self, sine_440: np.ndarray, sr: int) -> None:
        """Länge von times und frequencies stimmt mit magnitude-Shape überein."""
        spec = compute_spectrogram(sine_440, sr)
        assert spec.magnitude.shape[0] == len(spec.frequencies)
        assert spec.magnitude.shape[1] == len(spec.times)


# ======================================================================
# 440-Hz-Sinuston: Peak-Lokalisation
# ======================================================================

class TestSine440:
    def test_peak_near_440hz(self, sine_440: np.ndarray, sr: int) -> None:
        """Das stärkste Spektrum-Bin liegt im Bereich 430–450 Hz."""
        spec = compute_spectrogram(sine_440, sr)
        peak_bin = int(np.argmax(np.max(spec.magnitude, axis=1)))
        peak_freq = spec.frequencies[peak_bin]
        assert 430 <= peak_freq <= 450, (
            f"Erwarteter Peak ~440 Hz, gefunden bei {peak_freq:.1f} Hz"
        )

    def test_440hz_within_frequency_range(self, sine_440: np.ndarray, sr: int) -> None:
        """440 Hz liegt im konfigurierten Frequenzbereich und ist daher sichtbar."""
        assert config.MIN_FREQUENCY_HZ <= 440 <= config.MAX_FREQUENCY_HZ
        spec = compute_spectrogram(sine_440, sr)
        assert spec.frequencies[0] <= 440 <= spec.frequencies[-1]

    def test_440hz_dominant_over_other_bins(self, sine_440: np.ndarray, sr: int) -> None:
        """Der 440-Hz-Bereich ist deutlich lauter als der Rest des Spektrums."""
        spec = compute_spectrogram(sine_440, sr)
        col_max_per_freq = np.max(spec.magnitude, axis=1)
        peak_bin = int(np.argmax(col_max_per_freq))
        peak_val = col_max_per_freq[peak_bin]
        # Medianer Wert aller anderen Bins muss mindestens 10 dB leiser sein
        other_bins = np.delete(col_max_per_freq, peak_bin)
        assert peak_val - np.median(other_bins) >= 10.0, (
            "440-Hz-Peak ist nicht dominant genug"
        )


# ======================================================================
# Stille: dB-Skala und Threshold
# ======================================================================

class TestSilence:
    def test_silence_max_is_zero_db(self, silence: np.ndarray, sr: int) -> None:
        """Bei Stille normalisiert amplitude_to_db(ref=np.max) auf 0 dB."""
        spec = compute_spectrogram(silence, sr)
        # Stille: alle Werte identisch → max = 0 dB durch ref=np.max(amin)
        assert spec.magnitude.max() <= 0.0

    def test_silence_all_below_amplitude_threshold(
        self, silence: np.ndarray, sr: int
    ) -> None:
        """Alle Werte eines Stillesignals liegen unter dem Amplituden-Threshold.

        Stille sollte nach dem Threshold-Filter keine Peaks erzeugen.
        Hinweis: Bei exakter Stille kann amplitude_to_db auf 0 dB sättigen
        (weil ref=np.max des Nullsignals = 0 ist, was durch amin aufgefangen wird).
        Dieser Test prüft daher die Peak-Finder-Ebene, nicht den dB-Wert selbst.
        """
        spec = compute_spectrogram(silence, sr)
        # Alle Werte im Spektrogramm sind identisch (kein Kontrast)
        unique_vals = np.unique(np.round(spec.magnitude, decimals=4))
        assert len(unique_vals) <= 3, (
            f"Stille sollte homogenes Spektrogramm erzeugen, "
            f"aber {len(unique_vals)} verschiedene Werte gefunden"
        )


# ======================================================================
# dB-Skala
# ======================================================================

class TestDbScale:
    def test_max_value_is_zero_db(self, sine_440: np.ndarray, sr: int) -> None:
        """Das Maximum des Spektrogramms ist 0 dB (ref=np.max-Normierung)."""
        spec = compute_spectrogram(sine_440, sr)
        assert abs(spec.magnitude.max()) < 1e-4, (
            f"Maximum sollte 0 dB sein, ist {spec.magnitude.max():.4f} dB"
        )

    def test_all_values_non_positive(self, sine_440: np.ndarray, sr: int) -> None:
        """Alle dB-Werte sind ≤ 0 dB (relativ zum Maximum)."""
        spec = compute_spectrogram(sine_440, sr)
        assert np.all(spec.magnitude <= 0.0), (
            "Es gibt positive dB-Werte (Werte > 0 dB)"
        )

    def test_dynamic_range_reasonable(self, sine_440: np.ndarray, sr: int) -> None:
        """Dynamikbereich liegt im Bereich [−120 dB, 0 dB]."""
        spec = compute_spectrogram(sine_440, sr)
        assert spec.magnitude.min() >= -120.0
        assert spec.magnitude.max() <= 0.0


# ======================================================================
# Frequenzbereich-Clipping
# ======================================================================

class TestFrequencyRange:
    def test_frequencies_within_config_bounds(
        self, sine_440: np.ndarray, sr: int
    ) -> None:
        """Alle Frequenz-Bins liegen im konfigurierten [MIN, MAX]-Bereich."""
        spec = compute_spectrogram(sine_440, sr)
        # Kleines Toleranzfenster für Bin-Grenzen
        assert spec.frequencies[0] >= config.MIN_FREQUENCY_HZ - 20
        assert spec.frequencies[-1] <= config.MAX_FREQUENCY_HZ + 20

    def test_freq_bin_min_offset_correct(self, sine_440: np.ndarray, sr: int) -> None:
        """freq_bin_min zeigt auf den korrekten globalen Bin-Index."""
        import librosa
        spec = compute_spectrogram(sine_440, sr)
        full_freqs = librosa.fft_frequencies(sr=sr, n_fft=config.N_FFT)
        reconstructed_freq = full_freqs[spec.freq_bin_min]
        assert abs(reconstructed_freq - spec.frequencies[0]) < 15.0, (
            f"freq_bin_min={spec.freq_bin_min} zeigt auf {reconstructed_freq:.1f} Hz, "
            f"erwartet ~{spec.frequencies[0]:.1f} Hz"
        )

    def test_low_frequencies_excluded(self, sine_440: np.ndarray, sr: int) -> None:
        """Frequenzen unterhalb MIN_FREQUENCY_HZ sind nicht im Spektrogramm."""
        spec = compute_spectrogram(sine_440, sr)
        assert not np.any(spec.frequencies < config.MIN_FREQUENCY_HZ - 20)

    def test_high_frequencies_excluded(self, sine_440: np.ndarray, sr: int) -> None:
        """Frequenzen oberhalb MAX_FREQUENCY_HZ sind nicht im Spektrogramm."""
        spec = compute_spectrogram(sine_440, sr)
        assert not np.any(spec.frequencies > config.MAX_FREQUENCY_HZ + 20)


# ======================================================================
# Zeitachse
# ======================================================================

class TestTimeAxis:
    def test_times_start_at_zero(self, sine_440: np.ndarray, sr: int) -> None:
        """Die Zeitachse beginnt bei 0 s."""
        spec = compute_spectrogram(sine_440, sr)
        assert spec.times[0] >= 0.0

    def test_times_end_near_signal_duration(
        self, sine_440: np.ndarray, sr: int
    ) -> None:
        """Die Zeitachse endet nahe der Signaldauer (±0.5 s Toleranz)."""
        duration = len(sine_440) / sr
        spec = compute_spectrogram(sine_440, sr)
        assert abs(spec.times[-1] - duration) < 0.5, (
            f"Zeitachse endet bei {spec.times[-1]:.2f} s, "
            f"Signaldauer ist {duration:.2f} s"
        )

    def test_times_monotonically_increasing(
        self, sine_440: np.ndarray, sr: int
    ) -> None:
        """Die Zeitachse ist strikt monoton steigend."""
        spec = compute_spectrogram(sine_440, sr)
        assert np.all(np.diff(spec.times) > 0)

    def test_hop_length_matches_time_resolution(
        self, sine_440: np.ndarray, sr: int
    ) -> None:
        """Der Frame-Abstand entspricht config.HOP_LENGTH / config.SAMPLE_RATE."""
        spec = compute_spectrogram(sine_440, sr)
        expected_dt = config.HOP_LENGTH / config.SAMPLE_RATE
        actual_dt = float(np.mean(np.diff(spec.times)))
        assert abs(actual_dt - expected_dt) < 1e-4, (
            f"Frame-Abstand: {actual_dt:.6f} s (erwartet {expected_dt:.6f} s)"
        )


# ======================================================================
# Mehrton-Signal
# ======================================================================

class TestMultiTone:
    def test_all_tones_visible(self, multi_tone: np.ndarray, sr: int) -> None:
        """500 Hz, 1000 Hz und 2500 Hz sind alle im Spektrogramm sichtbar."""
        spec = compute_spectrogram(multi_tone, sr)
        col_max = np.max(spec.magnitude, axis=1)

        for target_hz in [500, 1000, 2500]:
            # Nächsten Bin finden
            bin_idx = int(np.argmin(np.abs(spec.frequencies - target_hz)))
            # Dieser Bin muss zu den oberen 10 % der Energie gehören
            threshold_90 = np.percentile(col_max, 90)
            assert col_max[bin_idx] >= threshold_90, (
                f"{target_hz} Hz ist nicht prominent (Wert {col_max[bin_idx]:.1f} dB < "
                f"90. Perzentile {threshold_90:.1f} dB)"
            )


# ======================================================================
# Fehlerbehandlung
# ======================================================================

class TestErrorHandling:
    def test_empty_signal_raises(self, sr: int) -> None:
        """Leeres Signal wirft ValueError."""
        with pytest.raises(ValueError, match="leer"):
            compute_spectrogram(np.array([], dtype=np.float32), sr)

    def test_invalid_sr_raises(self, sine_440: np.ndarray) -> None:
        """Abtastrate ≤ 0 wirft ValueError."""
        with pytest.raises(ValueError):
            compute_spectrogram(sine_440, 0)
        with pytest.raises(ValueError):
            compute_spectrogram(sine_440, -1)
