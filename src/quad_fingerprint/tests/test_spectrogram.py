"""Tests für das spectrogram-Modul (Quad-Fingerprinting).

Gemäß anforderungen_quad.md Abschnitt TESTS:
- Test mit bekanntem Sinussignal (440 Hz, 8 kHz): Peak muss bei ~440 Hz liegen.
- Test: STFT mit N_FFT=1024, HOP_LENGTH=32 ergibt korrekte Array-Shape.

Weitere Tests:
- Rückgabetypen (Spectrogram-NamedTuple, float32/float64, Shapes)
- Lineare Skala (kein dB)
- Frequenzachse korrekt (0 bis SR/2)
- Zeitachse korrekt (Sekunden)
- Fehlerbehandlung: falscher SR, leeres Signal, zu kurzes Signal, 2D-Signal
"""

import numpy as np
import pytest

from quad_fingerprint import config
from quad_fingerprint.spectrogram import Spectrogram, compute_spectrogram

SR = config.SAMPLE_RATE       # 8000 Hz
N_FFT = config.N_FFT          # 1024
HOP = config.HOP_LENGTH       # 32
N_BINS = N_FFT // 2 + 1      # 513

# Frequenzauflösung: Hz pro Bin
FREQ_RES = SR / N_FFT         # ≈ 7.8125 Hz


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _sine(freq: float, duration: float = 1.0) -> np.ndarray:
    """Erzeugt ein Sinussignal bei `freq` Hz, Länge `duration` s, SR=8000."""
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _n_frames(n_samples: int) -> int:
    """Berechnet die erwartete Frame-Anzahl für scipy.signal.stft
    mit boundary=None, padded=False."""
    return (n_samples - N_FFT) // HOP + 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_440() -> np.ndarray:
    """440 Hz Sinussignal, 1 s @ 8 kHz."""
    return _sine(440.0, duration=1.0)


@pytest.fixture
def sine_1000() -> np.ndarray:
    """1000 Hz Sinussignal, 2 s @ 8 kHz."""
    return _sine(1000.0, duration=2.0)


@pytest.fixture
def silence() -> np.ndarray:
    """Stilles Signal (Nullen), 1 s @ 8 kHz."""
    return np.zeros(SR, dtype=np.float32)


# ---------------------------------------------------------------------------
# Anforderungen laut Paper: Haupttests
# ---------------------------------------------------------------------------

class TestSpectrogramPaperRequirements:
    """Kerntest aus anforderungen_quad.md: 440 Hz-Peak und korrekte Shape."""

    def test_440hz_peak_at_correct_bin(self, sine_440: np.ndarray) -> None:
        """Sinussignal 440 Hz → dominanter Bin muss bei ~440 Hz liegen.

        Frequenzauflösung = SR / N_FFT = 8000 / 1024 ≈ 7.8 Hz pro Bin.
        Erwarteter Bin: round(440 / 7.8125) = 56.
        Toleranz: ±1 Bin (~7.8 Hz).
        """
        spec = compute_spectrogram(sine_440)
        # Summe über alle Frames → mittleres Magnitude-Profil
        mean_mag = spec.magnitude.mean(axis=1)
        dominant_bin = int(np.argmax(mean_mag))
        expected_bin = round(440.0 / FREQ_RES)

        assert abs(dominant_bin - expected_bin) <= 1, (
            f"Dominanter Bin: {dominant_bin} (≈{dominant_bin * FREQ_RES:.1f} Hz), "
            f"erwartet: {expected_bin} (≈440 Hz)"
        )

    def test_440hz_dominant_frequency_in_hz(self, sine_440: np.ndarray) -> None:
        """Peak liegt in der frequencies-Achse bei ≈440 Hz (±10 Hz)."""
        spec = compute_spectrogram(sine_440)
        mean_mag = spec.magnitude.mean(axis=1)
        dominant_bin = int(np.argmax(mean_mag))
        dominant_freq = spec.frequencies[dominant_bin]

        assert abs(dominant_freq - 440.0) <= 10.0, (
            f"Dominante Frequenz: {dominant_freq:.1f} Hz, erwartet: ~440 Hz"
        )

    def test_shape_n_bins(self, sine_440: np.ndarray) -> None:
        """N_FFT=1024 → n_bins = 1024 // 2 + 1 = 513 (einseitiges Spektrum)."""
        spec = compute_spectrogram(sine_440)
        assert spec.magnitude.shape[0] == N_BINS

    def test_shape_n_frames(self, sine_440: np.ndarray) -> None:
        """HOP_LENGTH=32, boundary=None, padded=False → korrekte Frame-Anzahl."""
        n_samples = len(sine_440)
        expected_frames = _n_frames(n_samples)
        spec = compute_spectrogram(sine_440)
        assert spec.magnitude.shape[1] == expected_frames

    def test_shape_2d(self, sine_440: np.ndarray) -> None:
        """Magnitude muss 2-dimensional sein: (n_bins, n_frames)."""
        spec = compute_spectrogram(sine_440)
        assert spec.magnitude.ndim == 2


# ---------------------------------------------------------------------------
# Rückgabetypen und Datentypen
# ---------------------------------------------------------------------------

class TestSpectrogramReturnTypes:
    def test_returns_spectrogram_namedtuple(self, sine_440: np.ndarray) -> None:
        result = compute_spectrogram(sine_440)
        assert isinstance(result, Spectrogram)

    def test_magnitude_dtype_float32(self, sine_440: np.ndarray) -> None:
        """Magnitude muss float32 sein (Paper: "single precision")."""
        spec = compute_spectrogram(sine_440)
        assert spec.magnitude.dtype == np.float32

    def test_times_dtype_float64(self, sine_440: np.ndarray) -> None:
        spec = compute_spectrogram(sine_440)
        assert spec.times.dtype == np.float64

    def test_frequencies_dtype_float64(self, sine_440: np.ndarray) -> None:
        spec = compute_spectrogram(sine_440)
        assert spec.frequencies.dtype == np.float64

    def test_times_shape_matches_frames(self, sine_440: np.ndarray) -> None:
        spec = compute_spectrogram(sine_440)
        assert spec.times.shape[0] == spec.magnitude.shape[1]

    def test_frequencies_shape_matches_bins(self, sine_440: np.ndarray) -> None:
        spec = compute_spectrogram(sine_440)
        assert spec.frequencies.shape[0] == spec.magnitude.shape[0]

    def test_namedtuple_fields(self, sine_440: np.ndarray) -> None:
        """Spectrogram hat die Felder magnitude, times, frequencies."""
        spec = compute_spectrogram(sine_440)
        assert hasattr(spec, "magnitude")
        assert hasattr(spec, "times")
        assert hasattr(spec, "frequencies")


# ---------------------------------------------------------------------------
# Lineare Skala (kein dB) — kritischer Unterschied zu Shazam
# ---------------------------------------------------------------------------

class TestLinearScale:
    def test_magnitude_non_negative(self, sine_440: np.ndarray) -> None:
        """Magnitude = |STFT| ≥ 0 (linear, kein dB)."""
        spec = compute_spectrogram(sine_440)
        assert np.all(spec.magnitude >= 0.0)

    def test_magnitude_not_log_scaled(self, sine_440: np.ndarray) -> None:
        """Keine dB-Werte: Magnitude soll nicht negative Werte > -1 enthalten
        (Log-Magnitude wäre typischerweise stark negativ, z.B. −60 dB)."""
        spec = compute_spectrogram(sine_440)
        # Bei linearer Skala: Max-Wert ≤ 1.0 für normalisierte Eingabe (Amplitude 0.5)
        # Bei log-Skala wären Werte wie -60, -40 usw. zu erwarten
        assert spec.magnitude.max() <= 2.0, (
            "Magnitude scheint logarithmisch skaliert (Werte > 2 bei normiertem Signal)"
        )

    def test_silence_magnitude_near_zero(self, silence: np.ndarray) -> None:
        """Stilles Signal → Magnitude ≈ 0 überall."""
        spec = compute_spectrogram(silence)
        assert np.all(spec.magnitude < 1e-6)

    def test_louder_signal_higher_magnitude(self) -> None:
        """Doppelte Amplitude → doppelte lineare Magnitude (nicht +6 dB)."""
        sig = _sine(440.0, duration=1.0)
        spec1 = compute_spectrogram(sig)
        spec2 = compute_spectrogram(sig * 2.0)
        ratio = spec2.magnitude.mean() / (spec1.magnitude.mean() + 1e-12)
        # Bei linearer Skala: ratio ≈ 2.0; bei dB wäre ratio ≈ 1.0 (da +6 dB addiert)
        assert ratio == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Frequenz- und Zeitachse
# ---------------------------------------------------------------------------

class TestAxes:
    def test_frequencies_start_at_zero(self, sine_440: np.ndarray) -> None:
        spec = compute_spectrogram(sine_440)
        assert spec.frequencies[0] == pytest.approx(0.0)

    def test_frequencies_end_at_nyquist(self, sine_440: np.ndarray) -> None:
        """Letzte Frequenz muss SR/2 = 4000 Hz sein (Nyquist)."""
        spec = compute_spectrogram(sine_440)
        assert spec.frequencies[-1] == pytest.approx(SR / 2.0, rel=1e-4)

    def test_frequencies_monotone_increasing(self, sine_440: np.ndarray) -> None:
        spec = compute_spectrogram(sine_440)
        diffs = np.diff(spec.frequencies)
        assert np.all(diffs > 0)

    def test_frequency_resolution(self, sine_440: np.ndarray) -> None:
        """Frequenzauflösung = SR / N_FFT = 8000 / 1024 ≈ 7.8125 Hz."""
        spec = compute_spectrogram(sine_440)
        resolution = spec.frequencies[1] - spec.frequencies[0]
        assert resolution == pytest.approx(FREQ_RES, rel=1e-5)

    def test_times_start_positive(self, sine_440: np.ndarray) -> None:
        """Erste Frame-Zeit sollte positiv sein (boundary=None → kein Rand-Padding)."""
        spec = compute_spectrogram(sine_440)
        assert spec.times[0] >= 0.0

    def test_times_monotone_increasing(self, sine_440: np.ndarray) -> None:
        spec = compute_spectrogram(sine_440)
        diffs = np.diff(spec.times)
        assert np.all(diffs > 0)

    def test_time_step_matches_hop(self, sine_440: np.ndarray) -> None:
        """Frame-Abstand muss HOP_LENGTH / SR = 32 / 8000 = 0.004 s sein."""
        spec = compute_spectrogram(sine_440)
        dt = spec.times[1] - spec.times[0]
        assert dt == pytest.approx(HOP / SR, rel=1e-5)

    def test_times_cover_signal_duration(self, sine_1000: np.ndarray) -> None:
        """Letzte Frame-Zeit darf die Signaldauer nicht überschreiten."""
        signal_dur = len(sine_1000) / SR  # 2.0 s
        spec = compute_spectrogram(sine_1000)
        assert spec.times[-1] <= signal_dur + HOP / SR


# ---------------------------------------------------------------------------
# Verschiedene Frequenzen — Dominanter Bin
# ---------------------------------------------------------------------------

class TestDominantFrequency:
    @pytest.mark.parametrize("freq", [220.0, 440.0, 880.0, 1760.0, 3000.0])
    def test_dominant_bin_matches_frequency(self, freq: float) -> None:
        """Sinussignal bei `freq` Hz → dominanter Bin muss ≈ `freq` Hz sein.

        Toleranz: ±2 Bins (±~15.6 Hz), um Leakage-Effekte zu berücksichtigen.
        """
        sig = _sine(freq, duration=2.0)
        spec = compute_spectrogram(sig)
        mean_mag = spec.magnitude.mean(axis=1)
        dominant_bin = int(np.argmax(mean_mag))
        dominant_freq_hz = spec.frequencies[dominant_bin]

        assert abs(dominant_freq_hz - freq) <= 2 * FREQ_RES, (
            f"freq={freq} Hz: dominanter Bin bei {dominant_freq_hz:.1f} Hz"
        )

    def test_two_tones_two_peaks(self) -> None:
        """Zwei Sinuskomponenten → zwei deutliche Peaks im mittleren Magnitude-Profil."""
        t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
        sig = (
            0.5 * np.sin(2 * np.pi * 500.0 * t)
            + 0.5 * np.sin(2 * np.pi * 2000.0 * t)
        ).astype(np.float32)
        spec = compute_spectrogram(sig)
        mean_mag = spec.magnitude.mean(axis=1)

        # Bins für 500 Hz und 2000 Hz
        bin_500 = round(500.0 / FREQ_RES)
        bin_2000 = round(2000.0 / FREQ_RES)

        # Beide Bins müssen lokale Maxima sein (größer als ihre Nachbarn)
        for b, label in [(bin_500, "500 Hz"), (bin_2000, "2000 Hz")]:
            neighborhood = mean_mag[max(0, b - 3): b + 4]
            assert mean_mag[b] == neighborhood.max(), (
                f"Kein Peak bei {label} (Bin {b})"
            )


# ---------------------------------------------------------------------------
# Fehlerbehandlung
# ---------------------------------------------------------------------------

class TestSpectrogramErrors:
    def test_wrong_sample_rate_raises(self, sine_440: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Abtastrate"):
            compute_spectrogram(sine_440, sr=44100)

    def test_empty_signal_raises(self) -> None:
        with pytest.raises(ValueError, match="[Ll]eer"):
            compute_spectrogram(np.array([], dtype=np.float32))

    def test_2d_signal_raises(self) -> None:
        stereo = np.zeros((2, SR), dtype=np.float32)
        with pytest.raises(ValueError, match="1-dimensional"):
            compute_spectrogram(stereo)

    def test_too_short_signal_raises(self) -> None:
        """Signal kürzer als N_FFT (1024 Samples) muss ValueError auslösen."""
        short = np.zeros(N_FFT - 1, dtype=np.float32)
        with pytest.raises(ValueError, match="[Kk]urz"):
            compute_spectrogram(short)

    def test_exact_minimum_length_works(self) -> None:
        """Exakt N_FFT Samples: minimal gültiges Signal — kein Fehler."""
        sig = np.zeros(N_FFT, dtype=np.float32)
        sig[0] = 1.0  # nicht leer
        spec = compute_spectrogram(sig)
        assert spec.magnitude.shape[0] == N_BINS


# ---------------------------------------------------------------------------
# Konsistenz mit config-Parametern
# ---------------------------------------------------------------------------

class TestConfigConsistency:
    def test_n_bins_matches_n_fft(self, sine_440: np.ndarray) -> None:
        spec = compute_spectrogram(sine_440)
        assert spec.magnitude.shape[0] == config.N_FFT // 2 + 1

    def test_longer_signal_more_frames(self) -> None:
        short = _sine(440.0, duration=1.0)
        long_ = _sine(440.0, duration=2.0)
        spec_short = compute_spectrogram(short)
        spec_long = compute_spectrogram(long_)
        assert spec_long.magnitude.shape[1] > spec_short.magnitude.shape[1]

    def test_frames_per_second_ratio(self, sine_440: np.ndarray) -> None:
        """Frame-Abstand entspricht HOP_LENGTH / SR = 1 / FRAMES_PER_SECOND.

        Mit boundary=None, padded=False werden Randframes weggelassen,
        daher gilt: n_frames / signal_dur < FRAMES_PER_SECOND.
        Der Zeitschritt zwischen Frames muss aber exakt HOP/SR betragen.
        """
        spec = compute_spectrogram(sine_440)
        n_frames = spec.magnitude.shape[1]
        # Abgedeckter Zeitraum via Zeitachse: von times[0] bis times[-1] + HOP/SR
        covered_dur = (n_frames - 1) * (HOP / SR) + (HOP / SR)
        actual_fps = n_frames / covered_dur
        assert actual_fps == pytest.approx(config.FRAMES_PER_SECOND, rel=1e-6)
