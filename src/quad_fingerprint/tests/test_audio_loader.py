"""Tests für das audio_loader-Modul (Quad-Fingerprinting).

Überprüft:
- Korrekte Rückgabetypen und -formen (float32, 1D, sr=8000)
- Peak-Normalisierung auf [-1, 1]
- Mono-Konvertierung bei Stereo-Eingabe
- offset_sec und duration_sec Ausschnitte
- Vollständige Metadaten-Keys
- FileNotFoundError / ValueError für Fehlereingaben
- load_audio_excerpt als Wrapper
- load_audio_directory als Generator
- get_audio_info ohne Signal-Dekodierung
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from quad_fingerprint import config
from quad_fingerprint.audio_loader import (
    SUPPORTED_EXTENSIONS,
    get_audio_info,
    load_audio,
    load_audio_directory,
    load_audio_excerpt,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetische WAV-Dateien
# ---------------------------------------------------------------------------

SR_NATIVE = 16_000  # Quell-SR != 8000 → erzwingt Resampling


def _write_sine(
    path: Path,
    freq: float = 440.0,
    duration: float = 2.0,
    sr: int = SR_NATIVE,
    channels: int = 1,
) -> None:
    """Schreibt eine Sinus-WAV-Datei (ggf. Stereo)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (0.6 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels == 2:
        signal = np.column_stack([mono, mono * 0.5])
    else:
        signal = mono
    sf.write(str(path), signal, sr)


@pytest.fixture
def mono_wav(tmp_path: Path) -> Path:
    """Mono-WAV, 2 s, 440 Hz, SR=16000."""
    p = tmp_path / "sine_mono.wav"
    _write_sine(p, freq=440.0, duration=2.0, sr=SR_NATIVE, channels=1)
    return p


@pytest.fixture
def stereo_wav(tmp_path: Path) -> Path:
    """Stereo-WAV, 3 s, 880 Hz, SR=16000."""
    p = tmp_path / "sine_stereo.wav"
    _write_sine(p, freq=880.0, duration=3.0, sr=SR_NATIVE, channels=2)
    return p


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """Stilles WAV (alle Nullen), 1 s."""
    p = tmp_path / "silence.wav"
    sf.write(str(p), np.zeros(SR_NATIVE, dtype=np.float32), SR_NATIVE)
    return p


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    """Verzeichnis mit 3 WAV-Dateien."""
    d = tmp_path / "audio"
    d.mkdir()
    for i, freq in enumerate([220.0, 440.0, 880.0], start=1):
        _write_sine(d / f"song{i}.wav", freq=freq, duration=1.5, sr=SR_NATIVE)
    return d


# ---------------------------------------------------------------------------
# load_audio — Rückgabetypen und Basisverhalten
# ---------------------------------------------------------------------------

class TestLoadAudioReturnTypes:
    def test_returns_tuple_of_three(self, mono_wav: Path) -> None:
        result = load_audio(mono_wav)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_signal_is_float32(self, mono_wav: Path) -> None:
        signal, _, _ = load_audio(mono_wav)
        assert signal.dtype == np.float32

    def test_signal_is_1d(self, mono_wav: Path) -> None:
        signal, _, _ = load_audio(mono_wav)
        assert signal.ndim == 1

    def test_sample_rate_is_config(self, mono_wav: Path) -> None:
        _, sr, _ = load_audio(mono_wav)
        assert sr == config.SAMPLE_RATE

    def test_metadata_is_dict(self, mono_wav: Path) -> None:
        _, _, meta = load_audio(mono_wav)
        assert isinstance(meta, dict)


# ---------------------------------------------------------------------------
# load_audio — Resampling
# ---------------------------------------------------------------------------

class TestLoadAudioResampling:
    def test_resampled_length_approx(self, mono_wav: Path) -> None:
        """2 s bei 16 kHz → 2 s bei 8 kHz → ~16000 Samples."""
        signal, sr, meta = load_audio(mono_wav)
        expected = 2.0 * config.SAMPLE_RATE
        # Kleine Abweichung durch Resampling-Randeffekte erlaubt (±1%)
        assert abs(len(signal) - expected) / expected < 0.02

    def test_duration_metadata_matches_signal(self, mono_wav: Path) -> None:
        signal, sr, meta = load_audio(mono_wav)
        assert meta["duration_sec"] == pytest.approx(len(signal) / sr, rel=1e-3)


# ---------------------------------------------------------------------------
# load_audio — Peak-Normalisierung
# ---------------------------------------------------------------------------

class TestLoadAudioNormalization:
    def test_peak_amplitude_at_most_one(self, mono_wav: Path) -> None:
        signal, _, _ = load_audio(mono_wav)
        assert float(np.max(np.abs(signal))) == pytest.approx(1.0, abs=1e-5)

    def test_values_in_minus_one_to_one(self, mono_wav: Path) -> None:
        signal, _, _ = load_audio(mono_wav)
        assert np.all(signal >= -1.0)
        assert np.all(signal <= 1.0)

    def test_silent_audio_no_crash(self, silent_wav: Path) -> None:
        """Stilles Audio (Peak=0) soll keinen Division-by-Zero-Fehler werfen."""
        signal, sr, meta = load_audio(silent_wav)
        assert signal is not None
        assert np.all(signal == 0.0)


# ---------------------------------------------------------------------------
# load_audio — Mono-Konvertierung
# ---------------------------------------------------------------------------

class TestLoadAudioMono:
    def test_stereo_becomes_1d(self, stereo_wav: Path) -> None:
        signal, _, _ = load_audio(stereo_wav)
        assert signal.ndim == 1

    def test_stereo_normalized(self, stereo_wav: Path) -> None:
        signal, _, _ = load_audio(stereo_wav)
        assert float(np.max(np.abs(signal))) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# load_audio — Metadaten-Keys
# ---------------------------------------------------------------------------

class TestLoadAudioMetadata:
    EXPECTED_KEYS = {
        "file_path",
        "file_name",
        "duration_sec",
        "original_sr",
        "original_duration_sec",
        "load_time_ms",
        "offset_sec",
        "peak_amplitude",
        "n_channels",
    }

    def test_all_keys_present(self, mono_wav: Path) -> None:
        _, _, meta = load_audio(mono_wav)
        assert self.EXPECTED_KEYS.issubset(set(meta.keys()))

    def test_file_name_correct(self, mono_wav: Path) -> None:
        _, _, meta = load_audio(mono_wav)
        assert meta["file_name"] == "sine_mono.wav"

    def test_original_sr_is_native(self, mono_wav: Path) -> None:
        _, _, meta = load_audio(mono_wav)
        assert meta["original_sr"] == SR_NATIVE

    def test_offset_sec_stored(self, mono_wav: Path) -> None:
        _, _, meta = load_audio(mono_wav, offset_sec=0.5)
        assert meta["offset_sec"] == pytest.approx(0.5)

    def test_peak_amplitude_positive(self, mono_wav: Path) -> None:
        _, _, meta = load_audio(mono_wav)
        assert meta["peak_amplitude"] > 0.0

    def test_load_time_ms_positive(self, mono_wav: Path) -> None:
        _, _, meta = load_audio(mono_wav)
        assert meta["load_time_ms"] >= 0.0


# ---------------------------------------------------------------------------
# load_audio — Ausschnitte (offset_sec, duration_sec)
# ---------------------------------------------------------------------------

class TestLoadAudioExcerpt:
    def test_duration_limits_length(self, mono_wav: Path) -> None:
        """duration_sec=0.5 s → ~4000 Samples bei 8 kHz."""
        signal, sr, _ = load_audio(mono_wav, duration_sec=0.5)
        expected = 0.5 * config.SAMPLE_RATE
        assert abs(len(signal) - expected) / expected < 0.02

    def test_offset_shortens_signal(self, mono_wav: Path) -> None:
        """offset_sec=1.0 → Signal kürzer als die volle Datei (2 s)."""
        full, _, _ = load_audio(mono_wav)
        excerpt, _, _ = load_audio(mono_wav, offset_sec=1.0)
        assert len(excerpt) < len(full)

    def test_offset_and_duration(self, mono_wav: Path) -> None:
        signal, sr, _ = load_audio(mono_wav, offset_sec=0.5, duration_sec=0.5)
        expected = 0.5 * config.SAMPLE_RATE
        assert abs(len(signal) - expected) / expected < 0.02


# ---------------------------------------------------------------------------
# load_audio — Fehlerbehandlung
# ---------------------------------------------------------------------------

class TestLoadAudioErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_audio(tmp_path / "nonexistent.wav")

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_text("dummy")
        with pytest.raises(ValueError, match="Nicht unterstütztes Dateiformat"):
            load_audio(p)

    def test_negative_offset_raises(self, mono_wav: Path) -> None:
        with pytest.raises(ValueError, match="offset_sec"):
            load_audio(mono_wav, offset_sec=-1.0)

    def test_zero_duration_raises(self, mono_wav: Path) -> None:
        with pytest.raises(ValueError, match="duration_sec"):
            load_audio(mono_wav, duration_sec=0.0)

    def test_negative_duration_raises(self, mono_wav: Path) -> None:
        with pytest.raises(ValueError, match="duration_sec"):
            load_audio(mono_wav, duration_sec=-1.0)

    def test_path_as_string_works(self, mono_wav: Path) -> None:
        """Auch str-Pfade müssen funktionieren."""
        signal, sr, _ = load_audio(str(mono_wav))
        assert signal.dtype == np.float32
        assert sr == config.SAMPLE_RATE


# ---------------------------------------------------------------------------
# load_audio_excerpt
# ---------------------------------------------------------------------------

class TestLoadAudioExcerptWrapper:
    def test_matches_load_audio(self, mono_wav: Path) -> None:
        """load_audio_excerpt muss identisches Ergebnis liefern wie load_audio."""
        sig1, sr1, _ = load_audio(mono_wav, offset_sec=0.3, duration_sec=0.8)
        sig2, sr2, _ = load_audio_excerpt(mono_wav, offset_sec=0.3, duration_sec=0.8)
        assert sr1 == sr2
        np.testing.assert_array_equal(sig1, sig2)

    def test_returns_tuple_of_three(self, mono_wav: Path) -> None:
        result = load_audio_excerpt(mono_wav, offset_sec=0.0, duration_sec=1.0)
        assert isinstance(result, tuple)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# load_audio_directory
# ---------------------------------------------------------------------------

class TestLoadAudioDirectory:
    def test_yields_correct_count(self, audio_dir: Path) -> None:
        results = list(load_audio_directory(audio_dir))
        assert len(results) == 3

    def test_each_result_is_tuple_of_three(self, audio_dir: Path) -> None:
        for result in load_audio_directory(audio_dir):
            assert isinstance(result, tuple)
            assert len(result) == 3

    def test_all_signals_float32(self, audio_dir: Path) -> None:
        for signal, sr, _ in load_audio_directory(audio_dir):
            assert signal.dtype == np.float32

    def test_all_sr_are_config(self, audio_dir: Path) -> None:
        for _, sr, _ in load_audio_directory(audio_dir):
            assert sr == config.SAMPLE_RATE

    def test_is_generator(self, audio_dir: Path) -> None:
        import types
        gen = load_audio_directory(audio_dir)
        assert isinstance(gen, types.GeneratorType)

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list(load_audio_directory(tmp_path / "nonexistent"))

    def test_not_a_directory_raises(self, mono_wav: Path) -> None:
        with pytest.raises(ValueError):
            list(load_audio_directory(mono_wav))

    def test_empty_directory_yields_nothing(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        results = list(load_audio_directory(empty))
        assert results == []

    def test_recursive_finds_subdirectory_files(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        sub = root / "subdir"
        sub.mkdir(parents=True)
        _write_sine(root / "a.wav", duration=1.0, sr=SR_NATIVE)
        _write_sine(sub / "b.wav", duration=1.0, sr=SR_NATIVE)

        # Nicht-rekursiv: nur root/a.wav
        non_recursive = list(load_audio_directory(root, recursive=False))
        assert len(non_recursive) == 1

        # Rekursiv: root/a.wav + sub/b.wav
        recursive = list(load_audio_directory(root, recursive=True))
        assert len(recursive) == 2

    def test_duration_applied_to_all(self, audio_dir: Path) -> None:
        target_dur = 0.5
        for signal, sr, _ in load_audio_directory(audio_dir, duration_sec=target_dur):
            expected = target_dur * config.SAMPLE_RATE
            assert abs(len(signal) - expected) / expected < 0.02

    def test_ignores_non_audio_files(self, tmp_path: Path) -> None:
        d = tmp_path / "mixed"
        d.mkdir()
        _write_sine(d / "song.wav", duration=1.0, sr=SR_NATIVE)
        (d / "notes.txt").write_text("ignore me")
        (d / "image.png").write_bytes(b"\x89PNG")
        results = list(load_audio_directory(d))
        assert len(results) == 1


# ---------------------------------------------------------------------------
# get_audio_info
# ---------------------------------------------------------------------------

class TestGetAudioInfo:
    EXPECTED_KEYS = {
        "file_path",
        "file_name",
        "original_sr",
        "original_duration_sec",
        "n_channels",
        "file_size_bytes",
        "suffix",
    }

    def test_returns_dict_with_all_keys(self, mono_wav: Path) -> None:
        info = get_audio_info(mono_wav)
        assert self.EXPECTED_KEYS.issubset(set(info.keys()))

    def test_original_sr(self, mono_wav: Path) -> None:
        info = get_audio_info(mono_wav)
        assert info["original_sr"] == SR_NATIVE

    def test_duration_approx(self, mono_wav: Path) -> None:
        info = get_audio_info(mono_wav)
        assert info["original_duration_sec"] == pytest.approx(2.0, abs=0.01)

    def test_mono_channels(self, mono_wav: Path) -> None:
        info = get_audio_info(mono_wav)
        assert info["n_channels"] == 1

    def test_stereo_channels(self, stereo_wav: Path) -> None:
        info = get_audio_info(stereo_wav)
        assert info["n_channels"] == 2

    def test_file_size_positive(self, mono_wav: Path) -> None:
        info = get_audio_info(mono_wav)
        assert info["file_size_bytes"] > 0

    def test_suffix_lowercase(self, mono_wav: Path) -> None:
        info = get_audio_info(mono_wav)
        assert info["suffix"] == ".wav"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            get_audio_info(tmp_path / "nonexistent.wav")

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_text("dummy")
        with pytest.raises(ValueError):
            get_audio_info(p)

    def test_does_not_load_signal(self, mono_wav: Path, monkeypatch) -> None:
        """get_audio_info darf kein librosa.load aufrufen."""
        import librosa
        calls = []
        original_load = librosa.load

        def mock_load(*args, **kwargs):
            calls.append(args)
            return original_load(*args, **kwargs)

        monkeypatch.setattr(librosa, "load", mock_load)
        get_audio_info(mono_wav)
        assert len(calls) == 0, "get_audio_info darf librosa.load nicht aufrufen"


# ---------------------------------------------------------------------------
# SUPPORTED_EXTENSIONS
# ---------------------------------------------------------------------------

class TestSupportedExtensions:
    def test_wav_supported(self) -> None:
        assert ".wav" in SUPPORTED_EXTENSIONS

    def test_mp3_supported(self) -> None:
        assert ".mp3" in SUPPORTED_EXTENSIONS

    def test_flac_supported(self) -> None:
        assert ".flac" in SUPPORTED_EXTENSIONS

    def test_unsupported_not_present(self) -> None:
        assert ".txt" not in SUPPORTED_EXTENSIONS
        assert ".png" not in SUPPORTED_EXTENSIONS
        assert ".pdf" not in SUPPORTED_EXTENSIONS
