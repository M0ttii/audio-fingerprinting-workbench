"""
Modul zum Laden und Vorverarbeiten von Audiodateien.

Unterstützte Formate: WAV, MP3, FLAC, OGG (via librosa/soundfile).
Die Vorverarbeitung umfasst:
  - Konvertierung zu Mono (Kanal-Mittelung)
  - Resampling auf SAMPLE_RATE (8000 Hz)
  - Peak-Normalisierung auf [-1.0, 1.0]

Rückgabe immer als Tuple (signal, sr, metadata).
"""

import logging
import pathlib
import time
from typing import Generator

import librosa
import numpy as np
import soundfile as sf

from quad_fingerprint import config

logger = logging.getLogger(__name__)

# Unterstützte Dateiendungen
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
)


def load_audio(
    file_path: str | pathlib.Path,
    offset_sec: float = 0.0,
    duration_sec: float | None = None,
) -> tuple[np.ndarray, int, dict]:
    """Lädt eine Audiodatei, konvertiert zu Mono, resampled auf SAMPLE_RATE
    und normalisiert auf Peak-Amplitude 1.0.

    Args:
        file_path: Pfad zur Audiodatei (WAV, MP3, FLAC, OGG, M4A, AAC).
        offset_sec: Startposition des Ausschnitts in Sekunden (Standard: 0.0).
        duration_sec: Länge des zu ladenden Ausschnitts in Sekunden.
            Bei None wird die gesamte Datei (ab offset_sec) geladen.

    Returns:
        Tuple aus:
            - signal: float32-Array der Form (N,), normalisiert auf [-1.0, 1.0]
            - sr: Abtastrate in Hz (immer config.SAMPLE_RATE = 8000)
            - metadata: Dict mit Datei-Metadaten (siehe unten)

    Raises:
        FileNotFoundError: Wenn file_path nicht existiert.
        ValueError: Wenn das Dateiformat nicht unterstützt wird oder
            offset_sec/duration_sec ungültig sind.

    Metadaten-Keys:
        file_path (str): Absoluter Pfad zur Datei.
        file_name (str): Dateiname ohne Verzeichnis.
        duration_sec (float): Länge des geladenen Ausschnitts in Sekunden.
        original_sr (int): Original-Abtastrate der Datei.
        original_duration_sec (float): Gesamtdauer der Originaldatei in Sekunden.
        load_time_ms (float): Ladezeit in Millisekunden.
        offset_sec (float): Verwendeter Startoffset.
        peak_amplitude (float): Peak-Amplitude vor der Normalisierung.
        n_channels (int): Anzahl der Kanäle im Original.
    """
    file_path = pathlib.Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Audiodatei nicht gefunden: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Nicht unterstütztes Dateiformat '{file_path.suffix}'. "
            f"Unterstützt: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if offset_sec < 0.0:
        raise ValueError(f"offset_sec muss >= 0 sein, erhalten: {offset_sec}")
    if duration_sec is not None and duration_sec <= 0.0:
        raise ValueError(f"duration_sec muss > 0 sein, erhalten: {duration_sec}")

    t_start = time.perf_counter()

    # Original-Metadaten vor dem Laden auslesen (effizient, ohne Dekodierung)
    try:
        info = sf.info(str(file_path))
        original_sr = info.samplerate
        original_duration_sec = info.duration
        n_channels = info.channels
    except Exception:
        # soundfile unterstützt MP3 nicht direkt — Fallback via librosa
        original_sr = None
        original_duration_sec = None
        n_channels = None

    # Laden mit librosa:
    # - mono=True: Kanal-Mittelung
    # - sr=config.SAMPLE_RATE: direktes Resampling beim Laden
    # - res_type='soxr_hq': hohe Qualität via soxr (schneller als kaiser_best/resampy)
    signal, sr = librosa.load(
        str(file_path),
        sr=config.SAMPLE_RATE,
        mono=True,
        offset=offset_sec,
        duration=duration_sec,
        res_type="soxr_hq",
        dtype=np.float32,
    )

    # Original-SR aus librosa, falls soundfile es nicht lesen konnte
    if original_sr is None:
        # librosa gibt immer die Ziel-SR zurück; echten Wert über soundfile
        # nachladen — bei reinen MP3 nicht verfügbar, daher Schätzung
        original_sr = sr  # Fallback
    if original_duration_sec is None:
        original_duration_sec = len(signal) / sr
    if n_channels is None:
        n_channels = 1  # nach mono=True ohnehin 1

    # Peak-Normalisierung
    peak_amplitude = float(np.max(np.abs(signal)))
    if peak_amplitude > 0.0:
        signal = signal / peak_amplitude
    else:
        logger.warning("Stilles Audio geladen (Peak-Amplitude = 0): %s", file_path)

    load_time_ms = (time.perf_counter() - t_start) * 1000.0

    metadata: dict = {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "duration_sec": len(signal) / sr,
        "original_sr": original_sr,
        "original_duration_sec": original_duration_sec,
        "load_time_ms": round(load_time_ms, 2),
        "offset_sec": offset_sec,
        "peak_amplitude": round(peak_amplitude, 6),
        "n_channels": n_channels,
    }

    logger.debug(
        "Geladen: %s | %.2f s | SR %d→%d | %.1f ms",
        file_path.name,
        metadata["duration_sec"],
        original_sr,
        sr,
        load_time_ms,
    )

    return signal, sr, metadata


def load_audio_excerpt(
    file_path: str | pathlib.Path,
    offset_sec: float,
    duration_sec: float,
) -> tuple[np.ndarray, int, dict]:
    """Lädt einen definierten Ausschnitt aus einer Audiodatei.

    Convenience-Wrapper um load_audio() mit explizit gesetztem offset und
    duration — nützlich für Query-Snippets fixer Länge (z.B. 20 s).

    Args:
        file_path: Pfad zur Audiodatei.
        offset_sec: Startposition des Ausschnitts in Sekunden (>= 0).
        duration_sec: Länge des Ausschnitts in Sekunden (> 0).

    Returns:
        Tuple (signal, sr, metadata) wie load_audio().
    """
    return load_audio(file_path, offset_sec=offset_sec, duration_sec=duration_sec)


def load_audio_directory(
    directory: str | pathlib.Path,
    recursive: bool = False,
    offset_sec: float = 0.0,
    duration_sec: float | None = None,
) -> Generator[tuple[np.ndarray, int, dict], None, None]:
    """Lädt alle unterstützten Audiodateien aus einem Verzeichnis.

    Generator: Gibt jeweils ein (signal, sr, metadata)-Tuple zurück, ohne
    alle Dateien gleichzeitig im Speicher zu halten.

    Args:
        directory: Pfad zum Verzeichnis.
        recursive: Bei True werden Unterverzeichnisse rekursiv durchsucht.
        offset_sec: Startposition für alle Dateien in Sekunden.
        duration_sec: Länge des Ausschnitts für alle Dateien in Sekunden.
            Bei None wird jeweils die gesamte Datei geladen.

    Yields:
        Tuple (signal, sr, metadata) wie load_audio().

    Raises:
        FileNotFoundError: Wenn directory nicht existiert.
        ValueError: Wenn directory kein Verzeichnis ist.
    """
    directory = pathlib.Path(directory).resolve()

    if not directory.exists():
        raise FileNotFoundError(f"Verzeichnis nicht gefunden: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Pfad ist kein Verzeichnis: {directory}")

    pattern = "**/*" if recursive else "*"
    all_files = sorted(
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not all_files:
        logger.warning("Keine unterstützten Audiodateien in: %s", directory)
        return

    logger.info(
        "Batch-Laden: %d Datei(en) aus %s%s",
        len(all_files),
        directory,
        " (rekursiv)" if recursive else "",
    )

    for file_path in all_files:
        try:
            yield load_audio(file_path, offset_sec=offset_sec, duration_sec=duration_sec)
        except Exception as exc:  # noqa: BLE001
            logger.error("Fehler beim Laden von %s: %s", file_path.name, exc)
            # Datei überspringen, Batch-Verarbeitung fortsetzen


def get_audio_info(file_path: str | pathlib.Path) -> dict:
    """Liest Metadaten einer Audiodatei, ohne das Signal zu dekodieren.

    Args:
        file_path: Pfad zur Audiodatei.

    Returns:
        Dict mit: file_path, file_name, original_sr, original_duration_sec,
        n_channels, file_size_bytes, suffix.

    Raises:
        FileNotFoundError: Wenn file_path nicht existiert.
        ValueError: Wenn das Format nicht unterstützt wird.
    """
    file_path = pathlib.Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Audiodatei nicht gefunden: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Nicht unterstütztes Dateiformat: {file_path.suffix}")

    try:
        info = sf.info(str(file_path))
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "original_sr": info.samplerate,
            "original_duration_sec": info.duration,
            "n_channels": info.channels,
            "file_size_bytes": file_path.stat().st_size,
            "suffix": file_path.suffix.lower(),
        }
    except Exception as exc:
        # Fallback für Formate, die soundfile nicht direkt liest (z.B. MP3)
        logger.debug("soundfile-Info fehlgeschlagen für %s: %s — nutze librosa", file_path.name, exc)
        duration = librosa.get_duration(path=str(file_path))
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "original_sr": None,
            "original_duration_sec": duration,
            "n_channels": None,
            "file_size_bytes": file_path.stat().st_size,
            "suffix": file_path.suffix.lower(),
        }
