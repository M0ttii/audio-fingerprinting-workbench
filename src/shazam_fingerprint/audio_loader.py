"""
Modul zum Laden, Normalisieren und Vorverarbeiten von Audiodateien.

Unterstützte Formate: WAV, MP3, FLAC, OGG.
Alle geladenen Signale werden zu Mono konvertiert, auf SAMPLE_RATE geresampelt
und peak-normalisiert auf [-1.0, 1.0].
"""

import logging
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from shazam_fingerprint import config

logger = logging.getLogger(__name__)

# Unterstützte Dateiendungen
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".wav", ".mp3", ".flac", ".ogg"})

# Typ-Alias für das Rückgabe-Tuple
AudioTuple = tuple[np.ndarray, int, dict]


def load_audio(
    file_path: str | Path,
    start_sec: float | None = None,
    duration_sec: float | None = None,
) -> AudioTuple:
    """Lädt eine einzelne Audiodatei und bereitet sie für das Fingerprinting vor.

    Das Signal wird zu Mono konvertiert, auf config.SAMPLE_RATE geresampelt
    und peak-normalisiert. Optionale Parameter ermöglichen das Laden eines
    Ausschnitts (wichtig für Query-Generierung, vgl. Wang Section 3.1).

    Args:
        file_path: Pfad zur Audiodatei (WAV, MP3, FLAC, OGG).
        start_sec: Startzeit des zu ladenden Ausschnitts in Sekunden.
            None bedeutet Beginn der Datei.
        duration_sec: Länge des zu ladenden Ausschnitts in Sekunden.
            None bedeutet bis zum Ende der Datei (bzw. config.DURATION).

    Returns:
        Tuple aus:
            - signal (np.ndarray): Mono-Signal, float32, normalisiert auf [-1.0, 1.0].
            - sr (int): Abtastrate (immer config.SAMPLE_RATE).
            - metadata (dict): Dateiname, Dauer, Original-Abtastrate, Ausschnitt-Info.

    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert.
        ValueError: Wenn das Dateiformat nicht unterstützt wird oder das Signal
            nach dem Laden leer ist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audiodatei nicht gefunden: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Nicht unterstütztes Format '{file_path.suffix}'. "
            f"Erlaubt: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    effective_duration = duration_sec if duration_sec is not None else config.DURATION

    logger.debug(
        "Lade '%s' | offset=%.2f s | duration=%s s",
        file_path.name,
        start_sec or 0.0,
        effective_duration if effective_duration is not None else "full",
    )

    signal, original_sr = librosa.load(
        path=str(file_path),
        sr=config.SAMPLE_RATE,
        mono=config.MONO,
        offset=start_sec or 0.0,
        duration=effective_duration,
        res_type="kaiser_best",
    )

    if signal.size == 0:
        raise ValueError(f"Signal nach dem Laden leer: {file_path}")

    signal = _peak_normalize(signal)

    duration_loaded = signal.shape[0] / config.SAMPLE_RATE

    metadata: dict = {
        "filename": file_path.name,
        "filepath": str(file_path),
        "duration": duration_loaded,
        "original_sr": original_sr,
        "start_sec": start_sec or 0.0,
        "duration_sec": duration_loaded,
    }

    logger.debug(
        "Geladen: '%s' | %.2f s | %d Samples | orig. SR: %d Hz",
        file_path.name,
        duration_loaded,
        signal.shape[0],
        original_sr,
    )

    return signal, config.SAMPLE_RATE, metadata


def load_query(
    file_path: str | Path,
    start_sec: float | None = None,
) -> AudioTuple:
    """Lädt einen Query-Ausschnitt fester Länge (config.QUERY_DURATION_SEC).

    Komfortfunktion für die Evaluation: Lädt immer genau config.QUERY_DURATION_SEC
    Sekunden, beginnend bei start_sec. Entspricht dem Query-Modell aus
    Wang Section 3.1 ("The query was restricted to 10 seconds.").

    Args:
        file_path: Pfad zur Audiodatei.
        start_sec: Startzeit in Sekunden. None bedeutet Beginn der Datei.

    Returns:
        Tuple (signal, sr, metadata) wie load_audio().

    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert.
        ValueError: Wenn das Dateiformat nicht unterstützt wird.
    """
    return load_audio(
        file_path=file_path,
        start_sec=start_sec,
        duration_sec=config.QUERY_DURATION_SEC,
    )


def load_directory(
    directory: str | Path,
    recursive: bool = False,
    show_progress: bool = True,
) -> list[AudioTuple]:
    """Lädt alle unterstützten Audiodateien aus einem Verzeichnis.

    Durchsucht das Verzeichnis nach allen unterstützten Audioformaten und
    lädt jede Datei vollständig (kein Ausschnitt). Dateien, die nicht geladen
    werden können, werden mit einer Warnung übersprungen.

    Args:
        directory: Pfad zum Verzeichnis.
        recursive: Falls True, werden auch Unterverzeichnisse durchsucht.
        show_progress: Falls True, wird ein tqdm-Fortschrittsbalken angezeigt.

    Returns:
        Liste von (signal, sr, metadata)-Tupeln für alle erfolgreich geladenen
        Dateien, sortiert nach Dateiname.

    Raises:
        FileNotFoundError: Wenn das Verzeichnis nicht existiert.
        ValueError: Wenn das Verzeichnis keine unterstützten Audiodateien enthält.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Verzeichnis nicht gefunden: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Kein Verzeichnis: {directory}")

    glob_pattern = "**/*" if recursive else "*"
    all_files = sorted(directory.glob(glob_pattern))
    audio_files = [f for f in all_files if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not audio_files:
        raise ValueError(
            f"Keine unterstützten Audiodateien in '{directory}'. "
            f"Erlaubt: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    logger.info(
        "Batch-Laden: %d Audiodateien in '%s' gefunden.", len(audio_files), directory
    )

    results: list[AudioTuple] = []
    iterator = tqdm(audio_files, desc="Audio laden", unit="file") if show_progress else audio_files

    for file_path in iterator:
        try:
            result = load_audio(file_path)
            results.append(result)
        except (FileNotFoundError, ValueError, Exception) as exc:
            logger.warning("Überspringe '%s': %s", file_path.name, exc)

    logger.info(
        "Batch-Laden abgeschlossen: %d/%d Dateien erfolgreich geladen.",
        len(results),
        len(audio_files),
    )

    return results


def _peak_normalize(signal: np.ndarray) -> np.ndarray:
    """Normalisiert ein Signal auf den Bereich [-1.0, 1.0] (Peak-Normalisierung).

    Die Peak-Normalisierung skaliert das Signal so, dass der betragsmäßig größte
    Sample-Wert genau 1.0 beträgt. Dadurch sind Lautstärkeunterschiede zwischen
    Aufnahmen kompensiert, was die Robustheit des Fingerprinting gegenüber
    Lautstärke-Variationen verbessert.

    Wang Section 2.1: "the highest amplitude peaks are most likely to survive
    the distortions" — konsequenterweise normalisieren wir auf ein einheitliches
    Amplitudenniveau, damit der Threshold AMPLITUDE_THRESHOLD_DB konsistent wirkt.

    Args:
        signal: Eingabe-Signal als np.ndarray (beliebige Form).

    Returns:
        Normalisiertes Signal als float32-Array. Stilles Signal (max=0) wird
        unverändert zurückgegeben.
    """
    peak = np.max(np.abs(signal))
    if peak < 1e-9:
        logger.debug("Stilles Signal erkannt (peak < 1e-9), Normalisierung übersprungen.")
        return signal.astype(np.float32)
    return (signal / peak).astype(np.float32)


def get_duration(file_path: str | Path) -> float:
    """Gibt die Dauer einer Audiodatei in Sekunden zurück, ohne sie vollständig zu laden.

    Nützlich für Vorfilterung (z.B. zu kurze Tracks überspringen) und
    für die Berechnung sinnvoller Startzeiten bei Query-Ausschnitten.

    Args:
        file_path: Pfad zur Audiodatei.

    Returns:
        Dauer der Audiodatei in Sekunden.

    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audiodatei nicht gefunden: {file_path}")
    return librosa.get_duration(path=str(file_path))
