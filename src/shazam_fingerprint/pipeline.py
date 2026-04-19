"""
End-to-End-Pipeline für das Shazam Audio-Fingerprinting.

Orchestriert den gesamten Datenfluss:
  audio_loader → spectrogram → peak_finder → fingerprint → database → matcher → evaluate

Drei Hauptfunktionen:
  1. ingest_directory: Referenz-Songs laden und in die Datenbank einfügen.
  2. query: Einzelne Audiodatei gegen die Datenbank abfragen.
  3. evaluate_robustness: Batch-Evaluation über ein Query-Verzeichnis.
"""

import logging
import time
from pathlib import Path

from tqdm import tqdm

from shazam_fingerprint import config
from shazam_fingerprint.audio_loader import SUPPORTED_EXTENSIONS, load_audio
from shazam_fingerprint.database import FingerprintDatabase
from shazam_fingerprint.evaluate import EvalReport, EvalResult, compute_metrics
from shazam_fingerprint.fingerprint import generate_fingerprints
from shazam_fingerprint.matcher import MatchResult, match
from shazam_fingerprint.peak_finder import find_peaks
from shazam_fingerprint.spectrogram import compute_spectrogram

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Ingest
# ======================================================================

def ingest_directory(
    audio_dir: str | Path,
    database: FingerprintDatabase,
    recursive: bool = False,
    show_progress: bool = True,
) -> dict:
    """Lädt alle Audiodateien aus einem Verzeichnis und fügt sie in die DB ein.

    Ablauf pro Datei:
      load_audio → compute_spectrogram → find_peaks → generate_fingerprints → db.insert

    Songs, die bereits in der Datenbank vorhanden sind, werden übersprungen.
    Dateien, die beim Verarbeiten Fehler werfen, werden gewarnt und übersprungen.

    Args:
        audio_dir: Pfad zum Verzeichnis mit Referenz-Audiodateien.
        database: FingerprintDatabase, in die eingefügt wird.
        recursive: Falls True, werden Unterverzeichnisse mit durchsucht.
        show_progress: Falls True, wird ein tqdm-Fortschrittsbalken angezeigt.

    Returns:
        Dict mit Ingest-Statistiken:
            processed (int): Anzahl erfolgreich verarbeiteter Songs.
            skipped (int): Bereits in DB vorhandene Songs.
            failed (int): Fehlgeschlagene Dateien.
            total_hashes (int): Gesamtzahl eingefügter Hashes.
            total_time_s (float): Gesamtverarbeitungszeit in Sekunden.

    Raises:
        FileNotFoundError: Wenn audio_dir nicht existiert.
    """
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"Verzeichnis nicht gefunden: {audio_dir}")

    glob_pattern = "**/*" if recursive else "*"
    audio_files = sorted(
        f for f in audio_dir.glob(glob_pattern)
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not audio_files:
        logger.warning("Keine Audiodateien in '%s' gefunden.", audio_dir)
        return {"processed": 0, "skipped": 0, "failed": 0,
                "total_hashes": 0, "total_time_s": 0.0}

    logger.info("Ingest: %d Audiodateien in '%s'", len(audio_files), audio_dir)

    processed = 0
    skipped = 0
    failed = 0
    total_hashes = 0
    t_start = time.perf_counter()

    iterator = (
        tqdm(audio_files, desc="Ingest", unit="song")
        if show_progress
        else audio_files
    )

    for file_path in iterator:
        song_id = file_path.stem

        if song_id in database:
            logger.debug("Überspringe '%s' (bereits in DB).", song_id)
            skipped += 1
            continue

        try:
            signal, sr, meta = load_audio(file_path)
            spec = compute_spectrogram(signal, sr)
            peaks = find_peaks(spec)
            fps = generate_fingerprints(peaks)

            if not fps:
                logger.warning("Keine Fingerprints für '%s' erzeugt.", song_id)
                failed += 1
                continue

            database.insert(song_id, fps)
            processed += 1
            total_hashes += len(fps)

            logger.debug(
                "Ingest '%s': %d Peaks → %d Hashes (%.1f s)",
                song_id, len(peaks), len(fps), meta["duration"],
            )
        except Exception as exc:
            logger.warning("Fehler bei '%s': %s", file_path.name, exc)
            failed += 1

    total_time = time.perf_counter() - t_start

    logger.info(
        "Ingest abgeschlossen: %d verarbeitet, %d übersprungen, %d fehlgeschlagen | "
        "%d Hashes | %.1f s",
        processed, skipped, failed, total_hashes, total_time,
    )

    return {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_hashes": total_hashes,
        "total_time_s": total_time,
    }


# ======================================================================
# 2. Query
# ======================================================================

def query(
    audio_path: str | Path,
    database: FingerprintDatabase,
    start_sec: float | None = None,
    duration_sec: float | None = None,
) -> MatchResult:
    """Führt eine einzelne Abfrage gegen die Datenbank durch.

    Ablauf:
      load_audio → compute_spectrogram → find_peaks → generate_fingerprints → match

    Args:
        audio_path: Pfad zur Query-Audiodatei.
        database: FingerprintDatabase mit indexierten Referenz-Songs.
        start_sec: Optionaler Startpunkt des Ausschnitts in Sekunden.
        duration_sec: Optionale Länge des Ausschnitts in Sekunden.
            Wenn None, wird config.DURATION verwendet (Standard: gesamte Datei).

    Returns:
        MatchResult mit best_match, score und Timing-Informationen.

    Raises:
        FileNotFoundError: Wenn audio_path nicht existiert.
    """
    signal, sr, meta = load_audio(audio_path, start_sec=start_sec,
                                  duration_sec=duration_sec)
    spec = compute_spectrogram(signal, sr)
    peaks = find_peaks(spec)
    fps = generate_fingerprints(peaks)

    result = match(fps, database)

    logger.info(
        "Query '%s': %s (Score %d) | %d Hashes | %.1f ms",
        meta["filename"],
        result.best_match or "KEIN MATCH",
        result.best_score,
        len(fps),
        result.processing_time_ms,
    )

    return result


# ======================================================================
# 3. Evaluation
# ======================================================================

def evaluate_robustness(
    query_dir: str | Path,
    database: FingerprintDatabase,
    ground_truth: dict[str, str] | None = None,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    show_progress: bool = True,
) -> EvalReport:
    """Batch-Evaluation: Alle Queries in einem Verzeichnis gegen die DB testen.

    Für jede Query-Datei wird der vollständige Pipeline-Durchlauf gemessen
    (Fingerprinting + Matching) und ein EvalResult erzeugt. Am Ende werden
    alle Metriken aggregiert.

    Ground-Truth-Zuordnung:
      - Wenn ground_truth übergeben wird: Dict {query_filename → expected_song_id}.
      - Wenn nicht: Der Dateiname (ohne Suffix und optionales Verzerrungssuffix)
        wird als expected_song_id interpretiert. Z.B. "song_a_noise20dB.wav" → "song_a"
        erfordert dann eine Konvention: der Referenz-Song-ID-Anteil steht am Anfang.
        Standard-Fallback: query_stem == expected song_id.

    Args:
        query_dir: Verzeichnis mit Query-Audiodateien.
        database: FingerprintDatabase mit indexierten Referenz-Songs.
        ground_truth: Optionales Dict {query_filename → expected_song_id}.
            Wenn None, wird der Dateiname (stem) als expected_song_id verwendet.
        start_sec: Optionaler Startpunkt des zu ladenden Ausschnitts.
        duration_sec: Optionale Query-Länge. Standard: config.QUERY_DURATION_SEC.
        show_progress: Falls True, wird ein tqdm-Fortschrittsbalken angezeigt.

    Returns:
        EvalReport mit allen aggregierten Robustheits- und Effizienz-Metriken.

    Raises:
        FileNotFoundError: Wenn query_dir nicht existiert.
        ValueError: Wenn keine Query-Dateien gefunden werden oder keine
            erfolgreich verarbeitet werden konnten.
    """
    query_dir = Path(query_dir)
    if not query_dir.exists():
        raise FileNotFoundError(f"Query-Verzeichnis nicht gefunden: {query_dir}")

    query_files = sorted(
        f for f in query_dir.glob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not query_files:
        raise ValueError(f"Keine Audiodateien in '{query_dir}' gefunden.")

    effective_duration = (
        duration_sec if duration_sec is not None else config.QUERY_DURATION_SEC
    )

    logger.info(
        "Evaluation: %d Queries in '%s' | duration=%.1f s",
        len(query_files), query_dir, effective_duration,
    )

    eval_results: list[EvalResult] = []

    iterator = (
        tqdm(query_files, desc="Evaluation", unit="query")
        if show_progress
        else query_files
    )

    for file_path in iterator:
        expected = _resolve_ground_truth(file_path, ground_truth)

        try:
            # Fingerprinting messen
            t0 = time.perf_counter()
            signal, sr, meta = load_audio(file_path, start_sec=start_sec,
                                          duration_sec=effective_duration)
            spec = compute_spectrogram(signal, sr)
            peaks = find_peaks(spec)
            fps = generate_fingerprints(peaks)
            fp_time_ms = (time.perf_counter() - t0) * 1000.0

            # Matching (misst sich intern selbst)
            result = match(fps, database)

            eval_results.append(EvalResult(
                query_file=file_path.name,
                expected_match=expected,
                predicted_match=result.best_match,
                score=result.best_score,
                fingerprint_time_ms=fp_time_ms,
                query_time_ms=result.processing_time_ms,
                num_query_hashes=len(fps),
            ))

        except Exception as exc:
            logger.warning("Evaluation-Fehler bei '%s': %s", file_path.name, exc)

    if not eval_results:
        raise ValueError("Keine Queries konnten erfolgreich verarbeitet werden.")

    report = compute_metrics(eval_results, database=database)

    logger.info(
        "Evaluation abgeschlossen: RR=%.1f%% | FNR=%.1f%% | %d/%d Queries",
        report.recognition_rate * 100,
        report.false_negative_rate * 100,
        report.correct_count,
        report.total_queries,
    )

    return report


def _resolve_ground_truth(
    file_path: Path,
    ground_truth: dict[str, str] | None,
) -> str:
    """Ermittelt die erwartete song_id für eine Query-Datei.

    Args:
        file_path: Pfad zur Query-Datei.
        ground_truth: Optionales Mapping {filename → expected_song_id}.

    Returns:
        Erwartete song_id. Wenn kein ground_truth-Dict vorhanden, wird der
        Dateiname (stem) als song_id verwendet.
    """
    if ground_truth is not None:
        return ground_truth.get(file_path.name, file_path.stem)
    return file_path.stem
