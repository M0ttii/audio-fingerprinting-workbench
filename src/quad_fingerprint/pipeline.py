"""
End-to-End-Pipeline für das Quad-basierte Audio-Fingerprinting.

Orchestriert den gesamten Datenfluss:
  audio_loader → spectrogram → peak_finder → quad_builder → database → matcher → evaluate

Drei Hauptfunktionen:
  1. ingest_directory: Referenz-Songs laden und in die Datenbank einfügen.
     Searchtree wird EINMALIG am Ende gebaut, nicht nach jedem Song.
  2. query: Einzelne Audiodatei gegen die Datenbank abfragen.
     Verwendet Query-Parametersätze (kleinere Filter, erweiterte Fenster).
  3. evaluate_robustness: Batch-Evaluation über ein Query-Verzeichnis.
     Extrahiert distortion_type und distortion_level aus der Verzeichnisstruktur.

Referenz: Sonnleitner & Widmer (2016), "Robust Quad-Based Audio Fingerprinting".
"""

import logging
import re
import time
from pathlib import Path

from tqdm import tqdm

from quad_fingerprint import config
from quad_fingerprint.audio_loader import SUPPORTED_EXTENSIONS, load_audio
from quad_fingerprint.database import ReferenceDatabase
from quad_fingerprint.evaluate import (
    EvalReport,
    EvalResult,
    compute_metrics,
    recognition_by_scale_factor,
)
from quad_fingerprint.matcher import MatchResult, identify
from quad_fingerprint.peak_finder import extract_query_peaks, extract_reference_peaks
from quad_fingerprint.quad_builder import build_query_quads, build_reference_quads
from quad_fingerprint.spectrogram import compute_spectrogram

logger = logging.getLogger(__name__)

# Regex für Verzeichnisnamen wie "tempo_80", "pitch_120", "speed_90"
_DISTORTION_DIR_RE = re.compile(
    r"^(?P<type>tempo|pitch|speed|noise)_(?P<level>\d+)$"
)

# Regex für Dateinamen wie "song1_tempo_80.wav" → Ground-Truth "song1"
_DISTORTION_FILE_RE = re.compile(
    r"^(?P<song>.+?)_(?:tempo|pitch|speed|noise)_\d+$"
)


# ======================================================================
# 1. Ingest
# ======================================================================

def ingest_directory(
    audio_dir: str | Path,
    database: ReferenceDatabase,
    recursive: bool = False,
    show_progress: bool = True,
) -> dict:
    """Lädt alle Audiodateien aus einem Verzeichnis und fügt sie in die DB ein.

    Ablauf pro Datei:
      load_audio → compute_spectrogram → extract_reference_peaks
      → build_reference_quads → database.add_file

    Der Searchtree wird EINMALIG am Ende via database.finalize() gebaut,
    nicht nach jeder einzelnen Datei — bei großen Datensätzen ist das
    deutlich effizienter (Section V, cKDTree über alle Hashes).

    Args:
        audio_dir: Pfad zum Verzeichnis mit Referenz-Audiodateien.
        database: ReferenceDatabase, in die eingefügt wird.
        recursive: Falls True, werden Unterverzeichnisse mit durchsucht.
        show_progress: Falls True, wird ein tqdm-Fortschrittsbalken angezeigt.

    Returns:
        Dict mit Ingest-Statistiken:
            processed (int): Anzahl erfolgreich verarbeiteter Songs.
            skipped (int): Bereits in DB vorhandene Songs.
            failed (int): Fehlgeschlagene Dateien.
            total_quads (int): Gesamtzahl eingefügter Quad-Records.
            total_peaks (int): Gesamtzahl extrahierter Peaks.
            total_time_s (float): Gesamtverarbeitungszeit in Sekunden.
            db_memory_mb (float): Speicherbedarf der finalisierten DB.

    Raises:
        FileNotFoundError: Wenn audio_dir nicht existiert.
    """
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"Verzeichnis nicht gefunden: {audio_dir}")

    glob_pattern = "**/*" if recursive else "*"
    audio_files = sorted(
        f for f in audio_dir.glob(glob_pattern)
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not audio_files:
        logger.warning("Keine Audiodateien in '%s' gefunden.", audio_dir)
        return {
            "processed": 0, "skipped": 0, "failed": 0,
            "total_quads": 0, "total_peaks": 0,
            "total_time_s": 0.0, "db_memory_mb": 0.0,
        }

    logger.info("Ingest: %d Audiodateien in '%s'", len(audio_files), audio_dir)

    processed = 0
    skipped = 0
    failed = 0
    total_quads = 0
    total_peaks = 0
    t_start = time.perf_counter()

    # Existierende Dateinamen prüfen, um Duplikate zu überspringen
    existing_names: set[str] = set()
    for fid in range(database.n_files):
        try:
            existing_names.add(database.get_file_name(fid))
        except KeyError:
            pass

    iterator = (
        tqdm(audio_files, desc="Ingest", unit="song")
        if show_progress
        else audio_files
    )

    for file_path in iterator:
        file_name = file_path.name

        if file_name in existing_names:
            logger.debug("Überspringe '%s' (bereits in DB).", file_name)
            skipped += 1
            continue

        try:
            signal, sr, meta = load_audio(file_path)
            spec = compute_spectrogram(signal, sr)
            peaks = extract_reference_peaks(spec.magnitude)

            if len(peaks) < 4:
                logger.warning(
                    "Zu wenige Peaks für '%s' (%d) — übersprungen.",
                    file_name, len(peaks),
                )
                failed += 1
                continue

            # file_id wird von add_file vergeben
            file_id = database.n_files
            quads = build_reference_quads(peaks, spec.magnitude, file_id)

            if not quads:
                logger.warning("Keine Quads für '%s' erzeugt.", file_name)
                failed += 1
                continue

            database.add_file(
                file_name=file_name,
                peaks=peaks,
                quad_records=quads,
                duration_sec=meta.get("duration", 0.0),
            )

            processed += 1
            total_quads += len(quads)
            total_peaks += len(peaks)

            logger.debug(
                "Ingest '%s': %d Peaks → %d Quads (%.1f s)",
                file_name, len(peaks), len(quads),
                meta.get("duration", 0.0),
            )
        except Exception as exc:
            logger.warning("Fehler bei '%s': %s", file_path.name, exc)
            failed += 1

    # Searchtree EINMALIG bauen (Section V: cKDTree über alle Hashes)
    if processed > 0:
        logger.info("Baue Searchtree über %d Quad-Records ...", database.n_records)
        database.finalize()

    total_time = time.perf_counter() - t_start

    db_mem = database.memory_usage_mb()["total_mb"] if database.is_finalized else 0.0

    logger.info(
        "Ingest abgeschlossen: %d verarbeitet, %d übersprungen, %d fehlgeschlagen | "
        "%d Peaks, %d Quads | DB %.2f MB | %.1f s",
        processed, skipped, failed, total_peaks, total_quads, db_mem, total_time,
    )

    return {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_quads": total_quads,
        "total_peaks": total_peaks,
        "total_time_s": total_time,
        "db_memory_mb": db_mem,
    }


# ======================================================================
# 2. Query
# ======================================================================

def query(
    audio_path: str | Path,
    database: ReferenceDatabase,
    start_sec: float | None = None,
    duration_sec: float | None = None,
) -> MatchResult:
    """Führt eine einzelne Abfrage gegen die Datenbank durch.

    Ablauf (mit Query-Parametersätzen):
      load_audio → compute_spectrogram → extract_query_peaks
      → build_query_quads → identify

    WICHTIG: Verwendet Query-Filtergrößen (config.QUERY_MAX_FILTER_*)
    und erweiterte Quad-Fenster (Eq. 4a–4c), nicht die Referenz-Parameter.

    Args:
        audio_path: Pfad zur Query-Audiodatei.
        database: Finalisierte ReferenceDatabase mit indexierten Referenz-Songs.
        start_sec: Optionaler Startpunkt des Ausschnitts in Sekunden.
        duration_sec: Optionale Länge des Ausschnitts in Sekunden.
            Wenn None, wird config.DEFAULT_QUERY_DURATION_SEC verwendet.

    Returns:
        MatchResult mit best_match, best_score, best_scale_factors und
        Timing-Informationen.

    Raises:
        FileNotFoundError: Wenn audio_path nicht existiert.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audiodatei nicht gefunden: {audio_path}")

    effective_duration = (
        duration_sec if duration_sec is not None
        else config.DEFAULT_QUERY_DURATION_SEC
    )

    # Audio laden (ggf. nur Ausschnitt)
    offset = start_sec if start_sec is not None else 0.0
    signal, sr, meta = load_audio(
        audio_path, offset_sec=offset, duration_sec=effective_duration,
    )
    query_duration_sec = len(signal) / sr

    # Spektrogramm → Peaks (Query-Filter!) → Query-Quads
    spec = compute_spectrogram(signal, sr)
    peaks = extract_query_peaks(spec.magnitude)
    quads = build_query_quads(peaks, spec.magnitude)

    # Matching: Kandidatenauswahl → Sequenzschätzung → Verifikation
    result = identify(quads, peaks, database, query_duration_sec)

    logger.info(
        "Query '%s': %s (Score %.3f) | %d Peaks → %d Quads | %.1f ms",
        meta.get("file_name", audio_path.name),
        result.best_match or "KEIN MATCH",
        result.best_score,
        len(peaks),
        len(quads),
        result.processing_time_ms,
    )

    return result


# ======================================================================
# 3. Evaluation
# ======================================================================

def evaluate_robustness(
    query_dir: str | Path,
    database: ReferenceDatabase,
    ground_truth: dict[str, str] | None = None,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    show_progress: bool = True,
) -> tuple[EvalReport, list[EvalResult], dict[str, dict[float, float]]]:
    """Batch-Evaluation: Alle Queries in einem Verzeichnis gegen die DB testen.

    Für jede Query-Datei wird der vollständige Pipeline-Durchlauf gemessen
    (Fingerprinting + Matching) und ein EvalResult erzeugt. Am Ende werden
    alle Metriken aggregiert, inklusive Scale-Robustheit.

    Distortion-Erkennung aus Verzeichnisstruktur:
        queries/tempo_80/song1_tempo_80.wav → distortion_type="tempo", level=0.80
        queries/pitch_120/song1_pitch_120.wav → distortion_type="pitch", level=1.20
        queries/speed_90/song1_speed_90.wav → distortion_type="speed", level=0.90
        queries/noise_20/song1_noise_20.wav → distortion_type="noise", level=20.0

    Ground-Truth-Zuordnung:
        - Wenn ground_truth übergeben: Dict {query_filename → expected_song_id}.
        - Sonst: Aus Dateiname extrahiert. "song1_tempo_80.wav" → "song1".
          Falls kein Distortion-Suffix erkannt wird: stem = expected_song_id.

    Args:
        query_dir: Verzeichnis mit Query-Audiodateien (ggf. mit Unterverzeichnissen
            für verschiedene Distortion-Typen).
        database: Finalisierte ReferenceDatabase.
        ground_truth: Optionales Dict {query_filename → expected_song_id}.
        start_sec: Optionaler Startpunkt des zu ladenden Ausschnitts.
        duration_sec: Optionale Query-Länge in Sekunden.
            Standard: config.DEFAULT_QUERY_DURATION_SEC.
        show_progress: Falls True, wird ein tqdm-Fortschrittsbalken angezeigt.

    Returns:
        Tuple aus:
            - EvalReport mit allen aggregierten Metriken.
            - Liste aller EvalResult-Objekte (für Export/Detailanalyse).
            - Scale-Robustheit-Dict von recognition_by_scale_factor()
              (für Fig. 5 Reproduktion).

    Raises:
        FileNotFoundError: Wenn query_dir nicht existiert.
        ValueError: Wenn keine Query-Dateien gefunden oder verarbeitet werden.
    """
    query_dir = Path(query_dir)
    if not query_dir.exists():
        raise FileNotFoundError(f"Query-Verzeichnis nicht gefunden: {query_dir}")

    # Sammle Query-Dateien rekursiv (Unterverzeichnisse = Distortion-Typen)
    query_files = sorted(
        f for f in query_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not query_files:
        raise ValueError(f"Keine Audiodateien in '{query_dir}' gefunden.")

    effective_duration = (
        duration_sec if duration_sec is not None
        else config.DEFAULT_QUERY_DURATION_SEC
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
        # Ground-Truth und Distortion-Info extrahieren
        expected = _resolve_ground_truth(file_path, ground_truth)
        dist_type, dist_level = _parse_distortion(file_path, query_dir)

        try:
            # Fingerprinting-Zeit messen
            t0 = time.perf_counter()
            offset = start_sec if start_sec is not None else 0.0
            signal, sr, meta = load_audio(
                file_path, offset_sec=offset, duration_sec=effective_duration,
            )
            spec = compute_spectrogram(signal, sr)
            peaks = extract_query_peaks(spec.magnitude)
            quads = build_query_quads(peaks, spec.magnitude)
            fp_time_ms = (time.perf_counter() - t0) * 1000.0

            query_duration_sec = len(signal) / sr

            # Matching (misst sich intern über processing_time_ms)
            result = identify(quads, peaks, database, query_duration_sec)

            er = EvalResult(
                query_file=file_path.name,
                expected_match=expected,
                predicted_match=result.best_match,
                score=result.best_score,
                fingerprint_time_ms=fp_time_ms,
                query_time_ms=result.processing_time_ms,
                num_query_quads=len(quads),
                scale_factors=result.best_scale_factors,
                distortion_type=dist_type,
                distortion_level=dist_level,
            )
            eval_results.append(er)

        except Exception as exc:
            logger.warning("Evaluation-Fehler bei '%s': %s", file_path.name, exc)

    if not eval_results:
        raise ValueError("Keine Queries konnten erfolgreich verarbeitet werden.")

    report = compute_metrics(eval_results, database=database)
    scale_rates = recognition_by_scale_factor(eval_results)

    logger.info(
        "Evaluation abgeschlossen: RR=%.1f%% | FNR=%.1f%% | Spec=%.1f%% | "
        "%d/%d Queries",
        report.recognition_rate * 100,
        report.false_negative_rate * 100,
        report.specificity * 100,
        report.correct_count,
        report.total_queries,
    )

    return report, eval_results, scale_rates


# ======================================================================
# Hilfsfunktionen
# ======================================================================

def _resolve_ground_truth(
    file_path: Path,
    ground_truth: dict[str, str] | None,
) -> str | None:
    """Ermittelt die erwartete song_id für eine Query-Datei.

    Strategie:
      1. Wenn ground_truth-Dict vorhanden und Dateiname enthalten → verwende es.
      2. Sonst: Entferne Distortion-Suffix aus dem Stem.
         "song1_tempo_80" → "song1"
      3. Fallback: Stem direkt als song_id.

    Args:
        file_path: Pfad zur Query-Datei.
        ground_truth: Optionales Mapping {filename → expected_song_id}.

    Returns:
        Erwartete song_id oder None (falls als nicht-in-DB markiert).
    """
    if ground_truth is not None:
        return ground_truth.get(file_path.name)

    stem = file_path.stem
    m = _DISTORTION_FILE_RE.match(stem)
    if m:
        return m.group("song")
    return stem


def _parse_distortion(
    file_path: Path,
    query_dir: Path,
) -> tuple[str | None, float | None]:
    """Extrahiert Distortion-Typ und -Level aus der Verzeichnisstruktur.

    Erwartet Verzeichnisnamen wie "tempo_80", "pitch_120", "speed_90".
    Das Level wird als Skalierungsfaktor (0.0–2.0) für tempo/pitch/speed
    interpretiert (80 → 0.80), bei noise direkt als Wert (20 → 20.0).

    Args:
        file_path: Pfad zur Query-Datei.
        query_dir: Basis-Query-Verzeichnis (um relative Position zu bestimmen).

    Returns:
        Tuple (distortion_type, distortion_level) oder (None, None).
    """
    try:
        rel = file_path.relative_to(query_dir)
    except ValueError:
        return None, None

    # Prüfe Verzeichnis direkt über der Datei
    if len(rel.parts) < 2:
        return None, None

    parent_name = rel.parts[0]
    m = _DISTORTION_DIR_RE.match(parent_name)
    if not m:
        return None, None

    dist_type = m.group("type")
    raw_level = int(m.group("level"))

    # tempo/pitch/speed: 80 → 0.80 (Skalierungsfaktor)
    # noise: 20 → 20.0 (direkt, z.B. SNR in dB)
    if dist_type in ("tempo", "pitch", "speed"):
        dist_level = raw_level / 100.0
    else:
        dist_level = float(raw_level)

    return dist_type, dist_level
