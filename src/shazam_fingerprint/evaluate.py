"""
Modul zur Berechnung von Evaluationsmetriken für das Shazam-Fingerprinting.

Zwei Evaluationsdimensionen gemäß Bachelorarbeit-Anforderungen:
- Robustheit: recognition_rate, false_positive_rate, false_negative_rate
- Effizienz: avg_fingerprint_time, avg_query_time, db_memory_usage, hashes_per_second

Eingabe ist eine Liste von EvalResult-Objekten, die pro Query gesammelt werden.
Ausgabe ist ein EvalReport-Objekt, exportierbar als JSON oder CSV.
"""

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shazam_fingerprint.database import FingerprintDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Ergebnis einer einzelnen Query-Auswertung.

    Attributes:
        query_file: Dateiname der Query-Audiodatei.
        expected_match: Ground-Truth song_id (welcher Song sollte erkannt werden).
        predicted_match: Vom Matcher zurückgegebene song_id, oder None.
        score: Histogram-Peak-Score des besten Treffers.
        fingerprint_time_ms: Zeit für Fingerprint-Generierung der Query in ms.
        query_time_ms: Zeit für den Matching-Vorgang in ms.
        num_query_hashes: Anzahl der erzeugten Query-Hashes.
        is_correct: True wenn predicted_match == expected_match.
    """

    query_file: str
    expected_match: str
    predicted_match: str | None
    score: int
    fingerprint_time_ms: float
    query_time_ms: float
    num_query_hashes: int = 0
    is_correct: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_correct = self.predicted_match == self.expected_match


@dataclass
class EvalReport:
    """Zusammenfassung aller Evaluationsmetriken über eine Query-Menge.

    Robustheit (Wang Section 3.1 — Metriken für Audio-Suchsysteme):
        recognition_rate: Anteil korrekt erkannter Queries (True Positives / Gesamt).
        false_positive_rate: Anteil falsch positiver Erkennungen (Kein Match vorhanden,
            aber Match gemeldet — relevant bei Queries ohne Datenbankentsprechung).
        false_negative_rate: Anteil nicht erkannter Queries (Song in DB, aber kein Match).

    Effizienz:
        avg_fingerprint_time_ms: Durchschnittliche Fingerprint-Generierungszeit pro Query.
        avg_query_time_ms: Durchschnittliche Matching-Zeit pro Query.
        db_memory_mb: Speicherbedarf der Datenbank in MB (aus database.get_stats()).
        hashes_per_second: Durchsatz der Fingerprint-Generierung.
        total_queries: Gesamtzahl ausgewerteter Queries.
    """

    recognition_rate: float
    false_positive_rate: float
    false_negative_rate: float
    avg_fingerprint_time_ms: float
    avg_query_time_ms: float
    db_memory_mb: float
    hashes_per_second: float
    total_queries: int
    correct_count: int
    false_positive_count: int
    false_negative_count: int


def compute_metrics(
    results: list[EvalResult],
    database: FingerprintDatabase | None = None,
) -> EvalReport:
    """Berechnet alle Evaluationsmetriken aus einer Liste von Query-Ergebnissen.

    Begriffsdefinitionen für diesen Kontext:
    - True Positive (TP): expected_match in DB, predicted_match == expected_match.
    - False Negative (FN): expected_match in DB, predicted_match != expected_match
      (Song vorhanden aber nicht erkannt — Robustheitsproblem).
    - False Positive (FP): expected_match == None (kein Match erwartet),
      aber predicted_match ist gesetzt (Falscherkennung).

    Args:
        results: Liste von EvalResult-Objekten, ein Eintrag pro Query.
        database: Optionale FingerprintDatabase für Speicherstatistiken.
            None führt zu db_memory_mb=0.0.

    Returns:
        EvalReport mit allen berechneten Metriken.

    Raises:
        ValueError: Wenn results leer ist.
    """
    if not results:
        raise ValueError("Keine EvalResult-Objekte übergeben.")

    n = len(results)

    # Robustheit-Metriken
    # Queries, bei denen ein Match erwartet wird (Song ist in DB)
    in_db = [r for r in results if r.expected_match is not None]
    # Queries, bei denen kein Match erwartet wird (Song nicht in DB)
    not_in_db = [r for r in results if r.expected_match is None]

    correct = sum(1 for r in in_db if r.is_correct)
    false_negatives = sum(1 for r in in_db if not r.is_correct)
    false_positives = sum(1 for r in not_in_db if r.predicted_match is not None)

    n_in_db = len(in_db)
    n_not_in_db = len(not_in_db)

    recognition_rate = correct / n_in_db if n_in_db > 0 else 0.0
    false_negative_rate = false_negatives / n_in_db if n_in_db > 0 else 0.0
    false_positive_rate = false_positives / n_not_in_db if n_not_in_db > 0 else 0.0

    # Effizienz-Metriken
    avg_fp_time = sum(r.fingerprint_time_ms for r in results) / n
    avg_q_time = sum(r.query_time_ms for r in results) / n

    total_hashes = sum(r.num_query_hashes for r in results)
    total_fp_time_s = sum(r.fingerprint_time_ms for r in results) / 1000.0
    hashes_per_sec = total_hashes / total_fp_time_s if total_fp_time_s > 0 else 0.0

    db_memory_mb = 0.0
    if database is not None:
        db_memory_mb = database.get_stats()["memory_mb"]

    report = EvalReport(
        recognition_rate=recognition_rate,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        avg_fingerprint_time_ms=avg_fp_time,
        avg_query_time_ms=avg_q_time,
        db_memory_mb=db_memory_mb,
        hashes_per_second=hashes_per_sec,
        total_queries=n,
        correct_count=correct,
        false_positive_count=false_positives,
        false_negative_count=false_negatives,
    )

    logger.info(
        "Evaluation: %d Queries | RR=%.1f%% | FNR=%.1f%% | FPR=%.1f%% | "
        "FP-Zeit=%.1f ms | Q-Zeit=%.1f ms | %.0f H/s",
        n,
        recognition_rate * 100,
        false_negative_rate * 100,
        false_positive_rate * 100,
        avg_fp_time,
        avg_q_time,
        hashes_per_sec,
    )

    return report


def export_json(
    report: EvalReport,
    results: list[EvalResult],
    path: str | Path,
) -> None:
    """Exportiert Report und Einzel-Ergebnisse als JSON-Datei.

    Args:
        report: Aggregierter EvalReport.
        results: Liste der EvalResult-Objekte für Detailansicht.
        path: Zieldatei (.json).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": asdict(report),
        "results": [asdict(r) for r in results],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("JSON-Export: '%s' (%d Queries)", path, len(results))


def export_csv(results: list[EvalResult], path: str | Path) -> None:
    """Exportiert die Einzel-Ergebnisse als CSV-Datei.

    Jede Zeile entspricht einem EvalResult. Nützlich für Tabellenauswertung
    in Excel oder pandas für die Bachelorarbeit.

    Args:
        results: Liste der EvalResult-Objekte.
        path: Zieldatei (.csv).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        logger.warning("Keine Ergebnisse für CSV-Export.")
        return

    fieldnames = list(asdict(results[0]).keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    logger.info("CSV-Export: '%s' (%d Zeilen)", path, len(results))
