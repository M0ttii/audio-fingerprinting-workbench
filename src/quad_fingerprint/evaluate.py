"""
Modul zur Berechnung von Evaluationsmetriken für das Quad-Fingerprinting.

Zwei Evaluationsdimensionen gemäß Bachelorarbeit-Anforderungen:
- Robustheit: recognition_rate, false_positive_rate, false_negative_rate, specificity
- Effizienz: avg_fingerprint_time, avg_query_time, db_memory_usage, quads_per_second

Zusätzlich quad-spezifisch:
- recognition_by_scale_factor: Erkennungsrate nach Verzerrungsgrad (70%–130%)
  Ermöglicht Reproduktion von Paper Fig. 5 und direkten Vergleich mit Shazam.

Schnittstelle ist IDENTISCH zum Shazam-Modul (shazam_fingerprint/evaluate.py),
um direkten Vergleich zu ermöglichen. Erweitert um quad-spezifische Felder.

Referenz: Sonnleitner & Widmer (2016), Section VIII, Eq. (14).
"""

import csv
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from quad_fingerprint.database import ReferenceDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Ergebnis einer einzelnen Query-Auswertung.

    Kompatibel mit shazam_fingerprint.evaluate.EvalResult,
    erweitert um quad-spezifische Felder (scale_factors, distortion_type,
    distortion_level).

    Attributes:
        query_file: Dateiname der Query-Audiodatei.
        expected_match: Ground-Truth song_id (welcher Song sollte erkannt werden).
            None falls Song nicht in der Datenbank.
        predicted_match: Vom Matcher zurückgegebene song_id, oder None.
        score: Verifikationsscore (float 0.0–1.0, nicht Integer wie bei Shazam).
        fingerprint_time_ms: Zeit für Fingerprint-Generierung der Query in ms.
        query_time_ms: Zeit für den Matching-Vorgang in ms.
        num_query_quads: Anzahl der erzeugten Query-Quads.
        scale_factors: (s_time, s_freq) des besten Matches, oder None.
        distortion_type: Art der Verzerrung ("speed", "tempo", "pitch", "noise", etc.).
        distortion_level: Verzerrungsgrad als Faktor (z.B. 0.8 für 80% Tempo).
        is_correct: True wenn predicted_match == expected_match.
    """

    query_file: str
    expected_match: str | None
    predicted_match: str | None
    score: float
    fingerprint_time_ms: float
    query_time_ms: float
    num_query_quads: int = 0
    scale_factors: tuple[float, float] | None = None
    distortion_type: str | None = None
    distortion_level: float | None = None
    is_correct: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_correct = self.predicted_match == self.expected_match


@dataclass
class EvalReport:
    """Zusammenfassung aller Evaluationsmetriken über eine Query-Menge.

    Kompatibel mit shazam_fingerprint.evaluate.EvalReport,
    erweitert um specificity und quads_per_second.

    Robustheit (Sonnleitner & Widmer Section VIII):
        recognition_rate: Anteil korrekt erkannter Queries (TP / in_db).
        false_positive_rate: Anteil falsch positiver Erkennungen (FP / not_in_db).
        false_negative_rate: Anteil nicht erkannter Queries (FN / in_db).
        specificity: Anteil korrekter Abstentionen (TN / not_in_db), Eq. (14).

    Effizienz:
        avg_fingerprint_time_ms: Durchschnittliche Fingerprint-Generierungszeit pro Query.
        avg_query_time_ms: Durchschnittliche Matching-Zeit pro Query.
        db_memory_mb: Speicherbedarf der Datenbank in MB.
        quads_per_second: Durchsatz der Quad-Generierung.
        total_queries: Gesamtzahl ausgewerteter Queries.
    """

    recognition_rate: float
    false_positive_rate: float
    false_negative_rate: float
    specificity: float
    avg_fingerprint_time_ms: float
    avg_query_time_ms: float
    db_memory_mb: float
    quads_per_second: float
    total_queries: int
    correct_count: int
    false_positive_count: int
    false_negative_count: int
    true_negative_count: int


def compute_metrics(
    results: list[EvalResult],
    database: ReferenceDatabase | None = None,
) -> EvalReport:
    """Berechnet alle Evaluationsmetriken aus einer Liste von Query-Ergebnissen.

    Begriffsdefinitionen (identisch zu Shazam-Modul):
    - True Positive (TP): expected_match in DB, predicted_match == expected_match.
    - False Negative (FN): expected_match in DB, predicted_match != expected_match
      (Song vorhanden aber nicht erkannt — Robustheitsproblem).
    - False Positive (FP): expected_match == None (kein Match erwartet),
      aber predicted_match ist gesetzt (Falscherkennung).
    - True Negative (TN): expected_match == None, predicted_match == None
      (korrekte Abstenion).

    Specificity = TN / (TN + FP), Sonnleitner & Widmer, Eq. (14).

    Args:
        results: Liste von EvalResult-Objekten, ein Eintrag pro Query.
        database: Optionale ReferenceDatabase für Speicherstatistiken.
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
    true_negatives = sum(1 for r in not_in_db if r.predicted_match is None)

    n_in_db = len(in_db)
    n_not_in_db = len(not_in_db)

    recognition_rate = correct / n_in_db if n_in_db > 0 else 0.0
    false_negative_rate = false_negatives / n_in_db if n_in_db > 0 else 0.0
    false_positive_rate = false_positives / n_not_in_db if n_not_in_db > 0 else 0.0
    # Specificity = TN / (TN + FP), Paper Eq. (14)
    specificity = (
        true_negatives / n_not_in_db if n_not_in_db > 0 else 1.0
    )

    # Effizienz-Metriken
    avg_fp_time = sum(r.fingerprint_time_ms for r in results) / n
    avg_q_time = sum(r.query_time_ms for r in results) / n

    total_quads = sum(r.num_query_quads for r in results)
    total_fp_time_s = sum(r.fingerprint_time_ms for r in results) / 1000.0
    quads_per_sec = total_quads / total_fp_time_s if total_fp_time_s > 0 else 0.0

    db_memory_mb = 0.0
    if database is not None:
        db_memory_mb = database.memory_usage_mb()["total_mb"]

    report = EvalReport(
        recognition_rate=recognition_rate,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        specificity=specificity,
        avg_fingerprint_time_ms=avg_fp_time,
        avg_query_time_ms=avg_q_time,
        db_memory_mb=db_memory_mb,
        quads_per_second=quads_per_sec,
        total_queries=n,
        correct_count=correct,
        false_positive_count=false_positives,
        false_negative_count=false_negatives,
        true_negative_count=true_negatives,
    )

    logger.info(
        "Evaluation: %d Queries | RR=%.1f%% | FNR=%.1f%% | FPR=%.1f%% | "
        "Spec=%.1f%% | FP-Zeit=%.1f ms | Q-Zeit=%.1f ms | %.0f Q/s",
        n,
        recognition_rate * 100,
        false_negative_rate * 100,
        false_positive_rate * 100,
        specificity * 100,
        avg_fp_time,
        avg_q_time,
        quads_per_sec,
    )

    return report


def recognition_by_scale_factor(
    results: list[EvalResult],
) -> dict[str, dict[float, float]]:
    """Berechnet Erkennungsrate nach Verzerrungsgrad und -typ.

    Ermöglicht Reproduktion von Fig. 5 aus Sonnleitner & Widmer (2016):
    Erkennungsrate aufgeschlüsselt nach Distortion-Level (70%–130%) pro
    Distortion-Typ (speed, tempo, pitch).

    Die Query-Verzeichnisstruktur enkodiert Typ und Level:
        queries/tempo_80/song1_tempo_80.wav → distortion_type="tempo", distortion_level=0.80

    Args:
        results: Liste von EvalResult-Objekten mit gesetzten distortion_type
            und distortion_level Feldern.

    Returns:
        Dict der Form {distortion_type: {distortion_level: recognition_rate}}.
        Beispiel: {"tempo": {0.80: 0.95, 1.00: 1.0, 1.20: 0.90},
                   "pitch": {0.80: 0.85, ...}}.
        Nur EvalResults mit expected_match != None werden berücksichtigt.
    """
    # Gruppiere nach (distortion_type, distortion_level)
    groups: dict[str, dict[float, list[EvalResult]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in results:
        if r.distortion_type is None or r.distortion_level is None:
            continue
        if r.expected_match is None:
            continue
        groups[r.distortion_type][r.distortion_level].append(r)

    rate_by_type: dict[str, dict[float, float]] = {}
    for dist_type, levels in sorted(groups.items()):
        rate_by_level: dict[float, float] = {}
        for level, group_results in sorted(levels.items()):
            n_correct = sum(1 for r in group_results if r.is_correct)
            rate_by_level[level] = n_correct / len(group_results)
        rate_by_type[dist_type] = rate_by_level

    logger.info(
        "Scale-Robustheit: %d Distortion-Typen, %d Stufen gesamt",
        len(rate_by_type),
        sum(len(v) for v in rate_by_type.values()),
    )

    return rate_by_type


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
        "results": [_eval_result_to_dict(r) for r in results],
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

    fieldnames = list(_eval_result_to_dict(results[0]).keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(_eval_result_to_dict(r))

    logger.info("CSV-Export: '%s' (%d Zeilen)", path, len(results))


def _eval_result_to_dict(result: EvalResult) -> dict:
    """Konvertiert ein EvalResult in ein serialisierbares Dict.

    Tuples (scale_factors) werden als Liste gespeichert, damit JSON/CSV
    korrekt funktioniert.

    Args:
        result: Einzelnes EvalResult-Objekt.

    Returns:
        Dict mit allen Feldern, scale_factors als Liste.
    """
    d = asdict(result)
    # scale_factors: tuple → list für JSON-Kompatibilität
    if d["scale_factors"] is not None:
        d["scale_factors"] = list(d["scale_factors"])
    return d
