"""
Modul für das Matching von Query-Fingerprints gegen die Datenbank.

Implementiert das Scoring-Verfahren nach Wang (2003) Section 2.3
"Searching and Scoring": Für jede Query wird ein Hash-Lookup durchgeführt,
die resultierenden Treffer nach Song gruppiert und mittels δt-Histogram
auf Zeitkohärenz geprüft.

Wang: "The time pairs are distributed into bins according to the track ID
associated with the matching database hash. Then we calculate a histogram
of δt values and scan for a peak."

Wang: "The score of the match is the number of matching points in the
histogram peak."
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from shazam_fingerprint import config
from shazam_fingerprint.database import FingerprintDatabase

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Ergebnis eines Matching-Vorgangs.

    Attributes:
        best_match: Song-ID des besten Treffers, oder None falls kein Match
            den Schwellwert überschreitet.
        best_score: Höhe des höchsten Histogram-Peaks des besten Treffers.
            Wang: "The score of the match is the number of matching points
            in the histogram peak."
        all_scores: Dict aller Kandidaten mit ihrem jeweiligen höchsten
            Histogram-Peak-Score. Nützlich für Analyse und Visualisierung.
        match_found: True wenn best_score >= config.MATCH_THRESHOLD.
        num_query_hashes: Anzahl der Query-Hashes, die zum Lookup verwendet wurden.
        processing_time_ms: Dauer des gesamten Matching-Vorgangs in Millisekunden.
            Wichtig für die Effizienz-Evaluation.
    """

    best_match: str | None
    best_score: int
    all_scores: dict[str, int] = field(default_factory=dict)
    match_found: bool = False
    num_query_hashes: int = 0
    processing_time_ms: float = 0.0


def match(
    query_fingerprints: list[tuple[int, int]],
    database: FingerprintDatabase,
) -> MatchResult:
    """Sucht die beste Übereinstimmung für Query-Fingerprints in der Datenbank.

    Algorithmus nach Wang Section 2.3:
    1. Für jeden Query-Hash: Lookup in der Datenbank → (song_id, t_db)-Treffer.
    2. Berechne δt = t_db - t_query für jeden Treffer.
    3. Gruppiere alle δt-Werte nach song_id ("distribute into bins by track ID").
    4. Erstelle pro song_id ein Histogram der δt-Werte.
    5. Score = Höhe des höchsten Histogram-Peaks.
    6. Best Match = song_id mit dem höchsten Score.

    Wang Section 2.3.1: "an acceptable false positive rate is chosen, then a
    threshold score is chosen that meets or exceeds the false-positive criterion."

    Args:
        query_fingerprints: Liste von (hash_value, anchor_time)-Tupeln der Query,
            erzeugt durch fingerprint.generate_fingerprints().
        database: FingerprintDatabase mit indexierten Referenz-Songs.

    Returns:
        MatchResult mit dem besten Treffer, Score und Timing-Informationen.
    """
    start = time.perf_counter()

    if not query_fingerprints:
        logger.warning("Leere Query-Fingerprints — kein Matching möglich.")
        return MatchResult(
            best_match=None,
            best_score=0,
            processing_time_ms=0.0,
        )

    # --- Schritt 1+2: Hash-Lookup und δt-Berechnung ---
    # Sammle pro song_id alle δt-Werte.
    # Wang: "The time pairs are distributed into bins according to the
    # track ID associated with the matching database hash."
    delta_t_by_song: defaultdict[str, list[int]] = defaultdict(list)

    for hash_query, t_query in query_fingerprints:
        hits = database.lookup(hash_query)
        for song_id, t_db in hits:
            delta_t = t_db - t_query
            delta_t_by_song[song_id].append(delta_t)

    # --- Schritt 3: Vor-Filterung ---
    # Songs mit zu wenigen Hash-Matches überspringen, um Rechenzeit für
    # das Histogram-Scoring zu sparen.
    candidates = {
        sid: dt_list
        for sid, dt_list in delta_t_by_song.items()
        if len(dt_list) >= config.MIN_HASH_MATCHES
    }

    # --- Schritt 4+5: Histogram-Scoring ---
    # Wang Section 2.3: "calculate a histogram of δt values and scan for a peak."
    all_scores: dict[str, int] = {}

    for song_id, dt_list in candidates.items():
        score = _histogram_peak_score(dt_list)
        all_scores[song_id] = score

    # --- Schritt 6: Bestes Ergebnis bestimmen ---
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if not all_scores:
        logger.info(
            "Kein Kandidat mit >= %d Hash-Matches (Query: %d Hashes, %.1f ms)",
            config.MIN_HASH_MATCHES,
            len(query_fingerprints),
            elapsed_ms,
        )
        return MatchResult(
            best_match=None,
            best_score=0,
            all_scores={},
            match_found=False,
            num_query_hashes=len(query_fingerprints),
            processing_time_ms=elapsed_ms,
        )

    best_song = max(all_scores, key=all_scores.get)  # type: ignore[arg-type]
    best_score = all_scores[best_song]
    match_found = best_score >= config.MATCH_THRESHOLD

    logger.info(
        "Match: '%s' (Score %d, Threshold %d, %s) | %d Kandidaten | %.1f ms",
        best_song,
        best_score,
        config.MATCH_THRESHOLD,
        "MATCH" if match_found else "KEIN MATCH",
        len(all_scores),
        elapsed_ms,
    )

    return MatchResult(
        best_match=best_song if match_found else None,
        best_score=best_score,
        all_scores=all_scores,
        match_found=match_found,
        num_query_hashes=len(query_fingerprints),
        processing_time_ms=elapsed_ms,
    )


def _histogram_peak_score(delta_t_values: list[int]) -> int:
    """Berechnet den höchsten Histogram-Peak-Score aus δt-Werten.

    Erstellt ein Histogram der Zeitdifferenzen (δt = t_db - t_query).
    Ein echter Match zeigt eine starke Häufung bei einem bestimmten δt,
    da alle Hash-Paare den gleichen zeitlichen Offset zum Datenbank-Track haben.

    Wang Section 2.3: "calculate a histogram of these δt values and scan for
    a peak. An example of a matching track and its histogram can be found
    in Fig. 3."

    Args:
        delta_t_values: Liste von δt-Werten (t_db - t_query) in Frames.

    Returns:
        Höhe des höchsten Histogram-Bins. Bei config.HISTOGRAM_BIN_WIDTH=1
        entspricht dies der Anzahl Hash-Paare mit exakt identischem Zeitoffset.
    """
    if not delta_t_values:
        return 0

    if config.HISTOGRAM_BIN_WIDTH == 1:
        # Optimierter Pfad: Einfaches Zählen ohne numpy-Overhead.
        # Zähle die häufigste δt-Wert.
        counts: defaultdict[int, int] = defaultdict(int)
        for dt in delta_t_values:
            counts[dt] += 1
        return max(counts.values())

    # Allgemeiner Pfad für HISTOGRAM_BIN_WIDTH > 1:
    # Quantisiere δt-Werte in Bins und zähle die häufigste Bin.
    counts_binned: defaultdict[int, int] = defaultdict(int)
    for dt in delta_t_values:
        bin_idx = dt // config.HISTOGRAM_BIN_WIDTH
        counts_binned[bin_idx] += 1
    return max(counts_binned.values())
