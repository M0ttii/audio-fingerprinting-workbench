"""
Drei-stufiger Matching-Algorithmus nach Sonnleitner & Widmer, Section VI.

Stufe 1 — Kandidatenauswahl + Filterung (Section VI-A):
  Fixed-Radius NN-Suche im 4D-Hash-Raum, dann drei sequentielle Filter:
    Filter 1: Grobe Pitch-Kohärenz (Eq. 6a, 6b)
    Filter 2: Transform-Toleranz mit Skalierungsfaktoren (Eq. 7a/7b → 8a/8b)
    Filter 3: Feine Pitch-Kohärenz (Eq. 9)

Stufe 2 — Sequenzschätzung (Section VI-B):
  Adaptierter Histogram-Ansatz mit skalierten Query-Zeiten.
  Varianz-basiertes Outlier-Removal. Early Exit wenn möglich.

Stufe 3 — Verifikation (Section VI-C):
  Peak-Ausrichtung Referenz → Query-Raum via Skalierungstransformation
  (Eq. 10, 11). Toleranz-Rechtecke für Peak-Matching.
  Verification Score + Coverage-Check.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from quad_fingerprint import config
from quad_fingerprint.database import ReferenceDatabase
from quad_fingerprint.quad_builder import QueryQuad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Ergebnis eines Matching-Durchlaufs.

    Attributes:
        best_match: Dateiname des besten Matches oder None.
        best_score: Durchschnittlicher Verifikationsscore (0.0–1.0).
        best_scale_factors: (s_time, s_freq) des besten Matches.
        match_position_sec: Position der Query im Referenz-Track (Sekunden).
        all_scores: Dict file_name → avg_verification_score für alle geprüften.
        match_found: True wenn ein akzeptierter Match gefunden wurde.
        processing_time_ms: Gesamte Matching-Dauer in Millisekunden.
        candidates_processed: Anzahl geprüfter file_ids.
    """
    best_match: str | None = None
    best_score: float = 0.0
    best_scale_factors: tuple[float, float] | None = None
    match_position_sec: float | None = None
    all_scores: dict[str, float] = field(default_factory=dict)
    match_found: bool = False
    processing_time_ms: float = 0.0
    candidates_processed: int = 0


# ---------------------------------------------------------------------------
# Vorberechnete Toleranzgrenzen (einmalig statt pro Kandidat)
# ---------------------------------------------------------------------------

# Filter 1 + 2: Pitch-/Zeit-Toleranzbereiche
# Eq. 6a/6b, 8a: 1/(1+ε_p) ≤ ratio ≤ 1/(1−ε_p)
# Eq. 8b:        1/(1+ε_t) ≤ s_time ≤ 1/(1−ε_t)
_PITCH_LO: float = 1.0 / (1.0 + config.EPSILON_P)  # ≈ 0.763
_PITCH_HI: float = 1.0 / (1.0 - config.EPSILON_P)  # ≈ 1.449
_TIME_LO: float = 1.0 / (1.0 + config.EPSILON_T)   # ≈ 0.763
_TIME_HI: float = 1.0 / (1.0 - config.EPSILON_T)   # ≈ 1.449

# Histogramm-Bin-Breite in Frames (für Sequenzschätzung)
_HIST_BIN_FRAMES: float = config.HISTOGRAM_BIN_SIZE_SEC * config.FRAMES_PER_SECOND

# Verifikations-Zeitbereich in Frames (±1.8 s → ±450 Frames)
_VERIF_TIME_FRAMES: float = (
    config.VERIFICATION_TIME_RANGE_SEC * config.FRAMES_PER_SECOND
)

# Verifikations-Toleranz: Halbbreiten der Toleranz-Rechtecke
# Paper Section VIII: "rectangular alignment regions of height 12 frequency bins
# and 18 STFT frames (0.072s)"
_VERIF_FREQ_HALF: float = config.VERIFICATION_FREQ_TOLERANCE / 2.0  # ±6 Bins
_VERIF_TIME_HALF: float = config.VERIFICATION_TIME_TOLERANCE / 2.0  # ±9 Frames


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def identify(
    query_quads: list[QueryQuad],
    query_peaks: np.ndarray,
    database: ReferenceDatabase,
    query_duration_sec: float,
) -> MatchResult:
    """Identifiziert ein Query-Audio gegen die Referenz-Datenbank.

    Drei-stufiger Matching-Prozess nach Section VI:
      1. Kandidatenauswahl + Filterung (VI-A)
      2. Sequenzschätzung (VI-B)
      3. Verifikation (VI-C)

    Args:
        query_quads: Query-Quads von build_query_quads().
        query_peaks: (N, 2) float32 Query-Peaks [(time, freq), ...],
            nach Zeit sortiert.
        database: Finalisierte ReferenceDatabase.
        query_duration_sec: Dauer des Query-Snippets in Sekunden.

    Returns:
        MatchResult mit allen Ergebnisfeldern.
    """
    t_start = time.perf_counter()
    result = MatchResult()

    if len(query_quads) == 0:
        result.processing_time_ms = (time.perf_counter() - t_start) * 1000
        logger.warning("Keine Query-Quads — kein Match möglich.")
        return result

    # === Stufe 1: Kandidatenauswahl + Filterung (Section VI-A) ===
    candidates_by_fid = _select_and_filter_candidates(query_quads, database)

    if not candidates_by_fid:
        result.processing_time_ms = (time.perf_counter() - t_start) * 1000
        logger.debug("Keine Kandidaten nach Filterung.")
        return result

    # Query-Peaks nach Zeit sortieren (für effiziente Bisection in Verifikation)
    q_sort_idx = np.argsort(query_peaks[:, 0])
    q_times_sorted = query_peaks[q_sort_idx, 0].astype(np.float64)
    q_freqs_sorted = query_peaks[q_sort_idx, 1].astype(np.float64)

    # Sortiere file_ids absteigend nach Kandidatenanzahl (für Early Exit)
    # Paper Section VI-B: "sort the groups by the number of match candidates,
    # in decreasing order"
    sorted_fids = sorted(
        candidates_by_fid.keys(),
        key=lambda fid: len(candidates_by_fid[fid]),
        reverse=True,
    )

    query_duration_frames = query_duration_sec * config.FRAMES_PER_SECOND
    best_verified_count = 0

    # === Stufe 2 + 3: Sequenzschätzung + Verifikation pro file_id ===
    for fid in sorted_fids:
        cands = candidates_by_fid[fid]
        result.candidates_processed += 1

        # Early Exit (Section VI-C):
        # "return the best verified sequence ... as soon as it becomes evident
        # that the next file-ID has a smaller number of associated match
        # candidates than the number of verified matches"
        if len(cands) <= best_verified_count:
            logger.debug(
                "Early Exit bei file_id=%d: %d Kandidaten ≤ %d verifizierte",
                fid, len(cands), best_verified_count,
            )
            break

        # --- Stufe 2: Sequenzschätzung (Section VI-B) ---
        sequence = _estimate_sequence(cands)
        if sequence is None or len(sequence) < config.SEQUENCE_MIN_MATCHES:
            continue

        # --- Stufe 3: Verifikation (Section VI-C) ---
        avg_score, s_time_med, s_freq_med, match_pos_frames = _verify_sequence(
            sequence, q_times_sorted, q_freqs_sorted, database, fid,
        )

        file_name = database.get_file_name(fid)
        result.all_scores[file_name] = avg_score

        # Akzeptanzkriterien:
        # 1. Avg Verification Score >= VERIFICATION_THRESHOLD (0.53)
        # 2. Sequenz deckt >= MATCH_COVERAGE_THRESHOLD (15%) der Query ab
        if avg_score >= config.VERIFICATION_THRESHOLD:
            seq_times_query = np.array([c['a_time_query'] for c in sequence])
            time_span = float(np.max(seq_times_query) - np.min(seq_times_query))
            coverage = time_span / query_duration_frames if query_duration_frames > 0 else 0

            if coverage >= config.MATCH_COVERAGE_THRESHOLD:
                if len(sequence) > best_verified_count:
                    best_verified_count = len(sequence)
                    result.best_match = file_name
                    result.best_score = round(avg_score, 6)
                    result.best_scale_factors = (
                        round(float(s_time_med), 6),
                        round(float(s_freq_med), 6),
                    )
                    result.match_position_sec = round(
                        float(match_pos_frames) / config.FRAMES_PER_SECOND, 4
                    )
                    result.match_found = True

    result.processing_time_ms = round(
        (time.perf_counter() - t_start) * 1000, 2
    )

    if result.match_found:
        logger.info(
            "Match: '%s' | Score=%.3f | s=(%.3f, %.3f) | Pos=%.2f s | %.0f ms",
            result.best_match, result.best_score,
            result.best_scale_factors[0], result.best_scale_factors[1],
            result.match_position_sec, result.processing_time_ms,
        )
    else:
        logger.debug(
            "Kein Match | %d file_ids geprüft | %.0f ms",
            result.candidates_processed, result.processing_time_ms,
        )

    return result


# ---------------------------------------------------------------------------
# Stufe 1: Kandidatenauswahl + Filterung (Section VI-A)
# ---------------------------------------------------------------------------

def _select_and_filter_candidates(
    query_quads: list[QueryQuad],
    database: ReferenceDatabase,
) -> dict[int, list[dict]]:
    """Wählt und filtert Match-Kandidaten für alle Query-Quads.

    Für jeden Query-Quad wird eine Fixed-Radius NN-Suche durchgeführt.
    Die Rohkandidaten werden durch drei sequentielle Filter reduziert:
      Filter 1: Grobe Pitch-Kohärenz (Eq. 6a, 6b)
      Filter 2: Transform-Toleranz (Eq. 7a/7b → 8a/8b)
      Filter 3: Feine Pitch-Kohärenz (Eq. 9)

    Paper Section VI-A: "From this point onwards, this stage and subsequent
    stages operate on the spectral quads rather than on their hashes."

    Args:
        query_quads: Alle Query-Quads.
        database: Finalisierte ReferenceDatabase.

    Returns:
        Dict: file_id → Liste von Kandidaten-Dicts.
    """
    # Batch-Suche: alle Query-Hashes auf einmal an den cKDTree
    query_hashes = np.array(
        [qq.hash for qq in query_quads], dtype=np.float32
    )
    all_nn_results = database.query_radius(
        query_hashes, radius=config.SEARCH_RADIUS,
    )

    candidates_by_fid: dict[int, list[dict]] = {}
    n_raw = 0
    n_f1 = 0
    n_f2 = 0
    n_f3 = 0

    for qq, nn_indices in zip(query_quads, all_nn_results):
        if len(nn_indices) == 0:
            continue

        n_raw += len(nn_indices)
        records = database.get_refrecords_by_indices(nn_indices)

        # Query-Quad-Werte (einmalig pro Quad)
        a_x_q = float(qq.root_point[0])
        a_y_q = float(qq.root_point[1])
        s_x_q = float(qq.quad_size[0])   # S_x^query = B_x − A_x
        s_y_q = float(qq.quad_size[1])   # S_y^query = B_y − A_y

        for rec in records:
            a_x_r = float(rec['a_time'])
            a_y_r = float(rec['a_freq'])
            s_x_r = float(rec['s_time'])  # S_x^ref
            s_y_r = float(rec['s_freq'])  # S_y^ref
            fid = int(rec['file_id'])

            # --- Filter 1: Grobe Pitch-Kohärenz (Eq. 6a, 6b) ---
            # A_y^query / A_y^cand ∈ [1/(1+ε_p), 1/(1−ε_p)]
            if a_y_r <= 0.0:
                continue
            ratio = a_y_q / a_y_r
            if ratio < _PITCH_LO or ratio > _PITCH_HI:
                continue
            n_f1 += 1

            # --- Filter 2: Transform-Toleranz (Eq. 7a/7b → 8a/8b) ---
            # Skalierungsfaktoren berechnen
            if abs(s_x_r) < 1e-8 or abs(s_y_r) < 1e-8:
                continue
            s_time = s_x_q / s_x_r  # Eq. 7a: s_time = S_x^query / S_x^ref
            s_freq = s_y_q / s_y_r  # Eq. 7b: s_freq = S_y^query / S_y^ref

            # Eq. 8a: s_freq (pitch) in Toleranzbereich
            if s_freq < _PITCH_LO or s_freq > _PITCH_HI:
                continue
            # Eq. 8b: s_time in Toleranzbereich
            if s_time < _TIME_LO or s_time > _TIME_HI:
                continue
            n_f2 += 1

            # --- Filter 3: Feine Pitch-Kohärenz (Eq. 9) ---
            # |A_y^query − A_y^ref · s_freq| ≤ ε_pfine
            # Paper: "ε_pfine = 1.8, which we determined empirically"
            if abs(a_y_q - a_y_r * s_freq) > config.FINE_PITCH_COHERENCE_THRESHOLD:
                continue
            n_f3 += 1

            # Kandidat akzeptiert — Skalierungsfaktoren mitspeichern
            # Paper: "The scale factors are stored with the accepted candidates"
            cand = {
                'a_time_query': a_x_q,
                'a_freq_query': a_y_q,
                'a_time_ref': a_x_r,
                'a_freq_ref': a_y_r,
                's_time': s_time,
                's_freq': s_freq,
            }
            candidates_by_fid.setdefault(fid, []).append(cand)

    logger.debug(
        "Kandidaten: %d roh → %d F1 → %d F2 → %d F3 | %d file_ids",
        n_raw, n_f1, n_f2, n_f3, len(candidates_by_fid),
    )
    return candidates_by_fid


# ---------------------------------------------------------------------------
# Stufe 2: Sequenzschätzung (Section VI-B)
# ---------------------------------------------------------------------------

def _estimate_sequence(candidates: list[dict]) -> list[dict] | None:
    """Findet die längste Match-Sequenz via adaptiertem Histogram-Ansatz.

    Paper Section VI-B: "We adapt the [Shazam histogram] method such that
    the query time is scaled according to the uncovered time scale factor
    s_time. The file-ID for the largest histogram bin (the longest match
    sequence) is returned."

    Anschließend varianzbasiertes Outlier-Removal der s_time-Werte.

    Args:
        candidates: Kandidaten einer file_id (nach Filterung aus Stufe 1).

    Returns:
        Bereinigte Sequenz (Liste von Kandidaten-Dicts) oder None.
    """
    if len(candidates) < config.SEQUENCE_MIN_MATCHES:
        return None

    # Skalierte Zeitdifferenz für Histogramm (Section VI-B):
    # offset = A_x^query / s_time − A_x^ref
    # Wenn die Query eine skalierte Version des Referenztracks ist,
    # ergeben alle korrekten Matches denselben Offset.
    offsets = np.array([
        c['a_time_query'] / c['s_time'] - c['a_time_ref']
        for c in candidates
    ], dtype=np.float64)

    # Histogramm-Binning
    min_off = float(np.min(offsets))
    max_off = float(np.max(offsets))
    span = max_off - min_off

    if span < _HIST_BIN_FRAMES:
        # Alle Kandidaten in einem Bin
        best_indices = np.arange(len(candidates))
    else:
        n_bins = max(1, int(np.ceil(span / _HIST_BIN_FRAMES)) + 1)
        bin_ids = ((offsets - min_off) / _HIST_BIN_FRAMES).astype(np.intp)
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)

        counts = np.bincount(bin_ids, minlength=n_bins)
        best_bin = int(np.argmax(counts))
        best_indices = np.where(bin_ids == best_bin)[0]

    if len(best_indices) < config.SEQUENCE_MIN_MATCHES:
        return None

    sequence = [candidates[i] for i in best_indices]

    # Varianz-basiertes Outlier-Removal (Section VI-B):
    # "Resulting sequences with a variance of scale transforms larger than a
    # threshold value are cleaned up using a simple variance based outlier
    # detection method."
    if len(sequence) > 2:
        s_times = np.array([c['s_time'] for c in sequence])
        median_st = float(np.median(s_times))
        std_st = float(np.std(s_times))
        if std_st > 0:
            keep = np.abs(s_times - median_st) <= config.OUTLIER_SIGMA_FACTOR * std_st
            sequence = [c for c, k in zip(sequence, keep) if k]

    if len(sequence) < config.SEQUENCE_MIN_MATCHES:
        return None

    return sequence


# ---------------------------------------------------------------------------
# Stufe 3: Verifikation (Section VI-C)
# ---------------------------------------------------------------------------

def _verify_sequence(
    sequence: list[dict],
    q_times: np.ndarray,
    q_freqs: np.ndarray,
    database: ReferenceDatabase,
    file_id: int,
) -> tuple[float, float, float, float]:
    """Verifiziert eine Match-Sequenz über Peak-Ausrichtung.

    Paper Section VI-C: "spectral peaks that were extracted nearby a matching
    reference quad in the reference audio should also be present in the query
    audio."

    Für jeden Kandidaten in der Sequenz:
      1. Referenz-Peaks im Zeitbereich ±VERIFICATION_TIME_RANGE_SEC um A^ref laden
      2. Jeden Referenz-Peak in den Query-Raum transformieren (Eq. 10, 11)
      3. In einem Toleranz-Rechteck nach einem Query-Peak suchen
      4. Verification Score = Anteil erfolgreich ausgerichteter Peaks

    Paper Section VIII: "threshold for the average verification score of
    sequences is set to 0.53"

    Args:
        sequence: Kandidaten-Sequenz einer file_id.
        q_times: Nach Zeit sortierte Query-Peak-Zeiten (float64).
        q_freqs: Zugehörige Query-Peak-Frequenzen (float64).
        database: ReferenceDatabase mit Peakfile.
        file_id: Datei-ID des Kandidaten.

    Returns:
        Tuple (avg_score, median_s_time, median_s_freq, match_position_frames).
    """
    ref_peaks = database.get_peaks_for_file(file_id)
    # ref_peaks ist nach Zeit sortiert (für Bisection)
    ref_times = ref_peaks[:, 0].astype(np.float64)
    ref_freqs = ref_peaks[:, 1].astype(np.float64)

    # Mediane Skalierungsfaktoren der Sequenz
    s_times_seq = np.array([c['s_time'] for c in sequence])
    s_freqs_seq = np.array([c['s_freq'] for c in sequence])
    s_time_med = float(np.median(s_times_seq))
    s_freq_med = float(np.median(s_freqs_seq))

    total_score = 0.0
    n_scored = 0

    for cand in sequence:
        a_x_r = cand['a_time_ref']
        a_y_r = cand['a_freq_ref']
        a_x_q = cand['a_time_query']
        a_y_q = cand['a_freq_query']
        s_t = cand['s_time']
        s_f = cand['s_freq']

        # Referenz-Peaks im Zeitbereich ±VERIFICATION_TIME_RANGE um A^ref
        # Paper: "peaks in the range of ±1.8s near the reference candidate
        # rootpoint" → bisecting the time values
        t_lo = a_x_r - _VERIF_TIME_FRAMES
        t_hi = a_x_r + _VERIF_TIME_FRAMES
        idx_lo = int(np.searchsorted(ref_times, t_lo, side='left'))
        idx_hi = int(np.searchsorted(ref_times, t_hi, side='right'))

        n_nearby = idx_hi - idx_lo
        if n_nearby == 0:
            continue

        nearby_t = ref_times[idx_lo:idx_hi]
        nearby_f = ref_freqs[idx_lo:idx_hi]

        # Transformation: Referenz-Peaks → Query-Raum (Eq. 10, 11)
        # off = (P^ref − A^ref) · s
        # P^query_est = A^query + off
        est_times = a_x_q + (nearby_t - a_x_r) * s_t
        est_freqs = a_y_q + (nearby_f - a_y_r) * s_f

        # Für jeden geschätzten Punkt: suche Query-Peak in Toleranz-Rechteck
        # Paper Section VIII: "rectangular alignment regions of height
        # 12 frequency bins and 18 STFT frames"
        aligned = 0
        for j in range(n_nearby):
            et = est_times[j]
            ef = est_freqs[j]

            # Bisection in sortierten Query-Zeiten → Zeit-Fenster
            qlo = int(np.searchsorted(q_times, et - _VERIF_TIME_HALF, side='left'))
            qhi = int(np.searchsorted(q_times, et + _VERIF_TIME_HALF, side='right'))
            if qlo >= qhi:
                continue

            # Frequenz-Check innerhalb des Zeit-Fensters
            if np.any(np.abs(q_freqs[qlo:qhi] - ef) <= _VERIF_FREQ_HALF):
                aligned += 1

        # Verification Score = v / N
        score = aligned / n_nearby
        total_score += score
        n_scored += 1

    avg_score = total_score / n_scored if n_scored > 0 else 0.0

    # Match-Position: minimaler A_x^ref-Wert in der Sequenz
    # Paper: "the minimal time value A_x of the peaks in the histogram bin"
    match_pos_frames = min(c['a_time_ref'] for c in sequence)

    return avg_score, s_time_med, s_freq_med, match_pos_frames
