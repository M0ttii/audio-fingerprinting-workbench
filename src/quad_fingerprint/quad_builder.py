"""
Quad-Gruppierung und Hash-Berechnung — der Kern des Quad-Algorithmus.

Aus Gruppen von vier spektralen Peaks (Quads) werden translations- und
skalierungsinvariante 4D-Hashes im Einheitsquadrat [0,1]^4 berechnet.
KEIN Integer-Hash — das ist der fundamentale Unterschied zu Shazam.

Algorithmus (Section IV-A bis IV-C):
  1. Jeder Peak wird als Root-Point A versucht.
  2. Im Gruppierungsfenster [A_x + r_start, A_x + r_end] werden alle (m C 3)
     Triples als potenzielle (B, C, D) geprüft.
  3. Gültigkeitsbedingung: Eq. (1a)–(1c) — C, D im Rechteck A–B.
  4. Hash: Normierung ins Einheitsquadrat → (C'_x, C'_y, D'_x, D'_y).
  5. Stärke-Auswahl: Top-q Quads pro Sekunde nach Magnitude-Score.

Zwei Parametersätze:
  - Referenz: Kleine Fenster, wenige Quads (q=9/s) → kompakte DB.
  - Query: Große Fenster (Eq. 4a–4c), viele Quads (q=1500/s) → hohe Trefferquote.

Referenz: Sonnleitner & Widmer (2016), Sections IV-A bis IV-C.
"""

import logging
from dataclasses import dataclass

import numpy as np

from quad_fingerprint import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------

@dataclass
class QuadRecord:
    """Referenz-Quad-Record für die Datenbank (36 Bytes/Record im Paper).

    Paper Section V: "A quad record consist of spectral peak A and S, where S
    denotes the height and width of the quad in spectrogram space (i.e.
    S_x = B_x − A_x and S_y = B_y − A_y), the quad hash C', D', and the
    file-ID."

    Attributes:
        root_point: Koordinaten von A als float32[2]: (time, freq).
        quad_size: Größe S als float32[2]: (S_x = B_x − A_x, S_y = B_y − A_y).
        hash: 4D-Hash als float32[4]: (C'_x, C'_y, D'_x, D'_y) in [0,1]^4.
        file_id: Eindeutige Datei-ID (uint32).
    """
    root_point: np.ndarray  # float32[2]
    quad_size: np.ndarray   # float32[2]
    hash: np.ndarray        # float32[4]
    file_id: int


@dataclass
class QueryQuad:
    """Query-Quad für die Suche.

    Wie QuadRecord, aber ohne file_id. quad_size wird im Matcher für die
    Skalierungsfaktoren s_time, s_freq benötigt (Eq. 7a, 7b).

    Attributes:
        root_point: Koordinaten von A als float32[2]: (time, freq).
        quad_size: Größe S als float32[2]: (S_x, S_y).
        hash: 4D-Hash als float32[4]: (C'_x, C'_y, D'_x, D'_y) in [0,1]^4.
    """
    root_point: np.ndarray  # float32[2]
    quad_size: np.ndarray   # float32[2]
    hash: np.ndarray        # float32[4]


# ---------------------------------------------------------------------------
# Öffentliches API
# ---------------------------------------------------------------------------

def compute_quad_hash(
    a_time: float, a_freq: float,
    b_time: float, b_freq: float,
    c_time: float, c_freq: float,
    d_time: float, d_freq: float,
) -> np.ndarray:
    """Berechnet den translations- und skalierungsinvarianten 4D-Hash.

    Paper Section IV-C: Normierung ins Einheitsquadrat mit A' = (0,0), B' = (1,1).
    Hash = (C'_x, C'_y, D'_x, D'_y), wobei:
        C'_x = (C_x − A_x) / (B_x − A_x)
        C'_y = (C_y − A_y) / (B_y − A_y)
    Translationsinvarianz: Verschiebung um −A.
    Skalierungsinvarianz: Normierung durch (B − A).

    Args:
        a_time, a_freq: Root-Point A (P_x, P_y).
        b_time, b_freq: Point B (definiert Rechteck-Größe).
        c_time, c_freq: Point C (innerhalb Rechteck).
        d_time, d_freq: Point D (innerhalb Rechteck).

    Returns:
        float32[4]: (C'_x, C'_y, D'_x, D'_y) in [0,1]^4.
    """
    dx = b_time - a_time  # > 0, da B rechts von A
    dy = b_freq - a_freq  # > 0, da B_y > A_y (Eq. 1a)

    return np.array([
        (c_time - a_time) / dx,   # C'_x
        (c_freq - a_freq) / dy,   # C'_y
        (d_time - a_time) / dx,   # D'_x
        (d_freq - a_freq) / dy,   # D'_y
    ], dtype=np.float32)


def is_valid_quad(
    a_time: float, a_freq: float,
    b_time: float, b_freq: float,
    c_time: float, c_freq: float,
    d_time: float, d_freq: float,
) -> bool:
    """Prüft die Gültigkeitsbedingung eines Quads (Equations 1a–1c).

    Paper Section IV-A:
        A_y < B_y                        (1a)
        A_x < C_x ≤ D_x ≤ B_x           (1b)
        A_y < C_y, D_y ≤ B_y             (1c)

    "We define a quad to be valid if the points C, D reside in the
    axis-parallel rectangle that is spanned by points A, B."

    Returns:
        True wenn das Quad gültig ist.
    """
    # (1a): B hat strikt höhere Frequenz als A
    if a_freq >= b_freq:
        return False
    # (1b): Zeitliche Ordnung: A < C ≤ D ≤ B
    if not (a_time < c_time and c_time <= d_time and d_time <= b_time):
        return False
    # (1c): C, D im Frequenzbereich (A_y, B_y]
    if not (a_freq < c_freq <= b_freq and a_freq < d_freq <= b_freq):
        return False
    return True


def build_reference_quads(
    peaks: np.ndarray,
    magnitude: np.ndarray,
    file_id: int,
) -> list[QuadRecord]:
    """Erstellt Referenz-Quads für ein Audiosignal.

    Paper Section IV-A: "the quad grouping process proceeds through an audio
    file from left to right, trying each spectral peak as a potential root
    point A of a set of quads, and aims to create up to a number of q quads
    for each second of audio."

    Verwendet Referenz-Fenstergrenzen (config.REF_R_START/END_FRAMES) und
    max. q = REF_QUADS_PER_SECOND (= 9) Quads pro Sekunde.

    Args:
        peaks: Peaks als float32 (N, 2): [(time, freq), ...], nach time sortiert.
        magnitude: Magnitude-Spektrogramm (n_bins, n_frames) für Stärke-Auswahl.
        file_id: Eindeutige ID dieser Audiodatei.

    Returns:
        Liste von QuadRecord-Objekten.
    """
    params, hashes, scores = _generate_all_candidates(
        peaks, magnitude,
        r_start=float(config.REF_R_START_FRAMES),
        r_end=float(config.REF_R_END_FRAMES),
    )

    if len(params) == 0:
        logger.warning("Keine gültigen Quads für file_id=%d.", file_id)
        return []

    sel_p, sel_h = _select_strong_quads(
        params, hashes, scores, config.REF_QUADS_PER_SECOND
    )

    records = _params_to_quad_records(sel_p, sel_h, file_id)

    logger.debug(
        "Referenz-Quads: %d (von %d Kandidaten) | file_id=%d",
        len(records), len(params), file_id,
    )
    return records


def build_query_quads(
    peaks: np.ndarray,
    magnitude: np.ndarray,
) -> list[QueryQuad]:
    """Erstellt Query-Quads mit erweiterten Fenstergrenzen und Subspace-Check.

    Fenstergrenzen nach Eq. (4a)–(4c), Subspace-Check nach Eq. (5).

    Paper Section IV-C: "44.7% of possible quads could be rejected" durch den
    Relevanter-Teilraum-Check. Quads mit C'_x < HASH_CX_MIN − ε_L werden
    verworfen, da kein äquivalentes Referenz-Quad existiert.

    Args:
        peaks: Peaks als float32 (N, 2): [(time, freq), ...].
        magnitude: Magnitude-Spektrogramm (n_bins, n_frames).

    Returns:
        Liste von QueryQuad-Objekten.
    """
    params, hashes, scores = _generate_all_candidates(
        peaks, magnitude,
        r_start=config.QUERY_R_START_FRAMES,
        r_end=config.QUERY_R_END_FRAMES,
    )

    if len(params) == 0:
        logger.warning("Keine gültigen Query-Quads.")
        return []

    # --- Relevanter-Teilraum-Check (Section IV-C, Eq. 5) ---
    # C'_x ≥ HASH_CX_MIN − ε_L (es reicht C'_x zu prüfen, da C'_x ≤ D'_x)
    threshold = config.HASH_CX_MIN - config.SEARCH_RADIUS
    subspace_mask = hashes[:, 0] >= threshold
    n_total = len(params)
    n_rejected = int(np.sum(~subspace_mask))

    params = params[subspace_mask]
    hashes = hashes[subspace_mask]
    scores = scores[subspace_mask]

    if len(params) == 0:
        logger.warning("Alle Query-Quads durch Subspace-Check verworfen.")
        return []

    # --- Stärke-Auswahl ---
    sel_p, sel_h = _select_strong_quads(
        params, hashes, scores, config.QUERY_QUADS_PER_SECOND
    )

    quads = _params_to_query_quads(sel_p, sel_h)

    logger.debug(
        "Query-Quads: %d | Subspace-Verwerfung: %d / %d (%.1f%%)",
        len(quads), n_rejected, n_total,
        100.0 * n_rejected / max(n_total, 1),
    )
    return quads


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------

def _generate_all_candidates(
    peaks: np.ndarray,
    magnitude: np.ndarray,
    r_start: float,
    r_end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Erzeugt alle gültigen Quad-Kandidaten für alle Root-Points (vektorisiert).

    Für jeden Peak als Root-Point A werden alle Peaks im Gruppierungsfenster
    [A_x + r_start, A_x + r_end] gesammelt und alle (m C 3) Triples als
    potenzielle (B, C, D) geprüft.

    Implementierung: Der Python-`combinations`-Loop wurde durch numpy-Broadcasting
    ersetzt. Für jedes Root-Peak werden alle Tripel-Indizes (i<j<k) per
    np.triu_indices + Reset-Cumsum-Trick komplett in numpy erzeugt, dann
    Validity-Check und Hash-Berechnung vektorisiert durchgeführt.
    Speedup: ~10–15× gegenüber der itertools-Version.

    Args:
        peaks: (N, 2) float32, [(time, freq), ...], nach time sortiert.
        magnitude: (n_bins, n_frames) Spektrogramm.
        r_start: Start des Gruppierungsfensters in Frames (relativ zu A).
        r_end: Ende des Gruppierungsfensters in Frames (relativ zu A).

    Returns:
        Tuple (params, hashes, scores):
            - params: (K, 4) float32 — [a_time, a_freq, b_time, b_freq]
            - hashes: (K, 4) float32 — Hash-Vektoren
            - scores: (K,) float32 — Magnitude-Scores (Summe B + C + D)
    """
    n_peaks = len(peaks)
    empty = (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )

    if n_peaks < 4:
        return empty

    times = peaks[:, 0].astype(np.float64)
    freqs = peaks[:, 1].astype(np.float64)
    n_bins, n_frames = magnitude.shape

    # Magnitude für jeden Peak vorab berechnen (gerundete Integer-Indizes)
    peak_t_int = np.clip(np.round(times).astype(int), 0, n_frames - 1)
    peak_f_int = np.clip(np.round(freqs).astype(int), 0, n_bins - 1)
    peak_mags = magnitude[peak_f_int, peak_t_int].astype(np.float64)

    all_params_list: list[np.ndarray] = []
    all_hashes_list: list[np.ndarray] = []
    all_scores_list: list[np.ndarray] = []

    for root_idx in range(n_peaks):
        a_t = times[root_idx]
        a_f = freqs[root_idx]

        # Gruppierungsfenster: [A_x + r_start, A_x + r_end]
        region_lo = a_t + r_start
        region_hi = a_t + r_end

        # Binäre Suche (Peaks sind time-sortiert)
        i_lo = int(np.searchsorted(times, region_lo, side='left'))
        i_hi = int(np.searchsorted(times, region_hi, side='right'))

        n_region = i_hi - i_lo
        if n_region < 3:
            continue

        r_times = times[i_lo:i_hi]
        r_freqs = freqs[i_lo:i_hi]
        r_mags  = peak_mags[i_lo:i_hi]

        # Quick check: Mindestens ein Peak mit freq > a_f muss existieren (Eq. 1a)
        if np.max(r_freqs) <= a_f:
            continue

        # ------------------------------------------------------------------
        # Vektorisierte Tripel-Generierung: alle (i, j, k) mit i < j < k
        # ------------------------------------------------------------------
        # Schritt 1: alle Paare (ii_2d, jj_2d) mit i < j via np.triu_indices
        ii_2d, jj_2d = np.triu_indices(n_region, k=1)  # (nC2,)
        rep = n_region - jj_2d - 1   # Anzahl gültiger k-Werte je (i,j)-Paar
        valid_pairs = rep > 0
        ii_2d = ii_2d[valid_pairs]
        jj_2d = jj_2d[valid_pairs]
        rep   = rep[valid_pairs]

        if len(rep) == 0:
            continue

        # Schritt 2: kk-Array ohne Python-Loop via Reset-Cumsum-Trick.
        # Für jedes Paar (ii_2d[p], jj_2d[p]) mit rep[p] Wiederholungen soll
        # kk die Werte jj_2d[p]+1, jj_2d[p]+2, ..., jj_2d[p]+rep[p] enthalten.
        #
        # Herleitung: starte kk als Array aus Einsen (Schrittgröße = 1).
        # Am Anfang jeder Gruppe p setze kk[starts[p]] so, dass nach cumsum
        # der gewünschte Startwert jj_2d[p]+1 herauskommt:
        #   kk[starts[p]] = jj_2d[p]+1 − (jj_2d[p-1]+rep[p-1])   für p ≥ 1
        #   kk[starts[0]] = jj_2d[0]+1
        total  = int(rep.sum())
        ends   = np.cumsum(rep)
        starts = ends - rep

        kk_delta = np.ones(total, dtype=np.int64)
        kk_delta[0] = int(jj_2d[0]) + 1
        if len(starts) > 1:
            kk_delta[starts[1:]] = (
                jj_2d[1:].astype(np.int64) + 1
                - jj_2d[:-1].astype(np.int64)
                - rep[:-1].astype(np.int64)
            )
        kk = np.cumsum(kk_delta)          # (total,) — k-Indices

        ii = np.repeat(ii_2d, rep)        # (total,) — i-Indices
        jj = np.repeat(jj_2d, rep)        # (total,) — j-Indices

        # ------------------------------------------------------------------
        # Koordinaten aller Tripel sammeln: (M, 3) Arrays
        # ------------------------------------------------------------------
        t_tri = np.stack([r_times[ii], r_times[jj], r_times[kk]], axis=1)
        f_tri = np.stack([r_freqs[ii], r_freqs[jj], r_freqs[kk]], axis=1)
        m_tri = np.stack([r_mags[ii],  r_mags[jj],  r_mags[kk]],  axis=1)

        # ------------------------------------------------------------------
        # Zeilenweise Lexsort nach (time, freq) → C=[:, 0], D=[:, 1], B=[:, 2]
        # Entspricht Python: sorted([(t_i,f_i,m_i), (t_j,f_j,m_j), (t_k,f_k,m_k)])
        # Peaks haben durch parabolische Interpolation float-Koordinaten → kein
        # einfaches combined-key int-Trick; stattdessen zwei stabile Sorts:
        #   1. Stable sort by freq  (sekundärer Schlüssel)
        #   2. Stable sort by time  (primärer Schlüssel, bricht Ties nach freq auf)
        # ------------------------------------------------------------------
        row_idx = np.arange(len(ii))[:, None]          # (M, 1)

        # Schritt 1: stabil nach freq sortieren
        idx1     = np.argsort(f_tri, axis=1, kind='stable')    # (M, 3)
        t_by_f   = t_tri[row_idx, idx1]                        # times nach freq-Sort

        # Schritt 2: stabil nach time sortieren (erhält freq-Reihenfolge bei Gleichstand)
        idx2     = np.argsort(t_by_f, axis=1, kind='stable')   # (M, 3)
        sort_idx = idx1[row_idx, idx2]                         # kombinierter Sort-Index

        t_s = t_tri[row_idx, sort_idx]
        f_s = f_tri[row_idx, sort_idx]
        m_s = m_tri[row_idx, sort_idx]

        c_t = t_s[:, 0];  c_f = f_s[:, 0];  c_m = m_s[:, 0]
        d_t = t_s[:, 1];  d_f = f_s[:, 1];  d_m = m_s[:, 1]
        b_t = t_s[:, 2];  b_f = f_s[:, 2];  b_m = m_s[:, 2]

        # ------------------------------------------------------------------
        # Gültigkeitscheck (Eq. 1a–1c) — vektorisiert
        # (1a): a_f < b_f
        # (1b): a_t < c_t (durch r_start > 0 garantiert); c_t ≤ d_t ≤ b_t (Sortierung)
        # (1c): a_f < c_f ≤ b_f  AND  a_f < d_f ≤ b_f
        # ------------------------------------------------------------------
        valid = (
            (a_f < b_f) &
            (a_f < c_f) & (c_f <= b_f) &
            (a_f < d_f) & (d_f <= b_f)
        )
        if not np.any(valid):
            continue

        c_t = c_t[valid];  c_f = c_f[valid];  c_m = c_m[valid]
        d_t = d_t[valid];  d_f = d_f[valid];  d_m = d_m[valid]
        b_t = b_t[valid];  b_f = b_f[valid];  b_m = b_m[valid]

        # ------------------------------------------------------------------
        # Hash-Berechnung (Section IV-C) — vektorisiert
        # ------------------------------------------------------------------
        dx = b_t - a_t   # > 0 (B in Region, Region rechts von A)
        dy = b_f - a_f   # > 0 (Eq. 1a)
        k  = len(dx)

        params_batch = np.empty((k, 4), dtype=np.float32)
        params_batch[:, 0] = a_t
        params_batch[:, 1] = a_f
        params_batch[:, 2] = b_t
        params_batch[:, 3] = b_f

        hashes_batch = np.empty((k, 4), dtype=np.float32)
        hashes_batch[:, 0] = (c_t - a_t) / dx   # C'_x
        hashes_batch[:, 1] = (c_f - a_f) / dy   # C'_y
        hashes_batch[:, 2] = (d_t - a_t) / dx   # D'_x
        hashes_batch[:, 3] = (d_f - a_f) / dy   # D'_y

        scores_batch = (b_m + c_m + d_m).astype(np.float32)

        all_params_list.append(params_batch)
        all_hashes_list.append(hashes_batch)
        all_scores_list.append(scores_batch)

    if not all_params_list:
        return empty

    return (
        np.concatenate(all_params_list, axis=0),
        np.concatenate(all_hashes_list, axis=0),
        np.concatenate(all_scores_list, axis=0),
    )




def _select_strong_quads(
    params: np.ndarray,
    hashes: np.ndarray,
    scores: np.ndarray,
    quads_per_second: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Wählt die stärksten Quads pro 1-Sekunden-Zeitbin aus.

    Paper Section IV-A: "We pick as many of the strongest quads, such that
    we get close to a maximum of q quads per second of audio. We do this
    by first creating all valid quads for each root point A. Then we bin
    the quads into groups according to the time values of their root points A,
    and select the strong ones ... until we reach the number q quads per second."

    "This grouping method extracts the robust quads from the set of all
    possible quads." — Ergibt eine nahezu uniforme Zeitverteilung.

    Args:
        params: (K, 4) — [a_time, a_freq, b_time, b_freq].
        hashes: (K, 4) — Hash-Vektoren.
        scores: (K,) — Magnitude-Scores.
        quads_per_second: Maximale Quads pro Zeitbin.

    Returns:
        Tuple (selected_params, selected_hashes).
    """
    if len(scores) == 0:
        return np.empty((0, 4), np.float32), np.empty((0, 4), np.float32)

    # 1-Sekunden-Bins basierend auf Root-Point-Zeit
    bin_indices = (params[:, 0] / config.FRAMES_PER_SECOND).astype(np.int32)

    # Sortierung: primär nach Bin (aufsteigend), sekundär nach Score (absteigend)
    sort_order = np.lexsort((-scores, bin_indices))

    # Top-q pro Bin auswählen
    selected = np.zeros(len(scores), dtype=bool)
    prev_bin = -1
    count = 0
    for idx in sort_order:
        current_bin = bin_indices[idx]
        if current_bin != prev_bin:
            prev_bin = current_bin
            count = 0
        if count < quads_per_second:
            selected[idx] = True
            count += 1

    return params[selected], hashes[selected]


def _params_to_quad_records(
    params: np.ndarray,
    hashes: np.ndarray,
    file_id: int,
) -> list[QuadRecord]:
    """Konvertiert Arrays in eine Liste von QuadRecord-Objekten."""
    records = []
    for i in range(len(params)):
        a_t, a_f, b_t, b_f = params[i]
        records.append(QuadRecord(
            root_point=np.array([a_t, a_f], dtype=np.float32),
            quad_size=np.array([b_t - a_t, b_f - a_f], dtype=np.float32),
            hash=hashes[i].copy(),
            file_id=file_id,
        ))
    return records


def _params_to_query_quads(
    params: np.ndarray,
    hashes: np.ndarray,
) -> list[QueryQuad]:
    """Konvertiert Arrays in eine Liste von QueryQuad-Objekten."""
    quads = []
    for i in range(len(params)):
        a_t, a_f, b_t, b_f = params[i]
        quads.append(QueryQuad(
            root_point=np.array([a_t, a_f], dtype=np.float32),
            quad_size=np.array([b_t - a_t, b_f - a_f], dtype=np.float32),
            hash=hashes[i].copy(),
        ))
    return quads
