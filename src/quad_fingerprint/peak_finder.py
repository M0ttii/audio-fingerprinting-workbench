"""
Peak-Extraktion via Zwei-Filter-Verfahren nach Sonnleitner & Widmer, Section IV-A.

Das Verfahren unterscheidet sich grundlegend von Shazams einfacher Constellation Map:

1. Max-Filter: Identifiziert lokale Maxima im Spektrogramm.
2. Min-Filter (3×3): Eliminiert uniforme Regionen (Stille, Klicks, digitale Töne).
3. Adjacency-Cleanup: Garantiert genau einen Peak pro Max-Filter-Fenster.
4. Parabolische Interpolation: Liefert sub-sample-genaue float32-Koordinaten.

Zwei Parametersätze: Referenz-Audio verwendet große Filter (wenige, robuste Peaks),
Query-Audio verwendet kleinere Filter (dichtere Peaks für höhere Trefferquote bei
Skalierungsverzerrungen). Die Query-Filtergrößen werden gemäß Eq. (2) und (3)
aus den Referenz-Parametern und den Toleranzen EPSILON_T, EPSILON_P berechnet.

Referenz: Sonnleitner & Widmer (2016), Section IV-A.
"""

import logging
from collections import defaultdict

import numpy as np
import scipy.ndimage

from quad_fingerprint import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hauptfunktionen (öffentliches API)
# ---------------------------------------------------------------------------

def extract_peaks(
    magnitude: np.ndarray,
    max_filter_width: int = config.REF_MAX_FILTER_WIDTH,
    max_filter_height: int = config.REF_MAX_FILTER_HEIGHT,
) -> np.ndarray:
    """Extrahiert spektrale Peaks aus einem Magnitude-Spektrogramm.

    Implementiert das Vier-Schritt-Verfahren aus dem Paper (Section IV-A):
    Max-Filter → Min-Filter → Adjacency-Cleanup → Parabolische Interpolation.

    Args:
        magnitude: Magnitude-Spektrogramm in linearer Skala, float32,
            Form (n_bins, n_frames) mit n_bins = Frequenz, n_frames = Zeit.
        max_filter_width: Breite des Max-Filters in STFT-Frames (Zeit-Dimension).
            Standard: config.REF_MAX_FILTER_WIDTH = 151 (Referenz-Audio).
        max_filter_height: Höhe des Max-Filters in Frequenz-Bins.
            Standard: config.REF_MAX_FILTER_HEIGHT = 75 (Referenz-Audio).

    Returns:
        np.ndarray der Form (N, 2) mit float32-Koordinaten [(time, freq), ...],
        sortiert nach time aufsteigend. Koordinaten sind STFT-Frame- und
        Frequenz-Bin-Indizes als float (nach parabolischer Interpolation).
        Paper: "From this point onwards, peaks are represented as single
        precision floating point values." (Section IV-A)

    Raises:
        ValueError: Wenn magnitude leer, nicht 2D oder kleiner als Filter.
    """
    _validate_input(magnitude, max_filter_width, max_filter_height)

    # Schritt 1: Max-Filter — Lokale Maxima identifizieren
    # Paper: "Locations of the spectrogram that are identical to the
    # respective max-filter values are peaks." (Section IV-A)
    max_filtered = apply_max_filter(magnitude, max_filter_height, max_filter_width)
    is_max = (magnitude == max_filtered)

    # Schritt 2: Min-Filter (3×3) — Uniforme Regionen eliminieren
    # Paper: "discard peak candidates if they are detected by both,
    # the min and the max filter." (Section IV-A)
    min_filtered = apply_min_filter(magnitude)
    is_min = (magnitude == min_filtered)

    # Peak-Kandidaten: Max-Filter-Match OHNE Min-Filter-Match
    is_peak = is_max & ~is_min

    # Koordinaten extrahieren: argwhere gibt (freq_bin, time_frame) zurück
    peak_coords = np.argwhere(is_peak)  # Shape (N, 2): [[freq, time], ...]

    if len(peak_coords) == 0:
        logger.debug(
            "Keine Peaks gefunden (Filter: %d×%d)", max_filter_height, max_filter_width
        )
        return np.empty((0, 2), dtype=np.float32)

    n_before_cleanup = len(peak_coords)

    # Schritt 3: Adjacency-Cleanup — Genau ein Peak pro Max-Filter-Fenster
    # Paper: "We group all peaks by magnitude, and search for adjacent peaks
    # in each group. If adjacent peaks are found, we keep the first peak in
    # time and frequency." (Section IV-A)
    peak_coords = adjacency_cleanup(peak_coords, magnitude)

    n_after_cleanup = len(peak_coords)

    if len(peak_coords) == 0:
        logger.debug("Alle Peaks durch Adjacency-Cleanup eliminiert.")
        return np.empty((0, 2), dtype=np.float32)

    # Schritt 4: Parabolische Interpolation → float32-Koordinaten
    # Paper: "parabolic interpolation based on their neighbourhood of
    # 3×3 in spectrogram space" (Section IV-A)
    peaks_float = parabolic_interpolation(peak_coords, magnitude)

    # Konvertierung: (freq, time) → (time, freq) für Ausgabe
    # Paper-Konvention: P = (P_x, P_y) mit P_x = time, P_y = freq
    peaks_out = peaks_float[:, ::-1].copy()  # Swap: [freq, time] → [time, freq]

    # Sortierung nach time aufsteigend
    sort_idx = np.argsort(peaks_out[:, 0])
    peaks_out = peaks_out[sort_idx].astype(np.float32)

    logger.debug(
        "Peaks extrahiert: %d (nach Cleanup: %d entfernt) | Filter: %d×%d",
        len(peaks_out),
        n_before_cleanup - n_after_cleanup,
        max_filter_height,
        max_filter_width,
    )

    return peaks_out


def extract_reference_peaks(magnitude: np.ndarray) -> np.ndarray:
    """Extrahiert Peaks für Referenz-Audio mit Referenz-Filtergrößen.

    Verwendet config.REF_MAX_FILTER_WIDTH (151 Frames) und
    config.REF_MAX_FILTER_HEIGHT (75 Bins), wie im Paper Section V spezifiziert.

    Args:
        magnitude: Magnitude-Spektrogramm (n_bins, n_frames), float32, linear.

    Returns:
        np.ndarray (N, 2), float32, [(time, freq), ...].
    """
    return extract_peaks(
        magnitude,
        max_filter_width=config.REF_MAX_FILTER_WIDTH,
        max_filter_height=config.REF_MAX_FILTER_HEIGHT,
    )


def extract_query_peaks(magnitude: np.ndarray) -> np.ndarray:
    """Extrahiert Peaks für Query-Audio mit kleineren Filtergrößen.

    Verwendet die Query-Filtergrößen aus config.py, die gemäß Eq. (2) und (3)
    aus den Referenz-Parametern und EPSILON_T, EPSILON_P berechnet werden:
        m_w^query = m_w^ref / (1 + ε_t)   [Eq. 2]
        m_h^query = m_h^ref · (1 − ε_p)   [Eq. 3]

    Paper: "we extract peaks from query audio at higher density, by using
    smaller max filter sizes" (Section IV-B)

    Args:
        magnitude: Magnitude-Spektrogramm (n_bins, n_frames), float32, linear.

    Returns:
        np.ndarray (N, 2), float32, [(time, freq), ...].
    """
    return extract_peaks(
        magnitude,
        max_filter_width=config.QUERY_MAX_FILTER_WIDTH,
        max_filter_height=config.QUERY_MAX_FILTER_HEIGHT,
    )


def compute_query_filter_sizes(
    ref_width: int = config.REF_MAX_FILTER_WIDTH,
    ref_height: int = config.REF_MAX_FILTER_HEIGHT,
    epsilon_t: float = config.EPSILON_T,
    epsilon_p: float = config.EPSILON_P,
) -> tuple[int, int]:
    """Berechnet die Query-Filtergrößen aus Referenz-Parametern und Toleranzen.

    Eq. (2): m_w^query = m_w^ref / (1 + ε_t)
    Eq. (3): m_h^query = m_h^ref · (1 − ε_p)

    Args:
        ref_width: Referenz-Max-Filter-Breite in STFT-Frames.
        ref_height: Referenz-Max-Filter-Höhe in Frequenz-Bins.
        epsilon_t: Zeittoleranz (dimensionslos).
        epsilon_p: Frequenztoleranz (dimensionslos).

    Returns:
        Tuple (query_width, query_height) in ganzzahligen Werten.
    """
    query_width = round(ref_width / (1 + epsilon_t))
    query_height = round(ref_height * (1 - epsilon_p))
    return query_width, query_height


# ---------------------------------------------------------------------------
# Einzelschritte (für Testbarkeit öffentlich, aber primär intern verwendet)
# ---------------------------------------------------------------------------

def apply_max_filter(
    magnitude: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Wendet einen 2D-Max-Filter auf das Spektrogramm an.

    Schritt 1 des Peak-Extraktionsverfahrens.
    Paper: "We first apply the max filter to the spectrogram." (Section IV-A)

    Args:
        magnitude: Spektrogramm (n_bins, n_frames).
        height: Filterhöhe in Frequenz-Bins (Achse 0).
        width: Filterbreite in STFT-Frames (Achse 1).

    Returns:
        Max-gefiltertes Spektrogramm, gleiche Form wie Eingabe.
    """
    return scipy.ndimage.maximum_filter(magnitude, size=(height, width))


def apply_min_filter(magnitude: np.ndarray) -> np.ndarray:
    """Wendet den 3×3-Min-Filter auf das Spektrogramm an.

    Schritt 2 des Peak-Extraktionsverfahrens. Dient der Erkennung uniformer
    Regionen. Peaks, bei denen sowohl Max- als auch Min-Filter mit dem
    Spektrogramm übereinstimmen, werden verworfen.
    Paper: "a min filter of size 3×3" (Section IV-A)

    Args:
        magnitude: Spektrogramm (n_bins, n_frames).

    Returns:
        Min-gefiltertes Spektrogramm, gleiche Form wie Eingabe.
    """
    return scipy.ndimage.minimum_filter(
        magnitude, size=(config.MIN_FILTER_HEIGHT, config.MIN_FILTER_WIDTH)
    )


def adjacency_cleanup(
    peak_coords: np.ndarray,
    magnitude: np.ndarray,
) -> np.ndarray:
    """Entfernt benachbarte Peaks gleicher Magnitude (Adjacency-Cleanup).

    Schritt 3: Gruppiert Peaks nach Magnitude und sucht nach benachbarten Peaks
    innerhalb jeder Gruppe (8-Konnektivität). Pro Cluster benachbarter Peaks
    wird nur der erste Peak (nach Zeit, dann Frequenz) behalten.

    Paper: "We group all peaks by magnitude, and search for adjacent peaks in
    each group. If adjacent peaks are found, we keep the first peak in time
    and frequency and delete all other peaks in the current filter window.
    This way we ensure to report exactly one peak per max filter window."
    (Section IV-A)

    Paper: "the cleanup procedure has a quadratic time complexity in size of
    peak magnitude groups, but in practice it seems well applicable to audio
    spectrograms." (Section IV-A)

    Args:
        peak_coords: Peak-Koordinaten (N, 2) als int, [[freq_bin, time_frame], ...].
        magnitude: Spektrogramm (n_bins, n_frames) für Magnitude-Lookup.

    Returns:
        Bereinigte Peak-Koordinaten (M, 2) als int, M ≤ N.
    """
    if len(peak_coords) <= 1:
        return peak_coords

    # Magnitudenwerte an den Peak-Positionen
    magnitudes = magnitude[peak_coords[:, 0], peak_coords[:, 1]]

    # Gruppierung nach Magnitude (exakte Gleichheit für Spektrogrammwerte)
    mag_to_indices: dict[float, list[int]] = defaultdict(list)
    for i, mag_val in enumerate(magnitudes):
        mag_to_indices[float(mag_val)].append(i)

    keep_mask = np.ones(len(peak_coords), dtype=bool)

    for indices in mag_to_indices.values():
        if len(indices) <= 1:
            continue

        group_peaks = peak_coords[indices]
        n = len(indices)

        # Positions-Set für O(1)-Nachbarschafts-Lookup
        pos_to_local: dict[tuple[int, int], int] = {}
        for local_idx in range(n):
            pos = (int(group_peaks[local_idx, 0]), int(group_peaks[local_idx, 1]))
            pos_to_local[pos] = local_idx

        # Union-Find für Connected Components (8-Konnektivität)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # Pfadkompression
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Prüfe 8 Nachbarn jedes Peaks in der Gruppe
        for local_idx in range(n):
            f, t = int(group_peaks[local_idx, 0]), int(group_peaks[local_idx, 1])
            for df in (-1, 0, 1):
                for dt in (-1, 0, 1):
                    if df == 0 and dt == 0:
                        continue
                    neighbor_pos = (f + df, t + dt)
                    neighbor_local = pos_to_local.get(neighbor_pos)
                    if neighbor_local is not None:
                        union(local_idx, neighbor_local)

        # Gruppierung nach Komponente
        components: dict[int, list[int]] = defaultdict(list)
        for local_idx in range(n):
            root = find(local_idx)
            components[root].append(local_idx)

        # Pro Komponente: Behalte nur den ersten Peak (nach time, dann freq)
        for comp_members in components.values():
            if len(comp_members) <= 1:
                continue
            # Sortierung: time (Spalte 1) aufsteigend, dann freq (Spalte 0)
            comp_members.sort(
                key=lambda idx: (group_peaks[idx, 1], group_peaks[idx, 0])
            )
            # Alle außer dem ersten zum Löschen markieren
            for local_idx in comp_members[1:]:
                keep_mask[indices[local_idx]] = False

    return peak_coords[keep_mask]


def parabolic_interpolation(
    peak_coords: np.ndarray,
    magnitude: np.ndarray,
) -> np.ndarray:
    """Verfeinert Peak-Positionen mittels parabolischer Interpolation.

    Schritt 4: Für jeden Peak wird in Zeit- und Frequenz-Richtung unabhängig
    eine Parabel durch die drei Nachbarwerte gelegt. Das Parabel-Maximum
    bestimmt die sub-sample-genaue Position. Falls der interpolierte Offset
    |δ| > 1.0 (außerhalb der 3×3-Nachbarschaft), wird der ursprüngliche
    Integer-Wert für diese Dimension beibehalten.

    Paper: "resulting peak coordinates are now subjected to parabolic
    interpolation based on their neighbourhood of 3×3 in spectrogram space.
    If the interpolated value lies outside the neighbourhood in any dimension,
    the original non-interpolated value is kept." (Section IV-A)

    Standard-Formel für 1D parabolische Interpolation:
        δ = 0.5 · (f_{-1} − f_{+1}) / (f_{-1} − 2·f_0 + f_{+1})

    Args:
        peak_coords: Peak-Koordinaten (N, 2) als int, [[freq_bin, time_frame], ...].
        magnitude: Spektrogramm (n_bins, n_frames).

    Returns:
        Interpolierte Peak-Koordinaten (N, 2) als float64, [[freq, time], ...].
    """
    n_bins, n_frames = magnitude.shape
    peaks_float = peak_coords.astype(np.float64)

    for i in range(len(peak_coords)):
        freq, time = int(peak_coords[i, 0]), int(peak_coords[i, 1])

        # Randpixel: Interpolation nur möglich, wenn vollständige 3×3-Nachbarschaft
        if freq <= 0 or freq >= n_bins - 1 or time <= 0 or time >= n_frames - 1:
            continue

        center = float(magnitude[freq, time])

        # Zeit-Interpolation (horizontale Parabel)
        left = float(magnitude[freq, time - 1])
        right = float(magnitude[freq, time + 1])
        denom_t = left - 2.0 * center + right
        if denom_t != 0.0:
            delta_t = 0.5 * (left - right) / denom_t
            if abs(delta_t) <= 1.0:
                peaks_float[i, 1] = time + delta_t

        # Frequenz-Interpolation (vertikale Parabel)
        below = float(magnitude[freq - 1, time])
        above = float(magnitude[freq + 1, time])
        denom_f = below - 2.0 * center + above
        if denom_f != 0.0:
            delta_f = 0.5 * (below - above) / denom_f
            if abs(delta_f) <= 1.0:
                peaks_float[i, 0] = freq + delta_f

    return peaks_float


# ---------------------------------------------------------------------------
# Validierung
# ---------------------------------------------------------------------------

def _validate_input(
    magnitude: np.ndarray,
    max_filter_width: int,
    max_filter_height: int,
) -> None:
    """Prüft Eingabeparameter für extract_peaks."""
    if magnitude.ndim != 2:
        raise ValueError(
            f"magnitude muss 2D sein (n_bins, n_frames), erhalten: {magnitude.ndim}D"
        )
    if magnitude.size == 0:
        raise ValueError("Leeres Spektrogramm — keine Peak-Extraktion möglich.")
    if max_filter_width < 1 or max_filter_height < 1:
        raise ValueError(
            f"Filtergrößen müssen ≥ 1 sein, erhalten: "
            f"width={max_filter_width}, height={max_filter_height}"
        )
    n_bins, n_frames = magnitude.shape
    if n_bins < config.MIN_FILTER_HEIGHT or n_frames < config.MIN_FILTER_WIDTH:
        raise ValueError(
            f"Spektrogramm zu klein ({n_bins}×{n_frames}) für Min-Filter "
            f"({config.MIN_FILTER_HEIGHT}×{config.MIN_FILTER_WIDTH})."
        )
