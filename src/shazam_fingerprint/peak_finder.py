"""
Modul zur Peak-Extraktion aus einem Spektrogramm (Constellation Map).

Implementiert den ersten Schritt des Shazam-Algorithmus nach Wang (2003) Section 2.1:
Lokale Maxima im Spektrogramm werden als zeit-frequenz-Landmarken identifiziert.
Die resultierende Constellation Map enthält nur die Koordinaten (frequency_bin, time_frame)
— die Amplitudeninformation wird bewusst verworfen.

Wang: "Notice that at this point the amplitude component has been eliminated,
leaving only a set of time-frequency points. This representation of the audio
signal is very sparse, occupying only a small fraction of all possible
time-frequency points."
"""

import logging

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from shazam_fingerprint import config
from shazam_fingerprint.spectrogram import Spectrogram

logger = logging.getLogger(__name__)


def find_peaks(spec: Spectrogram) -> list[tuple[int, int]]:
    """Extrahiert lokale Maxima aus dem Spektrogramm (Constellation Map).

    Algorithmus in drei Schritten:
    1. Lokale-Maxima-Detektion via scipy.ndimage.maximum_filter().
       Ein Punkt (f, t) ist Peak, wenn er strikt größer ist als alle Nachbarn
       im konfigurierten Rechteck.
       Wang Section 2.1: "A time-frequency point is a candidate peak if it has a
       higher energy content than all its neighbors in a region centered around
       the point."

    2. Amplitude-Threshold: Punkte unterhalb von AMPLITUDE_THRESHOLD_DB werden
       verworfen, da sie wahrscheinlich Hintergrundrauschen repräsentieren.
       Wang Section 2.1: "the highest amplitude peaks are most likely to survive
       the distortions to which the audio has been subjected."

    3. Dichte-Kriterium: Pro Zeitsegment (1 Sekunde) werden maximal
       MAX_PEAKS_PER_SECOND Peaks behalten, ausgewählt nach absteigender Amplitude.
       Wang Section 2.1: "Candidate peaks are chosen according to a density
       criterion in order to assure that the time-frequency strip for the audio
       file has reasonably uniform coverage."

    Die zurückgegebenen Koordinaten verwenden globale Bin-Indizes (bezogen auf
    das volle FFT-Spektrum mit N_FFT//2+1 Bins), sodass fingerprint.py korrekt
    auf FREQ_BITS quantisieren kann. Der time_frame-Index bezieht sich auf den
    STFT-Frame-Index (ab 0).

    Args:
        spec: Spectrogram-NamedTuple aus spectrogram.compute_spectrogram().
            spec.magnitude enthält dB-Werte (0 dB = Maximum der Aufnahme).

    Returns:
        Liste von (frequency_bin, time_frame)-Tupeln, sortiert aufsteigend
        nach time_frame. frequency_bin ist der globale Bin-Index im vollen
        FFT-Spektrum (d.h. spec.freq_bin_min wurde hinzuaddiert).
        KEINE Amplituden – Wang: "the amplitude component has been eliminated."

    Raises:
        ValueError: Wenn das Spektrogramm leer ist.
    """
    magnitude = spec.magnitude  # Shape: (n_freqs, n_frames)

    if magnitude.size == 0:
        raise ValueError("Spektrogramm ist leer.")

   # --- Schritt 1: Lokale Maxima via maximum_filter ---
    peak_mask = _detect_local_maxima(magnitude)

    # --- Schritt 2: Amplitude-Threshold ---
    # Dynamischer Threshold: 40 dB unterhalb des Spektrogramm-Maximums.
    # Verhindert dass ein fixer Threshold zu viele Peaks bei leisen Aufnahmen
    # herausfiltert. Wang: "highest amplitude peaks are most likely to survive."
    dynamic_threshold = magnitude.max() - 40.0
    effective_threshold = max(dynamic_threshold, config.AMPLITUDE_THRESHOLD_DB)
    peak_mask &= magnitude >= effective_threshold

    logger.debug(
        "Threshold: dynamisch=%.1f dB, config=%.1f dB, effektiv=%.1f dB",
        dynamic_threshold, config.AMPLITUDE_THRESHOLD_DB, effective_threshold
    )

    # --- Schritt 3: Dichte-Kriterium ---
    peaks = _apply_density_criterion(magnitude, peak_mask, spec.times)

    # Lokale Frequenz-Indizes auf globale Bin-Indizes umrechnen
    peaks = [(f + spec.freq_bin_min, t) for f, t in peaks]

    logger.info(
        "Peak-Extraktion: %d Peaks (%.1f Peaks/s) aus %d×%d Spektrogramm",
        len(peaks),
        len(peaks) / (spec.times[-1] if spec.times[-1] > 0 else 1.0),
        magnitude.shape[0],
        magnitude.shape[1],
    )

    return peaks


def _detect_local_maxima(magnitude: np.ndarray) -> np.ndarray:
    """Findet alle lokalen Maxima im Spektrogramm mittels maximum_filter.

    Ein Punkt ist ein lokales Maximum, wenn sein Wert dem Maximum in seiner
    Nachbarschaft entspricht UND er strikt größer als das umliegende Minimum
    ist (verhindert Plateaus in stillen Bereichen).

    Wang Section 2.1: "A time-frequency point is a candidate peak if it has a
    higher energy content than all its neighbors in a region centered around
    the point."

    Args:
        magnitude: 2D-Array (n_freqs × n_frames) in dB-Skala.

    Returns:
        Boolesche Maske gleicher Shape. True = lokales Maximum.
    """
    # Nachbarschaftsgröße: (2*SIZE+1) entlang jeder Achse
    neighborhood_size = (
        2 * config.PEAK_NEIGHBORHOOD_SIZE_FREQ + 1,
        2 * config.PEAK_NEIGHBORHOOD_SIZE_TIME + 1,
    )

    local_max = maximum_filter(magnitude, size=neighborhood_size, mode="constant")
    local_min = minimum_filter(magnitude, size=neighborhood_size, mode="constant")

    # Punkt ist Peak, wenn er dem lokalen Maximum entspricht UND strikt über
    # dem lokalen Minimum liegt. Die zweite Bedingung schließt Plateaus aus
    # (z.B. stille Bereiche, in denen alle Werte identisch sind).
    peak_mask = (magnitude == local_max) & (magnitude > local_min)

    return peak_mask


def _apply_density_criterion(
    magnitude: np.ndarray,
    peak_mask: np.ndarray,
    times: np.ndarray,
) -> list[tuple[int, int]]:
    """Wendet das Dichte-Kriterium an: max MAX_PEAKS_PER_SECOND pro Sekunde.

    Das Spektrogramm wird in Zeitsegmente von je 1 Sekunde unterteilt.
    In jedem Segment werden die Peaks nach absteigender Amplitude sortiert
    und nur die stärksten MAX_PEAKS_PER_SECOND behalten. Das garantiert
    räumlich gleichmäßige Abdeckung ("uniform coverage").

    Wang Section 2.1: "Candidate peaks are chosen according to a density
    criterion in order to assure that the time-frequency strip for the audio
    file has reasonably uniform coverage."

    Args:
        magnitude: 2D-Array (n_freqs × n_frames) in dB-Skala.
        peak_mask: Boolesche Maske der lokalen Maxima.
        times: 1D-Array mit Zeitstempeln der STFT-Frames in Sekunden.

    Returns:
        Liste von (frequency_bin_local, time_frame)-Tupeln, sortiert nach
        time_frame. Frequenz-Indizes sind lokal (bezogen auf das gecropte
        Spektrogramm, noch ohne freq_bin_min-Offset).
    """
    if not np.any(peak_mask):
        return []

    # Alle Kandidaten-Peaks extrahieren: (freq_idx, time_idx)
    freq_indices, time_indices = np.nonzero(peak_mask)
    amplitudes = magnitude[freq_indices, time_indices]

    total_duration = times[-1] if times.size > 0 else 0.0
    n_frames = magnitude.shape[1]

    # Berechne Frames pro Sekunde für die Segmentierung
    if total_duration > 0 and n_frames > 1:
        frames_per_sec = (n_frames - 1) / total_duration
    else:
        frames_per_sec = config.SAMPLE_RATE / config.HOP_LENGTH

    # Segment-Zuordnung: Jeder Frame gehört zu einem 1-Sekunden-Segment
    segment_ids = (time_indices / frames_per_sec).astype(int)
    n_segments = segment_ids.max() + 1

    # Indizes nach absteigender Amplitude sortieren (stärkste zuerst)
    sorted_order = np.argsort(-amplitudes)

    selected: list[tuple[int, int]] = []
    segment_counts = np.zeros(n_segments, dtype=int)

    for idx in sorted_order:
        seg = segment_ids[idx]
        if segment_counts[seg] < config.MAX_PEAKS_PER_SECOND:
            segment_counts[seg] += 1
            selected.append((int(freq_indices[idx]), int(time_indices[idx])))

    # Nach time_frame sortieren (aufsteigend)
    selected.sort(key=lambda p: p[1])

    return selected
