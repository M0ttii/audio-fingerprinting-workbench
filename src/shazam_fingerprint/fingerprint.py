"""
Modul zur Fingerprint-Generierung mittels Combinatorial Hashing.

Implementiert das Kern-Verfahren nach Wang (2003) Section 2.2:
Jeder Peak der Constellation Map wird als Anchor-Point behandelt und mit Peaks
in einer rechteckigen Target Zone gepaart. Aus jedem Paar entsteht ein 32-bit-Hash,
der zusammen mit dem Zeitoffset des Anchors gespeichert wird.

Wang: "Anchor points are chosen, each anchor point having a target zone associated
with it. Each anchor point is sequentially paired with points within its target zone,
each pair yielding two frequency components plus the time difference between the
points — combined to form a hash."

Wang: "each hash can be packed into a 32-bit unsigned integer"
Hash-Layout:  f_anchor[10 bit] | f_target[10 bit] | delta_t[12 bit]  =  32 bit

Wang: "Each hash is also associated with the time offset from the beginning of the
respective file to its anchor point, though the absolute time is not a part of the
hash itself."
"""

import logging

import numpy as np

from shazam_fingerprint import config

logger = logging.getLogger(__name__)

# Abgeleitete Konstanten für Bit-Masken und maximale Werte
_MAX_FREQ_VAL: int = (1 << config.FREQ_BITS) - 1       # 1023 bei 10 Bit
_MAX_DELTA_T_VAL: int = (1 << config.DELTA_T_BITS) - 1  # 4095 bei 12 Bit


def generate_fingerprints(
    peaks: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Erzeugt Fingerprint-Hashes aus einer Constellation Map.

    Für jeden Peak (Anchor) werden die Peaks in der Target Zone gesucht,
    nach Zeitabstand sortiert, und die ersten FAN_OUT davon gepaart.
    Jedes Paar ergibt einen 32-bit-Hash plus den Zeitoffset des Anchors.

    Wang Section 2.2: "fan-out of size F" — begrenzt die Anzahl der
    Hashes pro Anchor und damit den Speicherverbrauch.

    Args:
        peaks: Liste von (frequency_bin, time_frame)-Tupeln aus peak_finder.
            frequency_bin ist ein globaler Bin-Index (0 … N_FFT//2).
            Muss nach time_frame aufsteigend sortiert sein.

    Returns:
        Liste von (hash_value, anchor_time)-Tupeln.
            hash_value: 32-bit unsigned int, gepackt als
                (f_anchor_q << 22) | (f_target_q << 12) | delta_t
            anchor_time: STFT-Frame-Index des Anchor-Peaks.
    """
    if len(peaks) < 2:
        logger.warning("Weniger als 2 Peaks — keine Fingerprints erzeugbar.")
        return []

    # Peaks in numpy-Arrays für schnellen Zugriff
    peak_arr = np.array(peaks, dtype=np.int32)  # Shape: (N, 2)
    freqs = peak_arr[:, 0]
    times = peak_arr[:, 1]
    n_peaks = len(peaks)

    fingerprints: list[tuple[int, int]] = []

    for i in range(n_peaks):
        f_a = int(freqs[i])
        t_a = int(times[i])

        # Target Zone: zeitlich vorwärts, frequenzmäßig symmetrisch um Anchor.
        # Wang Section 2.2 zeigt die Target Zone als Rechteck rechts vom Anchor.
        # Da peaks nach time_frame sortiert sind, suchen wir vorwärts ab i+1.
        targets = _find_targets_in_zone(freqs, times, i, n_peaks, f_a, t_a)

        # Fan-Out limitieren: nur die FAN_OUT nächsten (nach Zeitabstand).
        # targets ist bereits nach delta_t aufsteigend sortiert.
        for f_t, t_t in targets[:config.FAN_OUT]:
            delta_t = t_t - t_a
            hash_value = _compute_hash(f_a, f_t, delta_t)
            fingerprints.append((hash_value, t_a))

    logger.info(
        "Fingerprints: %d Hashes aus %d Peaks (Ø %.1f Hashes/Peak)",
        len(fingerprints),
        n_peaks,
        len(fingerprints) / n_peaks if n_peaks > 0 else 0.0,
    )

    return fingerprints


def _find_targets_in_zone(
    freqs: np.ndarray,
    times: np.ndarray,
    anchor_idx: int,
    n_peaks: int,
    f_a: int,
    t_a: int,
) -> list[tuple[int, int]]:
    """Findet alle Peaks in der Target Zone eines Anchors.

    Die Target Zone ist ein Rechteck im Zeit-Frequenz-Raum, definiert durch:
    - Zeitachse:    t_a + TARGET_ZONE_T_MIN  ≤  t_target  ≤  t_a + TARGET_ZONE_T_MAX
    - Frequenzachse: f_a + TARGET_ZONE_F_MIN  ≤  f_target  ≤  f_a + TARGET_ZONE_F_MAX

    Wang Section 2.2: "Anchor points are chosen, each anchor point having a
    target zone associated with it."

    Args:
        freqs: 1D int-Array aller Peak-Frequenzen (globale Bin-Indizes).
        times: 1D int-Array aller Peak-Zeitframes (aufsteigend sortiert).
        anchor_idx: Index des aktuellen Anchors in freqs/times.
        n_peaks: Gesamtzahl der Peaks.
        f_a: Frequenz-Bin des Anchors.
        t_a: Zeitframe des Anchors.

    Returns:
        Liste von (f_target, t_target)-Tupeln, sortiert nach aufsteigendem
        Zeitabstand (delta_t = t_target - t_a). Maximal so viele Einträge
        wie in der Zone liegen; der FAN_OUT-Schnitt erfolgt im Aufrufer.
    """
    t_min = t_a + config.TARGET_ZONE_T_MIN
    t_max = t_a + config.TARGET_ZONE_T_MAX
    f_min = f_a + config.TARGET_ZONE_F_MIN
    f_max = f_a + config.TARGET_ZONE_F_MAX

    # Da times aufsteigend sortiert ist, können wir ab anchor_idx+1 suchen
    # und bei t > t_max abbrechen (Early Exit).
    targets: list[tuple[int, int]] = []

    for j in range(anchor_idx + 1, n_peaks):
        t_j = int(times[j])

        if t_j > t_max:
            break
        if t_j < t_min:
            continue

        f_j = int(freqs[j])
        if f_min <= f_j <= f_max:
            targets.append((f_j, t_j))

    return targets


def _compute_hash(f_anchor: int, f_target: int, delta_t: int) -> int:
    """Packt ein Anchor-Target-Paar in einen 32-bit unsigned integer Hash.

    Hash-Layout (MSB → LSB):
        [f_anchor_quantized : 10 bit] [f_target_quantized : 10 bit] [delta_t : 12 bit]

    Frequenzen werden auf FREQ_BITS (10 Bit, 0–1023) quantisiert:
        quantized = f_bin * (2^FREQ_BITS - 1) // MAX_FREQ_BIN
    Wang Section 2.2: "each hash can be packed into a 32-bit unsigned integer"

    Args:
        f_anchor: Globaler Frequenz-Bin-Index des Anchors (0 … N_FFT//2).
        f_target: Globaler Frequenz-Bin-Index des Targets (0 … N_FFT//2).
        delta_t: Zeitdifferenz in Frames (t_target - t_anchor, ≥ 1).

    Returns:
        32-bit unsigned integer mit gepacktem Hash.
    """
    f_a_q = _quantize_frequency(f_anchor)
    f_t_q = _quantize_frequency(f_target)
    dt = min(delta_t, _MAX_DELTA_T_VAL)

    return (f_a_q << (config.FREQ_BITS + config.DELTA_T_BITS)) | (f_t_q << config.DELTA_T_BITS) | dt


def _quantize_frequency(freq_bin: int) -> int:
    """Quantisiert einen globalen Frequenz-Bin-Index auf FREQ_BITS.

    Mappt den Wertebereich [0, MAX_FREQ_BIN] linear auf [0, 2^FREQ_BITS - 1].

    Wang Section 2.2 impliziert Frequenz-Quantisierung, da der Hash nur
    begrenzte Bits pro Frequenzkomponente hat.

    Args:
        freq_bin: Globaler Frequenz-Bin-Index (0 … N_FFT//2).

    Returns:
        Quantisierter Wert im Bereich [0, 2^FREQ_BITS - 1].
    """
    return min(freq_bin * _MAX_FREQ_VAL // config.MAX_FREQ_BIN, _MAX_FREQ_VAL)
