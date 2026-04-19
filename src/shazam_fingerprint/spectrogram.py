"""
Modul zur STFT-Berechnung und Spektrogramm-Erzeugung.

Berechnet das Magnitude-Spektrogramm in dB-Skala aus einem Audio-Signal.
Die Amplituden werden erst nach der Peak-Extraktion verworfen (vgl. Wang Section 2.1:
"the amplitude component has been eliminated" – erst nach der Constellation Map).
"""

import logging
from typing import NamedTuple

import librosa
import numpy as np

from shazam_fingerprint import config

logger = logging.getLogger(__name__)


class Spectrogram(NamedTuple):
    """Container für ein berechnetes Spektrogramm mit zugehörigen Achsen.

    Attributes:
        magnitude: 2D-Array (n_freqs × n_frames) mit Amplituden in dB.
            Frequenzbereich ist auf [config.MIN_FREQUENCY_HZ, config.MAX_FREQUENCY_HZ]
            eingeschränkt.
        times: 1D-Array mit Zeitstempeln der STFT-Frames in Sekunden.
        frequencies: 1D-Array mit Frequenzwerten der Bins in Hz.
            Länge entspricht magnitude.shape[0].
        freq_bin_min: Ursprünglicher Bin-Index des untersten beibehaltenen Frequenz-Bins
            (vor dem Frequency-Clipping). Wird für die Rückrechnung auf globale
            Bin-Indizes benötigt (z.B. in fingerprint.py).
    """

    magnitude: np.ndarray
    times: np.ndarray
    frequencies: np.ndarray
    freq_bin_min: int


def compute_spectrogram(signal: np.ndarray, sr: int) -> Spectrogram:
    """Berechnet das Magnitude-Spektrogramm eines Audio-Signals.

    Führt eine STFT mit den Parametern aus config.py durch, wandelt das
    Ergebnis in eine dB-Skala um und schneidet das Spektrogramm auf den
    konfigurierten Frequenzbereich [MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ] zu.

    Die Amplitudeninformation bleibt in diesem Modul erhalten, da sie für
    das Dichte-Kriterium der Peak-Extraktion benötigt wird. Erst peak_finder.py
    verwirft sie nach der Lokalisation der Peaks.
    Wang Section 2.1: "Notice that at this point the amplitude component has
    been eliminated."

    Args:
        signal: Mono-Audiosignal als 1D float32-Array (normalisiert auf [-1.0, 1.0]).
        sr: Abtastrate des Signals in Hz. Sollte config.SAMPLE_RATE entsprechen.

    Returns:
        Spectrogram-NamedTuple mit magnitude (dB), times (s), frequencies (Hz)
        und freq_bin_min (globaler Index des ersten Frequenz-Bins).

    Raises:
        ValueError: Wenn signal leer ist oder sr <= 0.
    """
    if signal.size == 0:
        raise ValueError("Signal ist leer.")
    if sr <= 0:
        raise ValueError(f"Ungültige Abtastrate: {sr}")

    logger.debug(
        "STFT: n_fft=%d, hop_length=%d, window='%s', sr=%d Hz",
        config.N_FFT,
        config.HOP_LENGTH,
        config.WINDOW,
        sr,
    )

    stft = librosa.stft(
        y=signal,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        window=config.WINDOW,
        center=True,
    )

    magnitude_full = np.abs(stft)

    # Konvertierung in dB-Skala relativ zum globalen Maximum.
    # ref=np.max normiert so, dass der lauteste Bin 0 dB ist und alle anderen
    # Werte negativ sind. Das macht AMPLITUDE_THRESHOLD_DB direkt interpretierbar
    # als "X dB unterhalb des Peaks der gesamten Aufnahme".
    magnitude_db = librosa.amplitude_to_db(magnitude_full, ref=np.max, top_db=None)

    # Frequenz- und Zeitachsen berechnen
    frequencies_full = librosa.fft_frequencies(sr=sr, n_fft=config.N_FFT)
    times = librosa.frames_to_time(
        np.arange(magnitude_db.shape[1]),
        sr=sr,
        hop_length=config.HOP_LENGTH,
    )

    # Frequenzbereich einschränken auf [MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ].
    # Bins außerhalb dieses Bereichs enthalten entweder DC/Rauschen (zu tief)
    # oder für Musik irrelevante Obertöne (zu hoch).
    bin_min, bin_max = _frequency_bin_range(frequencies_full)

    magnitude = magnitude_db[bin_min:bin_max, :]
    frequencies = frequencies_full[bin_min:bin_max]

    logger.debug(
        "Spektrogramm: %d Freq-Bins (%.0f–%.0f Hz) × %d Frames | %.2f s",
        magnitude.shape[0],
        frequencies[0],
        frequencies[-1],
        magnitude.shape[1],
        times[-1] if times.size > 0 else 0.0,
    )

    return Spectrogram(
        magnitude=magnitude,
        times=times,
        frequencies=frequencies,
        freq_bin_min=bin_min,
    )


def _frequency_bin_range(frequencies: np.ndarray) -> tuple[int, int]:
    """Bestimmt den Bin-Index-Bereich für den konfigurierten Frequenzbereich.

    Args:
        frequencies: 1D-Array mit Frequenzwerten aller Bins in Hz
            (Ausgabe von librosa.fft_frequencies).

    Returns:
        Tuple (bin_min, bin_max): Halboffenes Intervall [bin_min, bin_max),
        sodass frequencies[bin_min:bin_max] ⊆ [MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ].
    """
    bin_min = int(np.searchsorted(frequencies, config.MIN_FREQUENCY_HZ, side="left"))
    bin_max = int(np.searchsorted(frequencies, config.MAX_FREQUENCY_HZ, side="right"))
    return bin_min, bin_max
