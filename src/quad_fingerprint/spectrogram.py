"""
Modul zur STFT-Spektrogramm-Berechnung.

Berechnet das Magnitude-Spektrogramm in LINEARER Skala (kein dB!).
Das ist ein kritischer Unterschied zum Shazam-Modul, das logarithmische
Magnitudewerte verwendet.

Paper: "We compute the STFT magnitude spectrogram using a Hann-window of size
1024 samples (128 ms) and a hop size of 32 samples (4 ms), discarding the
phases." (Sonnleitner & Widmer, Section IV)
"""

import logging
from typing import NamedTuple

import numpy as np
import scipy.signal

from quad_fingerprint import config

logger = logging.getLogger(__name__)


class Spectrogram(NamedTuple):
    """Ergebnis der STFT-Berechnung.

    Attributes:
        magnitude: Magnitude-Spektrogramm in linearer Skala, float32,
            Form (n_bins, n_frames) mit n_bins = N_FFT // 2 + 1 = 513.
        times: Zeitachse der STFT-Frames in Sekunden, Form (n_frames,).
        frequencies: Frequenzachse der Bins in Hz, Form (n_bins,).
    """

    magnitude: np.ndarray   # float32, (n_bins, n_frames), lineare Skala
    times: np.ndarray       # float64, (n_frames,), Sekunden
    frequencies: np.ndarray # float64, (n_bins,), Hz


def compute_spectrogram(signal: np.ndarray, sr: int = config.SAMPLE_RATE) -> Spectrogram:
    """Berechnet das STFT-Magnitude-Spektrogramm eines Audiosignals.

    Verwendet ein Hann-Fenster der Größe N_FFT mit Hop-Länge HOP_LENGTH,
    wie im Paper spezifiziert. Die Phasen werden verworfen — nur die Magnitude
    wird in LINEARER Skala zurückgegeben.

    Args:
        signal: Mono-Audiosignal als float32-Array der Form (N,).
            Muss auf sr Hz abgetastet sein (Standard: config.SAMPLE_RATE = 8000 Hz).
        sr: Abtastrate des Signals in Hz. Muss config.SAMPLE_RATE entsprechen,
            wird für Achsenberechnung und Validierung verwendet.

    Returns:
        Spectrogram-NamedTuple mit:
            - magnitude: float32-Array (n_bins, n_frames) in linearer Skala.
                n_bins = N_FFT // 2 + 1 = 513 (einseitiges Spektrum).
            - times: float64-Array (n_frames,) mit Zeitwerten in Sekunden.
            - frequencies: float64-Array (n_bins,) mit Frequenzwerten in Hz.

    Raises:
        ValueError: Wenn sr nicht config.SAMPLE_RATE entspricht oder das
            Signal leer ist.
    """
    if sr != config.SAMPLE_RATE:
        raise ValueError(
            f"Abtastrate {sr} Hz stimmt nicht mit config.SAMPLE_RATE "
            f"({config.SAMPLE_RATE} Hz) überein. Bitte zuerst resamplen."
        )
    if signal.ndim != 1:
        raise ValueError(
            f"Signal muss 1-dimensional sein (Mono), erhalten: {signal.ndim}D"
        )
    if len(signal) == 0:
        raise ValueError("Leeres Signal — STFT nicht berechenbar.")
    if len(signal) < config.N_FFT:
        raise ValueError(
            f"Signal zu kurz ({len(signal)} Samples) für N_FFT={config.N_FFT}. "
            f"Mindestlänge: {config.N_FFT} Samples."
        )

    # Hann-Fenster der Größe N_FFT
    # Paper: "Hann-window of size 1024 samples" (Section IV)
    window = scipy.signal.get_window("hann", config.N_FFT)

    # STFT-Berechnung via scipy.signal.stft
    # - nperseg = N_FFT = 1024 Samples
    # - noverlap = N_FFT - HOP_LENGTH = 992 Samples (Hop = 32 Samples = 4 ms)
    # - boundary=None: kein Zero-Padding am Rand (kein künstlicher erster/letzter Frame)
    # - padded=False: Signal wird nicht auf Vielfaches von HOP_LENGTH aufgefüllt
    frequencies, times, stft_matrix = scipy.signal.stft(
        signal.astype(np.float32),
        fs=sr,
        window=window,
        nperseg=config.N_FFT,
        noverlap=config.N_FFT - config.HOP_LENGTH,
        boundary=None,
        padded=False,
    )

    # Magnitude: |STFT| — Phasen werden verworfen
    # Paper: "STFT magnitude spectrogram ... discarding the phases" (Section IV)
    # WICHTIG: Lineare Skala, KEIN 20*log10(...) / dB-Konvertierung!
    magnitude = np.abs(stft_matrix).astype(np.float32)

    logger.debug(
        "Spektrogramm: %d Bins × %d Frames | %.3f s | SR=%d Hz",
        magnitude.shape[0],
        magnitude.shape[1],
        len(signal) / sr,
        sr,
    )

    return Spectrogram(
        magnitude=magnitude,
        times=times.astype(np.float64),
        frequencies=frequencies.astype(np.float64),
    )
