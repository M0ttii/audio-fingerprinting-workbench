"""
Visualisierungsmodul für den Shazam Audio-Fingerprinting-Algorithmus.

Erzeugt Plots für die Bachelorarbeit (matplotlib). Alle Funktionen geben eine
matplotlib Figure zurück und können optional als PNG gespeichert werden.
Die Visualisierungen entsprechen den Abbildungen im Whitepaper:
  - Fig. 1A: Spektrogramm           → plot_spectrogram
  - Fig. 1B: Constellation Map      → plot_constellation_map
  - Fig. 1B+C: Spektrogramm + Peaks → plot_spectrogram_with_peaks
  - Fig. 1C: Anchor-Target-Paare    → plot_hash_pairs
  - Fig. 2B / 3B: δt-Histogram      → plot_match_histogram
  - Fig. 2A / 3A: Scatterplot       → plot_scatterplot
"""

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shazam_fingerprint import config
from shazam_fingerprint.spectrogram import Spectrogram

logger = logging.getLogger(__name__)

# Einheitliches Farbschema für die Bachelorarbeit
_CMAP_SPEC = "magma"
_COLOR_PEAK = "#FF4444"
_COLOR_ANCHOR = "#2196F3"
_COLOR_TARGET = "#FF9800"
_COLOR_LINE = "#AAAAAA"
_FIGSIZE_DEFAULT = (10, 4)
_FIGSIZE_SQUARE = (6, 5)
_DPI = 150


def plot_spectrogram(
    spec: Spectrogram,
    title: str = "Spektrogramm",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Visualisiert das Magnitude-Spektrogramm als Heatmap (Wang Fig. 1A).

    Args:
        spec: Spectrogram-NamedTuple aus spectrogram.compute_spectrogram().
        title: Titel des Plots.
        save_path: Optionaler Pfad zum Speichern als PNG. None = nicht speichern.

    Returns:
        matplotlib Figure-Objekt.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE_DEFAULT)

    img = ax.imshow(
        spec.magnitude,
        aspect="auto",
        origin="lower",
        extent=[spec.times[0], spec.times[-1],
                spec.frequencies[0], spec.frequencies[-1]],
        cmap=_CMAP_SPEC,
        vmin=config.AMPLITUDE_THRESHOLD_DB,
        vmax=0,
    )

    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label("Amplitude (dB)", fontsize=10)

    ax.set_xlabel("Zeit (s)", fontsize=11)
    ax.set_ylabel("Frequenz (Hz)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(spec.times[0], spec.times[-1])
    ax.set_ylim(spec.frequencies[0], spec.frequencies[-1])

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_constellation_map(
    peaks: list[tuple[int, int]],
    spec: Spectrogram,
    title: str = "Constellation Map",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Visualisiert die Constellation Map als Scatter-Plot (Wang Fig. 1B).

    Zeigt nur die Peak-Koordinaten auf einem leeren Zeit-Frequenz-Raster,
    ohne das Spektrogramm im Hintergrund.

    Args:
        peaks: Liste von (frequency_bin_global, time_frame)-Tupeln aus peak_finder.
        spec: Spectrogram-NamedTuple für Achsengrenzen und Zeitachse.
        title: Titel des Plots.
        save_path: Optionaler Pfad zum Speichern als PNG.

    Returns:
        matplotlib Figure-Objekt.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE_DEFAULT)

    if peaks:
        freqs_hz, times_s = _peaks_to_axes(peaks, spec)
        ax.scatter(times_s, freqs_hz, s=6, c=_COLOR_PEAK, alpha=0.8,
                   linewidths=0, zorder=3)

    ax.set_facecolor("#111111")
    ax.set_xlabel("Zeit (s)", fontsize=11)
    ax.set_ylabel("Frequenz (Hz)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(spec.times[0], spec.times[-1])
    ax.set_ylim(spec.frequencies[0], spec.frequencies[-1])

    n_peaks = len(peaks)
    duration = spec.times[-1] - spec.times[0]
    density = n_peaks / duration if duration > 0 else 0
    ax.text(0.01, 0.98, f"{n_peaks} Peaks  |  {density:.1f} Peaks/s",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", alpha=0.8))

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_spectrogram_with_peaks(
    spec: Spectrogram,
    peaks: list[tuple[int, int]],
    title: str = "Spektrogramm mit Peaks",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Überlagert das Spektrogramm mit der Constellation Map (Wang Fig. 1B).

    Args:
        spec: Spectrogram-NamedTuple.
        peaks: Liste von (frequency_bin_global, time_frame)-Tupeln.
        title: Titel des Plots.
        save_path: Optionaler Pfad zum Speichern als PNG.

    Returns:
        matplotlib Figure-Objekt.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE_DEFAULT)

    ax.imshow(
        spec.magnitude,
        aspect="auto",
        origin="lower",
        extent=[spec.times[0], spec.times[-1],
                spec.frequencies[0], spec.frequencies[-1]],
        cmap=_CMAP_SPEC,
        vmin=config.AMPLITUDE_THRESHOLD_DB,
        vmax=0,
    )

    if peaks:
        freqs_hz, times_s = _peaks_to_axes(peaks, spec)
        ax.scatter(times_s, freqs_hz, s=12, c=_COLOR_PEAK, alpha=0.9,
                   linewidths=0.5, edgecolors="white", zorder=3,
                   label=f"{len(peaks)} Peaks")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.7)

    ax.set_xlabel("Zeit (s)", fontsize=11)
    ax.set_ylabel("Frequenz (Hz)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(spec.times[0], spec.times[-1])
    ax.set_ylim(spec.frequencies[0], spec.frequencies[-1])

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_hash_pairs(
    peaks: list[tuple[int, int]],
    fingerprints: list[tuple[int, int]],
    spec: Spectrogram,
    max_pairs: int = 200,
    title: str = "Anchor-Target-Paare",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Visualisiert Anchor-Target-Verbindungen der Combinatorial Hashing (Wang Fig. 1C).

    Anchor-Peaks werden blau, Target-Peaks orange dargestellt. Verbindungslinien
    zeigen die erzeugten Hash-Paare. Zur Übersichtlichkeit werden maximal
    max_pairs Paare gezeichnet.

    Args:
        peaks: Liste von (frequency_bin_global, time_frame)-Tupeln.
        fingerprints: Liste von (hash_value, anchor_time)-Tupeln aus fingerprint.py.
            Wird genutzt, um zu bestimmen welche Peaks Anchors sind.
        spec: Spectrogram-NamedTuple für Achsenskalierung.
        max_pairs: Maximale Anzahl dargestellter Paare (Übersichtlichkeit).
        title: Titel des Plots.
        save_path: Optionaler Pfad zum Speichern als PNG.

    Returns:
        matplotlib Figure-Objekt.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE_DEFAULT)
    ax.set_facecolor("#111111")

    if not peaks:
        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        _maybe_save(fig, save_path)
        return fig

    # Peaks indizieren für schnellen Zugriff: time_frame → (freq_bin, idx)
    time_to_peaks: defaultdict[int, list[int]] = defaultdict(list)
    for idx, (_, tf) in enumerate(peaks):
        time_to_peaks[tf].append(idx)

    freqs_full = np.linspace(
        spec.frequencies[0], spec.frequencies[-1],
        config.N_FFT // 2 + 1
    )
    frames_per_sec = (
        (spec.magnitude.shape[1] - 1) / spec.times[-1]
        if spec.times[-1] > 0 else 1.0
    )

    def bin_to_hz(fb: int) -> float:
        return float(np.interp(fb, np.arange(len(freqs_full)), freqs_full))

    def frame_to_sec(tf: int) -> float:
        return tf / frames_per_sec

    # Alle Peaks plotten
    peak_times_s = [frame_to_sec(tf) for _, tf in peaks]
    peak_freqs_hz = [bin_to_hz(fb) for fb, _ in peaks]
    ax.scatter(peak_times_s, peak_freqs_hz, s=6, c=_COLOR_PEAK,
               alpha=0.5, linewidths=0, zorder=2)

    # Anchor-Zeitframes extrahieren
    anchor_times = {t_a for _, t_a in fingerprints}

    # Anchor-Peaks hervorheben
    for fb, tf in peaks:
        if tf in anchor_times:
            ax.scatter(frame_to_sec(tf), bin_to_hz(fb), s=20,
                       c=_COLOR_ANCHOR, alpha=0.9, linewidths=0, zorder=4)

    # Verbindungslinien: Anchor → Target
    # Wir rekonstruieren Paare aus fingerprints (anchor_time) und peaks
    # Die Paare werden durch den Fan-Out begrenzt.
    pairs_drawn = 0
    seen_anchors: set[int] = set()

    # Aufbau: anchor_time_frame → anchor_peak(s)
    anchor_peak_map: dict[int, tuple[int, int]] = {}
    for fb, tf in peaks:
        if tf in anchor_times and tf not in anchor_peak_map:
            anchor_peak_map[tf] = (fb, tf)

    # Für jeden Fingerprint: Suche nächste Peaks in Target Zone
    for _, t_a in fingerprints:
        if pairs_drawn >= max_pairs:
            break
        if t_a in seen_anchors:
            continue
        seen_anchors.add(t_a)

        if t_a not in anchor_peak_map:
            continue
        f_a, _ = anchor_peak_map[t_a]

        # Finde Targets: Peaks in Target Zone
        targets_drawn = 0
        for fb_t, tf_t in peaks:
            if targets_drawn >= config.FAN_OUT:
                break
            dt = tf_t - t_a
            df = fb_t - f_a
            if (config.TARGET_ZONE_T_MIN <= dt <= config.TARGET_ZONE_T_MAX
                    and config.TARGET_ZONE_F_MIN <= df <= config.TARGET_ZONE_F_MAX):
                ax.plot(
                    [frame_to_sec(t_a), frame_to_sec(tf_t)],
                    [bin_to_hz(f_a), bin_to_hz(fb_t)],
                    color=_COLOR_LINE, alpha=0.4, linewidth=0.5, zorder=1,
                )
                ax.scatter(frame_to_sec(tf_t), bin_to_hz(fb_t), s=16,
                           c=_COLOR_TARGET, alpha=0.8, linewidths=0, zorder=3)
                targets_drawn += 1
                pairs_drawn += 1

    # Legende
    from matplotlib.lines import Line2D
    legend_elements = [
        plt.scatter([], [], s=20, c=_COLOR_ANCHOR, label="Anchor"),
        plt.scatter([], [], s=16, c=_COLOR_TARGET, label="Target"),
        Line2D([0], [0], color=_COLOR_LINE, alpha=0.6, label="Hash-Paar"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.7, facecolor="#333333", labelcolor="white")

    ax.set_xlabel("Zeit (s)", fontsize=11)
    ax.set_ylabel("Frequenz (Hz)", fontsize=11)
    ax.set_title(f"{title} ({pairs_drawn} Paare)", fontsize=13, fontweight="bold")
    ax.set_xlim(spec.times[0], spec.times[-1])
    ax.set_ylim(spec.frequencies[0], spec.frequencies[-1])

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_match_histogram(
    delta_t_values: list[int],
    song_id: str,
    expected_peak: int | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Visualisiert das δt-Histogram eines Match-Kandidaten (Wang Fig. 2B / 3B).

    Ein echter Match zeigt einen klaren Histogram-Peak (Wang Fig. 3B).
    Bei Nicht-Match ist das Histogram gleichmäßig verteilt (Wang Fig. 2B).

    Wang Section 2.3: "calculate a histogram of these δt values and scan for a peak."

    Args:
        delta_t_values: Liste von δt-Werten (t_db - t_query) in Frames.
        song_id: Song-ID des Kandidaten (für Beschriftung).
        expected_peak: Optionaler erwarteter Peak-Wert (für Annotation).
        title: Optionaler Titel. Standard: auto-generiert.
        save_path: Optionaler Pfad zum Speichern als PNG.

    Returns:
        matplotlib Figure-Objekt.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)

    if delta_t_values:
        dt_arr = np.array(delta_t_values)

        n_bins = min(200, max(20, len(set(delta_t_values)) // 2))
        counts, bin_edges = np.histogram(dt_arr, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]),
               color="#2196F3", alpha=0.8, edgecolor="none")

        # Peak markieren
        peak_idx = np.argmax(counts)
        peak_val = bin_centers[peak_idx]
        peak_count = counts[peak_idx]
        ax.axvline(peak_val, color="#FF4444", linewidth=1.5,
                   linestyle="--", label=f"Peak: δt={peak_val:.0f}  (Score={peak_count})")

        ax.text(peak_val, peak_count * 1.02, f" Score={peak_count}",
                color="#FF4444", fontsize=9, va="bottom")

        if expected_peak is not None:
            ax.axvline(expected_peak, color="#4CAF50", linewidth=1.5,
                       linestyle=":", label=f"Erwartet: δt={expected_peak}")

        ax.legend(fontsize=9, framealpha=0.8)

    ax.set_xlabel("Zeitoffset δt (Frames)", fontsize=11)
    ax.set_ylabel("Anzahl übereinstimmender Hash-Paare", fontsize=11)
    auto_title = title or f"δt-Histogram: '{song_id}'"
    ax.set_title(auto_title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_scatterplot(
    time_pairs: list[tuple[int, int]],
    song_id: str,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Visualisiert den Scatterplot von Database-Time vs. Sample-Time (Wang Fig. 2A / 3A).

    Bei einem echten Match liegen die Punkte auf einer Diagonalen (Wang Fig. 3A),
    da t_db = t_query + const. Bei keinem Match ist die Punktwolke gleichmäßig
    verteilt (Wang Fig. 2A).

    Wang Section 2.3: "The offset of the query to the database track is evident
    in the diagonal of the scatterplot."

    Args:
        time_pairs: Liste von (t_query, t_db)-Tupeln der Hash-Treffer (in Frames).
        song_id: Song-ID des Kandidaten (für Beschriftung).
        title: Optionaler Titel. Standard: auto-generiert.
        save_path: Optionaler Pfad zum Speichern als PNG.

    Returns:
        matplotlib Figure-Objekt.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)

    if time_pairs:
        t_query = np.array([p[0] for p in time_pairs])
        t_db = np.array([p[1] for p in time_pairs])

        ax.scatter(t_query, t_db, s=8, alpha=0.5, color="#2196F3",
                   linewidths=0)

        # Diagonalen-Fit (bei echtem Match sollte t_db ≈ t_query + offset)
        if len(time_pairs) >= 5:
            delta_t_values = t_db - t_query
            best_offset = int(np.bincount(
                (delta_t_values - delta_t_values.min()).astype(int)
            ).argmax()) + int(delta_t_values.min())
            t_range = np.array([t_query.min(), t_query.max()])
            ax.plot(t_range, t_range + best_offset, color="#FF4444",
                    linewidth=1.5, linestyle="--",
                    label=f"Fit: δt={best_offset} Frames", alpha=0.9)
            ax.legend(fontsize=9, framealpha=0.8)

    ax.set_xlabel("Query-Zeit t_q (Frames)", fontsize=11)
    ax.set_ylabel("Datenbank-Zeit t_db (Frames)", fontsize=11)
    auto_title = title or f"Scatterplot: '{song_id}'"
    ax.set_title(auto_title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ======================================================================
# Hilfsfunktionen
# ======================================================================

def _peaks_to_axes(
    peaks: list[tuple[int, int]],
    spec: Spectrogram,
) -> tuple[np.ndarray, np.ndarray]:
    """Konvertiert (freq_bin_global, time_frame) in (Hz, Sekunden).

    Args:
        peaks: Liste von (frequency_bin_global, time_frame)-Tupeln.
        spec: Spectrogram für Achsenskalierung.

    Returns:
        Tuple (frequencies_hz, times_s) als numpy-Arrays.
    """
    frames_per_sec = (
        (spec.magnitude.shape[1] - 1) / spec.times[-1]
        if spec.times[-1] > 0 else 1.0
    )
    hz_per_bin = config.SAMPLE_RATE / config.N_FFT

    freqs_hz = np.array([fb * hz_per_bin for fb, _ in peaks])
    times_s = np.array([tf / frames_per_sec for _, tf in peaks])
    return freqs_hz, times_s


def _maybe_save(fig: plt.Figure, save_path: str | Path | None) -> None:
    """Speichert eine Figure als PNG, falls save_path angegeben.

    Args:
        fig: matplotlib Figure-Objekt.
        save_path: Zielpfad oder None.
    """
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    logger.info("Plot gespeichert: '%s'", save_path)
