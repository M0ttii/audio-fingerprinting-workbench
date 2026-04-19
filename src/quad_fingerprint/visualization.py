"""
Visualisierungsmodul für das Quad-basierte Audio-Fingerprinting.

Erstellt Plots für die Bachelorarbeit (matplotlib). Alle Funktionen:
  - Haben Achsenbeschriftungen und Titel
  - Geben eine matplotlib Figure zurück
  - Speichern optional als PNG (save_path)
  - Zeigen optional inline an (show=True für Jupyter)

Referenz: Sonnleitner & Widmer (2016), Figures 2, 3, 5.
"""

import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from quad_fingerprint import config
from quad_fingerprint.spectrogram import Spectrogram

logger = logging.getLogger(__name__)

# Farben konsistent durch alle Plots
_COLOR_PEAKS = "#555555"
_COLOR_A = "#e74c3c"       # Root-Point A: Rot
_COLOR_B = "#2ecc71"       # B: Grün
_COLOR_C = "#3498db"       # C: Blau
_COLOR_D = "#f39c12"       # D: Orange
_COLOR_REF = "#2980b9"     # Referenz-Peaks: Blau
_COLOR_QUERY = "#e67e22"   # Query-Peaks: Orange
_COLOR_VERIFIED = "#27ae60"      # Verifiziert: Grün
_COLOR_NOT_VERIFIED = "#c0392b"  # Nicht verifiziert: Rot


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------

def _peaks_to_sec_hz(
    peaks: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    hop_length: int = config.HOP_LENGTH,
    n_fft: int = config.N_FFT,
) -> tuple[np.ndarray, np.ndarray]:
    """Wandelt Peak-Koordinaten (Frames, Bins) in (Sekunden, Hz) um.

    Args:
        peaks: (N, 2) float32, [(time_frame, freq_bin), ...].
        sr: Abtastrate in Hz.
        hop_length: STFT-Hopgröße in Samples.
        n_fft: STFT-Fenstergröße in Samples.

    Returns:
        Tuple (times_sec, freqs_hz), je (N,) float64.
    """
    times_sec = peaks[:, 0] * hop_length / sr
    freqs_hz = peaks[:, 1] * sr / n_fft
    return times_sec, freqs_hz


def _save_and_show(
    fig: plt.Figure,
    save_path: str | Path | None,
    show: bool,
) -> None:
    """Speichert Figure als PNG und/oder zeigt sie an.

    Args:
        fig: matplotlib Figure-Objekt.
        save_path: Pfad zur PNG-Ausgabedatei, oder None.
        show: Falls True, wird plt.show() aufgerufen.
    """
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Plot gespeichert: '%s'", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 1. plot_spectrogram
# ---------------------------------------------------------------------------

def plot_spectrogram(
    spectrogram: Spectrogram,
    title: str = "Magnitude-Spektrogramm",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Visualisiert das Magnitude-Spektrogramm als Heatmap.

    Y-Achse: Frequenz in Hz, X-Achse: Zeit in Sekunden.
    Magnitude wird logarithmisch dargestellt (20·log10) für bessere
    visuelle Unterscheidbarkeit, analog zu Shazam-Spektrogrammen.

    Args:
        spectrogram: Spectrogram-NamedTuple mit magnitude, times, frequencies.
        title: Titel des Plots.
        save_path: Optionaler Pfad zum Speichern als PNG.
        show: Falls True, wird der Plot angezeigt (plt.show()).

    Returns:
        matplotlib Figure-Objekt.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    mag = spectrogram.magnitude
    # Logarithmische Skalierung für Darstellung (nicht für Algorithmus!)
    mag_db = 20.0 * np.log10(mag + 1e-9)

    t_min, t_max = spectrogram.times[0], spectrogram.times[-1]
    f_min, f_max = spectrogram.frequencies[0], spectrogram.frequencies[-1]

    img = ax.imshow(
        mag_db,
        aspect="auto",
        origin="lower",
        extent=[t_min, t_max, f_min, f_max],
        cmap="inferno",
        interpolation="nearest",
    )

    plt.colorbar(img, ax=ax, label="Magnitude (dB)")
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Frequenz (Hz)")
    ax.set_title(title)
    fig.tight_layout()

    _save_and_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# 2. plot_peaks
# ---------------------------------------------------------------------------

def plot_peaks(
    peaks: np.ndarray,
    spectrogram_shape: tuple[int, int],
    sr: int = config.SAMPLE_RATE,
    hop_length: int = config.HOP_LENGTH,
    title: str = "Spektrale Peaks (Constellation Map)",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Visualisiert extrahierte Peaks als Scatter-Plot (Constellation Map).

    Analog zur Shazam Constellation Map: Zeit-Frequenz-Raster mit
    extrahierten Peaks als Punkte. Zeigt die Dichte und Verteilung
    der Peaks im Spektrogramm.

    Args:
        peaks: (N, 2) float32, [(time_frame, freq_bin), ...].
        spectrogram_shape: (n_bins, n_frames) des zugehörigen Spektrogramms.
        sr: Abtastrate in Hz.
        hop_length: STFT-Hopgröße in Samples.
        title: Titel des Plots.
        save_path: Optionaler Pfad zum Speichern als PNG.
        show: Falls True, wird der Plot angezeigt.

    Returns:
        matplotlib Figure-Objekt.
    """
    n_bins, n_frames = spectrogram_shape
    t_max = n_frames * hop_length / sr
    f_max = (n_bins - 1) * sr / config.N_FFT

    times_sec, freqs_hz = _peaks_to_sec_hz(peaks, sr, hop_length)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(
        times_sec, freqs_hz,
        s=4, c=_COLOR_PEAKS, alpha=0.7, linewidths=0,
    )
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, f_max)
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Frequenz (Hz)")
    ax.set_title(f"{title}  ({len(peaks)} Peaks)")
    ax.set_facecolor("#f8f8f8")
    fig.tight_layout()

    _save_and_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# 3. plot_spectrogram_with_peaks
# ---------------------------------------------------------------------------

def plot_spectrogram_with_peaks(
    spectrogram: Spectrogram,
    peaks: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    hop_length: int = config.HOP_LENGTH,
    title: str = "Spektrogramm mit Peaks",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Überlagert Peaks auf das Magnitude-Spektrogramm.

    Kombination aus plot_spectrogram und plot_peaks:
    Spektrogramm als Heatmap, Peaks als rote Punkte darüber.

    Args:
        spectrogram: Spectrogram-NamedTuple.
        peaks: (N, 2) float32 Peak-Koordinaten [(time_frame, freq_bin), ...].
        sr: Abtastrate in Hz.
        hop_length: STFT-Hopgröße in Samples.
        title: Titel des Plots.
        save_path: Optionaler Pfad zum Speichern als PNG.
        show: Falls True, wird der Plot angezeigt.

    Returns:
        matplotlib Figure-Objekt.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    mag_db = 20.0 * np.log10(spectrogram.magnitude + 1e-9)
    t_min, t_max = spectrogram.times[0], spectrogram.times[-1]
    f_min, f_max = spectrogram.frequencies[0], spectrogram.frequencies[-1]

    img = ax.imshow(
        mag_db,
        aspect="auto",
        origin="lower",
        extent=[t_min, t_max, f_min, f_max],
        cmap="inferno",
        interpolation="nearest",
    )
    plt.colorbar(img, ax=ax, label="Magnitude (dB)")

    times_sec, freqs_hz = _peaks_to_sec_hz(peaks, sr, hop_length)
    ax.scatter(
        times_sec, freqs_hz,
        s=8, c="#ff3333", alpha=0.85,
        linewidths=0, label=f"Peaks ({len(peaks)})",
    )

    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Frequenz (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", markerscale=2)
    fig.tight_layout()

    _save_and_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# 4. plot_quad
# ---------------------------------------------------------------------------

def plot_quad(
    peaks: np.ndarray,
    quad_abcd: tuple[int, int, int, int],
    spectrogram_shape: tuple[int, int],
    sr: int = config.SAMPLE_RATE,
    hop_length: int = config.HOP_LENGTH,
    title: str = "Quad-Visualisierung (analog Paper Fig. 2)",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Visualisiert einen einzelnen Quad mit A, B, C, D und B-D-Box.

    Zeigt Root-Point A (rot) und die drei Punkte B (grün), C (blau),
    D (orange). Das achsenparallele Rechteck, das von B und D aufgespannt
    wird (B-D-Box), wird als Normierungsrahmen gezeichnet.
    Analog zu Paper Fig. 2.

    Die Gültigkeitsbedingung (Eq. 1a–1c) erfordert, dass C innerhalb
    der B-D-Box liegt.

    Args:
        peaks: (N, 2) float32 alle Peaks [(time_frame, freq_bin), ...].
        quad_abcd: Tuple von 4 Indizes (i_A, i_B, i_C, i_D) in peaks.
        spectrogram_shape: (n_bins, n_frames) für Achsengrenzen.
        sr: Abtastrate in Hz.
        hop_length: STFT-Hopgröße in Samples.
        title: Titel des Plots.
        save_path: Optionaler Pfad zum Speichern als PNG.
        show: Falls True, wird der Plot angezeigt.

    Returns:
        matplotlib Figure-Objekt.
    """
    i_a, i_b, i_c, i_d = quad_abcd
    n_bins, n_frames = spectrogram_shape
    t_max = n_frames * hop_length / sr
    f_max = (n_bins - 1) * sr / config.N_FFT

    all_times, all_freqs = _peaks_to_sec_hz(peaks, sr, hop_length)

    a_t, a_f = all_times[i_a], all_freqs[i_a]
    b_t, b_f = all_times[i_b], all_freqs[i_b]
    c_t, c_f = all_times[i_c], all_freqs[i_c]
    d_t, d_f = all_times[i_d], all_freqs[i_d]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Alle Hintergrund-Peaks (grau, klein)
    ax.scatter(all_times, all_freqs, s=3, c=_COLOR_PEAKS, alpha=0.3,
               linewidths=0, label="Alle Peaks")

    # B-D-Box (achsenparalleles Rechteck)
    box_t_min = min(b_t, d_t)
    box_f_min = min(b_f, d_f)
    box_w = abs(d_t - b_t)
    box_h = abs(d_f - b_f)
    rect = mpatches.FancyBboxPatch(
        (box_t_min, box_f_min), box_w, box_h,
        boxstyle="square,pad=0",
        linewidth=1.5, edgecolor="#7f8c8d", facecolor="#ecf0f1", alpha=0.4,
        label="B-D-Box (Normierung)",
    )
    ax.add_patch(rect)

    # Verbindungslinien A→B, A→D
    ax.plot([a_t, b_t], [a_f, b_f], "--", color="#95a5a6", linewidth=1.2, alpha=0.7)
    ax.plot([a_t, d_t], [a_f, d_f], "--", color="#95a5a6", linewidth=1.2, alpha=0.7)

    # Quad-Punkte hervorheben
    marker_size = 100
    ax.scatter([a_t], [a_f], s=marker_size, c=_COLOR_A, marker="*",
               zorder=5, label="A (Root-Point)")
    ax.scatter([b_t], [b_f], s=marker_size, c=_COLOR_B, marker="s",
               zorder=5, label="B")
    ax.scatter([c_t], [c_f], s=marker_size, c=_COLOR_C, marker="^",
               zorder=5, label="C")
    ax.scatter([d_t], [d_f], s=marker_size, c=_COLOR_D, marker="D",
               zorder=5, label="D")

    ax.set_xlim(0, t_max)
    ax.set_ylim(0, f_max)
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Frequenz (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    _save_and_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# 5. plot_quad_hash_space
# ---------------------------------------------------------------------------

def plot_quad_hash_space(
    hashes: np.ndarray,
    title: str = "Hash-Raum: C'_x vs C'_y (2D-Projektion)",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Visualisiert die Hash-Verteilung im 2D-Unterraum (C'_x vs C'_y).

    Zeigt die Verteilung aller Hash-Vektoren im Einheitsquadrat [0,1]^2
    für die erste Projektion (C'_x, C'_y). Demonstriert den Unterschied
    zu diskreten Shazam-Hashes: kontinuierliche Verteilung statt
    quantisierter Gitterpunkte.

    Paper Section IV-C: Hash = (C'_x, C'_y, D'_x, D'_y) in [0,1]^4.

    Args:
        hashes: (N, 4) float32 Hash-Vektoren (C'_x, C'_y, D'_x, D'_y).
        title: Titel des Plots.
        save_path: Optionaler Pfad zum Speichern als PNG.
        show: Falls True, wird der Plot angezeigt.

    Returns:
        matplotlib Figure-Objekt.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linke Seite: C'_x vs C'_y
    ax_c = axes[0]
    ax_c.scatter(
        hashes[:, 0], hashes[:, 1],
        s=1, c=_COLOR_REF, alpha=0.3, linewidths=0,
    )
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.set_xlabel("C'_x")
    ax_c.set_ylabel("C'_y")
    ax_c.set_title("C'-Projektion")
    ax_c.set_aspect("equal")
    ax_c.axvline(
        x=config.HASH_CX_MIN, color="#e74c3c", linewidth=1.2, linestyle="--",
        label=f"Subspace-Grenze (C'_x > {config.HASH_CX_MIN:.3f})",
    )
    ax_c.legend(fontsize=8)

    # Rechte Seite: D'_x vs D'_y
    ax_d = axes[1]
    ax_d.scatter(
        hashes[:, 2], hashes[:, 3],
        s=1, c=_COLOR_QUERY, alpha=0.3, linewidths=0,
    )
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.set_xlabel("D'_x")
    ax_d.set_ylabel("D'_y")
    ax_d.set_title("D'-Projektion")
    ax_d.set_aspect("equal")

    fig.suptitle(f"{title}  (N = {len(hashes):,} Hashes)")
    fig.tight_layout()

    _save_and_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# 6. plot_verification
# ---------------------------------------------------------------------------

def plot_verification(
    ref_peaks: np.ndarray,
    query_peaks: np.ndarray,
    root_ref: np.ndarray,
    root_query: np.ndarray,
    s_time: float,
    s_freq: float,
    verified_mask: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    hop_length: int = config.HOP_LENGTH,
    title: str = "Verifikation einer Match-Hypothese (analog Paper Fig. 3)",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Visualisiert den Verifikationsprozess für eine Match-Hypothese.

    Zeigt Referenz-Peaks und transformierte Query-Peaks im gemeinsamen
    Zeit-Frequenz-Raum. Verifizierte Peaks werden als Kreuz (×) dargestellt,
    nicht verifizierte als Plus (+). Toleranz-Rechtecke um Referenz-Peaks
    zeigen das Ausrichtungsfenster (analog zu Paper Fig. 3).

    Transformation Query → Referenz-Raum:
        ref_time = root_ref[0] + (query_time - root_query[0]) * s_time
        ref_freq = root_ref[1] + (query_freq - root_query[1]) * s_freq

    Args:
        ref_peaks: (N, 2) float32 Referenz-Peaks [(time_frame, freq_bin), ...].
        query_peaks: (M, 2) float32 Query-Peaks (im Query-Koordinatensystem).
        root_ref: (2,) float32 Root-Point A im Referenz-Koordinatensystem.
        root_query: (2,) float32 Root-Point A im Query-Koordinatensystem.
        s_time: Zeitlicher Skalierungsfaktor (s_time = ref_S_x / query_S_x).
        s_freq: Frequenz-Skalierungsfaktor (s_freq = ref_S_y / query_S_y).
        verified_mask: (N,) bool-Array, True = Referenz-Peak verifiziert.
        sr: Abtastrate in Hz.
        hop_length: STFT-Hopgröße in Samples.
        title: Titel des Plots.
        save_path: Optionaler Pfad zum Speichern als PNG.
        show: Falls True, wird der Plot angezeigt.

    Returns:
        matplotlib Figure-Objekt.
    """
    # Query-Peaks in Referenz-Koordinatenraum transformieren
    query_transformed = np.empty_like(query_peaks)
    query_transformed[:, 0] = root_ref[0] + (query_peaks[:, 0] - root_query[0]) * s_time
    query_transformed[:, 1] = root_ref[1] + (query_peaks[:, 1] - root_query[1]) * s_freq

    # Frames → Sekunden / Hz
    ref_t, ref_f = _peaks_to_sec_hz(ref_peaks, sr, hop_length)
    q_t, q_f = _peaks_to_sec_hz(query_transformed, sr, hop_length)

    # Toleranzfenster in Sekunden / Hz
    tol_time_s = config.VERIFICATION_TIME_TOLERANCE * hop_length / sr
    tol_freq_hz = config.VERIFICATION_FREQ_TOLERANCE * sr / config.N_FFT

    fig, ax = plt.subplots(figsize=(12, 6))

    # Toleranz-Rechtecke um Referenz-Peaks
    for t, f, verified in zip(ref_t, ref_f, verified_mask):
        color = _COLOR_VERIFIED if verified else _COLOR_NOT_VERIFIED
        rect = mpatches.Rectangle(
            (t - tol_time_s / 2, f - tol_freq_hz / 2),
            tol_time_s, tol_freq_hz,
            linewidth=0.8, edgecolor=color, facecolor=color, alpha=0.15,
        )
        ax.add_patch(rect)

    # Referenz-Peaks (verifiziert / nicht verifiziert)
    verified_idx = verified_mask.astype(bool)
    if verified_idx.any():
        ax.scatter(
            ref_t[verified_idx], ref_f[verified_idx],
            s=60, c=_COLOR_VERIFIED, marker="x", linewidths=1.5,
            zorder=4, label=f"Verifiziert ({verified_idx.sum()})",
        )
    if (~verified_idx).any():
        ax.scatter(
            ref_t[~verified_idx], ref_f[~verified_idx],
            s=60, c=_COLOR_NOT_VERIFIED, marker="+", linewidths=1.5,
            zorder=4, label=f"Nicht verifiziert ({(~verified_idx).sum()})",
        )

    # Transformierte Query-Peaks
    ax.scatter(
        q_t, q_f,
        s=25, c=_COLOR_QUERY, alpha=0.6, linewidths=0,
        zorder=3, label=f"Query (transformiert, {len(q_t)})",
    )

    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Frequenz (Hz)")
    ax.set_title(
        f"{title}\n"
        f"s_time={s_time:.3f}, s_freq={s_freq:.3f} | "
        f"Score={verified_idx.sum()}/{len(verified_mask)}"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    _save_and_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# 7. plot_scale_robustness
# ---------------------------------------------------------------------------

def plot_scale_robustness(
    results_by_scale: dict[str, dict[float, float]],
    title: str = "Skalen-Robustheit (analog Paper Fig. 5)",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Liniendiagramm der Erkennungsrate über Skalierungsfaktoren.

    X-Achse: Skalierungsfaktor (70%–130%), Y-Achse: Erkennungsrate.
    Jeder Distortion-Typ (tempo, pitch, speed) erscheint als eigene Kurve.
    Analog zu Paper Fig. 5: zeigt die Robustheit des Algorithmus gegenüber
    Tempo-, Pitch- und Speed-Änderungen.

    Args:
        results_by_scale: Dict {distortion_type → {scale_level → recognition_rate}}.
            Direkte Ausgabe von evaluate.recognition_by_scale_factor().
        title: Titel des Plots.
        save_path: Optionaler Pfad zum Speichern als PNG.
        show: Falls True, wird der Plot angezeigt.

    Returns:
        matplotlib Figure-Objekt.
    """
    # Farbpalette und Linienstile für bis zu 5 Distortion-Typen
    style_map = {
        "tempo": {"color": "#2980b9", "marker": "o", "linestyle": "-"},
        "pitch": {"color": "#e74c3c", "marker": "s", "linestyle": "--"},
        "speed": {"color": "#27ae60", "marker": "^", "linestyle": "-."},
        "noise": {"color": "#8e44ad", "marker": "D", "linestyle": ":"},
    }
    default_colors = ["#f39c12", "#16a085", "#c0392b", "#2c3e50"]

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (dist_type, level_rates) in enumerate(sorted(results_by_scale.items())):
        levels = sorted(level_rates.keys())
        rates = [level_rates[l] * 100.0 for l in levels]  # → Prozent
        x = [l * 100.0 for l in levels]  # → Prozent-Achse (80 statt 0.80)

        style = style_map.get(dist_type, {
            "color": default_colors[i % len(default_colors)],
            "marker": "P",
            "linestyle": "-",
        })
        ax.plot(
            x, rates,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=6,
            label=dist_type.capitalize(),
        )

    # 100%-Referenzlinie
    ax.axvline(x=100, color="#95a5a6", linewidth=1.0, linestyle=":", alpha=0.8)

    ax.set_xlim(65, 135)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Skalierungsfaktor (%)")
    ax.set_ylabel("Erkennungsrate (%)")
    ax.set_title(title)
    ax.set_xticks(range(70, 135, 5))
    ax.legend(loc="lower center", ncol=max(1, len(results_by_scale)))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    _save_and_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# 8. plot_comparison_with_shazam
# ---------------------------------------------------------------------------

def plot_comparison_with_shazam(
    shazam_results: dict[str, float],
    quad_results: dict[str, float],
    title: str = "Erkennungsraten-Vergleich: Quad vs. Shazam",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Balkendiagramm: Direkter Vergleich Quad- vs. Shazam-Erkennungsraten.

    Für jeden Distortion-Typ wird je ein Balken für Shazam und Quad
    nebeneinander dargestellt. Ermöglicht direkten Vergleich der Robustheit
    beider Algorithmen aus der Bachelorarbeit.

    Args:
        shazam_results: Dict {distortion_type → recognition_rate (0.0–1.0)}.
        quad_results: Dict {distortion_type → recognition_rate (0.0–1.0)}.
        title: Titel des Plots.
        save_path: Optionaler Pfad zum Speichern als PNG.
        show: Falls True, wird der Plot angezeigt.

    Returns:
        matplotlib Figure-Objekt.
    """
    # Gemeinsame Distortion-Typen, sortiert
    all_types = sorted(set(shazam_results) | set(quad_results))

    shazam_rates = [shazam_results.get(t, 0.0) * 100.0 for t in all_types]
    quad_rates = [quad_results.get(t, 0.0) * 100.0 for t in all_types]

    x = np.arange(len(all_types))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(all_types) * 1.5), 6))

    bars_shazam = ax.bar(
        x - bar_width / 2, shazam_rates,
        width=bar_width,
        color="#3498db", alpha=0.85,
        label="Shazam",
    )
    bars_quad = ax.bar(
        x + bar_width / 2, quad_rates,
        width=bar_width,
        color="#e74c3c", alpha=0.85,
        label="Quad (Sonnleitner & Widmer)",
    )

    # Werte über Balken annotieren
    for bar in bars_shazam:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8,
            )
    for bar in bars_quad:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in all_types])
    ax.set_ylim(0, 115)
    ax.set_xlabel("Distortion-Typ")
    ax.set_ylabel("Erkennungsrate (%)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    _save_and_show(fig, save_path, show)
    return fig
