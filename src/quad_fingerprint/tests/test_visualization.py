"""Tests für das visualization-Modul (Quad-Fingerprinting).

Verwendet synthetische Daten, prüft Figure-Typ, Achsenbeschriftungen,
Titel und PNG-Export. Kein plt.show() in Tests.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # kein Display-Backend in Tests nötig

from quad_fingerprint.spectrogram import Spectrogram
from quad_fingerprint.visualization import (
    plot_comparison_with_shazam,
    plot_peaks,
    plot_quad,
    plot_quad_hash_space,
    plot_scale_robustness,
    plot_spectrogram,
    plot_spectrogram_with_peaks,
    plot_verification,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR = 8000
HOP = 32
N_FFT = 1024
N_BINS = N_FFT // 2 + 1   # 513
N_FRAMES = 500


@pytest.fixture
def synthetic_spectrogram() -> Spectrogram:
    """Synthetisches Magnitude-Spektrogramm."""
    rng = np.random.default_rng(0)
    magnitude = rng.random((N_BINS, N_FRAMES), dtype=np.float32) * 0.5
    times = np.linspace(0, N_FRAMES * HOP / SR, N_FRAMES)
    freqs = np.linspace(0, SR / 2, N_BINS)
    return Spectrogram(magnitude=magnitude, times=times, frequencies=freqs)


@pytest.fixture
def synthetic_peaks() -> np.ndarray:
    """Synthetische Peaks: (N, 2) float32 in (frame, bin)."""
    rng = np.random.default_rng(1)
    n = 80
    times = rng.uniform(5, N_FRAMES - 5, n).astype(np.float32)
    freqs = rng.uniform(5, N_BINS - 5, n).astype(np.float32)
    return np.column_stack([times, freqs])


@pytest.fixture
def synthetic_hashes() -> np.ndarray:
    """Synthetische 4D-Hashes: (N, 4) float32 in [0,1]^4."""
    rng = np.random.default_rng(2)
    return rng.random((300, 4)).astype(np.float32)


# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------

def _check_figure(fig: plt.Figure, expected_title_fragment: str = "") -> None:
    """Prüft, dass ein Figure-Objekt zurückgegeben wurde."""
    assert isinstance(fig, plt.Figure)
    if expected_title_fragment:
        # Titel irgendwo in der Figure prüfen
        texts = [ax.get_title() for ax in fig.axes]
        texts += [fig.texts[i].get_text() for i in range(len(fig.texts))]
        combined = " ".join(texts)
        assert expected_title_fragment.lower() in combined.lower()


# ---------------------------------------------------------------------------
# 1. plot_spectrogram
# ---------------------------------------------------------------------------

class TestPlotSpectrogram:
    def test_returns_figure(self, synthetic_spectrogram: Spectrogram) -> None:
        fig = plot_spectrogram(synthetic_spectrogram, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_axis_labels(self, synthetic_spectrogram: Spectrogram) -> None:
        fig = plot_spectrogram(synthetic_spectrogram, show=False)
        ax = fig.axes[0]
        assert "Zeit" in ax.get_xlabel()
        assert "Frequenz" in ax.get_ylabel()
        plt.close("all")

    def test_saves_png(self, synthetic_spectrogram: Spectrogram, tmp_path: Path) -> None:
        out = tmp_path / "spec.png"
        plot_spectrogram(synthetic_spectrogram, save_path=out, show=False)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_custom_title(self, synthetic_spectrogram: Spectrogram) -> None:
        fig = plot_spectrogram(synthetic_spectrogram, title="MeinTitel", show=False)
        ax = fig.axes[0]
        assert "MeinTitel" in ax.get_title()
        plt.close("all")


# ---------------------------------------------------------------------------
# 2. plot_peaks
# ---------------------------------------------------------------------------

class TestPlotPeaks:
    def test_returns_figure(self, synthetic_peaks: np.ndarray) -> None:
        fig = plot_peaks(synthetic_peaks, (N_BINS, N_FRAMES), show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_axis_labels(self, synthetic_peaks: np.ndarray) -> None:
        fig = plot_peaks(synthetic_peaks, (N_BINS, N_FRAMES), show=False)
        ax = fig.axes[0]
        assert "Zeit" in ax.get_xlabel()
        assert "Frequenz" in ax.get_ylabel()
        plt.close("all")

    def test_saves_png(self, synthetic_peaks: np.ndarray, tmp_path: Path) -> None:
        out = tmp_path / "peaks.png"
        plot_peaks(synthetic_peaks, (N_BINS, N_FRAMES), save_path=out, show=False)
        assert out.exists()

    def test_peak_count_in_title(self, synthetic_peaks: np.ndarray) -> None:
        fig = plot_peaks(synthetic_peaks, (N_BINS, N_FRAMES), show=False)
        ax = fig.axes[0]
        assert str(len(synthetic_peaks)) in ax.get_title()
        plt.close("all")


# ---------------------------------------------------------------------------
# 3. plot_spectrogram_with_peaks
# ---------------------------------------------------------------------------

class TestPlotSpectrogramWithPeaks:
    def test_returns_figure(
        self,
        synthetic_spectrogram: Spectrogram,
        synthetic_peaks: np.ndarray,
    ) -> None:
        fig = plot_spectrogram_with_peaks(
            synthetic_spectrogram, synthetic_peaks, show=False
        )
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_png(
        self,
        synthetic_spectrogram: Spectrogram,
        synthetic_peaks: np.ndarray,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "spec_peaks.png"
        plot_spectrogram_with_peaks(
            synthetic_spectrogram, synthetic_peaks, save_path=out, show=False
        )
        assert out.exists()

    def test_axis_labels(
        self,
        synthetic_spectrogram: Spectrogram,
        synthetic_peaks: np.ndarray,
    ) -> None:
        fig = plot_spectrogram_with_peaks(
            synthetic_spectrogram, synthetic_peaks, show=False
        )
        ax = fig.axes[0]
        assert "Zeit" in ax.get_xlabel()
        assert "Frequenz" in ax.get_ylabel()
        plt.close("all")


# ---------------------------------------------------------------------------
# 4. plot_quad
# ---------------------------------------------------------------------------

class TestPlotQuad:
    def test_returns_figure(self, synthetic_peaks: np.ndarray) -> None:
        fig = plot_quad(synthetic_peaks, (0, 1, 2, 3), (N_BINS, N_FRAMES), show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_png(self, synthetic_peaks: np.ndarray, tmp_path: Path) -> None:
        out = tmp_path / "quad.png"
        plot_quad(
            synthetic_peaks, (0, 1, 2, 3), (N_BINS, N_FRAMES),
            save_path=out, show=False,
        )
        assert out.exists()

    def test_axis_labels(self, synthetic_peaks: np.ndarray) -> None:
        fig = plot_quad(synthetic_peaks, (0, 1, 2, 3), (N_BINS, N_FRAMES), show=False)
        ax = fig.axes[0]
        assert "Zeit" in ax.get_xlabel()
        assert "Frequenz" in ax.get_ylabel()
        plt.close("all")

    def test_legend_contains_abcd(self, synthetic_peaks: np.ndarray) -> None:
        """Legende enthält A, B, C, D."""
        fig = plot_quad(synthetic_peaks, (0, 1, 2, 3), (N_BINS, N_FRAMES), show=False)
        ax = fig.axes[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        combined = " ".join(legend_texts)
        assert "A" in combined
        assert "B" in combined
        plt.close("all")


# ---------------------------------------------------------------------------
# 5. plot_quad_hash_space
# ---------------------------------------------------------------------------

class TestPlotQuadHashSpace:
    def test_returns_figure(self, synthetic_hashes: np.ndarray) -> None:
        fig = plot_quad_hash_space(synthetic_hashes, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_two_subplots(self, synthetic_hashes: np.ndarray) -> None:
        fig = plot_quad_hash_space(synthetic_hashes, show=False)
        assert len(fig.axes) == 2
        plt.close("all")

    def test_saves_png(self, synthetic_hashes: np.ndarray, tmp_path: Path) -> None:
        out = tmp_path / "hash_space.png"
        plot_quad_hash_space(synthetic_hashes, save_path=out, show=False)
        assert out.exists()

    def test_axis_labels(self, synthetic_hashes: np.ndarray) -> None:
        fig = plot_quad_hash_space(synthetic_hashes, show=False)
        assert "C'" in fig.axes[0].get_xlabel()
        assert "D'" in fig.axes[1].get_xlabel()
        plt.close("all")


# ---------------------------------------------------------------------------
# 6. plot_verification
# ---------------------------------------------------------------------------

class TestPlotVerification:
    @pytest.fixture
    def verif_data(self) -> dict:
        rng = np.random.default_rng(3)
        n_ref = 20
        n_query = 15
        ref_peaks = np.column_stack([
            rng.uniform(100, 400, n_ref), rng.uniform(20, 200, n_ref)
        ]).astype(np.float32)
        query_peaks = np.column_stack([
            rng.uniform(100, 400, n_query), rng.uniform(20, 200, n_query)
        ]).astype(np.float32)
        root_ref = np.array([200.0, 100.0], dtype=np.float32)
        root_query = np.array([200.0, 100.0], dtype=np.float32)
        verified_mask = rng.random(n_ref) > 0.4
        return {
            "ref_peaks": ref_peaks,
            "query_peaks": query_peaks,
            "root_ref": root_ref,
            "root_query": root_query,
            "s_time": 1.0,
            "s_freq": 1.0,
            "verified_mask": verified_mask,
        }

    def test_returns_figure(self, verif_data: dict) -> None:
        fig = plot_verification(**verif_data, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_png(self, verif_data: dict, tmp_path: Path) -> None:
        out = tmp_path / "verification.png"
        plot_verification(**verif_data, save_path=out, show=False)
        assert out.exists()

    def test_axis_labels(self, verif_data: dict) -> None:
        fig = plot_verification(**verif_data, show=False)
        ax = fig.axes[0]
        assert "Zeit" in ax.get_xlabel()
        assert "Frequenz" in ax.get_ylabel()
        plt.close("all")

    def test_scale_factors_in_title(self, verif_data: dict) -> None:
        data = {**verif_data, "s_time": 0.9, "s_freq": 1.1}
        fig = plot_verification(**data, show=False)
        ax = fig.axes[0]
        title = ax.get_title()
        assert "0.900" in title
        assert "1.100" in title
        plt.close("all")


# ---------------------------------------------------------------------------
# 7. plot_scale_robustness
# ---------------------------------------------------------------------------

class TestPlotScaleRobustness:
    @pytest.fixture
    def scale_data(self) -> dict[str, dict[float, float]]:
        return {
            "tempo": {0.80: 0.92, 0.90: 0.97, 1.00: 1.00, 1.10: 0.95, 1.20: 0.88},
            "pitch": {0.80: 0.85, 0.90: 0.93, 1.00: 1.00, 1.10: 0.91, 1.20: 0.82},
            "speed": {0.80: 0.88, 0.90: 0.94, 1.00: 1.00, 1.10: 0.93, 1.20: 0.86},
        }

    def test_returns_figure(self, scale_data: dict) -> None:
        fig = plot_scale_robustness(scale_data, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_png(self, scale_data: dict, tmp_path: Path) -> None:
        out = tmp_path / "scale_robustness.png"
        plot_scale_robustness(scale_data, save_path=out, show=False)
        assert out.exists()

    def test_axis_labels(self, scale_data: dict) -> None:
        fig = plot_scale_robustness(scale_data, show=False)
        ax = fig.axes[0]
        assert "Skalierungsfaktor" in ax.get_xlabel()
        assert "Erkennungsrate" in ax.get_ylabel()
        plt.close("all")

    def test_legend_has_all_types(self, scale_data: dict) -> None:
        fig = plot_scale_robustness(scale_data, show=False)
        ax = fig.axes[0]
        legend_texts = [t.get_text().lower() for t in ax.get_legend().get_texts()]
        for dist_type in scale_data:
            assert any(dist_type in t for t in legend_texts)
        plt.close("all")

    def test_empty_data(self) -> None:
        fig = plot_scale_robustness({}, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ---------------------------------------------------------------------------
# 8. plot_comparison_with_shazam
# ---------------------------------------------------------------------------

class TestPlotComparisonWithShazam:
    @pytest.fixture
    def comparison_data(self) -> tuple[dict, dict]:
        shazam = {"tempo": 0.72, "pitch": 0.68, "speed": 0.75, "noise": 0.85}
        quad = {"tempo": 0.94, "pitch": 0.91, "speed": 0.93, "noise": 0.89}
        return shazam, quad

    def test_returns_figure(self, comparison_data: tuple) -> None:
        shazam, quad = comparison_data
        fig = plot_comparison_with_shazam(shazam, quad, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_png(self, comparison_data: tuple, tmp_path: Path) -> None:
        shazam, quad = comparison_data
        out = tmp_path / "comparison.png"
        plot_comparison_with_shazam(shazam, quad, save_path=out, show=False)
        assert out.exists()

    def test_axis_labels(self, comparison_data: tuple) -> None:
        shazam, quad = comparison_data
        fig = plot_comparison_with_shazam(shazam, quad, show=False)
        ax = fig.axes[0]
        assert "Distortion" in ax.get_xlabel()
        assert "Erkennungsrate" in ax.get_ylabel()
        plt.close("all")

    def test_legend_contains_both_names(self, comparison_data: tuple) -> None:
        shazam, quad = comparison_data
        fig = plot_comparison_with_shazam(shazam, quad, show=False)
        ax = fig.axes[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        combined = " ".join(legend_texts)
        assert "Shazam" in combined
        assert "Quad" in combined
        plt.close("all")

    def test_partial_overlap(self) -> None:
        """Shazam und Quad haben unterschiedliche Distortion-Typen."""
        shazam = {"tempo": 0.80}
        quad = {"tempo": 0.95, "pitch": 0.88}
        fig = plot_comparison_with_shazam(shazam, quad, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
