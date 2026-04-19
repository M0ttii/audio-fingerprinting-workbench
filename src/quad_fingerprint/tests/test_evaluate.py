"""Tests für das evaluate-Modul (Quad-Fingerprinting).

Testet Robustheits- und Effizienz-Metriken, specificity,
recognition_by_scale_factor, sowie JSON/CSV-Export.
"""

import json
from pathlib import Path

import pytest

from quad_fingerprint.evaluate import (
    EvalReport,
    EvalResult,
    compute_metrics,
    export_csv,
    export_json,
    recognition_by_scale_factor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_results() -> list[EvalResult]:
    """Einfache Ergebnismenge: 3 korrekt, 1 falsch negativ."""
    return [
        EvalResult("q1.wav", "song_a", "song_a", 0.85, 50.0, 120.0, 200),
        EvalResult("q2.wav", "song_b", "song_b", 0.72, 45.0, 110.0, 180),
        EvalResult("q3.wav", "song_c", "song_c", 0.91, 55.0, 130.0, 220),
        EvalResult("q4.wav", "song_d", None, 0.0, 60.0, 140.0, 190),
    ]


@pytest.fixture
def results_with_negatives() -> list[EvalResult]:
    """Ergebnismenge mit True Negatives und False Positives."""
    return [
        # In DB, korrekt erkannt
        EvalResult("q1.wav", "song_a", "song_a", 0.85, 50.0, 100.0, 200),
        # In DB, nicht erkannt (FN)
        EvalResult("q2.wav", "song_b", None, 0.0, 45.0, 110.0, 180),
        # Nicht in DB, korrekt abgelehnt (TN)
        EvalResult("q3.wav", None, None, 0.0, 40.0, 90.0, 160),
        # Nicht in DB, fälschlicherweise erkannt (FP)
        EvalResult("q4.wav", None, "song_x", 0.55, 55.0, 120.0, 210),
        # Nicht in DB, korrekt abgelehnt (TN)
        EvalResult("q5.wav", None, None, 0.0, 42.0, 95.0, 170),
    ]


@pytest.fixture
def scale_results() -> list[EvalResult]:
    """Ergebnismenge mit distortion_type und distortion_level."""
    results = []
    # Tempo 80%: 2 korrekt, 1 falsch
    for i, correct in enumerate([True, True, False]):
        predicted = "song_a" if correct else None
        r = EvalResult(
            f"tempo80_{i}.wav", "song_a", predicted,
            0.7 if correct else 0.0, 50.0, 100.0, 200,
        )
        r.distortion_type = "tempo"
        r.distortion_level = 0.80
        results.append(r)
    # Tempo 100%: 3 korrekt
    for i in range(3):
        r = EvalResult(
            f"tempo100_{i}.wav", "song_a", "song_a",
            0.9, 50.0, 100.0, 200,
        )
        r.distortion_type = "tempo"
        r.distortion_level = 1.00
        results.append(r)
    # Pitch 120%: 1 korrekt, 1 falsch
    for i, correct in enumerate([True, False]):
        predicted = "song_a" if correct else None
        r = EvalResult(
            f"pitch120_{i}.wav", "song_a", predicted,
            0.6 if correct else 0.0, 50.0, 100.0, 200,
        )
        r.distortion_type = "pitch"
        r.distortion_level = 1.20
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

class TestEvalResult:
    def test_is_correct_true(self) -> None:
        r = EvalResult("q.wav", "song_a", "song_a", 0.8, 50.0, 100.0)
        assert r.is_correct is True

    def test_is_correct_false_wrong_match(self) -> None:
        r = EvalResult("q.wav", "song_a", "song_b", 0.6, 50.0, 100.0)
        assert r.is_correct is False

    def test_is_correct_false_no_match(self) -> None:
        r = EvalResult("q.wav", "song_a", None, 0.0, 50.0, 100.0)
        assert r.is_correct is False

    def test_is_correct_both_none(self) -> None:
        """Wenn kein Match erwartet und keiner gefunden → is_correct=True."""
        r = EvalResult("q.wav", None, None, 0.0, 50.0, 100.0)
        assert r.is_correct is True

    def test_scale_factors_optional(self) -> None:
        r = EvalResult("q.wav", "song_a", "song_a", 0.8, 50.0, 100.0)
        assert r.scale_factors is None
        assert r.distortion_type is None
        assert r.distortion_level is None

    def test_score_is_float(self) -> None:
        """Score ist float (0.0–1.0), nicht int wie bei Shazam."""
        r = EvalResult("q.wav", "song_a", "song_a", 0.85, 50.0, 100.0)
        assert isinstance(r.score, float)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="Keine EvalResult"):
            compute_metrics([])

    def test_basic_recognition_rate(self, basic_results: list[EvalResult]) -> None:
        report = compute_metrics(basic_results)
        assert report.recognition_rate == pytest.approx(3 / 4)
        assert report.correct_count == 3
        assert report.total_queries == 4

    def test_basic_false_negative_rate(
        self, basic_results: list[EvalResult]
    ) -> None:
        report = compute_metrics(basic_results)
        assert report.false_negative_rate == pytest.approx(1 / 4)
        assert report.false_negative_count == 1

    def test_no_not_in_db_queries(self, basic_results: list[EvalResult]) -> None:
        """Wenn keine not-in-db Queries: FPR=0, Specificity=1.0."""
        report = compute_metrics(basic_results)
        assert report.false_positive_rate == 0.0
        assert report.specificity == 1.0

    def test_specificity(
        self, results_with_negatives: list[EvalResult]
    ) -> None:
        """Specificity = TN / (TN + FP) = 2 / 3, Paper Eq. (14)."""
        report = compute_metrics(results_with_negatives)
        assert report.true_negative_count == 2
        assert report.false_positive_count == 1
        assert report.specificity == pytest.approx(2 / 3)

    def test_false_positive_rate(
        self, results_with_negatives: list[EvalResult]
    ) -> None:
        report = compute_metrics(results_with_negatives)
        # 1 FP out of 3 not-in-db queries
        assert report.false_positive_rate == pytest.approx(1 / 3)

    def test_efficiency_metrics(self, basic_results: list[EvalResult]) -> None:
        report = compute_metrics(basic_results)
        expected_fp_time = (50.0 + 45.0 + 55.0 + 60.0) / 4
        expected_q_time = (120.0 + 110.0 + 130.0 + 140.0) / 4
        assert report.avg_fingerprint_time_ms == pytest.approx(expected_fp_time)
        assert report.avg_query_time_ms == pytest.approx(expected_q_time)

    def test_quads_per_second(self, basic_results: list[EvalResult]) -> None:
        report = compute_metrics(basic_results)
        total_quads = 200 + 180 + 220 + 190
        total_time_s = (50.0 + 45.0 + 55.0 + 60.0) / 1000.0
        assert report.quads_per_second == pytest.approx(
            total_quads / total_time_s
        )

    def test_db_memory_without_database(
        self, basic_results: list[EvalResult]
    ) -> None:
        report = compute_metrics(basic_results)
        assert report.db_memory_mb == 0.0

    def test_returns_eval_report(self, basic_results: list[EvalResult]) -> None:
        report = compute_metrics(basic_results)
        assert isinstance(report, EvalReport)

    def test_all_correct(self) -> None:
        results = [
            EvalResult("q.wav", "s", "s", 0.9, 10.0, 20.0, 100)
            for _ in range(5)
        ]
        report = compute_metrics(results)
        assert report.recognition_rate == 1.0
        assert report.false_negative_rate == 0.0

    def test_all_wrong(self) -> None:
        results = [
            EvalResult("q.wav", "s", None, 0.0, 10.0, 20.0, 100)
            for _ in range(5)
        ]
        report = compute_metrics(results)
        assert report.recognition_rate == 0.0
        assert report.false_negative_rate == 1.0


# ---------------------------------------------------------------------------
# recognition_by_scale_factor
# ---------------------------------------------------------------------------

class TestRecognitionByScaleFactor:
    def test_basic_grouping(self, scale_results: list[EvalResult]) -> None:
        rates = recognition_by_scale_factor(scale_results)
        assert "tempo" in rates
        assert "pitch" in rates

    def test_tempo_rates(self, scale_results: list[EvalResult]) -> None:
        rates = recognition_by_scale_factor(scale_results)
        assert rates["tempo"][0.80] == pytest.approx(2 / 3)
        assert rates["tempo"][1.00] == pytest.approx(1.0)

    def test_pitch_rates(self, scale_results: list[EvalResult]) -> None:
        rates = recognition_by_scale_factor(scale_results)
        assert rates["pitch"][1.20] == pytest.approx(0.5)

    def test_ignores_none_distortion(self) -> None:
        """EvalResults ohne distortion_type/level werden ignoriert."""
        results = [
            EvalResult("q.wav", "song_a", "song_a", 0.8, 50.0, 100.0, 200),
        ]
        rates = recognition_by_scale_factor(results)
        assert len(rates) == 0

    def test_ignores_not_in_db(self) -> None:
        """Queries ohne expected_match werden nicht berücksichtigt."""
        r = EvalResult("q.wav", None, None, 0.0, 50.0, 100.0, 200)
        r.distortion_type = "tempo"
        r.distortion_level = 1.0
        rates = recognition_by_scale_factor([r])
        assert len(rates) == 0

    def test_empty_results(self) -> None:
        rates = recognition_by_scale_factor([])
        assert len(rates) == 0


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExportJson:
    def test_creates_file(
        self, basic_results: list[EvalResult], tmp_path: Path
    ) -> None:
        report = compute_metrics(basic_results)
        out = tmp_path / "report.json"
        export_json(report, basic_results, out)
        assert out.exists()

    def test_json_structure(
        self, basic_results: list[EvalResult], tmp_path: Path
    ) -> None:
        report = compute_metrics(basic_results)
        out = tmp_path / "report.json"
        export_json(report, basic_results, out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert "summary" in data
        assert "results" in data
        assert len(data["results"]) == len(basic_results)
        assert "recognition_rate" in data["summary"]
        assert "specificity" in data["summary"]

    def test_creates_parent_dirs(
        self, basic_results: list[EvalResult], tmp_path: Path
    ) -> None:
        out = tmp_path / "sub" / "dir" / "report.json"
        report = compute_metrics(basic_results)
        export_json(report, basic_results, out)
        assert out.exists()


class TestExportCsv:
    def test_creates_file(
        self, basic_results: list[EvalResult], tmp_path: Path
    ) -> None:
        out = tmp_path / "results.csv"
        export_csv(basic_results, out)
        assert out.exists()

    def test_csv_rows(
        self, basic_results: list[EvalResult], tmp_path: Path
    ) -> None:
        out = tmp_path / "results.csv"
        export_csv(basic_results, out)

        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == len(basic_results) + 1  # header + data rows

    def test_csv_header_contains_quad_fields(
        self, basic_results: list[EvalResult], tmp_path: Path
    ) -> None:
        out = tmp_path / "results.csv"
        export_csv(basic_results, out)

        header = out.read_text(encoding="utf-8").split("\n")[0]
        assert "scale_factors" in header
        assert "distortion_type" in header
        assert "distortion_level" in header
        assert "num_query_quads" in header

    def test_empty_results_no_file(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.csv"
        export_csv([], out)
        assert not out.exists()
