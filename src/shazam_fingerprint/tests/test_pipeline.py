"""
Tests für pipeline.py.

End-to-End-Tests: Synthetische WAV-Dateien werden auf Disk geschrieben,
per ingest_directory in die Datenbank eingelesen und anschließend per
query() abgefragt. Song 3 muss als Match zurückkommen.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from shazam_fingerprint import config
from shazam_fingerprint.database import FingerprintDatabase
from shazam_fingerprint.matcher import MatchResult
from shazam_fingerprint.pipeline import ingest_directory, query, evaluate_robustness


# ======================================================================
# Fixtures und Hilfsfunktionen
# ======================================================================

# Fünf verschiedene Trägerfrequenzen für synthetische Songs.
# Jeder Song besteht aus einem Sinuston mit einer einzigartigen Frequenz,
# sodass die Fingerprints der Songs sich deutlich unterscheiden.
_SONG_FREQS_HZ = [440.0, 880.0, 1320.0, 1760.0, 2200.0]
_SONG_NAMES = [f"song_{i + 1:02d}" for i in range(5)]
_SONG_DURATION_SEC = 15.0   # Genug Content für stabile Fingerprints
_QUERY_SONG_IDX = 2         # Song 3 (0-basiert: Index 2)


def _write_sine_wav(
    path: Path,
    freq_hz: float,
    duration_sec: float,
    sr: int,
) -> None:
    """Schreibt einen Sinuston als 16-Bit-WAV-Datei.

    Args:
        path: Zielpfad der WAV-Datei.
        freq_hz: Frequenz des Sinustons in Hz.
        duration_sec: Dauer des Signals in Sekunden.
        sr: Abtastrate in Hz.
    """
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    signal = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    sf.write(str(path), signal, sr, subtype="PCM_16")


@pytest.fixture(scope="module")
def song_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporäres Verzeichnis mit 5 synthetischen WAV-Dateien (Module-Scope).

    Der Module-Scope stellt sicher, dass die Dateien nur einmal erzeugt werden
    und von allen Tests in diesem Modul geteilt werden.
    """
    directory = tmp_path_factory.mktemp("songs")
    sr = config.SAMPLE_RATE
    for name, freq in zip(_SONG_NAMES, _SONG_FREQS_HZ):
        _write_sine_wav(directory / f"{name}.wav", freq, _SONG_DURATION_SEC, sr)
    return directory


@pytest.fixture(scope="module")
def filled_db(song_dir: Path) -> FingerprintDatabase:
    """Datenbank mit allen 5 Songs (Module-Scope, einmaliger Ingest)."""
    db = FingerprintDatabase()
    ingest_directory(song_dir, db, show_progress=False)
    return db


@pytest.fixture(scope="module")
def query_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """WAV-Datei mit einem Ausschnitt aus Song 3 als Query (Module-Scope)."""
    query_dir = tmp_path_factory.mktemp("queries")
    path = query_dir / f"{_SONG_NAMES[_QUERY_SONG_IDX]}.wav"
    _write_sine_wav(
        path,
        freq_hz=_SONG_FREQS_HZ[_QUERY_SONG_IDX],
        duration_sec=_SONG_DURATION_SEC,
        sr=config.SAMPLE_RATE,
    )
    return path


# ======================================================================
# ingest_directory: Statistiken und Korrektheit
# ======================================================================

class TestIngestDirectory:
    def test_all_songs_ingested(self, song_dir: Path, filled_db: FingerprintDatabase) -> None:
        """Alle 5 Songs aus dem Verzeichnis werden in die Datenbank eingefügt."""
        stats = filled_db.get_stats()
        assert stats["num_songs"] == 5, (
            f"Erwartet 5 Songs, gefunden {stats['num_songs']}"
        )

    def test_returns_statistics_dict(self, song_dir: Path) -> None:
        """ingest_directory gibt ein Dict mit Statistiken zurück."""
        db = FingerprintDatabase()
        stats = ingest_directory(song_dir, db, show_progress=False)
        assert isinstance(stats, dict)
        assert "processed" in stats
        assert "skipped" in stats
        assert "failed" in stats
        assert "total_hashes" in stats
        assert "total_time_s" in stats

    def test_processed_count_correct(self, song_dir: Path) -> None:
        """processed-Zähler entspricht der Anzahl erfolgreich verarbeiteter Songs."""
        db = FingerprintDatabase()
        stats = ingest_directory(song_dir, db, show_progress=False)
        assert stats["processed"] == 5
        assert stats["failed"] == 0

    def test_total_hashes_nonzero(self, song_dir: Path) -> None:
        """total_hashes ist größer 0 nach dem Ingest."""
        db = FingerprintDatabase()
        stats = ingest_directory(song_dir, db, show_progress=False)
        assert stats["total_hashes"] > 0

    def test_skip_already_indexed_songs(self, song_dir: Path) -> None:
        """Bereits indexierte Songs werden beim zweiten Ingest übersprungen."""
        db = FingerprintDatabase()
        ingest_directory(song_dir, db, show_progress=False)
        # Zweiter Ingest: alle Songs sind bereits drin → skipped=5, processed=0
        stats2 = ingest_directory(song_dir, db, show_progress=False)
        assert stats2["processed"] == 0
        assert stats2["skipped"] == 5

    def test_nonexistent_directory_raises(self) -> None:
        """Nicht-existierendes Verzeichnis wirft FileNotFoundError."""
        db = FingerprintDatabase()
        with pytest.raises(FileNotFoundError):
            ingest_directory("/tmp/does_not_exist_xyz_12345", db, show_progress=False)

    def test_empty_directory_returns_zero_processed(self, tmp_path: Path) -> None:
        """Leeres Verzeichnis liefert processed=0 ohne Fehler."""
        db = FingerprintDatabase()
        stats = ingest_directory(tmp_path, db, show_progress=False)
        assert stats["processed"] == 0


# ======================================================================
# End-to-End: Ingest 5 Songs, Query mit Song 3 → Song 3 wird erkannt
# ======================================================================

class TestEndToEnd:
    def test_song3_matched(
        self, filled_db: FingerprintDatabase, query_file: Path
    ) -> None:
        """Haupttest: Query aus Song 3 → Song 3 wird als Match zurückgegeben.

        Dies ist der zentrale End-to-End-Test laut Anforderungen:
        'Ingest 5 Songs, Query mit Ausschnitt aus Song 3 → Song 3 muss als
        Match zurückkommen.'
        """
        result = query(query_file, filled_db)
        assert result.match_found is True, (
            f"Kein Match gefunden (Score {result.best_score}, "
            f"Threshold {config.MATCH_THRESHOLD})"
        )
        assert result.best_match == _SONG_NAMES[_QUERY_SONG_IDX], (
            f"Erwarteter Match: '{_SONG_NAMES[_QUERY_SONG_IDX]}', "
            f"gefunden: '{result.best_match}'"
        )

    def test_correct_song_has_highest_score(
        self, filled_db: FingerprintDatabase, query_file: Path
    ) -> None:
        """Song 3 hat den höchsten Score aller Kandidaten."""
        result = query(query_file, filled_db)
        expected_id = _SONG_NAMES[_QUERY_SONG_IDX]
        assert result.all_scores.get(expected_id, 0) == result.best_score

    def test_returns_match_result(
        self, filled_db: FingerprintDatabase, query_file: Path
    ) -> None:
        """query() gibt ein MatchResult-Objekt zurück."""
        result = query(query_file, filled_db)
        assert isinstance(result, MatchResult)

    def test_processing_time_recorded(
        self, filled_db: FingerprintDatabase, query_file: Path
    ) -> None:
        """processing_time_ms ist nach dem Query gesetzt."""
        result = query(query_file, filled_db)
        assert result.processing_time_ms >= 0.0

    def test_num_query_hashes_nonzero(
        self, filled_db: FingerprintDatabase, query_file: Path
    ) -> None:
        """Die Query erzeugt mindestens einen Hash."""
        result = query(query_file, filled_db)
        assert result.num_query_hashes > 0


# ======================================================================
# query(): Fehlerbehandlung und optionale Parameter
# ======================================================================

class TestQueryFunction:
    def test_nonexistent_file_raises(
        self, filled_db: FingerprintDatabase
    ) -> None:
        """Nicht-existierende Audiodatei wirft FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            query("/tmp/does_not_exist.wav", filled_db)

    def test_query_with_duration_sec(
        self, filled_db: FingerprintDatabase, query_file: Path
    ) -> None:
        """Query mit duration_sec-Parameter liefert immer noch ein Ergebnis."""
        result = query(query_file, filled_db, duration_sec=5.0)
        assert isinstance(result, MatchResult)

    def test_query_with_start_sec(
        self, filled_db: FingerprintDatabase, query_file: Path
    ) -> None:
        """Query mit start_sec-Parameter (Ausschnitt aus der Mitte) funktioniert."""
        result = query(query_file, filled_db, start_sec=2.0, duration_sec=8.0)
        assert isinstance(result, MatchResult)

    def test_query_start_sec_still_matches_song3(
        self, filled_db: FingerprintDatabase, query_file: Path
    ) -> None:
        """Query-Ausschnitt ab Sekunde 3 erkennt trotzdem Song 3.

        Wang Section 3.1: Shazam erkennt Songs auch anhand kurzer Ausschnitte.
        """
        result = query(query_file, filled_db, start_sec=3.0, duration_sec=10.0)
        assert result.match_found is True
        assert result.best_match == _SONG_NAMES[_QUERY_SONG_IDX]


# ======================================================================
# Datenbank-Integrität nach dem Ingest
# ======================================================================

class TestDatabaseAfterIngest:
    def test_all_song_ids_in_db(self, filled_db: FingerprintDatabase) -> None:
        """Alle 5 Song-IDs sind nach dem Ingest in der Datenbank."""
        for name in _SONG_NAMES:
            assert name in filled_db, f"Song '{name}' fehlt in der Datenbank"

    def test_db_has_hashes(self, filled_db: FingerprintDatabase) -> None:
        """Die Datenbank enthält Hashes nach dem Ingest."""
        assert len(filled_db) > 0

    def test_db_stats_consistent(self, filled_db: FingerprintDatabase) -> None:
        """get_stats() liefert konsistente Werte nach dem Ingest."""
        stats = filled_db.get_stats()
        assert stats["num_songs"] == 5
        assert stats["num_hashes"] > 0
        assert stats["num_entries"] >= stats["num_hashes"]
        assert stats["memory_mb"] > 0.0


# ======================================================================
# evaluate_robustness: Batch-Evaluation
# ======================================================================

class TestEvaluateRobustness:
    def test_batch_evaluation_all_correct(
        self, song_dir: Path, filled_db: FingerprintDatabase
    ) -> None:
        """Batch-Evaluation über alle 5 Songs: alle werden korrekt erkannt.

        Verwendet song_dir direkt als Query-Verzeichnis (Dateinamen = song_ids).
        """
        report = evaluate_robustness(
            song_dir,
            filled_db,
            show_progress=False,
        )
        assert report.total_queries == 5
        assert report.correct_count == 5
        assert report.recognition_rate == pytest.approx(1.0)
        assert report.false_negative_rate == pytest.approx(0.0)

    def test_evaluate_returns_eval_report(
        self, song_dir: Path, filled_db: FingerprintDatabase
    ) -> None:
        """evaluate_robustness gibt ein EvalReport-Objekt zurück."""
        from shazam_fingerprint.evaluate import EvalReport
        report = evaluate_robustness(
            song_dir, filled_db, show_progress=False
        )
        assert isinstance(report, EvalReport)

    def test_evaluate_nonexistent_dir_raises(
        self, filled_db: FingerprintDatabase
    ) -> None:
        """Nicht-existierendes Query-Verzeichnis wirft FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            evaluate_robustness("/tmp/no_such_dir_xyz", filled_db, show_progress=False)

    def test_evaluate_with_ground_truth_dict(
        self, song_dir: Path, filled_db: FingerprintDatabase
    ) -> None:
        """Explizites ground_truth-Dict wird für die Metriken verwendet."""
        ground_truth = {f"{name}.wav": name for name in _SONG_NAMES}
        report = evaluate_robustness(
            song_dir,
            filled_db,
            ground_truth=ground_truth,
            show_progress=False,
        )
        assert report.total_queries == 5
        assert report.recognition_rate == pytest.approx(1.0)
