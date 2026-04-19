"""Tests für das database-Modul (Quad-Fingerprinting).

Testet alle vier Datenstrukturen (Paper Section V):
  peakfile, refrecords, fidindex, searchtree

Schwerpunkte:
- Workflow: add_file → finalize → query_radius
- Searchtree wird EINMALIG nach finalize() gebaut (nicht nach add_file)
- Fixed-Radius NN-Suche: Treffer innerhalb Radius, keine Treffer außerhalb
- Persistenz: save/load ergibt identische Ergebnisse
- Fehlerbehandlung: add_file nach finalize, Duplikat-Namen, unbekannte file_id
- Speicherstatistiken: memory_usage_mb
"""

from pathlib import Path

import numpy as np
import pytest

from quad_fingerprint import config
from quad_fingerprint.database import REFRECORD_DTYPE, ReferenceDatabase
from quad_fingerprint.quad_builder import QuadRecord


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _make_peaks(n: int = 10, seed: int = 0) -> np.ndarray:
    """Synthetische Peaks (N, 2) float32."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, 500, n)).astype(np.float32)
    freqs = rng.uniform(10, 200, n).astype(np.float32)
    return np.column_stack([times, freqs])


def _make_quad_records(n: int = 5, file_id: int = 0, seed: int = 1) -> list[QuadRecord]:
    """Synthetische QuadRecord-Liste mit Hashes im Einheitsquadrat."""
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n):
        records.append(QuadRecord(
            root_point=rng.random(2).astype(np.float32),
            quad_size=rng.uniform(0.1, 5.0, 2).astype(np.float32),
            hash=rng.random(4).astype(np.float32),
            file_id=file_id,
        ))
    return records


def _populated_db(n_files: int = 3) -> tuple[ReferenceDatabase, list[str]]:
    """Erstellt und finalisiert eine DB mit n_files Dateien."""
    db = ReferenceDatabase()
    names = []
    for i in range(n_files):
        name = f"song_{i}.wav"
        db.add_file(
            file_name=name,
            peaks=_make_peaks(10, seed=i),
            quad_records=_make_quad_records(5, file_id=i, seed=i + 10),
        )
        names.append(name)
    db.finalize()
    return db, names


# ---------------------------------------------------------------------------
# 1. Initialisierung
# ---------------------------------------------------------------------------

class TestInit:
    def test_not_finalized_initially(self) -> None:
        db = ReferenceDatabase()
        assert db.is_finalized is False

    def test_zero_files_initially(self) -> None:
        db = ReferenceDatabase()
        assert db.n_files == 0

    def test_zero_records_initially(self) -> None:
        db = ReferenceDatabase()
        assert db.n_records == 0

    def test_zero_peaks_initially(self) -> None:
        db = ReferenceDatabase()
        assert db.n_peaks == 0


# ---------------------------------------------------------------------------
# 2. add_file
# ---------------------------------------------------------------------------

class TestAddFile:
    def test_returns_file_id(self) -> None:
        db = ReferenceDatabase()
        fid = db.add_file("a.wav", _make_peaks(), _make_quad_records())
        assert isinstance(fid, int)

    def test_file_ids_sequential(self) -> None:
        db = ReferenceDatabase()
        fid0 = db.add_file("a.wav", _make_peaks(seed=0), _make_quad_records(seed=0))
        fid1 = db.add_file("b.wav", _make_peaks(seed=1), _make_quad_records(seed=1))
        assert fid1 == fid0 + 1

    def test_n_files_increments(self) -> None:
        db = ReferenceDatabase()
        db.add_file("a.wav", _make_peaks(), _make_quad_records())
        assert db.n_files == 1
        db.add_file("b.wav", _make_peaks(seed=1), _make_quad_records(seed=1))
        assert db.n_files == 2

    def test_n_peaks_accumulates(self) -> None:
        db = ReferenceDatabase()
        peaks_a = _make_peaks(8, seed=0)
        peaks_b = _make_peaks(12, seed=1)
        db.add_file("a.wav", peaks_a, [])
        db.add_file("b.wav", peaks_b, [])
        assert db.n_peaks == 8 + 12

    def test_n_records_accumulates(self) -> None:
        db = ReferenceDatabase()
        db.add_file("a.wav", _make_peaks(), _make_quad_records(3, seed=0))
        db.add_file("b.wav", _make_peaks(seed=1), _make_quad_records(7, seed=1))
        assert db.n_records == 3 + 7

    def test_duplicate_name_raises(self) -> None:
        db = ReferenceDatabase()
        db.add_file("a.wav", _make_peaks(), _make_quad_records())
        with pytest.raises(ValueError, match="bereits in Datenbank"):
            db.add_file("a.wav", _make_peaks(seed=1), _make_quad_records(seed=1))

    def test_add_after_finalize_raises(self) -> None:
        db = ReferenceDatabase()
        db.add_file("a.wav", _make_peaks(), _make_quad_records())
        db.finalize()
        with pytest.raises(RuntimeError, match="finalisiert"):
            db.add_file("b.wav", _make_peaks(seed=1), _make_quad_records(seed=1))

    def test_add_file_with_no_quads(self) -> None:
        """Datei ohne Quads (z.B. zu wenig Peaks) darf keinen Fehler auslösen."""
        db = ReferenceDatabase()
        fid = db.add_file("quiet.wav", _make_peaks(3), [])
        assert fid == 0
        assert db.n_files == 1
        assert db.n_records == 0


# ---------------------------------------------------------------------------
# 3. finalize (Searchtree wird EINMALIG gebaut)
# ---------------------------------------------------------------------------

class TestFinalize:
    def test_is_finalized_after_finalize(self) -> None:
        db = ReferenceDatabase()
        db.add_file("a.wav", _make_peaks(), _make_quad_records())
        db.finalize()
        assert db.is_finalized is True

    def test_finalize_idempotent(self) -> None:
        """Doppeltes finalize() soll keine Exception werfen."""
        db = ReferenceDatabase()
        db.add_file("a.wav", _make_peaks(), _make_quad_records())
        db.finalize()
        db.finalize()  # zweiter Aufruf: kein Fehler, nur Warning

    def test_n_files_unchanged_after_finalize(self) -> None:
        db, _ = _populated_db(3)
        assert db.n_files == 3

    def test_n_records_unchanged_after_finalize(self) -> None:
        db = ReferenceDatabase()
        db.add_file("a.wav", _make_peaks(), _make_quad_records(5, seed=0))
        db.add_file("b.wav", _make_peaks(seed=1), _make_quad_records(4, seed=1))
        db.finalize()
        assert db.n_records == 9

    def test_n_peaks_unchanged_after_finalize(self) -> None:
        db = ReferenceDatabase()
        peaks_a = _make_peaks(10, seed=0)
        db.add_file("a.wav", peaks_a, _make_quad_records(seed=0))
        db.finalize()
        assert db.n_peaks == 10

    def test_finalize_empty_db(self) -> None:
        """Leere DB kann finalisiert werden (kein Searchtree, keine Fehler)."""
        db = ReferenceDatabase()
        db.finalize()
        assert db.is_finalized is True
        assert db.n_records == 0

    def test_searchtree_not_built_before_finalize(self) -> None:
        """Zwischen add_file und finalize: query_radius muss RuntimeError auslösen."""
        db = ReferenceDatabase()
        db.add_file("a.wav", _make_peaks(), _make_quad_records())
        with pytest.raises(RuntimeError):
            db.query_radius(np.random.rand(1, 4).astype(np.float32))


# ---------------------------------------------------------------------------
# 4. query_radius (Fixed-Radius NN-Suche)
# ---------------------------------------------------------------------------

class TestQueryRadius:
    @pytest.fixture
    def db_with_known_hash(self) -> tuple[ReferenceDatabase, np.ndarray]:
        """DB mit einem bekannten Hash-Vektor für gezieltes Testen."""
        known_hash = np.array([0.3, 0.4, 0.7, 0.6], dtype=np.float32)
        rec = QuadRecord(
            root_point=np.array([0.0, 0.0], dtype=np.float32),
            quad_size=np.array([1.0, 1.0], dtype=np.float32),
            hash=known_hash,
            file_id=0,
        )
        db = ReferenceDatabase()
        db.add_file("song.wav", _make_peaks(), [rec])
        db.finalize()
        return db, known_hash

    def test_exact_hash_found(
        self, db_with_known_hash: tuple[ReferenceDatabase, np.ndarray]
    ) -> None:
        """Exakter Query-Hash → mindestens ein Treffer."""
        db, known_hash = db_with_known_hash
        results = db.query_radius(known_hash.reshape(1, 4))
        assert len(results) == 1
        assert len(results[0]) >= 1

    def test_hash_within_radius_found(
        self, db_with_known_hash: tuple[ReferenceDatabase, np.ndarray]
    ) -> None:
        """Hash um epsilon/2 verschoben → noch innerhalb des Radius → Treffer."""
        db, known_hash = db_with_known_hash
        query = known_hash.copy()
        query[0] += config.SEARCH_RADIUS * 0.5
        results = db.query_radius(query.reshape(1, 4))
        assert len(results[0]) >= 1

    def test_hash_outside_radius_not_found(
        self, db_with_known_hash: tuple[ReferenceDatabase, np.ndarray]
    ) -> None:
        """Hash um 10×epsilon verschoben → außerhalb des Radius → kein Treffer."""
        db, known_hash = db_with_known_hash
        query = known_hash.copy()
        query[0] += config.SEARCH_RADIUS * 10.0
        results = db.query_radius(query.reshape(1, 4))
        assert len(results[0]) == 0

    def test_returns_list_of_length_m(self) -> None:
        """Für M Query-Hashes werden M Ergebnis-Arrays zurückgegeben."""
        db, _ = _populated_db(2)
        m = 5
        queries = np.random.default_rng(0).random((m, 4)).astype(np.float32)
        results = db.query_radius(queries)
        assert len(results) == m

    def test_each_result_is_ndarray(self) -> None:
        db, _ = _populated_db(2)
        queries = np.random.default_rng(1).random((3, 4)).astype(np.float32)
        results = db.query_radius(queries)
        for r in results:
            assert isinstance(r, np.ndarray)

    def test_result_indices_valid(self) -> None:
        """Zurückgegebene Indizes müssen im Bereich [0, n_records) liegen."""
        db, _ = _populated_db(2)
        queries = np.random.default_rng(2).random((10, 4)).astype(np.float32)
        results = db.query_radius(queries)
        for r in results:
            if len(r) > 0:
                assert np.all(r >= 0)
                assert np.all(r < db.n_records)

    def test_not_finalized_raises(self) -> None:
        db = ReferenceDatabase()
        db.add_file("a.wav", _make_peaks(), _make_quad_records())
        with pytest.raises(RuntimeError):
            db.query_radius(np.zeros((1, 4), dtype=np.float32))

    def test_empty_db_returns_empty_results(self) -> None:
        """Leere DB: Suche gibt für jeden Query leeres Array zurück."""
        db = ReferenceDatabase()
        db.finalize()
        queries = np.zeros((3, 4), dtype=np.float32)
        results = db.query_radius(queries)
        assert len(results) == 3
        for r in results:
            assert len(r) == 0

    def test_custom_radius(
        self, db_with_known_hash: tuple[ReferenceDatabase, np.ndarray]
    ) -> None:
        """Mit größerem Radius mehr Treffer als mit kleinerem."""
        db, known_hash = db_with_known_hash
        query = known_hash.reshape(1, 4)
        res_small = db.query_radius(query, radius=1e-6)
        res_large = db.query_radius(query, radius=0.5)
        assert len(res_large[0]) >= len(res_small[0])


# ---------------------------------------------------------------------------
# 5. get_refrecords_by_indices
# ---------------------------------------------------------------------------

class TestGetRefrecordsByIndices:
    def test_returns_structured_array(self) -> None:
        db, _ = _populated_db(2)
        indices = np.array([0], dtype=np.intp)
        rec = db.get_refrecords_by_indices(indices)
        assert rec.dtype == REFRECORD_DTYPE

    def test_correct_number_of_records(self) -> None:
        db, _ = _populated_db(2)
        indices = np.array([0, 1], dtype=np.intp)
        recs = db.get_refrecords_by_indices(indices)
        assert len(recs) == 2

    def test_hash_field_accessible(self) -> None:
        db, _ = _populated_db(2)
        indices = np.array([0], dtype=np.intp)
        rec = db.get_refrecords_by_indices(indices)
        assert rec['hash'].shape == (1, 4)

    def test_file_id_in_valid_range(self) -> None:
        db, _ = _populated_db(3)
        indices = np.arange(min(5, db.n_records), dtype=np.intp)
        recs = db.get_refrecords_by_indices(indices)
        assert np.all(recs['file_id'] < db.n_files)


# ---------------------------------------------------------------------------
# 6. get_peaks_for_file
# ---------------------------------------------------------------------------

class TestGetPeaksForFile:
    def test_returns_correct_peaks(self) -> None:
        db = ReferenceDatabase()
        peaks_a = _make_peaks(8, seed=0)
        peaks_b = _make_peaks(12, seed=1)
        fid_a = db.add_file("a.wav", peaks_a, [])
        fid_b = db.add_file("b.wav", peaks_b, [])
        db.finalize()

        result_a = db.get_peaks_for_file(fid_a)
        result_b = db.get_peaks_for_file(fid_b)
        np.testing.assert_array_equal(result_a, peaks_a)
        np.testing.assert_array_equal(result_b, peaks_b)

    def test_peaks_shape(self) -> None:
        db = ReferenceDatabase()
        peaks = _make_peaks(10)
        fid = db.add_file("a.wav", peaks, [])
        db.finalize()
        result = db.get_peaks_for_file(fid)
        assert result.shape == (10, 2)

    def test_peaks_dtype_float32(self) -> None:
        db = ReferenceDatabase()
        fid = db.add_file("a.wav", _make_peaks(), [])
        db.finalize()
        result = db.get_peaks_for_file(fid)
        assert result.dtype == np.float32

    def test_unknown_file_id_raises(self) -> None:
        db, _ = _populated_db(2)
        with pytest.raises(KeyError):
            db.get_peaks_for_file(9999)

    def test_no_peaks_file(self) -> None:
        """Datei ohne Peaks: Ergebnis ist leeres (0, 2)-Array."""
        db = ReferenceDatabase()
        fid = db.add_file("silent.wav", np.empty((0, 2), dtype=np.float32), [])
        db.finalize()
        result = db.get_peaks_for_file(fid)
        assert result.shape == (0, 2)

    def test_peaks_isolation(self) -> None:
        """Peaks zweier Dateien dürfen sich nicht vermischen."""
        db = ReferenceDatabase()
        peaks_a = np.array([[10.0, 5.0], [20.0, 8.0]], dtype=np.float32)
        peaks_b = np.array([[100.0, 50.0], [200.0, 80.0]], dtype=np.float32)
        fid_a = db.add_file("a.wav", peaks_a, [])
        fid_b = db.add_file("b.wav", peaks_b, [])
        db.finalize()
        np.testing.assert_array_equal(db.get_peaks_for_file(fid_a), peaks_a)
        np.testing.assert_array_equal(db.get_peaks_for_file(fid_b), peaks_b)


# ---------------------------------------------------------------------------
# 7. get_file_name / get_file_id / get_file_info
# ---------------------------------------------------------------------------

class TestFidindex:
    def test_get_file_name(self) -> None:
        db = ReferenceDatabase()
        fid = db.add_file("mysong.wav", _make_peaks(), _make_quad_records())
        db.finalize()
        assert db.get_file_name(fid) == "mysong.wav"

    def test_get_file_id(self) -> None:
        db = ReferenceDatabase()
        fid = db.add_file("mysong.wav", _make_peaks(), _make_quad_records())
        db.finalize()
        assert db.get_file_id("mysong.wav") == fid

    def test_unknown_file_name_raises(self) -> None:
        db, _ = _populated_db(2)
        with pytest.raises(KeyError):
            db.get_file_id("nonexistent.wav")

    def test_unknown_file_id_raises(self) -> None:
        db, _ = _populated_db(2)
        with pytest.raises(KeyError):
            db.get_file_name(9999)

    def test_get_file_info_keys(self) -> None:
        db = ReferenceDatabase()
        db.add_file("a.wav", _make_peaks(5), _make_quad_records(3), duration_sec=2.5)
        db.finalize()
        info = db.get_file_info(0)
        assert "file_name" in info
        assert "duration_sec" in info
        assert "n_peaks" in info
        assert "n_quads" in info

    def test_get_file_info_values(self) -> None:
        db = ReferenceDatabase()
        peaks = _make_peaks(7)
        quads = _make_quad_records(4)
        db.add_file("b.wav", peaks, quads, duration_sec=3.0)
        db.finalize()
        info = db.get_file_info(0)
        assert info["file_name"] == "b.wav"
        assert info["n_peaks"] == 7
        assert info["n_quads"] == 4
        assert info["duration_sec"] == pytest.approx(3.0)

    def test_file_info_unknown_raises(self) -> None:
        db, _ = _populated_db(1)
        with pytest.raises(KeyError):
            db.get_file_info(9999)


# ---------------------------------------------------------------------------
# 8. Persistenz: save / load
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_files(self, tmp_path: Path) -> None:
        db, _ = _populated_db(2)
        db.save(tmp_path / "mydb")
        db_dir = tmp_path / "mydb"
        assert any(db_dir.iterdir())

    def test_load_is_finalized(self, tmp_path: Path) -> None:
        db, _ = _populated_db(2)
        db.save(tmp_path / "db")
        loaded = ReferenceDatabase.load(tmp_path / "db")
        assert loaded.is_finalized is True

    def test_load_same_n_files(self, tmp_path: Path) -> None:
        db, _ = _populated_db(3)
        db.save(tmp_path / "db")
        loaded = ReferenceDatabase.load(tmp_path / "db")
        assert loaded.n_files == db.n_files

    def test_load_same_n_records(self, tmp_path: Path) -> None:
        db, _ = _populated_db(3)
        db.save(tmp_path / "db")
        loaded = ReferenceDatabase.load(tmp_path / "db")
        assert loaded.n_records == db.n_records

    def test_load_same_n_peaks(self, tmp_path: Path) -> None:
        db, _ = _populated_db(3)
        db.save(tmp_path / "db")
        loaded = ReferenceDatabase.load(tmp_path / "db")
        assert loaded.n_peaks == db.n_peaks

    def test_load_same_file_names(self, tmp_path: Path) -> None:
        db, names = _populated_db(3)
        db.save(tmp_path / "db")
        loaded = ReferenceDatabase.load(tmp_path / "db")
        for name in names:
            assert loaded.get_file_id(name) == db.get_file_id(name)

    def test_load_same_peaks(self, tmp_path: Path) -> None:
        db = ReferenceDatabase()
        peaks = _make_peaks(10, seed=99)
        fid = db.add_file("x.wav", peaks, _make_quad_records(seed=99))
        db.finalize()
        db.save(tmp_path / "db")

        loaded = ReferenceDatabase.load(tmp_path / "db")
        result = loaded.get_peaks_for_file(fid)
        np.testing.assert_array_equal(result, peaks)

    def test_query_radius_after_load(self, tmp_path: Path) -> None:
        """Nach save/load muss die Suche funktionieren."""
        known_hash = np.array([0.3, 0.4, 0.7, 0.6], dtype=np.float32)
        rec = QuadRecord(
            root_point=np.zeros(2, dtype=np.float32),
            quad_size=np.ones(2, dtype=np.float32),
            hash=known_hash,
            file_id=0,
        )
        db = ReferenceDatabase()
        db.add_file("s.wav", _make_peaks(), [rec])
        db.finalize()
        db.save(tmp_path / "db")

        loaded = ReferenceDatabase.load(tmp_path / "db")
        results = loaded.query_radius(known_hash.reshape(1, 4))
        assert len(results[0]) >= 1

    def test_save_requires_finalized(self, tmp_path: Path) -> None:
        db = ReferenceDatabase()
        db.add_file("a.wav", _make_peaks(), _make_quad_records())
        with pytest.raises(RuntimeError):
            db.save(tmp_path / "db")

    def test_load_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises((FileNotFoundError, Exception)):
            ReferenceDatabase.load(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# 9. memory_usage_mb
# ---------------------------------------------------------------------------

class TestMemoryUsage:
    def test_returns_dict_with_keys(self) -> None:
        db, _ = _populated_db(2)
        stats = db.memory_usage_mb()
        assert "peakfile_mb" in stats
        assert "refrecords_mb" in stats
        assert "searchtree_est_mb" in stats
        assert "total_mb" in stats

    def test_total_is_sum_of_parts(self) -> None:
        db, _ = _populated_db(2)
        stats = db.memory_usage_mb()
        expected = stats["peakfile_mb"] + stats["refrecords_mb"] + stats["searchtree_est_mb"]
        assert stats["total_mb"] == pytest.approx(expected, rel=1e-5)

    def test_non_negative_values(self) -> None:
        db, _ = _populated_db(2)
        for val in db.memory_usage_mb().values():
            assert val >= 0.0

    def test_more_records_more_memory(self) -> None:
        db_small = ReferenceDatabase()
        db_small.add_file("a.wav", _make_peaks(5), _make_quad_records(5, seed=0))
        db_small.finalize()

        db_large = ReferenceDatabase()
        db_large.add_file("a.wav", _make_peaks(5), _make_quad_records(50, seed=1))
        db_large.finalize()

        assert (
            db_large.memory_usage_mb()["total_mb"]
            > db_small.memory_usage_mb()["total_mb"]
        )

    def test_empty_db_zero_memory(self) -> None:
        db = ReferenceDatabase()
        db.finalize()
        stats = db.memory_usage_mb()
        assert stats["peakfile_mb"] == pytest.approx(0.0, abs=1e-6)
        assert stats["refrecords_mb"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 10. REFRECORD_DTYPE (36 Bytes pro Record)
# ---------------------------------------------------------------------------

class TestRefrecordDtype:
    def test_dtype_itemsize(self) -> None:
        """Paper: 'a size of 36 bytes' pro Record (Section V)."""
        assert REFRECORD_DTYPE.itemsize == 36

    def test_dtype_has_hash_field(self) -> None:
        assert 'hash' in REFRECORD_DTYPE.names

    def test_dtype_has_file_id_field(self) -> None:
        assert 'file_id' in REFRECORD_DTYPE.names

    def test_hash_field_shape(self) -> None:
        """Hash-Feld ist float32 mit Form (4,)."""
        assert REFRECORD_DTYPE['hash'].shape == (4,)
        assert REFRECORD_DTYPE['hash'].base == np.float32

    def test_file_id_is_uint32(self) -> None:
        assert REFRECORD_DTYPE['file_id'].base == np.uint32
