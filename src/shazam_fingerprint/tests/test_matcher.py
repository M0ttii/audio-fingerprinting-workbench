"""
Tests für matcher.py.

Prüft korrektes Matching bei identischem Audio, Nicht-Matching bei fremdem Audio
und die Zeitkohärenz-Logik anhand synthetischer Fingerprints.
"""

import pytest

from shazam_fingerprint import config
from shazam_fingerprint.database import FingerprintDatabase
from shazam_fingerprint.matcher import MatchResult, _histogram_peak_score, match


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def empty_db() -> FingerprintDatabase:
    """Leere Datenbank."""
    return FingerprintDatabase()


@pytest.fixture
def single_song_db() -> FingerprintDatabase:
    """Datenbank mit einem einzigen Song."""
    db = FingerprintDatabase()
    fps = [(hash_val, t) for hash_val, t in _make_fingerprints("song_a", n=50)]
    db.insert("song_a", fps)
    return db


@pytest.fixture
def multi_song_db() -> FingerprintDatabase:
    """Datenbank mit drei verschiedenen Songs."""
    db = FingerprintDatabase()
    db.insert("song_a", _make_fingerprints("song_a", n=60))
    db.insert("song_b", _make_fingerprints("song_b", n=60))
    db.insert("song_c", _make_fingerprints("song_c", n=60))
    return db


# ======================================================================
# Hilfsfunktionen
# ======================================================================

def _make_fingerprints(
    song_id: str,
    n: int,
    offset: int = 0,
) -> list[tuple[int, int]]:
    """Erzeugt deterministisch eindeutige Fingerprints für einen Song.

    Die Hashes sind so gewählt, dass verschiedene Songs keine Kollisionen
    teilen (durch song_id-basierten Seed-Offset).

    Args:
        song_id: Name des Songs (bestimmt den Hash-Bereich).
        n: Anzahl der Fingerprints.
        offset: Zeitversatz der Anchor-Times (in Frames).

    Returns:
        Liste von (hash_value, anchor_time)-Tupeln.
    """
    seed = sum(ord(c) for c in song_id) * 1000
    return [(seed + i, i + offset) for i in range(n)]


def _build_coherent_query(
    song_id: str,
    n_matching: int,
    time_offset: int = 0,
) -> list[tuple[int, int]]:
    """Erzeugt Query-Fingerprints die zeitkohärent zu einem DB-Song passen.

    Die Hashes stimmen mit dem Song überein, aber die Anchor-Times der Query
    sind um `time_offset` verschoben — genau wie beim echten Shazam-Szenario,
    wo ein Ausschnitt ab Sekunde X verglichen wird.

    Args:
        song_id: Song-ID, dessen Hashes imitiert werden.
        n_matching: Anzahl übereinstimmender Hashes.
        time_offset: Zeitversatz der Query-Anchor-Times.

    Returns:
        Query-Fingerprints mit zeitkohärentem Zeitoffset zum DB-Song.
    """
    seed = sum(ord(c) for c in song_id) * 1000
    # Hash-Werte identisch mit DB, aber anchor_time um time_offset verschoben
    return [(seed + i, i + time_offset) for i in range(n_matching)]


# ======================================================================
# Rückgabetyp und Grundstruktur
# ======================================================================

class TestReturnType:
    def test_returns_match_result(self, single_song_db: FingerprintDatabase) -> None:
        """match() gibt ein MatchResult-Objekt zurück."""
        fps = _make_fingerprints("song_a", n=20)
        result = match(fps, single_song_db)
        assert isinstance(result, MatchResult)

    def test_match_result_has_required_fields(
        self, single_song_db: FingerprintDatabase
    ) -> None:
        """MatchResult enthält alle notwendigen Felder."""
        fps = _make_fingerprints("song_a", n=20)
        result = match(fps, single_song_db)
        assert hasattr(result, "best_match")
        assert hasattr(result, "best_score")
        assert hasattr(result, "all_scores")
        assert hasattr(result, "match_found")
        assert hasattr(result, "num_query_hashes")
        assert hasattr(result, "processing_time_ms")

    def test_empty_query_returns_no_match(
        self, single_song_db: FingerprintDatabase
    ) -> None:
        """Leere Query-Fingerprints liefern kein Match."""
        result = match([], single_song_db)
        assert result.best_match is None
        assert result.match_found is False
        assert result.best_score == 0

    def test_num_query_hashes_correct(
        self, single_song_db: FingerprintDatabase
    ) -> None:
        """num_query_hashes entspricht der Länge der übergebenen Fingerprints."""
        fps = _make_fingerprints("song_a", n=30)
        result = match(fps, single_song_db)
        assert result.num_query_hashes == 30

    def test_processing_time_is_positive(
        self, single_song_db: FingerprintDatabase
    ) -> None:
        """processing_time_ms ist ein nicht-negativer float."""
        fps = _make_fingerprints("song_a", n=20)
        result = match(fps, single_song_db)
        assert isinstance(result.processing_time_ms, float)
        assert result.processing_time_ms >= 0.0


# ======================================================================
# Identisches Audio: Score muss hoch sein, Match korrekt
# ======================================================================

class TestIdenticalAudio:
    def test_identical_fingerprints_match(
        self, single_song_db: FingerprintDatabase
    ) -> None:
        """Identische Fingerprints (Query = DB-Song) werden erkannt.

        Wenn Query-Hashes exakt mit den DB-Hashes übereinstimmen und
        zeitkohärent sind (δt konstant), muss ein Match gefunden werden.
        """
        fps = _make_fingerprints("song_a", n=50)
        result = match(fps, single_song_db)
        assert result.match_found is True
        assert result.best_match == "song_a"

    def test_identical_audio_high_score(
        self, single_song_db: FingerprintDatabase
    ) -> None:
        """Identische Fingerprints erzeugen einen hohen Score.

        Der Score muss mindestens MATCH_THRESHOLD überschreiten.
        """
        fps = _make_fingerprints("song_a", n=50)
        result = match(fps, single_song_db)
        assert result.best_score >= config.MATCH_THRESHOLD

    def test_correct_song_identified_in_multi_song_db(
        self, multi_song_db: FingerprintDatabase
    ) -> None:
        """Der korrekte Song wird aus einer Mehrfach-Song-Datenbank identifiziert."""
        for song_id in ("song_a", "song_b", "song_c"):
            fps = _make_fingerprints(song_id, n=50)
            result = match(fps, multi_song_db)
            assert result.match_found is True
            assert result.best_match == song_id, (
                f"Erwartet '{song_id}', gefunden '{result.best_match}'"
            )

    def test_best_score_highest_among_all_scores(
        self, multi_song_db: FingerprintDatabase
    ) -> None:
        """Der best_score ist der höchste Wert in all_scores."""
        fps = _make_fingerprints("song_b", n=50)
        result = match(fps, multi_song_db)
        assert result.best_score == max(result.all_scores.values())


# ======================================================================
# Fremdes Audio: Kein Match
# ======================================================================

class TestNoMatch:
    def test_unrelated_fingerprints_no_match(
        self, single_song_db: FingerprintDatabase
    ) -> None:
        """Vollständig fremde Fingerprints erzeugen keinen Match.

        Hashes die nicht in der DB vorhanden sind, liefern keine Treffer.
        """
        # Hashes im Bereich, der keinem DB-Song entspricht
        foreign_fps = [(0xDEAD0000 + i, i) for i in range(50)]
        result = match(foreign_fps, single_song_db)
        assert result.match_found is False
        assert result.best_match is None

    def test_no_match_empty_db(self) -> None:
        """Query gegen leere Datenbank ergibt keinen Match."""
        db = FingerprintDatabase()
        fps = _make_fingerprints("any_song", n=30)
        result = match(fps, db)
        assert result.match_found is False
        assert result.best_match is None

    def test_below_min_hash_matches_no_match(
        self, single_song_db: FingerprintDatabase
    ) -> None:
        """Weniger als MIN_HASH_MATCHES Treffer führen zu keinem Match.

        Die Vorfilterung in match() schließt Songs mit zu wenigen Hash-Matches
        vom Histogram-Scoring aus.
        """
        # Nur MIN_HASH_MATCHES - 1 übereinstimmende Hashes
        n = max(config.MIN_HASH_MATCHES - 1, 1)
        fps = _make_fingerprints("song_a", n=n)
        result = match(fps, single_song_db)
        assert result.match_found is False

    def test_score_below_threshold_no_match(self) -> None:
        """Score unterhalb MATCH_THRESHOLD liefert match_found=False.

        Direkte Prüfung: best_score und match_found sind konsistent.
        """
        db = FingerprintDatabase()
        # Song mit vielen Hashes in der DB
        db.insert("song_x", _make_fingerprints("song_x", n=100))

        # Query mit MIN_HASH_MATCHES Treffern, aber inkohärentem δt
        # → Histogram-Peak bleibt bei 1 (kein Kohärenz-Peak)
        seed = sum(ord(c) for c in "song_x") * 1000
        incoherent_fps = [(seed + i, i * 100) for i in range(config.MIN_HASH_MATCHES)]
        result = match(incoherent_fps, db)
        # Score ≤ 1 pro Hash (alle δt verschieden) → kein Peak
        assert result.match_found is False


# ======================================================================
# Zeitkohärenz: Künstliche Time-Pairs mit bekanntem Offset
# ======================================================================

class TestTimeCohérence:
    def test_coherent_time_offset_produces_high_score(self) -> None:
        """Zeitkohärente Hashes (konstantes δt) erzeugen hohen Histogram-Peak.

        Wang Section 2.3: Ein echter Match zeigt eine starke Häufung im
        δt-Histogram bei einem bestimmten Zeitoffset.
        """
        db = FingerprintDatabase()
        known_offset = 42  # DB-Anchor-Time liegt um 42 Frames vor Query-Anchor-Time

        # Song: Anchor-Times 0..n-1
        n = 40
        db_fps = [(1000 + i, i) for i in range(n)]
        db.insert("target_song", db_fps)

        # Query: gleiche Hashes, Anchor-Times um known_offset verschoben
        # → δt = t_db - t_query = i - (i - known_offset) = known_offset (konstant)
        query_fps = [(1000 + i, i - known_offset) for i in range(n)]
        result = match(query_fps, db)

        assert result.match_found is True
        assert result.best_match == "target_song"
        assert result.best_score >= config.MATCH_THRESHOLD

    def test_incoherent_time_offsets_low_score(self) -> None:
        """Zeitinkohärente Hashes (zufällige δt-Werte) erzeugen niedrigen Score.

        Wang Fig. 2B: Ohne Zeitkohärenz ist das δt-Histogram flach.
        """
        db = FingerprintDatabase()
        n = 40
        db_fps = [(2000 + i, i) for i in range(n)]
        db.insert("other_song", db_fps)

        # Query: gleiche Hashes, aber Anchor-Times zufällig verteilt
        # → δt = t_db - t_query variiert stark → kein Histogram-Peak
        query_fps = [(2000 + i, i * 17 + 3) for i in range(n)]  # Primzahl-Schritt
        result = match(query_fps, db)
        # Score sollte niedrig sein (≤ 1 pro Hash bei völlig verschiedenen δt)
        if result.all_scores:
            assert result.best_score <= max(3, config.MATCH_THRESHOLD // 3)

    def test_partial_overlap_still_matches(self) -> None:
        """Ein Query-Ausschnitt (Teilüberlappung) wird trotzdem erkannt.

        Shazam erkennt Songs auch anhand kurzer Ausschnitte. Die zeitkohärenten
        Hashes im Überschneidungsbereich genügen für ein Match.
        """
        db = FingerprintDatabase()
        n_total = 100
        n_query = 40   # Nur 40 % der Song-Hashes in der Query

        db_fps = [(3000 + i, i) for i in range(n_total)]
        db.insert("long_song", db_fps)

        # Query: Ausschnitt ab Frame 30, gleiche Hashes
        start = 30
        query_fps = [(3000 + start + i, i) for i in range(n_query)]
        result = match(query_fps, db)

        assert result.match_found is True
        assert result.best_match == "long_song"

    def test_known_delta_t_value(self) -> None:
        """Das δt im Histogram-Peak entspricht dem tatsächlichen Zeitversatz.

        Wenn Query-Fingerprints gegenüber der DB um `offset` Frames verschoben
        sind, muss der dominierende δt-Wert dem Versatz entsprechen.
        Dieser Test prüft _histogram_peak_score() direkt.
        """
        known_offset = 77
        n = 50
        # Alle δt-Werte sind identisch = known_offset
        delta_ts = [known_offset] * n
        score = _histogram_peak_score(delta_ts)
        assert score == n

    def test_multiple_offsets_score_is_peak_height(self) -> None:
        """Bei gemischten δt-Werten ist der Score die Höhe des höchsten Peaks.

        Wang: "The score of the match is the number of matching points
        in the histogram peak."
        """
        peak_height = 20
        noise_count = 10
        dominant_dt = 100

        # 20 × dominant_dt + 10 verschiedene Zufallswerte
        delta_ts = [dominant_dt] * peak_height
        delta_ts += list(range(200, 200 + noise_count))  # Rauschen

        score = _histogram_peak_score(delta_ts)
        assert score == peak_height

    def test_winner_has_most_coherent_offset(self) -> None:
        """Der Match-Winner hat den zeitkohärentesten Hash-Overlap.

        Zwei Songs in der DB: song_target mit vielen kohärenten Hashes,
        song_noise mit nur wenigen. Die Query muss song_target wählen.
        """
        db = FingerprintDatabase()
        offset = 10

        # song_target: 50 kohärente Hashes
        n_target = 50
        db.insert("song_target", [(4000 + i, i) for i in range(n_target)])

        # song_noise: wenige Hashes
        db.insert("song_noise", [(5000 + i, i) for i in range(10)])

        # Query: zeitkohärent zu song_target (Offset = 10)
        query_fps = [(4000 + i, i - offset) for i in range(n_target)]
        result = match(query_fps, db)

        assert result.best_match == "song_target"
        assert result.all_scores.get("song_target", 0) > result.all_scores.get("song_noise", 0)


# ======================================================================
# Histogram-Peak-Score (interne Funktion)
# ======================================================================

class TestHistogramPeakScore:
    def test_empty_list_returns_zero(self) -> None:
        """Leere δt-Liste liefert Score 0."""
        assert _histogram_peak_score([]) == 0

    def test_single_value_returns_one(self) -> None:
        """Eine einziger δt-Wert liefert Score 1."""
        assert _histogram_peak_score([42]) == 1

    def test_all_identical_returns_count(self) -> None:
        """Alle identischen δt-Werte: Score = Anzahl der Werte."""
        n = 30
        assert _histogram_peak_score([99] * n) == n

    def test_all_different_returns_one(self) -> None:
        """Alle verschiedenen δt-Werte: Score = 1 (kein Peak)."""
        unique_dts = list(range(50))
        assert _histogram_peak_score(unique_dts) == 1

    def test_mixed_returns_peak_height(self) -> None:
        """Gemischte δt-Werte: Score = Häufigkeit des häufigsten Werts."""
        dts = [5] * 15 + [10] * 7 + [20] * 3
        assert _histogram_peak_score(dts) == 15

    def test_negative_delta_t_handled(self) -> None:
        """Negative δt-Werte (Query nach DB-Zeit) werden korrekt verarbeitet."""
        dts = [-5] * 10 + [3] * 4
        assert _histogram_peak_score(dts) == 10


# ======================================================================
# all_scores enthält alle Kandidaten
# ======================================================================

class TestAllScores:
    def test_all_scores_contains_matched_song(
        self, multi_song_db: FingerprintDatabase
    ) -> None:
        """all_scores enthält den gematchten Song."""
        fps = _make_fingerprints("song_a", n=50)
        result = match(fps, multi_song_db)
        assert "song_a" in result.all_scores

    def test_all_scores_values_are_positive_ints(
        self, multi_song_db: FingerprintDatabase
    ) -> None:
        """Alle Scores in all_scores sind positive Integer."""
        fps = _make_fingerprints("song_b", n=50)
        result = match(fps, multi_song_db)
        for song_id, score in result.all_scores.items():
            assert isinstance(score, int), f"Score für '{song_id}' ist kein int"
            assert score > 0, f"Score für '{song_id}' ist nicht positiv"

    def test_match_found_consistent_with_threshold(
        self, multi_song_db: FingerprintDatabase
    ) -> None:
        """match_found ist konsistent mit best_score >= MATCH_THRESHOLD."""
        fps = _make_fingerprints("song_c", n=50)
        result = match(fps, multi_song_db)
        expected = result.best_score >= config.MATCH_THRESHOLD
        assert result.match_found == expected
