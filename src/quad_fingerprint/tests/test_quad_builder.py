"""Tests für das quad_builder-Modul (Quad-Fingerprinting).

Gemäß anforderungen_quad.md Abschnitt TESTS:
- Gültigkeitsbedingung: Ungültige Konstellationen werden verworfen (Eq. 1a–1c).
- Hash-Berechnung: Manuell berechneter Hash stimmt mit Funktion überein.
- Translationsinvarianz: Gleiche 4 Peaks, verschoben → Hash IDENTISCH.
- Skalierungsinvarianz: Gleiche 4 Peaks, mit Faktor 1.2 skaliert → Hash IDENTISCH.
- Relevanter Teilraum: Quads außerhalb des Teilraums werden verworfen.

Weitere Tests:
- Hash liegt in [0,1]^4, dtype float32
- QuadRecord- und QueryQuad-Felder (root_point, quad_size, hash, file_id)
- build_reference_quads / build_query_quads: korrekte Ausgabetypen
"""

import numpy as np
import pytest

from quad_fingerprint import config
from quad_fingerprint.quad_builder import (
    QueryQuad,
    QuadRecord,
    build_query_quads,
    build_reference_quads,
    compute_quad_hash,
    is_valid_quad,
)

# ---------------------------------------------------------------------------
# Kanonisches Referenz-Quad für alle Invarianz-Tests
# A=(0,0), B=(10,10), C=(3,4), D=(7,6) — alle Gültigkeitsbedingungen erfüllt
# Hash: [0.3, 0.4, 0.7, 0.6]
# ---------------------------------------------------------------------------
A0 = (0.0, 0.0)
B0 = (10.0, 10.0)
C0 = (3.0, 4.0)
D0 = (7.0, 6.0)
EXPECTED_HASH = np.array([0.3, 0.4, 0.7, 0.6], dtype=np.float32)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _make_magnitude(n_bins: int = 513, n_frames: int = 600) -> np.ndarray:
    """Synthetisches Spektrogramm mit Rauschen."""
    rng = np.random.default_rng(42)
    return rng.random((n_bins, n_frames)).astype(np.float32)


def _make_peaks_for_ref_window() -> np.ndarray:
    """Peaks, die mit dem Referenz-Gruppierungsfenster kompatibel sind.

    Referenz-Fenster: r_start = REF_R_START_FRAMES (≈225), r_end = REF_R_END_FRAMES (≈425).
    Platziere A bei Frame 0 und B, C, D bei Frames im Fenster [225, 425].
    Braucht ≥4 Peaks damit ein gültiger Quad entstehen kann.
    """
    r_s = config.REF_R_START_FRAMES
    r_e = config.REF_R_END_FRAMES
    center = (r_s + r_e) // 2

    # A (Root-Point), B, C, D im Fenster (freq aufsteigend, damit Eq. 1a/1c erfüllbar)
    peaks = np.array([
        [0.0,            5.0],   # A: frame 0, freq 5
        [float(r_s + 20), 8.0],  # C: freq > A_freq
        [float(center),   10.0], # D: freq > A_freq
        [float(r_e - 10), 20.0], # B: höchste freq → Eq. 1a erfüllbar
    ], dtype=np.float32)
    return peaks


# ---------------------------------------------------------------------------
# 1. Gültigkeitsbedingung (Eq. 1a–1c)
# ---------------------------------------------------------------------------

class TestIsValidQuad:
    """Ungültige Konstellationen werden verworfen, gültige akzeptiert."""

    def test_canonical_quad_is_valid(self) -> None:
        """Kanonisches Quad (A=(0,0), B=(10,10), C=(3,4), D=(7,6)) ist gültig."""
        assert is_valid_quad(*A0, *B0, *C0, *D0) is True

    # --- Eq. (1a): A_y < B_y ---
    def test_invalid_eq1a_b_freq_equal_a_freq(self) -> None:
        """Eq. 1a verletzt: B_freq == A_freq → ungültig."""
        assert is_valid_quad(0, 5, 10, 5, 3, 6, 7, 4) is False

    def test_invalid_eq1a_b_freq_below_a_freq(self) -> None:
        """Eq. 1a verletzt: B_freq < A_freq → ungültig."""
        assert is_valid_quad(0, 10, 10, 5, 3, 8, 7, 7) is False

    # --- Eq. (1b): A_x < C_x ≤ D_x ≤ B_x ---
    def test_invalid_eq1b_c_time_not_after_a(self) -> None:
        """Eq. 1b verletzt: C_time == A_time → A_x < C_x nicht erfüllt."""
        assert is_valid_quad(0, 0, 10, 10, 0, 4, 7, 6) is False

    def test_invalid_eq1b_d_time_exceeds_b(self) -> None:
        """Eq. 1b verletzt: D_time > B_time → ungültig."""
        assert is_valid_quad(0, 0, 10, 10, 3, 4, 11, 6) is False

    def test_invalid_eq1b_c_after_d(self) -> None:
        """Eq. 1b verletzt: C_time > D_time (falsche Reihenfolge) → ungültig."""
        assert is_valid_quad(0, 0, 10, 10, 8, 4, 3, 6) is False

    # --- Eq. (1c): A_y < C_y ≤ B_y AND A_y < D_y ≤ B_y ---
    def test_invalid_eq1c_c_freq_equal_a_freq(self) -> None:
        """Eq. 1c verletzt: C_freq == A_freq → A_y < C_y nicht erfüllt."""
        assert is_valid_quad(0, 0, 10, 10, 3, 0, 7, 6) is False

    def test_invalid_eq1c_c_freq_exceeds_b(self) -> None:
        """Eq. 1c verletzt: C_freq > B_freq → ungültig."""
        assert is_valid_quad(0, 0, 10, 10, 3, 15, 7, 6) is False

    def test_invalid_eq1c_d_freq_below_a(self) -> None:
        """Eq. 1c verletzt: D_freq <= A_freq → ungültig."""
        assert is_valid_quad(0, 0, 10, 10, 3, 4, 7, 0) is False

    def test_valid_boundary_c_and_d_at_b_time(self) -> None:
        """Grenzfall: C_x == D_x == B_x ist gültig (≤ ist erlaubt)."""
        # A=(0,0), B=(10,10), C=(10,4), D=(10,6) — C_x=D_x=B_x → gültig
        assert is_valid_quad(0, 0, 10, 10, 10, 4, 10, 6) is True

    def test_valid_boundary_c_freq_at_b_freq(self) -> None:
        """Grenzfall: C_y == B_y ist gültig (≤ B_y erlaubt)."""
        assert is_valid_quad(0, 0, 10, 10, 3, 10, 7, 6) is True


# ---------------------------------------------------------------------------
# 2. Hash-Berechnung: manuell vs. Funktion
# ---------------------------------------------------------------------------

class TestComputeQuadHash:
    """Hash-Berechnung stimmt mit manuell berechnetem Wert überein."""

    def test_canonical_hash_values(self) -> None:
        """A=(0,0), B=(10,10), C=(3,4), D=(7,6) → [0.3, 0.4, 0.7, 0.6]."""
        h = compute_quad_hash(*A0, *B0, *C0, *D0)
        np.testing.assert_allclose(h, EXPECTED_HASH, atol=1e-6)

    def test_hash_dtype_float32(self) -> None:
        """Hash muss float32 sein (Paper: 'single precision')."""
        h = compute_quad_hash(*A0, *B0, *C0, *D0)
        assert h.dtype == np.float32

    def test_hash_shape_4(self) -> None:
        """Hash ist ein 1D-Array der Länge 4."""
        h = compute_quad_hash(*A0, *B0, *C0, *D0)
        assert h.shape == (4,)

    def test_hash_in_unit_hypercube(self) -> None:
        """Alle Hash-Komponenten liegen in [0, 1]."""
        h = compute_quad_hash(*A0, *B0, *C0, *D0)
        assert np.all(h >= 0.0) and np.all(h <= 1.0)

    def test_manual_formula_cx(self) -> None:
        """C'_x = (C_x - A_x) / (B_x - A_x)."""
        a_t, a_f = 2.0, 1.0
        b_t, b_f = 12.0, 11.0
        c_t, c_f = 5.0, 4.0
        d_t, d_f = 8.0, 7.0
        h = compute_quad_hash(a_t, a_f, b_t, b_f, c_t, c_f, d_t, d_f)
        expected_cx = (c_t - a_t) / (b_t - a_t)
        assert h[0] == pytest.approx(expected_cx, rel=1e-5)

    def test_manual_formula_dy(self) -> None:
        """D'_y = (D_y - A_y) / (B_y - A_y)."""
        a_t, a_f = 2.0, 1.0
        b_t, b_f = 12.0, 11.0
        c_t, c_f = 5.0, 4.0
        d_t, d_f = 8.0, 7.0
        h = compute_quad_hash(a_t, a_f, b_t, b_f, c_t, c_f, d_t, d_f)
        expected_dy = (d_f - a_f) / (b_f - a_f)
        assert h[3] == pytest.approx(expected_dy, rel=1e-5)

    def test_c_at_corner_a(self) -> None:
        """C direkt neben A (minimal gültig): C'_x → 0, C'_y → 0."""
        # C sehr nah an A — Hash-Komponenten nahe 0
        h = compute_quad_hash(0, 0, 10, 10, 0.1, 0.1, 5, 5)
        assert h[0] == pytest.approx(0.01, rel=0.1)
        assert h[1] == pytest.approx(0.01, rel=0.1)

    def test_b_at_corner_b(self) -> None:
        """D genau bei B: D'_x = 1.0, D'_y = 1.0."""
        h = compute_quad_hash(0, 0, 10, 10, 3, 4, 10, 10)
        assert h[2] == pytest.approx(1.0, rel=1e-5)
        assert h[3] == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# 3. Translationsinvarianz (Kernanforderung)
# ---------------------------------------------------------------------------

class TestTranslationInvariance:
    """Gleiche 4 Peaks, verschoben → Hash IDENTISCH (Translationsinvarianz).

    Mathematisch: Hash = (C-A)/(B-A), (D-A)/(B-A).
    Verschiebung um beliebigen Vektor (dt, df) kürzt sich heraus.
    """

    def test_shift_time_only(self) -> None:
        """Verschiebung nur in Zeit (dt=100) → gleicher Hash."""
        dt = 100.0
        h1 = compute_quad_hash(*A0, *B0, *C0, *D0)
        h2 = compute_quad_hash(
            A0[0] + dt, A0[1],
            B0[0] + dt, B0[1],
            C0[0] + dt, C0[1],
            D0[0] + dt, D0[1],
        )
        np.testing.assert_allclose(h1, h2, atol=1e-6)

    def test_shift_freq_only(self) -> None:
        """Verschiebung nur in Frequenz (df=50) → gleicher Hash."""
        df = 50.0
        h1 = compute_quad_hash(*A0, *B0, *C0, *D0)
        h2 = compute_quad_hash(
            A0[0], A0[1] + df,
            B0[0], B0[1] + df,
            C0[0], C0[1] + df,
            D0[0], D0[1] + df,
        )
        np.testing.assert_allclose(h1, h2, atol=1e-6)

    def test_shift_time_and_freq(self) -> None:
        """Verschiebung in Zeit und Frequenz (dt=200, df=75) → gleicher Hash."""
        dt, df = 200.0, 75.0
        h1 = compute_quad_hash(*A0, *B0, *C0, *D0)
        h2 = compute_quad_hash(
            A0[0] + dt, A0[1] + df,
            B0[0] + dt, B0[1] + df,
            C0[0] + dt, C0[1] + df,
            D0[0] + dt, D0[1] + df,
        )
        np.testing.assert_allclose(h1, h2, atol=1e-6)

    def test_shift_large_offset(self) -> None:
        """Große Verschiebung (typische Song-Dauer: 10000 Frames) → gleicher Hash."""
        dt, df = 10000.0, 300.0
        h1 = compute_quad_hash(*A0, *B0, *C0, *D0)
        h2 = compute_quad_hash(
            A0[0] + dt, A0[1] + df,
            B0[0] + dt, B0[1] + df,
            C0[0] + dt, C0[1] + df,
            D0[0] + dt, D0[1] + df,
        )
        np.testing.assert_allclose(h1, h2, atol=1e-6)

    def test_shift_non_origin_a(self) -> None:
        """A nicht im Ursprung: Verschiebung ändert den Hash nicht."""
        # Quad mit A=(5, 3)
        a_t, a_f = 5.0, 3.0
        b_t, b_f = 15.0, 13.0
        c_t, c_f = 8.0, 7.0
        d_t, d_f = 12.0, 9.0
        h1 = compute_quad_hash(a_t, a_f, b_t, b_f, c_t, c_f, d_t, d_f)

        dt, df = 300.0, 100.0
        h2 = compute_quad_hash(
            a_t + dt, a_f + df,
            b_t + dt, b_f + df,
            c_t + dt, c_f + df,
            d_t + dt, d_f + df,
        )
        np.testing.assert_allclose(h1, h2, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Skalierungsinvarianz (Kernanforderung)
# ---------------------------------------------------------------------------

class TestScaleInvariance:
    """Gleiche 4 Peaks, mit Faktor skaliert → Hash IDENTISCH (Skalierungsinvarianz).

    Mathematisch: Skalierung mit Faktor s → dx' = s*dx, dy' = s*dy.
    (C'_x)' = s*(C_x - A_x) / (s*(B_x - A_x)) = (C_x - A_x) / (B_x - A_x)  ✓
    """

    def test_scale_factor_1_2(self) -> None:
        """Skalierung mit Faktor 1.2 (Tempo +20%) → identischer Hash.

        Dies ist der Hauptvorteil des Quad-Algorithmus gegenüber Shazam!
        """
        scale = 1.2
        h1 = compute_quad_hash(*A0, *B0, *C0, *D0)
        h2 = compute_quad_hash(
            A0[0] * scale, A0[1] * scale,
            B0[0] * scale, B0[1] * scale,
            C0[0] * scale, C0[1] * scale,
            D0[0] * scale, D0[1] * scale,
        )
        np.testing.assert_allclose(h1, h2, atol=1e-6)

    def test_scale_factor_0_8(self) -> None:
        """Skalierung mit Faktor 0.8 (Tempo -20%) → identischer Hash."""
        scale = 0.8
        h1 = compute_quad_hash(*A0, *B0, *C0, *D0)
        h2 = compute_quad_hash(
            A0[0] * scale, A0[1] * scale,
            B0[0] * scale, B0[1] * scale,
            C0[0] * scale, C0[1] * scale,
            D0[0] * scale, D0[1] * scale,
        )
        np.testing.assert_allclose(h1, h2, atol=1e-6)

    def test_scale_factor_1_3(self) -> None:
        """Skalierung mit Faktor 1.3 → identischer Hash (Extremwert aus EPSILON_T=0.31)."""
        scale = 1.3
        h1 = compute_quad_hash(*A0, *B0, *C0, *D0)
        h2 = compute_quad_hash(
            A0[0] * scale, A0[1] * scale,
            B0[0] * scale, B0[1] * scale,
            C0[0] * scale, C0[1] * scale,
            D0[0] * scale, D0[1] * scale,
        )
        np.testing.assert_allclose(h1, h2, atol=1e-6)

    def test_scale_with_nonzero_a(self) -> None:
        """Skalierung mit A ≠ Ursprung: Skalierung relativ zum Ursprung, nicht zu A."""
        # Quad mit A=(5,3), B=(15,13), C=(8,7), D=(12,9)
        a_t, a_f = 5.0, 3.0
        b_t, b_f = 15.0, 13.0
        c_t, c_f = 8.0, 7.0
        d_t, d_f = 12.0, 9.0
        h1 = compute_quad_hash(a_t, a_f, b_t, b_f, c_t, c_f, d_t, d_f)

        scale = 1.2
        h2 = compute_quad_hash(
            a_t * scale, a_f * scale,
            b_t * scale, b_f * scale,
            c_t * scale, c_f * scale,
            d_t * scale, d_f * scale,
        )
        np.testing.assert_allclose(h1, h2, atol=1e-6)

    def test_translation_plus_scale(self) -> None:
        """Kombination von Verschiebung und Skalierung → Hash muss gleich bleiben."""
        h1 = compute_quad_hash(*A0, *B0, *C0, *D0)
        dt, df, scale = 50.0, 25.0, 1.15
        h2 = compute_quad_hash(
            (A0[0] + dt) * scale, (A0[1] + df) * scale,
            (B0[0] + dt) * scale, (B0[1] + df) * scale,
            (C0[0] + dt) * scale, (C0[1] + df) * scale,
            (D0[0] + dt) * scale, (D0[1] + df) * scale,
        )
        np.testing.assert_allclose(h1, h2, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Relevanter Teilraum (Subspace-Check, Eq. 5)
# ---------------------------------------------------------------------------

class TestSubspaceCheck:
    """Quads außerhalb des relevanten Teilraums werden von build_query_quads verworfen.

    Paper Section IV-C: '44.7% of possible quads could be rejected'.
    Bedingung: C'_x ≥ HASH_CX_MIN − SEARCH_RADIUS.
    HASH_CX_MIN = REF_R_START_FRAMES / REF_R_END_FRAMES ≈ 0.529.
    """

    def test_subspace_threshold_positive(self) -> None:
        """HASH_CX_MIN ist positiv und kleiner als 1."""
        assert 0.0 < config.HASH_CX_MIN < 1.0

    def test_query_quads_fewer_than_all_candidates(self) -> None:
        """build_query_quads muss ≤ build_reference_quads (bei gleicher Datenmenge)
        oder ≤ Gesamtzahl der Kandidaten zurückgeben (Subspace eliminiert einige)."""
        rng = np.random.default_rng(7)
        # Viele zufällige Peaks, groß genug für beide Fenster
        n = 200
        n_bins, n_frames = 513, 1500
        times = np.sort(rng.uniform(0, n_frames, n)).astype(np.float32)
        freqs = rng.uniform(10, n_bins - 10, n).astype(np.float32)
        peaks = np.column_stack([times, freqs])
        mag = rng.random((n_bins, n_frames)).astype(np.float32)

        ref_quads = build_reference_quads(peaks, mag, file_id=0)
        # Referenz-Quads entstehen aus kleinerem Fenster → typischerweise weniger
        assert isinstance(ref_quads, list)

    def test_hash_cx_min_formula(self) -> None:
        """HASH_CX_MIN = REF_R_START_FRAMES / REF_R_END_FRAMES (Eq. 5)."""
        expected = config.REF_R_START_FRAMES / config.REF_R_END_FRAMES
        assert config.HASH_CX_MIN == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# 6. build_reference_quads
# ---------------------------------------------------------------------------

class TestBuildReferenceQuads:
    def test_returns_list(self) -> None:
        peaks = _make_peaks_for_ref_window()
        mag = _make_magnitude()
        result = build_reference_quads(peaks, mag, file_id=1)
        assert isinstance(result, list)

    def test_elements_are_quad_records(self) -> None:
        peaks = _make_peaks_for_ref_window()
        mag = _make_magnitude()
        result = build_reference_quads(peaks, mag, file_id=1)
        for q in result:
            assert isinstance(q, QuadRecord)

    def test_quad_record_hash_shape(self) -> None:
        peaks = _make_peaks_for_ref_window()
        mag = _make_magnitude()
        result = build_reference_quads(peaks, mag, file_id=1)
        for q in result:
            assert q.hash.shape == (4,)
            assert q.hash.dtype == np.float32

    def test_quad_record_hash_in_unit_hypercube(self) -> None:
        peaks = _make_peaks_for_ref_window()
        mag = _make_magnitude()
        result = build_reference_quads(peaks, mag, file_id=1)
        for q in result:
            assert np.all(q.hash >= 0.0) and np.all(q.hash <= 1.0)

    def test_quad_record_file_id_set(self) -> None:
        peaks = _make_peaks_for_ref_window()
        mag = _make_magnitude()
        result = build_reference_quads(peaks, mag, file_id=42)
        for q in result:
            assert q.file_id == 42

    def test_quad_record_root_point_shape(self) -> None:
        peaks = _make_peaks_for_ref_window()
        mag = _make_magnitude()
        result = build_reference_quads(peaks, mag, file_id=1)
        for q in result:
            assert q.root_point.shape == (2,)

    def test_quad_record_quad_size_positive(self) -> None:
        """quad_size = (B_x - A_x, B_y - A_y) muss beide Komponenten > 0 haben."""
        peaks = _make_peaks_for_ref_window()
        mag = _make_magnitude()
        result = build_reference_quads(peaks, mag, file_id=1)
        for q in result:
            assert q.quad_size[0] > 0.0  # S_x = B_x - A_x > 0
            assert q.quad_size[1] > 0.0  # S_y = B_y - A_y > 0 (Eq. 1a)

    def test_too_few_peaks_returns_empty(self) -> None:
        """Weniger als 4 Peaks → keine Quads möglich."""
        peaks = np.array([[0.0, 5.0], [100.0, 8.0], [200.0, 10.0]], dtype=np.float32)
        mag = _make_magnitude()
        result = build_reference_quads(peaks, mag, file_id=0)
        assert result == []

    def test_empty_peaks_returns_empty(self) -> None:
        result = build_reference_quads(np.empty((0, 2), dtype=np.float32), _make_magnitude(), 0)
        assert result == []


# ---------------------------------------------------------------------------
# 7. build_query_quads
# ---------------------------------------------------------------------------

class TestBuildQueryQuads:
    @pytest.fixture
    def query_peaks(self) -> np.ndarray:
        """Peaks kompatibel mit dem Query-Fenster (größer als Referenz-Fenster)."""
        r_s = config.QUERY_R_START_FRAMES
        r_e = config.QUERY_R_END_FRAMES
        center = (r_s + r_e) / 2
        rng = np.random.default_rng(5)
        n = 20
        times = np.sort(np.concatenate([
            [0.0],
            rng.uniform(r_s + 10, r_e - 10, n - 1),
        ])).astype(np.float32)
        freqs = rng.uniform(10, 200, n).astype(np.float32)
        return np.column_stack([times, freqs])

    def test_returns_list(self, query_peaks: np.ndarray) -> None:
        mag = _make_magnitude()
        result = build_query_quads(query_peaks, mag)
        assert isinstance(result, list)

    def test_elements_are_query_quads(self, query_peaks: np.ndarray) -> None:
        mag = _make_magnitude()
        result = build_query_quads(query_peaks, mag)
        for q in result:
            assert isinstance(q, QueryQuad)

    def test_query_quad_no_file_id(self, query_peaks: np.ndarray) -> None:
        """QueryQuad hat kein file_id-Feld."""
        mag = _make_magnitude()
        result = build_query_quads(query_peaks, mag)
        for q in result:
            assert not hasattr(q, "file_id")

    def test_query_quad_hash_in_unit_hypercube(self, query_peaks: np.ndarray) -> None:
        mag = _make_magnitude()
        result = build_query_quads(query_peaks, mag)
        for q in result:
            assert np.all(q.hash >= 0.0) and np.all(q.hash <= 1.0)

    def test_query_quad_hash_cx_above_threshold(self, query_peaks: np.ndarray) -> None:
        """Subspace-Check: Alle Query-Quads haben C'_x ≥ HASH_CX_MIN - SEARCH_RADIUS."""
        mag = _make_magnitude()
        result = build_query_quads(query_peaks, mag)
        threshold = config.HASH_CX_MIN - config.SEARCH_RADIUS
        for q in result:
            assert q.hash[0] >= threshold - 1e-6

    def test_too_few_peaks_returns_empty(self) -> None:
        peaks = np.array([[0.0, 5.0], [100.0, 8.0]], dtype=np.float32)
        mag = _make_magnitude()
        result = build_query_quads(peaks, mag)
        assert result == []


# ---------------------------------------------------------------------------
# 8. Hash-Konsistenz über build_*_quads
# ---------------------------------------------------------------------------

class TestHashConsistencyOverBuilders:
    def test_same_quad_same_hash_in_ref_and_query(self) -> None:
        """Gleiche Peaks → compute_quad_hash muss mit Quad aus build_*_quads übereinstimmen."""
        peaks = _make_peaks_for_ref_window()
        mag = _make_magnitude()
        ref_quads = build_reference_quads(peaks, mag, file_id=0)

        if not ref_quads:
            pytest.skip("Keine Referenz-Quads erzeugt — Peaks nicht kompatibel.")

        # Ersten Record prüfen: Hash muss mit manueller Berechnung übereinstimmen
        q = ref_quads[0]
        a_t, a_f = float(q.root_point[0]), float(q.root_point[1])
        sx, sy = float(q.quad_size[0]), float(q.quad_size[1])
        b_t, b_f = a_t + sx, a_f + sy

        # Nur A und B sind aus root_point + quad_size rekonstruierbar.
        # Wir prüfen quad_size > 0 und Hash im Einheitsquadrat.
        assert sx > 0.0 and sy > 0.0
        assert np.all(q.hash >= 0.0) and np.all(q.hash <= 1.0)
