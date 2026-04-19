"""
Tests für fingerprint.py.

Prüft Hash-Korrektheit, Fan-Out-Limitierung, 32-Bit-Größe und
Frequenz-Quantisierung anhand synthetischer Peaks.
"""

import numpy as np
import pytest

from shazam_fingerprint import config
from shazam_fingerprint.fingerprint import (
    generate_fingerprints,
    _compute_hash,
    _quantize_frequency,
    _find_targets_in_zone,
    _MAX_FREQ_VAL,
    _MAX_DELTA_T_VAL,
)


# ======================================================================
# Hilfsfunktionen
# ======================================================================

def _make_peaks(*args: tuple[int, int]) -> list[tuple[int, int]]:
    """Erzeugt eine nach time_frame sortierte Peak-Liste."""
    return sorted(args, key=lambda p: p[1])


# ======================================================================
# Rückgabetyp und Grundstruktur
# ======================================================================

class TestReturnType:
    def test_returns_list_of_tuples(self) -> None:
        """generate_fingerprints gibt eine Liste von (int, int)-Tupeln zurück."""
        peaks = _make_peaks((100, 0), (200, 5), (300, 10))
        fps = generate_fingerprints(peaks)
        assert isinstance(fps, list)
        for fp in fps:
            assert isinstance(fp, tuple)
            assert len(fp) == 2
            assert isinstance(fp[0], int)
            assert isinstance(fp[1], int)

    def test_less_than_two_peaks_returns_empty(self) -> None:
        """Weniger als 2 Peaks liefern eine leere Liste."""
        assert generate_fingerprints([]) == []
        assert generate_fingerprints([(100, 0)]) == []

    def test_no_targets_in_zone_returns_empty(self) -> None:
        """Peaks die nicht in der Target Zone liegen erzeugen keine Fingerprints."""
        # Zwei Peaks mit einem Zeitabstand größer als TARGET_ZONE_T_MAX
        t_far = config.TARGET_ZONE_T_MAX + 100
        peaks = _make_peaks((100, 0), (200, t_far))
        fps = generate_fingerprints(peaks)
        assert fps == []


# ======================================================================
# Bekannte Hashes manuell nachrechnen
# ======================================================================

class TestKnownHashValues:
    def test_single_pair_hash(self) -> None:
        """Hash eines bekannten Anchor-Target-Paares stimmt mit manuellem Ergebnis überein.

        Für Anchor (f_a, t_a) und Target (f_t, t_t) muss gelten:
            hash = (quantize(f_a) << 22) | (quantize(f_t) << 12) | delta_t
        """
        f_a, t_a = 100, 0
        f_t, t_t = 200, config.TARGET_ZONE_T_MIN
        delta_t = t_t - t_a

        expected_fa_q = f_a * _MAX_FREQ_VAL // config.MAX_FREQ_BIN
        expected_ft_q = f_t * _MAX_FREQ_VAL // config.MAX_FREQ_BIN
        expected_hash = (
            (expected_fa_q << (config.FREQ_BITS + config.DELTA_T_BITS))
            | (expected_ft_q << config.DELTA_T_BITS)
            | delta_t
        )

        peaks = _make_peaks((f_a, t_a), (f_t, t_t))
        fps = generate_fingerprints(peaks)

        assert len(fps) >= 1
        hash_val, anchor_time = fps[0]
        assert hash_val == expected_hash
        assert anchor_time == t_a

    def test_anchor_time_not_in_hash(self) -> None:
        """Der Anchor-Zeitoffset ist Metadatum, NICHT Teil des Hashes.

        Wang: "Each hash is also associated with the time offset from the
        beginning of the respective file to its anchor point, though the
        absolute time is not a part of the hash itself."

        Zwei identische Anchor-Target-Paare mit verschiedenen Anker-Zeitpunkten
        müssen denselben Hash produzieren.
        """
        f_a, f_t = 100, 200
        delta_t = config.TARGET_ZONE_T_MIN

        # Erster Anchor bei t=0, zweiter bei t=100
        peaks1 = _make_peaks((f_a, 0), (f_t, delta_t))
        peaks2 = _make_peaks((f_a, 100), (f_t, 100 + delta_t))

        fps1 = generate_fingerprints(peaks1)
        fps2 = generate_fingerprints(peaks2)

        hash1 = fps1[0][0]
        hash2 = fps2[0][0]
        anchor1 = fps1[0][1]
        anchor2 = fps2[0][1]

        assert hash1 == hash2, "Gleiche Paar-Geometrie muss gleichen Hash liefern"
        assert anchor1 != anchor2, "Anchor-Zeitoffsets müssen verschieden sein"

    def test_different_freq_gives_different_hash(self) -> None:
        """Verschiedene Frequenzen erzeugen verschiedene Hashes.

        Beide Targets liegen innerhalb der Frequenzgrenzen der Target Zone
        (f_a + TARGET_ZONE_F_MIN … f_a + TARGET_ZONE_F_MAX).
        """
        t_a, t_t = 0, config.TARGET_ZONE_T_MIN
        f_a = 500
        # Beide Targets innerhalb der Frequenz-Zone
        f_t1 = f_a + 10
        f_t2 = f_a + 20

        peaks_a = _make_peaks((f_a, t_a), (f_t1, t_t))
        peaks_b = _make_peaks((f_a, t_a), (f_t2, t_t))

        fps_a = generate_fingerprints(peaks_a)
        fps_b = generate_fingerprints(peaks_b)

        assert fps_a[0][0] != fps_b[0][0]

    def test_different_delta_t_gives_different_hash(self) -> None:
        """Verschiedene delta_t-Werte erzeugen verschiedene Hashes."""
        peaks_a = _make_peaks((100, 0), (200, config.TARGET_ZONE_T_MIN))
        peaks_b = _make_peaks((100, 0), (200, config.TARGET_ZONE_T_MIN + 1))

        fps_a = generate_fingerprints(peaks_a)
        fps_b = generate_fingerprints(peaks_b)

        assert fps_a[0][0] != fps_b[0][0]


# ======================================================================
# Fan-Out-Limit
# ======================================================================

class TestFanOut:
    def test_fan_out_not_exceeded_per_anchor(self) -> None:
        """Pro Anchor werden maximal FAN_OUT Hashes erzeugt.

        Wang Section 2.2: "fan-out of size F=10" begrenzt die Anzahl der
        Target-Peaks pro Anchor.
        """
        # Erzeuge einen Anchor und viele Targets in der Zone
        t_a = 0
        n_targets = config.FAN_OUT * 3
        f_a = 500
        # Targets in kleinen Zeitschritten, alle in der Zone
        peaks = [(f_a, t_a)]
        for i in range(n_targets):
            t_t = t_a + config.TARGET_ZONE_T_MIN + i
            if t_t > config.TARGET_ZONE_T_MAX:
                break
            peaks.append((f_a + 10, t_t))

        peaks = sorted(peaks, key=lambda p: p[1])
        fps = generate_fingerprints(peaks)

        # Alle Fingerprints mit anchor_time == t_a zählen
        anchor_fps = [(h, t) for h, t in fps if t == t_a]
        assert len(anchor_fps) <= config.FAN_OUT

    def test_fan_out_chooses_nearest_targets(self) -> None:
        """Fan-Out wählt die zeitlich nächsten Targets in der Zone aus.

        Targets werden nach aufsteigendem delta_t sortiert, bevor FAN_OUT
        angewendet wird.
        """
        t_a = 0
        f_a = 500
        n_extra = 5  # Mehr als FAN_OUT Targets

        peaks = [(f_a, t_a)]
        times = list(range(
            t_a + config.TARGET_ZONE_T_MIN,
            t_a + config.TARGET_ZONE_T_MIN + config.FAN_OUT + n_extra,
        ))
        for t in times:
            if t <= t_a + config.TARGET_ZONE_T_MAX:
                peaks.append((f_a + 50, t))

        peaks = sorted(peaks, key=lambda p: p[1])
        fps = generate_fingerprints(peaks)

        anchor_fps = [(h, t) for h, t in fps if t == t_a]
        # Aus den Hashes die delta_t-Werte extrahieren
        dt_mask = (1 << config.DELTA_T_BITS) - 1
        delta_ts = sorted(h & dt_mask for h, _ in anchor_fps)

        expected_dt_min = config.TARGET_ZONE_T_MIN
        expected_dt_max = config.TARGET_ZONE_T_MIN + config.FAN_OUT - 1
        assert delta_ts[0] == expected_dt_min, "Erstes Target muss das zeitlich nächste sein"
        assert delta_ts[-1] <= expected_dt_max, "Letztes Target darf nicht weiter sein"


# ======================================================================
# 32-Bit-Größe aller Hashes
# ======================================================================

class TestHashSize:
    def test_all_hashes_fit_in_32_bit(self) -> None:
        """Alle erzeugten Hashes passen in einen 32-bit unsigned integer.

        Wang: "each hash can be packed into a 32-bit unsigned integer."
        """
        # Verschiedene extreme Frequenz- und Zeitkombinationen
        max_bin = config.MAX_FREQ_BIN
        rng = np.random.default_rng(config.RANDOM_SEED)
        peaks = [(int(rng.integers(0, max_bin + 1)), i * config.TARGET_ZONE_T_MIN)
                 for i in range(50)]
        peaks = sorted(peaks, key=lambda p: p[1])
        fps = generate_fingerprints(peaks)

        assert len(fps) > 0
        for hash_val, _ in fps:
            assert 0 <= hash_val <= 0xFFFFFFFF, (
                f"Hash {hash_val:#010x} passt nicht in 32 Bit"
            )

    def test_extreme_freq_bins_fit_in_32_bit(self) -> None:
        """Auch bei maximalen Frequenz-Bins passen Hashes in 32 Bit."""
        max_bin = config.MAX_FREQ_BIN
        t_min = config.TARGET_ZONE_T_MIN

        peaks = [(max_bin, 0), (max_bin, t_min)]
        fps = generate_fingerprints(peaks)
        for hash_val, _ in fps:
            assert 0 <= hash_val <= 0xFFFFFFFF

    def test_max_delta_t_clamped(self) -> None:
        """delta_t-Werte größer als DELTA_T_BITS werden geclamped (nicht überlaufen).

        _compute_hash klemmt delta_t auf _MAX_DELTA_T_VAL, sodass kein Overflow
        in den Hash-Bits entsteht.
        """
        # delta_t knapp über dem Maximum der DELTA_T_BITS
        f_a, f_t = 100, 200
        delta_t_overflow = _MAX_DELTA_T_VAL + 100
        hash_val = _compute_hash(f_a, f_t, delta_t_overflow)
        assert 0 <= hash_val <= 0xFFFFFFFF

        # Geclamped delta_t muss mit dem Maximum übereinstimmen
        hash_max = _compute_hash(f_a, f_t, _MAX_DELTA_T_VAL)
        assert hash_val == hash_max


# ======================================================================
# Frequenz-Quantisierung
# ======================================================================

class TestFrequencyQuantization:
    def test_zero_freq_quantizes_to_zero(self) -> None:
        """Frequenz-Bin 0 wird zu 0 quantisiert."""
        assert _quantize_frequency(0) == 0

    def test_max_freq_quantizes_to_max(self) -> None:
        """MAX_FREQ_BIN wird zum Maximalwert (2^FREQ_BITS - 1) quantisiert."""
        assert _quantize_frequency(config.MAX_FREQ_BIN) == _MAX_FREQ_VAL

    def test_quantization_in_range(self) -> None:
        """Alle gültigen Frequenz-Bins bleiben im Bereich [0, 2^FREQ_BITS - 1]."""
        for f in range(0, config.MAX_FREQ_BIN + 1, config.MAX_FREQ_BIN // 50):
            q = _quantize_frequency(f)
            assert 0 <= q <= _MAX_FREQ_VAL, (
                f"_quantize_frequency({f}) = {q} liegt außerhalb [0, {_MAX_FREQ_VAL}]"
            )

    def test_quantization_monotone(self) -> None:
        """Die Quantisierung ist monoton steigend."""
        prev = _quantize_frequency(0)
        for f in range(1, config.MAX_FREQ_BIN + 1, config.MAX_FREQ_BIN // 100):
            cur = _quantize_frequency(f)
            assert cur >= prev, f"Quantisierung nicht monoton: f={f}, cur={cur}, prev={prev}"
            prev = cur

    def test_overflow_clamp(self) -> None:
        """Frequenz-Bins über MAX_FREQ_BIN werden auf _MAX_FREQ_VAL geklemmt."""
        assert _quantize_frequency(config.MAX_FREQ_BIN + 1000) == _MAX_FREQ_VAL


# ======================================================================
# Hash-Dekodierung: Bit-Layout überprüfen
# ======================================================================

class TestHashLayout:
    def test_hash_bit_layout(self) -> None:
        """Das Hash-Bit-Layout (f_a | f_t | delta_t) ist korrekt codiert.

        Wang Section 2.2: Hash = f_anchor[10 bit] | f_target[10 bit] | delta_t[12 bit]
        Durch Bit-Shift kann jede Komponente wieder extrahiert werden.
        """
        f_a_q = 512   # Beliebiger quantisierter Frequenzwert
        f_t_q = 256
        delta_t = 100

        hash_val = _compute_hash(
            f_a_q * config.MAX_FREQ_BIN // _MAX_FREQ_VAL,
            f_t_q * config.MAX_FREQ_BIN // _MAX_FREQ_VAL,
            delta_t,
        )

        extracted_dt = hash_val & ((1 << config.DELTA_T_BITS) - 1)
        extracted_ft = (hash_val >> config.DELTA_T_BITS) & ((1 << config.FREQ_BITS) - 1)
        extracted_fa = (hash_val >> (config.DELTA_T_BITS + config.FREQ_BITS)) & (
            (1 << config.FREQ_BITS) - 1
        )

        assert extracted_dt == delta_t
        # Quantisierte Werte können durch Integer-Division leicht abweichen
        assert abs(extracted_fa - f_a_q) <= 1
        assert abs(extracted_ft - f_t_q) <= 1

    def test_total_hash_bits_32(self) -> None:
        """FREQ_BITS + FREQ_BITS + DELTA_T_BITS ergibt genau 32 Bit."""
        total = config.FREQ_BITS + config.FREQ_BITS + config.DELTA_T_BITS
        assert total == 32, f"Hash-Layout ist {total} Bit statt 32 Bit"


# ======================================================================
# Target Zone
# ======================================================================

class TestTargetZone:
    def test_target_zone_excludes_too_early(self) -> None:
        """Peaks mit delta_t < TARGET_ZONE_T_MIN werden nicht gepaart."""
        t_a = 10
        # Target zu nah am Anchor (delta_t = 0, also gleiche Zeit)
        peaks = [(100, t_a), (200, t_a)]  # delta_t = 0 < T_MIN
        fps = generate_fingerprints(peaks)
        # Kein Hash darf mit delta_t < T_MIN erzeugt werden
        dt_mask = (1 << config.DELTA_T_BITS) - 1
        for h, _ in fps:
            assert (h & dt_mask) >= config.TARGET_ZONE_T_MIN

    def test_target_zone_excludes_too_late(self) -> None:
        """Peaks mit delta_t > TARGET_ZONE_T_MAX werden nicht gepaart."""
        t_a = 0
        t_too_late = t_a + config.TARGET_ZONE_T_MAX + 1
        peaks = _make_peaks((100, t_a), (200, t_too_late))
        fps = generate_fingerprints(peaks)
        assert fps == []

    def test_target_zone_boundary_t_min_included(self) -> None:
        """delta_t == TARGET_ZONE_T_MIN liegt noch in der Zone (Grenzwert)."""
        t_a = 0
        t_t = t_a + config.TARGET_ZONE_T_MIN
        peaks = _make_peaks((100, t_a), (200, t_t))
        fps = generate_fingerprints(peaks)
        assert len(fps) >= 1

    def test_target_zone_boundary_t_max_included(self) -> None:
        """delta_t == TARGET_ZONE_T_MAX liegt noch in der Zone (Grenzwert)."""
        t_a = 0
        t_t = t_a + config.TARGET_ZONE_T_MAX
        peaks = _make_peaks((100, t_a), (200, t_t))
        fps = generate_fingerprints(peaks)
        assert len(fps) >= 1

    def test_target_zone_freq_bounds(self) -> None:
        """Frequenzen außerhalb [f_a + F_MIN, f_a + F_MAX] werden ausgeschlossen."""
        t_a, t_t = 0, config.TARGET_ZONE_T_MIN
        f_a = 500

        # Target zu weit in Frequenz (außerhalb F_MAX)
        f_out = f_a + config.TARGET_ZONE_F_MAX + 50
        if f_out <= config.MAX_FREQ_BIN:
            peaks = _make_peaks((f_a, t_a), (f_out, t_t))
            fps = generate_fingerprints(peaks)
            assert fps == []

        # Target knapp innerhalb F_MAX
        f_in = f_a + config.TARGET_ZONE_F_MAX
        if f_in <= config.MAX_FREQ_BIN:
            peaks = _make_peaks((f_a, t_a), (f_in, t_t))
            fps = generate_fingerprints(peaks)
            assert len(fps) >= 1


# ======================================================================
# Anchor-Time als Metadatum
# ======================================================================

class TestAnchorTime:
    def test_anchor_time_equals_t_a(self) -> None:
        """Der zweite Wert jedes Fingerprints entspricht dem Anchor-Zeitframe."""
        peaks = [
            (100, 0),
            (200, config.TARGET_ZONE_T_MIN),
            (300, config.TARGET_ZONE_T_MIN * 2),
        ]
        peaks = sorted(peaks, key=lambda p: p[1])
        fps = generate_fingerprints(peaks)

        anchor_times = {t for _, t in fps}
        peak_times = {t for _, t in peaks}

        # Alle Anchor-Times müssen aus den Peak-Zeitframes stammen
        assert anchor_times.issubset(peak_times)

    def test_fingerprints_preserve_multiple_anchor_times(self) -> None:
        """Bei mehreren Anchors werden alle Anchor-Zeitoffsets korrekt gespeichert."""
        # Drei Anchors mit je einem Target in der Zone
        peaks = []
        for i in range(3):
            t_a = i * (config.TARGET_ZONE_T_MAX + 1)
            t_t = t_a + config.TARGET_ZONE_T_MIN
            peaks.append((100, t_a))
            peaks.append((200, t_t))

        peaks = sorted(peaks, key=lambda p: p[1])
        fps = generate_fingerprints(peaks)

        anchor_times = sorted({t for _, t in fps})
        # Mindestens die drei Anchor-Zeitpunkte müssen vorkommen
        expected_anchors = [i * (config.TARGET_ZONE_T_MAX + 1) for i in range(3)]
        for expected_t in expected_anchors:
            assert expected_t in anchor_times, (
                f"Anchor-Zeit {expected_t} fehlt in {anchor_times}"
            )
