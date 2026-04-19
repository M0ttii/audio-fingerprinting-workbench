"""
Quad pipeline helpers for the Audio Fingerprinting Benchmark (NB 03).

Wraps the low-level quad_fingerprint package API so that NB 03 can build
the reference index from a list of track IDs and produce result rows in the
verbindlichen Raw-Result-Format, including the Quad-specific
scale factor columns.

Prerequisite: src/ must be on sys.path so that 'quad_fingerprint' is
importable (the notebook sets this up via sys.path.insert).

Reference: Testkonzept Phase 2 — Quad Pipeline.
Reference: Sonnleitner & Widmer (2016), Sections IV–VI.
"""

import logging
import re
import time
from typing import List, Optional, Tuple
from tqdm import tqdm

import pandas as pd

from quad_fingerprint.audio_loader import load_audio
from quad_fingerprint.database import ReferenceDatabase
from quad_fingerprint.peak_finder import extract_reference_peaks
from quad_fingerprint.quad_builder import build_reference_quads
from quad_fingerprint.spectrogram import compute_spectrogram
from quad_fingerprint.pipeline import query as _quad_query

logger = logging.getLogger(__name__)

# Regex to parse condition names like "A_tempo_95", "A_speed_120", "A_pitch_m2"
_COND_TEMPO_RE = re.compile(r"^A_tempo_(\d+)$")
_COND_SPEED_RE = re.compile(r"^A_speed_(\d+)$")
_COND_PITCH_RE = re.compile(r"^A_pitch_")


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_quad_index(
    track_ids: List[int],
    metadata_df: pd.DataFrame,
) -> Tuple[ReferenceDatabase, dict]:
    """Build a Quad ReferenceDatabase from a list of FMA track IDs.

    Processes each track through the full quad_fingerprint pipeline:
        load_audio → compute_spectrogram → extract_reference_peaks →
        build_reference_quads → database.add_file(song_id, peaks, quads)
    Then calls database.finalize() once at the end (efficient: builds
    the cKDTree over all records at once — Sonnleitner & Widmer Section V).

    The file_name stored in the database is the zero-padded 6-digit string
    of track_id plus ".mp3" extension (e.g. track_id=52 → "000052.mp3").
    This convention must match what run_quad_query() uses to parse best_match
    back to int.

    Tracks that fail to load, yield too few peaks (<4), or yield zero quads
    are skipped with a warning — never crashes on bad files.

    Args:
        track_ids:   List of integer FMA track IDs to index (dry_ref).
        metadata_df: DataFrame from load_fma_metadata() with a 'filepath'
                     column indexed by track_id.

    Returns:
        Tuple of:
            database (ReferenceDatabase): Finalized index ready for queries.
            stats (dict): processed, skipped, failed, total_quads,
                          total_peaks, build_time_s, db_memory_mb.
    """
    db = ReferenceDatabase()
    processed = 0
    skipped = 0
    failed = 0
    total_quads = 0
    total_peaks = 0

    t0 = time.perf_counter()

    for track_id in tqdm(track_ids, desc="Quad-Index"):
        if track_id not in metadata_df.index:
            logger.warning("build_quad_index: track_id=%d not in metadata.", track_id)
            skipped += 1
            continue

        filepath = metadata_df.at[track_id, "filepath"]
        file_name = f"{track_id:06d}.mp3"

        try:
            signal, sr, meta = load_audio(filepath)
            spec = compute_spectrogram(signal, sr)
            peaks = extract_reference_peaks(spec.magnitude)

            if len(peaks) < 4:
                logger.warning(
                    "build_quad_index: too few peaks (%d) for track %d — skipping.",
                    len(peaks), track_id,
                )
                failed += 1
                continue

            # file_id is assigned as the current n_files (before add_file)
            file_id = db.n_files
            quads = build_reference_quads(peaks, spec.magnitude, file_id)

            if not quads:
                logger.warning(
                    "build_quad_index: no quads for track %d — skipping.", track_id
                )
                failed += 1
                continue

            db.add_file(
                file_name=file_name,
                peaks=peaks,
                quad_records=quads,
                duration_sec=meta.get("duration", 0.0),
            )
            total_quads += len(quads)
            total_peaks += len(peaks)
            processed += 1

        except Exception as exc:
            logger.warning("build_quad_index: error on track %d — %s", track_id, exc)
            failed += 1

    # Build the cKDTree once over all records (Section V)
    if processed > 0:
        db.finalize()

    build_time_s = time.perf_counter() - t0
    db_memory_mb = db.memory_usage_mb()["total_mb"] if db.is_finalized else 0.0

    stats = {
        "processed":    processed,
        "skipped":      skipped,
        "failed":       failed,
        "total_quads":  total_quads,
        "total_peaks":  total_peaks,
        "build_time_s": round(build_time_s, 3),
        "db_memory_mb": round(db_memory_mb, 3),
    }
    print(
        f"Quad index built: {processed} tracks, {total_quads:,} quads, "
        f"{failed} failed, {build_time_s:.1f}s, {db_memory_mb:.1f} MB"
    )
    return db, stats


# ---------------------------------------------------------------------------
# Single query
# ---------------------------------------------------------------------------

def run_quad_query(
    audio_path: str,
    database: ReferenceDatabase,
) -> Tuple[Optional[int], float, float, Optional[float], Optional[float]]:
    """Run one Quad query and return result tuple.

    Wraps quad_fingerprint.pipeline.query() and converts the string file_name
    back to an integer track_id. Returns None as pred_id when no match is
    found.

    Total elapsed time is measured externally with time.perf_counter() to
    include fingerprinting + matching.

    Args:
        audio_path: Absolute path to the query WAV file.
        database:   ReferenceDatabase built by build_quad_index().

    Returns:
        Tuple of:
            pred_id           (int | None):   Predicted track ID, or None.
            score             (float):        Verification score (0.0–1.0).
            query_time_ms     (float):        Total wall-clock time in ms.
            detected_time_scale (float|None): s_time from best_scale_factors.
            detected_freq_scale (float|None): s_freq from best_scale_factors.
    """
    t0 = time.perf_counter()
    result = _quad_query(audio_path, database)
    query_time_ms = (time.perf_counter() - t0) * 1000.0

    pred_id: Optional[int] = None
    if result.best_match is not None:
        try:
            # file_name stored as "000052.mp3" → strip extension → parse int
            stem = result.best_match.rsplit(".", 1)[0]
            pred_id = int(stem)
        except (ValueError, TypeError):
            logger.warning(
                "run_quad_query: cannot convert best_match=%r to int.",
                result.best_match,
            )
            pred_id = None

    detected_time_scale: Optional[float] = None
    detected_freq_scale: Optional[float] = None
    if result.best_scale_factors is not None:
        detected_time_scale, detected_freq_scale = result.best_scale_factors

    return pred_id, result.best_score, round(query_time_ms, 3), detected_time_scale, detected_freq_scale


# ---------------------------------------------------------------------------
# True scale factor extraction
# ---------------------------------------------------------------------------

def get_true_scales(condition: str) -> Tuple[float, float]:
    """Return (true_time_scale, true_freq_scale) for a given condition name.

    Extracts the ground-truth scale factors that a correctly functioning
    Quad matcher should detect for each distortion condition.

    Conventions (Sonnleitner & Widmer 2016, Section IV):
        A_tempo_XX  → time-stretch only: s_time = 1/(XX/100), s_freq = 1.0
                      apply_tempo_change(rate=XX/100) makes audio 1/rate longer,
                      so S_x^query/S_x^ref = 1/rate.
        A_speed_XX  → playback speed: s_time = 1/(XX/100), s_freq = XX/100
                      apply_speed_change(rate=XX/100) stretches time by 1/rate
                      and shifts frequency by rate (resampling scales pitch
                      proportionally to playback rate).
        A_pitch_XX  → semitone shift, no linear scale: (1.0, 1.0)
        All others  → no scale change: (1.0, 1.0)

    Args:
        condition: Condition string, e.g. "A_tempo_95", "A_speed_120",
                   "A_pitch_m2", "B_snr_10", "A_original", etc.

    Returns:
        Tuple (true_time_scale, true_freq_scale) as floats.
    """
    m_tempo = _COND_TEMPO_RE.match(condition)
    if m_tempo:
        rate = int(m_tempo.group(1)) / 100.0
        return (1.0 / rate, 1.0)  # s_time = 1/rate (audio is 1/rate longer)

    m_speed = _COND_SPEED_RE.match(condition)
    if m_speed:
        rate = int(m_speed.group(1)) / 100.0
        return (1.0 / rate, rate)  # time stretches by 1/rate, freq shifts by rate

    # Pitch shift: no quad-detectable linear scale (semitone shift is not
    # a uniform linear scaling of the spectrogram axes)
    if _COND_PITCH_RE.match(condition):
        return (1.0, 1.0)

    # A_original and all B/C/D conditions: no scale change
    return (1.0, 1.0)
