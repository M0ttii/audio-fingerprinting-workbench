"""
Shazam pipeline helpers for the Audio Fingerprinting Benchmark (NB 02).

Wraps the low-level shazam_fingerprint package API so that NB 02 can build
the reference index from a list of track IDs (rather than a flat directory)
and produce result rows in the verbindlichen Raw-Result-Format.

Prerequisite: src/ must be on sys.path so that 'shazam_fingerprint' is
importable (the notebook sets this up via sys.path.insert).

Reference: Testkonzept Phase 2 — Shazam Pipeline.
"""

import logging
import time
from typing import List, Optional, Tuple
from tqdm import tqdm

import pandas as pd

from shazam_fingerprint.audio_loader import load_audio
from shazam_fingerprint.database import FingerprintDatabase
from shazam_fingerprint.fingerprint import generate_fingerprints
from shazam_fingerprint.peak_finder import find_peaks
from shazam_fingerprint.pipeline import query as _shazam_query
from shazam_fingerprint.spectrogram import compute_spectrogram

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_shazam_index(
    track_ids: List[int],
    metadata_df: pd.DataFrame,
) -> Tuple[FingerprintDatabase, dict]:
    """Build a Shazam FingerprintDatabase from a list of FMA track IDs.

    Processes each track through the full shazam_fingerprint pipeline:
        load_audio → compute_spectrogram → find_peaks →
        generate_fingerprints → database.insert(song_id, fps)

    The song_id stored in the database is the zero-padded 6-digit string
    of track_id (e.g. track_id=52 → song_id="000052"). This convention
    must match what run_shazam_query() uses to convert best_match back to int.

    Tracks that fail to load or yield zero fingerprints are skipped with a
    warning — never crashes on bad files.

    Args:
        track_ids:   List of integer FMA track IDs to index (dry_ref).
        metadata_df: DataFrame from load_fma_metadata() with a 'filepath'
                     column indexed by track_id.

    Returns:
        Tuple of:
            database (FingerprintDatabase): Populated index ready for queries.
            stats (dict): processed, skipped, failed, total_hashes, build_time_s.
    """
    db = FingerprintDatabase()
    processed = 0
    skipped = 0
    failed = 0
    total_hashes = 0

    t0 = time.perf_counter()

    for track_id in tqdm(track_ids, desc="Shazam-Index"):
        if track_id not in metadata_df.index:
            logger.warning("build_shazam_index: track_id=%d not in metadata.", track_id)
            skipped += 1
            continue

        filepath = metadata_df.at[track_id, "filepath"]
        song_id = f"{track_id:06d}"

        try:
            signal, sr, _ = load_audio(filepath)
            spec = compute_spectrogram(signal, sr)
            peaks = find_peaks(spec)
            fps = generate_fingerprints(peaks)

            if not fps:
                logger.warning("build_shazam_index: no fingerprints for track %d.", track_id)
                failed += 1
                continue

            db.insert(song_id, fps)
            total_hashes += len(fps)
            processed += 1

        except Exception as exc:
            logger.warning("build_shazam_index: error on track %d — %s", track_id, exc)
            failed += 1

    build_time_s = time.perf_counter() - t0

    stats = {
        "processed":    processed,
        "skipped":      skipped,
        "failed":       failed,
        "total_hashes": total_hashes,
        "build_time_s": round(build_time_s, 3),
    }
    print(
        f"Shazam index built: {processed} tracks, {total_hashes:,} hashes, "
        f"{failed} failed, {build_time_s:.1f}s"
    )
    return db, stats


# ---------------------------------------------------------------------------
# Single query
# ---------------------------------------------------------------------------

def run_shazam_query(
    audio_path: str,
    database: FingerprintDatabase,
) -> Tuple[Optional[int], int, float]:
    """Run one Shazam query and return (pred_id, score, query_time_ms).

    Wraps shazam_fingerprint.pipeline.query() and converts the string
    song_id back to an integer track_id.  Returns None as pred_id when no
    match is found.

    Total elapsed time is measured externally with time.perf_counter() to
    include fingerprinting + matching (MatchResult.processing_time_ms only
    covers the matching step).

    Args:
        audio_path: Absolute path to the query WAV file.
        database:   FingerprintDatabase built by build_shazam_index().

    Returns:
        Tuple of:
            pred_id (int | None): Predicted track ID, or None if no match.
            score   (int):        Best histogram peak height.
            query_time_ms (float): Total wall-clock time in ms.
    """
    t0 = time.perf_counter()
    result = _shazam_query(audio_path, database)
    query_time_ms = (time.perf_counter() - t0) * 1000.0

    if result.best_match is not None:
        try:
            pred_id: Optional[int] = int(result.best_match)
        except (ValueError, TypeError):
            logger.warning(
                "run_shazam_query: cannot convert best_match=%r to int.", result.best_match
            )
            pred_id = None
    else:
        pred_id = None

    return pred_id, result.best_score, round(query_time_ms, 3)
