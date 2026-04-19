"""
Query generation logic for the Audio Fingerprinting Benchmark (NB 01).

Implements generate_track_queries(), which applies all distortion conditions
from Groups A–D to a single FMA track and saves the resulting WAV files.
Returns manifest rows compatible with the Raw-Result-Format defined in.

Reference: Testkonzept Phase 1 — Query-Generierung.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from distortions import (
    load_and_resample,
    extract_segment,
    apply_tempo_change,
    apply_pitch_shift,
    apply_speed_change,
    apply_noise,
    apply_room_ir,
    apply_mp3_compression,
    apply_combined,
    save_wav,
)

logger = logging.getLogger(__name__)


def generate_track_queries(
    track_id: int,
    is_ood: bool,
    filepath: str,
    out_dir: str,
    musan_eval_files: List[str],
    ir_files: List[str],
    sr: int = 8000,
) -> List[dict]:
    """Generate all query WAV files for a single track and return manifest rows.

    Applies all conditions from groups A (15), B (8), C (2), and D (8) to
    segments extracted from the track. Audio is loaded once; the 10-second
    base segment is extracted with seed=track_id for reproducibility.

    Group A — tempo / pitch / speed (15 conditions):
        A_original, A_tempo_80/90/95/105/110/120,
        A_pitch_m1/m2/p1/p2, A_speed_80/90/110/120.
        Speed conditions change output length — stored as actual duration_sec.

    Group B — noise / room IR / compression (8 conditions):
        B_snr_20/10/5/0/m5, B_room_ir, B_mp3_128, B_mp3_64.

    Group C — combined real-world scenarios (2 conditions):
        C_club  = tempo_110 + SNR_5 + room_IR
        C_radio = pitch_+1  + MP3_64 + SNR_10

    Group D — variable query lengths × 2 noise levels (8 conditions):
        D_clean_3s/5s/10s/15s, D_snr10_3s/5s/10s/15s.

    Seed policy: all random choices use seed=track_id for reproducibility.

    Args:
        track_id:         Integer FMA track ID.
        is_ood:           True if the track is NOT in the reference database.
        filepath:         Absolute path to the .mp3 audio file.
        out_dir:          Directory for output WAV files.
                          Files are written as ``<out_dir>/<tid>_<cond>.wav``.
        musan_eval_files: MUSAN eval .wav paths (noise addition + pseudo-IR).
        ir_files:         Impulse response .wav paths (use musan_eval_files
                          as fallback for dry runs without a real IR dataset).
        sr:               Sample rate in Hz. Default: 8 000.

    Returns:
        List of manifest row dicts (one per generated file) with keys:
            track_id, query_path, ref_track_id, is_ood, group,
            condition, duration_sec, nominal_dur_s.
        Returns an empty list if the audio file cannot be loaded
        (warning is logged; the caller should count skipped tracks).
    """
    out_dir = Path(out_dir)
    ref_track_id = None if is_ood else track_id

    try:
        audio = load_and_resample(str(filepath), sr=sr)
    except Exception as exc:
        logger.warning(
            "generate_track_queries: cannot load %s — %s. Skipping track %d.",
            filepath, exc, track_id,
        )
        return []

    # 10-second base segment, seed=track_id → deterministic
    seg10 = extract_segment(audio, sr, duration_sec=10.0, seed=track_id)

    rows: List[dict] = []

    def _row(cond: str, out_audio: np.ndarray, nominal: int) -> None:
        """Save WAV and append a manifest row."""
        path = out_dir / f"{track_id}_{cond}.wav"
        save_wav(out_audio, sr, str(path))
        rows.append({
            "track_id":      track_id,
            "query_path":    str(path),
            "ref_track_id":  ref_track_id,
            "is_ood":        is_ood,
            "group":         cond.split("_")[0],
            "condition":     cond,
            "duration_sec":  round(len(out_audio) / sr, 6),
            "nominal_dur_s": nominal,
        })

    # ------------------------------------------------------------------
    # Group A — 15 conditions (tempo / pitch / speed)
    # ------------------------------------------------------------------
    _row("A_original",  seg10.copy(), 10)
    for rate_str, rate in [("80", 0.80), ("90", 0.90), ("95", 0.95),
                            ("105", 1.05), ("110", 1.10), ("120", 1.20)]:
        _row(f"A_tempo_{rate_str}", apply_tempo_change(seg10, sr, rate), 10)
    for step_str, step in [("m1", -1), ("m2", -2), ("p1", +1), ("p2", +2)]:
        _row(f"A_pitch_{step_str}", apply_pitch_shift(seg10, sr, step), 10)
    for rate_str, rate in [("80", 0.80), ("90", 0.90), ("110", 1.10), ("120", 1.20)]:
        _row(f"A_speed_{rate_str}", apply_speed_change(seg10, sr, rate), 10)

    # ------------------------------------------------------------------
    # Group B — 8 conditions (noise / IR / compression)
    # ------------------------------------------------------------------
    for lbl, snr in [("20", 20), ("10", 10), ("5", 5), ("0", 0), ("m5", -5)]:
        noisy, _ = apply_noise(seg10, sr, musan_eval_files, snr_db=snr,
                               seed=track_id)
        _row(f"B_snr_{lbl}", noisy, 10)
    _row("B_room_ir", apply_room_ir(seg10, sr, ir_files, seed=track_id), 10)
    _row("B_mp3_128", apply_mp3_compression(seg10, sr, 128), 10)
    _row("B_mp3_64",  apply_mp3_compression(seg10, sr, 64),  10)

    # ------------------------------------------------------------------
    # Group C — 2 combined conditions
    # ------------------------------------------------------------------
    _row("C_club", apply_combined(seg10, sr, [
        (apply_tempo_change,    {"rate": 1.10}),
        (apply_noise,           {"snr_db": 5,  "noise_files": musan_eval_files,
                                 "seed": track_id}),
        (apply_room_ir,         {"ir_files": ir_files, "seed": track_id}),
    ]), 10)
    _row("C_radio", apply_combined(seg10, sr, [
        (apply_pitch_shift,     {"semitones": +1}),
        (apply_mp3_compression, {"bitrate_kbps": 64}),
        (apply_noise,           {"snr_db": 10, "noise_files": musan_eval_files,
                                 "seed": track_id}),
    ]), 10)

    # ------------------------------------------------------------------
    # Group D — 8 conditions (4 lengths × clean + SNR10)
    # ------------------------------------------------------------------
    for dur in [3, 5, 10, 15]:
        seg_d = extract_segment(audio, sr, duration_sec=float(dur), seed=track_id)
        _row(f"D_clean_{dur}s", seg_d, dur)
        noisy_d, _ = apply_noise(seg_d, sr, musan_eval_files, snr_db=10,
                                 seed=track_id)
        _row(f"D_snr10_{dur}s", noisy_d, dur)

    return rows


def build_manifest(rows: List[dict]) -> pd.DataFrame:
    """Convert a flat list of manifest row dicts to a validated DataFrame.
    ref_track_id is stored as a nullable integer (pd.Int64Dtype) so that
    None/NaN values are preserved correctly when the CSV is written and
    re-read.

    Args:
        rows: List of dicts as returned by generate_track_queries().

    Returns:
        DataFrame with columns:
            track_id (int), query_path (str), ref_track_id (Int64 nullable),
            is_ood (bool), group (str), condition (str),
            duration_sec (float), nominal_dur_s (int).
    """
    df = pd.DataFrame(rows, columns=[
        "track_id", "query_path", "ref_track_id", "is_ood",
        "group", "condition", "duration_sec", "nominal_dur_s",
    ])
    df["track_id"]      = df["track_id"].astype(int)
    df["ref_track_id"]  = df["ref_track_id"].astype(pd.Int64Dtype())
    df["is_ood"]        = df["is_ood"].astype(bool)
    df["duration_sec"]  = df["duration_sec"].astype(float)
    df["nominal_dur_s"] = df["nominal_dur_s"].astype(int)
    return df
