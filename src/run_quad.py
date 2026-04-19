#!/opt/conda/envs/audiofp_classical/bin/python3
"""
run_quad.py — Standalone Quad query runner for the live run benchmark.

Replicates the logic of NB 03 (Section 5: Run all queries) but writes
results row-by-row to CSV with f.flush() after each row, so partial
results are safe after a crash or kernel restart.

Usage:
    python /home/jupyter/run_quad.py
    python /home/jupyter/run_quad.py --test 5    # run only first N queries
    python /home/jupyter/run_quad.py --resume     # skip already-written rows

Reference: Sonnleitner & Widmer (2016), Sections IV–VI.
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jupyter/liverun")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from utils import load_fma_metadata, load_partitions
from metrics import classify_result
from quad_pipeline import build_quad_index, run_quad_query, get_true_scales

logging.basicConfig(level=logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────
FMA_DIR      = PROJECT_ROOT / "data" / "fma_medium"
PARTS_DIR    = PROJECT_ROOT / "data" / "partitions"
MANIFEST_CSV = PROJECT_ROOT / "data" / "queries" / "manifest_live.csv"
RESULTS_DIR  = PROJECT_ROOT / "results" / "live_run"
RAW_CSV      = RESULTS_DIR / "quad_raw.csv"
EFFIC_JSON   = RESULTS_DIR / "quad_efficiency.json"

COLUMNS = [
    "system",
    "track_id",
    "ref_track_id",
    "is_ood",
    "predicted_id",
    "score",
    "result_class",
    "query_time_ms",
    "group",
    "condition",
    "duration_sec",
    "detected_time_scale",
    "detected_freq_scale",
    "true_time_scale",
    "true_freq_scale",
]


def _fmt(v) -> str:
    """Format a single CSV cell value.

    None and pd.NA become empty string (same as pandas to_csv behaviour for
    Int64Dtype NA values).  Floats are rounded to 6 decimal places to avoid
    spurious precision drift.
    """
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, float):
        return repr(round(v, 6))
    return str(v)


def load_existing_rows(path: Path) -> set:
    """Return set of (track_id, condition) tuples already written to CSV."""
    seen = set()
    if not path.exists():
        return seen
    try:
        df = pd.read_csv(path, usecols=["track_id", "condition"])
        for _, r in df.iterrows():
            seen.add((int(r["track_id"]), str(r["condition"])))
    except Exception as exc:
        print(f"[WARN] Could not read existing CSV for resume: {exc}", flush=True)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Quad queries for the live run benchmark."
    )
    parser.add_argument(
        "--test", type=int, default=None, metavar="N",
        help="Run only the first N queries (smoke-test).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip rows already present in quad_raw.csv.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load partitions and manifest ──────────────────────────────────────────
    print("Loading partitions ...", flush=True)
    partitions = load_partitions(str(PARTS_DIR))
    ref_ids    = partitions["live_ref"]
    print(f"live_ref: {len(ref_ids)} tracks", flush=True)

    print("Loading FMA metadata ...", flush=True)
    metadata_df = load_fma_metadata(str(FMA_DIR))
    print(f"Metadata loaded: {len(metadata_df)} tracks", flush=True)

    print(f"Loading manifest: {MANIFEST_CSV}", flush=True)
    manifest = pd.read_csv(MANIFEST_CSV, dtype={"ref_track_id": "Int64"})
    print(
        f"Manifest: {manifest.shape[0]} queries, "
        f"{manifest['condition'].nunique()} conditions",
        flush=True,
    )

    # ── Build Quad index ───────────────────────────────────────────────────────
    print(f"\nBuilding Quad index on {len(ref_ids)} tracks (live_ref) ...", flush=True)
    quad_db, build_stats = build_quad_index(ref_ids, metadata_df)
    print(json.dumps(build_stats, indent=2), flush=True)

    # ── Determine query slice ──────────────────────────────────────────────────
    query_rows = list(manifest.iterrows())
    if args.test is not None:
        query_rows = query_rows[: args.test]
        print(f"\n[TEST MODE] Running only {args.test} queries.", flush=True)

    # ── Resume: find already-written rows ─────────────────────────────────────
    skip_set: set = set()
    append_mode = False
    if args.resume and RAW_CSV.exists():
        skip_set = load_existing_rows(RAW_CSV)
        append_mode = bool(skip_set)
        print(f"[RESUME] {len(skip_set)} rows already written — skipping.", flush=True)

    # ── Open CSV and write rows ────────────────────────────────────────────────
    file_mode  = "a" if append_mode else "w"
    write_header = not append_mode

    print(f"\nWriting results → {RAW_CSV}", flush=True)
    t_wall_start = time.perf_counter()
    n_written = 0
    n_skipped_resume = 0

    with open(RAW_CSV, file_mode, newline="") as fh:
        writer = csv.writer(fh)

        if write_header:
            writer.writerow(COLUMNS)
            fh.flush()

        for _, qrow in query_rows:
            track_id     = int(qrow["track_id"])
            query_path   = qrow["query_path"]
            ref_track_id = None if pd.isna(qrow["ref_track_id"]) else int(qrow["ref_track_id"])
            is_ood       = bool(qrow["is_ood"])
            condition    = str(qrow["condition"])

            # Resume check
            if (track_id, condition) in skip_set:
                n_skipped_resume += 1
                continue

            # Run query
            pred_id, score, q_ms, det_t, det_f = run_quad_query(query_path, quad_db)
            result_class = classify_result(pred_id, ref_track_id, is_ood)
            true_t, true_f = get_true_scales(condition)

            writer.writerow([
                "quad",
                track_id,
                "" if ref_track_id is None else ref_track_id,
                is_ood,
                "" if pred_id is None else pred_id,
                round(score, 6),
                result_class,
                round(q_ms, 3),
                qrow["group"],
                condition,
                round(float(qrow["duration_sec"]), 6),
                _fmt(det_t),
                _fmt(det_f),
                round(true_t, 6),
                round(true_f, 6),
            ])
            fh.flush()
            n_written += 1

            # Progress report every 500 queries
            done = n_written + n_skipped_resume
            if done % 500 == 0:
                elapsed = time.perf_counter() - t_wall_start
                rate = done / elapsed if elapsed > 0 else 0.0
                total = len(query_rows)
                eta_s = (total - done) / rate if rate > 0 else float("inf")
                print(
                    f"[{done:>6}/{total}]  elapsed={elapsed:.0f}s  "
                    f"rate={rate:.1f} q/s  ETA={eta_s:.0f}s",
                    flush=True,
                )

    total_wall_ms = (time.perf_counter() - t_wall_start) * 1000.0
    print(
        f"\nDone. {n_written} rows written, {n_skipped_resume} skipped (resume). "
        f"Wall time: {total_wall_ms / 1000:.1f}s",
        flush=True,
    )

    # ── Write efficiency JSON ──────────────────────────────────────────────────
    total_rows = n_written + n_skipped_resume
    efficiency = {
        "system":        "quad",
        "run_mode":      "live",
        "build_stats":   build_stats,
        "total_queries": total_rows,
        "total_wall_ms": round(total_wall_ms, 3),
        "avg_query_ms":  round(total_wall_ms / total_rows, 3) if total_rows else 0.0,
    }
    with open(EFFIC_JSON, "w") as f:
        json.dump(efficiency, f, indent=2)
    print(f"Efficiency JSON → {EFFIC_JSON}", flush=True)
    print(json.dumps(efficiency, indent=2), flush=True)


if __name__ == "__main__":
    main()
