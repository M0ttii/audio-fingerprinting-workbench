"""
Profiler: Laufzeit-Aufteilung pro Quad-Query-Stage.

Misst für 5 Queries die Zeit in:
  (A) load_audio
  (B) compute_spectrogram
  (C) extract_query_peaks
  (D) build_query_quads
  (E) database.query_radius (Batch-NN-Suche)
  (F) _select_and_filter_candidates gesamt (inkl. E)
  (G) identify() gesamt (Matching: F + Sequenz + Verif.)
  (H) Gesamtzeit (A–G)

Ausführen:
  conda run -n audiofp_classical python profile_quad.py
"""

import sys, os, time, json
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from utils import load_fma_metadata, load_partitions
from quad_pipeline import build_quad_index

from quad_fingerprint.audio_loader import load_audio
from quad_fingerprint.spectrogram import compute_spectrogram
from quad_fingerprint.peak_finder import extract_query_peaks
from quad_fingerprint.quad_builder import build_query_quads, QueryQuad
from quad_fingerprint.database import ReferenceDatabase
from quad_fingerprint.matcher import identify, _select_and_filter_candidates
from quad_fingerprint import config

# ── Setup ──────────────────────────────────────────────────────────────────
FMA_DIR   = Path("data/fma_medium")
PARTS_DIR = Path("data/partitions")
MAN_CSV   = Path("data/queries/manifest_dry.csv")

partitions  = load_partitions(str(PARTS_DIR))
dry_ref_ids = partitions["dry_ref"]
metadata_df = load_fma_metadata(str(FMA_DIR))

manifest = pd.read_csv(MAN_CSV, dtype={"ref_track_id": "Int64"})

# ── Build index ────────────────────────────────────────────────────────────
print("Building Quad index …")
quad_db, _ = build_quad_index(dry_ref_ids, metadata_df)
print(f"  n_files={quad_db.n_files}, n_records={quad_db.n_records}\n")

# ── Pick 5 queries (A_original, first 5 in-DB) ────────────────────────────
sample = manifest[manifest["condition"] == "A_original"].head(5)
results = []

for _, row in sample.iterrows():
    path = row["query_path"]
    times = {}

    # (A) load_audio
    t0 = time.perf_counter()
    signal, sr, meta = load_audio(path)
    times["A_load_audio_ms"] = (time.perf_counter() - t0) * 1e3

    # (B) compute_spectrogram
    t0 = time.perf_counter()
    spec = compute_spectrogram(signal, sr)
    times["B_spectrogram_ms"] = (time.perf_counter() - t0) * 1e3

    # (C) extract_query_peaks
    t0 = time.perf_counter()
    peaks = extract_query_peaks(spec.magnitude)
    times["C_peaks_ms"] = (time.perf_counter() - t0) * 1e3

    # (D) build_query_quads
    t0 = time.perf_counter()
    quads = build_query_quads(peaks, spec.magnitude)
    times["D_build_quads_ms"] = (time.perf_counter() - t0) * 1e3

    # (E) batch NN-search only
    query_hashes = np.array([qq.hash for qq in quads], dtype=np.float32)
    t0 = time.perf_counter()
    nn_results = quad_db.query_radius(query_hashes, radius=config.SEARCH_RADIUS)
    times["E_nn_search_ms"] = (time.perf_counter() - t0) * 1e3
    n_total_hits = sum(len(r) for r in nn_results)

    # (F) full _select_and_filter_candidates (includes E inside)
    t0 = time.perf_counter()
    cands_by_fid = _select_and_filter_candidates(quads, quad_db)
    times["F_filter_loop_ms"] = (time.perf_counter() - t0) * 1e3

    # (G) full identify() (includes F + sequence + verification)
    query_duration_sec = len(signal) / sr
    t0 = time.perf_counter()
    match_result = identify(quads, peaks, quad_db, query_duration_sec)
    times["G_identify_ms"] = (time.perf_counter() - t0) * 1e3

    times["H_total_ms"] = sum(
        times[k] for k in ["A_load_audio_ms", "B_spectrogram_ms",
                            "C_peaks_ms", "D_build_quads_ms", "G_identify_ms"]
    )

    extra = {
        "n_quads":       len(quads),
        "n_peaks":       len(peaks),
        "n_nn_hits":     n_total_hits,
        "n_cand_fids":   len(cands_by_fid),
        "best_match":    match_result.best_match,
    }
    results.append({**times, **extra})
    print(f"Query {row['track_id']}: {times['H_total_ms']:.0f} ms total | "
          f"quads={extra['n_quads']}, nn_hits={extra['n_nn_hits']}, "
          f"cand_fids={extra['n_cand_fids']}, match={extra['best_match']}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n── Averages ──────────────────────────────────────────────────────────")
cols = ["A_load_audio_ms","B_spectrogram_ms","C_peaks_ms","D_build_quads_ms",
        "E_nn_search_ms","F_filter_loop_ms","G_identify_ms","H_total_ms",
        "n_quads","n_nn_hits","n_cand_fids"]
df = pd.DataFrame(results)
for col in cols:
    print(f"  {col:<25}: {df[col].mean():>8.1f}  (min={df[col].min():.1f}, max={df[col].max():.1f})")

print("\nNote: F_filter_loop_ms includes E_nn_search_ms.")
print("      G_identify_ms includes F_filter_loop_ms + sequence + verification.")
