"""
NeuralFP (pfann) helper functions for the Audio Fingerprinting Benchmark.

Provides utilities for:
- Exporting query lists for pfann's matcher.py
- Per-query matching with individual FAISS timing (match_queries_timed)
- Parsing matcher output (TSV + detail.csv) into the standard raw-result format

Reference: Chang et al. (2021) Sections 3–4.
"""

import logging
import sys
import time as time_mod
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from metrics import classify_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query List Export
# ---------------------------------------------------------------------------

def export_query_list(manifest: pd.DataFrame, out_path: Union[str, Path]) -> None:
    """Write a pfann-compatible .txt file of query audio paths.

    One absolute path per line, in manifest row order.  Order is critical:
    pfann processes queries in list order, so TSV and detail.csv rows align
    positionally with manifest rows.

    Args:
        manifest: Query manifest DataFrame with a 'query_path' column
                  containing absolute paths to WAV files.
        out_path: Destination .txt file.  Parent directories are created
                  if they do not exist.

    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = manifest["query_path"].tolist()
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(str(p) for p in paths) + "\n")

    logger.info("Wrote query list: %s (%d entries)", out_path, len(paths))
    print(f"Query list written: {out_path} ({len(paths)} queries).")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_track_id_from_path(path_str) -> Optional[int]:
    """Extract FMA track_id from a matched-song path string.

    FMA filename convention: zero-padded 6-digit track ID (e.g. 000002.mp3).
    Parses the filename stem as an integer.

    Args:
        path_str: Path string to the matched audio file, or None / 'error'.

    Returns:
        Integer track ID, or None if path is empty, None, or 'error'.

    """
    if path_str is None:
        return None
    path_str = str(path_str).strip()
    if path_str in ("", "error"):
        return None
    try:
        return int(Path(path_str).stem)
    except (ValueError, TypeError):
        logger.warning("_extract_track_id_from_path: cannot parse '%s'", path_str)
        return None


# ---------------------------------------------------------------------------
# Per-query matching with individual FAISS timing
# ---------------------------------------------------------------------------

def match_queries_timed(
    query_list_path: Union[str, Path],
    db_dir: Union[str, Path],
    manifest: pd.DataFrame,
    out_path: Union[str, Path],
    score_threshold: float = 0.7,
    pfann_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Match queries against pfann DB with per-query timing.

    Measures wall-clock time for the full query pipeline:
    embedding generation (mel spectrogram → FpNetwork → L2 norm)
    AND FAISS lookup (db.query_embeddings).

    This matches the methodology used by Shazam and Quad, which time
    fingerprint extraction + DB lookup together.

    Pipeline per query:
        1. Load audio via pfann's MusicDataset (resample, mono, segment)
        2. time.perf_counter() START
        3. Generate embeddings via model (mel spectrogram → FpNetwork → L2 norm)
        4. db.query_embeddings(embeddings)  — FAISS search + reranking
        5. time.perf_counter() STOP
        6. query_time_ms = (stop - start) * 1000

    Args:
        query_list_path: Path to .txt file with one query WAV path per line
                         (manifest order).
        db_dir:          Path to pfann DB directory (contains model.pt,
                         configs.json, landmarkValue, landmarkKey, songList.txt).
        manifest:        Query manifest DataFrame in verbindliches Format.
        out_path:        Destination CSV path for the raw-result file.
        score_threshold: Minimum FAISS similarity score to accept a match.
                         Below threshold → predicted_id = None.  Default 0.7.
        pfann_dir:       Path to pfann source directory for imports.
                         Defaults to src/pfann/ relative to this file.

    """
    import torch
    from torch.utils.data import DataLoader
    import faiss  # noqa: F811 — needed for normalize_L2 assertion

    db_dir = Path(db_dir)
    out_path = Path(out_path)

    if pfann_dir is None:
        pfann_dir = Path(__file__).resolve().parent / "pfann"
    pfann_dir = Path(pfann_dir).resolve()

    # Temporarily add pfann to sys.path for its internal imports
    pfann_str = str(pfann_dir)
    added_to_path = pfann_str not in sys.path
    if added_to_path:
        sys.path.insert(0, pfann_str)

    try:
        import simpleutils
        from model import FpNetwork
        from datautil.melspec import build_mel_spec_layer
        from datautil.musicdata import MusicDataset
        from database import Database

        # --- Load config from DB directory ---
        params = simpleutils.read_config(str(db_dir / "configs.json"))

        d = params["model"]["d"]
        h = params["model"]["h"]
        u = params["model"]["u"]
        F_bin = params["n_mels"]
        segn = int(params["segment_size"] * params["sample_rate"])
        T = (segn + params["stft_hop"] - 1) // params["stft_hop"]

        # --- Load model ---
        device = torch.device("cpu")
        model = FpNetwork(d, h, u, F_bin, T, params["model"]).to(device)
        model.load_state_dict(
            torch.load(str(db_dir / "model.pt"), map_location=device)
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print(f"Model loaded from {db_dir / 'model.pt'}")

        # --- Load FAISS database ---
        db = Database(str(db_dir), params["indexer"], params["hop_size"])
        print(f"Database loaded: {db.index.ntotal} embeddings, "
              f"{len(db.songList)} songs")

        # --- Mel spectrogram layer ---
        mel = build_mel_spec_layer(params).to(device)

        # --- Dataset + loader (same as pfann matcher.py) ---
        dataset = MusicDataset(str(query_list_path), params)
        loader = DataLoader(dataset, num_workers=0, batch_size=None)

        # --- Per-query matching loop ---
        import tqdm
        raw_rows = []

        for dat in tqdm.tqdm(loader, desc="Matching queries", total=len(dataset)):
            i, name, wav = dat
            i = int(i)

            if wav.shape[0] == 0:
                logger.warning("match_queries_timed: load error for query %d: %s", i, name)
                raw_rows.append({
                    "query_idx": i,
                    "answer_path": None,
                    "score": float("nan"),
                    "query_time_ms": float("nan"),
                })
                continue

            # --- Embedding generation + FAISS lookup (both timed) ---
            t0 = time_mod.perf_counter()

            emb_parts = []
            for batch in torch.split(wav, 16):
                g = batch.to(device)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    g = mel(g)
                z = model.forward(g, norm=False).cpu()
                z = torch.nn.functional.normalize(z, p=2)
                emb_parts.append(z)
            embeddings = torch.cat(emb_parts).numpy()

            sco, (ans_idx, _offset), _song_score = db.query_embeddings(embeddings)
            t1 = time_mod.perf_counter()

            query_time_ms = (t1 - t0) * 1000.0

            # L2 norm assertion (outside timing — debug check, not pipeline)
            norms = np.linalg.norm(embeddings, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5), (
                f"Embeddings not L2-normalized: norms in "
                f"[{norms.min():.6f}, {norms.max():.6f}]"
            )

            ans_path = db.songList[ans_idx] if ans_idx >= 0 else None

            raw_rows.append({
                "query_idx": i,
                "answer_path": ans_path,
                "score": float(sco),
                "query_time_ms": query_time_ms,
            })

    finally:
        if added_to_path and pfann_str in sys.path:
            sys.path.remove(pfann_str)

    # --- Build result DataFrame ---
    res_df = pd.DataFrame(raw_rows)

    manifest_clean = manifest.copy().reset_index(drop=True)
    manifest_clean["query_idx"] = manifest_clean.index
    result = manifest_clean.merge(res_df, on="query_idx", how="left")

    # Extract predicted_id from matched path
    result["predicted_id"] = result["answer_path"].apply(_extract_track_id_from_path)

    # Score thresholding: reject low-confidence matches
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    below_threshold = result["score"] < score_threshold
    n_rejected = int(below_threshold.sum())
    if n_rejected > 0:
        logger.info(
            "Score threshold %.3f: rejecting %d / %d matches.",
            score_threshold, n_rejected, len(result),
        )
        print(f"  Score threshold {score_threshold}: {n_rejected}/{len(result)} "
              f"matches rejected (score too low → predicted_id=None).")
        result.loc[below_threshold, "predicted_id"] = None

    # Classify results via src/metrics.py
    result["result_class"] = result.apply(
        lambda row: classify_result(
            row["predicted_id"],
            row.get("ref_track_id"),
            bool(row["is_ood"]),
        ),
        axis=1,
    )

    result["system"] = "neuralFP"
    out_cols = [
        "system", "track_id", "ref_track_id", "is_ood",
        "predicted_id", "score", "result_class", "query_time_ms",
        "group", "condition", "duration_sec",
    ]
    output_df = result[out_cols].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False)

    # Summary
    times = output_df["query_time_ms"].dropna()
    print(f"\nneuralFP raw result CSV: {out_path} ({len(output_df)} rows)")
    if len(times) > 0:
        print(f"  query_time_ms — mean: {times.mean():.2f}, "
              f"median: {times.median():.2f}, std: {times.std():.2f}, "
              f"p95: {times.quantile(0.95):.2f}")

    return output_df
# ---------------------------------------------------------------------------
# Result Parsing (legacy — subprocess-based matcher)
# ---------------------------------------------------------------------------

def parse_pfann_results(
    tsv_path: Union[str, Path],
    detail_path: Union[str, Path],
    manifest: pd.DataFrame,
    out_path: Union[str, Path],
    score_threshold: float = 0.7,
    matcher_wall_time_s: Optional[float] = None,
) -> pd.DataFrame:
    """Parse pfann matcher output into the standard raw-result CSV format.

    Combines the tab-separated TSV output (query_path → matched_path),
    the detail.csv (score, time), and the manifest (ground-truth labels)
    to produce the benchmark's unified result format.

    Matching strategy:
        Primary join on query_path string.  Positional fallback is applied
        if the path-based join fails (e.g. path representation differs),
        provided TSV and manifest have the same number of rows.

    Track-ID extraction:
        matched_path filename stem (e.g. '000002') is parsed as int.
        'error' or empty matched_path → pred_id = None.

    Score thresholding:
        If the best FAISS similarity score for a query is below
        *score_threshold*, the match is rejected: predicted_id is set to
        None while the raw score is preserved.  This enables FN (in-DB)
        and TN (OOD) result classes.

    Time handling:
        pfann's detail.csv 'time' column contains an alignment offset,
        NOT wall-clock query latency.  Per-query latency is instead
        derived from *matcher_wall_time_s* (total subprocess wall-clock
        time divided evenly across queries).  If matcher_wall_time_s is
        not provided, query_time_ms is set to NaN.

    Args:
        tsv_path:    Path to neuralFP_matches.tsv (no header, tab-separated:
                     query_path <TAB> matched_path).
        detail_path: Path to neuralFP_matches_detail.csv (header: query,
                     answer, score, time, part_scores).
        manifest:    Query manifest DataFrame in verbindliches Format.
        out_path:    Destination CSV path for the raw-result file.
        score_threshold: Minimum FAISS similarity score to accept a match.
                     Matches below this threshold → predicted_id = None.
                     Default 0.7.
        matcher_wall_time_s: Total wall-clock seconds the matcher subprocess
                     took.  Used to compute per-query query_time_ms.
                     None → query_time_ms = NaN.

    Returns:
        DataFrame in the verbindliches Raw-Result-Format.

    Reference: Raw-Result-Format; pfann-Workflow Schritt 4.
    """
    tsv_path = Path(tsv_path)
    detail_path = Path(detail_path)
    out_path = Path(out_path)

    # --- Load TSV (no header, tab-separated) ---
    tsv_df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["query_path", "answer_path"],
        dtype=str,
    )
    tsv_df["query_path"] = tsv_df["query_path"].str.strip()
    tsv_df["answer_path"] = tsv_df["answer_path"].str.strip()

    # --- Load detail.csv ---
    detail_df = pd.read_csv(detail_path, dtype={"query": str})
    detail_df["query"] = detail_df["query"].astype(str).str.strip()

    # --- Join TSV + detail on query_path ---
    tsv_merged = tsv_df.merge(
        detail_df[["query", "score", "time"]],
        left_on="query_path",
        right_on="query",
        how="left",
    ).drop(columns=["query"])

    # Positional fallback: fill unmatched rows from detail.csv by row position
    null_mask = tsv_merged["score"].isna()
    if null_mask.any() and len(tsv_merged) == len(detail_df):
        n_fallback = int(null_mask.sum())
        logger.warning(
            "parse_pfann_results: %d rows had no path-match in detail.csv; "
            "using positional fallback.",
            n_fallback,
        )
        print(f"  WARNING: {n_fallback} detail rows matched positionally (path mismatch).")
        tsv_merged.loc[null_mask, "score"] = detail_df["score"].values[null_mask]
        tsv_merged.loc[null_mask, "time"] = detail_df["time"].values[null_mask]

    # --- Merge with manifest on query_path ---
    manifest_clean = manifest.copy()
    manifest_clean["query_path"] = manifest_clean["query_path"].astype(str).str.strip()

    result = manifest_clean.merge(tsv_merged, on="query_path", how="left")

    # Positional fallback if path-based merge failed (row count mismatch)
    if len(result) != len(manifest_clean):
        logger.warning(
            "parse_pfann_results: manifest merge produced %d rows (expected %d); "
            "falling back to positional alignment.",
            len(result), len(manifest_clean),
        )
        print(
            f"  WARNING: positional alignment used "
            f"(merge produced {len(result)} rows, expected {len(manifest_clean)})."
        )
        result = manifest_clean.copy()
        if len(tsv_merged) == len(manifest_clean):
            result["answer_path"] = tsv_merged["answer_path"].values
            result["score"] = pd.to_numeric(tsv_merged["score"].values, errors="coerce")
            result["time"] = pd.to_numeric(tsv_merged["time"].values, errors="coerce")
        else:
            logger.error(
                "parse_pfann_results: TSV has %d rows but manifest has %d — "
                "cannot align positionally.",
                len(tsv_merged), len(manifest_clean),
            )
            result["answer_path"] = None
            result["score"] = float("nan")
            result["time"] = float("nan")

    # --- Extract predicted_id (None for empty / 'error' matches) ---
    result["predicted_id"] = result["answer_path"].apply(_extract_track_id_from_path)

    # --- Numeric conversions ---
    result["score"] = pd.to_numeric(result["score"], errors="coerce")

    # --- Score thresholding: reject low-confidence matches ---
    below_threshold = result["score"] < score_threshold
    n_rejected = int(below_threshold.sum())
    if n_rejected > 0:
        logger.info(
            "Score threshold %.3f: rejecting %d / %d matches.",
            score_threshold, n_rejected, len(result),
        )
        print(f"  Score threshold {score_threshold}: {n_rejected}/{len(result)} "
              f"matches rejected (score too low → predicted_id=None).")
        result.loc[below_threshold, "predicted_id"] = None

    # --- Per-query latency from total matcher wall-clock time ---
    if matcher_wall_time_s is not None and len(result) > 0:
        per_query_ms = (matcher_wall_time_s / len(result)) * 1000.0
        result["query_time_ms"] = per_query_ms
        logger.info(
            "query_time_ms = %.2f ms (total %.1f s / %d queries).",
            per_query_ms, matcher_wall_time_s, len(result),
        )
    else:
        result["query_time_ms"] = float("nan")

    # --- result_class via classify_result() from src/metrics.py ---
    result["result_class"] = result.apply(
        lambda row: classify_result(
            row["predicted_id"],
            row.get("ref_track_id"),
            bool(row["is_ood"]),
        ),
        axis=1,
    )

    result["system"] = "neuralFP"

    out_cols = [
        "system", "track_id", "ref_track_id", "is_ood",
        "predicted_id", "score", "result_class", "query_time_ms",
        "group", "condition", "duration_sec",
    ]
    output_df = result[out_cols].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False)

    logger.info("Wrote neuralFP raw result: %s (%d rows)", out_path, len(output_df))
    print(f"neuralFP raw result CSV written: {out_path} ({len(output_df)} rows).")
    return output_df
