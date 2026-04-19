"""
GTZAN-specific helpers for the Audio Fingerprinting Benchmark (NB 07).

Provides utilities for:
- Building a metadata DataFrame for GTZAN (compatible with build_shazam_index /
  build_quad_index which expect a 'filepath' column indexed by track_id).
- Exporting a pfann-compatible reference list from GTZAN file paths.
- Per-query matching with individual FAISS timing, using a path→track_id dict
  instead of stem-parsing (GTZAN filenames like blues.00000.wav cannot be
  parsed as integers).

GTZAN file convention:
    gtzan/<genre>/<genre>.<NNNNN>.wav  (N = zero-padded 5-digit index 0–99)
    e.g. gtzan/blues/blues.00000.wav

Track-ID assignment (10 genres × 100 songs, alphabetical genre order):
    blues:      0–99     classical: 100–199   country: 200–299
    disco:     300–399   hiphop:    400–499   jazz:    500–599
    metal:     600–699   pop:       700–799   reggae:  800–899
    rock:      900–999

Reference: NB 07 — GTZAN Experiment.
"""

import logging
import sys
import time as time_mod
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from metrics import classify_result

logger = logging.getLogger(__name__)

# Canonical alphabetical genre order — determines track_id assignment.
GTZAN_GENRES: List[str] = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]


# ---------------------------------------------------------------------------
# Metadata DataFrame
# ---------------------------------------------------------------------------

def build_gtzan_metadata_df(gtzan_dir: Union[str, Path]) -> pd.DataFrame:
    """Build a metadata DataFrame for GTZAN, compatible with build_shazam_index.

    Assigns integer track_ids 0–999 by iterating over genres in alphabetical
    order (GTZAN_GENRES), 100 songs per genre.  The resulting DataFrame mirrors
    the structure returned by load_fma_metadata():

        index:    track_id (int, 0–999)
        columns:  genre (str), filepath (str)

    Args:
        gtzan_dir: Absolute or relative path to the GTZAN audio root directory,
                   containing subdirectories blues/, classical/, …, rock/.

    Returns:
        DataFrame indexed by track_id (0–999) with columns:
            genre    (str): Genre label (e.g. "blues", "classical").
            filepath (str): Absolute path to the .wav file.
    """
    gtzan_dir = Path(gtzan_dir).resolve()
    rows: List[dict] = []

    for genre_idx, genre in enumerate(GTZAN_GENRES):
        genre_dir = gtzan_dir / genre
        for song_idx in range(100):
            filename = f"{genre}.{song_idx:05d}.wav"
            filepath = genre_dir / filename
            track_id = genre_idx * 100 + song_idx
            rows.append({
                "track_id": track_id,
                "genre":    genre,
                "filepath": str(filepath),
            })

    df = pd.DataFrame(rows).set_index("track_id")
    df.index.name = "track_id"
    return df


# ---------------------------------------------------------------------------
# pfann Reference List Export
# ---------------------------------------------------------------------------

def export_gtzan_pfann_ref_list(
    gtzan_df: pd.DataFrame,
    out_path: Union[str, Path],
) -> None:
    """Write a pfann-compatible .txt file of GTZAN audio paths.

    One absolute path per line, ordered by track_id (ascending).
    The pfann builder.py reads this file and stores the paths in songList.txt
    inside the constructed DB directory.

    Args:
        gtzan_df: DataFrame as returned by build_gtzan_metadata_df().
        out_path: Destination .txt file. Parent directories are created
                  if they do not already exist.

    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = gtzan_df.sort_index()["filepath"].tolist()
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(paths) + "\n")

    logger.info("Wrote GTZAN pfann ref list: %s (%d entries)", out_path, len(paths))
    print(f"GTZAN pfann ref list written: {out_path} ({len(paths)} tracks).")


# ---------------------------------------------------------------------------
# Path-to-Track-ID Mapping
# ---------------------------------------------------------------------------

def build_path_to_id_mapping(gtzan_df: pd.DataFrame) -> Dict[str, int]:
    """Build a reverse mapping from absolute filepath to track_id.

    Used by match_gtzan_queries_timed() to convert the answer path returned
    by the pfann FAISS index back to an integer track_id.

    Args:
        gtzan_df: DataFrame as returned by build_gtzan_metadata_df().

    Returns:
        Dict mapping filepath (str) → track_id (int).
    """
    return {str(row["filepath"]): int(tid) for tid, row in gtzan_df.iterrows()}


# ---------------------------------------------------------------------------
# Per-query matching with individual FAISS timing (GTZAN version)
# ---------------------------------------------------------------------------

def match_gtzan_queries_timed(
    query_list_path: Union[str, Path],
    db_dir: Union[str, Path],
    manifest: pd.DataFrame,
    path_to_id: Dict[str, int],
    out_path: Union[str, Path],
    score_threshold: float = 0.54,
    pfann_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Match GTZAN queries against pfann DB with per-query timing.

    Identical to neural_fp.match_queries_timed() except that predicted_id is
    resolved via a path_to_id dict instead of stem-based int-parsing.  This is
    necessary because GTZAN filenames (e.g. blues.00000.wav) cannot be parsed
    as integers directly.

    Measures wall-clock time for the full query pipeline:
    embedding generation (mel spectrogram → FpNetwork → L2 norm)
    AND FAISS lookup (db.query_embeddings).  This matches the methodology
    used by Shazam and Quad, which time fingerprint extraction + DB lookup
    together.

    Pipeline per query:
        1. Load audio via pfann's MusicDataset (resample, mono, segment)
        2. time.perf_counter() START
        3. Generate embeddings via model (mel spectrogram → FpNetwork → L2 norm)
        4. db.query_embeddings(embeddings)  — FAISS search + reranking
        5. time.perf_counter() STOP
        6. query_time_ms = (stop - start) * 1000

    Args:
        query_list_path: Path to .txt file with one query WAV path per line
                         (manifest order — see neural_fp.export_query_list()).
        db_dir:          Path to pfann DB directory (contains model.pt,
                         configs.json, landmarkValue, landmarkKey, songList.txt).
        manifest:        Query manifest DataFrame in verbindliches Format.
        path_to_id:      Dict mapping absolute GTZAN filepath (str) → track_id
                         (int) as built by build_path_to_id_mapping().
        out_path:        Destination CSV path for the raw-result file.
        score_threshold: Minimum FAISS similarity score to accept a match.
                         Below threshold → predicted_id = None.  Default 0.54.
        pfann_dir:       Path to pfann source directory for imports.
                         Defaults to src/pfann/ relative to this file.

    Returns:
        DataFrame in the verbindliches Raw-Result-Format.

    Reference: neural_fp.match_queries_timed();
    """
    import torch
    from torch.utils.data import DataLoader
    import tqdm as tqdm_mod

    db_dir = Path(db_dir)
    out_path = Path(out_path)

    if pfann_dir is None:
        pfann_dir = Path(__file__).resolve().parent / "pfann"
    pfann_dir = Path(pfann_dir).resolve()

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

        # --- Dataset + loader ---
        dataset = MusicDataset(str(query_list_path), params)
        loader = DataLoader(dataset, num_workers=0, batch_size=None)

        # --- Per-query matching loop ---
        raw_rows = []

        for dat in tqdm_mod.tqdm(loader, desc="Matching GTZAN queries",
                                 total=len(dataset)):
            i, name, wav = dat
            i = int(i)

            if wav.shape[0] == 0:
                logger.warning("Load error for GTZAN query %d: %s", i, name)
                raw_rows.append({
                    "query_idx":    i,
                    "answer_path":  None,
                    "score":        float("nan"),
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

            # L2 norm assertion (outside timing — debug check, not pipeline)
            norms = np.linalg.norm(embeddings, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5), (
                f"Embeddings not L2-normalised: norms in "
                f"[{norms.min():.6f}, {norms.max():.6f}]"
            )

            query_time_ms = (t1 - t0) * 1000.0
            ans_path = db.songList[ans_idx] if ans_idx >= 0 else None

            raw_rows.append({
                "query_idx":    i,
                "answer_path":  ans_path,
                "score":        float(sco),
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

    # --- GTZAN-specific: map answer_path → track_id via path_to_id dict ---
    def _lookup_gtzan_id(path_str) -> Optional[int]:
        if path_str is None:
            return None
        path_str = str(path_str).strip()
        if path_str in ("", "error"):
            return None
        # Try exact match, then normalised absolute path
        if path_str in path_to_id:
            return path_to_id[path_str]
        normed = str(Path(path_str).resolve())
        return path_to_id.get(normed, None)

    result["predicted_id"] = result["answer_path"].apply(_lookup_gtzan_id)

    # --- Score thresholding ---
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    below_threshold = result["score"] < score_threshold
    n_rejected = int(below_threshold.sum())
    if n_rejected > 0:
        print(f"  Score threshold {score_threshold}: {n_rejected}/{len(result)} "
              f"matches rejected (score too low → predicted_id=None).")
        result.loc[below_threshold, "predicted_id"] = None

    # --- Classify results ---
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

    times = output_df["query_time_ms"].dropna()
    print(f"\nneuralFP GTZAN raw result: {out_path} ({len(output_df)} rows)")
    if len(times) > 0:
        print(f"  query_time_ms — mean: {times.mean():.2f}, "
              f"median: {times.median():.2f}, std: {times.std():.2f}, "
              f"p95: {times.quantile(0.95):.2f}")

    return output_df