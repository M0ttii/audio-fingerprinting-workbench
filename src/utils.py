"""
Utility functions for the Audio Fingerprinting Benchmark (Bachelorarbeit).

Handles FMA metadata loading, genre-stratified dataset partitioning,
MUSAN splitting, pfann list export, and diagnostic helpers.

All random operations use seed=42 by default for reproducibility.
"""

import json
import logging
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FMA Metadata Loading
# ---------------------------------------------------------------------------

def load_fma_metadata(fma_path: Union[str, Path]) -> pd.DataFrame:
    """Load FMA tracks.csv, filter to medium subset, and check file presence.

    Reads the multi-level-header tracks.csv from the fma_metadata/ directory
    (expected as a sibling of fma_path), retains only 'medium'-subset tracks,
    derives each track's expected .mp3 path, and drops tracks whose file is
    missing from disk.

    FMA directory layout assumed:
        fma_path/         (e.g. data/fma_medium/)
            000/000002.mp3
            000/000005.mp3
            ...
        fma_path/../fma_metadata/tracks.csv

    Args:
        fma_path: Absolute or relative path to the fma_medium audio directory.

    Returns:
        DataFrame indexed by track_id (int) with columns:
            genre    (str):   Top-level genre; 'Unknown' when missing in CSV.
            duration (float): Track duration in seconds.
            filepath (str):   Absolute path to the .mp3 file.
        Only tracks whose .mp3 file exists on disk are included.

    Raises:
        FileNotFoundError: If tracks.csv cannot be found.
        KeyError: If expected multi-level columns are missing in tracks.csv.
    """
    fma_path = Path(fma_path).resolve()

    # tracks.csv lives in fma_metadata/, which is a sibling of fma_medium/
    candidates = [
        fma_path.parent / "fma_metadata" / "tracks.csv",
        fma_path / "fma_metadata" / "tracks.csv",
    ]
    tracks_csv = next((c for c in candidates if c.exists()), None)
    if tracks_csv is None:
        raise FileNotFoundError(
            f"tracks.csv not found. Searched:\n"
            + "\n".join(f"  {c}" for c in candidates)
        )

    logger.info("Reading FMA metadata from %s", tracks_csv)

    # FMA tracks.csv uses a two-row column header plus an index-name row.
    # pandas reads the index-name row ("track_id") as a data row — drop it
    # afterwards by coercing the index to int (non-numeric rows become NaN).
    df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df[df.index.notna()].copy()
    df.index = df.index.astype(int)
    df.index.name = "track_id"

    # Filter to fma_medium subset
    subset_col = ("set", "subset")
    if subset_col not in df.columns:
        raise KeyError(
            f"Column {subset_col} not found in tracks.csv. "
            f"Available top-level groups: {df.columns.get_level_values(0).unique().tolist()}"
        )
    df = df[df[subset_col].isin(["small", "medium"])].copy()
    logger.info("fma_medium entries in tracks.csv: %d", len(df))

    # Extract genre and duration
    genre_col = ("track", "genre_top")
    dur_col = ("track", "duration")

    if genre_col not in df.columns:
        raise KeyError(f"Column {genre_col} not found in tracks.csv.")
    if dur_col not in df.columns:
        raise KeyError(f"Column {dur_col} not found in tracks.csv.")

    genres = df[genre_col].fillna("Unknown").astype(str)
    durations = pd.to_numeric(df[dur_col], errors="coerce").fillna(0.0)

    # Build filepath: FMA uses NNN/NNNNNN.mp3 where NNN = first 3 digits of
    # zero-padded 6-character track ID. (e.g. track 2 → 000/000002.mp3)
    filepaths = [
        str(fma_path / f"{tid:06d}"[:3] / f"{tid:06d}.mp3")
        for tid in df.index
    ]

    result = pd.DataFrame(
        {"genre": genres.values, "duration": durations.values, "filepath": filepaths},
        index=df.index,
    )

    # Drop tracks whose audio file is not present on disk
    exists_mask = np.array([Path(fp).exists() for fp in result["filepath"]])
    n_total = len(result)
    result = result[exists_mask].copy()
    n_found = len(result)
    n_missing = n_total - n_found

    if n_missing > 0:
        logger.warning(
            "%d / %d fma_medium tracks have no .mp3 on disk and are excluded.",
            n_missing, n_total,
        )
    print(
        f"FMA metadata loaded: {n_found} usable tracks "
        f"({n_missing} missing files excluded)."
    )
    return result


# ---------------------------------------------------------------------------
# Dataset Partitioning
# ---------------------------------------------------------------------------

def create_partitions(
    track_df: pd.DataFrame,
    n_train: int = 15_000,
    n_ref: int = 8_000,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Split FMA tracks into disjoint train / ref / ood_pool sets, genre-stratified.

    Draws n_train tracks for NeuralFP training and n_ref tracks for the
    shared reference database. The remainder becomes the OOD pool. Genre
    proportions are preserved across all three splits using stratified
    sampling. Tracks with missing genre labels are assigned 'Unknown'.

    Args:
        track_df: DataFrame as returned by load_fma_metadata() — indexed by
            track_id with a 'genre' column.
        n_train: Number of tracks for the training split.  Default: 15 000.
        n_ref:   Number of tracks for the reference-DB split. Default: 8 000.
        seed:    Random seed for reproducibility.

    Returns:
        Dict with keys 'train', 'ref', 'ood_pool', each mapping to a sorted
        list of integer track IDs.

    Raises:
        ValueError: If n_train + n_ref > len(track_df).
    """
    total = len(track_df)
    if n_train + n_ref > total:
        raise ValueError(
            f"Requested {n_train} train + {n_ref} ref = {n_train + n_ref} tracks, "
            f"but only {total} are available."
        )

    all_ids = track_df.index.tolist()
    all_genres = track_df["genre"].tolist()

    # Step 1: sample train set (stratified by genre)
    train_ids = _stratified_sample(all_ids, all_genres, n_train, seed=seed)
    train_set = set(train_ids)

    # Step 2: from remaining tracks, sample ref set (stratified by genre)
    remaining_ids = [tid for tid in all_ids if tid not in train_set]
    remaining_genres = [track_df.at[tid, "genre"] for tid in remaining_ids]
    ref_ids = _stratified_sample(remaining_ids, remaining_genres, n_ref, seed=seed + 1)
    ref_set = set(ref_ids)

    # Step 3: everything left is the OOD pool
    ood_pool_ids = [tid for tid in remaining_ids if tid not in ref_set]

    partitions = {
        "train":    sorted(train_ids),
        "ref":      sorted(ref_ids),
        "ood_pool": sorted(ood_pool_ids),
    }

    # Verify disjointness
    assert_disjoint(partitions["train"], partitions["ref"], partitions["ood_pool"])

    print(
        f"Partitions created: "
        f"train={len(partitions['train'])}, "
        f"ref={len(partitions['ref'])}, "
        f"ood_pool={len(partitions['ood_pool'])}"
    )
    return partitions


def create_dry_run_subsets(
    ref_ids: List[int],
    train_ids: List[int],
    ood_pool_ids: List[int],
    metadata_df: pd.DataFrame,
    n_ref: int = 50,
    n_train: int = 200,
    n_ood: int = 10,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Draw genre-stratified dry-run subsets from each partition.

    Takes small genre-proportional samples from the three partition sets so
    that the full benchmark pipeline can be tested end-to-end quickly without
    using the full dataset.

    Args:
        ref_ids:      Track IDs from the reference-DB partition.
        train_ids:    Track IDs from the training partition.
        ood_pool_ids: Track IDs from the OOD pool partition.
        metadata_df:  DataFrame from load_fma_metadata() — used to look up
                      genres for stratified sampling.
        n_ref:    Number of dry-run reference tracks.  Default: 50.
        n_train:  Number of dry-run training tracks.   Default: 200.
        n_ood:    Number of dry-run OOD tracks.        Default: 10.
        seed:     Random seed for reproducibility.

    Returns:
        Dict with keys 'dry_ref', 'dry_train', 'dry_ood', each a sorted
        list of integer track IDs.

    Raises:
        ValueError: If any requested n exceeds the size of its source partition.
    """
    def _get_genres(ids: List[int]) -> List[str]:
        return [
            metadata_df.at[tid, "genre"] if tid in metadata_df.index else "Unknown"
            for tid in ids
        ]

    if n_ref > len(ref_ids):
        raise ValueError(f"n_ref={n_ref} > len(ref_ids)={len(ref_ids)}")
    if n_train > len(train_ids):
        raise ValueError(f"n_train={n_train} > len(train_ids)={len(train_ids)}")
    if n_ood > len(ood_pool_ids):
        raise ValueError(f"n_ood={n_ood} > len(ood_pool_ids)={len(ood_pool_ids)}")

    dry_ref = _stratified_sample(ref_ids, _get_genres(ref_ids), n_ref, seed=seed)
    dry_train = _stratified_sample(train_ids, _get_genres(train_ids), n_train, seed=seed + 1)
    dry_ood = _stratified_sample(ood_pool_ids, _get_genres(ood_pool_ids), n_ood, seed=seed + 2)

    subsets = {
        "dry_ref":   sorted(dry_ref),
        "dry_train": sorted(dry_train),
        "dry_ood":   sorted(dry_ood),
    }

    assert_disjoint(subsets["dry_ref"], subsets["dry_train"], subsets["dry_ood"])

    print(
        f"Dry-run subsets: "
        f"dry_ref={len(subsets['dry_ref'])}, "
        f"dry_train={len(subsets['dry_train'])}, "
        f"dry_ood={len(subsets['dry_ood'])}"
    )
    return subsets


# ---------------------------------------------------------------------------
# Disjointness Assertion
# ---------------------------------------------------------------------------

def assert_disjoint(*id_lists: List[int]) -> None:
    """Assert that all provided ID lists are pairwise disjoint.

    Checks every pair of lists for overlap. Prints a diagnostic line for each
    pair regardless of outcome. Raises ValueError if any overlap is found.

    Args:
        *id_lists: Two or more lists (or sets) of integer track IDs.

    Raises:
        ValueError: If any two lists share at least one common ID.
                    The error message lists all overlapping pairs and their
                    overlapping elements.
    """
    if len(id_lists) < 2:
        print("assert_disjoint: only one list provided — nothing to check.")
        return

    sets = [set(lst) for lst in id_lists]
    violations = []

    for (i, a), (j, b) in combinations(enumerate(sets), 2):
        overlap = a & b
        if overlap:
            msg = f"  Lists [{i}] and [{j}]: {len(overlap)} common IDs — OVERLAP!"
            print(msg)
            violations.append((i, j, sorted(overlap)))
        else:
            print(f"  Lists [{i}] and [{j}]: no overlap — OK")

    if violations:
        detail = "; ".join(
            f"[{i}]∩[{j}]={ids[:5]}{'...' if len(ids) > 5 else ''}"
            for i, j, ids in violations
        )
        raise ValueError(f"Data leakage detected — overlapping partition(s): {detail}")

    print("assert_disjoint: all lists are disjoint. ✓")


# ---------------------------------------------------------------------------
# MUSAN Splitting
# ---------------------------------------------------------------------------

def split_musan(
    musan_path: Union[str, Path],
    split: float = 0.8,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split MUSAN speech and noise files into train and eval sets.

    Collects all .wav files from the 'speech/' and 'noise/' subdirectories
    of musan_path. Deliberately excludes the 'music/' category to avoid
    a second audio source interfering with fingerprint evaluation.
    Files are shuffled deterministically and split by the given ratio.

    80% of files are assigned to training augmentation (noise for NeuralFP).
    20% are reserved for evaluation queries (applied to generate distorted
    queries in NB 01). The same files must never appear in both splits.

    Args:
        musan_path: Path to the MUSAN root directory containing
                    'speech/', 'noise/', and optionally 'music/'.
        split:      Fraction of files to assign to the training set.
                    Default: 0.8 (80 % train, 20 % eval).
        seed:       Random seed for reproducibility.

    Returns:
        Dict with keys:
            'train' (List[str]): Absolute paths of training noise files.
            'eval'  (List[str]): Absolute paths of evaluation noise files.

    Raises:
        FileNotFoundError: If musan_path does not exist.
        ValueError: If neither 'speech/' nor 'noise/' subdirectory is found.
    """
    musan_path = Path(musan_path).resolve()
    if not musan_path.exists():
        raise FileNotFoundError(f"MUSAN directory not found: {musan_path}")

    categories = ["speech", "noise"]  # music excluded intentionally
    wav_files: List[str] = []

    for category in categories:
        cat_dir = musan_path / category
        if not cat_dir.exists():
            logger.warning("MUSAN category directory not found: %s", cat_dir)
            continue
        found = sorted(str(p) for p in cat_dir.rglob("*.wav"))
        logger.info("MUSAN %s: %d .wav files", category, len(found))
        wav_files.extend(found)

    if not wav_files:
        raise ValueError(
            f"No .wav files found in {musan_path}/speech/ or {musan_path}/noise/."
        )

    # Deterministic shuffle
    rng = random.Random(seed)
    wav_files_shuffled = sorted(wav_files)  # sort first for stable base order
    rng.shuffle(wav_files_shuffled)

    n_train = int(len(wav_files_shuffled) * split)
    train_files = wav_files_shuffled[:n_train]
    eval_files = wav_files_shuffled[n_train:]

    # Verify disjointness
    overlap = set(train_files) & set(eval_files)
    assert len(overlap) == 0, f"MUSAN split produced overlap: {overlap}"

    print(
        f"MUSAN split: {len(train_files)} train files, "
        f"{len(eval_files)} eval files "
        f"(total {len(wav_files_shuffled)}, split={split:.0%})."
    )
    return {"train": train_files, "eval": eval_files}


# ---------------------------------------------------------------------------
# pfann List Export
# ---------------------------------------------------------------------------

def reanchor_paths(
    paths: List[str],
    new_base: Union[str, Path],
) -> List[str]:
    """Re-anchor absolute paths to a new base directory on the current machine.

    Searches each path for a component whose name matches ``new_base.name``
    and replaces everything up to and including that component with
    ``new_base``.  This allows paths generated on machine A (e.g. a Mac
    laptop) to be used unchanged on machine B (e.g. a Vertex node) as long as
    both machines share the same directory name for the data root.

    Example::
        paths    = ["/Users/mac/proj/data/musan/speech/a.wav"]
        new_base = Path("/home/vertex/proj/data/musan")
        → ["/home/vertex/proj/data/musan/speech/a.wav"]

    Args:
        paths:    List of absolute path strings to re-anchor.
        new_base: Absolute path to the directory that replaces the old root.
                  Matching is done on ``new_base.name`` (the last component).

    Returns:
        List of re-anchored path strings.  Paths that do not contain
        ``new_base.name`` as a component are returned unchanged.
    """
    new_base = Path(new_base).resolve()
    result: List[str] = []
    for p in paths:
        parts = Path(p).parts
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == new_base.name:
                result.append(str(new_base / Path(*parts[i + 1:])))
                break
        else:
            result.append(p)  # no matching component — return unchanged
    return result


def write_path_list(
    paths: List[str],
    out_path: Union[str, Path],
    relative_to: Optional[Union[str, Path]] = None,
) -> None:
    """Write a list of file paths to a plain-text .txt file (one path per line).

    Used to write pfann-compatible audio lists for paths that are not FMA
    track IDs (e.g. MUSAN noise files).  With ``relative_to`` the paths are
    made relative to a base directory so the list remains portable across
    machines.  The corresponding pfann config key (``noise.dir``) must then
    be set to the absolute path of that base directory at runtime.

    Args:
        paths:       List of absolute path strings to write.
        out_path:    Destination .txt file.  Parent directories are created
                     automatically.
        relative_to: If provided, each path is written relative to this
                     directory.  All paths must be descendants of it.
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base: Optional[Path] = Path(relative_to).resolve() if relative_to is not None else None

    lines: List[str] = []
    for p in paths:
        if base is not None:
            lines.append(str(Path(p).relative_to(base)))
        else:
            lines.append(str(p))

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info("Wrote path list: %s (%d entries)", out_path, len(lines))
    print(f"pfann list written: {out_path} ({len(lines)} files).")


def export_pfann_list(
    track_ids: List[int],
    audio_dir: Union[str, Path],
    out_path: Union[str, Path],
    relative_to: Optional[Union[str, Path]] = None,
) -> None:
    """Write a .txt file of FMA audio paths for pfann's builder.py / train.py.

    pfann expects a plain text file with one audio path per line and no
    header.  The FMA directory structure is NNN/NNNNNN.mp3, where NNN is
    the first three characters of the zero-padded 6-digit track ID.

    By default absolute paths are written.  Pass ``relative_to`` to write
    paths relative to a base directory instead — the corresponding pfann
    config key (``music_dir``) must then be set to the absolute path of that
    base directory at runtime.  This makes the list portable across machines.

    Example for track_id=2:    <audio_dir>/000/000002.mp3
    Example for track_id=1234: <audio_dir>/001/001234.mp3

    Args:
        track_ids:   Iterable of integer FMA track IDs to include.
        audio_dir:   Path to the fma_medium audio directory (parent of NNN/).
        out_path:    Destination .txt file.  Parent directories are created
                     if they do not already exist.
        relative_to: If provided, paths are written relative to this
                     directory.  All constructed paths must be descendants
                     of it.
    """
    audio_dir = Path(audio_dir).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base: Optional[Path] = Path(relative_to).resolve() if relative_to is not None else None

    lines = []
    for tid in track_ids:
        tid_str = f"{tid:06d}"
        filepath = audio_dir / tid_str[:3] / f"{tid_str}.mp3"
        if base is not None:
            lines.append(str(filepath.relative_to(base)))
        else:
            lines.append(str(filepath))

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info("Wrote pfann list: %s (%d entries)", out_path, len(lines))
    print(f"pfann list written: {out_path} ({len(lines)} tracks).")


# ---------------------------------------------------------------------------
# Diagnostic Helpers
# ---------------------------------------------------------------------------

def print_genre_distribution(
    track_ids: List[int],
    metadata_df: pd.DataFrame,
) -> None:
    """Print the genre distribution for a set of track IDs.

    Looks up each track ID in metadata_df and counts genres. Useful for
    verifying that stratified sampling produced balanced splits.

    Args:
        track_ids:   Iterable of integer FMA track IDs.
        metadata_df: DataFrame from load_fma_metadata() with a 'genre' column
                     indexed by track_id.
    """
    ids = [tid for tid in track_ids if tid in metadata_df.index]
    missing = len(track_ids) - len(ids)

    genres = [metadata_df.at[tid, "genre"] for tid in ids]
    counts = Counter(genres)
    total = sum(counts.values())

    print(f"Genre distribution ({total} tracks, {missing} IDs not in metadata):")
    for genre, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"  {genre:<25} {count:>6}  ({pct:5.1f}%)")


def print_missing_files(
    track_ids: List[int],
    metadata_df: pd.DataFrame,
) -> None:
    """Print track IDs whose expected audio file is missing from disk.

    Checks the 'filepath' column in metadata_df for each given track ID and
    reports any file that does not exist. Useful for diagnosing incomplete
    downloads before running the pipeline.

    Args:
        track_ids:   Iterable of integer FMA track IDs to check.
        metadata_df: DataFrame from load_fma_metadata() with a 'filepath'
                     column indexed by track_id.
    """
    missing = []
    not_in_meta = []

    for tid in track_ids:
        if tid not in metadata_df.index:
            not_in_meta.append(tid)
            continue
        fp = Path(metadata_df.at[tid, "filepath"])
        if not fp.exists():
            missing.append((tid, str(fp)))

    if not_in_meta:
        print(f"Track IDs not in metadata ({len(not_in_meta)}): {not_in_meta[:10]}"
              + ("..." if len(not_in_meta) > 10 else ""))

    if missing:
        print(f"Missing audio files ({len(missing)}):")
        for tid, fp in missing[:20]:
            print(f"  track_id={tid}: {fp}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more.")
    else:
        print(f"All {len(track_ids)} files present on disk. ✓")


# ---------------------------------------------------------------------------
# Duration Filtering
# ---------------------------------------------------------------------------

def filter_and_replenish_by_duration(
    selected_ids: List[int],
    pool_ids: List[int],
    metadata_df: pd.DataFrame,
    min_dur: float = 30.0,
    seed: int = 42,
) -> List[int]:
    """Filter selected IDs to those meeting minimum duration, replenishing from pool.

    Used after partitioning to enforce the ≥ 30 s duration requirement.
    Tracks shorter than min_dur are dropped and replaced by random draws from
    pool_ids (which must not overlap selected_ids). Replenishment is done
    randomly (not genre-stratified) for simplicity.

    Args:
        selected_ids: Initial list of track IDs (e.g. dry_ref from
                      create_dry_run_subsets).
        pool_ids:     Reservoir of additional track IDs to draw from.
                      Must be disjoint from selected_ids.
        metadata_df:  DataFrame from load_fma_metadata() with a 'duration'
                      column indexed by track_id.
        min_dur:      Minimum duration in seconds.  Default: 30.0.
        seed:         Random seed for reproducibility.

    Returns:
        List of track IDs all meeting min_dur.  Length equals len(selected_ids)
        if enough candidates are available; otherwise as many as possible.

    Raises:
        ValueError: If pool_ids overlaps with selected_ids.
    """
    overlap = set(selected_ids) & set(pool_ids)
    if overlap:
        raise ValueError(
            f"filter_and_replenish_by_duration: pool_ids overlaps selected_ids "
            f"({len(overlap)} common IDs)."
        )

    def _dur(tid: int) -> float:
        if tid not in metadata_df.index:
            return 0.0
        return float(metadata_df.at[tid, "duration"])

    kept = [tid for tid in selected_ids if _dur(tid) >= min_dur]
    n_dropped = len(selected_ids) - len(kept)

    if n_dropped == 0:
        print(f"filter_and_replenish: all {len(kept)} tracks meet ≥{min_dur}s. ✓")
        return kept

    print(
        f"filter_and_replenish: {n_dropped} tracks < {min_dur}s dropped; "
        f"replenishing from pool ({len(pool_ids)} candidates)."
    )

    # Candidates: pool tracks that meet the duration requirement, not already kept
    kept_set = set(kept)
    candidates = [tid for tid in pool_ids if _dur(tid) >= min_dur and tid not in kept_set]

    rng = np.random.default_rng(seed)
    n_fill = min(n_dropped, len(candidates))
    if n_fill < n_dropped:
        logger.warning(
            "filter_and_replenish: can only fill %d / %d slots (pool exhausted).",
            n_fill, n_dropped,
        )

    if n_fill > 0:
        chosen = rng.choice(candidates, size=n_fill, replace=False).tolist()
        kept.extend(chosen)

    print(f"filter_and_replenish: final count = {len(kept)} tracks.")
    return kept


# ---------------------------------------------------------------------------
# Partition Persistence
# ---------------------------------------------------------------------------

def save_partitions(
    partitions: Dict[str, List[int]],
    out_dir: Union[str, Path],
) -> None:
    """Save each partition as a separate JSON file under out_dir.

    Each key in partitions is written to <out_dir>/<key>.json as a JSON
    array of integers.  Creates out_dir if it does not exist.

    Args:
        partitions: Dict mapping partition name → list of int track IDs.
                    E.g. {'train': [...], 'ref': [...], 'ood_pool': [...],
                           'dry_ref': [...], 'dry_train': [...], 'dry_ood': [...]}.
        out_dir:    Destination directory.  Created if absent.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, ids in partitions.items():
        out_path = out_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(sorted(ids), fh)
        print(f"Saved {name}.json  ({len(ids)} IDs) → {out_path}")


def load_partitions(out_dir: Union[str, Path]) -> Dict[str, List[int]]:
    """Load all partition JSON files from out_dir.

    Args:
        out_dir: Directory produced by save_partitions().

    Returns:
        Dict mapping partition name → list of int track IDs.

    Raises:
        FileNotFoundError: If out_dir does not exist.
    """
    out_dir = Path(out_dir).resolve()
    if not out_dir.exists():
        raise FileNotFoundError(f"Partitions directory not found: {out_dir}")

    partitions: Dict[str, List[int]] = {}
    for json_file in sorted(out_dir.glob("*.json")):
        name = json_file.stem
        with open(json_file, "r", encoding="utf-8") as fh:
            partitions[name] = json.load(fh)
    return partitions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stratified_sample(
    ids: List[int],
    genres: List[str],
    n: int,
    seed: int,
) -> List[int]:
    """Sample n IDs from ids with genre-proportional stratification.

    Each genre's quota is computed as round(n * genre_fraction). The last
    genre absorbs any rounding remainder so that exactly n IDs are returned.
    When a genre has fewer members than its quota, all members are taken.

    Args:
        ids:    List of integer IDs to sample from.
        genres: Parallel list of genre labels for each ID.
        n:      Number of IDs to return.
        seed:   NumPy random seed for reproducibility.

    Returns:
        List of n selected IDs (order is not guaranteed).

    Raises:
        ValueError: If n > len(ids).
    """
    if n > len(ids):
        raise ValueError(f"Cannot sample {n} from {len(ids)} IDs.")
    if n == len(ids):
        return list(ids)

    rng = np.random.default_rng(seed)

    # Group IDs by genre (deterministic order via sorted)
    genre_to_ids: Dict[str, List[int]] = {}
    for tid, genre in zip(ids, genres):
        genre_to_ids.setdefault(genre, []).append(tid)

    total = len(ids)
    genres_sorted = sorted(genre_to_ids.keys())

    selected: List[int] = []
    allocated = 0

    for i, genre in enumerate(genres_sorted):
        g_ids = genre_to_ids[genre]
        is_last = (i == len(genres_sorted) - 1)

        if is_last:
            # Last genre takes exactly the remaining quota
            quota = n - allocated
        else:
            quota = round(n * len(g_ids) / total)

        # Cannot take more than available in this genre
        quota = min(quota, len(g_ids))
        quota = max(0, quota)

        if quota > 0:
            chosen = rng.choice(g_ids, size=quota, replace=False).tolist()
            selected.extend(chosen)
            allocated += quota

    # Safety: if rounding left us short (rare edge case), top up randomly
    if len(selected) < n:
        remaining_pool = list(set(ids) - set(selected))
        shortfall = n - len(selected)
        if shortfall <= len(remaining_pool):
            extra = rng.choice(remaining_pool, size=shortfall, replace=False).tolist()
            selected.extend(extra)

    return selected
