"""
Evaluation metrics for the Audio Fingerprinting Benchmark.

All functions operate on the verbindliche Raw-Result-Format.  
The DataFrame must contain at least the columns:

    system, track_id, ref_track_id, is_ood, predicted_id,
    score, result_class, query_time_ms, group, condition, duration_sec

Quad-specific functions additionally require:
    detected_time_scale, detected_freq_scale,
    true_time_scale, true_freq_scale

Reference: Testkonzept Phase 6 — Evaluation & Metriken.
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _is_none(value) -> bool:
    """Return True if value is None, NaN, or pd.NA.

    Handles all 'missing' representations that can appear when reading
    predicted_id / ref_track_id from CSV files.

    Args:
        value: Any scalar value.

    Returns:
        True if value represents a missing / no-match result.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return pd.isna(value)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Result classification
# ---------------------------------------------------------------------------

def classify_result(pred_id, ref_track_id, is_ood: bool) -> str:
    """Classify a single query result into one of four outcome classes.

    The four classes follow the standard confusion-matrix convention adapted
    for audio fingerprinting (Testkonzept Phase 2, Phase 6):

    In-DB queries (is_ood == False):
        TP  pred_id == ref_track_id   — correct match
        FN  pred_id is None           — missed match (song known but not found)
        FP  pred_id != ref_track_id   — wrong song returned

    OOD queries (is_ood == True, ref_track_id is None):
        TN  pred_id is None           — correctly returned no match
        FP  pred_id is not None       — spurious match to an unrelated song

    None/NaN detection handles both Python None and pandas NaN values
    that arise when reading CSV files.

    Args:
        pred_id:      Predicted track ID returned by the system, or None / NaN
                      when the system found no match.
        ref_track_id: Ground-truth track ID for in-DB queries.  None / NaN for
                      OOD queries.
        is_ood:       True if the query song is NOT in the reference database.

    Returns:
        One of "TP", "FP", "FN", "TN".
    """
    pred_missing = _is_none(pred_id)

    if not is_ood:
        # In-DB query
        if not pred_missing and not _is_none(ref_track_id):
            # Both present — compare IDs (cast to int to avoid float vs int)
            if int(pred_id) == int(ref_track_id):
                return "TP"
            else:
                return "FP"
        elif pred_missing:
            return "FN"
        else:
            # pred_id present but ref_track_id missing — should not happen
            # in well-formed data; treat conservatively as FP
            logger.warning(
                "classify_result: pred_id=%s present but ref_track_id is None "
                "for non-OOD query. Treating as FP.", pred_id,
            )
            return "FP"
    else:
        # OOD query
        if pred_missing:
            return "TN"
        else:
            return "FP"


# ---------------------------------------------------------------------------
# Hit Rate (Recall)
# ---------------------------------------------------------------------------

def compute_hit_rate(
    results_df: pd.DataFrame,
    filter_col: Optional[str] = None,
    filter_val=None,
) -> float:
    """Compute Hit Rate (Top-1 Recall) on in-DB queries.

    Hit Rate = TP / (TP + FN + FP)

    Because every in-DB query is classified as exactly one of TP, FN, or FP,
    the denominator equals the total number of in-DB queries after filtering.
    This matches the Top-1 Recall definition in Chang et al. (2021).

    Only rows where is_ood == False are included.  An optional column filter
    allows computing hit rate for a specific condition, group, or system.

    Args:
        results_df:  Raw-result DataFrame (see module docstring).
        filter_col:  Column name to filter on before computing the metric.
                     E.g. "condition", "group", "system".  None = no filter.
        filter_val:  Value to match in filter_col.
                     E.g. "A_original", "A", "shazam".

    Returns:
        Hit rate as a float in [0.0, 1.0].  Returns 0.0 if no qualifying
        rows exist after filtering.
    """
    indb = results_df[~results_df["is_ood"]].copy()

    if filter_col is not None:
        indb = indb[indb[filter_col] == filter_val]

    if len(indb) == 0:
        logger.warning(
            "compute_hit_rate: no in-DB rows after filter (%s=%s). "
            "Returning 0.0.", filter_col, filter_val,
        )
        return 0.0

    tp = (indb["result_class"] == "TP").sum()
    return float(tp) / len(indb)


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------

def compute_precision(results_df: pd.DataFrame) -> float:
    """Compute Precision over all queries (in-DB and OOD combined).

    Precision = TP / (TP + FP)

    False Positives arise from two sources:
      - In-DB queries where the wrong track was returned.
      - OOD queries where any track was returned (spurious match).

    Args:
        results_df: Raw-result DataFrame (see module docstring).

    Returns:
        Precision as a float in [0.0, 1.0].  Returns 1.0 if no positive
        predictions were made (TP + FP == 0), following the convention that
        a system making no predictions has trivially perfect precision.
    """
    tp = (results_df["result_class"] == "TP").sum()
    fp = (results_df["result_class"] == "FP").sum()
    denom = tp + fp
    return float(tp) / float(denom) if denom > 0 else 1.0


# ---------------------------------------------------------------------------
# Specificity
# ---------------------------------------------------------------------------

def compute_specificity(results_df: pd.DataFrame) -> float:
    """Compute Specificity on OOD queries only.

    Specificity = TN / (TN + FP)

    Measures how often the system correctly returns no match when the query
    song is not in the reference database.  A high specificity means the
    system rarely makes spurious matches.

    Only rows where is_ood == True are included.

    Args:
        results_df: Raw-result DataFrame (see module docstring).

    Returns:
        Specificity as a float in [0.0, 1.0].  Returns 1.0 if no OOD rows
        exist (vacuously perfect specificity).
    """
    ood = results_df[results_df["is_ood"]].copy()

    if len(ood) == 0:
        logger.warning("compute_specificity: no OOD rows. Returning 1.0.")
        return 1.0

    tn = (ood["result_class"] == "TN").sum()
    fp = (ood["result_class"] == "FP").sum()
    denom = tn + fp
    return float(tn) / float(denom) if denom > 0 else 1.0


# ---------------------------------------------------------------------------
# No-Match Rate
# ---------------------------------------------------------------------------

def compute_no_match_rate(results_df: pd.DataFrame) -> float:
    """Compute the False-Negative (no-match) rate on in-DB queries.

    No-Match Rate = FN / (TP + FN + FP)  =  FN / len(in-DB queries)

    Measures the fraction of in-DB queries for which the system returned
    no match at all, even though the song is in the reference database.
    Complement of Hit Rate when FP = 0; differs when FP > 0.

    Only rows where is_ood == False are included.

    Args:
        results_df: Raw-result DataFrame (see module docstring).

    Returns:
        No-match rate as a float in [0.0, 1.0].  Returns 0.0 if no in-DB
        rows exist.
    """
    indb = results_df[~results_df["is_ood"]]

    if len(indb) == 0:
        logger.warning("compute_no_match_rate: no in-DB rows. Returning 0.0.")
        return 0.0

    fn = (indb["result_class"] == "FN").sum()
    return float(fn) / len(indb)


# ---------------------------------------------------------------------------
# Timing statistics
# ---------------------------------------------------------------------------

def compute_time_stats(results_df: pd.DataFrame) -> dict:
    """Compute descriptive statistics of per-query latency.

    Summarises the query_time_ms column with mean, median, standard
    deviation, and 95th percentile.  Useful for the efficiency comparison
    table in NB 06.

    Args:
        results_df: Raw-result DataFrame containing a 'query_time_ms' column.

    Returns:
        Dict with keys:
            mean   (float): Mean query time in ms.
            median (float): Median query time in ms.
            std    (float): Standard deviation in ms.
            p95    (float): 95th-percentile query time in ms.

    Raises:
        KeyError: If 'query_time_ms' column is missing.
    """
    t = results_df["query_time_ms"].dropna()

    if len(t) == 0:
        logger.warning("compute_time_stats: no query_time_ms values.")
        return {"mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "p95": float("nan")}

    return {
        "mean":   float(t.mean()),
        "median": float(t.median()),
        "std":    float(t.std(ddof=1)),
        "p95":    float(t.quantile(0.95)),
    }


# ---------------------------------------------------------------------------
# Scale estimation error (Quad-specific)
# ---------------------------------------------------------------------------

def compute_scale_estimation_error(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute scale-factor estimation errors for the Quad system, Group A.

    Filters results_df to Quad (system == "quad"), Group A queries
    (group == "A"), and in-DB rows only.  Computes the absolute error
    between detected and true time/frequency scale factors.

    The true scale factors are known from the condition parameters:
        A_tempo_120 → true_time_scale = 1.20, true_freq_scale = 1.0
        A_speed_90  → true_time_scale = 0.90, true_freq_scale = 0.90
        A_original  → true_time_scale = 1.0,  true_freq_scale = 1.0

    Reference: Testkonzept Phase 3

    Args:
        results_df: Raw-result DataFrame including Quad-specific columns:
                    detected_time_scale, detected_freq_scale,
                    true_time_scale, true_freq_scale.

    Returns:
        DataFrame with columns:
            condition        (str):   Distortion condition name.
            time_scale_err   (float): |detected_time_scale - true_time_scale|
            freq_scale_err   (float): |detected_freq_scale - true_freq_scale|
        Returns an empty DataFrame with these columns if no matching rows
        exist.
    """
    required_cols = {
        "system", "group", "is_ood",
        "detected_time_scale", "detected_freq_scale",
        "true_time_scale", "true_freq_scale", "condition",
    }
    missing = required_cols - set(results_df.columns)
    if missing:
        logger.warning(
            "compute_scale_estimation_error: missing columns %s. "
            "Returning empty DataFrame.", missing,
        )
        return pd.DataFrame(columns=["condition", "time_scale_err", "freq_scale_err"])

    quad_A = results_df[
        (results_df["system"] == "quad") &
        (results_df["group"] == "A") &
        (~results_df["is_ood"])
    ].copy()

    if len(quad_A) == 0:
        logger.warning(
            "compute_scale_estimation_error: no Quad Group-A in-DB rows found."
        )
        return pd.DataFrame(columns=["condition", "time_scale_err", "freq_scale_err"])

    quad_A["time_scale_err"] = (
        quad_A["detected_time_scale"] - quad_A["true_time_scale"]
    ).abs()
    quad_A["freq_scale_err"] = (
        quad_A["detected_freq_scale"] - quad_A["true_freq_scale"]
    ).abs()

    return quad_A[["condition", "time_scale_err", "freq_scale_err"]].reset_index(drop=True)
