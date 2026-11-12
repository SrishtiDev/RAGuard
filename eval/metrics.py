"""
eval/metrics.py
---------------
Evaluation metrics for hallucination detection with positive class = hallucinated (1).
Includes ranking metrics (ROC-AUC, PR-AUC), threshold metrics (confusion matrix, F1, precision, recall),
threshold tuning guard, grouped bootstrap confidence intervals, subset/model breakdowns,
and Section 6 additions: f1_all_positive baseline, precision_at_recall, f1_delta_over_all_positive.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


def compute_prevalence(labels: Sequence[Any]) -> float:
    """Calculate prevalence (proportion of positive/hallucinated instances)."""
    y = np.asarray(labels, dtype=int)
    if len(y) == 0:
        return 0.0
    return float(np.mean(y == 1))


def compute_ranking_metrics(
    scores: Sequence[float],
    labels: Sequence[Any],
) -> dict[str, Any]:
    """
    Compute ranking metrics: prevalence, ROC-AUC, and average precision (PR-AUC).
    Positive class is hallucinated (1).

    Returns
    -------
    dict[str, Any]
        Dictionary with prevalence, roc_auc, average_precision, and flags for undefined cases.
    """
    y_scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(labels, dtype=int)
    flags: dict[str, str] = {}

    if len(y_true) == 0:
        return {
            "prevalence": 0.0,
            "roc_auc": None,
            "average_precision": None,
            "flags": {"all": "empty_input"},
        }

    prevalence = float(np.mean(y_true == 1))

    # Check for single-class ground truth
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        reason = "single_class_labels"
        flags["roc_auc"] = reason
        flags["average_precision"] = reason
        return {
            "prevalence": prevalence,
            "roc_auc": None,
            "average_precision": None,
            "flags": flags,
        }

    # Check for constant scores
    if len(np.unique(y_scores)) == 1:
        # Constant predictions have uninformative AUC 0.5 and AP = prevalence
        return {
            "prevalence": prevalence,
            "roc_auc": 0.5,
            "average_precision": prevalence,
            "flags": {"scores": "constant_scores"},
        }

    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except Exception as e:
        logger.warning("Error computing roc_auc_score: %s", e)
        roc_auc = None
        flags["roc_auc"] = f"error: {e}"

    try:
        avg_precision = float(average_precision_score(y_true, y_scores))
    except Exception as e:
        logger.warning("Error computing average_precision_score: %s", e)
        avg_precision = None
        flags["average_precision"] = f"error: {e}"

    return {
        "prevalence": prevalence,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "flags": flags,
    }


def compute_threshold_metrics(
    scores: Sequence[float],
    labels: Sequence[Any],
    threshold: float,
) -> dict[str, Any]:
    """
    Compute classification metrics at a given threshold.
    Positive prediction if score >= threshold.

    Undefined cases return None with a flag (never a silent 0 or NaN).
    """
    y_scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(labels, dtype=int)
    y_pred = (y_scores >= threshold).astype(int)
    flags: dict[str, str] = {}

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    total = len(y_true)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    # Precision: undefined if no positive predictions
    if tp + fp == 0:
        precision = None
        flags["precision"] = "no_positive_predictions"
    else:
        precision = float(tp / (tp + fp))

    # Recall: undefined if no positive ground-truth labels
    if tp + fn == 0:
        recall = None
        flags["recall"] = "no_positive_labels"
    else:
        recall = float(tp / (tp + fn))

    # F1: undefined if precision or recall is undefined; 0.0 if precision + recall == 0
    if precision is None:
        f1 = None
        flags["f1"] = flags.get("precision", "undefined_precision")
    elif recall is None:
        f1 = None
        flags["f1"] = flags.get("recall", "undefined_recall")
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall / (precision + recall))

    return {
        "threshold": float(threshold),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flags": flags,
    }


def best_threshold_f1(
    scores: Sequence[float],
    labels: Sequence[Any],
    split_name: str,
) -> float:
    """
    Scan unique scores to find the threshold that maximizes F1 score.

    Tuning is strictly restricted to split_name == "val".
    Raises ValueError on any other split.
    """
    if split_name != "val":
        raise ValueError(
            f"Tuning threshold is only permitted on split 'val', got '{split_name}'. "
            "Never tune thresholds on the test split."
        )

    y_scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(labels, dtype=int)

    if len(y_scores) == 0:
        return 0.5

    unique_scores = np.sort(np.unique(y_scores))
    if len(unique_scores) == 1:
        return float(unique_scores[0])

    best_thresh = 0.5
    best_f1 = -1.0

    for candidate in unique_scores:
        res = compute_threshold_metrics(y_scores, y_true, threshold=candidate)
        f1 = res["f1"]
        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_thresh = candidate

    return float(best_thresh)


def grouped_bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], Optional[float]],
    scores: Sequence[float],
    labels: Sequence[Any],
    groups: Sequence[Any],
    n: int = 1000,
    seed: int = 42,
) -> tuple[Optional[float], Optional[float]]:
    """
    Compute a 95% confidence interval using grouped bootstrap resampling with replacement.
    Groups (group_id = source_id) are resampled to preserve cluster dependencies.

    Parameters
    ----------
    metric_fn : Callable
        Function taking (scores_sample, labels_sample) and returning float or None.
    scores : Sequence[float]
        Detector scores.
    labels : Sequence[Any]
        Binary labels (1 = positive/hallucinated).
    groups : Sequence[Any]
        Group identifiers (source_ids).
    n : int
        Number of bootstrap iterations (default: 1000).
    seed : int
        Random seed for determinism (default: 42).

    Returns
    -------
    tuple[Optional[float], Optional[float]]
        (ci_lower, ci_upper) 95% interval, or (None, None) if undefined.
    """
    y_scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(labels, dtype=int)
    y_groups = np.asarray(groups)

    if len(y_groups) == 0:
        return None, None

    # Sort unique groups first to ensure cross-platform deterministic order for a fixed seed
    unique_groups = np.array(sorted(list(set(y_groups))))
    group_indices = {g: np.where(y_groups == g)[0] for g in unique_groups}

    rng = np.random.default_rng(seed)
    bootstrap_vals: list[float] = []

    for _ in range(n):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sample_idx = np.concatenate([group_indices[g] for g in sampled_groups])

        sub_scores = y_scores[sample_idx]
        sub_labels = y_true[sample_idx]

        val = metric_fn(sub_scores, sub_labels)
        if val is not None and not np.isnan(val):
            bootstrap_vals.append(val)

    if len(bootstrap_vals) == 0:
        return None, None

    ci_lower, ci_upper = np.percentile(bootstrap_vals, [2.5, 97.5])
    return float(ci_lower), float(ci_upper)


def compute_all_metrics(
    scores: Sequence[float],
    labels: Sequence[Any],
    groups: Optional[Sequence[Any]] = None,
    threshold: Optional[float] = None,
    split_name: Optional[str] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Compute comprehensive metrics: ranking (prevalence, ROC-AUC, PR-AUC),
    threshold metrics at the given or tuned threshold, and 95% bootstrap CIs.
    """
    y_scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(labels, dtype=int)

    # Resolve threshold
    tuned = False
    if threshold is None:
        if split_name == "val":
            threshold = best_threshold_f1(y_scores, y_true, split_name="val")
            tuned = True
        else:
            threshold = 0.5

    ranking = compute_ranking_metrics(y_scores, y_true)
    thresh_metrics = compute_threshold_metrics(y_scores, y_true, threshold=threshold)

    # Grouped bootstrap CIs
    roc_ci: tuple[Optional[float], Optional[float]] = (None, None)
    pr_ci: tuple[Optional[float], Optional[float]] = (None, None)
    f1_ci: tuple[Optional[float], Optional[float]] = (None, None)

    if groups is not None and len(groups) > 0 and len(np.unique(y_true)) > 1:
        def _auc_fn(s: np.ndarray, y: np.ndarray) -> Optional[float]:
            if len(np.unique(y)) < 2:
                return None
            if len(np.unique(s)) == 1:
                return 0.5
            try:
                return float(roc_auc_score(y, s))
            except Exception:
                return None

        def _ap_fn(s: np.ndarray, y: np.ndarray) -> Optional[float]:
            if len(np.unique(y)) < 2:
                return None
            if len(np.unique(s)) == 1:
                return float(np.mean(y == 1))
            try:
                return float(average_precision_score(y, s))
            except Exception:
                return None

        def _f1_fn(s: np.ndarray, y: np.ndarray) -> Optional[float]:
            return compute_threshold_metrics(s, y, threshold=threshold)["f1"]

        roc_ci = grouped_bootstrap_ci(_auc_fn, y_scores, y_true, groups, n=n_bootstrap, seed=seed)
        pr_ci = grouped_bootstrap_ci(_ap_fn, y_scores, y_true, groups, n=n_bootstrap, seed=seed)
        f1_ci = grouped_bootstrap_ci(_f1_fn, y_scores, y_true, groups, n=n_bootstrap, seed=seed)

    # Section 6: f1_all_positive baseline and precision_at_recall
    prevalence = ranking["prevalence"]
    f1_floor = f1_all_positive(prevalence)
    f1_val = thresh_metrics["f1"]
    f1_delta = f1_delta_over_all_positive(f1_val, prevalence)
    prec_at_recalls = compute_precision_at_recalls(y_scores, y_true, targets=(0.8, 0.9))

    return {
        "prevalence": prevalence,
        "roc_auc": ranking["roc_auc"],
        "roc_auc_ci_95": roc_ci,
        "average_precision": ranking["average_precision"],
        "average_precision_ci_95": pr_ci,
        "threshold": threshold,
        "threshold_tuned_on_val": tuned,
        "confusion_matrix": thresh_metrics["confusion_matrix"],
        "accuracy": thresh_metrics["accuracy"],
        "precision": thresh_metrics["precision"],
        "recall": thresh_metrics["recall"],
        "f1": f1_val,
        "f1_ci_95": f1_ci,
        "f1_all_positive_floor": f1_floor,
        "f1_delta_over_all_positive": f1_delta,
        **prec_at_recalls,
        "flags": {**ranking.get("flags", {}), **thresh_metrics.get("flags", {})},
    }


def compute_breakdowns(
    predictions: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """
    Compute metric breakdowns by subset (task_type) and by model.
    """
    by_subset_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rec in predictions:
        subset = rec.get("subset", "unknown")
        model = rec.get("model", "unknown")
        by_subset_records[subset].append(rec)
        by_model_records[model].append(rec)

    def _eval_records(recs: list[dict[str, Any]]) -> dict[str, Any]:
        s = [r["score"] for r in recs]
        y = [1 if r["label"] else 0 for r in recs]
        rank = compute_ranking_metrics(s, y)
        t = compute_threshold_metrics(s, y, threshold=threshold)
        return {
            "n": len(recs),
            "prevalence": rank["prevalence"],
            "roc_auc": rank["roc_auc"],
            "average_precision": rank["average_precision"],
            "precision": t["precision"],
            "recall": t["recall"],
            "f1": t["f1"],
            "accuracy": t["accuracy"],
            "confusion_matrix": t["confusion_matrix"],
            "flags": {**rank.get("flags", {}), **t.get("flags", {})},
        }

    return {
        "by_subset": {k: _eval_records(v) for k, v in sorted(by_subset_records.items())},
        "by_model": {k: _eval_records(v) for k, v in sorted(by_model_records.items())},
    }


# ---------------------------------------------------------------------------
# Section 6 additions
# ---------------------------------------------------------------------------

def f1_all_positive(prevalence: float) -> Optional[float]:
    """
    F1 score of a hypothetical detector that labels everything as positive (hallucinated).

    Formula: 2p / (1 + p)  where p = prevalence.

    This is the floor that any useful detector must beat on F1.
    Returns None if prevalence is out of [0, 1].
    """
    if prevalence < 0.0 or prevalence > 1.0:
        return None
    if prevalence == 0.0:
        return 0.0
    return 2.0 * prevalence / (1.0 + prevalence)


def precision_at_recall(
    scores: Sequence[float],
    labels: Sequence[Any],
    target_recall: float,
) -> Optional[float]:
    """
    Find the highest precision achievable at or above *target_recall*.

    Scans thresholds (from low to high score, increasing threshold) and returns
    the precision at the first threshold where recall >= target_recall.

    Parameters
    ----------
    scores : Sequence[float]
        Detector scores.
    labels : Sequence[Any]
        Binary labels (1 = positive/hallucinated).
    target_recall : float
        Minimum recall required (e.g. 0.8 or 0.9).

    Returns
    -------
    Optional[float]
        Precision at the threshold giving >= target_recall, or None if unachievable
        (e.g. single-class labels, no thresholds meet the target).
    """
    y_scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(labels, dtype=int)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None

    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return None

    # Sort by ascending threshold (high score = positive prediction)
    # We want the highest precision where recall >= target_recall.
    # Try all candidate thresholds.
    unique_thresholds = np.sort(np.unique(y_scores))
    best_precision: Optional[float] = None

    for thresh in unique_thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        recall = tp / n_pos
        if recall >= target_recall:
            denom = tp + fp
            prec = float(tp / denom) if denom > 0 else None
            if prec is not None:
                if best_precision is None or prec > best_precision:
                    best_precision = prec

    return best_precision


def f1_delta_over_all_positive(f1: Optional[float], prevalence: float) -> Optional[float]:
    """
    Compute F1 delta relative to the all-positive baseline.

    Returns f1 - f1_all_positive(prevalence), or None if either is None.
    """
    floor = f1_all_positive(prevalence)
    if f1 is None or floor is None:
        return None
    return f1 - floor


def compute_precision_at_recalls(
    scores: Sequence[float],
    labels: Sequence[Any],
    targets: tuple[float, ...] = (0.8, 0.9),
) -> dict[str, Optional[float]]:
    """
    Compute precision_at_recall for each target in *targets*.

    Returns a dict keyed as "precision_at_recall_0.80" etc.
    """
    return {
        f"precision_at_recall_{t:.2f}": precision_at_recall(scores, labels, t)
        for t in targets
    }

