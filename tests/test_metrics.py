"""
tests/test_metrics.py
---------------------
Unit tests for eval/metrics.py.
Verifies ranking metrics, threshold metrics, undefined-case handling,
threshold tuning guards, grouped bootstrap determinism, and breakdowns.
Uses neutral dummy values only.
"""

import numpy as np
import pytest

from eval.metrics import (
    best_threshold_f1,
    compute_all_metrics,
    compute_breakdowns,
    compute_ranking_metrics,
    compute_threshold_metrics,
    grouped_bootstrap_ci,
)


def test_ranking_metrics_perfect_separation():
    # Negative examples score 0.1, 0.2; Positive examples score 0.8, 0.9
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    res = compute_ranking_metrics(scores, labels)
    assert res["prevalence"] == 0.5
    assert res["roc_auc"] == 1.0
    assert res["average_precision"] == 1.0
    assert len(res["flags"]) == 0


def test_ranking_metrics_inverted():
    # Negative examples score high, Positive score low -> AUC = 0.0
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [0, 0, 1, 1]
    res = compute_ranking_metrics(scores, labels)
    assert res["prevalence"] == 0.5
    assert res["roc_auc"] == 0.0
    assert res["average_precision"] < 0.5


def test_ranking_metrics_constant_scores():
    # All scores are identical -> AUC = 0.5, AP = prevalence
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 0, 1, 1]
    res = compute_ranking_metrics(scores, labels)
    assert res["prevalence"] == 0.5
    assert res["roc_auc"] == 0.5
    assert res["average_precision"] == 0.5
    assert res["flags"]["scores"] == "constant_scores"


def test_ranking_metrics_single_class_labels():
    # All positive
    res_pos = compute_ranking_metrics([0.1, 0.5, 0.9], [1, 1, 1])
    assert res_pos["prevalence"] == 1.0
    assert res_pos["roc_auc"] is None
    assert res_pos["average_precision"] is None
    assert res_pos["flags"]["roc_auc"] == "single_class_labels"

    # All negative
    res_neg = compute_ranking_metrics([0.1, 0.5, 0.9], [0, 0, 0])
    assert res_neg["prevalence"] == 0.0
    assert res_neg["roc_auc"] is None
    assert res_neg["average_precision"] is None
    assert res_neg["flags"]["roc_auc"] == "single_class_labels"


def test_threshold_metrics_hand_computed():
    scores = [0.2, 0.4, 0.6, 0.8]
    labels = [0, 1, 0, 1]
    # threshold = 0.5 -> pred = [0, 0, 1, 1]
    # y_true = [0, 1, 0, 1]
    # tp: idx 3 (pred=1, true=1) -> 1
    # fp: idx 2 (pred=1, true=0) -> 1
    # fn: idx 1 (pred=0, true=1) -> 1
    # tn: idx 0 (pred=0, true=0) -> 1
    res = compute_threshold_metrics(scores, labels, threshold=0.5)
    cm = res["confusion_matrix"]
    assert cm == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert res["accuracy"] == 0.5
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["f1"] == 0.5


def test_threshold_metrics_undefined_cases():
    # No positive predictions (threshold = 0.9 on max score 0.5)
    scores = [0.1, 0.2, 0.3, 0.4]
    labels = [0, 1, 0, 1]
    res_no_pos = compute_threshold_metrics(scores, labels, threshold=0.9)
    assert res_no_pos["confusion_matrix"]["tp"] == 0
    assert res_no_pos["confusion_matrix"]["fp"] == 0
    assert res_no_pos["precision"] is None
    assert res_no_pos["f1"] is None
    assert res_no_pos["flags"]["precision"] == "no_positive_predictions"

    # No positive labels in dataset
    scores_neg = [0.1, 0.8]
    labels_neg = [0, 0]
    res_no_labels = compute_threshold_metrics(scores_neg, labels_neg, threshold=0.5)
    assert res_no_labels["recall"] is None
    assert res_no_labels["f1"] is None
    assert res_no_labels["flags"]["recall"] == "no_positive_labels"


def test_best_threshold_f1_guard_and_tuning():
    scores = [0.1, 0.3, 0.6, 0.9]
    labels = [0, 0, 1, 1]

    # Guard: must raise on test or train
    with pytest.raises(ValueError) as exc_info:
        best_threshold_f1(scores, labels, split_name="test")
    assert "Tuning threshold is only permitted on split 'val'" in str(exc_info.value)

    with pytest.raises(ValueError):
        best_threshold_f1(scores, labels, split_name="train")

    # Allowed on val
    thresh = best_threshold_f1(scores, labels, split_name="val")
    # At threshold 0.6: tp=2, fp=0, fn=0, tn=2 -> F1 = 1.0
    assert thresh in (0.3, 0.6)
    m = compute_threshold_metrics(scores, labels, threshold=thresh)
    assert m["f1"] == 1.0


def test_grouped_bootstrap_ci_deterministic():
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 10
    labels = [0, 0, 0, 1, 1, 1] * 10
    groups = [f"src_{i % 5}" for i in range(len(scores))]

    def metric_fn(s, y):
        if len(np.unique(y)) < 2:
            return None
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s))

    ci1 = grouped_bootstrap_ci(metric_fn, scores, labels, groups, n=100, seed=42)
    ci2 = grouped_bootstrap_ci(metric_fn, scores, labels, groups, n=100, seed=42)
    assert ci1 == ci2
    assert ci1[0] is not None and ci1[1] is not None
    assert ci1[0] <= ci1[1]


def test_compute_breakdowns():
    predictions = [
        {"id": "1", "score": 0.8, "label": True, "subset": "QA", "model": "test-model-1"},
        {"id": "2", "score": 0.2, "label": False, "subset": "QA", "model": "test-model-1"},
        {"id": "3", "score": 0.9, "label": True, "subset": "Summary", "model": "test-model-2"},
        {"id": "4", "score": 0.1, "label": False, "subset": "Summary", "model": "test-model-2"},
    ]
    bd = compute_breakdowns(predictions, threshold=0.5)
    assert "by_subset" in bd
    assert "by_model" in bd
    assert "QA" in bd["by_subset"]
    assert "Summary" in bd["by_subset"]
    assert "test-model-1" in bd["by_model"]
    assert "test-model-2" in bd["by_model"]
    assert bd["by_subset"]["QA"]["accuracy"] == 1.0
    assert bd["by_model"]["test-model-1"]["accuracy"] == 1.0



# ---------------------------------------------------------------------------
# Section 6 additions: f1_all_positive, precision_at_recall, f1_delta
# ---------------------------------------------------------------------------

import pytest
from eval.metrics import (
    f1_all_positive,
    precision_at_recall,
    f1_delta_over_all_positive,
    compute_precision_at_recalls,
)


def test_f1_all_positive_known_values():
    """f1_all_positive(p) = 2p / (1 + p)."""
    # prevalence = 0.5 -> F1 = 2*0.5 / 1.5 = 2/3
    result = f1_all_positive(0.5)
    assert abs(result - 2 / 3) < 1e-9

    # prevalence = 1.0 -> F1 = 2 / 2 = 1.0
    assert abs(f1_all_positive(1.0) - 1.0) < 1e-9

    # prevalence = 0.0 -> F1 = 0.0
    assert f1_all_positive(0.0) == 0.0

    # prevalence = 0.3140 (RAGTruth val)
    p = 0.3140
    expected = 2 * p / (1 + p)
    assert abs(f1_all_positive(p) - expected) < 1e-9


def test_f1_all_positive_out_of_range():
    assert f1_all_positive(-0.1) is None
    assert f1_all_positive(1.1) is None


def test_precision_at_recall_perfect_scores():
    """Perfect scores -> precision at any recall target is 1.0."""
    scores = [1.0, 1.0, 0.0, 0.0]
    labels = [1,   1,   0,   0]
    result = precision_at_recall(scores, labels, target_recall=1.0)
    assert result is not None
    assert abs(result - 1.0) < 1e-9


def test_precision_at_recall_target_80():
    """Hand-computed: threshold that gives recall >= 0.8."""
    # 5 positives, 5 negatives; sorted by score descending
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    labels = [1,   1,   1,   1,   1,   0,   0,   0,   0,   0  ]
    # At threshold=0.1: recall = 5/5 = 1.0, precision = 5/10 = 0.5
    # At threshold=0.5: recall = 5/5 = 1.0, precision = 5/5 = 1.0
    # At threshold=0.6: recall = 4/5 = 0.8, precision = 4/4 = 1.0
    # Best precision at recall >= 0.8 = 1.0
    result = precision_at_recall(scores, labels, target_recall=0.8)
    assert result is not None
    assert abs(result - 1.0) < 1e-9


def test_precision_at_recall_target_90_unachievable():
    """If only 80% recall is reachable before precision collapses, target 0.9 may still work."""
    scores = [1.0, 0.0, 0.0, 0.0, 0.0]
    labels = [1,   1,   1,   0,   0  ]
    # recall at score >= 0.0: recall=3/3=1.0, precision=3/5=0.6
    # We achieve recall=1.0, so target 0.9 is achievable
    result = precision_at_recall(scores, labels, target_recall=0.9)
    assert result is not None


def test_precision_at_recall_single_class_returns_none():
    scores = [0.9, 0.5, 0.1]
    labels = [0, 0, 0]
    result = precision_at_recall(scores, labels, target_recall=0.8)
    assert result is None


def test_f1_delta_over_all_positive():
    """f1_delta = f1 - f1_all_positive(prevalence)."""
    prevalence = 0.5
    floor = f1_all_positive(prevalence)  # 2/3 ~ 0.6667
    # Suppose model F1 = 0.75
    delta = f1_delta_over_all_positive(0.75, prevalence)
    assert delta is not None
    assert abs(delta - (0.75 - floor)) < 1e-9


def test_f1_delta_none_when_f1_is_none():
    assert f1_delta_over_all_positive(None, 0.5) is None


def test_compute_precision_at_recalls_keys():
    scores = [0.9, 0.1]
    labels = [1, 0]
    result = compute_precision_at_recalls(scores, labels, targets=(0.8, 0.9))
    assert "precision_at_recall_0.80" in result
    assert "precision_at_recall_0.90" in result


def test_compute_all_metrics_includes_new_fields():
    """compute_all_metrics must include f1_all_positive_floor, f1_delta, and precision_at_recall keys."""
    from eval.metrics import compute_all_metrics
    scores = [0.9, 0.8, 0.1, 0.1]
    labels = [1,   1,   0,   0  ]
    result = compute_all_metrics(scores, labels, split_name="val")
    assert "f1_all_positive_floor" in result
    assert "f1_delta_over_all_positive" in result
    assert "precision_at_recall_0.80" in result
    assert "precision_at_recall_0.90" in result
    # Sanity: floor should be less than 1 for prevalence 0.5
    assert result["f1_all_positive_floor"] is not None
    assert result["f1_all_positive_floor"] < 1.0
