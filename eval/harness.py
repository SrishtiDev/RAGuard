"""
eval/harness.py
---------------
Evaluation execution harness for hallucination detectors.
Manages evaluation runs, threshold resolution, metric computation, split verification,
and artifact logging to runs/<timestamp>_<detector>_<split>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from detectors.base import Detector
from detectors.dummy import get_dummy_detector
from eval.datasets import load_ragtruth
from eval.logger import (
    create_run_dir,
    get_environment_versions,
    get_git_info,
    load_run_threshold,
    save_run_artifacts,
)
from eval.metrics import best_threshold_f1, compute_all_metrics, compute_breakdowns
from eval.schema import Example
from eval.splits import compute_id_hash

logger = logging.getLogger(__name__)


def load_and_verify_split_ids(
    splits_dir: Union[str, Path],
    split_name: str,
) -> tuple[list[str], str]:
    """
    Load split file from data/processed/splits/ragtruth_{split_name}.json and verify its SHA-256.

    Returns
    -------
    tuple[list[str], str]
        (list of example IDs, verified SHA-256 hash)

    Raises
    ------
    FileNotFoundError
        If split file is missing.
    ValueError
        If computed SHA-256 does not match stored hash.
    """
    split_file = Path(splits_dir) / f"ragtruth_{split_name}.json"
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Split file not found at {split_file}. Run split generation first."
        )

    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = data.get("ids", [])
    expected_sha = data.get("sha256", "")
    computed_sha = compute_id_hash(ids)

    if computed_sha != expected_sha:
        raise ValueError(
            f"SHA-256 verification failed for split '{split_name}'. "
            f"Expected {expected_sha}, computed {computed_sha}."
        )

    return ids, computed_sha


def format_console_summary(summary: dict[str, Any]) -> str:
    """Format a human-readable console summary string."""
    m = summary["metrics"]
    cm = m.get("confusion_matrix", {})
    roc_ci = m.get("roc_auc_ci_95", (None, None))
    pr_ci = m.get("average_precision_ci_95", (None, None))
    f1_ci = m.get("f1_ci_95", (None, None))

    def _fmt(v: Optional[float]) -> str:
        return f"{v:.4f}" if v is not None else "N/A"

    def _fmt_ci(ci: tuple[Optional[float], Optional[float]]) -> str:
        if ci[0] is not None and ci[1] is not None:
            return f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        return "N/A"

    lines = [
        "=" * 70,
        f"  EVALUATION SUMMARY: {summary['detector']} on split '{summary['split']}'",
        "=" * 70,
        f"Examples Evaluated    : {summary['n_examples']}",
        f"Prevalence (Positive) : {_fmt(m.get('prevalence'))} ({m.get('prevalence', 0.0) * 100:.1f}%)",
        f"ROC-AUC               : {_fmt(m.get('roc_auc'))} (95% CI: {_fmt_ci(roc_ci)})",
        f"Average Precision (PR): {_fmt(m.get('average_precision'))} (95% CI: {_fmt_ci(pr_ci)})",
        "-" * 70,
        f"Decision Threshold    : {_fmt(summary.get('threshold'))}"
        + (" (Tuned on val)" if summary.get("threshold_tuned_on_val") else " (Fixed/External)"),
        f"Accuracy              : {_fmt(m.get('accuracy'))}",
        f"Precision             : {_fmt(m.get('precision'))}",
        f"Recall                : {_fmt(m.get('recall'))}",
        f"F1 Score              : {_fmt(m.get('f1'))} (95% CI: {_fmt_ci(f1_ci)})",
        f"Confusion Matrix      : TP={cm.get('tp')}, FP={cm.get('fp')}, FN={cm.get('fn')}, TN={cm.get('tn')}",
        "-" * 70,
        f"Latency per predict() : Mean={summary['latency']['mean_ms']:.3f} ms | p95={summary['latency']['p95_ms']:.3f} ms",
    ]

    # Breakdowns
    bd = summary.get("breakdowns", {})
    if "by_subset" in bd:
        lines.append("-" * 70)
        lines.append("Breakdown by Task Subset:")
        for subset, sub_m in bd["by_subset"].items():
            lines.append(
                f"  - {subset:8s}: N={sub_m['n']} | Prev={_fmt(sub_m['prevalence'])} | "
                f"ROC-AUC={_fmt(sub_m['roc_auc'])} | PR-AUC={_fmt(sub_m['average_precision'])} | "
                f"F1={_fmt(sub_m['f1'])} | Prec={_fmt(sub_m['precision'])} | Rec={_fmt(sub_m['recall'])}"
            )

    if "run_dir" in summary:
        lines.append("-" * 70)
        lines.append(f"Artifacts saved to: {summary['run_dir']}")
    lines.append("=" * 70)

    return "\n".join(lines)


def run(
    detector: Detector,
    examples: Sequence[Example],
    split_name: str,
    threshold: Optional[float] = None,
    split_sha256: Optional[str] = None,
    seed: int = 42,
    output_dir: Optional[Union[str, Path]] = "runs",
) -> dict[str, Any]:
    """
    Execute detector evaluation across examples and log artifacts.

    Rules
    -----
    - split "val": if threshold is None, tune it on val and record in summary.
    - split "test": threshold is REQUIRED (e.g. from prior val run) or raises ValueError.
    - Leakage safety: input to detector is strictly DetectorInput.
    """
    clean_split = split_name.lower().strip()

    # Guard: test split must never tune threshold
    if clean_split == "test" and threshold is None:
        raise ValueError(
            "A threshold is REQUIRED for split 'test' (e.g. via --threshold-from a prior val run). "
            "Threshold tuning is strictly forbidden on the test split."
        )

    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []

    # Run inference using strictly leakage-safe DetectorInput
    for ex in examples:
        det_input = ex.to_detector_input()
        t0 = time.perf_counter()
        res = detector.predict(det_input)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        latencies.append(lat_ms)
        predictions.append(
            {
                "id": ex.id,
                "group_id": ex.group_id,
                "subset": ex.subset,
                "model": ex.meta.get("model", "unknown"),
                "label": bool(ex.label_response_level),
                "score": float(res.score),
                "latency_ms": lat_ms,
            }
        )

    scores = [p["score"] for p in predictions]
    labels = [p["label"] for p in predictions]
    groups = [p["group_id"] for p in predictions]

    # Resolve threshold
    tuned_on_val = False
    if threshold is None and clean_split == "val":
        threshold = best_threshold_f1(scores, labels, split_name="val")
        tuned_on_val = True
    elif threshold is None:
        threshold = 0.5

    # Compute metrics
    metrics = compute_all_metrics(
        scores=scores,
        labels=labels,
        groups=groups,
        threshold=threshold,
        split_name=clean_split,
        seed=seed,
    )
    breakdowns = compute_breakdowns(predictions, threshold=threshold)

    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    summary: dict[str, Any] = {
        "detector": detector.name,
        "split": clean_split,
        "n_examples": len(predictions),
        "threshold": float(threshold),
        "threshold_tuned_on_val": tuned_on_val,
        "metrics": metrics,
        "latency": {
            "mean_ms": mean_latency,
            "p95_ms": p95_latency,
        },
        "breakdowns": breakdowns,
    }

    commit, is_dirty = get_git_info()
    config = {
        "detector": detector.name,
        "split": clean_split,
        "threshold": float(threshold),
        "seed": seed,
        "split_sha256": split_sha256 or "unspecified",
        "git_commit": commit,
        "git_dirty": is_dirty,
        "versions": get_environment_versions(),
    }

    if output_dir is not None:
        run_dir = create_run_dir(
            base_dir=output_dir,
            detector_name=detector.name,
            split_name=clean_split,
        )
        save_run_artifacts(run_dir, config, predictions, summary)
        summary["run_dir"] = str(run_dir)

    console_output = format_console_summary(summary)
    print(console_output)

    return summary


def main(args_list: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for python -m eval.harness."""
    parser = argparse.ArgumentParser(description="RAGuard Detector Evaluation Harness")
    parser.add_argument(
        "--detector",
        required=True,
        choices=["always", "never", "random", "length"],
        help="Detector to evaluate.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate (default: val).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed classification threshold.",
    )
    parser.add_argument(
        "--threshold-from",
        type=str,
        default=None,
        help="Path to previous run directory from which to load threshold.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/ragtruth",
        help="Directory containing RAGTruth raw data.",
    )
    parser.add_argument(
        "--splits-dir",
        default="data/processed/splits",
        help="Directory containing split ID files.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Output directory for runs (default: runs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for detector and bootstrap CI.",
    )

    args = parser.parse_args(args_list)

    # Resolve threshold
    threshold = args.threshold
    if args.threshold_from:
        threshold = load_run_threshold(args.threshold_from)
        logger.info("Loaded threshold %f from %s", threshold, args.threshold_from)

    # Instantiate detector
    detector = get_dummy_detector(args.detector, seed=args.seed)

    # Verify and load split IDs
    split_ids, split_sha = load_and_verify_split_ids(args.splits_dir, args.split)
    split_set = set(split_ids)

    # Load all examples and filter by split IDs
    all_examples = load_ragtruth(args.raw_dir, task_types=("QA", "Summary"))
    split_examples = [ex for ex in all_examples if ex.id in split_set]

    if len(split_examples) != len(split_ids):
        logger.warning(
            "Found %d examples matching %d split IDs.",
            len(split_examples),
            len(split_ids),
        )

    run(
        detector=detector,
        examples=split_examples,
        split_name=args.split,
        threshold=threshold,
        split_sha256=split_sha,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

