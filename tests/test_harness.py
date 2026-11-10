"""
tests/test_harness.py
---------------------
Unit and integration tests for eval/harness.py.
Verifies threshold guards, end-to-end execution, artifact generation,
and real-data val split evaluation.
Uses neutral dummy values only.
"""

from pathlib import Path
import pytest

from detectors.dummy import AlwaysHallucinated, LengthBaseline, NeverHallucinated
from eval.harness import run
from eval.schema import Example, Span


def _make_synth_example(idx: int, task: str = "QA", label: bool = False, original_split: str = "val") -> Example:
    spans = []
    resp = "Factual synthetic response text."
    if label:
        resp = "Hallucinated synthetic response text."
        spans = [Span(start=0, end=12, text="Hallucinated", label_type="Evident Conflict")]
    return Example(
        id=f"test-ds:{idx}",
        source_dataset="test-ds",
        subset=task,
        group_id=f"grp_{idx % 3}",
        query="Synthetic test query?",
        context="Synthetic context text.",
        response=resp,
        label_response_level=label,
        spans=spans,
        meta={"model": "test-model", "original_split": original_split},
    )


def test_harness_test_split_without_threshold_raises():
    detector = LengthBaseline()
    examples = [_make_synth_example(1, original_split="test")]
    with pytest.raises(ValueError) as exc_info:
        run(detector, examples, split_name="test", threshold=None)
    assert "threshold is REQUIRED for split 'test'" in str(exc_info.value)


def test_harness_val_split_tunes_threshold():
    detector = LengthBaseline()
    # Create examples where longer responses are hallucinated
    examples = [
        _make_synth_example(1, label=False),
        _make_synth_example(2, label=True),
    ]
    summary = run(detector, examples, split_name="val", threshold=None, output_dir=None)
    assert summary["threshold"] is not None
    assert summary["threshold_tuned_on_val"] is True
    assert "metrics" in summary


def test_harness_end_to_end_artifacts(tmp_path):
    detector = LengthBaseline()
    examples = [
        _make_synth_example(1, label=False),
        _make_synth_example(2, label=True),
    ]
    summary = run(
        detector,
        examples,
        split_name="val",
        threshold=0.5,
        output_dir=tmp_path / "runs",
    )
    run_dir = Path(summary["run_dir"])
    assert run_dir.is_dir()

    # All three files must exist
    config_file = run_dir / "config.json"
    pred_file = run_dir / "predictions.jsonl"
    summary_file = run_dir / "summary.json"

    assert config_file.is_file()
    assert pred_file.is_file()
    assert summary_file.is_file()

    # Verify predictions.jsonl does NOT contain context or response text
    with open(pred_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Synthetic context text." not in content
    assert "Factual synthetic response text." not in content


def test_harness_real_val_split_integration():
    """Integration test on real cached data; skipped if data or split file is missing."""
    raw_path = Path("data/raw/ragtruth")
    split_file = Path("data/processed/splits/ragtruth_val.json")

    if not raw_path.exists() or not split_file.exists():
        pytest.skip("Local cached RAGTruth or val split file not found.")

    from eval.datasets import load_ragtruth
    from eval.harness import load_and_verify_split_ids

    split_ids, split_sha = load_and_verify_split_ids("data/processed/splits", "val")
    all_examples = load_ragtruth(raw_path, task_types=("QA", "Summary"))
    val_set = set(split_ids)
    val_examples = [ex for ex in all_examples if ex.id in val_set]

    assert len(val_examples) == 1449

    detector = AlwaysHallucinated()
    summary = run(
        detector,
        val_examples,
        split_name="val",
        threshold=0.5,
        output_dir=None,
    )
    assert summary["n_examples"] == 1449
    # AlwaysHallucinated at 0.5 threshold predicts 1 for everything -> recall 1.0
    assert summary["metrics"]["recall"] == 1.0

