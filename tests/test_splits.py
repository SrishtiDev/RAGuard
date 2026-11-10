"""
tests/test_splits.py
--------------------
Unit tests for eval/splits.py.
Verifies split determinism, source_id grouping, leakage detection, test-split preservation,
and neutral dummy value handling.
"""

from pathlib import Path
import pytest

from eval.schema import Example, Span
from eval.splits import (
    assert_no_leakage,
    compute_id_hash,
    format_split_table,
    save_splits,
    split_ragtruth,
)


def _make_example(ex_id: str, source_id: str, task: str, original_split: str, hallucinated: bool = False) -> Example:
    """Helper to construct minimal synthetic Example with neutral dummy values."""
    spans = []
    resp = "This is a factual answer."
    if hallucinated:
        resp = "This is an ungrounded hallucination."
        spans = [
            Span(
                start=11,
                end=35,
                text="ungrounded hallucination",
                label_type="Evident Baseless Info",
            )
        ]
    return Example(
        id=f"test-ds:{ex_id}",
        source_dataset="test-ds",
        subset=task,
        group_id=source_id,
        query="Test query?",
        context="Test context passage.",
        response=resp,
        label_response_level=hallucinated,
        spans=spans,
        meta={"model": "test-model", "original_split": original_split},
    )


@pytest.fixture
def synthetic_split_examples():
    """
    Creates synthetic examples across 20 train sources (10 QA, 10 Summary)
    and 4 test sources (2 QA, 2 Summary), with 3 responses per source.
    """
    examples = []
    # Train sources: 1 to 20
    for s_idx in range(1, 21):
        task = "QA" if s_idx <= 10 else "Summary"
        sid = f"src_{s_idx}"
        for r_idx in range(1, 4):
            ex_id = f"{sid}_resp_{r_idx}"
            is_halluc = (s_idx + r_idx) % 2 == 0
            examples.append(_make_example(ex_id, sid, task, "train", is_halluc))

    # Test sources: 21 to 24
    for s_idx in range(21, 25):
        task = "QA" if s_idx <= 22 else "Summary"
        sid = f"src_{s_idx}"
        for r_idx in range(1, 4):
            ex_id = f"{sid}_resp_{r_idx}"
            is_halluc = (s_idx + r_idx) % 2 == 0
            examples.append(_make_example(ex_id, sid, task, "test", is_halluc))

    return examples


def test_splits_deterministic(synthetic_split_examples):
    splits1 = split_ragtruth(synthetic_split_examples, val_ratio=0.20, seed=42)
    splits2 = split_ragtruth(synthetic_split_examples, val_ratio=0.20, seed=42)

    for sp in ("train", "val", "test"):
        ids1 = [ex.id for ex in splits1[sp]]
        ids2 = [ex.id for ex in splits2[sp]]
        assert ids1 == ids2
        assert compute_id_hash(ids1) == compute_id_hash(ids2)


def test_splits_disjoint_and_grouped(synthetic_split_examples):
    splits = split_ragtruth(synthetic_split_examples, val_ratio=0.20, seed=42)

    # 1. No leakage check passes
    assert_no_leakage(splits)

    train_sources = set(ex.group_id for ex in splits["train"])
    val_sources = set(ex.group_id for ex in splits["val"])
    test_sources = set(ex.group_id for ex in splits["test"])

    # Disjointness
    assert len(train_sources.intersection(val_sources)) == 0
    assert len(train_sources.intersection(test_sources)) == 0
    assert len(val_sources.intersection(test_sources)) == 0

    # Grouping: every response of a source must be in the same split
    all_sources = train_sources | val_sources | test_sources
    for sid in all_sources:
        matching_splits = [sp for sp, exs in splits.items() if any(ex.group_id == sid for ex in exs)]
        assert len(matching_splits) == 1


def test_leakage_assertion_catches_overlap(synthetic_split_examples):
    splits = split_ragtruth(synthetic_split_examples, val_ratio=0.20, seed=42)
    # Intentionally corrupt splits by copying a train example into test
    corrupted = {
        "train": list(splits["train"]),
        "val": list(splits["val"]),
        "test": list(splits["test"]) + [splits["train"][0]],
    }
    with pytest.raises(AssertionError) as exc_info:
        assert_no_leakage(corrupted)
    assert "Data leakage detected" in str(exc_info.value)


def test_test_split_unchanged_and_val_subset_of_train(synthetic_split_examples):
    splits = split_ragtruth(synthetic_split_examples, val_ratio=0.20, seed=42)

    # Test split has exactly the original test items
    orig_test_ids = set(ex.id for ex in synthetic_split_examples if ex.meta["original_split"] == "test")
    split_test_ids = set(ex.id for ex in splits["test"])
    assert orig_test_ids == split_test_ids

    # Val items all originated from train
    orig_train_ids = set(ex.id for ex in synthetic_split_examples if ex.meta["original_split"] == "train")
    for ex in splits["val"]:
        assert ex.id in orig_train_ids


def test_save_splits_produces_identical_files(synthetic_split_examples, tmp_path):
    splits = split_ragtruth(synthetic_split_examples, val_ratio=0.20, seed=42)

    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    saved1 = save_splits(splits, output_dir=dir1, seed=42)
    saved2 = save_splits(splits, output_dir=dir2, seed=42)

    for sp in ("train", "val", "test"):
        content1 = saved1[sp].read_text(encoding="utf-8")
        content2 = saved2[sp].read_text(encoding="utf-8")
        assert content1 == content2


def test_format_split_table(synthetic_split_examples):
    splits = split_ragtruth(synthetic_split_examples, val_ratio=0.20, seed=42)
    table_str = format_split_table(splits)
    assert "| Split | Task Type | N Sources | N Responses | Hallucinated | Hallucination Rate |" in table_str
    assert "train" in table_str
    assert "val" in table_str
    assert "test" in table_str

