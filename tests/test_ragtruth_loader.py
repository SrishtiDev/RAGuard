"""
tests/test_ragtruth_loader.py
-----------------------------
Unit and integration tests for load_ragtruth in eval/datasets.py.
Uses neutral dummy values only (no real model names or API keys).
"""

import json
from pathlib import Path
import pytest

from eval.datasets import load_ragtruth
from eval.schema import Example


@pytest.fixture
def synthetic_ragtruth_dir(tmp_path):
    """
    Creates a minimal synthetic RAGTruth directory with 1 QA and 1 Summary source,
    and 3 responses (including one with an excluded quality flag).
    """
    source_info = [
        {
            "source_id": "1001",
            "task_type": "QA",
            "source": "MARCO",
            "source_info": {
                "question": "What is the capital of Wonderland?",
                "passages": "passage 1: The capital of Wonderland is Emerald City.",
            },
            "prompt": "Answer the question based on passages.",
        },
        {
            "source_id": "1002",
            "task_type": "Summary",
            "source": "CNN/DM",
            "source_info": "Wonderland celebrated its annual jubilee today with festivities.",
            "prompt": "Summarize the news.",
        },
    ]

    responses = [
        {
            "id": "2001",
            "source_id": "1001",
            "model": "test-model-1",
            "temperature": 0.0,
            "quality": "good",
            "split": "train",
            "response": "The capital of Wonderland is Emerald City.",
            "labels": [],
        },
        {
            "id": "2002",
            "source_id": "1001",
            "model": "test-model-2",
            "temperature": 0.0,
            "quality": "incorrect_refusal",  # should be excluded by default
            "split": "train",
            "response": "I cannot answer this question.",
            "labels": [],
        },
        {
            "id": "2003",
            "source_id": "1002",
            "model": "test-model-1",
            "temperature": 0.7,
            "quality": "good",
            "split": "test",
            "response": "Wonderland held a massive parade yesterday.",
            "labels": [
                {
                    "start": 11,
                    "end": 32,
                    "text": "held a massive parade",
                    "label_type": "Evident Baseless Info",
                    "implicit_true": False,
                    "due_to_null": False,
                }
            ],
        },
    ]

    with open(tmp_path / "source_info.jsonl", "w", encoding="utf-8") as f:
        for item in source_info:
            f.write(json.dumps(item) + "\n")

    with open(tmp_path / "response.jsonl", "w", encoding="utf-8") as f:
        for item in responses:
            f.write(json.dumps(item) + "\n")

    return tmp_path


def test_load_ragtruth_synthetic(synthetic_ragtruth_dir):
    examples = load_ragtruth(synthetic_ragtruth_dir)
    assert len(examples) == 2  # 2002 was excluded due to incorrect_refusal

    qa_ex = next(ex for ex in examples if ex.subset == "QA")
    assert qa_ex.id == "ragtruth:2001"
    assert qa_ex.group_id == "1001"
    assert qa_ex.query == "What is the capital of Wonderland?"
    assert qa_ex.context == "passage 1: The capital of Wonderland is Emerald City."
    assert qa_ex.context_chunks is None
    assert qa_ex.label_response_level is False
    assert len(qa_ex.spans) == 0
    assert qa_ex.meta["model"] == "test-model-1"

    sum_ex = next(ex for ex in examples if ex.subset == "Summary")
    assert sum_ex.id == "ragtruth:2003"
    assert sum_ex.group_id == "1002"
    assert sum_ex.query is None  # Must not parse query from prompt
    assert sum_ex.context == "Wonderland celebrated its annual jubilee today with festivities."
    assert sum_ex.label_response_level is True
    assert len(sum_ex.spans) == 1
    assert sum_ex.spans[0].text == "held a massive parade"
    assert sum_ex.has_span_mismatches is False


def test_load_ragtruth_passages_list_handling(tmp_path):
    """Verify context_chunks is populated if passages is natively a list."""
    source_info = [
        {
            "source_id": "3001",
            "task_type": "QA",
            "source": "MARCO",
            "source_info": {
                "question": "Test question?",
                "passages": ["Passage 1 content.", "Passage 2 content."],
            },
            "prompt": "Test prompt",
        }
    ]
    responses = [
        {
            "id": "4001",
            "source_id": "3001",
            "model": "test-model",
            "temperature": 0.0,
            "quality": "good",
            "split": "train",
            "response": "Test answer.",
            "labels": [],
        }
    ]
    with open(tmp_path / "source_info.jsonl", "w", encoding="utf-8") as f:
        for item in source_info:
            f.write(json.dumps(item) + "\n")
    with open(tmp_path / "response.jsonl", "w", encoding="utf-8") as f:
        for item in responses:
            f.write(json.dumps(item) + "\n")

    examples = load_ragtruth(tmp_path, task_types=("QA",))
    assert len(examples) == 1
    assert examples[0].context_chunks == ["Passage 1 content.", "Passage 2 content."]
    assert examples[0].context == "Passage 1 content.\n\nPassage 2 content."


def test_data2txt_raises_not_implemented(synthetic_ragtruth_dir):
    with pytest.raises(NotImplementedError) as exc_info:
        load_ragtruth(synthetic_ragtruth_dir, task_types=("Data2txt",))
    assert "Data2txt" in str(exc_info.value)


def test_invalid_task_type_raises(synthetic_ragtruth_dir):
    with pytest.raises(ValueError) as exc_info:
        load_ragtruth(synthetic_ragtruth_dir, task_types=("UnknownTask",))
    assert "Unsupported task_type" in str(exc_info.value)


def test_schema_structure_assertion(tmp_path):
    """Corrupt QA source_info to non-dict: should raise TypeError, not silently coerce."""
    source_info = [
        {
            "source_id": "5001",
            "task_type": "QA",
            "source": "MARCO",
            "source_info": "malformed string where dict expected",
            "prompt": "prompt",
        }
    ]
    responses = [
        {
            "id": "6001",
            "source_id": "5001",
            "model": "test-model",
            "temperature": 0.0,
            "quality": "good",
            "split": "train",
            "response": "response",
            "labels": [],
        }
    ]
    with open(tmp_path / "source_info.jsonl", "w", encoding="utf-8") as f:
        for item in source_info:
            f.write(json.dumps(item) + "\n")
    with open(tmp_path / "response.jsonl", "w", encoding="utf-8") as f:
        for item in responses:
            f.write(json.dumps(item) + "\n")

    with pytest.raises(TypeError) as exc_info:
        load_ragtruth(tmp_path, task_types=("QA",))
    assert "Expected dict for QA source_info" in str(exc_info.value)


def test_real_ragtruth_integration():
    """Optional integration test on actual cached files; skipped if missing."""
    real_path = Path("data/raw/ragtruth")
    if not (real_path / "source_info.jsonl").exists() or not (real_path / "response.jsonl").exists():
        pytest.skip("Local cached data/raw/ragtruth files not found.")

    examples = load_ragtruth(real_path, task_types=("QA", "Summary"))
    # Total QA (5934 - 167) + Summary (5658 - 3) = 5767 + 5655 = 11422
    assert len(examples) == 11422

    qa_count = sum(1 for ex in examples if ex.subset == "QA")
    sum_count = sum(1 for ex in examples if ex.subset == "Summary")
    assert qa_count == 5767
    assert sum_count == 5655

    for ex in examples[:10]:
        assert isinstance(ex, Example)
        assert ex.source_dataset == "ragtruth"
        assert ex.context
        assert ex.response
