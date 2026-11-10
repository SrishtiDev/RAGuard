"""
tests/test_dummy_detectors.py
-----------------------------
Unit tests for dummy detectors and leakage-safe DetectorInput.
Uses neutral dummy values only.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from detectors.base import DetectionResult, Detector
from detectors.dummy import (
    AlwaysHallucinated,
    LengthBaseline,
    NeverHallucinated,
    RandomDetector,
    get_dummy_detector,
)
from eval.schema import DetectorInput, Example, Span


def test_detector_input_attributes():
    """Verify DetectorInput exposes strictly the intended fields, frozen, and no label leakage."""
    inp = DetectorInput(
        id="test-id:1",
        query="test query",
        context="test context",
        context_chunks=["test context"],
        response="test response",
    )
    # Check exact attribute set
    expected_fields = {"id", "query", "context", "context_chunks", "response"}
    assert set(DetectorInput.model_fields.keys()) == expected_fields

    # Verify frozen immutability
    with pytest.raises(Exception):
        inp.id = "modified"  # type: ignore

    # Verify Example.to_detector_input() does not leak labels, spans, or meta
    ex = Example(
        id="test-id:1",
        source_dataset="test-ds",
        subset="QA",
        group_id="group-1",
        query="test query",
        context="test context",
        response="test response",
        label_response_level=True,
        spans=[Span(start=0, end=4, text="test", label_type="Evident Conflict")],
        meta={"model": "test-model", "quality": "good"},
    )
    converted = ex.to_detector_input()
    assert isinstance(converted, DetectorInput)
    assert not hasattr(converted, "label_response_level")
    assert not hasattr(converted, "spans")
    assert not hasattr(converted, "meta")
    assert not hasattr(converted, "group_id")
    assert not hasattr(converted, "subset")


def test_always_and_never_detectors():
    inp = DetectorInput(
        id="test:1",
        query=None,
        context="context",
        response="short answer",
    )
    always = AlwaysHallucinated()
    never = NeverHallucinated()

    res_always = always.predict(inp)
    assert res_always.score == 1.0

    res_never = never.predict(inp)
    assert res_never.score == 0.0


def test_random_detector_determinism_and_order_independence():
    inp1 = DetectorInput(id="doc:1", context="ctx", response="resp1")
    inp2 = DetectorInput(id="doc:2", context="ctx", response="resp2")

    rand_a = RandomDetector(seed=42)
    rand_b = RandomDetector(seed=42)

    # Determinism across instances
    assert rand_a.predict(inp1).score == rand_b.predict(inp1).score

    # Order independence: evaluating in different order yields same score per id
    score1_first = rand_a.predict(inp1).score
    score2_first = rand_a.predict(inp2).score

    rand_c = RandomDetector(seed=42)
    score2_second = rand_c.predict(inp2).score
    score1_second = rand_c.predict(inp1).score

    assert score1_first == score1_second
    assert score2_first == score2_second


def test_random_detector_auc_near_half():
    """On a sufficiently large seeded synthetic set, RandomDetector AUC should be near 0.5."""
    detector = RandomDetector(seed=123)
    n = 2000
    labels = np.array([1 if i % 2 == 0 else 0 for i in range(n)])
    scores = np.array(
        [
            detector.predict(
                DetectorInput(id=f"synth:{i}", context="ctx", response="resp")
            ).score
            for i in range(n)
        ]
    )
    auc = roc_auc_score(labels, scores)
    assert 0.45 <= auc <= 0.55


def test_length_baseline():
    detector = LengthBaseline()
    # 0 words
    inp0 = DetectorInput(id="doc:0", context="ctx", response="")
    assert detector.predict(inp0).score == 0.0

    # 100 words -> 100 / (100 + 100) = 0.5
    inp100 = DetectorInput(id="doc:100", context="ctx", response=" ".join(["word"] * 100))
    res100 = detector.predict(inp100)
    assert abs(res100.score - 0.5) < 1e-6
    assert res100.metadata["word_count"] == 100

    # 300 words -> 300 / 400 = 0.75
    inp300 = DetectorInput(id="doc:300", context="ctx", response=" ".join(["word"] * 300))
    assert abs(detector.predict(inp300).score - 0.75) < 1e-6


def test_get_dummy_detector_factory():
    assert isinstance(get_dummy_detector("always"), AlwaysHallucinated)
    assert isinstance(get_dummy_detector("never"), NeverHallucinated)
    assert isinstance(get_dummy_detector("random", seed=99), RandomDetector)
    assert isinstance(get_dummy_detector("length"), LengthBaseline)

    with pytest.raises(ValueError):
        get_dummy_detector("nonexistent")
