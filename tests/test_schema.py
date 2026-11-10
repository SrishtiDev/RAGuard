"""
tests/test_schema.py
--------------------
Unit tests for eval/schema.py.
Verifies Pydantic v2 Example and Span validation, span mismatch detection,
and neutral dummy value handling.
"""

import logging
import pytest

from eval.schema import Example, Span, SpanMismatch


def test_span_creation_and_defaults():
    span = Span(
        start=0,
        end=4,
        text="test",
        label_type="Evident Conflict",
    )
    assert span.start == 0
    assert span.end == 4
    assert span.text == "test"
    assert span.label_type == "Evident Conflict"
    assert span.implicit_true is False
    assert span.due_to_null is False


def test_example_valid_spans():
    text = "The quick brown fox jumps over the lazy dog."
    span = Span(
        start=4,
        end=9,
        text="quick",
        label_type="Subtle Baseless Info",
    )
    ex = Example(
        id="test-ds:1",
        source_dataset="test-ds",
        subset="test-task",
        group_id="group-1",
        query="What did the fox do?",
        context="Context passage for test.",
        context_chunks=["Context passage for test."],
        response=text,
        label_response_level=True,
        spans=[span],
        meta={"model": "test-model", "temperature": 0.0},
    )

    assert ex.id == "test-ds:1"
    assert ex.has_span_mismatches is False
    assert ex.find_span_mismatches() == []


def test_example_span_mismatch_detected(caplog):
    text = "The quick brown fox jumps over the lazy dog."
    # Mismatched span: claims slice [4:9] is "slow" instead of "quick"
    mismatched_span = Span(
        start=4,
        end=9,
        text="slow",
        label_type="Evident Conflict",
    )
    with caplog.at_level(logging.WARNING):
        ex = Example(
            id="test-ds:2",
            source_dataset="test-ds",
            subset="test-task",
            group_id="group-2",
            query=None,
            context="Reference context text.",
            response=text,
            label_response_level=True,
            spans=[mismatched_span],
            meta={"model": "test-model"},
        )

    # Record is not dropped, but mismatch is detected
    assert ex.has_span_mismatches is True
    mismatches = ex.find_span_mismatches()
    assert len(mismatches) == 1
    assert isinstance(mismatches[0], SpanMismatch)
    assert mismatches[0].span_index == 0
    assert mismatches[0].start == 4
    assert mismatches[0].end == 9
    assert mismatches[0].expected_text == "slow"
    assert mismatches[0].actual_text == "quick"

    # Verify a warning was logged
    assert "has 1 span mismatch(es)" in caplog.text


def test_example_multiple_spans_detection():
    text = "Alpha Beta Gamma Delta"
    span_ok = Span(start=0, end=5, text="Alpha", label_type="Evident Conflict")
    span_bad1 = Span(start=6, end=10, text="Wrong", label_type="Evident Conflict")  # actual: Beta
    span_bad2 = Span(start=11, end=16, text="Wrong", label_type="Evident Conflict")  # actual: Gamma

    ex = Example(
        id="test-ds:3",
        source_dataset="test-ds",
        subset="test-task",
        group_id="group-3",
        query=None,
        context="Context text",
        response=text,
        label_response_level=True,
        spans=[span_ok, span_bad1, span_bad2],
    )

    mismatches = ex.find_span_mismatches()
    assert len(mismatches) == 2
    assert mismatches[0].span_index == 1
    assert mismatches[0].actual_text == "Beta"
    assert mismatches[1].span_index == 2
    assert mismatches[1].actual_text == "Gamma"

