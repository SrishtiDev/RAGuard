"""
eval/schema.py
--------------
Canonical Pydantic v2 schema for evaluation examples and fine-grained hallucination spans.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class DetectorInput(BaseModel):
    """
    Leakage-safe input container for detectors.
    Exposes strictly the textual information required for inference,
    completely stripped of labels, spans, or metadata.
    """
    model_config = ConfigDict(frozen=True)

    id: str
    query: Optional[str] = None
    context: str
    context_chunks: Optional[list[str]] = None
    response: str


class Span(BaseModel):
    """
    Span-level hallucination annotation.
    """
    start: int
    end: int
    text: str
    label_type: str
    implicit_true: bool = False
    due_to_null: bool = False


class SpanMismatch(BaseModel):
    """
    Record of a discrepancy between a span annotation and response[start:end].
    """
    span_index: int
    start: int
    end: int
    expected_text: str
    actual_text: str


class Example(BaseModel):
    """
    Canonical representation of a single benchmark instance.
    """
    id: str  # e.g. "ragtruth:<response id>"
    source_dataset: str  # e.g. "ragtruth"
    subset: str  # e.g. task_type ("QA", "Summary", etc.)
    group_id: str  # e.g. source_id; all instances sharing group_id stay in one split
    query: Optional[str] = None
    context: str  # full context text
    context_chunks: Optional[list[str]] = None  # only if dataset natively provides passages, else None
    response: str
    label_response_level: bool  # True = hallucinated (labels non-empty)
    spans: list[Span] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)  # model, temperature, quality, original_split

    def find_span_mismatches(self) -> list[SpanMismatch]:
        """
        Check whether response[span.start:span.end] == span.text for every span.
        Returns a list of SpanMismatch records.
        """
        mismatches: list[SpanMismatch] = []
        for i, span in enumerate(self.spans):
            actual = self.response[span.start : span.end]
            if actual != span.text:
                mismatches.append(
                    SpanMismatch(
                        span_index=i,
                        start=span.start,
                        end=span.end,
                        expected_text=span.text,
                        actual_text=actual,
                    )
                )
        return mismatches

    @property
    def has_span_mismatches(self) -> bool:
        """True if any span text differs from response[start:end]."""
        return len(self.find_span_mismatches()) > 0

    @model_validator(mode="after")
    def _check_span_text_matches(self) -> Example:
        """
        Check span offsets against response text.
        Logs violations without dropping records.
        """
        mismatches = self.find_span_mismatches()
        if mismatches:
            logger.warning(
                "Example %s (%s) has %d span mismatch(es): %s",
                self.id,
                self.source_dataset,
                len(mismatches),
                mismatches,
            )
        return self

    def to_detector_input(self) -> DetectorInput:
        """
        Convert this Example into a leakage-safe DetectorInput.
        Exposes strictly: id, query, context, context_chunks, response.
        """
        return DetectorInput(
            id=self.id,
            query=self.query,
            context=self.context,
            context_chunks=self.context_chunks,
            response=self.response,
        )

