"""
tests/test_nli.py
-----------------
Unit tests for detectors/nli.py.
Uses an injected fake scorer — no model download required.
A real-model integration test runs only when NLI_MODEL is set AND RUN_MODEL_TESTS=1.
"""

from __future__ import annotations

import os
import math
import pytest

from eval.schema import DetectorInput
from detectors.nli import (
    NLIDetector,
    _resolve_label_indices,
    _window_text,
    AggregationMode,
)
from eval.cache import NLICache
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(response: str, context: str = "Some context.") -> DetectorInput:
    return DetectorInput(
        id="test-example-1",
        query="What happened?",
        context=context,
        response=response,
    )


def _fake_scorer(premises: list[str], hypotheses: list[str]) -> list[tuple[float, float, float, bool]]:
    """Always returns entailment=0.7, neutral=0.2, contradiction=0.1, truncated=False."""
    return [(0.7, 0.2, 0.1, False)] * len(premises)


def _make_detector(
    aggregation: AggregationMode = "max",
    scorer=_fake_scorer,
    cache: NLICache | None = None,
) -> NLIDetector:
    return NLIDetector(
        model_name="test-model",
        aggregation=aggregation,
        window_words=180,
        stride_words=90,
        batch_size=8,
        cache=cache,
        _scorer=scorer,
    )


# ---------------------------------------------------------------------------
# Label index resolution
# ---------------------------------------------------------------------------

def test_resolve_label_indices_standard_order():
    """Standard entailment/neutral/contradiction in order 0/1/2."""
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    ent, neu, con = _resolve_label_indices(id2label)
    assert ent == 0
    assert neu == 1
    assert con == 2


def test_resolve_label_indices_shuffled_order():
    """Labels in a non-standard order must still be found by name."""
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    ent, neu, con = _resolve_label_indices(id2label)
    assert ent == 1
    assert neu == 2
    assert con == 0


def test_resolve_label_indices_case_insensitive():
    """Label matching is case-insensitive."""
    id2label = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
    ent, neu, con = _resolve_label_indices(id2label)
    assert ent == 0
    assert neu == 1
    assert con == 2


def test_resolve_label_indices_missing_label_raises():
    """Missing a required label raises ValueError with a clear message."""
    id2label = {0: "entailment", 1: "not_a_valid_label"}
    with pytest.raises(ValueError, match="neutral"):
        _resolve_label_indices(id2label)


# ---------------------------------------------------------------------------
# Window splitting
# ---------------------------------------------------------------------------

def test_window_short_text():
    """Text shorter than window_words returns a single window."""
    text = "Short text with few words."
    windows = _window_text(text, window_words=100, stride_words=50)
    assert windows == [text]


def test_window_exact_boundary():
    """Text with exactly window_words words returns a single window."""
    words = ["word"] * 10
    text = " ".join(words)
    windows = _window_text(text, window_words=10, stride_words=5)
    assert len(windows) == 1
    assert windows[0] == text


def test_window_overlap():
    """Text longer than window_words generates overlapping windows."""
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)
    windows = _window_text(text, window_words=10, stride_words=5)
    assert len(windows) > 1
    # Last window must include the final words
    last_window_words = windows[-1].split()
    assert last_window_words[-1] == words[-1]


def test_window_empty_text():
    windows = _window_text("", window_words=10, stride_words=5)
    assert windows == [""]


# ---------------------------------------------------------------------------
# Three aggregations
# ---------------------------------------------------------------------------

def test_aggregation_max():
    """max aggregation: max(claim_risk) = max(1 - support)."""
    # Fake scorer: always entailment=0.7 -> support=0.7 -> claim_risk=0.3
    det = _make_detector(aggregation="max")
    inp = _make_input("First claim. Second claim.")
    result = det.predict(inp)
    # Both claims have risk 0.3 -> max = 0.3
    assert abs(result.score - 0.3) < 1e-6
    assert result.metadata["aggregation"] == "max"


def test_aggregation_mean():
    det = _make_detector(aggregation="mean")
    inp = _make_input("First claim. Second claim.")
    result = det.predict(inp)
    # mean(0.3, 0.3) = 0.3
    assert abs(result.score - 0.3) < 1e-6


def test_aggregation_noisy_or():
    det = _make_detector(aggregation="noisy_or")
    inp = _make_input("First claim. Second claim.")
    result = det.predict(inp)
    # noisy_or = 1 - prod(support) = 1 - 0.7 * 0.7 = 1 - 0.49 = 0.51
    expected = 1.0 - (0.7 * 0.7)
    assert abs(result.score - expected) < 1e-6


def test_aggregation_noisy_or_single_claim():
    det = _make_detector(aggregation="noisy_or")
    inp = _make_input("Single claim only.")
    result = det.predict(inp)
    # 1 - 0.7 = 0.3
    assert abs(result.score - 0.3) < 1e-6


# ---------------------------------------------------------------------------
# Zero claims
# ---------------------------------------------------------------------------

def test_zero_claims_returns_zero_score():
    det = _make_detector()
    inp = _make_input("")  # empty response
    result = det.predict(inp)
    assert result.score == 0.0
    assert result.metadata["zero_claims"] is True


# ---------------------------------------------------------------------------
# Metadata keys
# ---------------------------------------------------------------------------

def test_metadata_keys_present():
    det = _make_detector()
    inp = _make_input("One claim. Two claims.")
    result = det.predict(inp)
    md = result.metadata
    for key in [
        "aggregation", "zero_claims", "claim_risks", "claim_texts",
        "claim_starts", "claim_ends", "claim_support", "claim_contra",
        "all_aggregations", "n_pairs", "n_truncated",
    ]:
        assert key in md, f"Missing metadata key: {key}"

    # all_aggregations must include all three modes
    agg = md["all_aggregations"]
    assert "max" in agg and "mean" in agg and "noisy_or" in agg


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------

def test_cache_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = NLICache(Path(tmpdir) / "test.db")
        cache.put("test-model", "premise text", "hypothesis text", 0.7, 0.2, 0.1, False)
        result = cache.get("test-model", "premise text", "hypothesis text")
        assert result is not None
        p_ent, p_neu, p_con, trunc = result
        assert abs(p_ent - 0.7) < 1e-9
        assert abs(p_neu - 0.2) < 1e-9
        assert abs(p_con - 0.1) < 1e-9
        assert trunc is False
        cache.close()


def test_cache_miss_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = NLICache(Path(tmpdir) / "test.db")
        result = cache.get("test-model", "premise", "hypothesis")
        assert result is None
        cache.close()


def test_cache_hit_rate():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = NLICache(Path(tmpdir) / "test.db")
        cache.put("m", "p1", "h1", 0.5, 0.3, 0.2, False)
        cache.get("m", "p1", "h1")   # hit
        cache.get("m", "p2", "h2")   # miss
        assert abs(cache.hit_rate - 0.5) < 1e-9
        cache.close()


def test_detector_with_cache(tmp_path):
    """Detector uses cache for second identical call; scorer should only be called once."""
    call_count = {"n": 0}

    def counting_scorer(premises, hypotheses):
        call_count["n"] += len(premises)
        return [(0.8, 0.1, 0.1, False)] * len(premises)

    cache = NLICache(tmp_path / "test.db")
    det = _make_detector(scorer=counting_scorer, cache=cache)
    inp = _make_input("One sentence.", context="Same context.")

    det.predict(inp)
    first_count = call_count["n"]
    # Second call should be fully cached
    det.predict(inp)
    second_count = call_count["n"]
    assert second_count == first_count, "Cache did not prevent re-scoring"
    cache.close()


# ---------------------------------------------------------------------------
# DetectorInput isolation (leakage check)
# ---------------------------------------------------------------------------

def test_detector_input_has_no_labels():
    """DetectorInput must not expose label_response_level, spans, or meta."""
    fields = set(DetectorInput.model_fields.keys())
    assert "label_response_level" not in fields
    assert "spans" not in fields
    assert "meta" not in fields
    assert fields == {"id", "query", "context", "context_chunks", "response"}


from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Real model integration test (skipped unless NLI_MODEL + RUN_MODEL_TESTS=1)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (os.environ.get("NLI_MODEL", "").strip() and os.environ.get("RUN_MODEL_TESTS") == "1"),
    reason="Set NLI_MODEL and RUN_MODEL_TESTS=1 to run real model tests",
)
def test_real_model_basic():
    """Basic sanity: clearly entailed claim should score lower risk than clearly contradicted."""
    from detectors.nli import load_nli_detector
    det = load_nli_detector(aggregation="max")

    inp_easy = DetectorInput(
        id="real-easy",
        query="What color is the sky?",
        context="The sky is blue on a clear day.",
        response="The sky is blue.",
    )
    inp_hard = DetectorInput(
        id="real-hard",
        query="What color is the sky?",
        context="The sky is blue on a clear day.",
        response="The sky is bright green.",
    )

    r_easy = det.predict(inp_easy)
    r_hard = det.predict(inp_hard)
    assert r_easy.score < r_hard.score, (
        f"Expected entailed claim to have lower risk: easy={r_easy.score:.3f}, hard={r_hard.score:.3f}"
    )

