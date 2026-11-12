"""
tests/test_splitter.py
----------------------
Unit tests for claims/splitter.py.
Verifies character offset correctness, empty-claim dropping,
and no-download (rule-based only) behavior.
"""

from __future__ import annotations

import pytest

from claims.splitter import Claim, split_claims


def test_basic_two_sentences():
    text = "The sky is blue. Water is wet."
    claims = split_claims(text)
    assert len(claims) == 2
    # Verify every claim's text matches the slice
    for c in claims:
        assert text[c.start : c.end] == c.text


def test_single_sentence():
    text = "Only one sentence here."
    claims = split_claims(text)
    assert len(claims) == 1
    assert claims[0].text == "Only one sentence here."
    assert text[claims[0].start : claims[0].end] == claims[0].text


def test_empty_string():
    assert split_claims("") == []


def test_whitespace_only():
    assert split_claims("   \n\t  ") == []


def test_offsets_are_exact():
    """Every claim's character slice must exactly reproduce claim.text."""
    text = "Alice went to the market. She bought apples. Then she went home."
    claims = split_claims(text)
    assert len(claims) >= 2
    for c in claims:
        assert text[c.start : c.end] == c.text, (
            f"Offset mismatch: text[{c.start}:{c.end}]={text[c.start:c.end]!r} "
            f"!= {c.text!r}"
        )


def test_leading_trailing_whitespace_stripped():
    """Claims should not start or end with whitespace."""
    text = "  First sentence.   Second sentence.  "
    claims = split_claims(text)
    for c in claims:
        assert c.text == c.text.strip(), f"Claim not stripped: {c.text!r}"


def test_claim_dataclass_offset_validation():
    """Claim rejects negative or inverted offsets."""
    with pytest.raises(ValueError):
        Claim(text="test", start=10, end=5)

    with pytest.raises(ValueError):
        Claim(text="test", start=-1, end=4)


def test_multiple_sentences_all_offsets():
    text = (
        "The model was released in 2020. "
        "It supports multiple languages. "
        "Performance benchmarks are impressive."
    )
    claims = split_claims(text)
    assert len(claims) == 3
    for c in claims:
        assert text[c.start : c.end] == c.text


def test_no_spacy_model_download(monkeypatch):
    """The pipeline must use only spacy.blank('en') + sentencizer, never downloading models."""
    import spacy as _spacy
    original_load = _spacy.load

    def _reject_load(name, *args, **kwargs):
        raise AssertionError(f"spacy.load('{name}') was called — model download is not allowed!")

    monkeypatch.setattr(_spacy, "load", _reject_load)
    # Re-create the splitter NLP pipeline through the public API
    from claims.splitter import _make_nlp
    nlp = _make_nlp()
    assert "sentencizer" in nlp.pipe_names

