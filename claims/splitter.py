"""
claims/splitter.py
------------------
Rule-based sentence/claim splitter using spaCy's blank English model
with the sentencizer component (no model download required).

Usage
-----
    from claims.splitter import split_claims, Claim
    claims = split_claims("The sky is blue. Water is wet.")
    # [Claim(text='The sky is blue.', start=0, end=16),
    #  Claim(text='Water is wet.', start=17, end=30)]
"""

from __future__ import annotations

from dataclasses import dataclass

import spacy
from spacy.language import Language


@dataclass(frozen=True)
class Claim:
    """A single atomic claim with its character offsets in the original text."""

    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"Claim.text must be str, got {type(self.text)}")
        if self.start < 0 or self.end < self.start:
            raise ValueError(
                f"Invalid offsets: start={self.start}, end={self.end}"
            )


def _make_nlp() -> Language:
    """Create a spaCy blank English pipeline with the rule-based sentencizer."""
    nlp = spacy.blank("en")
    # sentencizer is fully rule-based: no model weights, no download
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


# Module-level singleton — avoids re-creating the pipeline on every call
_NLP: Language | None = None


def _get_nlp() -> Language:
    global _NLP
    if _NLP is None:
        _NLP = _make_nlp()
    return _NLP


def split_claims(text: str) -> list[Claim]:
    """
    Split *text* into a list of Claim objects with verified character offsets.

    Rules
    -----
    - Uses spaCy's blank("en") + sentencizer (no model download).
    - Strips leading/trailing whitespace from each sentence boundary.
    - Skips empty claims after stripping.
    - Verifies that text[claim.start:claim.end] == claim.text for every claim.

    Parameters
    ----------
    text : str
        Input text to split into claims.

    Returns
    -------
    list[Claim]
        Ordered list of non-empty claims with verified character offsets.

    Raises
    ------
    AssertionError
        If any offset verification fails (indicates a bug in offset computation).
    """
    if not text or not text.strip():
        return []

    nlp = _get_nlp()
    doc = nlp(text)
    claims: list[Claim] = []

    for sent in doc.sents:
        # sent.start_char / sent.end_char are character offsets in the original text
        raw_text = sent.text

        # Strip leading whitespace and adjust start
        stripped_left = raw_text.lstrip()
        if not stripped_left:
            continue
        leading_spaces = len(raw_text) - len(stripped_left)
        char_start = sent.start_char + leading_spaces

        # Strip trailing whitespace and adjust end
        stripped_full = stripped_left.rstrip()
        if not stripped_full:
            continue
        char_end = char_start + len(stripped_full)

        # Verify offset correctness — must hold by construction
        actual_slice = text[char_start:char_end]
        assert actual_slice == stripped_full, (
            f"Offset mismatch: text[{char_start}:{char_end}]={actual_slice!r} "
            f"!= expected {stripped_full!r}"
        )

        claims.append(Claim(text=stripped_full, start=char_start, end=char_end))

    return claims

