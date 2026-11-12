"""
detectors/nli.py
----------------
NLI-based hallucination detector.

For each response:
  1. Split it into atomic claims using the rule-based sentencizer.
  2. Split the context into overlapping word windows.
  3. For each (claim, window) pair, run NLI to get
     [P(entailment), P(neutral), P(contradiction)].
  4. Per claim, aggregate over windows:
       support = max P(entailment)
       contra  = max P(contradiction)
       claim_risk = 1 - support
  5. Aggregate claim risks into a single response score.

Aggregation modes (selectable):
  "max"      : max(claim_risk)
  "mean"     : mean(claim_risk)
  "noisy_or" : 1 - prod(1 - claim_risk)   i.e. 1 - prod(support)

Design constraints:
  - Implements the Detector interface from detectors/base.py.
  - Scorer function is injectable so tests never download a model.
  - Labels are derived from config.id2label (case-insensitive); raises if any missing.
  - fp16 inference on GPU when available; float32 on CPU.
  - Results are cached in SQLite via eval/cache.py.
  - No labels, spans, or meta reach the detector (DetectorInput only).
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
import torch.nn.functional as F

from claims.splitter import Claim, split_claims
from detectors.base import DetectionResult, Detector
from eval.cache import NLICache, open_cache
from eval.schema import DetectorInput

logger = logging.getLogger(__name__)

AggregationMode = Literal["max", "mean", "noisy_or"]


# ---------------------------------------------------------------------------
# Label index resolution
# ---------------------------------------------------------------------------

def _resolve_label_indices(id2label: dict) -> tuple[int, int, int]:
    """
    Derive (entailment_idx, neutral_idx, contradiction_idx) from id2label.
    Matching is case-insensitive. Raises ValueError if any label is missing.
    """
    lower_map: dict[str, int] = {v.lower(): int(k) for k, v in id2label.items()}

    def _find(name: str) -> int:
        # Try exact match first, then prefix match
        for key, idx in lower_map.items():
            if key == name or key.startswith(name):
                return idx
        raise ValueError(
            f"Could not find '{name}' label in id2label={id2label}. "
            "Available labels: " + str(list(id2label.values()))
        )

    ent_idx = _find("entailment")
    neu_idx = _find("neutral")
    con_idx = _find("contradiction")
    return ent_idx, neu_idx, con_idx


# ---------------------------------------------------------------------------
# Context windowing
# ---------------------------------------------------------------------------

def _window_text(
    text: str,
    window_words: int = 180,
    stride_words: int = 90,
) -> list[str]:
    """
    Split *text* into overlapping word windows.

    Parameters
    ----------
    text : str
        Full context string.
    window_words : int
        Maximum number of whitespace-separated words per window.
    stride_words : int
        Step size between window starts.

    Returns
    -------
    list[str]
        List of window strings. Always at least one window (the full text if short).
    """
    words = text.split()
    if not words:
        return [""]
    if len(words) <= window_words:
        return [text]

    windows: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + window_words, len(words))
        windows.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride_words
    return windows


# ---------------------------------------------------------------------------
# Scorer type alias
# ---------------------------------------------------------------------------

# A scorer takes (premises: list[str], hypotheses: list[str]) and returns
# a list of (p_ent, p_neu, p_con, truncated) tuples.
ScorerFn = Callable[
    [list[str], list[str]],
    list[tuple[float, float, float, bool]],
]


# ---------------------------------------------------------------------------
# NLIDetector
# ---------------------------------------------------------------------------

class NLIDetector(Detector):
    """
    Hallucination detector based on Natural Language Inference.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (read from NLI_MODEL env var by default).
    aggregation : AggregationMode
        How to combine per-claim risks into a response score.
    window_words : int
        Max words per context window.
    stride_words : int
        Words to advance between context windows.
    batch_size : int
        NLI inference batch size (premise-hypothesis pairs per batch).
    cache : Optional[NLICache]
        Pre-opened cache. If None, one is opened using default path.
    _scorer : Optional[ScorerFn]
        Injectable scorer for testing (bypasses model loading).
    """

    def __init__(
        self,
        model_name: str,
        aggregation: AggregationMode = "max",
        window_words: int = 180,
        stride_words: int = 90,
        batch_size: int = 32,
        cache: Optional[NLICache] = None,
        _scorer: Optional[ScorerFn] = None,
    ) -> None:
        self._model_name = model_name
        self._aggregation = aggregation
        self._window_words = window_words
        self._stride_words = stride_words
        self._batch_size = batch_size
        self._cache = cache

        # Statistics
        self._total_pairs = 0
        self._truncated_pairs = 0

        if _scorer is not None:
            # Injected scorer — no model loading (for tests)
            self._scorer: ScorerFn = _scorer
            self._device = "cpu"
            self._ent_idx = 0
            self._neu_idx = 1
            self._con_idx = 2
        else:
            self._device, self._ent_idx, self._neu_idx, self._con_idx, self._scorer = (
                self._load_model(model_name, batch_size)
            )

        if self._cache is None:
            self._cache = open_cache(model_name)

    @staticmethod
    def _load_model(
        model_name: str, batch_size: int
    ) -> tuple[str, int, int, int, ScorerFn]:
        """Load model, tokenizer, resolve label indices, build scorer closure."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading NLI model '%s' on device '%s'", model_name, device)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model = model.to(device)
        if device == "cuda":
            model = model.half()  # fp16 on GPU
        model.eval()

        id2label: dict = model.config.id2label
        logger.info("Model id2label: %s", id2label)
        ent_idx, neu_idx, con_idx = _resolve_label_indices(id2label)
        logger.info(
            "Label indices — entailment: %d, neutral: %d, contradiction: %d",
            ent_idx, neu_idx, con_idx,
        )

        def _scorer(premises: list[str], hypotheses: list[str]) -> list[tuple[float, float, float, bool]]:
            results: list[tuple[float, float, float, bool]] = []
            for i in range(0, len(premises), batch_size):
                batch_p = premises[i : i + batch_size]
                batch_h = hypotheses[i : i + batch_size]
                enc = tokenizer(
                    batch_p,
                    batch_h,
                    return_tensors="pt",
                    truncation="only_first",   # truncate premise (context window) only
                    padding=True,
                    max_length=tokenizer.model_max_length,
                )
                # Detect truncation: input_ids contains [SEP] token at model_max_length - 1
                truncated_flags = [
                    int(ids[-1]) != tokenizer.sep_token_id
                    if tokenizer.sep_token_id is not None
                    else False
                    for ids in enc["input_ids"]
                ]
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = model(**enc).logits  # (B, num_labels)
                probs = F.softmax(logits.float(), dim=-1).cpu()
                for j in range(len(batch_p)):
                    p_ent = float(probs[j, ent_idx])
                    p_neu = float(probs[j, neu_idx])
                    p_con = float(probs[j, con_idx])
                    results.append((p_ent, p_neu, p_con, bool(truncated_flags[j])))
            return results

        return device, ent_idx, neu_idx, con_idx, _scorer

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "nli"

    @property
    def aggregation(self) -> str:
        return self._aggregation

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def truncation_rate(self) -> Optional[float]:
        if self._total_pairs == 0:
            return None
        return self._truncated_pairs / self._total_pairs

    # ------------------------------------------------------------------
    def predict(self, inp: DetectorInput) -> DetectionResult:
        """
        Score a single response for hallucination risk.

        Returns
        -------
        DetectionResult
            score   : hallucination risk in [0, 1]
            metadata: {
                "aggregation": str,
                "claim_risks": list[float],       # per-claim 1 - support
                "claim_texts": list[str],          # text of each claim
                "claim_starts": list[int],         # char start in response
                "claim_ends": list[int],           # char end in response
                "claim_support": list[float],      # max P(entailment) over windows
                "claim_contra": list[float],       # max P(contradiction) over windows
                "all_aggregations": dict,          # max/mean/noisy_or scores
                "zero_claims": bool,               # True if no claims found
                "n_pairs": int,                    # (claim, window) pairs evaluated
                "n_truncated": int,                # pairs where premise was truncated
            }
        """
        claims = split_claims(inp.response)
        if not claims:
            return DetectionResult(
                score=0.0,
                metadata={
                    "aggregation": self._aggregation,
                    "zero_claims": True,
                    "claim_risks": [],
                    "claim_texts": [],
                    "claim_starts": [],
                    "claim_ends": [],
                    "claim_support": [],
                    "claim_contra": [],
                    "all_aggregations": {"max": 0.0, "mean": 0.0, "noisy_or": 0.0},
                    "n_pairs": 0,
                    "n_truncated": 0,
                },
            )

        # Build context windows from full context
        windows = _window_text(inp.context, self._window_words, self._stride_words)

        # Collect per-claim aggregated probabilities
        claim_risks: list[float] = []
        claim_support: list[float] = []
        claim_contra: list[float] = []
        example_n_pairs = 0
        example_n_truncated = 0

        for claim in claims:
            # Gather (premise=window, hypothesis=claim) pairs
            premises: list[str] = []
            hypotheses: list[str] = []
            cached_results: list[Optional[tuple[float, float, float, bool]]] = []

            for window in windows:
                cached = self._cache.get(self._model_name, window, claim.text)
                cached_results.append(cached)
                if cached is None:
                    premises.append(window)
                    hypotheses.append(claim.text)

            # Run model for cache-miss pairs
            if premises:
                new_results = self._scorer(premises, hypotheses)
                miss_idx = 0
                for cached in cached_results:
                    if cached is None:
                        r = new_results[miss_idx]
                        miss_idx += 1
                        # Store in cache (no raw text stored — only hash key)
                        # Find which window this was
                        win_text = premises[miss_idx - 1]
                        self._cache.put(
                            self._model_name, win_text, claim.text,
                            r[0], r[1], r[2], r[3]
                        )

            # Merge cached + new results
            all_results: list[tuple[float, float, float, bool]] = []
            miss_idx = 0
            for cached in cached_results:
                if cached is not None:
                    all_results.append(cached)
                else:
                    all_results.append(new_results[miss_idx])
                    miss_idx += 1

            # Aggregate over windows
            max_ent = max(r[0] for r in all_results)
            max_con = max(r[2] for r in all_results)
            n_trunc = sum(1 for r in all_results if r[3])

            claim_support.append(max_ent)
            claim_contra.append(max_con)
            claim_risks.append(1.0 - max_ent)

            example_n_pairs += len(all_results)
            example_n_truncated += n_trunc

        # Update running statistics
        self._total_pairs += example_n_pairs
        self._truncated_pairs += example_n_truncated

        # Compute all three aggregations
        risks = claim_risks
        agg_max = max(risks)
        agg_mean = sum(risks) / len(risks)
        supports = claim_support
        agg_noisy_or = 1.0 - math.prod(s for s in supports)

        agg_scores = {"max": agg_max, "mean": agg_mean, "noisy_or": agg_noisy_or}
        final_score = float(agg_scores[self._aggregation])
        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, final_score))

        return DetectionResult(
            score=final_score,
            metadata={
                "aggregation": self._aggregation,
                "zero_claims": False,
                "claim_risks": claim_risks,
                "claim_texts": [c.text for c in claims],
                "claim_starts": [c.start for c in claims],
                "claim_ends": [c.end for c in claims],
                "claim_support": claim_support,
                "claim_contra": claim_contra,
                "all_aggregations": agg_scores,
                "n_pairs": example_n_pairs,
                "n_truncated": example_n_truncated,
            },
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_nli_detector(
    aggregation: AggregationMode = "max",
    window_words: int = 180,
    stride_words: int = 90,
    batch_size: int = 32,
    cache: Optional[NLICache] = None,
    _scorer: Optional[ScorerFn] = None,
) -> NLIDetector:
    """
    Instantiate NLIDetector reading NLI_MODEL from the environment.

    Raises
    ------
    RuntimeError
        If NLI_MODEL is not set in the environment.
    """
    from dotenv import load_dotenv
    load_dotenv()
    model_name = os.environ.get("NLI_MODEL", "").strip()
    if not model_name:
        raise RuntimeError(
            "NLI_MODEL environment variable is not set. "
            "Add NLI_MODEL=<hf-model-id> to your .env file."
        )
    return NLIDetector(
        model_name=model_name,
        aggregation=aggregation,
        window_words=window_words,
        stride_words=stride_words,
        batch_size=batch_size,
        cache=cache,
        _scorer=_scorer,
    )

