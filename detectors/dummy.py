"""
detectors/dummy.py
------------------
Baseline dummy detectors for sanity checks and harness verification.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from detectors.base import DetectionResult, Detector
from eval.schema import DetectorInput


class AlwaysHallucinated(Detector):
    """
    Trivial baseline detector that assigns maximum hallucination risk (1.0) to every example.
    """

    @property
    def name(self) -> str:
        return "always"

    def predict(self, inp: DetectorInput) -> DetectionResult:
        return DetectionResult(score=1.0, metadata={"strategy": "always_hallucinated"})


class NeverHallucinated(Detector):
    """
    Trivial baseline detector that assigns minimum hallucination risk (0.0) to every example.
    """

    @property
    def name(self) -> str:
        return "never"

    def predict(self, inp: DetectorInput) -> DetectionResult:
        return DetectionResult(score=0.0, metadata={"strategy": "never_hallucinated"})


class RandomDetector(Detector):
    """
    Stochastic baseline detector whose prediction is derived deterministically from
    hash(id + seed), ensuring independence from execution order and dataset batching.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    @property
    def name(self) -> str:
        return "random"

    def predict(self, inp: DetectorInput) -> DetectionResult:
        payload = f"{inp.id}:{self.seed}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        # Map first 8 bytes of SHA-256 to float in [0.0, 1.0)
        uint64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
        score = float(uint64 / (2**64))
        return DetectionResult(score=score, metadata={"seed": self.seed})


class LengthBaseline(Detector):
    """
    Length heuristic baseline: score = words / (words + 100).
    Longer responses receive higher hallucination scores.
    Sanity check to ensure genuine detectors perform better than simple verbosity artifacts.
    """

    @property
    def name(self) -> str:
        return "length"

    def predict(self, inp: DetectorInput) -> DetectionResult:
        words = len(inp.response.split())
        score = float(words / (words + 100.0))
        return DetectionResult(score=score, metadata={"word_count": words})


def get_dummy_detector(name: str, seed: int = 42) -> Detector:
    """
    Retrieve a dummy detector instance by name.
    """
    registry = {
        "always": AlwaysHallucinated,
        "never": NeverHallucinated,
        "random": lambda: RandomDetector(seed=seed),
        "length": LengthBaseline,
    }
    key = name.lower().strip()
    if key not in registry:
        raise ValueError(
            f"Unknown dummy detector '{name}'. Available: {list(registry.keys())}"
        )
    return registry[key]()

