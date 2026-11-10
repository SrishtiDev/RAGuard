"""
detectors/base.py
-----------------
Abstract base class and return schema for hallucination detectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

from eval.schema import DetectorInput


class DetectionResult(BaseModel):
    """
    Standard output produced by a detector prediction.

    Attributes
    ----------
    score : float
        Hallucination risk score strictly within [0.0, 1.0] (1.0 = definitely hallucinated).
    metadata : dict[str, Any]
        Diagnostic metadata, intermediate scores, or sub-signals.
    tokens : Optional[dict[str, Any]]
        Optional token consumption tracking (e.g. prompt/completion counts).
    """

    score: float = Field(ge=0.0, le=1.0, description="Hallucination risk score in [0.0, 1.0]")
    metadata: dict[str, Any] = Field(default_factory=dict)
    tokens: Optional[dict[str, Any]] = None


class Detector(ABC):
    """
    Abstract base class for hallucination detectors.
    Detectors receive a leakage-safe DetectorInput and return a DetectionResult.
    Thresholding is deliberately decoupled from detectors and managed by the evaluation harness.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this detector."""
        ...

    @abstractmethod
    def predict(self, inp: DetectorInput) -> DetectionResult:
        """
        Evaluate hallucination risk on the given input.

        Parameters
        ----------
        inp : DetectorInput
            Leakage-safe input containing query, context, and response text.

        Returns
        -------
        DetectionResult
            Risk score in [0.0, 1.0], metadata, and optional token usage.
        """
        ...

