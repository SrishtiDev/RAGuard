"""
eval/datasets.py
----------------
Dataset loaders for hallucination evaluation benchmarks.
Currently supports RAGTruth (QA and Summary tasks).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence, Union

from eval.schema import Example, Span

logger = logging.getLogger(__name__)


def _find_ragtruth_files(raw_dir: Union[str, Path]) -> tuple[Path, Path]:
    """Resolve paths to source_info.jsonl and response.jsonl."""
    raw_path = Path(raw_dir)
    candidates = [
        (raw_path / "source_info.jsonl", raw_path / "response.jsonl"),
        (raw_path / "ragtruth" / "source_info.jsonl", raw_path / "ragtruth" / "response.jsonl"),
    ]
    for src_file, resp_file in candidates:
        if src_file.is_file() and resp_file.is_file():
            return src_file, resp_file
    raise FileNotFoundError(
        f"Could not locate source_info.jsonl and response.jsonl in {raw_path} "
        f"or {raw_path / 'ragtruth'}"
    )


def load_ragtruth(
    raw_dir: Union[str, Path],
    task_types: Sequence[str] = ("QA", "Summary"),
    exclude_quality: Sequence[str] = ("incorrect_refusal", "truncated"),
    seed: Optional[int] = None,
) -> list[Example]:
    """
    Load RAGTruth benchmark examples into canonical Example instances.

    Parameters
    ----------
    raw_dir : Union[str, Path]
        Directory containing RAGTruth jsonl files or containing a 'ragtruth' subdirectory.
    task_types : Sequence[str]
        Task subsets to load. Default: ("QA", "Summary").
        Data2txt is not yet supported and will raise NotImplementedError.
    exclude_quality : Sequence[str]
        Generation quality flags to exclude. Default: ("incorrect_refusal", "truncated").
    seed : Optional[int]
        Optional random seed for reproducibility.

    Returns
    -------
    list[Example]
        List of canonical Example objects.
    """
    if "Data2txt" in task_types:
        raise NotImplementedError(
            "Data2txt task_type is not supported yet in load_ragtruth."
        )

    for tt in task_types:
        if tt not in ("QA", "Summary"):
            raise ValueError(f"Unsupported task_type: '{tt}'. Supported: 'QA', 'Summary'.")

    source_file, response_file = _find_ragtruth_files(raw_dir)

    # 1. Load source_info.jsonl
    sources: dict[str, dict] = {}
    with open(source_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            source_id = str(item["source_id"])
            tt = item["task_type"]
            si = item["source_info"]

            # Assert strict schema expectations per task_type
            if tt == "Summary":
                if not isinstance(si, str):
                    raise TypeError(
                        f"Expected str for Summary source_info at source_id={source_id}, "
                        f"got {type(si).__name__}"
                    )
            elif tt == "QA":
                if not isinstance(si, dict):
                    raise TypeError(
                        f"Expected dict for QA source_info at source_id={source_id}, "
                        f"got {type(si).__name__}"
                    )
                if "question" not in si or "passages" not in si:
                    raise KeyError(
                        f"QA source_info at source_id={source_id} must contain "
                        f"'question' and 'passages' keys. Got: {list(si.keys())}"
                    )
                if not isinstance(si["passages"], (str, list)):
                    raise TypeError(
                        f"Expected str or list for QA passages at source_id={source_id}, "
                        f"got {type(si['passages']).__name__}"
                    )
            elif tt == "Data2txt":
                if not isinstance(si, dict):
                    raise TypeError(
                        f"Expected dict for Data2txt source_info at source_id={source_id}, "
                        f"got {type(si).__name__}"
                    )

            sources[source_id] = item

    # 2. Load response.jsonl and join on source_id
    examples: list[Example] = []
    loaded_per_task: Counter = Counter()
    excluded_per_quality: Counter = Counter()
    span_mismatches_total: int = 0

    with open(response_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            resp = json.loads(line)
            resp_id = str(resp["id"])
            source_id = str(resp["source_id"])

            if source_id not in sources:
                raise KeyError(
                    f"Response ID '{resp_id}' references unknown source_id '{source_id}'."
                )

            src = sources[source_id]
            task_type = src["task_type"]
            if task_type not in task_types:
                continue

            quality = resp.get("quality", "good")
            if quality in exclude_quality:
                excluded_per_quality[quality] += 1
                continue

            resp_text = resp["response"]

            # Spans
            spans: list[Span] = []
            for lbl in resp.get("labels", []):
                span = Span(
                    start=lbl["start"],
                    end=lbl["end"],
                    text=lbl["text"],
                    label_type=lbl["label_type"],
                    implicit_true=bool(lbl.get("implicit_true", False)),
                    due_to_null=bool(lbl.get("due_to_null", False)),
                )
                actual_slice = resp_text[span.start : span.end]
                if actual_slice != span.text:
                    span_mismatches_total += 1
                    logger.warning(
                        "Span mismatch in response %s: expected %r, got %r",
                        resp_id,
                        span.text,
                        actual_slice,
                    )
                spans.append(span)

            # Task-specific context and query mapping
            si = src["source_info"]
            if task_type == "QA":
                query = si["question"]
                passages = si["passages"]
                if isinstance(passages, list):
                    context_chunks = passages
                    context = "\n\n".join(passages)
                else:
                    context = passages
                    context_chunks = None
            elif task_type == "Summary":
                query = None  # Do NOT parse it out of the prompt
                context = si
                context_chunks = None
            else:
                # Should not be reached due to upfront validation
                continue

            example = Example(
                id=f"ragtruth:{resp_id}",
                source_dataset="ragtruth",
                subset=task_type,
                group_id=source_id,
                query=query,
                context=context,
                context_chunks=context_chunks,
                response=resp_text,
                label_response_level=len(spans) > 0,
                spans=spans,
                meta={
                    "model": resp.get("model"),
                    "temperature": resp.get("temperature"),
                    "quality": quality,
                    "original_split": resp.get("split"),
                },
            )
            examples.append(example)
            loaded_per_task[task_type] += 1

    logger.info("Loaded RAGTruth records per task_type: %s", dict(loaded_per_task))
    logger.info("Excluded RAGTruth records per quality: %s", dict(excluded_per_quality))
    if span_mismatches_total > 0:
        logger.warning("Total span text mismatches observed: %d", span_mismatches_total)
    else:
        logger.info("Total span text mismatches observed: 0")

    return examples

