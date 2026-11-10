"""
eval/splits.py
--------------
Dataset partitioning and split generation for RAGTruth.
Preserves official test split and creates a source-grouped, task-stratified validation split.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Union

from eval.schema import Example

logger = logging.getLogger(__name__)


def compute_id_hash(ids: Sequence[str]) -> str:
    """
    Compute a deterministic SHA-256 hash for a sequence of IDs.
    Sorts IDs first to guarantee canonical ordering.
    """
    sorted_ids = sorted(ids)
    return hashlib.sha256("\n".join(sorted_ids).encode("utf-8")).hexdigest()


def assert_no_leakage(splits: dict[str, list[Example]]) -> None:
    """
    Verify that no group_id (source_id) appears across multiple splits.

    Parameters
    ----------
    splits : dict[str, list[Example]]
        Mapping of split names to lists of Example objects.

    Raises
    ------
    AssertionError
        If any group_id is found in more than one split.
    """
    group_map: dict[str, set[str]] = {}
    for name, examples in splits.items():
        group_map[name] = set(ex.group_id for ex in examples)

    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = group_map[s1].intersection(group_map[s2])
            if overlap:
                raise AssertionError(
                    f"Data leakage detected between splits '{s1}' and '{s2}': "
                    f"{len(overlap)} overlapping group_ids: {sorted(list(overlap))[:5]}"
                )


def split_ragtruth(
    examples: list[Example],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Example]]:
    """
    Split RAGTruth examples into train, val, and test partitions.

    - The official test split (original_split == 'test') is preserved exactly as given.
    - Val is carved from official train split (~15% of source_ids), grouped by
      source_id (group_id) and stratified by task_type (subset).
    - All responses of a single source_id stay in the same split.

    Parameters
    ----------
    examples : list[Example]
        Loaded RAGTruth examples.
    val_ratio : float
        Proportion of train source_ids to allocate to validation (default: 0.15).
    seed : int
        Random seed for shuffling source_ids (default: 42).

    Returns
    -------
    dict[str, list[Example]]
        Dictionary with keys 'train', 'val', 'test'.
    """
    test_examples = [ex for ex in examples if ex.meta.get("original_split") == "test"]
    train_pool = [ex for ex in examples if ex.meta.get("original_split") == "train"]

    # Group train source_ids by task_type (subset)
    train_sources_by_task: dict[str, set[str]] = defaultdict(set)
    for ex in train_pool:
        train_sources_by_task[ex.subset].add(ex.group_id)

    val_source_ids: set[str] = set()
    for task_type, sources in sorted(train_sources_by_task.items()):
        sorted_sources = sorted(list(sources))
        rng = random.Random(seed)
        rng.shuffle(sorted_sources)
        n_val = round(len(sorted_sources) * val_ratio)
        val_source_ids.update(sorted_sources[:n_val])

    train_examples = [ex for ex in train_pool if ex.group_id not in val_source_ids]
    val_examples = [ex for ex in train_pool if ex.group_id in val_source_ids]

    # Sort each split deterministically by example ID
    train_examples.sort(key=lambda ex: ex.id)
    val_examples.sort(key=lambda ex: ex.id)
    test_examples.sort(key=lambda ex: ex.id)

    splits = {
        "train": train_examples,
        "val": val_examples,
        "test": test_examples,
    }

    assert_no_leakage(splits)
    return splits


# Alias for convenience
create_ragtruth_splits = split_ragtruth


def save_splits(
    splits: dict[str, list[Example]],
    output_dir: Union[str, Path] = "data/processed/splits",
    seed: int = 42,
) -> dict[str, Path]:
    """
    Save split ID lists and metadata to JSON files.

    Each file contains:
      - split: split name
      - seed: the random seed used
      - sha256: SHA-256 hash of sorted example IDs
      - ids: sorted list of example IDs
      - source_ids: sorted list of source_ids (group_ids)
      - num_examples: count of examples
      - num_sources: count of unique source_ids

    Parameters
    ----------
    splits : dict[str, list[Example]]
        Dictionary with 'train', 'val', 'test' example lists.
    output_dir : Union[str, Path]
        Directory in which to save the JSON files.
    seed : int
        Seed used to generate the split.

    Returns
    -------
    dict[str, Path]
        Mapping from split name to the saved JSON file path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved_files: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        split_examples = splits.get(split_name, [])
        sorted_ids = sorted(ex.id for ex in split_examples)
        sorted_sources = sorted(set(ex.group_id for ex in split_examples))
        sha256_hash = compute_id_hash(sorted_ids)

        payload = {
            "split": split_name,
            "seed": seed,
            "sha256": sha256_hash,
            "ids": sorted_ids,
            "example_ids": sorted_ids,
            "source_ids": sorted_sources,
            "num_examples": len(sorted_ids),
            "num_sources": len(sorted_sources),
        }

        file_path = out_path / f"ragtruth_{split_name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        saved_files[split_name] = file_path
        logger.info(
            "Saved %s split to %s (sha256: %s, %d examples, %d sources)",
            split_name,
            file_path,
            sha256_hash,
            len(sorted_ids),
            len(sorted_sources),
        )

    return saved_files


def format_split_table(splits: dict[str, list[Example]]) -> str:
    """
    Format a markdown table summarizing count and hallucination rate per split x task_type.
    """
    lines = [
        "| Split | Task Type | N Sources | N Responses | Hallucinated | Hallucination Rate |",
        "|---|---|---|---|---|---|",
    ]

    total_sources_all: set[str] = set()
    total_resps_all = 0
    total_hal_all = 0

    for split_name in ("train", "val", "test"):
        if split_name not in splits:
            continue
        sp_exs = splits[split_name]
        by_task: dict[str, list[Example]] = defaultdict(list)
        for ex in sp_exs:
            by_task[ex.subset].append(ex)

        for tt in sorted(by_task.keys()):
            tt_exs = by_task[tt]
            n_src = len(set(ex.group_id for ex in tt_exs))
            n_resp = len(tt_exs)
            n_hal = sum(1 for ex in tt_exs if ex.label_response_level)
            rate = (n_hal / n_resp * 100) if n_resp > 0 else 0.0
            lines.append(
                f"| **{split_name}** | {tt} | {n_src} | {n_resp} | {n_hal} | {rate:.2f}% |"
            )

        sub_src = len(set(ex.group_id for ex in sp_exs))
        sub_resp = len(sp_exs)
        sub_hal = sum(1 for ex in sp_exs if ex.label_response_level)
        sub_rate = (sub_hal / sub_resp * 100) if sub_resp > 0 else 0.0
        lines.append(
            f"| **{split_name} (Subtotal)** | All | {sub_src} | {sub_resp} | {sub_hal} | **{sub_rate:.2f}%** |"
        )

        total_sources_all.update(ex.group_id for ex in sp_exs)
        total_resps_all += sub_resp
        total_hal_all += sub_hal

    total_rate = (total_hal_all / total_resps_all * 100) if total_resps_all > 0 else 0.0
    lines.append(
        f"| **Total** | All | {len(total_sources_all)} | {total_resps_all} | {total_hal_all} | **{total_rate:.2f}%** |"
    )

    return "\n".join(lines)


def print_split_table(splits: dict[str, list[Example]]) -> None:
    """Print the split table to standard output."""
    print(format_split_table(splits))

