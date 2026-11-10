"""
eval/logger.py
--------------
Experiment logging and artifact persistence for hallucination evaluation runs.
Saves run configuration, predictions, and summary metrics to runs/<timestamp>_<detector>_<split>/.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import sklearn

logger = logging.getLogger(__name__)


def get_git_info() -> tuple[str, bool]:
    """Return the current HEAD commit hash and whether the working tree is dirty."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        is_dirty = len(status) > 0
        return commit, is_dirty
    except Exception:
        return "unknown", False


def get_environment_versions() -> dict[str, str]:
    """Return current runtime and library versions."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def create_run_dir(
    base_dir: Union[str, Path] = "runs",
    detector_name: str = "detector",
    split_name: str = "split",
) -> Path:
    """
    Create a timestamped directory for the evaluation run:
    runs/<YYYYMMDD_HHMMSS>_<detector>_<split>/
    """
    utc_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_det = detector_name.lower().replace(" ", "_")
    clean_split = split_name.lower()
    dir_name = f"{utc_timestamp}_{clean_det}_{clean_split}"

    run_path = Path(base_dir) / dir_name
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def save_run_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    predictions: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Path]:
    """
    Write config.json, predictions.jsonl, and summary.json to run_dir.

    predictions.jsonl enforces the schema:
    id, group_id, subset, model, label, score, latency_ms.
    Textual response and context fields are strictly excluded.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. config.json
    config_file = run_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 2. predictions.jsonl
    pred_file = run_dir / "predictions.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            # Enforce that response and context are never stored
            row = {
                "id": str(pred["id"]),
                "group_id": str(pred["group_id"]),
                "subset": str(pred.get("subset", "unknown")),
                "model": str(pred.get("model", "unknown")),
                "label": bool(pred["label"]),
                "score": float(pred["score"]),
                "latency_ms": float(pred.get("latency_ms", 0.0)),
            }
            f.write(json.dumps(row) + "\n")

    # 3. summary.json
    summary_file = run_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved run artifacts to %s", run_dir)
    return {
        "config": config_file,
        "predictions": pred_file,
        "summary": summary_file,
    }


def load_run_threshold(run_dir: Union[str, Path]) -> float:
    """
    Extract the tuned threshold from a previous run directory's config.json or summary.json.
    """
    run_path = Path(run_dir)
    config_file = run_path / "config.json"
    summary_file = run_path / "summary.json"

    if config_file.is_file():
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            if "threshold" in cfg and cfg["threshold"] is not None:
                return float(cfg["threshold"])

    if summary_file.is_file():
        with open(summary_file, "r", encoding="utf-8") as f:
            summ = json.load(f)
            if "threshold" in summ and summ["threshold"] is not None:
                return float(summ["threshold"])

    raise ValueError(
        f"Could not load a valid threshold from run directory: {run_path}. "
        "Neither config.json nor summary.json contained a valid 'threshold'."
    )

