"""
eval/cache.py
-------------
SQLite-backed inference cache for NLI pair probabilities.

Key  = SHA-256(model_name + "|" + premise + "|" + hypothesis)
Value = three probabilities in canonical order [P(entailment), P(neutral), P(contradiction)]
        plus a boolean truncated flag.

Design constraints:
- Raw text (premise / hypothesis) is NEVER stored.
- Re-running with a different aggregation or threshold hits the cache; no new model calls.
- Thread-safe via a module-level lock (single-process, multi-thread safety).
- Database lives at data/processed/cache/<db_name>.db (directory is git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level lock for thread safety (single process, potentially multi-threaded)
_LOCK = threading.Lock()

# Default cache directory (created on first use)
_DEFAULT_CACHE_DIR = Path("data/processed/cache")


def _make_key(model_name: str, premise: str, hypothesis: str) -> str:
    """Compute a deterministic SHA-256 cache key from model + premise + hypothesis."""
    raw = f"{model_name}\x00{premise}\x00{hypothesis}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class NLICache:
    """
    SQLite-backed cache for NLI inference results.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file. Created (with parent directories) if absent.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_schema()
        self._hits = 0
        self._misses = 0

    def _init_schema(self) -> None:
        with _LOCK, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nli_cache (
                    key        TEXT PRIMARY KEY,
                    p_ent      REAL NOT NULL,
                    p_neu      REAL NOT NULL,
                    p_con      REAL NOT NULL,
                    truncated  INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def get(
        self, model_name: str, premise: str, hypothesis: str
    ) -> Optional[tuple[float, float, float, bool]]:
        """
        Retrieve cached probabilities.

        Returns
        -------
        Optional[tuple[float, float, float, bool]]
            (p_entailment, p_neutral, p_contradiction, truncated) or None if not cached.
        """
        key = _make_key(model_name, premise, hypothesis)
        with _LOCK:
            cur = self._conn.execute(
                "SELECT p_ent, p_neu, p_con, truncated FROM nli_cache WHERE key = ?", (key,)
            )
            row = cur.fetchone()

        if row is None:
            self._misses += 1
            return None

        self._hits += 1
        p_ent, p_neu, p_con, trunc = row
        return float(p_ent), float(p_neu), float(p_con), bool(trunc)

    def put(
        self,
        model_name: str,
        premise: str,
        hypothesis: str,
        p_ent: float,
        p_neu: float,
        p_con: float,
        truncated: bool = False,
    ) -> None:
        """Store NLI probabilities. Silently updates on key collision."""
        key = _make_key(model_name, premise, hypothesis)
        with _LOCK, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO nli_cache (key, p_ent, p_neu, p_con, truncated)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, float(p_ent), float(p_neu), float(p_con), int(truncated)),
            )

    @property
    def hit_rate(self) -> Optional[float]:
        """Cache hit rate since object creation. None if no lookups yet."""
        total = self._hits + self._misses
        if total == 0:
            return None
        return self._hits / total

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def close(self) -> None:
        with _LOCK:
            self._conn.close()

    def __enter__(self) -> "NLICache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def open_cache(
    model_name: str,
    cache_dir: Optional[Path] = None,
) -> NLICache:
    """
    Open (or create) an NLI cache database for the given model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (used to namespace the DB file).
    cache_dir : Optional[Path]
        Directory for the SQLite file. Defaults to data/processed/cache/.

    Returns
    -------
    NLICache
        Opened cache instance.
    """
    base = cache_dir or _DEFAULT_CACHE_DIR
    # Sanitize model_name for use as a filename
    safe_name = model_name.replace("/", "__").replace("\\", "__")
    db_path = base / f"{safe_name}.db"
    logger.info("NLI cache at: %s", db_path)
    return NLICache(db_path)

