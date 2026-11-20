"""
src/cache.py
------------
Lightweight in-memory query-answer cache for NexusRAG.

Design:
  - Normalizes and hashes the user question (lowercase, stripped).
  - Stores the full answer_question result dict keyed by hash.
  - LRU-style eviction: when the cache exceeds MAX_SIZE, the oldest
    entry is dropped (insertion-ordered dict).
  - Cache is session-scoped (lives as long as the Python process).
    For multi-user / persistent caching, swap the backing store for
    Redis with a TTL.
"""

import hashlib
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Maximum number of question-answer pairs to keep in memory.
_MAX_SIZE: int = 128

# {question_hash: {"result": ..., "ts": unix_time}}
_store: Dict[str, Dict[str, Any]] = {}

# Simple hit/miss counters (useful for the debug panel).
_hits: int = 0
_misses: int = 0


def _normalise(question: str) -> str:
    """Lower-case, collapse whitespace — same question, different casing = same key."""
    return " ".join(question.lower().split())


def _hash(question: str) -> str:
    return hashlib.sha256(_normalise(question).encode()).hexdigest()


def get(question: str) -> Optional[Dict[str, Any]]:
    """Return cached result or None."""
    global _hits, _misses
    key = _hash(question)
    entry = _store.get(key)
    if entry:
        _hits += 1
        logger.info("Cache HIT for: %s…", question[:60])
        return entry["result"]
    _misses += 1
    return None


def put(question: str, result: Dict[str, Any]) -> None:
    """Store a result. Evicts oldest entry when cache is full."""
    key = _hash(question)
    if key in _store:
        # Refresh existing entry.
        _store.pop(key)

    if len(_store) >= _MAX_SIZE:
        oldest_key = next(iter(_store))
        _store.pop(oldest_key)
        logger.debug("Cache evicted oldest entry (size=%d).", _MAX_SIZE)

    _store[key] = {"result": result, "ts": time.time()}
    logger.info("Cache SET for: %s…", question[:60])


def clear() -> None:
    """Wipe the entire cache (called when index is rebuilt)."""
    global _hits, _misses
    _store.clear()
    _hits = 0
    _misses = 0
    logger.info("Query cache cleared.")


def stats() -> Dict[str, int]:
    return {"size": len(_store), "hits": _hits, "misses": _misses}

