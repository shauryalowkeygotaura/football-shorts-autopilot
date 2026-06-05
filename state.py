"""Tiny JSON state store so the autonomous loop never repeats a topic.

Mirrors philosopher-pipeline/state.py: a single JSON file tracking which
topics have already been turned into docs, plus a published log for metrics.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_STATE_PATH = Path(__file__).parent / "state.json"


def _load() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"used_topics": [], "published": []}


def _save(state: dict) -> None:
    _STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def is_used(topic_key: str) -> bool:
    return topic_key.strip().lower() in {t.lower() for t in _load().get("used_topics", [])}


def mark_used(topic_key: str) -> None:
    state = _load()
    state.setdefault("used_topics", []).append(topic_key.strip())
    _save(state)


def log_published(record: dict) -> None:
    """record = {asset, kind ('doc'|'short'), video_id, title, topic, ...}."""
    state = _load()
    record["published_at"] = datetime.now(timezone.utc).isoformat()
    state.setdefault("published", []).append(record)
    _save(state)
