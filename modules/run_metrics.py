"""Run-metrics writer: emits the JSON the Command Center dashboard reads.

The Command Center is a static GitHub Pages app with no backend, so every
pipeline run drops a `runs/latest.json` (and appends to `runs/history.jsonl`)
into the repo; the GitHub Actions workflow commits it and the dashboard fetches
it over raw.githubusercontent. Schema is shared across all pipelines so one
dashboard component renders any of them:

    {
      "pipeline": "football-shorts",
      "ts":       "2026-06-05T08:04:13Z",   # UTC ISO8601
      "mode":     "cycle",                    # cycle | shorts-only | doc-only
      "status":   "ok" | "degraded" | "error",
      "summary":  "human one-liner",
      "metrics":  { ... arbitrary counters ... },
      "budgets":  { "<service>": {"used": int|null, "limit": int, "note": str} }
    }

`status` semantics:
  ok        produced and staged/uploaded assets
  degraded  ran clean but produced nothing (e.g. every topic already used)
  error     unhandled exception bubbled out
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repo root = parent of the modules/ package this file lives in.
_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

PIPELINE_NAME = "football-shorts"


def write(
    mode: str,
    status: str,
    summary: str,
    metrics: dict[str, Any] | None = None,
    budgets: dict[str, Any] | None = None,
) -> Path:
    """Write runs/latest.json and append to runs/history.jsonl. Never raises:
    a metrics-write failure must not take down a real pipeline run."""
    payload = {
        "pipeline": PIPELINE_NAME,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "status": status,
        "summary": summary,
        "metrics": metrics or {},
        "budgets": budgets or {},
    }
    try:
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        latest = _RUNS_DIR / "latest.json"
        latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with (_RUNS_DIR / "history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
        return latest
    except Exception as e:  # pragma: no cover - best effort
        print(f"    [run_metrics] failed to write metrics: {e}")
        return _RUNS_DIR / "latest.json"
