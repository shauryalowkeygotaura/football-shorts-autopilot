#!/usr/bin/env python
"""football-shorts-autopilot — end-to-end orchestrator.

Hybrid, fully autonomous:
    pick topic -> write doc script -> build 16:9 master doc
              -> slice 4 vertical shorts -> generate metadata
              -> upload doc + shorts to YouTube -> record state

Run:
    doppler run --project football-shorts-autopilot --config dev -- python pipeline.py --now
    YT_DRY_RUN=1 ... python pipeline.py --now      # render only, no upload, no quota

Cadence is enforced by GitHub Actions cron (.github/workflows/autopilot.yml);
this script does ONE full cycle per invocation.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config       # noqa: E402
import state        # noqa: E402
from modules import (topics, scriptwriter, composer, shorts, metadata,  # noqa: E402
                     thumbnail, uploader, run_metrics)

log = logging.getLogger("autopilot")


def run_cycle(shorts_only: bool = False, doc_only: bool = False) -> dict:
    config.PENDING.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    work = config.CACHE / run_id

    # 1. Idea machine
    topic = topics.pick_topic()
    log.info("Topic: %r (score=%.1f, %s)", topic.title, topic.score, topic.source)

    # 2. Script (hook -> supporting hook -> body -> payoff)
    script = scriptwriter.write_script(topic.title)
    log.info("Script: %d beats, %d shortable",
             len(script.beats), sum(b.shortable for b in script.beats))

    # 3. Master doc (16:9)
    doc = composer.build_doc(script, work)
    log.info("Doc: %.1fs -> %s", doc.duration, doc.video_path.name)

    # 4. Shorts (9:16) sliced from the doc
    short_paths = [] if doc_only else shorts.slice_shorts(doc, work)
    log.info("Shorts: %d sliced", len(short_paths))

    published = []

    # 5a. Upload the doc + a custom CTR thumbnail
    if not shorts_only:
        dmeta = metadata.make_meta(topic.title, "long-form doc")
        final_doc = config.PENDING / f"{run_id}-doc.mp4"
        final_doc.write_bytes(doc.video_path.read_bytes())
        # thumbnail from the doc's first still so it always matches the content
        thumb_src = doc.beats[0].image if doc.beats else None
        thumb = thumbnail.make_thumbnail(
            thumb_src, dmeta.title, config.PENDING / f"{run_id}-thumb.jpg"
        ) if thumb_src else None
        vid = uploader.upload(final_doc, dmeta.title, dmeta.description, dmeta.tags)
        if thumb:
            uploader.set_thumbnail(vid, thumb)
        state.log_published({"asset": final_doc.name, "kind": "doc",
                             "video_id": vid, "title": dmeta.title, "topic": topic.title})
        published.append(("doc", vid, dmeta.title))
        log.info("Uploaded doc: %s (%s)%s", dmeta.title, vid,
                 " +thumb" if thumb else "")

    # 5b. Upload the shorts
    for i, sp in enumerate(short_paths):
        smeta = metadata.make_meta(topic.title, "short")
        final_short = config.PENDING / f"{run_id}-short{i}.mp4"
        final_short.write_bytes(Path(sp).read_bytes())
        vid = uploader.upload(final_short, smeta.title, smeta.description, smeta.tags)
        state.log_published({"asset": final_short.name, "kind": "short",
                             "video_id": vid, "title": smeta.title, "topic": topic.title})
        published.append(("short", vid, smeta.title))
        log.info("Uploaded short %d: %s (%s)", i, smeta.title, vid)

    state.mark_used(topic.key)

    # Tell the Command Center dashboard what happened this cycle.
    n_shorts = sum(1 for k, _, _ in published if k == "short")
    n_doc = sum(1 for k, _, _ in published if k == "doc")
    dry = uploader.is_dry()  # also true when YT OAuth creds are absent
    run_metrics.write(
        mode="doc-only" if doc_only else ("shorts-only" if shorts_only else "cycle"),
        status="ok" if published else "degraded",
        summary=(f"{'[dry] ' if dry else ''}{topic.title[:60]} "
                 f"-> {n_doc} doc + {n_shorts} shorts"),
        metrics={"topic": topic.title, "beats": len(script.beats),
                 "shorts": n_shorts, "docs": n_doc, "dry_run": dry,
                 "doc_seconds": round(doc.duration, 1)},
        budgets={"youtube": {"used": None, "limit": 10000,
                             "note": "1600 units/upload; ~6/day on default quota"}},
    )
    return {"topic": topic.title, "published": published}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true", help="run one cycle immediately")
    ap.add_argument("--shorts-only", action="store_true")
    ap.add_argument("--doc-only", action="store_true")
    args = ap.parse_args()

    # Player/country names carry non-ASCII (Mbappé, Modrić); keep Windows
    # consoles from crashing on log output. CI (Linux) is already UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    if not args.now:
        ap.error("pass --now (scheduling is handled by GitHub Actions cron)")

    try:
        result = run_cycle(shorts_only=args.shorts_only, doc_only=args.doc_only)
    except Exception as e:
        # Surface the failure on the Command Center instead of going silently red.
        run_metrics.write(mode="cycle", status="error",
                          summary=f"cycle failed: {type(e).__name__}: {e}"[:160])
        raise
    log.info("DONE: %s -> %d assets", result["topic"], len(result["published"]))


if __name__ == "__main__":
    main()
