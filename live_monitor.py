#!/usr/bin/env python
"""live_monitor.py: first-to-post GOAL-NEWS short detector (DESIGN / DRY-RUN).

Polls a free football data API (football-data.org v4) for live matches, detects
a goal the moment a score increments, and triggers the SAME free render chain
the daily pipeline already uses (Groq script -> edge-tts narration -> Wikimedia
Commons stills -> Ken-Burns -> FFmpeg vertical short with burned captions ->
metadata -> uploader) to stage a "GOAL: X scores" short within minutes.

LEGAL LINE (do not cross):
  - STILLS ONLY. This path renders over Creative-Commons / public-domain
    photographs of players, teams, and stadiums. It NEVER downloads, clips,
    embeds, or posts broadcast match footage. The operator is a minor; a single
    posted broadcast clip is a copyright strike, so the stills-only rule is
    enforced in code (`_stills_only()` disables the Pexels video b-roll layer
    for goal-news renders) and not left to discipline.
  - POSTING STAYS DRY-RUN. There is no YouTube OAuth on this path by design.
    Auto-publish is OFF and gated behind LIVE_MONITOR_AUTOPOST=1 AND real YT
    creds; absent either, every "upload" is a dry-run that writes nothing to
    YouTube. The fixture replay proves a short is rendered while nothing posts.

Run (live, dry by default, needs FOOTBALL_DATA_TOKEN via Doppler):
    doppler run --project youtube-title-autoresearch --config dev -- \\
        python live_monitor.py --watch --once

Prove it offline (renders one short, uploads nothing):
    python test_goal_replay.py

Secrets: FOOTBALL_DATA_TOKEN (free key from football-data.org) lives in Doppler,
never hardcoded. GROQ_API_KEY is reused from the existing pipeline secrets.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import state   # noqa: E402
from modules import composer, metadata, run_metrics, shorts, uploader  # noqa: E402
from modules.scriptwriter import Beat, Script, _client, _parse_script  # noqa: E402

log = logging.getLogger("live-monitor")

# ── API + safety constants ────────────────────────────────────────────────────
FOOTBALL_DATA_API = "https://api.football-data.org/v4/matches"

# Whitelist over character-removal: only these status values may reach the API
# query string, and competition codes must match an exact shape. Anything else
# is rejected, never sanitized-then-used.
_ALLOWED_STATUS = {"LIVE", "IN_PLAY", "PAUSED", "FINISHED", "SCHEDULED", "TIMED"}
_COMP_CODE_RE = re.compile(r"^[A-Z0-9]{2,12}$")

# Persistent score memory so a restart resumes from the last good state instead
# of re-firing every in-progress scoreline (long unattended jobs checkpoint).
_STATE_PATH = config.ROOT / "live_state.json"


def _autopost_enabled() -> bool:
    """Auto-publish to YouTube is OFF unless the operator explicitly opts in.
    Mirrors the WhatsApp/IG draft-by-default rule for a minor's account: the
    default path posts nothing, ever."""
    return os.environ.get("LIVE_MONITOR_AUTOPOST", "0").strip() == "1"


@contextmanager
def _stills_only():
    """Force the render to use Commons STILLS + Ken-Burns only by disabling the
    Pexels video b-roll layer for the duration of a goal-news render. This is the
    code-level guarantee behind the stills-only legal rule: even if a Pexels key
    is configured for the daily docs, goal-news shorts can never pull video."""
    saved = os.environ.pop("PEXELS_API_KEY", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["PEXELS_API_KEY"] = saved


# ── Goal event model ──────────────────────────────────────────────────────────
@dataclass
class GoalEvent:
    match_id: str
    competition: str
    home: str
    away: str
    home_score: int
    away_score: int
    side: str               # "home" | "away": which team just scored
    scoring_team: str
    conceding_team: str
    minute: int | None = None
    scorer: str | None = None

    @property
    def event_id(self) -> str:
        """Stable id used to dedupe across restarts: one fire per (match, score,
        side). A correction/VAR rollback yields a different score and never an id
        we already fired."""
        return f"{self.match_id}-{self.home_score}-{self.away_score}-{self.side}"

    @property
    def headline(self) -> str:
        who = self.scorer or self.scoring_team
        mins = f" {self.minute}'" if self.minute else ""
        return (f"GOAL: {who} for {self.scoring_team} vs {self.conceding_team}"
                f"{mins} ({self.home_score}-{self.away_score})")


# ── API boundary (validated) ──────────────────────────────────────────────────
def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _extract(match: dict) -> dict | None:
    """Normalize one football-data.org v4 match object into the flat record the
    detector reasons over. Returns None if the object is missing the fields we
    need, so a malformed entry can never crash the poll loop."""
    if not isinstance(match, dict):
        return None
    mid = match.get("id")
    if mid is None:
        return None
    score = (match.get("score") or {}).get("fullTime") or {}
    home_team = (match.get("homeTeam") or {})
    away_team = (match.get("awayTeam") or {})
    home = home_team.get("shortName") or home_team.get("name") or "Home"
    away = away_team.get("shortName") or away_team.get("name") or "Away"
    rec = {
        "id": str(mid),
        "competition": (match.get("competition") or {}).get("name") or "Football",
        "home": home,
        "away": away,
        "home_score": _safe_int(score.get("home")),
        "away_score": _safe_int(score.get("away")),
        "minute": match.get("minute") if isinstance(match.get("minute"), int) else None,
    }
    # Optional scorer enrichment: football-data includes a goals[] array on the
    # match-detail payload (and we accept it inline in fixtures). Map the latest
    # goal per team so the detector can name the scorer when available.
    scorers = {"home": None, "away": None}
    for g in (match.get("goals") or []):
        if not isinstance(g, dict):
            continue
        team = (g.get("team") or {}).get("name") or ""
        name = (g.get("scorer") or {}).get("name")
        if not name:
            continue
        if team in (home, home_team.get("name")):
            scorers["home"] = name
        elif team in (away, away_team.get("name")):
            scorers["away"] = name
    rec["scorers"] = scorers
    return rec


def poll_live(token: str, status: str = "LIVE",
              competitions: Iterable[str] | None = None,
              timeout: int = 25) -> list[dict]:
    """One validated poll of the live-matches endpoint. Returns extracted records.

    Fails LOUDLY on a bad/absent token (credential gate) but degrades to an empty
    list on transient throttling (429) or a parse error, so a single bad poll
    never takes down the watcher.
    """
    import requests

    if not token:
        raise ValueError("FOOTBALL_DATA_TOKEN not set (provide via Doppler). "
                         "Get a free key at https://www.football-data.org/client/register")
    status = status.strip().upper()
    if status not in _ALLOWED_STATUS:
        raise ValueError(f"status {status!r} not in whitelist {sorted(_ALLOWED_STATUS)}")
    params: dict[str, str] = {"status": status}
    if competitions:
        codes = [c.strip().upper() for c in competitions if c and c.strip()]
        bad = [c for c in codes if not _COMP_CODE_RE.match(c)]
        if bad:
            raise ValueError(f"competition codes rejected by whitelist: {bad}")
        if codes:
            params["competitions"] = ",".join(codes)

    try:
        r = requests.get(FOOTBALL_DATA_API, headers={"X-Auth-Token": token},
                         params=params, timeout=timeout)
    except requests.RequestException as e:
        log.warning("poll network error (%s); skipping this tick", type(e).__name__)
        return []

    if r.status_code in (401, 403):
        # A scoped-key mismatch is a hard, up-front failure, not a silent skip.
        raise PermissionError(
            f"football-data.org rejected the token (HTTP {r.status_code}). "
            "Check FOOTBALL_DATA_TOKEN is a valid free key.")
    if r.status_code == 429:
        log.warning("football-data.org rate limited this tick (HTTP 429); backing off")
        return []
    if r.status_code != 200:
        log.warning("football-data.org HTTP %s; skipping this tick", r.status_code)
        return []
    try:
        payload = r.json()
    except ValueError:
        log.warning("football-data.org returned non-JSON; skipping this tick")
        return []

    records = []
    for m in payload.get("matches", []):
        rec = _extract(m)
        if rec is not None:
            records.append(rec)
    return records


# ── Detection ─────────────────────────────────────────────────────────────────
def detect_goals(records: list[dict], prev_scores: dict[str, list[int]]
                 ) -> list[GoalEvent]:
    """Compare this snapshot's scores against the last seen scores and emit a
    GoalEvent for every side whose tally increased. First sight of a match only
    sets a baseline (no event), so starting mid-match never back-fires goals.

    Mutates `prev_scores` in place to the new baseline.
    """
    events: list[GoalEvent] = []
    for rec in records:
        mid = rec["id"]
        h, a = rec["home_score"], rec["away_score"]
        prev = prev_scores.get(mid)
        prev_scores[mid] = [h, a]
        if prev is None:
            continue  # baseline only; do not fire on first observation
        ph, pa = prev
        if h > ph:
            events.append(_make_event(rec, "home"))
        if a > pa:
            events.append(_make_event(rec, "away"))
    return events


def _make_event(rec: dict, side: str) -> GoalEvent:
    home, away = rec["home"], rec["away"]
    scoring = home if side == "home" else away
    conceding = away if side == "home" else home
    scorer = (rec.get("scorers") or {}).get(side)
    return GoalEvent(
        match_id=rec["id"], competition=rec["competition"],
        home=home, away=away, home_score=rec["home_score"],
        away_score=rec["away_score"], side=side, scoring_team=scoring,
        conceding_team=conceding, minute=rec.get("minute"), scorer=scorer)


# ── Goal-news script (Groq, with deterministic fallback) ──────────────────────
# Static system prompt FIRST (Groq prefix-cache discipline), no f-strings.
GOAL_SYSTEM_PROMPT = """You are the breaking-news writer for a faceless football
channel. You turn a single goal that JUST happened into a punchy 30-45 second
vertical short script: fast, factual, hype but never invented.

Hard rules:
- Open with a 1-sentence GOAL hook naming the scorer (or scoring team) and the
  scoreline. It must feel like breaking news.
- Then 2 short beats: what this goal means in the match right now, and one true
  bit of context about the scorer or team. Stay general if unsure; never invent
  a stat, quote, assist, or minute you were not given.
- End on a forward-looking line (what happens next in this match).
- Narration only, no stage directions. No em dashes; use commas or colons.

Return STRICT JSON, no markdown fences:
{
  "title_working": "...",
  "beats": [
    {"narration": "spoken line", "visual": "Real Proper Noun", "shortable": true},
    ...
  ]
}

VISUAL FIELD RULES (these become a Wikimedia Commons photo search, STILLS ONLY):
- The visual MUST be the NAME of a real, photographable subject likely to have a
  Commons photo: the scoring player, the scoring team, the opponent, or the
  stadium. Examples: "Kylian Mbappe", "France national football team".
- NO scenes, actions, or emotions ("player celebrating"); those return junk.
- NO invented names. 1 to 3 words, just the entity.
Produce exactly 3 or 4 beats, all "shortable": true, totalling 70-110 words.
"""


def _goal_script(event: GoalEvent) -> Script:
    """Generate the goal-news script via Groq, falling back to a deterministic
    template so a Groq outage can never stall a live goal. All beats are forced
    shortable so the slicer always yields the vertical short."""
    from groq import GroqError

    facts = (
        f"Scorer: {event.scorer or 'unknown (name the scoring team instead)'}\n"
        f"Scoring team: {event.scoring_team}\n"
        f"Opponent: {event.conceding_team}\n"
        f"Current score: {event.home} {event.home_score} - "
        f"{event.away_score} {event.away}\n"
        f"Minute: {event.minute or 'unknown'}\n"
        f"Competition: {event.competition}"
    )
    try:
        client = _client()
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL_FAST,
            messages=[{"role": "system", "content": GOAL_SYSTEM_PROMPT},
                      {"role": "user", "content": f"Write the goal short for:\n\n{facts}"}],
            temperature=0.5, max_tokens=700,
            response_format={"type": "json_object"},
        )
        script = _parse_script(resp.choices[0].message.content, event.headline)
        # Force every beat shortable so shorts.slice_shorts always selects them.
        for b in script.beats:
            b.shortable = True
        return script
    except (KeyError, IndexError, TypeError, AttributeError, ValueError, GroqError) as e:
        log.warning("goal-script Groq path failed (%s); using deterministic fallback",
                    type(e).__name__)
        return _fallback_goal_script(event)


def _fallback_goal_script(event: GoalEvent) -> Script:
    who = event.scorer or event.scoring_team
    mins = f" in the {event.minute}th minute" if event.minute else ""
    hook = (f"Goal. {who} scores for {event.scoring_team} against "
            f"{event.conceding_team}{mins}.")
    body = (f"The scoreline is now {event.home} {event.home_score}, "
            f"{event.conceding_team if event.side=='home' else event.scoring_team} "
            f"{event.away_score if event.side=='home' else event.home_score}, "
            f"in the {event.competition}.")
    context = (f"{event.scoring_team} needed this one, and {who} delivered when it "
               "mattered.")
    payoff = (f"Plenty of football still to play: {event.conceding_team} now have "
              "to answer.")
    parts = [(hook, event.scorer or event.scoring_team),
             (body, event.scoring_team),
             (context, who if event.scorer else event.scoring_team),
             (payoff, event.conceding_team)]
    beats = [Beat(narration=n, visual=v, shortable=True) for n, v in parts]
    return Script(title_working=event.headline, beats=beats)


# ── Render + (dry) publish one goal short ─────────────────────────────────────
def produce_goal_short(event: GoalEvent, work_root: Path | None = None) -> dict:
    """Render ONE stills+caption vertical short for a goal and (dry) publish it.

    Reuses the existing chain end to end: Groq script -> edge-tts -> Commons
    stills -> Ken-Burns -> FFmpeg vertical short with captions -> Groq metadata
    -> uploader. Posting is dry unless LIVE_MONITOR_AUTOPOST=1 AND real creds.
    Returns a result dict; raises only on a genuine render failure.
    """
    run_id = time.strftime("%Y%m%d-%H%M%S")
    work = (work_root or config.CACHE / "goalnews") / run_id
    work.mkdir(parents=True, exist_ok=True)
    config.PENDING.mkdir(parents=True, exist_ok=True)

    # Belt and suspenders: if the operator has not explicitly opted into
    # auto-posting, force dry-run for THIS process so nothing can reach YouTube.
    if not _autopost_enabled():
        os.environ["YT_DRY_RUN"] = "1"

    script = _goal_script(event)
    log.info("Goal short: %r -> %d beats", event.headline, len(script.beats))

    with _stills_only():  # legal guard: Commons stills only, never video footage
        doc = composer.build_doc(script, work)
        short_paths = shorts.slice_shorts(doc, work)
    if not short_paths:
        raise RuntimeError("slicer produced no short for goal event "
                           f"{event.event_id!r} (no shortable window)")
    short = Path(short_paths[0])

    # metadata.make_meta builds title/description/tags via Groq; it raises only
    # when GROQ_API_KEY is absent (the daily pipeline always has it via Doppler).
    # A live goal must still ship, so fall back to the module's deterministic
    # metadata rather than letting a missing key sink the short.
    try:
        smeta = metadata.make_meta(event.headline, "short")
    except ValueError as e:
        log.warning("metadata Groq path unavailable (%s); using deterministic meta", e)
        smeta = metadata._fallback_meta(event.headline, "short")
    final_short = config.PENDING / f"{run_id}-goal-{event.match_id}.mp4"
    final_short.write_bytes(short.read_bytes())

    dry = uploader.is_dry()
    vid = uploader.upload(final_short, smeta.title, smeta.description, smeta.tags)
    state.log_published({"asset": final_short.name, "kind": "goal-short",
                         "video_id": vid, "title": smeta.title,
                         "topic": event.headline, "match_id": event.match_id})

    run_metrics.write(
        mode="goal-news",
        status="ok",
        summary=(f"{'[dry] ' if dry else ''}{event.headline[:80]}")[:160],
        metrics={"event": event.event_id, "scorer": event.scorer,
                 "scoring_team": event.scoring_team, "minute": event.minute,
                 "beats": len(script.beats), "dry_run": dry,
                 "stills_only": True, "video_id": vid},
        budgets={"football_data": {"used": None, "limit": 10,
                                   "note": "free tier ~10 req/min; poll every 30-60s"}},
    )
    log.info("Goal short staged: %s (%s)%s", final_short.name, vid,
             "" if dry else " UPLOADED")
    return {"event_id": event.event_id, "headline": event.headline,
            "short_path": str(final_short), "video_id": vid, "dry_run": dry}


# ── Checkpoint state ──────────────────────────────────────────────────────────
def _load_state() -> dict:
    if _STATE_PATH.exists():
        try:
            data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            data.setdefault("scores", {})
            data.setdefault("fired", [])
            return data
        except (ValueError, OSError):
            pass
    return {"scores": {}, "fired": []}


def _save_state(st: dict) -> None:
    try:
        _STATE_PATH.write_text(json.dumps(st, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("could not checkpoint live_state.json: %s", e)


# ── Live watch loop ───────────────────────────────────────────────────────────
def watch(once: bool = False, interval: int = 45, status: str = "LIVE",
          competitions: Iterable[str] | None = None) -> list[dict]:
    """Poll the live endpoint on an interval, render a short for each new goal.

    `once` does a single poll (useful for a cron tick); otherwise loops forever.
    State is checkpointed after every tick so a kill/restart resumes cleanly and
    never re-fires a goal it already handled.
    """
    token = os.environ.get("FOOTBALL_DATA_TOKEN", "").strip()
    st = _load_state()
    fired: set[str] = set(st.get("fired", []))
    results: list[dict] = []

    while True:
        try:
            records = poll_live(token, status=status, competitions=competitions)
            events = detect_goals(records, st["scores"])
            for ev in events:
                if ev.event_id in fired:
                    continue
                try:
                    results.append(produce_goal_short(ev))
                except Exception as e:  # one bad render must not kill the watcher
                    log.error("render failed for %s: %s: %s",
                              ev.event_id, type(e).__name__, e)
                fired.add(ev.event_id)
            st["fired"] = sorted(fired)
            _save_state(st)
        except (PermissionError, ValueError) as e:
            # Credential/whitelist problems are fatal: surface and stop.
            run_metrics.write(mode="goal-news", status="error",
                              summary=f"watch aborted: {type(e).__name__}: {e}"[:160])
            raise
        if once:
            return results
        time.sleep(max(20, interval))  # respect free-tier rate limits


# ── Fixture replay (the offline proof) ────────────────────────────────────────
def replay_fixture(fixture_path: str | Path, render: bool = True) -> list[dict]:
    """Replay a recorded sequence of poll snapshots to prove the detector fires
    and (optionally) the render chain produces a short while posting NOTHING.

    The fixture is JSON: {"snapshots": [[<match>, ...], ...]} where each match
    mirrors a football-data.org v4 match object. Detection runs exactly as it
    does live; only the network poll is replaced by the recorded snapshots.

    Forces YT_DRY_RUN=1 for the whole replay: a test harness must never post.
    """
    os.environ["YT_DRY_RUN"] = "1"  # hard guarantee: replay uploads nothing
    fixture_path = Path(fixture_path)
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    snapshots = data.get("snapshots", [])

    scores: dict[str, list[int]] = {}
    results: list[dict] = []
    for snap in snapshots:
        records = [r for r in (_extract(m) for m in snap) if r is not None]
        for ev in detect_goals(records, scores):
            log.info("REPLAY goal: %s", ev.headline)
            if render:
                results.append(produce_goal_short(ev))
            else:
                results.append({"event_id": ev.event_id, "headline": ev.headline,
                                "rendered": False})
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", action="store_true", help="poll the live API")
    ap.add_argument("--once", action="store_true", help="single poll then exit")
    ap.add_argument("--interval", type=int, default=45, help="seconds between polls")
    ap.add_argument("--status", default="LIVE", help="match status filter")
    ap.add_argument("--competitions", default="",
                    help="comma-separated competition codes (e.g. WC,CL)")
    ap.add_argument("--replay", metavar="FIXTURE.json",
                    help="replay a fixture instead of polling (renders, posts nothing)")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    comps = [c for c in args.competitions.split(",") if c.strip()]
    if args.replay:
        res = replay_fixture(args.replay)
        log.info("REPLAY DONE: %d goal short(s) rendered, 0 posted", len(res))
        return
    if args.watch:
        res = watch(once=args.once, interval=args.interval, status=args.status,
                    competitions=comps)
        log.info("WATCH DONE: %d goal short(s) handled", len(res))
        return
    ap.error("pass --watch (optionally --once) or --replay FIXTURE.json")


if __name__ == "__main__":
    main()
