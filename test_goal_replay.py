"""Offline-ish proof for the goal-news detector (live_monitor.py).

Two layers:
  1. Pure-logic, NO network: py_compile, the score-delta detector, the slug/
     whitelist guards, and the dry-run posting gate. Always runs.
  2. Full fixture replay that RENDERS one short and proves nothing posts. This
     needs edge-tts + Wikimedia Commons reachability (both free, no key); set
     GOAL_REPLAY_RENDER=0 to skip the render leg in an offline CI box.

Run: python test_goal_replay.py
"""
import os
import py_compile
from pathlib import Path

ROOT = Path(__file__).parent

# 1. Compile the new module + the chain it drives.
for f in ["live_monitor.py", "modules/composer.py", "modules/shorts.py",
          "modules/uploader.py"]:
    py_compile.compile(str(ROOT / f), doraise=True)
print("compile OK")

# Force dry-run + ensure no Pexels key leaks into the stills-only path.
os.environ["YT_DRY_RUN"] = "1"
os.environ.pop("PEXELS_API_KEY", None)

import live_monitor as lm  # noqa: E402

# 2. Detector: first sight is a baseline (no fire); a score increment fires once.
scores: dict = {}
snap0 = [lm._extract(m) for m in
         [{"id": 1, "homeTeam": {"name": "France"}, "awayTeam": {"name": "Argentina"},
           "score": {"fullTime": {"home": 0, "away": 0}}}]]
assert lm.detect_goals([r for r in snap0 if r], scores) == [], "baseline must not fire"

snap1 = [lm._extract(m) for m in
         [{"id": 1, "homeTeam": {"name": "France"}, "awayTeam": {"name": "Argentina"},
           "score": {"fullTime": {"home": 1, "away": 0}},
           "goals": [{"team": {"name": "France"}, "scorer": {"name": "Kylian Mbappe"}}]}]]
events = lm.detect_goals([r for r in snap1 if r], scores)
assert len(events) == 1, f"exactly one goal expected, got {len(events)}"
ev = events[0]
assert ev.side == "home" and ev.scoring_team == "France", ev
assert ev.scorer == "Kylian Mbappe", ev.scorer
assert ev.conceding_team == "Argentina", ev
assert ev.headline.startswith("GOAL: Kylian Mbappe"), ev.headline
print("detector OK:", ev.headline)

# Re-detecting the same scoreline does not fire again (idempotent baseline).
assert lm.detect_goals([r for r in snap1 if r], scores) == [], "no re-fire on same score"

# A VAR rollback (score decrease) updates baseline silently, no event.
snap_var = [lm._extract(m) for m in
            [{"id": 1, "homeTeam": {"name": "France"}, "awayTeam": {"name": "Argentina"},
              "score": {"fullTime": {"home": 0, "away": 0}}}]]
assert lm.detect_goals([r for r in snap_var if r], scores) == [], "rollback must not fire"
print("idempotency + rollback OK")

# 3. Whitelist guards (validation, not character-removal).
import requests  # noqa: E402
try:
    lm.poll_live("tok", status="DROP TABLE")  # not in whitelist
    raise AssertionError("bad status should be rejected")
except ValueError:
    pass
try:
    lm.poll_live("tok", competitions=["WC; rm -rf"])  # fails comp regex
    raise AssertionError("bad competition code should be rejected")
except ValueError:
    pass
try:
    lm.poll_live("")  # missing token = loud credential gate
    raise AssertionError("empty token should raise")
except ValueError:
    pass
print("whitelist + credential gate OK")

# 4. Posting gate: default is OFF (dry).
os.environ.pop("LIVE_MONITOR_AUTOPOST", None)
assert lm._autopost_enabled() is False, "auto-post must default OFF for a minor"
print("autopost-default-off OK")

# 5. Full fixture replay (renders a short, posts nothing). Network-dependent.
if os.environ.get("GOAL_REPLAY_RENDER", "1") == "1":
    results = lm.replay_fixture(ROOT / "fixtures" / "goal_replay.json", render=True)
    assert len(results) == 1, f"fixture should render exactly one short, got {len(results)}"
    res = results[0]
    short = Path(res["short_path"])
    assert short.exists() and short.stat().st_size > 1000, f"no short rendered: {short}"
    assert res["video_id"].startswith("DRYRUN"), f"must be a dry-run id, got {res['video_id']}"
    assert res["dry_run"] is True, "replay must never post"
    print(f"replay render OK: {short.name} ({short.stat().st_size} bytes), "
          f"video_id={res['video_id']} (posted nothing)")
else:
    print("replay render SKIPPED (GOAL_REPLAY_RENDER=0)")

print("\nALL GOAL-NEWS CHECKS PASSED")
