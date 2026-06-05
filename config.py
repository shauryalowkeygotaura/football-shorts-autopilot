"""Central config for football-shorts-autopilot.

Single source of truth for the niche, cadence, and voice so the whole
pipeline can be retargeted (e.g. to a basketball channel) by editing this
file alone. Secrets live in Doppler, NOT here.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.resolve()
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"
PENDING = OUTPUT / "pending"   # generated-but-not-yet-uploaded (autonomous mode skips review)

# ── Niche identity ────────────────────────────────────────────────────────────
# Locked sub-niche per the research synthesis: money/scandal/underdog STORY docs.
# Highest RPM + views, scriptable from public facts, low copyright risk (no live
# match clips — we narrate over CC-licensed stills).
CHANNEL_NAME = "World Cup Vault"
SUB_NICHE = "football money / scandal / underdog stories"

# The five money angles, ranked by the synthesis (RPM x view potential).
ANGLES = [
    "money / loophole / scandal / heist",   # highest RPM AND views
    "player origin / underdog rise",
    "country / national-team story",
    "how-football-works explainer (US first-timers)",
    "ranking / prediction / debate",
]

# ── Output format targets ─────────────────────────────────────────────────────
DOC_RESOLUTION = (1920, 1080)     # 16:9 master long-form doc
SHORT_RESOLUTION = (1080, 1920)   # 9:16 vertical short
DOC_TARGET_SEC = (300, 600)       # 5-10 min long-form
SHORT_TARGET_SEC = (25, 55)       # shorts sweet spot
SHORTS_PER_DOC = 4                # auto-slice N shorts from each doc

# ── Voice (edge-tts, free) ────────────────────────────────────────────────────
# Authoritative documentary narrator. en-US for the high-RPM North-American
# audience the tournament unlocks.
VOICE = "en-US-AndrewMultilingualNeural"
VOICE_RATE = "+6%"   # docs read slightly brisk; shorts inherit then get punchier

# ── LLM (Groq, free tier — per standing project rule, never Anthropic) ─────────
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL_FAST = "llama-3.1-8b-instant"

# ── YouTube upload ────────────────────────────────────────────────────────────
YT_CATEGORY_ID = "17"            # Sports
YT_PRIVACY = "public"
YT_DEFAULT_TAGS = ["FIFA World Cup 2026", "football", "soccer", "world cup"]

# ── Image sourcing (copyright-safe, free) ─────────────────────────────────────
# Wikimedia Commons (CC/public-domain). We narrate over stills + Ken-Burns,
# NOT live match footage, to stay transformative + demonetization-safe.
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
IMAGES_PER_DOC = 14              # one fresh visual per ~30-40s of narration
