"""Groq scriptwriter. Produces the long-form documentary script as ordered
beats, each beat being one narrated paragraph + a visual cue. The shorts
slicer later reuses the most self-contained beats as standalone shorts.

Hook discipline (from the Money Guy transcript): every script opens with a
1-sentence hook that is a question tied to a famous name, then a supporting
hook that defeats "I already know this", then the body with no fluff, then a
payoff that answers the opening question.

Prompt caching (standing rule): the big static SYSTEM block is sent FIRST and
never interpolates variables, so Groq's prefix cache hits across every call.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

# ── Static system prompt FIRST (cache prefix) — no f-strings here. ─────────────
SYSTEM_PROMPT = """You are the head writer for a faceless football documentary
YouTube channel. You write tight, factual, emotionally compelling narration
about real football history: scandals, money, underdog rises, national-team
stories, and explainers for new fans.

Hard rules:
- Open with ONE hook sentence: a question tied to a famous name or number that
  the viewer genuinely wants answered.
- Follow with ONE supporting-hook sentence that makes them doubt they know the
  answer, so they keep watching.
- Body: chronological, specific, no filler, no "in this video", no "let's dive
  in". Every sentence earns its place.
- End by paying off the opening question explicitly.
- Only state facts a reasonable football fan would accept as true. Never invent
  quotes, scores, or statistics. If unsure, stay general.
- Narration only. No stage directions inside the narration text.
- No em dashes anywhere. Use commas, colons, or parentheses.

Return STRICT JSON, no markdown fences, with this shape:
{
  "title_working": "...",
  "beats": [
    {"narration": "one paragraph of spoken narration",
     "visual": "Real Proper Noun(s) only",
     "shortable": true},
    ...
  ]
}

VISUAL FIELD RULES (critical — these become a Wikimedia Commons image search):
- The visual MUST be the NAME of a real, photographable subject that plausibly
  has a Wikipedia/Commons photo: a real footballer, manager, national team,
  club, stadium, trophy, or city. Examples: "Luka Modric", "Croatia national
  football team", "Ballon d'Or trophy", "Maracana Stadium".
- NO scenes, emotions, actions, or descriptions ("a player looking sad",
  "celebrating a goal", "young boy with a ball"). Those return junk images.
- NO invented names. Use only real people/places that actually exist.
- 1 to 3 words, just the entity. If a beat is general, name the most relevant
  real entity in the story (e.g. the main subject of the documentary).

The first beat MUST be the hook + supporting hook. Mark a beat "shortable": true
only if it stands alone as a 25-50s clip with its own mini hook and payoff.
Produce 9 to 14 beats totalling roughly 800-1100 spoken words.
"""


@dataclass
class Beat:
    narration: str
    visual: str
    shortable: bool = False


@dataclass
class Script:
    title_working: str
    beats: list[Beat]

    @property
    def full_narration(self) -> str:
        return " ".join(b.narration for b in self.beats)


def _client():
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise ValueError("GROQ_API_KEY not set (provide via Doppler).")
    return Groq(api_key=key)


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def _repair_json(text: str) -> str:
    """Trivial repairs for almost-valid model JSON: drop trailing commas before
    a closing brace/bracket and strip stray control characters. Best-effort only;
    the retry + deterministic fallback are the real safety net."""
    text = re.sub(r",\s*([}\]])", r"\1", text)          # trailing commas
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # control chars
    return text


def _parse_script(raw: str, topic_title: str) -> Script:
    """Parse model output into a Script, repairing trivially-broken JSON first."""
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except ValueError:
        data = json.loads(_repair_json(cleaned))   # may still raise -> caller retries
    raw_beats = data["beats"]
    if not isinstance(raw_beats, list):
        raise ValueError("'beats' is not a list")
    beats: list[Beat] = []
    for b in raw_beats:
        if not isinstance(b, dict):
            continue
        # Coerce defensively: the model occasionally emits non-string fields
        # (null, numbers) which would otherwise blow up .strip() with an
        # AttributeError and escape the retry loop.
        narration = str(b.get("narration") or "").strip()
        if not narration:
            continue
        visual = str(b.get("visual") or topic_title).strip() or topic_title
        beats.append(Beat(narration=narration, visual=visual,
                          shortable=bool(b.get("shortable", False))))
    if not beats:
        raise ValueError("model returned no usable beats")
    return Script(title_working=data.get("title_working", topic_title).strip(),
                  beats=beats)


def _fallback_script(topic_title: str) -> Script:
    """Deterministic script when Groq can't produce valid JSON. A plain factual
    narration beats a dead pipeline: it renders, slices, and uploads like any
    other doc. The visual on every beat is the topic itself, which is a real
    illustratable entity (topics.py only picks Commons-resolvable subjects)."""
    hook = (f"What really happened with {topic_title}? "
            "The story is bigger than the headline, and most fans only know half of it.")
    body = (f"{topic_title} sits at the crossroads of money, power, and football. "
            "The facts, laid out in order, tell a story that is stranger than the rumor.")
    payoff = (f"That is the real story behind {topic_title}: "
              "follow the money, and the football makes sense.")
    parts = [(hook, True), (body, True), (payoff, False)]
    beats = [Beat(narration=n, visual=topic_title, shortable=s) for n, s in parts]
    return Script(title_working=topic_title, beats=beats)


def write_script(topic_title: str) -> Script:
    """Generate the documentary script for one topic.

    Groq's json_object mode validates AFTER generation and 400s on malformed
    JSON (json_validate_failed), and the API can also throw transient 429/5xx.
    Either would kill the whole cycle, so we retry colder each attempt, repair
    trivially-broken JSON, and finally fall back to a deterministic template.
    """
    from groq import GroqError
    client = _client()
    # Variable content goes in the USER message, AFTER the cached system prefix.
    user = f"Write the documentary script for this topic:\n\n{topic_title}"
    for attempt, temp in enumerate((0.8, 0.5, 0.2)):
        try:
            resp = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=temp,
                max_tokens=2600,
                response_format={"type": "json_object"},
            )
            return _parse_script(resp.choices[0].message.content, topic_title)
        except (KeyError, IndexError, TypeError, AttributeError, ValueError, GroqError) as e:
            # ValueError covers json.JSONDecodeError; GroqError covers the
            # json_validate_failed 400, 429 rate limits, and 5xx/timeouts;
            # the rest catch malformed-but-200 payloads.
            print(f"[scriptwriter] attempt {attempt + 1} failed (temp={temp}): {e}",
                  file=sys.stderr)
    print("[scriptwriter] all Groq attempts failed; using deterministic fallback script",
          file=sys.stderr)
    return _fallback_script(topic_title)


if __name__ == "__main__":
    s = write_script("The stolen Jules Rimet trophy heist")
    print("TITLE:", s.title_working)
    for i, b in enumerate(s.beats):
        flag = " [SHORT]" if b.shortable else ""
        print(f"\n[{i}]{flag} visual={b.visual!r}\n{b.narration}")
