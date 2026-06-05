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


def write_script(topic_title: str) -> Script:
    """Generate the documentary script for one topic."""
    client = _client()
    # Variable content goes in the USER message, AFTER the cached system prefix.
    user = f"Write the documentary script for this topic:\n\n{topic_title}"
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.8,
        max_tokens=2600,
        response_format={"type": "json_object"},
    )
    data = json.loads(_strip_fences(resp.choices[0].message.content))
    beats = [
        Beat(
            narration=b["narration"].strip(),
            visual=b.get("visual", topic_title).strip(),
            shortable=bool(b.get("shortable", False)),
        )
        for b in data["beats"]
        if b.get("narration", "").strip()
    ]
    return Script(title_working=data.get("title_working", topic_title).strip(),
                  beats=beats)


if __name__ == "__main__":
    s = write_script("The stolen Jules Rimet trophy heist")
    print("TITLE:", s.title_working)
    for i, b in enumerate(s.beats):
        flag = " [SHORT]" if b.shortable else ""
        print(f"\n[{i}]{flag} visual={b.visual!r}\n{b.narration}")
