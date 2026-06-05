"""Groq-generated YouTube metadata: a click-worthy title, a description with
the legal CC-attribution note, and tags. Same prompt-caching discipline:
static system block first, variable topic in the user message.
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

SYSTEM_PROMPT = """You write YouTube metadata for a faceless football
documentary channel. Given a topic and whether the asset is a long-form doc or
a short, return STRICT JSON (no fences):
{"title": "...", "description": "...", "tags": ["...", ...]}

Title rules: under 70 chars, curiosity-driven, lead with the famous name or a
number, no clickbait lies, no emojis in docs (one allowed in shorts titles).
For shorts, append " #shorts" to the title.
Description: 2-3 sentences that restate the hook, then a blank line, then
3-5 hashtags. No em dashes anywhere.
Tags: 8-12 lowercase search phrases.
"""


@dataclass
class Meta:
    title: str
    description: str
    tags: list[str]


def _client():
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise ValueError("GROQ_API_KEY not set (provide via Doppler).")
    return Groq(api_key=key)


_ATTRIB = ("\n\nImagery: Creative Commons / public domain via Wikimedia Commons. "
           "Narration and edit are original. Not affiliated with FIFA.")


def make_meta(topic_title: str, kind: str) -> Meta:
    client = _client()
    user = f"Asset kind: {kind}\nTopic: {topic_title}"
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL_FAST,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
        temperature=0.7, max_tokens=600,
        response_format={"type": "json_object"},
    )
    raw = re.sub(r"^```(?:json)?|```$", "", resp.choices[0].message.content.strip()).strip()
    data = json.loads(raw)
    tags = list(dict.fromkeys([*data.get("tags", []), *config.YT_DEFAULT_TAGS]))[:15]
    return Meta(title=data["title"][:95],
                description=data["description"] + _ATTRIB,
                tags=tags)
