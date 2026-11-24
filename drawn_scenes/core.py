"""Code-drawn background scenes that stand in for stock b-roll, at any frame size.

Three looks, ported from the paper-reel, sketch and anatomy agent skills (no image/video model, no browser):
  papercut  cut paper on a coloured wall: flat fills, paper grain, soft drop shadows, back-eased pops
  sketch    soft hand-drawn ink on notebook paper: wobbly strokes, offset watercolour fill, lines that
            "boil" (3 jittered redraws cycled at 7.5 fps), drawn on with a wipe
  anatomy   an isometric bench instrument on a framed dark card: four-tone shading, the icon engraved on a
            plate, a carriage that slides and drives a dial needle and a live readout (opt-in, not in auto)

A scene = a 1-3 word headline + one icon from ICONS + light decor. Portrait frames stack headline over
icon (captions and a presenter live in the lower half); landscape frames put them side by side.

The LLM only ever picks words and an icon NAME from the whitelist, and (in mix mode) "stock" or "drawn"
per line. It never writes drawing code. Bring your own client via `chat=`; the default uses GROQ_API_KEY.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = Path(__file__).resolve().parent / "fonts"
FPS = 30
VERSION = 3            # part of the cache key: bump after changing the look
STYLES = ("papercut", "sketch", "anatomy")
AUTO_STYLES = ("papercut", "sketch")   # 'auto' keeps its old picks; anatomy is opt-in

Chat = Callable[[list, int], str]   # (messages, max_tokens) -> JSON text


def _bgr(hexs: str) -> tuple[int, int, int]:
    h = hexs.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


# ── icons: unit-square primitives, drawn by both styles ──────────────────────
# ("poly", pts, role) closed fill | ("circle", (cx, cy, r), role) | ("rect", (x0, y0, x1, y1), role)
# ("line", pts, width, role) open stroke
def _heart() -> list:
    pts = []
    for k in range(48):
        t = 2 * math.pi * k / 48
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((0.5 + x / 38, 0.47 - y / 38))
    return [("poly", pts, "red")]


def _star(n=8, r0=0.42, r1=0.18) -> list:
    pts = [(0.5 + (r0 if k % 2 == 0 else r1) * math.cos(math.pi * k / n - math.pi / 2),
            0.5 + (r0 if k % 2 == 0 else r1) * math.sin(math.pi * k / n - math.pi / 2)) for k in range(2 * n)]
    return [("poly", pts, "accent")]


def _arc(cx, cy, r, a0, a1, n=18) -> list:
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n))
            for k in range(n + 1)]


ICONS: dict[str, list] = {
    "spark": _star(),
    "bulb": [("circle", (0.5, 0.4, 0.28), "accent"), ("rect", (0.38, 0.62, 0.62, 0.77), "dark"),
             ("rect", (0.42, 0.79, 0.58, 0.86), "dark"), ("line", [(0.42, 0.42), (0.5, 0.52), (0.58, 0.42)], 0.03, "light")],
    "phone": [("rect", (0.28, 0.06, 0.72, 0.94), "dark"), ("rect", (0.32, 0.14, 0.68, 0.8), "light"),
              ("circle", (0.5, 0.87, 0.03), "light"), ("rect", (0.38, 0.24, 0.62, 0.3), "main"),
              ("rect", (0.38, 0.36, 0.56, 0.41), "accent")],
    "laptop": [("rect", (0.15, 0.2, 0.85, 0.66), "dark"), ("rect", (0.2, 0.25, 0.8, 0.61), "light"),
               ("poly", [(0.05, 0.7), (0.95, 0.7), (0.88, 0.8), (0.12, 0.8)], "main"),
               ("rect", (0.27, 0.33, 0.6, 0.38), "accent"), ("rect", (0.27, 0.44, 0.5, 0.49), "main")],
    "clock": [("circle", (0.5, 0.5, 0.42), "dark"), ("circle", (0.5, 0.5, 0.35), "light"),
              ("line", [(0.5, 0.5), (0.5, 0.25)], 0.05, "dark"), ("line", [(0.5, 0.5), (0.67, 0.6)], 0.05, "main")],
    "coin": [("circle", (0.5, 0.5, 0.4), "accent"), ("circle", (0.5, 0.5, 0.31), "main"),
             ("rect", (0.455, 0.3, 0.545, 0.7), "light")],
    "chart_up": [("rect", (0.14, 0.6, 0.3, 0.86), "main"), ("rect", (0.4, 0.45, 0.56, 0.86), "main"),
                 ("rect", (0.66, 0.26, 0.82, 0.86), "main"),
                 ("line", [(0.1, 0.5), (0.38, 0.32), (0.58, 0.38), (0.84, 0.14)], 0.05, "green"),
                 ("poly", [(0.92, 0.06), (0.9, 0.24), (0.76, 0.12)], "green")],
    "chart_down": [("rect", (0.14, 0.26, 0.3, 0.86), "main"), ("rect", (0.4, 0.45, 0.56, 0.86), "main"),
                   ("rect", (0.66, 0.6, 0.82, 0.86), "main"),
                   ("line", [(0.1, 0.14), (0.38, 0.34), (0.58, 0.3), (0.84, 0.52)], 0.05, "red"),
                   ("poly", [(0.92, 0.6), (0.74, 0.58), (0.86, 0.44)], "red")],
    "check": [("circle", (0.5, 0.5, 0.42), "green"), ("line", [(0.3, 0.52), (0.45, 0.66), (0.72, 0.36)], 0.08, "light")],
    "cross": [("circle", (0.5, 0.5, 0.42), "red"), ("line", [(0.33, 0.33), (0.67, 0.67)], 0.08, "light"),
              ("line", [(0.67, 0.33), (0.33, 0.67)], 0.08, "light")],
    "warning": [("poly", [(0.5, 0.08), (0.94, 0.86), (0.06, 0.86)], "accent"),
                ("rect", (0.465, 0.34, 0.535, 0.62), "dark"), ("circle", (0.5, 0.73, 0.045), "dark")],
    "chat": [("poly", [(0.1, 0.16), (0.9, 0.16), (0.9, 0.64), (0.42, 0.64), (0.24, 0.84), (0.27, 0.64), (0.1, 0.64)], "light"),
             ("circle", (0.32, 0.4, 0.05), "dark"), ("circle", (0.5, 0.4, 0.05), "dark"), ("circle", (0.68, 0.4, 0.05), "dark")],
    "code": [("rect", (0.06, 0.14, 0.94, 0.86), "dark"),
             ("line", [(0.36, 0.32), (0.2, 0.5), (0.36, 0.68)], 0.06, "accent"),
             ("line", [(0.64, 0.32), (0.8, 0.5), (0.64, 0.68)], 0.06, "accent"),
             ("line", [(0.56, 0.26), (0.44, 0.74)], 0.06, "light")],
    "lock": [("line", _arc(0.5, 0.44, 0.17, math.pi, 2 * math.pi), 0.07, "dark"),
             ("line", [(0.33, 0.44), (0.33, 0.5)], 0.07, "dark"), ("line", [(0.67, 0.44), (0.67, 0.5)], 0.07, "dark"),
             ("rect", (0.24, 0.48, 0.76, 0.9), "accent"), ("circle", (0.5, 0.64, 0.05), "dark"),
             ("rect", (0.48, 0.64, 0.52, 0.78), "dark")],
    "heart": _heart(),
    "arrow": [("poly", [(0.08, 0.4), (0.56, 0.4), (0.56, 0.2), (0.92, 0.5), (0.56, 0.8), (0.56, 0.6), (0.08, 0.6)], "main")],
    "rocket": [("poly", [(0.5, 0.04), (0.66, 0.3), (0.66, 0.68), (0.34, 0.68), (0.34, 0.3)], "light"),
               ("poly", [(0.34, 0.46), (0.18, 0.72), (0.34, 0.68)], "main"),
               ("poly", [(0.66, 0.46), (0.82, 0.72), (0.66, 0.68)], "main"),
               ("circle", (0.5, 0.34, 0.07), "dark"),
               ("poly", [(0.38, 0.7), (0.62, 0.7), (0.5, 0.96)], "accent")],
    "ball": [("circle", (0.5, 0.5, 0.42), "light"),
             ("poly", [(0.5, 0.34), (0.64, 0.44), (0.59, 0.6), (0.41, 0.6), (0.36, 0.44)], "dark"),
             ("line", [(0.5, 0.34), (0.5, 0.1)], 0.03, "dark"), ("line", [(0.64, 0.44), (0.88, 0.38)], 0.03, "dark"),
             ("line", [(0.36, 0.44), (0.12, 0.38)], 0.03, "dark"), ("line", [(0.59, 0.6), (0.72, 0.84)], 0.03, "dark"),
             ("line", [(0.41, 0.6), (0.28, 0.84)], 0.03, "dark")],
    "trophy": [("poly", [(0.26, 0.12), (0.74, 0.12), (0.68, 0.46), (0.5, 0.58), (0.32, 0.46)], "accent"),
               ("line", _arc(0.26, 0.28, 0.12, math.pi / 2, 3 * math.pi / 2), 0.05, "accent"),
               ("line", _arc(0.74, 0.28, 0.12, -math.pi / 2, math.pi / 2), 0.05, "accent"),
               ("rect", (0.45, 0.56, 0.55, 0.74), "accent"), ("rect", (0.3, 0.74, 0.7, 0.86), "dark")],
    "book": [("poly", [(0.1, 0.22), (0.48, 0.28), (0.48, 0.86), (0.1, 0.8)], "light"),
             ("poly", [(0.52, 0.28), (0.9, 0.22), (0.9, 0.8), (0.52, 0.86)], "light"),
             ("rect", (0.47, 0.26, 0.53, 0.88), "main"),
             ("line", [(0.18, 0.4), (0.4, 0.43)], 0.025, "dark"), ("line", [(0.18, 0.52), (0.4, 0.55)], 0.025, "dark"),
             ("line", [(0.6, 0.43), (0.82, 0.4)], 0.025, "dark"), ("line", [(0.6, 0.55), (0.82, 0.52)], 0.025, "dark")],
    "globe": [("circle", (0.5, 0.5, 0.42), "dark"), ("circle", (0.5, 0.5, 0.36), "green"),
              ("line", [(0.14, 0.5), (0.86, 0.5)], 0.03, "light"),
              ("line", _arc(0.5, 0.5, 0.36, -math.pi / 2, math.pi / 2), 0.03, "light"),
              ("line", [(0.5, 0.14), (0.5, 0.86)], 0.03, "light")],
    "eye": [("poly", [(0.06, 0.5), (0.28, 0.28), (0.5, 0.22), (0.72, 0.28), (0.94, 0.5), (0.72, 0.72), (0.5, 0.78),
                      (0.28, 0.72)], "light"),
            ("circle", (0.5, 0.5, 0.17), "main"), ("circle", (0.5, 0.5, 0.08), "dark")],
}

# keyword -> icon, used when the LLM is down or returns something off the list
KEYWORDS = [
    (r"money|price|cost|paid|pay|rupee|dollar|cash|earn|revenue|million|billion|\$|₹|£|€", "coin"),
    (r"goal|match|football|soccer|striker|kick|pitch|keeper", "ball"),
    (r"nuclear|radioactive|radiation|reactor|plutonium|uranium|fallout|toxic|contamina\w*", "warning"),
    (r"satellite|orbit|space|spacecraft|astronaut|nasa|soviet space", "rocket"),
    (r"trophy|champion|title|cup|final|won|victory", "trophy"),
    (r"code|bug|build|deploy|repo|github|script|api|server", "code"),
    (r"phone|call|ring|dial", "phone"),
    (r"laptop|computer|screen|website|app|dashboard", "laptop"),
    (r"time|hour|minute|second|late|fast|slow|wait|today|year|history|ancient", "clock"),
    (r"idea|think|smart|learn|why|how|mind|wisdom", "bulb"),
    (r"book|read|write|wrote|philosoph\w*|truth|virtue|meaning", "book"),
    (r"world|planet|earth|country|global|nation", "globe"),
    (r"see|look|watch|notice|perceive|observe", "eye"),
    (r"grow|more|up|scale|win|best|better|rise", "chart_up"),
    (r"fail|down|drop|lose|lost|less|worse|fall", "chart_down"),
    (r"safe|secure|password|private|lock|prison|trap", "lock"),
    (r"yes|works|done|right|correct|ship", "check"),
    (r"\bno\b|wrong|broke|broken|never|stop", "cross"),
    (r"warn|careful|risk|danger|mistake|death|die", "warning"),
    (r"talk|chat|message|say|tell|ask|reply|dm", "chat"),
    (r"love|like|heart|fan", "heart"),
    (r"launch|start|go|rocket|boost", "rocket"),
    (r"next|then|so|because|step", "arrow"),
]

STOP = set("""a an the and or but so to of in on at for with from by is are was were be been it its this that
these those i you he she we they my your our their me him her us them do does did not just like really very
can could will would should have has had get got make made what when where who which there here all any some
one about into than then also if as up out over more most""".split())

PAPER = {  # paper-reel palette (references/style.md)
    "main": "#d67555", "accent": "#e6b82a", "dark": "#1c2b56", "light": "#f4f1e8",
    "red": "#d2453a", "green": "#2f9e5b", "card": "#192833", "ink": "#1c2b56",
    "walls": ["#869997", "#2b3b5e", "#583656", "#6c9e8f", "#c9b28a", "#4a4566", "#c79a5b"],
    "scraps": ["#f4f1e8", "#e6b82a", "#d67555", "#95a9a7", "#f2c94c"],
}
SKETCH = {  # soft pastels under dark ink
    "main": "#f2a68a", "accent": "#f6d06b", "dark": "#8fa6c9", "light": "#fffdf7",
    "red": "#ee8a80", "green": "#9fd3a8", "ink": "#2b2a33", "paper": "#f6f0e2", "rule": "#c9d6e8",
}


# ── planning ─────────────────────────────────────────────────────────────────
def clean_word(s: str) -> str:
    """Whitelist (the text is drawn verbatim), max 3 whole words / 22 chars.
    Currency signs the bundled fonts lack become letters."""
    s = re.sub(r"₹\s*", "Rs ", s or "")
    s = re.sub(r"[^A-Za-z0-9 '&%$?!+-]", " ", s)
    out = ""
    for w in s.split()[:3]:
        nxt = f"{out} {w}".strip()
        if len(nxt) > 22:
            break
        out = nxt
    words = out.split()
    while len(words) > 1 and words[-1].lower() in STOP:   # no dangling "a" / "the" after the cap
        words.pop()
    return " ".join(words) or s.strip()[:22]


def guess_icon(text: str) -> str:
    t = (text or "").lower()
    for pat, icon in KEYWORDS:
        # whole words + common suffixes (not substring): "go" must not match "good", "up" not "update";
        # a lookbehind instead of \b so "$" and "₹" still match after a space
        if re.search(rf"(?<![a-z0-9])(?:{pat})(?:s|es|ed|ing|er|ers|ly)?(?![a-z])", t):
            return icon
    return "spark"


def fallback_spec(line: str) -> dict:
    words = [w for w in re.findall(r"[A-Za-z0-9']+", line or "") if w.lower() not in STOP]
    best = sorted(words, key=len, reverse=True)[:2]
    keep = [w for w in words if w in best][:2]       # keep spoken order
    return {"word": clean_word(" ".join(keep)) or "listen", "icon": guess_icon(line), "use": None}


def _groq_keys() -> list[str]:
    """The whole pool: GROQ_API_KEYS (comma list) + GROQ_API_KEY + GROQ_API_KEY_2, deduped, in order."""
    raw = os.environ.get("GROQ_API_KEYS", "").split(",") + [os.environ.get("GROQ_API_KEY", ""),
                                                             os.environ.get("GROQ_API_KEY_2", "")]
    out: list[str] = []
    for k in (x.strip() for x in raw):
        if k and k not in out:
            out.append(k)
    return out


def groq_chat(messages: list, max_tokens: int) -> str:
    """Default client: Groq SDK, rotating through the key pool on rate limits / auth errors."""
    from groq import Groq  # noqa: PLC0415
    keys = _groq_keys()
    if not keys:
        raise RuntimeError("no GROQ_API_KEY / GROQ_API_KEYS set")
    last: Exception | None = None
    for key in keys:
        try:
            r = Groq(api_key=key, max_retries=1).chat.completions.create(
                model=os.environ.get("DRAWN_SCENES_MODEL", "openai/gpt-oss-120b"), messages=messages,
                max_tokens=max_tokens, response_format={"type": "json_object"})
            content = r.choices[0].message.content
            if content:
                return content
            last = RuntimeError("empty LLM reply")
        except Exception as e:  # noqa: BLE001 - 429 / 401 on one key: try the next
            last = e
    raise RuntimeError(f"all {len(keys)} Groq keys failed: {str(last)[:120]}")


def _decide_rules(stock: str) -> str:
    return (
        f"Also decide 'use' for each line, picking whichever would look BEST behind it. \"stock\" = {stock}. "
        "That covers more than it seems: people and their reactions, everyday places and objects, real events. "
        "Real imagery feels human and is the default for story moments. \"drawn\" = a designed paper/sketch card "
        "with a short headline and an icon, better when the line's point is a number, a price, a short verdict "
        "or punchline, a lesson, a list, a call to action, or an idea real imagery could only show loosely. "
        "There is no quota and no fixed pattern; judge each line. But texture matters: where a line could go "
        "either way, choose the one that breaks a run of 3 or more of the same kind.")


def plan(lines: list[str], *, decide: bool = False, video: str = "a fast vertical reel",
         stock: str = "real stock footage or photos of what is said", chat: Chat | None = None) -> list[dict]:
    """One LLM call for every line: [{word, icon, use}], use = 'stock' | 'drawn' (decide=True) or None.
    Fails soft to keyword rules with use=None; use decide()/alternate() for the per-shot choice."""
    specs = [fallback_spec(x) for x in lines]
    if not lines:
        return specs
    chat = chat or groq_chat
    shape = '{"i": 0, "word": "...", "icon": "..."' + (', "use": "stock|drawn"}' if decide else "}")
    system = (f"You design the background for each numbered line of {video}. For each line give 'word': the "
              "1-3 word punch of the line, the phrase a viewer remembers (no filler, no full sentence), and "
              f"'icon': the ONE icon from this list that shows the idea: {sorted(ICONS)}. "
              + (_decide_rules(stock) + " " if decide else "")
              + f'Reply JSON {{"scenes": [{shape}]}}')
    try:
        raw = chat([{"role": "system", "content": system},
                    {"role": "user", "content": "\n".join(f"{i}: {x}" for i, x in enumerate(lines))}],
                   400 + 60 * len(lines))
        data = json.loads(raw)
        for s in data.get("scenes", []) if isinstance(data, dict) else []:
            if not isinstance(s, dict):
                continue
            i = s.get("i")
            if not isinstance(i, int) or not 0 <= i < len(lines):
                continue
            word = clean_word(s.get("word") if isinstance(s.get("word"), str) else "")
            icon = s.get("icon") if s.get("icon") in ICONS else guess_icon(lines[i])
            use = s.get("use") if decide and s.get("use") in ("stock", "drawn") else None
            specs[i] = {"word": word or specs[i]["word"], "icon": icon, "use": use}
    except Exception as e:  # noqa: BLE001 - cards still get drawn from the keyword rules
        print(f"  drawn-scenes: LLM plan unavailable ({str(e)[:90]}), using keyword rules", file=sys.stderr)
    return specs


def alternate(n: int, same_as_prev: list[bool] | None = None, keep_first: bool = True) -> set[int]:
    """Fixed fallback: every other LINE drawn (the first stays real when keep_first),
    shots that continue the previous line follow it."""
    same_as_prev = same_as_prev or [False] * n
    out: set[int] = set()
    k = 0
    for i in range(n):
        if i > 0 and same_as_prev[i]:
            if i - 1 in out:
                out.add(i)
            continue
        if (k % 2 == 1) if keep_first else (k % 2 == 0):
            out.add(i)
        k += 1
    return out


def decide(specs: list[dict], same_as_prev: list[bool] | None = None) -> set[int]:
    """Mix mode: the LLM's per-line call (spec['use']), no quota. Continuation shots follow their line.
    Lines the LLM didn't answer fall back to alternate(), so one bad reply can't empty the mix."""
    n = len(specs)
    same_as_prev = same_as_prev or [False] * n
    fb = alternate(n, same_as_prev)
    out: set[int] = set()
    for i, s in enumerate(specs):
        if i > 0 and same_as_prev[i]:
            if i - 1 in out:
                out.add(i)
            continue
        use = s.get("use")
        if use == "drawn" or (use is None and i in fb):
            out.add(i)
    return out


def choose_style(style: str, seed_text: str) -> str:
    """'auto' = one look per video (stable for re-renders), never mixed inside a video."""
    if style in STYLES:
        return style
    return AUTO_STYLES[int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest(), 16) % len(AUTO_STYLES)]


# ── drawing helpers ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Layout:
    w: int
    h: int

    @property
    def k(self) -> float:          # sizes are authored at 1080 short side
        return min(self.w, self.h) / 1080

    @property
    def portrait(self) -> bool:
        return self.h >= self.w

    @property
    def sign(self) -> tuple[float, float]:
        return (self.w / 2, self.h * 0.156) if self.portrait else (self.w * 0.31, self.h * 0.44)

    @property
    def icon(self) -> tuple[float, float]:
        return (self.w / 2, self.h * 0.417) if self.portrait else (self.w * 0.73, self.h * 0.5)

    @property
    def icon_size(self) -> int:
        return int(self.w * 0.556) if self.portrait else int(self.h * 0.6)

    @property
    def title_max(self) -> int:
        return int(self.w * (0.72 if self.portrait else 0.44))

    @property
    def floor(self) -> int:
        return int(self.h * (0.79 if self.portrait else 0.86))

    def clear_of_hero(self, x: float, y: float, margin: float) -> bool:
        ix, iy = self.icon
        s = self.icon_size / 2 + margin
        sx, sy = self.sign
        return not (abs(x - ix) < s and abs(y - iy) < s) and \
            not (abs(x - sx) < self.title_max / 2 + margin and abs(y - sy) < 150 * self.k + margin)


def _noise(h: int, w: int, seed: int, octaves=((6, 1.0), (24, 0.5), (96, 0.3), (300, 0.15))) -> np.ndarray:
    """Multi-octave value noise in [-1, 1]: the paper grain."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    for cells, amp in octaves:
        g = rng.random((max(2, cells * h // max(w, 1)), cells)).astype(np.float32)
        acc += cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC) * amp
    acc -= acc.mean()
    return acc / (np.abs(acc).max() + 1e-6)


def _vignette(w: int, h: int, strength: float = 0.22) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h * 0.45) / (h * 0.6)) ** 2)
    return (1 - strength * np.clip(d - 0.35, 0, 1)).astype(np.float32)[..., None]


def _to_px(prim, S: int):
    """Primitive -> (closed?, points in px, stroke width px, role)."""
    kind = prim[0]
    if kind == "poly":
        return True, [(x * S, y * S) for x, y in prim[1]], 0, prim[2]
    if kind == "circle":
        cx, cy, r = prim[1]
        n = max(24, int(r * S / 4))
        return True, [((cx + r * math.cos(2 * math.pi * k / n)) * S, (cy + r * math.sin(2 * math.pi * k / n)) * S)
                      for k in range(n)], 0, prim[2]
    if kind == "rect":
        x0, y0, x1, y1 = prim[1]
        return True, [(x0 * S, y0 * S), (x1 * S, y0 * S), (x1 * S, y1 * S), (x0 * S, y1 * S)], 0, prim[2]
    return False, [(x * S, y * S) for x, y in prim[1]], max(2, prim[2] * S), prim[3]


def _mask(closed: bool, pts, width: float, size: tuple[int, int], ss: int = 2) -> np.ndarray:
    """Antialiased float mask (supersampled PIL draw)."""
    w, h = size
    im = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(im)
    p = [(x * ss, y * ss) for x, y in pts]
    if closed:
        d.polygon(p, fill=255)
    else:
        lw = max(1, int(width * ss))
        d.line(p, fill=255, width=lw, joint="curve")
        for x, y in (p[0], p[-1]):  # round caps
            d.ellipse((x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2), fill=255)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255.0


def _subdivide(pts, closed: bool, step: float = 14.0):
    out = []
    seq = list(pts) + ([pts[0]] if closed else [])
    for (x0, y0), (x1, y1) in zip(seq, seq[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        out += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n)]
    out.append(seq[-1])
    return out


def _jitter(pts, rng: np.random.Generator, amt: float):
    """Smooth hand wobble: low-frequency offsets along the path, not per-point noise."""
    n = len(pts)
    if n < 2:
        return pts
    knots = rng.normal(0, amt, (max(2, n // 6 + 2), 2))
    xs = np.linspace(0, len(knots) - 1, n)
    off = np.stack([np.interp(xs, np.arange(len(knots)), knots[:, k]) for k in range(2)], 1)
    return [(x + dx, y + dy) for (x, y), (dx, dy) in zip(pts, off)]


class Sprite:
    """Premultiplied BGR + alpha, float32, blitted with scale/rotation about its centre."""

    def __init__(self, rgb: np.ndarray, a: np.ndarray):
        self.rgb, self.a = rgb, a

    @classmethod
    def empty(cls, w: int, h: int) -> "Sprite":
        return cls(np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32))

    def over(self, color, m: np.ndarray, tex: np.ndarray | None = None, tex_amt: float = 0.0):
        col = np.asarray(color, np.float32)[None, None, :] * np.ones_like(self.rgb)
        if tex is not None:
            col *= 1 + tex_amt * tex[..., None]
        m3 = m[..., None]
        self.rgb = self.rgb * (1 - m3) + col * m3
        self.a = self.a * (1 - m) + m

    def shadow(self, m: np.ndarray, dx: int, dy: int, blur: float, amt: float):
        sh = np.roll(np.roll(m, dy, 0), dx, 1)
        sh = cv2.GaussianBlur(sh, (0, 0), blur) * amt
        self.rgb *= (1 - sh)[..., None]
        self.a = self.a + sh * (1 - self.a)

    def blit(self, canvas: np.ndarray, cx: float, cy: float, scale: float = 1.0, angle: float = 0.0,
             reveal: float | None = None):
        if scale <= 0.01:
            return
        H, W = canvas.shape[:2]
        h, w = self.a.shape
        side = int(math.ceil(math.hypot(w, h) * scale)) + 4
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += side / 2 - w / 2
        M[1, 2] += side / 2 - h / 2
        rgb = cv2.warpAffine(self.rgb, M, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
        a = cv2.warpAffine(self.a, M, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
        if reveal is not None and reveal < 1:   # drawn-on wipe, left to right with a soft edge
            ramp = np.clip((reveal * (side + 120) - np.arange(side)) / 120, 0, 1).astype(np.float32)
            rgb *= ramp[None, :, None]
            a *= ramp[None, :]
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(W, x0 + side), min(H, y0 + side)
        if dx1 <= dx0 or dy1 <= dy0:
            return
        sub_a = a[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0][..., None]
        sub_rgb = rgb[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
        roi = canvas[dy0:dy1, dx0:dx1]
        roi *= 1 - sub_a
        roi += sub_rgb


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


def _fit_font(text: str, name: str, max_w: int, start: int, lo: int = 40) -> ImageFont.FreeTypeFont:
    size = start
    while size > lo:
        f = _font(name, size)
        if f.getbbox(text)[2] - f.getbbox(text)[0] <= max_w:
            return f
        size -= 6
    return _font(name, lo)


def _text_mask(text: str, font: ImageFont.FreeTypeFont, pad: int = 10) -> np.ndarray:
    l, t, r, b = font.getbbox(text)
    im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=font, fill=255)
    return np.asarray(im, np.float32) / 255.0


def _ease_back(u: float, s: float = 1.9) -> float:
    u = min(1.0, max(0.0, u)) - 1
    return u * u * ((s + 1) * u + s) + 1


def _scatter(L: Layout, rng: np.random.Generator, n: int, margin: float):
    pts = []
    for _ in range(n * 6):
        if len(pts) >= n:
            break
        x = float(rng.uniform(60 * L.k, L.w - 60 * L.k))
        y = float(rng.uniform(100 * L.k, L.floor - 80 * L.k))
        if L.clear_of_hero(x, y, margin):
            pts.append((x, y))
    return pts


# ── papercut ─────────────────────────────────────────────────────────────────
def _paper_icon(icon: str, S: int, tex: np.ndarray, k: float) -> Sprite:
    pad = int(40 * k)
    sp = Sprite.empty(S + 2 * pad, S + 2 * pad)
    for prim in ICONS[icon]:
        closed, pts, width, role = _to_px(prim, S)
        pts = [(x + pad, y + pad) for x, y in pts]
        m = _mask(closed, pts, width, (S + 2 * pad, S + 2 * pad))
        sp.shadow(m, int(6 * k), int(10 * k), 9 * k, 0.42)
        sp.over(_bgr(PAPER[role]), m, tex[: S + 2 * pad, : S + 2 * pad], 0.10)
    return sp


def _paper_sign(word: str, L: Layout, rng: np.random.Generator) -> Sprite:
    k = L.k
    text = word.upper()
    font = _fit_font(text, "ArchivoBlack-Regular.ttf", L.title_max - int(80 * k), int(132 * k), int(40 * k))
    tm = _text_mask(text, font)
    th, tw = tm.shape
    cw, ch = tw + int(90 * k), th + int(70 * k)
    off, pad = int(16 * k), int(40 * k)
    sw, sh = cw + off + 2 * pad, ch + off + 2 * pad
    sp = Sprite.empty(sw, sh)
    tex = _noise(sh, sw, int(rng.integers(1 << 30)), ((4, 1.0), (40, 0.5), (200, 0.3)))
    card = np.zeros((sh, sw), np.float32)
    cv2.rectangle(card, (pad + off, pad), (pad + off + cw, pad + ch), 1.0, -1)
    sp.shadow(card, int(6 * k), int(10 * k), 10 * k, 0.4)
    sp.over(_bgr(PAPER["card"]), card, tex, 0.08)
    paper = np.zeros((sh, sw), np.float32)
    cv2.rectangle(paper, (pad, pad + off), (pad + cw, pad + off + ch), 1.0, -1)
    sp.shadow(paper, int(5 * k), int(9 * k), 9 * k, 0.38)
    sp.over(_bgr(PAPER["light"]), paper, tex, 0.07)
    tx, ty = pad + int(45 * k), pad + off + int(35 * k)
    big = np.zeros((sh, sw), np.float32)
    big[ty:ty + th, tx:tx + tw] = tm
    sp.over(_bgr(PAPER["ink"]), big)
    return sp


def _paper_scrap(rng: np.random.Generator, tex: np.ndarray, k: float) -> Sprite:
    s = int(rng.integers(50, 110) * k)
    pad = int(24 * k)
    sp = Sprite.empty(s + 2 * pad, s + 2 * pad)
    kind = rng.integers(3)
    if kind == 0:
        pts = [(pad + s / 2 + s / 2 * math.cos(a), pad + s / 2 + s / 2 * math.sin(a))
               for a in np.linspace(0, 2 * math.pi, 28, endpoint=False)]
    elif kind == 1:
        pts = [(pad, pad + s), (pad + s / 2, pad), (pad + s, pad + s)]
    else:
        pts = [(pad, pad + s * 0.3), (pad + s, pad), (pad + s, pad + s * 0.7), (pad, pad + s)]
    m = _mask(True, pts, 0, (s + 2 * pad, s + 2 * pad))
    sp.shadow(m, int(4 * k), int(7 * k), 6 * k, 0.35)
    col = PAPER["scraps"][int(rng.integers(len(PAPER["scraps"])))]
    sp.over(_bgr(col), m, tex[: s + 2 * pad, : s + 2 * pad], 0.1)
    return sp


def _render_papercut(spec: dict, seconds: float, seed: int, L: Layout, sink) -> None:
    rng = np.random.default_rng(seed)
    k = L.k
    wall = _bgr(PAPER["walls"][seed % len(PAPER["walls"])])
    bg = np.ones((L.h, L.w, 3), np.float32) * np.asarray(wall, np.float32)
    bg *= (1 + 0.07 * _noise(L.h, L.w, seed))[..., None]
    bg[L.floor:] *= 0.82                          # a darker paper strip along the bottom
    bg[L.floor:L.floor + max(2, int(6 * k))] *= 0.7
    vig = _vignette(L.w, L.h)
    bare = bool(spec.get("notitle"))   # no headline: the icon takes the centre
    S = int(min(L.w, L.h) * 0.72) if bare else L.icon_size
    tex = _noise(S + int(80 * k), S + int(80 * k), seed + 1, ((5, 1.0), (40, 0.5), (220, 0.3)))
    icon = _paper_icon(spec["icon"], S, tex, k)
    sign = None if spec.get("notitle") else _paper_sign(spec["word"], L, rng)
    scraps = [(_paper_scrap(rng, tex, k), x, y, float(rng.uniform(0, 6.28)), float(rng.uniform(-25, 25)))
              for x, y in _scatter(L, rng, 7, 60 * k)
              if not (bare and abs(x - L.w / 2) < S / 2 + 40 * k and abs(y - L.h / 2) < S / 2 + 40 * k)]
    tilt = float(rng.uniform(-2.5, 2.5))
    ix, iy = (L.w / 2, L.h / 2) if bare else L.icon
    sx, sy = L.sign
    for f in range(max(1, int(round(seconds * FPS)))):
        t = f / FPS
        canvas = bg.copy()
        for sp, x, y, ph, rot in scraps:
            sp.blit(canvas, x + 6 * k * math.sin(t * 1.3 + ph), y + 9 * k * math.sin(t * 0.9 + ph),
                    _ease_back((t - 0.05) / 0.3), rot + 4 * math.sin(t + ph))
        icon.blit(canvas, ix, iy + 7 * k * math.sin(t * 2 * math.pi / 2.4), _ease_back((t - 0.12) / 0.38),
                  1.4 * math.sin(t * 2 * math.pi / 3.1))
        if sign is not None:
            sign.blit(canvas, sx, sy, _ease_back(t / 0.32), tilt)
        canvas *= vig
        sink(canvas)


# ── sketch ───────────────────────────────────────────────────────────────────
BOIL = 3          # redraws per element
BOIL_EVERY = 4    # frames per redraw (30 / 4 = 7.5 fps boil)


def _ink_layer(prims, S: int, pad: int, rng: np.random.Generator, wobble: float, ink_w: float) -> Sprite:
    size = (S + 2 * pad, S + 2 * pad)
    sp = Sprite.empty(*size)
    wash = _noise(size[1], size[0], int(rng.integers(1 << 30)), ((3, 1.0), (12, 0.6), (60, 0.3)))
    strokes = []
    for prim in prims:
        closed, pts, width, role = _to_px(prim, S)
        pts = [(x + pad, y + pad) for x, y in pts]
        fine = _subdivide(pts, closed)
        if closed:   # watercolour fill, offset from the line like a quick colouring-in
            off = S / 66
            fill_pts = _jitter([(x + off, y + off * 0.8) for x, y in fine], rng, wobble * 1.6)
            m = cv2.GaussianBlur(_mask(True, fill_pts, 0, size), (0, 0), 2.2) * (0.78 + 0.18 * wash)
            sp.over(_bgr(SKETCH[role]), np.clip(m, 0, 1))
            strokes.append((True, _jitter(fine, rng, wobble), ink_w))
        else:        # open strokes are thick marker lines in the role colour, inked on top
            line = _jitter(fine, rng, wobble)
            m = _mask(False, line, width * 0.8, size)
            sp.over(_bgr(SKETCH[role] if role not in ("light",) else SKETCH["ink"]), m)
            strokes.append((False, _jitter(fine, rng, wobble * 0.6), max(ink_w, width * 0.25)))
    for closed, pts, w_ in strokes:
        seq = pts + ([pts[0]] if closed else [])
        sp.over(_bgr(SKETCH["ink"]), _mask(False, seq, w_, size))
    return sp


def _sketch_title(word: str, L: Layout, rng: np.random.Generator) -> Sprite:
    k = L.k
    font = _fit_font(word, "PatrickHand-Regular.ttf", L.title_max, int(190 * k), int(50 * k))
    tm = _text_mask(word, font, pad=int(30 * k))
    th, tw = tm.shape
    pad = int(20 * k)
    sp = Sprite.empty(tw + 2 * pad, th + int(60 * k) + 2 * pad)
    big = np.zeros(sp.a.shape, np.float32)
    big[pad:pad + th, pad:pad + tw] = tm
    y = pad + th + 10 * k   # marker underline, slightly drunk
    line = _jitter(_subdivide([(pad + 30 * k, y), (pad + tw - 30 * k, y + rng.uniform(-14, 14) * k)], False),
                   rng, 4 * k)
    um = _mask(False, line, 16 * k, (sp.a.shape[1], sp.a.shape[0]))
    sp.over(_bgr(SKETCH["accent"]), um * 0.9)
    sp.over(_bgr(SKETCH["ink"]), big)
    return sp


DOODLES = [
    [("line", [(0.5 + 0.38 * (k / 30) * math.cos(k * 0.42), 0.5 + 0.38 * (k / 30) * math.sin(k * 0.42))
               for k in range(31)], 0.06, "ink")],                                            # spiral
    [("line", [(0.5, 0.1), (0.5, 0.9)], 0.07, "ink"), ("line", [(0.1, 0.5), (0.9, 0.5)], 0.07, "ink")],  # plus
    [("line", [(0.1, 0.6), (0.3, 0.4), (0.5, 0.6), (0.7, 0.4), (0.9, 0.6)], 0.06, "ink")],  # zigzag
    [("circle", (0.5, 0.5, 0.18), "accent")],                                                # dot
]


def _render_sketch(spec: dict, seconds: float, seed: int, L: Layout, sink) -> None:
    rng = np.random.default_rng(seed)
    k = L.k
    paper = np.ones((L.h, L.w, 3), np.float32) * np.asarray(_bgr(SKETCH["paper"]), np.float32)
    paper *= (1 + 0.045 * _noise(L.h, L.w, seed))[..., None]
    step = max(30, int(76 * k))
    for y in range(int(240 * k) % step + step, L.h, step):   # faint notebook rules
        cv2.line(paper, (0, y), (L.w, y), _bgr(SKETCH["rule"]), max(1, int(2 * k)), cv2.LINE_AA)
    mx = int(118 * k)
    cv2.line(paper, (mx, 0), (mx, L.h), _bgr("#efb3ad"), max(1, int(2 * k)), cv2.LINE_AA)
    vig = _vignette(L.w, L.h, 0.14)
    bare = bool(spec.get("notitle"))
    S = int(min(L.w, L.h) * 0.72) if bare else L.icon_size
    icons = [_ink_layer(ICONS[spec["icon"]], S, int(40 * k), np.random.default_rng(seed + 10 + j), 5 * k, 7 * k)
             for j in range(BOIL)]
    titles = [] if spec.get("notitle") else         [_sketch_title(spec["word"], L, np.random.default_rng(seed + 20 + j)) for j in range(BOIL)]
    doodles = []
    for x, y in ([] if bare else _scatter(L, rng, 5, 80 * k)):
        d = DOODLES[int(rng.integers(len(DOODLES)))]
        ds = int(90 * k)
        doodles.append(([_ink_layer(d, ds, int(16 * k), np.random.default_rng(seed + 40 + j + int(x)), 3 * k, 5 * k)
                         for j in range(BOIL)], x, y, float(rng.uniform(-30, 30))))
    ix, iy = (L.w / 2, L.h / 2) if bare else L.icon
    sx, sy = L.sign
    for f in range(max(1, int(round(seconds * FPS)))):
        t = f / FPS
        v = (f // BOIL_EVERY) % BOIL
        canvas = paper.copy()
        for frames, x, y, rot in doodles:
            frames[v].blit(canvas, x, y, 1.0, rot, reveal=min(1.0, max(0.0, (t - 0.35) / 0.4)))
        icons[v].blit(canvas, ix, iy, 1.0 + 0.012 * math.sin(t * 2.2), 0.0,
                      reveal=min(1.0, max(0.0, (t - 0.1) / 0.6)))
        if titles:
            titles[v].blit(canvas, sx, sy, 1.0, -1.5, reveal=min(1.0, t / 0.4))
        canvas *= vig
        sink(canvas)


# ── output ───────────────────────────────────────────────────────────────────
# ── anatomy ──────────────────────────────────────────────────────────────────
# Ported from the anatomy skill's rules (wheresryan22/anatomy, MIT): the idea is a real bench instrument,
# isometric, four-tone shading (top / left / right / edge), correct painter's order, a framed card with
# title, hint and a live readout, calm eased motion. One model drives the moving parts AND the number:
# the carriage position sets the needle angle and the readout, so they can never disagree.
ANAT = {"bg": "#0e1116", "card": "#151a22", "frame": "#2a3240", "grid": "#1f2733", "text": "#e9edf3",
        "dim": "#7d8899", "accent": "#ff8a3d", "engrave": "#39404d",
        "metal": ("#c9d1dc", "#8e99a8", "#5d6778", "#eef2f7"),     # top, left, right, edge
        "base": ("#3a4352", "#2a313d", "#1f252e", "#56617a"),
        "cart": ("#ffa45c", "#e07a2e", "#b45a1c", "#ffd2a8")}
_C30, _S30 = math.cos(math.pi / 6), 0.5
_SH = 2                                       # cv2 sub-pixel shift bits (coords x4)


def _ease_io(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _q(p) -> tuple[int, int]:
    return int(round(p[0] * 4)), int(round(p[1] * 4))


class _Iso:
    """x runs to the lower right, y to the lower left, z up. Bigger x + y = nearer the viewer."""

    def __init__(self, ox: float, oy: float, s: float):
        self.ox, self.oy, self.s = ox, oy, s

    def p(self, x, y, z):
        return (self.ox + (x - y) * _C30 * self.s, self.oy + (x + y) * _S30 * self.s - z * self.s)

    def box(self, img, x, y, z, dx, dy, dz, tones):
        P = self.p
        top = [P(x, y, z + dz), P(x + dx, y, z + dz), P(x + dx, y + dy, z + dz), P(x, y + dy, z + dz)]
        left = [P(x, y + dy, z), P(x + dx, y + dy, z), P(x + dx, y + dy, z + dz), P(x, y + dy, z + dz)]
        right = [P(x + dx, y, z), P(x + dx, y + dy, z), P(x + dx, y + dy, z + dz), P(x + dx, y, z + dz)]
        for face, tone in ((right, tones[2]), (left, tones[1]), (top, tones[0])):
            cv2.fillPoly(img, [np.array([_q(v) for v in face], np.int32)], _bgr(tone), cv2.LINE_AA, _SH)
        lw = max(1, int(round(self.s * 0.035)))
        for a, b in ((top[0], top[1]), (top[3], top[0]), (top[1], top[2]), (top[2], top[3])):
            cv2.line(img, _q(a), _q(b), _bgr(tones[3]), lw, cv2.LINE_AA, _SH)


def _anat_text(img: np.ndarray, text: str, font, x: float, y: float, color: str, anchor: str = "c",
               alpha: float = 1.0) -> None:
    m = _text_mask(text, font, pad=4) * alpha
    h, w = m.shape
    x0 = int(x - w / 2) if anchor == "c" else int(x)
    y0 = int(y - h / 2)
    H, W = img.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
        return
    roi = img[y0:y0 + h, x0:x0 + w].astype(np.float32)
    col = np.asarray(_bgr(color), np.float32)
    img[y0:y0 + h, x0:x0 + w] = (roi * (1 - m[..., None]) + col * m[..., None]).astype(np.uint8)


def _render_anatomy(spec: dict, seconds: float, seed: int, L: Layout, sink) -> None:
    rng = np.random.default_rng(seed)
    k = L.k
    img = np.zeros((L.h, L.w, 3), np.uint8)
    img[:] = _bgr(ANAT["bg"])
    m = int(36 * k)                                   # the framed card
    cv2.rectangle(img, (m, m), (L.w - m, L.h - m), _bgr(ANAT["card"]), -1)
    step = int(54 * k)
    for gx in range(m + step, L.w - m, step):        # engineer's dot grid
        for gy in range(m + step, L.h - m, step):
            cv2.circle(img, (gx, gy), max(1, int(2 * k)), _bgr(ANAT["grid"]), -1, cv2.LINE_AA)
    cv2.rectangle(img, (m, m), (L.w - m, L.h - m), _bgr(ANAT["frame"]), max(2, int(3 * k)), cv2.LINE_AA)

    bare = bool(spec.get("notitle"))
    # fit the whole bench (base corners + dial disc) into the hero box, then centre it there
    if bare:
        bx, by, bw, bh = L.w / 2, L.h * 0.45, L.w * 0.84, L.h * 0.6
    elif L.portrait:
        bx, by, bw, bh = L.w / 2, L.icon[1] + 0.06 * L.h, L.w * 0.86, L.h * 0.4
    else:
        bx, by, bw, bh = L.w * 0.73, L.h * 0.54, L.w * 0.44, L.h * 0.74
    unit = _Iso(0, 0, 1)
    pts = [unit.p(x, y, z) for x in (0, 10) for y in (0, 6) for z in (0, 1.35)]
    dx_, dy_ = unit.p(8.55, 1.05, 6.3)
    pts += [(dx_ - 1.7, dy_ - 1.7), (dx_ + 1.7, dy_ + 1.7)]
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    s = min(bw / (max(xs) - min(xs)), bh / (max(ys) - min(ys)))
    iso = _Iso(bx - s * (max(xs) + min(xs)) / 2, by - s * (max(ys) + min(ys)) / 2, s)

    # static instrument, painter's order: feet, base, icon plate, ruler + rail, gauge post, dial
    for fx, fy in ((0.4, 0.4), (8.9, 0.4), (0.4, 5.0), (8.9, 5.0)):
        iso.box(img, fx, fy, 0, 0.7, 0.6, 0.45, ANAT["base"])
    iso.box(img, 0, 0, 0.45, 10, 6, 0.9, ANAT["base"])
    px0, py0, pw, ph = 0.6, 0.5, 4.6, 2.9
    iso.box(img, px0, py0, 1.35, pw, ph, 0.12, ANAT["metal"])
    # the icon printed on the plate, layer by layer in its own roles (light parts cut back to bare metal)
    S = 320
    ink = {"light": ANAT["metal"][0], "dark": ANAT["engrave"], "ink": ANAT["engrave"], "accent": ANAT["accent"],
           "main": ANAT["accent"], "red": "#d2453a", "green": "#2f9e5b"}
    prgb = np.zeros((S, S, 3), np.float32)
    pa = np.zeros((S, S), np.float32)
    for prim in ICONS[spec["icon"]]:
        closed, pts, width, role = _to_px(prim, S)
        mk = _mask(closed, pts, width, (S, S))
        prgb = prgb * (1 - mk[..., None]) + np.asarray(_bgr(ink.get(role, ANAT["engrave"])), np.float32) * mk[..., None]
        pa = np.maximum(pa, mk)
    side = min(pw, ph) * 0.86
    ux, uy = px0 + (pw - side) / 2, py0 + (ph - side) / 2
    a, b, c = iso.p(ux, uy, 1.47), iso.p(ux + side, uy, 1.47), iso.p(ux, uy + side, 1.47)
    M = cv2.getAffineTransform(np.float32([[0, 0], [S, 0], [0, S]]), np.float32([a, b, c]))
    wa = cv2.warpAffine(pa, M, (L.w, L.h), flags=cv2.INTER_LINEAR)[..., None]
    wrgb = cv2.warpAffine(prgb, M, (L.w, L.h), flags=cv2.INTER_LINEAR)
    img[:] = (img * (1 - wa) + wrgb * wa).astype(np.uint8)
    iso.box(img, 0.5, 3.95, 1.35, 9.0, 0.5, 0.3, ANAT["metal"])            # rail
    for i in range(19):                                                     # tick ruler in front of it
        tx = 0.7 + i * 0.47
        h_ = 0.55 if i % 2 == 0 else 0.3
        cv2.line(img, _q(iso.p(tx, 4.75, 1.36)), _q(iso.p(tx, 4.75 + h_, 1.36)), _bgr(ANAT["dim"]),
                 max(1, int(2 * k)), cv2.LINE_AA, _SH)
    iso.box(img, 8.2, 0.7, 1.35, 0.7, 0.7, 3.6, ANAT["metal"])             # gauge post (back right)
    dcx, dcy = iso.p(8.55, 1.05, 6.3)
    R = 1.55 * s
    cv2.circle(img, (int(dcx + 0.14 * s), int(dcy + 0.14 * s)), int(R), _bgr("#07090c"), -1, cv2.LINE_AA)
    cv2.circle(img, (int(dcx), int(dcy)), int(R), _bgr(ANAT["metal"][2]), -1, cv2.LINE_AA)
    cv2.circle(img, (int(dcx), int(dcy)), int(R * 0.88), _bgr("#f3f5f8"), -1, cv2.LINE_AA)
    for i in range(41):                                                     # 270 degree dial
        ang = math.radians(225 - i * 270 / 40)
        r0 = R * (0.66 if i % 5 == 0 else 0.75)
        p0 = (int(dcx + r0 * math.cos(ang)), int(dcy - r0 * math.sin(ang)))
        p1 = (int(dcx + R * 0.82 * math.cos(ang)), int(dcy - R * 0.82 * math.sin(ang)))
        cv2.line(img, p0, p1, _bgr("#2b3240"), max(1, int((4 if i % 5 == 0 else 2) * k)), cv2.LINE_AA)

    word = spec["word"].upper()
    title_f = _fit_font(word, "ArchivoBlack-Regular.ttf", L.title_max, int(120 * k))
    small = _font("ArchivoBlack-Regular.ttf", max(18, int(30 * k)))
    num_f = _font("ArchivoBlack-Regular.ttf", max(22, int(50 * k)))
    fig = f"FIG. {int(rng.integers(1, 9)):02d}"
    static = img
    target = float(rng.uniform(0.62, 0.9))
    sx, sy = L.sign
    for f in range(max(1, int(round(seconds * FPS)))):
        t = f / FPS
        u = target * _ease_io((t - 0.15) / 1.3) + (0.015 * math.sin((t - 1.45) * 2.1) if t > 1.45 else 0.0)
        canvas = static.copy()
        iso.box(canvas, 0.7 + u * 7.0, 3.8, 1.65, 1.5, 0.8, 0.95, ANAT["cart"])   # the carriage = the model
        ang = math.radians(225 - u * 270)
        tip = (dcx + R * 0.8 * math.cos(ang), dcy - R * 0.8 * math.sin(ang))
        cv2.line(canvas, _q((dcx, dcy)), _q(tip), _bgr(ANAT["accent"]), max(2, int(6 * k)), cv2.LINE_AA, _SH)
        cv2.circle(canvas, (int(dcx), int(dcy)), max(3, int(11 * k)), _bgr("#2b3240"), -1, cv2.LINE_AA)
        if not bare:
            fade = _ease_io(t / 0.45)
            if fade > 0.02:
                _anat_text(canvas, fig, small, sx, sy - 100 * k, ANAT["accent"], alpha=fade)
                _anat_text(canvas, word, title_f, sx, sy + (1 - fade) * 30 * k, ANAT["text"], alpha=fade)
        _anat_text(canvas, f"READ {u * 100:05.1f}", num_f, m + 40 * k, m + 70 * k, ANAT["accent"], anchor="l")
        sink(canvas.astype(np.float32))


_RENDERERS = {"papercut": _render_papercut, "sketch": _render_sketch, "anatomy": _render_anatomy}


def _normalise(spec: dict) -> dict:
    """notitle=True: icon + decor only, for pipelines that lay their own text over the frame."""
    out = {"word": clean_word(spec.get("word", "")) or "listen",
           "icon": spec.get("icon") if spec.get("icon") in ICONS else "spark"}
    if spec.get("notitle"):
        out["notitle"] = True
    return out


def render(style: str, spec: dict, seconds: float, out_dir: Path, *, size: tuple[int, int] = (1080, 1920),
           seed: int = 7, crf: int = 18) -> Path:
    """One scene clip, cached in out_dir by content hash. size = (width, height), even numbers.
    Raises RuntimeError if ffmpeg fails."""
    style = style if style in STYLES else "papercut"
    w, h = int(size[0]) // 2 * 2, int(size[1]) // 2 * 2
    if w < 64 or h < 64:
        raise ValueError(f"frame too small: {w}x{h}")
    spec = _normalise(spec)
    seconds = max(0.5, round(float(seconds), 1))
    key = hashlib.sha1(json.dumps([VERSION, style, spec, seconds, seed, w, h], sort_keys=True).encode()).hexdigest()[:16]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"drawn_{style}_{w}x{h}_{key}.mp4"
    if out.exists() and out.stat().st_size > 10_000:
        return out
    tmp = out.with_suffix(".part.mp4")
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
                             "-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p", str(tmp)],
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def sink(canvas: np.ndarray) -> None:
        proc.stdin.write(np.clip(canvas, 0, 255).astype(np.uint8).tobytes())

    try:
        _RENDERERS[style](spec, seconds, int(seed), Layout(w, h), sink)
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"drawn-scenes: ffmpeg failed: {err[-300:]}")
    except BaseException:
        proc.kill()
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out)
    print(f"  drawn: {style} {w}x{h} '{spec['word']}' + {spec['icon']} ({seconds:g}s) -> {out.name}")
    return out


def still(style: str, spec: dict, out: Path, *, size: tuple[int, int] = (1080, 1920), at: float = 1.5,
          seed: int = 7) -> Path:
    """One frame as PNG, to eyeball a look without encoding video."""
    w, h = int(size[0]) // 2 * 2, int(size[1]) // 2 * 2
    target = int(at * FPS)
    got: list[np.ndarray] = []
    count = [0]

    def sink(canvas: np.ndarray) -> None:
        if count[0] == target:
            got.append(np.clip(canvas, 0, 255).astype(np.uint8))
        count[0] += 1
    _RENDERERS[style if style in STYLES else "papercut"](_normalise(spec), at + 1 / FPS, seed, Layout(w, h), sink)
    cv2.imwrite(str(out), got[0])
    return out
