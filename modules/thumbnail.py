"""Generate a high-CTR 1280x720 thumbnail for the long-form doc.

Faceless documentary channels live or die on the thumbnail. YouTube's
auto-picked frame-grabs (a blurry Ken-Burns mid-pan) destroy click-through, so
we composite our own: the doc's strongest still, darkened for legibility, with
2-4 BIG punchy words in the bottom-left and a thin brand strip.

Pure Pillow + a system font (no paid asset, no API). Shorts don't take a
thumbnail (YouTube ignores it for the Shorts shelf), so this is doc-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

THUMB_SIZE = (1280, 720)

# Windows ships these; fall back to Pillow's bundled font if absent.
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\impact.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _load_font(size: int):
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _punchy_words(title: str, max_words: int = 4) -> str:
    """Strip filler + channel boilerplate so the overlay is glanceable.
    'The Rise & Fall of Ronaldinho' -> 'RISE & FALL RONALDINHO'.
    Drops everything after a separator (| or -) and bare punctuation tokens."""
    import re
    # cut a trailing "| FIFA World Cup ..." / " - Episode ..." tail
    head = re.split(r"\s[|\-–—:]\s", title)[0]
    filler = {"the", "a", "an", "of", "how", "why", "is", "to", "and",
              "his", "her", "their", "story", "in", "on", "ep", "episode"}
    words = []
    for w in head.replace(":", " ").split():
        w = w.strip()
        if not w or (w != "&" and not re.search(r"[A-Za-z0-9]", w)):  # skip "|" etc, keep "&"
            continue
        if w.lower().rstrip(".") in filler:
            continue
        words.append(w)
    words = words or [w for w in head.split() if w]
    return " ".join(words[:max_words]).upper()


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def make_thumbnail(still: str | Path, title: str, out: str | Path) -> Path | None:
    """Returns the thumbnail path, or None if it couldn't be made (a bad/missing
    still must never abort the upload — the doc just keeps YouTube's default)."""
    from PIL import Image, ImageDraw, ImageEnhance

    out = Path(out)
    W, H = THUMB_SIZE

    # 1. Cover-fit the still to 1280x720 (crop overflow, never distort).
    try:
        img = Image.open(still).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        print(f"[thumbnail] cannot open still {still!r}: {e}; skipping thumbnail")
        return None
    scale = max(W / img.width, H / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)),
                     Image.LANCZOS)
    left = (img.width - W) // 2
    top = (img.height - H) // 2
    img = img.crop((left, top, left + W, top + H))

    # 2. Punch the colour, then a left-to-bottom dark gradient for text contrast.
    img = ImageEnhance.Color(img).enhance(1.25)
    img = ImageEnhance.Contrast(img).enhance(1.08)
    grad = Image.new("L", (1, H), 0)
    for y in range(H):
        # darker toward the bottom where the text sits
        grad.putpixel((0, y), int(200 * (y / H) ** 1.6))
    alpha = grad.resize((W, H))
    shade = Image.new("RGB", (W, H), (8, 8, 12))
    img = Image.composite(shade, img, alpha)

    draw = ImageDraw.Draw(img)

    # 3. Headline: biggest font that fits in <=2 lines within the safe margin.
    headline = _punchy_words(title)
    margin = 64
    max_text_w = W - 2 * margin
    font_size = 150
    while font_size > 60:
        font = _load_font(font_size)
        lines = _wrap(draw, headline, font, max_text_w)
        if len(lines) <= 2:
            break
        font_size -= 8
    line_h = font_size + 12
    total_h = line_h * len(lines)
    y = H - margin - total_h

    for line in lines:
        x = margin
        # heavy black stroke so text reads over any image
        draw.text((x, y), line, font=font, fill=(255, 255, 255),
                  stroke_width=max(6, font_size // 18), stroke_fill=(0, 0, 0))
        y += line_h

    # 4. Brand strip top-left.
    bfont = _load_font(34)
    btext = config.CHANNEL_NAME.upper()
    bw = draw.textlength(btext, font=bfont)
    draw.rectangle([0, 0, bw + 48, 56], fill=(176, 0, 32))  # burgundy accent
    draw.text((24, 9), btext, font=bfont, fill=(255, 255, 255))

    try:
        img.save(out, "JPEG", quality=90)
    except OSError as e:
        print(f"[thumbnail] cannot write {out!r}: {e}; skipping thumbnail")
        return None
    return out
