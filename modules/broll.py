"""Pexels stock b-roll layer (pattern lifted from MoneyPrinterTurbo).

Commons stills cover beats whose visual cue is a real named entity. When the
Commons search whiffs, the old behaviour was a blank gradient card. This module
upgrades that fallback: Groq turns the beat's cue into a generic 1-3 word stock
search term ("stadium floodlights", "counting money"), Pexels returns free
licensed video clips at the target orientation, and we render a trimmed,
size-normalized clip for the beat.

Reliability rules borrowed from MoneyPrinterTurbo:
- API key rotation: PEXELS_API_KEY may hold a comma-separated list; requests
  round-robin across keys to stretch free tiers.
- URL-hash cache: downloads land in cache/broll/vid-{md5}.mp4 so retried runs
  never redownload.
- Validate before use: every download is probed with ffprobe (duration > 0)
  and deleted if corrupt, so a truncated CDN response can't reach the render.
- Dry-safe: no PEXELS_API_KEY -> fetch_clip() returns None and the composer
  keeps its gradient-card fallback. The pipeline never hard-fails here.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

PEXELS_API = "https://api.pexels.com/videos/search"
MIN_CLIP_SEC = 4          # don't bother with clips shorter than a beat's floor
SEARCH_PER_PAGE = 15

_key_counter = 0
_key_lock = threading.Lock()

# ── Static system prompt FIRST (cache prefix) — no f-strings here. ─────────────
TERM_SYSTEM_PROMPT = """You convert a video beat description into ONE generic
stock-footage search term for Pexels.

Rules:
- 1 to 3 English words, lowercase.
- Generic and photographable: "stadium crowd", "counting money", "football
  training", "city skyline at night". Stock sites have no footage of specific
  people, so NEVER return a person's name, club name, or trophy name.
- Stay on the emotional register of the beat (money, tension, celebration,
  origin story) while keeping the football context when it fits.
- Return STRICT JSON, no markdown fences: {"term": "..."}
"""

# Deterministic rotation when Groq is unavailable: generic, always-stocked terms.
_FALLBACK_TERMS = ["football stadium", "stadium crowd", "soccer ball grass",
                   "stadium floodlights", "football training"]


def _api_keys() -> list[str]:
    raw = os.environ.get("PEXELS_API_KEY", "").strip()
    return [k.strip() for k in raw.split(",") if k.strip()]


def _next_key(keys: list[str]) -> str:
    """Round-robin across keys. search() already returns [] when no keys are
    configured; the empty-list guard here only protects future refactors from a
    ZeroDivisionError, and stays soft (empty string -> 401 -> caught upstream)
    to preserve the module's never-hard-fail contract."""
    if not keys:
        return ""
    global _key_counter
    with _key_lock:
        _key_counter += 1
        return keys[_key_counter % len(keys)]


def generic_term(cue: str, beat_index: int = 0) -> str:
    """Beat cue -> generic stock term via Groq fast model, with a deterministic
    rotation fallback so a Groq outage can't stall the visual layer."""
    try:
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise ValueError("GROQ_API_KEY not set")
        client = Groq(api_key=key)
        # Variable content goes in the USER message, AFTER the cached prefix.
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL_FAST,
            messages=[
                {"role": "system", "content": TERM_SYSTEM_PROMPT},
                {"role": "user", "content": f"Beat description: {cue}"},
            ],
            temperature=0.3,
            max_tokens=40,
            response_format={"type": "json_object"},
        )
        term = str(json.loads(resp.choices[0].message.content).get("term") or "").strip()
        term = re.sub(r"[^a-z0-9 ]", "", term.lower()).strip()
        if term and len(term.split()) <= 4:
            return term
        raise ValueError(f"unusable term {term!r}")
    except Exception as e:  # any failure -> deterministic fallback
        # Log the exception TYPE only: Groq/HTTP errors can echo request
        # headers, and stderr lands in public CI logs.
        print(f"[broll] term generation failed ({type(e).__name__}); using fallback",
              file=sys.stderr)
        return _FALLBACK_TERMS[beat_index % len(_FALLBACK_TERMS)]


def search(term: str, size: tuple[int, int], min_duration: int = MIN_CLIP_SEC) -> list[dict]:
    """One Pexels video search. Returns [{url, duration}] best-first, filtered
    to the target orientation and a usable minimum duration."""
    keys = _api_keys()
    if not keys:
        return []
    w, h = size
    orientation = "portrait" if h > w else "landscape"
    try:
        r = requests.get(PEXELS_API, timeout=(15, 30),
                         headers={"Authorization": _next_key(keys)},
                         params={"query": term, "per_page": SEARCH_PER_PAGE,
                                 "orientation": orientation})
        r.raise_for_status()
        items: list[dict] = []
        for v in r.json().get("videos", []):
            if v.get("duration", 0) < min_duration:
                continue
            # Pick the smallest rendition that still covers the target frame;
            # exact-resolution matching (MoneyPrinterTurbo's approach) skips too
            # many usable clips.
            files = sorted((f for f in v.get("video_files", [])
                            if f.get("width") and f.get("height")),
                           key=lambda f: f["width"] * f["height"])
            for f in files:
                if f["width"] >= w * 0.9 and f["height"] >= h * 0.9:
                    items.append({"url": f["link"], "duration": v["duration"]})
                    break
        return items
    except Exception as e:
        print(f"[broll] pexels search failed for {term!r}: {e}", file=sys.stderr)
        return []


def _probe_ok(path: Path) -> bool:
    """A video is real only if ffprobe reads a positive duration from it."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip() or 0) > 0
    except Exception:
        return False


def _download(url: str) -> Path | None:
    """Fetch into the URL-hash cache; skip when cached, delete when corrupt."""
    cache_dir = config.CACHE / "broll"
    cache_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.md5(url.split("?")[0].encode()).hexdigest()
    dest = cache_dir / f"vid-{url_hash}.mp4"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        r = requests.get(url, timeout=(30, 180))
        r.raise_for_status()
        dest.write_bytes(r.content)
    except Exception as e:
        print(f"[broll] download failed: {e}", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return None
    if _probe_ok(dest):
        return dest
    print(f"[broll] corrupt download removed: {dest.name}", file=sys.stderr)
    dest.unlink(missing_ok=True)
    return None


def _render_clip(src: Path, out: Path, duration: float, size: tuple[int, int]) -> Path:
    """Trim + cover-crop + normalize to the doc's frame spec (30fps, yuv420p,
    no audio) so the concat demuxer sees uniform streams."""
    w, h = size
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
          f"crop={w}:{h},fps=30,format=yuv420p")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-t", f"{duration:.3f}", "-vf", vf, "-an",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True)
    return out


def fetch_clip(cue: str, out: Path, duration: float, size: tuple[int, int],
               beat_index: int = 0) -> Path | None:
    """Resolve a beat with no Commons hit to a stock b-roll clip.

    Returns the rendered clip path, or None when Pexels is unconfigured or has
    nothing usable (caller keeps its gradient fallback either way).
    """
    if not _api_keys():
        return None
    term = generic_term(cue, beat_index)
    for item in search(term, size, min_duration=max(MIN_CLIP_SEC, int(min(duration, 15)))):
        src = _download(item["url"])
        if not src:
            continue
        try:
            return _render_clip(src, out, duration, size)
        except (subprocess.CalledProcessError, OSError) as e:
            # OSError covers a missing ffmpeg binary; the composer's gradient
            # fallback still saves the beat. Full detail is safe here: local
            # ffmpeg/OS errors carry no credentials.
            print(f"[broll] render failed for {src.name}: {e}", file=sys.stderr)
            continue  # try the next search result
    print(f"[broll] no usable clip for term {term!r}", file=sys.stderr)
    return None


if __name__ == "__main__":
    clip = fetch_clip("Ronaldinho celebrating a goal", Path("broll_test.mp4"),
                      6.0, config.DOC_RESOLUTION)
    print("clip:", clip)
