"""Copyright-safe imagery. Pulls CC / public-domain stills from Wikimedia
Commons for each beat's visual cue. We narrate over stills with Ken-Burns
motion instead of using live match footage, which keeps the content
transformative and demonetization-safe (a risk the synthesis flagged).

Falls back to a generated gradient card if Commons has no usable hit, so the
autonomous pipeline never hard-fails on a missing image.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

# Wikimedia's User-Agent policy 429-blocks generic/no-contact agents, so a real
# contact is REQUIRED or every image download fails and the doc silently falls
# back to blank gradient cards. Default to the public repo URL only; set
# WIKIMEDIA_CONTACT (e.g. an email) to override without committing PII.
_CONTACT = os.environ.get("WIKIMEDIA_CONTACT",
                          "https://github.com/shauryalowkeygotaura").strip()
_HEADERS = {"User-Agent": f"football-shorts-autopilot/1.0 ({_CONTACT})"}

# Capitalized words that start a sentence/phrase but aren't real subjects.
_STOP = {"the", "a", "an", "in", "his", "her", "their", "with", "and", "of",
         "on", "for", "to", "at", "as", "is", "from", "after", "before", "during",
         "young", "later", "early"}


def _proper_nouns(text: str) -> list[str]:
    """Pull the distinct proper-noun tokens from a descriptive cue.

    'Ronaldinho holding the Ballon d'Or trophy' -> ['Ronaldinho', 'Ballon'].
    Commons file titles are entity-named, so an entity query hits where the
    full descriptive sentence never would.
    """
    tokens = re.findall(r"[A-Z][A-Za-z'’.-]+", text)
    out: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if t.lower() in _STOP or len(t) <= 1:
            continue
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def _commons_query(query: str) -> str | None:
    """One Commons file search. Returns a direct image URL or None."""
    try:
        r = requests.get(config.WIKIMEDIA_API, headers=_HEADERS, timeout=20, params={
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": query, "gsrnamespace": "6", "gsrlimit": "5",
            "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": "1600",
        })
        pages = r.json().get("query", {}).get("pages", {})
        # search results aren't dict-ordered by rank, so respect the index
        ranked = sorted(pages.values(), key=lambda p: p.get("index", 999))
        for page in ranked:
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")
            url = info.get("thumburl") or info.get("url")
            if url and mime.startswith("image/") and mime != "image/svg+xml":
                return url
    except Exception:
        pass
    return None


def _search_commons(query: str, subject: str | None = None) -> str | None:
    """Resolve a (possibly descriptive) visual cue to a real Commons image by
    trying progressively broader entity queries before giving up:
      full proper nouns -> primary entity (+football) -> raw cue -> topic subject.
    """
    nouns = _proper_nouns(query)
    candidates: list[str] = []
    if nouns:
        candidates.append(" ".join(nouns[:3]))   # e.g. "Ronaldinho Barcelona"
        candidates.append(f"{nouns[0]} football")  # disambiguate a lone name
        candidates.append(nouns[0])
    candidates.append(query)                       # raw cue, last resort
    if subject:
        candidates.extend([subject, f"{subject} football"])

    seen: set[str] = set()
    for cand in candidates:
        cand = cand.strip()
        if not cand or cand.lower() in seen:
            continue
        seen.add(cand.lower())
        url = _commons_query(cand)
        if url:
            return url
    return None


_RASTER_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif",
    "image/bmp", "image/tiff",
}


def _is_decodable(path: Path) -> bool:
    """True when the bytes on disk really are an image ffmpeg can read.

    PIL verify() must run on a fresh handle and invalidates the object, so the
    file is opened twice on purpose: once to verify integrity, once to confirm
    it loads and has real dimensions.
    """
    try:
        from PIL import Image
    except ImportError:
        return True  # cannot verify here; let the render surface it as before
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            w, h = im.size
        return w > 0 and h > 0
    except Exception:
        return False


def _download(url: str, dest: Path, retries: int = 3) -> bool:
    """Fetch the image bytes, retrying on transient throttling (429/503).
    Returns True only on a real image payload."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=30)
            if r.status_code in (429, 503) and attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))  # linear backoff
                continue
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
            # startswith("image/") is not enough. Commons serves SVG as
            # image/svg+xml, which passes that test, gets written as .jpg, and
            # then ffmpeg dies with exit 8 on a file it cannot decode. Raster
            # types only.
            if ctype not in _RASTER_TYPES:
                return False
            dest.write_bytes(r.content)
            if dest.stat().st_size <= 1000:
                return False
            # Content-Type is a claim by the server, not a fact about the bytes.
            # A truncated or mislabelled payload still decodes as nothing, and
            # the only place that surfaced was a hard ffmpeg crash halfway
            # through a render. Verify it opens BEFORE the render depends on it.
            return _is_decodable(dest)
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            return False
    return False


def _gradient_card(dest: Path, w: int, h: int) -> None:
    """Deterministic fallback still (dark cinematic gradient)."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"gradients=s={w}x{h}:c0=0x0b0f1a:c1=0x16233a:duration=1",
         "-frames:v", "1", str(dest)], check=True, capture_output=True)


def fetch_still(query: str, dest: Path, size: tuple[int, int],
                subject: str | None = None) -> tuple[Path, bool]:
    """Resolve one beat's visual to a local image file (always returns a path).

    `subject` is the doc's topic, used as a final fallback so a beat with an
    unsearchable cue still gets a topic-relevant photo rather than a blank card.

    Returns (path, found): `found` is False when Commons had no usable hit and
    the still is the gradient placeholder, which lets the composer try a stock
    b-roll clip before settling for the blank card.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _search_commons(query, subject=subject)
    if url and _download(url, dest):
        return dest, True
    _gradient_card(dest, *size)
    return dest, False


def gradient_still(dest: Path, size: tuple[int, int]) -> Path:
    """The deterministic fallback still, as a public entry point.

    The composer needs it to salvage a beat whose real image will not render.
    """
    _gradient_card(dest, *size)
    return dest


def _describe_image(image: Path) -> str:
    """Whatever can be said about a still that ffmpeg refused to render.

    The image is the only variable in an otherwise fixed command, so its shape
    is the first thing worth knowing and the one thing the traceback omits.
    """
    try:
        size_b = image.stat().st_size
    except OSError as e:
        return "unreadable on disk (%s)" % e
    try:
        with Image.open(image) as im:
            return "%s %dx%d %s, %d bytes" % (
                im.format, im.width, im.height, im.mode, size_b)
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return "%d bytes, PIL cannot identify it (%s)" % (size_b, e)


def ken_burns_clip(image: Path, out: Path, duration: float,
                   size: tuple[int, int], zoom_in: bool = True) -> Path:
    """Render a slow Ken-Burns push/pull over a still -> mp4 (no audio).

    zoompan needs a frame budget (d) = duration * fps. Scale up first so the
    crop has headroom, then pan. fps=30.

    Raises RuntimeError carrying ffmpeg's own stderr. CalledProcessError prints
    the command and nothing else, so when this failed in CI on 2026-09-03 the
    log showed a 300-character argv and no reason - the run died on beat 8 of
    12 with exit status 8 and the cause was, and remains, unknowable from the
    log. Every one of these arguments is fixed except the image, so the image
    is described too.
    """
    w, h = size
    fps = 30
    frames = max(1, int(duration * fps))
    z_expr = "min(zoom+0.0008,1.18)" if zoom_in else "if(lte(zoom,1.0),1.18,max(zoom-0.0008,1.0))"
    vf = (
        f"scale={w*2}:-1,"
        f"zoompan=z='{z_expr}':d={frames}:s={w}x{h}:fps={fps}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',"
        f"format=yuv420p"
    )
    # `-loop 1` on an input ffmpeg cannot decode does NOT fail: it spins,
    # logging "No JPEG data found in image" forever. Verified locally on
    # 2026-09-04 against an HTML error page saved as .jpg - it ran until killed
    # and then exited 0. Unattended, that burns the whole CI job on the
    # workflow timeout rather than failing one beat. Rendering ~22s of 1080p
    # Ken Burns takes well under a minute on a runner, so this ceiling only
    # ever catches the pathological case.
    budget = max(120.0, duration * 10)
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(image),
             "-t", f"{duration:.3f}", "-vf", vf, "-r", str(fps),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
            capture_output=True, text=True, timeout=budget)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            "ffmpeg hung rendering %s and was killed after %.0fs\n  image: %s\n"
            "  An input it cannot decode makes `-loop 1` retry forever rather "
            "than error."
            % (image.name, budget, _describe_image(image))) from e
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(
            "ffmpeg could not render %s (exit %d)\n  image: %s\n  duration: %.3fs, "
            "target: %dx%d\n  ffmpeg said: %s"
            % (image.name, proc.returncode, _describe_image(image), duration, w, h,
               " | ".join(detail[-4:]) or "(nothing on stderr)"))
    return out
