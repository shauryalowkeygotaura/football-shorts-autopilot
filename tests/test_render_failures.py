"""What happens when one still will not render.

The 2026-09-03 autopilot run died like this:

    subprocess.CalledProcessError: Command '['ffmpeg', ..., 'beat8.jpg', ...]'
    returned non-zero exit status 8.

That is the entire diagnostic. The command was captured with
capture_output=True, so ffmpeg's own stderr existed and was thrown away -
CalledProcessError prints argv and nothing else. Four minutes of narration and
seven finished beats were lost to beat 8, and the reason is still unknown
because the log cannot say.

Two things are fixed here and pinned below:

  1. A render failure reports ffmpeg's stderr AND the image's real shape. Every
     argument in that command is fixed except the image, so the image is the
     variable worth describing.
  2. One unrenderable still costs a gradient card, not the video. fetch_still
     already treats a MISSING image that way; an unrenderable one is the same
     situation one step later. Past a third of the beats it still fails, because
     that is no longer one awkward photo.

Also pinned: `-loop 1` on an input ffmpeg cannot decode does not fail, it spins
forever. Verified locally against an HTML error page saved as .jpg. Unattended
that burns the CI job on the workflow timeout, so the call is bounded.

    python -m pytest tests/test_render_failures.py -q
"""
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import composer, visuals  # noqa: E402

SIZE = (1920, 1080)


@pytest.fixture
def a_real_jpg(tmp_path):
    p = tmp_path / "real.jpg"
    Image.new("RGB", (1200, 800), (20, 40, 60)).save(p, quality=80)
    return p


@pytest.fixture
def not_an_image(tmp_path):
    p = tmp_path / "beat8.jpg"
    p.write_bytes(b"<html><body>404 Not Found</body></html>")
    return p


# --- the description that was missing from the log ------------------------

def test_describe_reports_a_real_image(a_real_jpg):
    out = visuals._describe_image(a_real_jpg)
    assert "1200x800" in out and "JPEG" in out


def test_describe_calls_out_something_that_is_not_an_image(not_an_image):
    out = visuals._describe_image(not_an_image)
    assert "cannot identify" in out
    assert "39 bytes" in out


def test_describe_survives_a_missing_file(tmp_path):
    assert "unreadable" in visuals._describe_image(tmp_path / "gone.jpg")


# --- failures now say why -------------------------------------------------

def test_a_failed_render_carries_ffmpegs_own_words(monkeypatch, a_real_jpg, tmp_path):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 8, stdout="", stderr="[libx264] height not divisible by 2\nfatal")

    monkeypatch.setattr(visuals.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as e:
        visuals.ken_burns_clip(a_real_jpg, tmp_path / "o.mp4", 3.0, SIZE)
    msg = str(e.value)
    assert "exit 8" in msg
    assert "height not divisible by 2" in msg, "ffmpeg's stderr must reach the log"
    assert "1200x800" in msg, "the one variable input must be described"


def test_a_render_that_says_nothing_still_reports_that(monkeypatch, a_real_jpg, tmp_path):
    monkeypatch.setattr(visuals.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 8, "", ""))
    with pytest.raises(RuntimeError) as e:
        visuals.ken_burns_clip(a_real_jpg, tmp_path / "o.mp4", 3.0, SIZE)
    assert "nothing on stderr" in str(e.value)


def test_a_hung_ffmpeg_is_killed_rather_than_waited_on(monkeypatch, not_an_image, tmp_path):
    """`-loop 1` retries an undecodable input forever instead of erroring."""
    def fake_run(cmd, **kw):
        assert kw.get("timeout"), "the call must be bounded"
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(visuals.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as e:
        visuals.ken_burns_clip(not_an_image, tmp_path / "o.mp4", 3.0, SIZE)
    msg = str(e.value)
    assert "hung" in msg
    assert "cannot identify" in msg


def test_the_timeout_scales_with_the_clip_but_has_a_floor(monkeypatch, a_real_jpg, tmp_path):
    seen = {}

    def fake_run(cmd, **kw):
        seen["timeout"] = kw.get("timeout")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(visuals.subprocess, "run", fake_run)
    visuals.ken_burns_clip(a_real_jpg, tmp_path / "o.mp4", 2.0, SIZE)
    assert seen["timeout"] == 120.0, "a short clip still gets the floor"
    visuals.ken_burns_clip(a_real_jpg, tmp_path / "o.mp4", 30.0, SIZE)
    assert seen["timeout"] == 300.0, "a long clip gets proportionally longer"


# --- one bad beat must not cost the video ---------------------------------

class _Beat:
    def __init__(self, i):
        self.narration = f"beat {i}"
        self.visual = f"visual {i}"
        self.shortable = False


class _Script:
    title_working = "Test Doc"

    def __init__(self, n):
        self.beats = [_Beat(i) for i in range(n)]


class _Audio:
    duration_sec = 2.0
    word_timings: list = []

    def __init__(self, path):
        self.audio_path = path


def _wire(monkeypatch, tmp_path, failing: set[int]):
    """Stub everything but the failure policy under test."""
    rendered: list[Path] = []

    def fake_narrate(text, out_dir, *a, slug="x", **kw):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        p = Path(out_dir) / f"{slug}.mp3"
        p.write_bytes(b"audio")
        return _Audio(p)

    def fake_fetch_still(query, dest, size, subject=None):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"jpg")
        return Path(dest), True

    def fake_ken_burns(image, out, dur, size, zoom_in=True):
        idx = int("".join(c for c in Path(image).stem if c.isdigit()) or -1)
        if idx in failing and "fallback" not in Path(image).stem:
            raise RuntimeError("ffmpeg could not render %s (exit 8)" % Path(image).name)
        Path(out).write_bytes(b"clip")
        rendered.append(Path(out))
        return Path(out)

    def fake_gradient(dest, size):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"gradient")
        return Path(dest)

    monkeypatch.setattr(composer.tts, "narrate", fake_narrate)
    monkeypatch.setattr(composer.visuals, "fetch_still", fake_fetch_still)
    monkeypatch.setattr(composer.visuals, "ken_burns_clip", fake_ken_burns)
    monkeypatch.setattr(composer.visuals, "gradient_still", fake_gradient)
    monkeypatch.setattr(composer, "_mux_beat", lambda c, a, o: Path(o).write_bytes(b"m"))
    monkeypatch.setattr(composer, "_concat", lambda parts, doc, wd: Path(doc).write_bytes(b"d"))
    return rendered


def test_one_unrenderable_beat_becomes_a_gradient_card(monkeypatch, tmp_path, caplog):
    _wire(monkeypatch, tmp_path, failing={8})
    with caplog.at_level("WARNING"):
        result = composer.build_doc(_Script(12), tmp_path)
    assert result.video_path.exists(), "the video must still be produced"
    assert len(result.beats) == 12, "no beat is dropped"
    blob = " ".join(r.getMessage() for r in caplog.records)
    assert "gradient card" in blob
    assert "1 of 12" in blob


def test_a_run_where_most_beats_fail_is_not_shipped(monkeypatch, tmp_path):
    """Past a third, it is not one awkward photo, and a video of blank cards
    is worse than no video."""
    _wire(monkeypatch, tmp_path, failing={0, 1, 2, 3, 4, 5})
    with pytest.raises(RuntimeError) as e:
        composer.build_doc(_Script(12), tmp_path)
    assert "not one bad image" in str(e.value)


def test_a_clean_run_reports_nothing(monkeypatch, tmp_path, caplog):
    _wire(monkeypatch, tmp_path, failing=set())
    with caplog.at_level("WARNING"):
        composer.build_doc(_Script(6), tmp_path)
    assert "gradient card" not in " ".join(r.getMessage() for r in caplog.records)
