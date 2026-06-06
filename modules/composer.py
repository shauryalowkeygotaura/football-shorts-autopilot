"""Assemble the 16:9 master documentary.

For each beat: narrate -> fetch still -> Ken-Burns clip sized to the narration
length -> mux that beat's audio. Then concat all beats into one doc and lay a
soft music bed (optional) under the voice. Returns a DocResult carrying the
per-beat time offsets the shorts slicer needs.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402
from modules import broll, tts, visuals  # noqa: E402
from modules.scriptwriter import Script  # noqa: E402


@dataclass
class BeatTiming:
    index: int
    start: float
    end: float
    shortable: bool
    narration: str
    image: Path
    word_timings: list = field(default_factory=list)


@dataclass
class DocResult:
    video_path: Path
    duration: float
    beats: list[BeatTiming]


def _mux_beat(clip: Path, audio: Path, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-i", str(audio),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)],
        check=True, capture_output=True)


def _concat(parts: list[Path], out: Path, work: Path) -> None:
    listfile = work / "concat.txt"
    # Absolute paths: the concat demuxer resolves relative entries against the
    # list file's own directory, which would double the prefix.
    listfile.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts),
                        encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listfile), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", str(out)],
        check=True, capture_output=True)


def build_doc(script: Script, work_dir: Path) -> DocResult:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    size = config.DOC_RESOLUTION

    beat_parts: list[Path] = []
    timings: list[BeatTiming] = []
    cursor = 0.0

    for i, beat in enumerate(script.beats):
        a = tts.narrate(beat.narration, work_dir / "audio", config.VOICE,
                        config.VOICE_RATE, slug=f"beat{i}")
        dur = max(a.duration_sec, 1.0)

        still, found = visuals.fetch_still(beat.visual, work_dir / "img" / f"beat{i}.jpg",
                                           size, subject=script.title_working)
        clip = None
        if not found:
            # Commons whiffed: try generic stock b-roll (Pexels) before
            # accepting the blank gradient card. Dry-safe: returns None when
            # PEXELS_API_KEY is unset.
            clip = broll.fetch_clip(beat.visual, work_dir / f"clip{i}.mp4", dur,
                                    size, beat_index=i)
        if clip is None:
            clip = visuals.ken_burns_clip(still, work_dir / f"clip{i}.mp4", dur, size,
                                          zoom_in=(i % 2 == 0))
        muxed = work_dir / f"beat{i}.mp4"
        _mux_beat(clip, a.audio_path, muxed)
        beat_parts.append(muxed)

        # edge-tts returns timings local to THIS beat (0-based). Shift them by
        # the beat's doc offset so the shorts slicer can filter on the doc
        # timeline; otherwise every beat's words cluster near 0s and only the
        # first short ever gets captions.
        doc_relative = [(w, s + cursor, e + cursor) for (w, s, e) in a.word_timings]
        timings.append(BeatTiming(
            index=i, start=cursor, end=cursor + dur, shortable=beat.shortable,
            narration=beat.narration, image=still, word_timings=doc_relative))
        cursor += dur

    doc = work_dir / "doc_master.mp4"
    _concat(beat_parts, doc, work_dir)
    return DocResult(video_path=doc, duration=cursor, beats=timings)
