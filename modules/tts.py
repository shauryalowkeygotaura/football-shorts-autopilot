"""edge-tts narration (free, no key). Adapted from philosopher-pipeline/tts.py.

Renders one narration string to mp3 and returns per-word timings (needed so
the shorts slicer can cut on word boundaries and burn synced captions).
`boundary="WordBoundary"` is REQUIRED or word timings collapse to sentences.
"""
from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import edge_tts


@dataclass
class TTSResult:
    audio_path: Path
    duration_sec: float
    word_timings: List[Tuple[str, float, float]]  # (word, start_s, end_s)


async def _render(text: str, voice: str, rate: str, out: Path):
    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    audio = bytearray()
    words: List[Tuple[str, float, float]] = []
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            s = chunk["offset"] / 1e7
            e = (chunk["offset"] + chunk["duration"]) / 1e7
            words.append((chunk["text"], s, e))
    if not audio:
        raise RuntimeError(f"edge-tts produced no audio for {voice!r}")
    out.write_bytes(bytes(audio))
    return words


def _probe(path: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def narrate(text: str, out_dir: Path | str, voice: str, rate: str,
            slug: str) -> TTSResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp3 = out_dir / f"{slug}.mp3"
    timings = asyncio.run(_render(text, voice, rate, mp3))
    return TTSResult(audio_path=mp3, duration_sec=_probe(mp3), word_timings=timings)
