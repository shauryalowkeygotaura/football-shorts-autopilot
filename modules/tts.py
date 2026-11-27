"""Groq Orpheus narration via paralang (replaces edge-tts).

edge-tts failed 7 of 8 CI runs (2026-11-19..27) with NoAudioReceived, and the
voice-stack rule is Orpheus/Andromeda, never edge-tts. paralang does the
Orpheus call (key pool, 429/401 cooldowns, network retries) and the word
alignment (Groq Whisper word timestamps, captions keep the script's spelling).

Renders one narration string to mp3 and returns per-word timings (the shorts
slicer cuts on word boundaries and burns synced captions from them).
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from paralang import orpheus

MAX_CHARS = 190  # Groq Orpheus rejects input over 200 chars; keep a margin


@dataclass
class TTSResult:
    audio_path: Path
    duration_sec: float
    word_timings: List[Tuple[str, float, float]]  # (word, start_s, end_s)


def split_for_orpheus(text: str, limit: int = MAX_CHARS) -> List[str]:
    """Pieces <= limit chars, cut at sentence ends, then commas, then spaces."""
    text = " ".join(text.split())
    pieces: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        while len(sentence) > limit:
            cut = max(sentence.rfind(", ", 0, limit), sentence.rfind("; ", 0, limit))
            cut = cut + 1 if cut > 0 else sentence.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    # merge short neighbours back up to the limit: fewer requests, fewer seams
    merged: List[str] = []
    for p in pieces:
        if merged and len(merged[-1]) + 1 + len(p) <= limit:
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    return merged


def _to_mp3(pcm: bytes, mp3: Path) -> None:
    wav = mp3.with_suffix(".wav")
    wav.write_bytes(orpheus._wav(pcm, orpheus.SR))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                    "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)], check=True)
    wav.unlink(missing_ok=True)


def narrate(text: str, out_dir: Path | str, voice: str, rate: str,
            slug: str) -> TTSResult:
    """`rate` is kept for call-site compatibility; Orpheus has no speed control
    and time-stretching it is off the table (it degrades the voice)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp3 = out_dir / f"{slug}.mp3"
    pieces = split_for_orpheus(text)
    if not pieces:
        raise RuntimeError(f"empty narration for {slug}")
    pcm = b"".join(orpheus.synth(p, voice=voice) for p in pieces)
    timings = orpheus.align(pcm, text)
    if not timings:
        raise RuntimeError(f"Whisper alignment returned no words for {slug}")
    _to_mp3(pcm, mp3)
    duration = len(pcm) / 2 / orpheus.SR  # exact for 16-bit mono, no ffprobe needed
    return TTSResult(audio_path=mp3, duration_sec=duration, word_timings=timings)
