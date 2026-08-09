"""Down-sample the 16:9 master doc into vertical 9:16 shorts.

Picks the beats the scriptwriter marked `shortable` (self-contained mini
hook + payoff), pads each to land in the 25-55s sweet spot by bundling the
following beat when too short, then reframes 16:9 -> 9:16 with a blurred
fill background and burns word-synced captions (the visual hook the gurus
say matters more than the written one).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402
from modules.composer import DocResult, BeatTiming  # noqa: E402


def _select_segments(beats: list[BeatTiming]) -> list[tuple[float, float, str]]:
    """Return (start, end, caption_seed) windows to cut, max SHORTS_PER_DOC."""
    lo, hi = config.SHORT_TARGET_SEC
    segments: list[tuple[float, float, str]] = []
    i = 0
    n = len(beats)
    while i < n and len(segments) < config.SHORTS_PER_DOC:
        b = beats[i]
        if not b.shortable:
            i += 1
            continue
        start, end = b.start, b.end
        text = b.narration
        j = i + 1
        # Grow the window forward until it clears the minimum length.
        while (end - start) < lo and j < n:
            end = beats[j].end
            text += " " + beats[j].narration
            j += 1
        # Trim if we overshot the max.
        if (end - start) > hi:
            end = start + hi
        segments.append((start, end, text))
        i = max(j, i + 1)
    return segments


def _srt_timestamp(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_srt(words, seg_start: float, seg_end: float, out: Path) -> Path:
    """Caption file relative to the SHORT timeline (word timings are doc-relative)."""
    lines, idx, group = [], 1, []
    for w, s, e in words:
        if s < seg_start or e > seg_end:
            continue
        group.append((w, s - seg_start, e - seg_start))
        if len(group) >= 3:  # 3 words per caption flash
            txt = " ".join(g[0] for g in group)
            lines.append(f"{idx}\n{_srt_timestamp(group[0][1])} --> {_srt_timestamp(group[-1][2])}\n{txt}\n")
            idx += 1
            group = []
    if group:
        txt = " ".join(g[0] for g in group)
        lines.append(f"{idx}\n{_srt_timestamp(group[0][1])} --> {_srt_timestamp(group[-1][2])}\n{txt}\n")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _escape_filter_path(p: Path) -> str:
    """Escape a path for use inside an ffmpeg filter arg. The Windows drive
    colon (C:) is a filter-syntax separator and must be backslash-escaped, or
    ffmpeg rejects the whole filtergraph with EINVAL."""
    return p.as_posix().replace(":", "\\:")


def _to_ass(srt: Path, out: Path) -> Path:
    """Rewrite the SRT as ASS with an explicit canvas.

    SRT carries no resolution, so libass assumes a 384x288 script canvas and
    scales from there: `FontSize` and `MarginV` in a `force_style` string stop
    meaning pixels. A MarginV of 260 intended as "260px up from the bottom of a
    1920-tall frame" is 260 of 288 lines, which puts the captions at the TOP.
    Stating PlayResX/PlayResY makes every number real pixels.
    """
    w, h = config.SHORT_RESOLUTION
    font_size = int(h * 0.036)
    margin_v = int(h * 0.14)        # clear of the Shorts UI overlay
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{max(3, font_size // 12)},2,2,70,70,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    def ass_ts(srt_ts: str) -> str:
        hh, mm, rest = srt_ts.split(":")
        ss, ms = rest.split(",")
        return f"{int(hh):d}:{mm}:{ss}.{int(ms) // 10:02d}"

    events = []
    for block in srt.read_text(encoding="utf-8").split("\n\n"):
        rows = [r for r in block.strip().splitlines() if r.strip()]
        if len(rows) < 3 or "-->" not in rows[1]:
            continue
        start_ts, end_ts = (p.strip() for p in rows[1].split("-->"))
        text = " ".join(rows[2:]).replace("{", "(").replace("}", ")")
        events.append(f"Dialogue: 0,{ass_ts(start_ts)},{ass_ts(end_ts)},Cap,,0,0,0,,{text}")

    out.write_text((header + "\n".join(events) + "\n") if events else "",
                   encoding="utf-8")
    return out


def _cut_vertical(doc: Path, start: float, end: float, srt: Path, out: Path) -> Path:
    w, h = config.SHORT_RESOLUTION
    # Blurred 9:16 fill + centered 16:9 content + burned captions.
    base_vf = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={w}:{h},boxblur=30:5,setsar=1[bgb];"
        f"[fg]scale={w}:-1[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2[base]"
    )
    # An empty caption file crashes the subtitles filter, so only burn captions
    # when the window actually has caption lines.
    has_captions = srt.exists() and srt.stat().st_size > 0
    if has_captions:
        ass = _to_ass(srt, srt.with_suffix(".ass"))
        has_captions = ass.stat().st_size > 0
    if has_captions:
        # No force_style: the .ass file owns its own geometry (see _to_ass).
        vf = base_vf + f";[base]subtitles='{_escape_filter_path(ass)}'[v]"
    else:
        # No caption lines in this window: relabel the final output [base]->[v].
        vf = base_vf[:-len("[base]")] + "[v]"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(doc),
         "-filter_complex", vf, "-map", "[v]", "-map", "0:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         str(out)],
        check=True, capture_output=True)
    return out


def slice_shorts(doc: DocResult, work_dir: Path) -> list[Path]:
    work_dir = Path(work_dir)
    all_words = [wt for b in doc.beats for wt in b.word_timings]
    outputs: list[Path] = []
    for k, (start, end, _seed) in enumerate(_select_segments(doc.beats)):
        srt = _build_srt(all_words, start, end, work_dir / f"short{k}.srt")
        out = _cut_vertical(doc.video_path, start, end, srt, work_dir / f"short{k}.mp4")
        outputs.append(out)
    return outputs
