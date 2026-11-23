"""drawn-scenes papercut "ship it" rocket --size 1920x1080 [--still] [--seconds 3] [--out DIR]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import ICONS, STYLES, render, still


def main() -> int:
    ap = argparse.ArgumentParser(prog="drawn-scenes")
    ap.add_argument("style", choices=STYLES)
    ap.add_argument("word")
    ap.add_argument("icon", choices=sorted(ICONS))
    ap.add_argument("--size", default="1080x1920", help="WIDTHxHEIGHT")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--still", action="store_true", help="write one PNG frame instead of a clip")
    ap.add_argument("--out", type=Path, default=Path("drawn-out"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except ValueError:
        print("--size must look like 1920x1080", file=sys.stderr)
        return 2
    spec = {"word": a.word, "icon": a.icon}
    if a.still:
        a.out.mkdir(parents=True, exist_ok=True)
        print(still(a.style, spec, a.out / f"{a.style}_{a.icon}_{w}x{h}.png", size=(w, h), seed=a.seed))
    else:
        print(render(a.style, spec, a.seconds, a.out, size=(w, h), seed=a.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
