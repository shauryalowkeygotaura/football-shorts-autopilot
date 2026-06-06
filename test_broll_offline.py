"""Offline tests for the b-roll layer: no network, no API keys.
Run: python test_broll_offline.py
"""
import os
import py_compile
from pathlib import Path

for f in ["modules/broll.py", "modules/visuals.py", "modules/composer.py"]:
    py_compile.compile(f, doraise=True)
print("compile OK")

os.environ.pop("PEXELS_API_KEY", None)
os.environ.pop("GROQ_API_KEY", None)
from modules import broll  # noqa: E402

# dry-safe: no key -> None, no exception
r = broll.fetch_clip("Ronaldinho celebrating", Path("nope.mp4"), 5.0,
                     (1920, 1080), beat_index=2)
assert r is None, r

# term fallback without Groq is deterministic by beat index
t = broll.generic_term("Zidane headbutt", beat_index=3)
assert t == "stadium floodlights", t

# empty-key guard stays soft (never raises)
assert broll._next_key([]) == ""

# search with no keys -> [] (never raises)
assert broll.search("stadium", (1920, 1080)) == []

print("offline smoke OK")
