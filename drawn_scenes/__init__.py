"""drawn-scenes: code-drawn paper-cutout / hand-drawn b-roll, and an LLM stock-vs-drawn mix."""
from .core import (FPS, ICONS, STYLES, VERSION, Layout, alternate, choose_style, clean_word, decide,
                   fallback_spec, groq_chat, guess_icon, plan, render, still)

MODES = ("stock", "drawn", "mix")

__all__ = ["FPS", "ICONS", "MODES", "STYLES", "VERSION", "Layout", "alternate", "choose_style", "clean_word",
           "decide", "fallback_spec", "groq_chat", "guess_icon", "plan", "render", "still"]
