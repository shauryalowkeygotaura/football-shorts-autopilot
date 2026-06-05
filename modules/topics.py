"""Idea machine — the automated version of the gurus' "follow-only-football"
trick. Instead of a phone fed by an algorithm, we harvest candidate topics
with yt-dlp (what's already getting views) + an evergreen seed list, then
score and pick the single best UNUSED topic for today's doc.

Free + no API key: yt-dlp search scraping + a deterministic scorer.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402
import state   # noqa: E402
from modules import visuals  # noqa: E402  (proper-noun + Commons entity check)

# Search probes that surface what's currently pulling views in-niche. Kept
# football/soccer-EXPLICIT: bare "world cup" pulls cricket (huge Indian audience)
# and would poison a football channel.
SEARCH_PROBES = [
    "fifa world cup scandal",
    "fifa world cup untold story",
    "footballer rise to fame soccer",
    "fifa world cup underdog soccer",
    "fifa football corruption explained",
]

# Words that signal high-RPM / high-retention angles per the synthesis.
_HIGH_VALUE = re.compile(
    r"\b(scandal|heist|corruption|stole|fraud|secret|untold|loophole|"
    r"banned|rejected|refugee|underdog|rise|fortune|money|million)\b", re.I)

# Hard reject: other sports that share "World Cup"/"world champion" phrasing.
_OFF_NICHE = re.compile(
    r"\b(cricket|ipl|t20|odi|test match|bcci|rugby|nfl|nba|mlb|baseball|"
    r"tennis|wimbledon|field hockey|kabaddi|wwe|formula\s?1|\bf1\b|motogp|"
    r"badminton|volleyball|handball)\b", re.I)

# Positive football signal a TRENDING title must carry (bare "world cup" is not
# enough — it's sport-ambiguous). Seed topics are curated and skip this.
_FOOTBALL_CTX = re.compile(
    r"\b(football|soccer|footballer|fifa|uefa|premier league|la ?liga|bundesliga|"
    r"serie a|ligue 1|champions league|ballon d'?or|world cup final|messi|ronaldo|"
    r"mbappe|mbappé|neymar|pele|pelé|maradona|ronaldinho|modric|modrić|zidane|"
    r"beckham|haaland|striker|midfielder|goalkeeper|penalty|offside|club|"
    r"barcelona|real madrid|manchester|liverpool|chelsea|arsenal|juventus|"
    r"bayern|psg|brazil|argentina|germany|france|morocco|croatia)\b", re.I)

# Markers of scraped junk we must NOT clone: clickbait series, channel branding,
# hashtag/emoji spam (often the AI-faceless-spam titles that are fabricated and
# therefore can't be illustrated with real photos).
_JUNK = re.compile(
    r"(\bep\.?\s*\d|\bepisode\b|\bpart\s*\d+|\bvol\.?\s*\d|\bseason\b|"
    r"#\w+|\bsubscribe\b|\bshorts?\b|[\U0001F300-\U0001FAFF☀-➿])", re.I)


def _clean_title(title: str) -> str:
    """Strip branding tails and series markers from a scraped title.
    'Rica's Miracle Run | FIFA World Cup Underdog Stories Ep.1' -> 'Rica's Miracle Run'."""
    title = re.split(r"\s[|•·–—]\s", title)[0]                       # branding after a separator
    title = re.sub(r"\s*[-:]\s*(ep(isode)?|part|vol|season)\.?\s*\d.*$", "", title, flags=re.I)
    title = re.sub(r"#\w+", "", title)
    return re.sub(r"\s{2,}", " ", title).strip(" -:|")


@dataclass
class Topic:
    title: str
    source: str               # "trending" | "seed"
    view_count: int = 0
    angle_idx: int = 0
    score: float = field(default=0.0)

    @property
    def key(self) -> str:
        return re.sub(r"[^a-z0-9 ]", "", self.title.lower()).strip()


def _harvest_trending(per_probe: int = 5) -> list[Topic]:
    """Scrape candidate titles + view counts from yt-dlp search (flat, fast)."""
    out: list[Topic] = []
    for probe in SEARCH_PROBES:
        try:
            res = subprocess.run(
                [sys.executable, "-m", "yt_dlp",
                 f"ytsearch{per_probe}:{probe}",
                 "--print", "%(title)s\t%(view_count)s",
                 "--flat-playlist", "--no-warnings"],
                capture_output=True, text=True, timeout=120,
            )
            for line in res.stdout.splitlines():
                if "\t" not in line:
                    continue
                raw, _, views = line.partition("\t")
                if _JUNK.search(raw) or _OFF_NICHE.search(raw):  # spam or wrong sport
                    continue
                if not _FOOTBALL_CTX.search(raw):  # must be unambiguously football
                    continue
                title = _clean_title(raw)
                if len(title) < 12 or not visuals._proper_nouns(title):
                    continue                   # too vague / no concrete subject
                try:
                    vc = int(views.strip())
                except ValueError:
                    vc = 0
                out.append(Topic(title=title, source="trending", view_count=vc))
        except Exception:
            continue
    return out


def _load_seed() -> list[Topic]:
    seed_path = config.ROOT / "topics_seed.md"
    topics: list[Topic] = []
    if not seed_path.exists():
        return topics
    angle_idx = 0
    for line in seed_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("## "):
            angle_idx = min(angle_idx + 1, len(config.ANGLES) - 1)
        elif line.startswith("- "):
            topics.append(Topic(title=line[2:].strip(), source="seed",
                                angle_idx=angle_idx))
    return topics


def score_topic(topic: Topic) -> float:
    """Rank a candidate topic. Higher = make this one first.

    This is the channel's editorial brain. The weighting below is a sane
    default derived from the research synthesis (scandal/money angles win on
    RPM AND views; trending proves live demand; but pure trending decays
    while evergreen seed topics keep earning for years).

    Trade-offs worth tuning to taste:
      - Chase trends harder  -> raise the view_count weight.
      - Build an evergreen library -> raise the seed/high-value weight.
      - Avoid saturated topics -> subtract for very high competitor views.
    """
    score = 0.0
    # 1. High-value angle words (the money/scandal/underdog signal).
    if _HIGH_VALUE.search(topic.title):
        score += 40.0
    # 2. Live demand, but log-damped so one viral outlier can't dominate, and
    #    capped so we don't only chase saturated topics.
    if topic.view_count > 0:
        from math import log10
        score += min(25.0, 6.0 * log10(topic.view_count + 1))
    # 3. Evergreen seed bonus (assets that keep earning after the tournament).
    if topic.source == "seed":
        score += 18.0
    # 4. Prefer the top-ranked money angles (angle_idx 0 = highest value).
    score += max(0.0, 8.0 - 2.0 * topic.angle_idx)
    # 5. Punchy, specific titles retain better than vague ones.
    if 24 <= len(topic.title) <= 70:
        score += 6.0
    return score


def _illustratable(topic: Topic) -> bool:
    """True if the topic's primary entity resolves to a real Commons photo.

    This is the gate that keeps fabricated/AI-spam topics out: a real
    footballer, nation, or trophy has Commons imagery; an invented "Rica's
    Miracle Run" does not, so we'd only ever render blank gradients for it.
    Seed topics are hand-curated real subjects, so they skip the network check.
    """
    if topic.source == "seed":
        return True
    nouns = visuals._proper_nouns(topic.title)
    if not nouns:
        return False
    return visuals._search_commons(" ".join(nouns[:2]), subject="football team") is not None


def pick_topic() -> Topic:
    """Return the highest-scoring ILLUSTRATABLE topic not already turned into a doc."""
    candidates = _harvest_trending() + _load_seed()
    # De-dupe by key, keep the higher view_count instance.
    by_key: dict[str, Topic] = {}
    for t in candidates:
        if t.key in by_key:
            if t.view_count > by_key[t.key].view_count:
                by_key[t.key] = t
        else:
            by_key[t.key] = t

    fresh = [t for t in by_key.values() if not state.is_used(t.key)]
    if not fresh:
        # Everything used — fall back to the full pool (re-angle an old topic).
        fresh = list(by_key.values())

    for t in fresh:
        t.score = score_topic(t)
    fresh.sort(key=lambda t: t.score, reverse=True)

    # Walk down by score and return the first topic we can actually illustrate.
    # Only validate the top slice (network-bounded); seed topics always pass, so
    # a curated real topic is always there as the floor.
    for t in fresh[:12]:
        if _illustratable(t):
            return t
    # Last resort: the best seed (guaranteed real + illustratable).
    seeds = [t for t in fresh if t.source == "seed"]
    return seeds[0] if seeds else fresh[0]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    top = pick_topic()
    print(f"PICKED: {top.title!r}  (score={top.score:.1f}, src={top.source}, views={top.view_count})")
