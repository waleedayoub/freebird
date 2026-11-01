"""Species-name normalization shared by the pipeline, bot and eval scorer.

Models return free-text names ("House sparrow", "Dark-eyed Junco (Slate-colored)",
"European Starling provincial"). Comparing by canonical key keeps lifer detection and
scoring from treating spelling variants as different species. This does NOT restrict
which species a model may report.
"""
from __future__ import annotations

import re
import string

ALIASES = {
    "common starling": "european starling", "european starling provincial": "european starling",
    "starling": "european starling",
    "house sparrow pachysylvia": "house sparrow", "english sparrow": "house sparrow",
    "slate-colored junco": "dark-eyed junco", "junco": "dark-eyed junco",
    "eastern gray squirrel": "squirrel", "gray squirrel": "squirrel", "grey squirrel": "squirrel",
    "eastern grey squirrel": "squirrel", "black squirrel": "squirrel",
    "eastern chipmunk": "chipmunk",
    "rock dove": "rock pigeon", "feral pigeon": "rock pigeon",
    "american goldfinch": "american goldfinch",
    "unknown bird": "unknown", "bird": "unknown", "unidentified bird": "unknown",
}


def norm_species(s: str | None) -> str | None:
    """Lowercase, strip parentheticals, apply aliases. 'sp.'-style entries collapse to 'unknown'."""
    if not s:
        return None
    s = re.sub(r"\s*\(.*?\)", "", s.strip().lower())
    s = ALIASES.get(s, s)
    if s.startswith(("unidentified", "unknown")) or s.endswith(" sp.") or s.endswith(" sp"):
        return "unknown"
    return s


def is_real_species(s: str | None) -> bool:
    n = norm_species(s)
    return bool(n) and n != "unknown"


def display_name(s: str | None) -> str | None:
    """Canonical display form, e.g. 'House sparrow' -> 'House Sparrow'."""
    n = norm_species(s)
    return string.capwords(n) if n else None
