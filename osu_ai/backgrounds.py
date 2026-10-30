"""Map backgrounds prepared by scripts/prep_backgrounds.py (data/backgrounds/): which file a map
uses, and the images for training and testing."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data" / "backgrounds"
_FILE = re.compile(r'^\s*(\w+)\s*,\s*[^,]*\s*,\s*"?([^",]+?)"?\s*(?:,|$)')
_SPRITE = re.compile(r'^\s*Sprite\s*,\s*[^,]*,\s*[^,]*,\s*"?([^",]+?)"?\s*(?:,|$)')
_IMAGE = (".jpg", ".jpeg", ".png", ".bmp")


def background_name(osu_text: str) -> str | None:
    """The background file lazer shows for a map (LegacyBeatmapDecoder): the last `0,0,"file"`
    event; a Video event naming an image replaces it; failing both, the first storyboard Sprite."""
    if "[Events]" not in osu_text:
        return None
    events = osu_text.split("[Events]", 1)[1].split("\n[", 1)[0]
    bg = sprite = None
    for line in events.splitlines():
        if line.lstrip().startswith("//"):
            continue
        m = _FILE.match(line)
        if m and m.group(1) == "0":
            bg = m.group(2).strip()
        elif m and m.group(1) in ("1", "Video") and m.group(2).strip().lower().endswith(_IMAGE):
            bg = m.group(2).strip()
        elif sprite is None and (sm := _SPRITE.match(line)):
            sprite = sm.group(1).strip()
    return bg or sprite


def set_id(osu_text: str) -> int | None:
    m = re.search(r"^BeatmapSetID:\s*(-?\d+)", osu_text, re.M)
    return int(m.group(1)) if m else None


def index() -> list[dict]:
    path = DIR / "index.json"
    return json.loads(path.read_text()) if path.exists() else []


def paths(split: str) -> list[str]:
    """The .npy files of one split ("train" or "test")."""
    return [str(ROOT / i["npy"]) for i in index() if i["split"] == split]


def _known_set(raw: bytes) -> int | None:
    """Old .osu files carry no BeatmapSetID; data/real_maps.json knows the test maps' sets."""
    import hashlib

    path = ROOT / "data" / "real_maps.json"
    md5 = hashlib.md5(raw).hexdigest()
    maps = json.loads(path.read_text()) if path.exists() else []
    return next((m["set_id"] for m in maps if m.get("osu_md5") == md5), None)


def for_map(osu_path: str | Path) -> np.ndarray | None:
    """The map's own background (RGB uint8, covering 640x480), if it was prepared."""
    raw = Path(osu_path).read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    name = background_name(text)
    sid = set_id(text) if set_id(text) not in (None, -1) else _known_set(raw)
    idx = index()
    if name is None:  # lazer shows its default background (Backgrounds/bg1)
        hits = [i for i in idx if i["set"] == 0 and i["file"].lower() == "bg1.jpg"]
    else:
        same = [i for i in idx if i["file"].lower() == name.lower()]
        hits = [i for i in same if i["set"] == sid]
        if not hits and sid is None and len({i["set"] for i in same}) == 1:
            hits = same  # unknown set, but only one set has a file of that name
    return np.load(ROOT / hits[0]["npy"]) if hits else None
