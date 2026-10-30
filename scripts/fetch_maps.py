"""Download a corpus of osu!standard .osu files from osu.ppy.sh (no login needed).

  uv run scripts/fetch_maps.py --count 400

Samples random beatmap IDs, keeps osu!standard maps with enough objects, and
paces requests politely. Files land in data/corpus/<id>.osu.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import httpx

from osu_ai import beatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("data/corpus"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    have = len(list(args.out.glob("*.osu")))
    rng = random.Random(args.seed)
    tried = 0
    with httpx.Client(timeout=15, headers={"User-Agent": "osu-ai research (local)"}) as http:
        while have < args.count:
            bid = rng.randint(50_000, 4_800_000)
            path = args.out / f"{bid}.osu"
            if path.exists():
                continue
            tried += 1
            try:
                resp = http.get(f"https://osu.ppy.sh/osu/{bid}")
            except httpx.HTTPError:
                time.sleep(2)
                continue
            time.sleep(0.5)
            if resp.status_code != 200 or not resp.content:
                continue
            path.write_bytes(resp.content)
            try:
                bm = beatmap.parse(path)
                ok = len(bm.objects) >= 50
            except Exception:
                ok = False
            if not ok:
                path.unlink()
                continue
            have += 1
            if have % 25 == 0:
                print(f"{have}/{args.count} maps ({tried} ids tried)", flush=True)
    print(f"done: {have} maps in {args.out}")


if __name__ == "__main__":
    main()
