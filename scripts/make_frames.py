"""Render labeled 60 fps clips from the beatmap corpus, split by map.

  uv run scripts/make_frames.py --clips 12 --seconds 2

Writes data/frames/{train,val}/<map id>.npz. Demo maps in data/maps are never used.
"""

from __future__ import annotations

import argparse
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def work(job) -> str:
    path, out, clips, clip_frames, seed = job
    from osu_ai import beatmap, framedata

    try:
        bm = beatmap.parse(path)
        data = framedata.render_clips(bm, clips, clip_frames, random.Random(seed))
    except Exception as exc:
        return f"skip {Path(path).name}: {exc}"
    np.savez_compressed(out, **data)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--out", type=Path, default=Path("data/frames"))
    parser.add_argument("--clips", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--maps", type=int, default=400)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    held_out = {p.stem for p in Path("data/maps").glob("*.osu")}
    paths = sorted(p for p in args.corpus.glob("*.osu") if p.stem not in held_out)
    random.Random(0).shuffle(paths)
    paths = paths[:args.maps]
    n_val = max(4, int(len(paths) * args.val_frac))
    jobs = []
    for i, p in enumerate(paths):
        split = "val" if i < n_val else "train"
        (args.out / split).mkdir(parents=True, exist_ok=True)
        jobs.append((str(p), str(args.out / split / f"{p.stem}.npz"), args.clips, int(args.seconds * 60), i))
    with Pool() as pool:
        for i, msg in enumerate(pool.imap_unordered(work, jobs), 1):
            if msg:
                print(msg)
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} maps", flush=True)
    print(f"done: {len(jobs) - n_val} train maps, {n_val} val maps -> {args.out}")


if __name__ == "__main__":
    main()
