"""Prepare captured osu!lazer footage for training: align each recording to game time.

  uv run scripts/prep_real.py data/real/*.npz --out data/real/prepared.json

For every recording (realdata.Recording) this finds the clock offset that makes our renderer
match the footage, then writes the frames as a plain .npy (memory-mapped by the training
workers), each frame's game time, and what happened to every object: autoplay hits each one
exactly on time; a replay (ours, or lazer's export of a live run) is judged with the lazer
rules. About 15% of the maps go to validation, chosen by map hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from osu_ai import beatmap, judge
from osu_ai.lazer_frames import Look, perfect
from osu_ai.realdata import Recording, align, outcome_from_score, replay_from_osr

MIN_MATCH = 0.8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/real/prepared.json"))
    args = parser.parse_args()
    done = {it["source"]: it for it in json.loads(args.out.read_text())} if args.out.exists() else {}
    for path in args.recordings:
        if str(path) in done:
            continue
        rec = Recording.load(path)
        meta = rec.meta
        bm = beatmap.parse(meta["map"])
        look = Look(**meta.get("look", {}))
        mode = meta.get("mode", "autoplay")
        if mode == "autoplay":
            outcome = perfect(bm)
        elif meta.get("replay"):  # our replay, or lazer's export of a live run: judged as lazer would
            outcome = outcome_from_score(bm, judge.score(bm, replay_from_osr(Path(meta["replay"]), bm)))
        else:
            print(f"skip {path.name}: {mode} footage without a replay to judge")
            continue
        c, info = align(rec, bm, outcome, look)
        print(f"{path.name}: offset {c:.2f} ms, match {info['match']:.4f}", flush=True)
        if info["match"] < MIN_MATCH:
            print(f"  skipped: footage does not match the renderer well enough (< {MIN_MATCH})")
            continue
        frames_path, times_path = path.with_suffix(".frames.npy"), path.with_suffix(".times.npy")
        np.save(frames_path, rec.frames)
        np.save(times_path, rec.pts_ms + c)
        split = "val" if int(hashlib.md5(meta["map"].encode()).hexdigest(), 16) % 7 == 0 else "train"
        done[str(path)] = {
            "source": str(path), "map": meta["map"], "frames": str(frames_path), "times": str(times_path),
            "offset_ms": c, "match": info["match"], "split": split, "look": meta.get("look", {}),
            "judged": {str(i): v for i, v in outcome.judged.items()}, "spin_rate": {},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(list(done.values()), indent=1))
    print(f"{len(done)} recordings in {args.out}")


if __name__ == "__main__":
    main()
