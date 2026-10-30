"""Turn the beatmap corpus into train/valid/test sets (split by map).

  uv run scripts/build_dataset.py --per-object 3

Writes data/tev_osu/{train,valid,test}.jsonl as chat messages for mlx-lm, plus
records.*.jsonl in Together's tev1 schema (state/question/options/answer).
Maps used for live demos (data/maps) never go into training.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from osu_ai import beatmap
from osu_ai.dataset import chat, records_for_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--out", type=Path, default=Path("data/tev_osu"))
    parser.add_argument("--per-object", type=float, default=3.0)
    parser.add_argument("--max-per-map", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    held_out = {p.stem for p in Path("data/maps").glob("*.osu")}
    paths = sorted(p for p in args.corpus.glob("*.osu") if p.stem not in held_out)
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    n_valid, n_test = max(2, len(paths) // 20), max(4, len(paths) // 10)
    splits = {"valid": paths[:n_valid], "test": paths[n_valid:n_valid + n_test], "train": paths[n_valid + n_test:]}

    args.out.mkdir(parents=True, exist_ok=True)
    for split, split_paths in splits.items():
        records = []
        for p in split_paths:
            try:
                bm = beatmap.parse(p)
            except Exception as exc:
                print(f"skip {p.name}: {exc}")
                continue
            recs = records_for_map(bm, args.per_object, rng)
            rng.shuffle(recs)
            for r in recs[:args.max_per_map]:
                r["meta"]["map"] = p.stem
                records.append(r)
        rng.shuffle(records)
        with open(args.out / f"{split}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(chat(r)) + "\n")
        with open(args.out / f"records.{split}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        kinds = Counter(r["meta"]["kind"] for r in records)
        answers = Counter(r["meta"]["key"] for r in records if r["meta"]["kind"] == "timing")
        print(f"{split}: {len(split_paths)} maps, {len(records)} records {dict(kinds)}; "
              f"'later' share {answers['later'] / max(1, kinds['timing']):.0%}")


if __name__ == "__main__":
    main()
