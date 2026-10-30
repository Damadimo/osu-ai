"""Score a local decision model on held-out osu! questions.

  uv run --extra train scripts/eval_tev.py --model Qwen/Qwen3.5-0.8B [--adapter adapters/osu-0.8b] [--n 300]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from osu_ai.dataset import key_center_ms
from osu_ai.tev import TevModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--adapter")
    parser.add_argument("--data", type=Path, default=Path("data/tev_osu/records.test.jsonl"))
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    records = [json.loads(line) for line in open(args.data)]
    random.Random(0).shuffle(records)
    records = records[:args.n]
    model = TevModel(args.model, args.adapter)
    model.decide(records[0])  # warm up

    timing_right, near_right, errors, actions, lat = [], [], [], [], []
    for r in records:
        key, _, ms = model.decide(r)
        lat.append(ms)
        meta = r["meta"]
        if meta["kind"] == "action":
            actions.append(key == meta["key"])
            continue
        timing_right.append(key == meta["key"])
        if meta["key"] != "later":
            near_right.append(key == meta["key"])
            said = key_center_ms(key)
            errors.append(abs((said if said is not None else 400) - max(meta["ms_left"], 0)))

    print(f"{model.name} on {len(records)} held-out questions")
    print(f"  timing, exact 25 ms window: {sum(timing_right)}/{len(timing_right)} ({statistics.mean(timing_right):.0%})")
    print(f"  timing, rings closing within 300 ms: {statistics.mean(near_right):.0%} exact, "
          f"median error {statistics.median(errors):.0f} ms, within ±40 ms: "
          f"{sum(e <= 40 for e in errors) / len(errors):.0%}")
    if actions:
        print(f"  tap/hold/spin: {statistics.mean(actions):.0%}")
    print(f"  latency {statistics.median(lat):.0f} ms per question")


if __name__ == "__main__":
    main()
