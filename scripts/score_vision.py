"""Score the vision model playing maps offline (60 fps game time, as fast as possible).

  uv run --extra train scripts/score_vision.py data/maps/75.osu data/maps/129891.osu
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter

from osu_ai import beatmap, game
from osu_ai.vision_live import VisionAgent, VisionGame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("maps", nargs="+")
    parser.add_argument("--weights", default="models/osunet.safetensors")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    agent = VisionAgent(args.weights)
    for path in args.maps:
        bm = game.segment(beatmap.parse(path), None, args.seconds)
        agent.history.clear()
        g = VisionGame(bm, agent)
        score = g.run_offline()
        by_kind = {}
        for o, j in zip(bm.objects, score.judgments):
            by_kind.setdefault(o.kind.value, []).append(j)
        offs = [abs(x) for x in score.head_offsets]
        print(f"{bm.name[:50]}: {score.summary()}")
        print("   " + "  ".join(f"{k}: {sum(v) / (300 * len(v)):.0%} of {len(v)}" for k, v in by_kind.items())
              + (f" | median timing error {statistics.median(offs):.0f} ms" if offs else ""))


if __name__ == "__main__":
    main()
