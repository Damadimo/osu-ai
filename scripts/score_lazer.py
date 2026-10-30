"""Score the lazer-vision model on whole maps in closed loop (lazer-accurate frames and rules).

  uv run --extra train scripts/score_lazer.py data/lazer_osu/827932.osu [--weights ...] [--hit-animations]

This is the last offline check before the real game: same frames, same rules, no latency.
"""

from __future__ import annotations

import argparse
import statistics

from osu_ai import beatmap
from osu_ai.agent import LazerAgent
from osu_ai.lazer_frames import Look
from osu_ai.lazer_sim import LazerSim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("maps", nargs="+")
    parser.add_argument("--weights", default="models/osunet-lazer.safetensors")
    parser.add_argument("--hit-animations", action="store_true")
    parser.add_argument("--snaking", action="store_true")
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument("--lead-ms", type=float, default=0.0, help="decide presses this early (real-game latency)")
    args = parser.parse_args()
    agent = LazerAgent(args.weights)
    look = Look(hit_animations=args.hit_animations, snaking=args.snaking)
    for path in args.maps:
        bm = beatmap.parse(path)
        sim = LazerSim(bm, agent, look, jitter_ms=args.jitter_ms, lead_ms=args.lead_ms)
        score = sim.run()
        by_kind: dict[str, list[int]] = {}
        for o, j in zip(bm.objects, score.judgments):
            by_kind.setdefault(o.kind.value, []).append(j)
        offs = [abs(x) for x in score.head_offsets]
        print(f"{bm.name[:60]}: {score.summary()}")
        print("   " + "  ".join(f"{k}: {sum(v) / (300 * len(v)):.0%} of {len(v)}" for k, v in by_kind.items())
              + (f" | timing error median {statistics.median(offs):.0f} ms, p90 {sorted(offs)[int(0.9 * len(offs))]:.0f} ms"
                 if offs else "")
              + f" | model {statistics.median(sim.infer_ms):.1f} ms/frame", flush=True)


if __name__ == "__main__":
    main()
