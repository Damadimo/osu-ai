"""Score the lazer-vision model on whole maps in closed loop (lazer-accurate frames and rules).

  uv run --extra train scripts/score_lazer.py data/lazer_osu/827932.osu [--weights ...] [--hit-animations]
  uv run --extra train scripts/score_lazer.py data/lazer_osu/827932.osu --dim 0.7   # the map's background at 70% dim

This is the last offline check before the real game: same frames, same rules, no latency.
"""

from __future__ import annotations

import argparse
import statistics

from osu_ai import backgrounds, beatmap
from osu_ai.agent import LazerAgent
from osu_ai.lazer_frames import Look, View, backdrop, luma
from osu_ai.lazer_sim import LazerSim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("maps", nargs="+")
    parser.add_argument("--weights", default="models/osunet-lazer.safetensors")
    parser.add_argument("--hit-animations", action="store_true")
    parser.add_argument("--snaking", action="store_true")
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument("--lead-ms", type=float, default=0.0, help="decide presses this early (real-game latency)")
    parser.add_argument("--dim", type=float, default=1.0, help="background dim; below 1 shows the map's own background")
    args = parser.parse_args()
    agent = LazerAgent(args.weights)
    look = Look(hit_animations=args.hit_animations, snaking=args.snaking)
    for path in args.maps:
        bm = beatmap.parse(path)
        bg = None
        if args.dim < 1:
            rgb = backgrounds.for_map(path)
            if rgb is None:
                print(f"{path}: no prepared background (scripts/prep_backgrounds.py); skipping")
                continue
            bg = backdrop(luma(rgb), View(agent.input_size), dim=args.dim)
        sim = LazerSim(bm, agent, look, jitter_ms=args.jitter_ms, lead_ms=args.lead_ms, background=bg)
        score = sim.run()
        by_kind: dict[str, list[int]] = {}
        for o, j in zip(bm.objects, score.judgments):
            by_kind.setdefault(o.kind.value, []).append(j)
        offs = [abs(x) for x in score.head_offsets]
        print(f"{bm.name[:60]}{f' (dim {args.dim:g})' if args.dim < 1 else ''}: {score.summary()}")
        print("   " + "  ".join(f"{k}: {sum(v) / (300 * len(v)):.0%} of {len(v)}" for k, v in by_kind.items())
              + (f" | timing error median {statistics.median(offs):.0f} ms, p90 {sorted(offs)[int(0.9 * len(offs))]:.0f} ms"
                 if offs else "")
              + f" | model {statistics.median(sim.infer_ms):.1f} ms/frame", flush=True)


if __name__ == "__main__":
    main()
