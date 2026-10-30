"""Play an osu! beatmap with a brain, print the score, and save a run log.

  uv run scripts/play.py data/maps/75.osu                      # perfect-knowledge baseline
  uv run scripts/play.py data/maps/75.osu --latency 250 300    # same, with decision latency
  uv run scripts/play.py data/maps/75.osu --brain jev --seconds 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from osu_ai import beatmap, game, osr
from osu_ai.brains import AutoBrain, DelayedAutoBrain, JevBrain
from osu_ai.hands import CORRECT_ACTION
from osu_ai.systemone import BACKENDS, SystemOneClient

# brain name -> System One backend that serves it
MODEL_BRAINS = {"jev": "vercel", "jev-direct": "typesafe", "laya": "laya", "laya-typed": "laya-typed"}


def save(run: game.Run, map_path: str) -> Path:
    objs = {o.index: o for o in run.beatmap.objects}
    rows = []
    for rec, judgment in zip(run.records, run.score.judgments):
        a = rec.answer
        rows.append({
            "index": rec.index, "kind": rec.kind, "hit_time": objs[rec.index].time, "asked_at": rec.asked_at,
            "latency_ms": a and round(a.latency_ms, 1), "action": a and a.action.value,
            "correct_action": CORRECT_ACTION[objs[rec.index].kind].value, "confidence": a and a.confidence,
            "telemetry": a and a.telemetry, "error": rec.error, "judgment": judgment,
        })
    out = Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-{run.brain}-{Path(map_path).stem}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "map": run.beatmap.name, "map_md5": run.beatmap.md5, "brain": run.brain,
        "accuracy": run.score.accuracy, "counts": dict(run.score.counts), "max_combo": run.score.max_combo,
        "objects": rows,
    }, indent=1))
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map")
    parser.add_argument("--brain", choices=["auto", *MODEL_BRAINS], default="auto")
    parser.add_argument("--latency", type=float, nargs="+", help="auto brain: latency samples (ms)")
    parser.add_argument("--start", type=float, help="seconds into the map to start")
    parser.add_argument("--seconds", type=float, help="how many seconds to play")
    parser.add_argument("--speed", type=float, default=1.0, help="send requests faster than real time")
    args = parser.parse_args()

    load_dotenv()
    bm = game.segment(beatmap.parse(args.map), args.start, args.seconds)
    d = bm.difficulty
    print(f"{bm.name}: {len(bm.objects)} objects, AR{d.ar:g} OD{d.od:g} CS{d.cs:g} "
          f"(objects appear {d.preempt:.0f} ms early, 300 window ±{d.windows.great:.0f} ms)")
    if args.brain in MODEL_BRAINS:
        async with SystemOneClient(BACKENDS[MODEL_BRAINS[args.brain]]) as client:
            brain = JevBrain(client, name=args.brain)
            run = await game.run(bm, brain, realtime=True, speed=args.speed, progress=True)
    else:
        brain = DelayedAutoBrain(args.latency) if args.latency else AutoBrain()
        run = await game.run(bm, brain, realtime=False)
    print(f"{run.brain}: {run.score.summary()}")
    print(f"  {run.decision_summary()}")
    log = save(run, args.map)
    print(f"  log: {log}")
    print(f"  replay (drag into osu!lazer to watch): {osr.export(run.beatmap, run.replay, run.score, log.with_suffix('.osr'), run.brain)}")


if __name__ == "__main__":
    asyncio.run(main())
