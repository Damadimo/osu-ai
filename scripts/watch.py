"""Watch a model play osu! live in a window, with the song.

  uv run scripts/watch.py data/maps/75.osu                  # Jev via Vercel AI Gateway
  uv run scripts/watch.py data/maps/75.osu --brain laya     # local Laya server
  uv run scripts/watch.py data/maps/75.osu --brain oracle --headless --seconds 20   # harness check

The model decides when every ring will close and how to play each object;
the cursor only does what it says. Each session saves a log and an .osr replay.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

from osu_ai import beatmap, game, judge, osr
from osu_ai.beatmap import Kind
from osu_ai.live import Session
from osu_ai.pilot import ModelPilot, OraclePilot
from osu_ai.systemone import BACKENDS, SystemOneClient

MODEL_BRAINS = {"jev": "vercel", "jev-direct": "typesafe", "laya": "laya", "laya-typed": "laya-typed"}


def find_audio(map_path: str) -> Path | None:
    for ext in (".mp3", ".ogg"):
        p = Path("data/audio") / f"{Path(map_path).stem}{ext}"
        if p.exists():
            return p
    return None


async def headless(session: Session) -> None:
    thinker = asyncio.create_task(session.think())
    while not session.finished:
        session.step()
        await asyncio.sleep(0.004)
    thinker.cancel()


def report(session: Session, bm, map_path: str) -> None:
    score = judge.score(bm, session.hands.replay)
    st = session.stats
    reads = [p for p in st.reads if p.bucket != "spinner"]
    print(f"{session.pilot.name}: {score.summary()}")
    lat = sorted(st.all_latencies)
    if lat:
        print(f"  latency {statistics.median(lat):.0f} ms typical, {lat[int(0.95 * (len(lat) - 1))]:.0f} ms p95")
    print(f"  requests answered {st.ok}/{st.sent}" + "".join(f", {n}x {e}" for e, n in st.errors.items()))
    if reads:
        right = sum(p.read_correct for p in reads)
        print(f"  timing reads in the right 50 ms bucket: {right}/{len(reads)} ({right / len(reads):.0%})")
    if score.head_offsets:
        offs = [abs(o) for o in score.head_offsets]
        print(f"  clicks that landed: {len(offs)}, typical timing error {statistics.median(offs):.0f} ms")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path("runs") / f"{stamp}-live-{session.pilot.name}-{Path(map_path).stem}"
    out.parent.mkdir(exist_ok=True)
    kinds = {o.index: o.kind.value for o in bm.objects}
    out.with_suffix(".json").write_text(json.dumps({
        "map": bm.name, "pilot": session.pilot.name, "accuracy": score.accuracy,
        "counts": dict(score.counts), "max_combo": score.max_combo,
        "requests": {"sent": st.sent, "ok": st.ok, "errors": dict(st.errors), "input_tokens": st.tokens},
        "latencies_ms": [round(x, 1) for x in st.all_latencies],
        "reads": [{"index": p.index, "kind": kinds[p.index], "asked_at": p.asked_at, "said": p.bucket,
                   "truth_ms": round(p.truth_ms, 1), "action": p.action.value, "right": p.read_correct}
                  for p in st.reads],
        "judgments": score.judgments,
    }, indent=1))
    osr.export(bm, session.hands.replay, score, out.with_suffix(".osr"), session.pilot.name)
    print(f"  log: {out.with_suffix('.json')}\n  replay: {out.with_suffix('.osr')}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map")
    parser.add_argument("--brain", choices=[*MODEL_BRAINS, "oracle", "tev"], default="jev")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="tev brain: base model")
    parser.add_argument("--adapter", default="adapters/osu-qwen3-0.6b", help="tev brain: LoRA adapter ('' for none)")
    parser.add_argument("--start", type=float, help="seconds into the map to start")
    parser.add_argument("--seconds", type=float, help="how many seconds to play")
    parser.add_argument("--tick", type=float, help="ms between model requests (default 150; 40 for tev)")
    parser.add_argument("--in-flight", type=int, help="max concurrent requests (default 4; 1 for tev)")
    parser.add_argument("--headless", action="store_true", help="no window or sound")
    args = parser.parse_args()

    local = args.brain == "tev"
    args.tick = args.tick or (40 if local else 150)
    args.in_flight = args.in_flight or (1 if local else 4)
    load_dotenv()
    bm = game.segment(beatmap.parse(args.map), args.start, args.seconds)
    d = bm.difficulty
    print(f"{bm.name}: {len(bm.objects)} objects, AR{d.ar:g} OD{d.od:g} "
          f"(rings close in {d.preempt:.0f} ms, 300 window ±{d.windows.great:.0f} ms)")

    async def play(pilot) -> None:
        session = Session(bm, pilot, tick_ms=args.tick, max_in_flight=args.in_flight)
        if args.headless:
            await headless(session)
        else:
            from osu_ai.viewer import Viewer  # pygame only when there's a window
            await Viewer(session, bm.name, find_audio(args.map)).run()
        report(session, bm, args.map)

    if args.brain == "oracle":
        await play(OraclePilot(250))
    elif args.brain == "tev":
        from osu_ai.tev_pilot import TevPilot  # MLX only when used
        name = "Tev (fine-tuned)" if args.adapter else "Tev (base)"
        await play(TevPilot(bm, args.model, args.adapter or None, name))
    else:
        async with SystemOneClient(BACKENDS[MODEL_BRAINS[args.brain]]) as client:
            await play(ModelPilot(client, args.brain.capitalize() if args.brain == "jev" else args.brain))


if __name__ == "__main__":
    asyncio.run(main())
