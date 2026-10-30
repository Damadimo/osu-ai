"""Can Jev read osu! from a screenshot? Box the next circle, call its timing, measure latency.

  uv run scripts/probe_vision.py data/maps/75.osu --n 20 [--model typesafe-ai/jev-1.14.0]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import math
import os
import random
import statistics
import time
import uuid

import httpx
from dotenv import load_dotenv

from osu_ai import beatmap, frames
from osu_ai.beatmap import Kind

URL = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
BUCKETS = [f"{a}-{a + 50}ms" for a in range(0, 600, 50)] + ["later"]


def bucket(ms: float) -> str:
    return "later" if ms >= 600 else f"{int(max(ms, 0) // 50) * 50}-{int(max(ms, 0) // 50) * 50 + 50}ms"


def request_body(model: str, png: bytes, preempt: float) -> dict:
    ref = f"screen-{uuid.uuid4()}"
    return {
        "model": model,
        "state": {
            "screen": ref,
            "game": "osu!: each numbered circle must be clicked at the moment its thin outer approach ring "
                    f"shrinks onto it. Rings start at 4x the circle's size and close over {preempt / 1000:.2f} s.",
        },
        "assets": {ref: {"type": "image", "format": "png", "data_base64": base64.b64encode(png).decode()}},
        "questions": {
            "next": {"type": "box", "instructions": "the filled numbered circle in `screen` whose approach ring "
                                                    "is closest to touching it (the next one to click)"},
            "when": {"type": "choice",
                     "instructions": "How long from now until the approach ring of the next circle to click in "
                                     "`screen` touches it?",
                     "criteria": {b: None for b in BUCKETS}},
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--model", default="typesafe-ai/jev-1.14.0")
    args = parser.parse_args()
    load_dotenv()

    bm = beatmap.parse(args.map)
    colors = frames.combo_colors(bm)
    rng = random.Random(0)
    preempt, radius = bm.difficulty.preempt, bm.difficulty.radius
    circles = [o for o in bm.objects if o.kind is not Kind.SPINNER]
    headers = {"Authorization": f"Bearer {os.environ['AI_GATEWAY_API_KEY']}",
               "TypeSafe-Danger-Zone-Images": "true"}
    hits, near_right, errs, lat, tokens = 0, [], [], [], []
    async with httpx.AsyncClient(timeout=30, http2=True) as http:
        for i in range(args.n):
            target = rng.choice(circles)
            t = target.time - rng.uniform(0, 500)
            done = {o.index for o in bm.objects if o.time < t and o.kind is Kind.CIRCLE}
            # The real "next" object is the earliest one still due.
            nxt = min((o for o in circles if o.time >= t and o.index not in done), key=lambda o: o.time)
            png = frames.png_bytes(frames.render(bm, t, colors, done=done))
            start = time.perf_counter()
            resp = await http.post(URL, headers=headers, json=request_body(args.model, png, preempt))
            ms = (time.perf_counter() - start) * 1000
            if resp.status_code != 200:
                print(f"  {i}: HTTP {resp.status_code} {resp.text[:200]}")
                continue
            data = resp.json()
            lat.append(ms)
            tokens.append(data.get("usage", {}).get("input_tokens", 0))
            box = data["answers"]["next"]["box"]
            cx = (box["left"] + box["right"]) / 2 * frames.W
            cy = (box["top"] + box["bottom"]) / 2 * frames.H
            px, py = frames.to_playfield(cx, cy)
            off = math.dist((px, py), nxt.pos)
            hit = off <= radius
            hits += hit
            said = data["answers"]["when"]["choice"]
            truth = nxt.time - t
            right = said == bucket(truth)
            near_right.append(right)
            if said != "later":
                errs.append(abs(int(said.split("-")[0]) + 25 - truth))
            print(f"  {i:2d}: box {'HIT ' if hit else 'miss'} ({off:5.0f}px off)  said {said:>10}  real {truth:4.0f} ms "
                  f"{'ok' if right else 'x'}   {ms:4.0f} ms, {tokens[-1]} tokens")
    if lat:
        print(f"\nbox lands on the right circle: {hits}/{len(lat)}")
        print(f"timing in the right 50 ms bucket: {sum(near_right)}/{len(near_right)}"
              + (f", median error {statistics.median(errs):.0f} ms" if errs else ""))
        print(f"latency {statistics.median(lat):.0f} ms typical, max {max(lat):.0f}; ~{statistics.median(tokens):.0f} input tokens")


if __name__ == "__main__":
    asyncio.run(main())
