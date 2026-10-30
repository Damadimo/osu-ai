"""Measure System One round-trip latency with an osu!-sized request.

Usage: uv run scripts/probe_latency.py [--backend vercel] [--n 20] [--burst 5]
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import Counter

from dotenv import load_dotenv

from osu_ai.systemone import BACKENDS, SystemOneClient, SystemOneError

# A mid-map moment: three objects on screen, B is due first and is a slider.
STATE = {
    "playfield": "osu! playfield seen from the player's view",
    "cursor": "near the center",
    "visible_objects": {
        "A": {"kind": "hit circle", "where": "top left", "approach_ring": "just appeared, very large"},
        "B": {"kind": "slider", "where": "center right", "approach_ring": "almost touching its circle"},
        "C": {"kind": "hit circle", "where": "bottom center", "approach_ring": "about half closed"},
    },
}

QUESTIONS = {
    "next": {
        "type": "choice",
        "instructions": "Which object has to be hit first? It is the one whose approach ring is closest to touching its circle.",
        "criteria": {"A": None, "B": None, "C": None},
    },
    "action": {
        "type": "choice",
        "instructions": "How is object `B` played?",
        "criteria": {
            "tap": "click once (hit circle)",
            "hold": "press and follow the ball until the end (slider)",
            "spin": "hold and spin around the center (spinner)",
        },
    },
}
EXPECTED = {"next": "B", "action": "hold"}


def report(label: str, results: list[float | str]) -> None:
    ms = sorted(r for r in results if isinstance(r, float))
    errors = Counter(r for r in results if isinstance(r, str))
    line = f"{label:<24} ok {len(ms)}/{len(results)}"
    if ms:
        p95 = ms[min(len(ms) - 1, round(0.95 * (len(ms) - 1)))]
        line += (f"  p50={statistics.median(ms):5.0f} ms  p95={p95:5.0f} ms  "
                 f"min={ms[0]:4.0f}  max={ms[-1]:5.0f}")
    print(line)
    for err, n in errors.items():
        print(f"    {n}x {err}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="vercel", choices=sorted(BACKENDS))
    parser.add_argument("--n", type=int, default=20, help="sequential requests")
    parser.add_argument("--burst", type=int, default=5, help="concurrent requests per burst")
    parser.add_argument("--gap", type=float, default=0.2, help="seconds between sequential requests")
    parser.add_argument("--wait-minutes", type=float, default=2, help="keep retrying a refused first request")
    args = parser.parse_args()

    load_dotenv()
    async with SystemOneClient(BACKENDS[args.backend]) as client:
        deadline = time.monotonic() + args.wait_minutes * 60
        attempt = 0
        while True:  # the upstream sheds load with 429s; back off until it answers
            try:
                first = await client.decide(STATE, QUESTIONS)  # includes TLS + connection setup
                break
            except SystemOneError as exc:
                wait = min(2**attempt, 60)
                if time.monotonic() + wait > deadline:
                    raise SystemExit(f"still refused after {args.wait_minutes:g} min: {str(exc)[:120]}")
                attempt += 1
                print(f"  {time.strftime('%H:%M:%S')} attempt {attempt}: {str(exc)[:60]}… retry in {wait}s",
                      flush=True)
                await asyncio.sleep(wait)
        print(f"model={first.model}  usage={first.usage}")
        for key, want in EXPECTED.items():
            ans = first.answers[key]
            print(f"  {key}: got {ans['choice']!r} (want {want!r})  probs={ans['probabilities']}")
        print(f"{'cold (first request)':<28} {first.latency_ms:.0f} ms")

        async def timed() -> float | str:
            try:
                return (await client.decide(STATE, QUESTIONS)).latency_ms
            except SystemOneError as exc:
                return str(exc)[:60]

        seq = []
        for _ in range(args.n):  # paced like gameplay: a few decisions per second
            seq.append(await timed())
            await asyncio.sleep(args.gap)
        report("warm, sequential", seq)

        burst: list[float | str] = []
        for _ in range(2):
            burst += await asyncio.gather(*(timed() for _ in range(args.burst)))
            await asyncio.sleep(1)
        report(f"warm, {args.burst} at once", burst)


if __name__ == "__main__":
    asyncio.run(main())
