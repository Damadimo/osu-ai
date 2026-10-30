"""Run a brain over a beatmap: ask at each object's appearance, then play and score.

A decision about an object is requested the moment it appears on screen
(hit time minus the approach time) and only reaches the hands after the
brain's latency. What the brain sees never depends on the hands, so the
replay can be built after all decisions are in without changing the result.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, replace

from . import judge
from .beatmap import Beatmap
from .brains import Answer, Brain
from .hands import CORRECT_ACTION, Intent, perform
from .replay import Replay
from .systemone import SystemOneError
from .view import snapshot

RETRY_MARGIN_S = 0.3  # only retry if a typical answer could still arrive before the hit


@dataclass
class Record:
    index: int
    kind: str
    asked_at: float  # game ms
    answer: Answer | None
    error: str | None = None

    @property
    def ready_at(self) -> float | None:
        return None if self.answer is None else self.asked_at + self.answer.latency_ms


@dataclass
class Run:
    beatmap: Beatmap
    brain: str
    records: list[Record]
    replay: Replay
    score: judge.Score

    def decision_summary(self) -> str:
        answered = [r for r in self.records if r.answer]
        lat = sorted(r.answer.latency_ms for r in answered)
        objs = {o.index: o for o in self.beatmap.objects}
        wrong = sum(r.answer.action is not CORRECT_ACTION[objs[r.index].kind] for r in answered)
        late = sum(r.ready_at > objs[r.index].time for r in answered)
        firsts = [r.answer.telemetry["first_correct"] for r in answered if "first_correct" in r.answer.telemetry]
        parts = [f"decisions {len(answered)}/{len(self.records)}", f"wrong action {wrong}",
                 f"arrived after hit time {late}"]
        if lat and lat[-1] > 0:
            parts.append(f"latency p50 {statistics.median(lat):.0f} ms, p95 {lat[int(0.95 * (len(lat) - 1))]:.0f} ms")
        if firsts:
            parts.append(f"read 'which is first' right {sum(firsts)}/{len(firsts)}")
        retried = sum(r.answer.telemetry.get("attempts", 1) > 1 for r in answered)
        if retried:
            parts.append(f"needed retries {retried}")
        errors = len(self.records) - len(answered)
        if errors:
            parts.append(f"errors {errors} (first: {next(r.error for r in self.records if r.error)})")
        return " | ".join(parts)


def segment(bm: Beatmap, start_s: float | None, seconds: float | None) -> Beatmap:
    if start_s is None and seconds is None:
        return bm
    lo = (start_s or 0) * 1000
    hi = lo + seconds * 1000 if seconds else float("inf")
    return replace(bm, objects=[o for o in bm.objects if lo <= o.time < hi])


async def run(bm: Beatmap, brain: Brain, *, realtime: bool, speed: float = 1.0,
              timeout_s: float = 3.0, progress: bool = False) -> Run:
    preempt = bm.difficulty.preempt
    loop = asyncio.get_running_loop()
    t0_game = bm.objects[0].time - preempt
    t0_wall = loop.time() + 0.5
    done = 0

    async def ask(obj) -> Record:
        nonlocal done
        asked_at = obj.time - preempt
        snap = snapshot(bm, obj, asked_at)
        if not realtime:
            answer = await brain.decide(snap, bm)
            done += 1
            return Record(obj.index, obj.kind.value, asked_at, answer)

        await asyncio.sleep(max(0.0, t0_wall + (asked_at - t0_game) / 1000 / speed - loop.time()))
        started = loop.time()
        hit_at = started + preempt / 1000  # wall clock when the object must be hit
        backoff, attempts = 0.1, 0
        while True:
            attempts += 1
            try:
                answer = await asyncio.wait_for(brain.decide(snap, bm), timeout_s)
                # Retries count against latency: the hands only get the answer now.
                answer.latency_ms = (loop.time() - started) * 1000
                answer.telemetry["attempts"] = attempts
                record = Record(obj.index, obj.kind.value, asked_at, answer)
                break
            except Exception as exc:  # a failed call is a missed decision, like a real stall
                retryable = isinstance(exc, (SystemOneError, TimeoutError)) and getattr(exc, "retryable", True)
                if retryable and loop.time() + backoff + RETRY_MARGIN_S < hit_at:
                    await asyncio.sleep(backoff)
                    backoff *= 1.5
                    continue
                error = f"{type(exc).__name__} after {attempts} attempts: {exc}"[:200]
                record = Record(obj.index, obj.kind.value, asked_at, None, error)
                break
        done += 1
        return record

    async def report() -> None:
        started = time.monotonic()
        while True:
            await asyncio.sleep(5)
            print(f"  … {time.monotonic() - started:4.0f}s  {done}/{len(bm.objects)} decisions", flush=True)

    reporter = asyncio.create_task(report()) if progress else None
    try:
        records = await asyncio.gather(*(ask(o) for o in bm.objects))
    finally:
        if reporter:
            reporter.cancel()

    intents = {r.index: Intent(r.answer.action, r.ready_at) for r in records if r.answer}
    replay = perform(bm, intents)
    return Run(bm, brain.name, list(records), replay, judge.score(bm, replay))
