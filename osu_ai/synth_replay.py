"""Human-like replays for recording the real game: osu!lazer plays them back while we capture.

Each object is aimed at with a little timing and aim noise, a few are missed, and the
cursor eases between objects the way a hand would. The point is varied, realistic
footage (hit and miss animations, judgements, a moving cursor) to label from the
beatmap. For local data collection only: never submit these anywhere.
"""

from __future__ import annotations

import math
import random

from . import rules
from .beatmap import Beatmap, Kind
from .replay import Replay

SAMPLE_MS = 8.0


def _ease(f: float) -> float:
    return f * f * (3 - 2 * f)  # smoothstep: slow start and stop, like a hand


def human_replay(bm: Beatmap, rng: random.Random, timing_sd: float = 12.0, aim_sd: float = 0.2,
                 miss_rate: float = 0.03) -> Replay:
    """timing_sd: ms of press-time noise; aim_sd: aim noise as a fraction of the circle radius."""
    d = bm.difficulty
    r, meh = d.radius, d.windows.meh
    replay = Replay()
    cursor = rules.SPINNER_CENTER
    free_at = bm.objects[0].time - d.preempt  # when the hand is done with the previous object
    replay.move(free_at, cursor)

    def glide(t0: float, t1: float, start, end) -> None:
        t = t0 + SAMPLE_MS
        while t < t1:
            f = _ease((t - t0) / (t1 - t0))
            replay.move(t, (start[0] + (end[0] - start[0]) * f, start[1] + (end[1] - start[1]) * f))
            t += SAMPLE_MS
        replay.move(max(t1, replay.frames[-1][0]), end)

    for obj in bm.objects:
        if obj.kind is Kind.SPINNER:
            start = max(obj.time + abs(rng.gauss(0, 20)), free_at + SAMPLE_MS)
            rps, radius = rng.uniform(4.0, 7.0), rng.uniform(40.0, 90.0)
            cx, cy = rules.SPINNER_CENTER
            glide(free_at, start, cursor, (cx + radius, cy))
            t = start
            while t < obj.end_time:
                a = 2 * math.pi * rps * (t - start) / 1000
                replay.move(t, (cx + radius * math.cos(a), cy + radius * math.sin(a)))
                t += SAMPLE_MS
            replay.press(start, max(t, start + 50))
            free_at, cursor = t, replay.frames[-1][1:]
            continue

        press = obj.time + max(-0.8 * meh, min(0.8 * meh, rng.gauss(0, timing_sd)))
        press = max(press, free_at + SAMPLE_MS)
        miss = rng.random() < miss_rate or press > obj.time + 0.8 * meh
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(1.4, 2.5) * r if miss else min(abs(rng.gauss(0, aim_sd * r)), 0.8 * r)
        target = (obj.x + dist * math.cos(angle), obj.y + dist * math.sin(angle))
        glide(free_at, press, cursor, target)
        cursor = target
        if miss and rng.random() < 0.5:
            free_at = press  # never pressed: the object times out
            continue
        if obj.kind is Kind.SLIDER and not miss:
            lag = rng.uniform(0, 0.3 * r)
            t = press
            while t < obj.end_time:
                bx, by = obj.ball_at(t)
                replay.move(t, (bx + lag * math.cos(angle), by + lag * math.sin(angle)))
                t += SAMPLE_MS
            end = max(t, press + 40) + rng.uniform(0, 30)
            replay.move(end, obj.ball_at(obj.end_time))
            replay.press(press, end)
            free_at, cursor = end, replay.frames[-1][1:]
        else:
            replay.press(press, press + rng.uniform(40, 90))
            free_at = press
    return replay
