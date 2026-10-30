"""The bot's hands: turn per-object decisions into cursor movement and key presses.

The hands are exact (they aim at the true position and press on the true hit
time) but they cannot act on an object before its decision has arrived.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from . import rules
from .beatmap import Beatmap, HitObject, Kind
from .replay import Replay

TAP_HOLD_MS = 30.0
FRAME_MS = 8.0  # cursor sampling while following sliders or spinning
SPIN_RPS = 6.0
SPIN_RADIUS = 60.0


class Action(str, Enum):
    TAP = "tap"  # click once
    HOLD = "hold"  # press and follow the slider ball to its end
    SPIN = "spin"  # hold and circle the playfield center
    SKIP = "skip"


CORRECT_ACTION = {Kind.CIRCLE: Action.TAP, Kind.SLIDER: Action.HOLD, Kind.SPINNER: Action.SPIN}


@dataclass(frozen=True)
class Intent:
    action: Action
    ready_at: float  # game ms when the decision reaches the hands


def _spin(replay: Replay, start: float, end: float) -> None:
    cx, cy = rules.SPINNER_CENTER
    t = start
    while t <= end:
        a = 2 * math.pi * SPIN_RPS * (t - start) / 1000
        replay.move(t, (cx + SPIN_RADIUS * math.cos(a), cy + SPIN_RADIUS * math.sin(a)))
        t += FRAME_MS
    replay.press(start, end)


def _follow(replay: Replay, obj: HitObject, start: float) -> float:
    """Hold from `start` and track the ball until the slider ends. Returns when the hand is free."""
    end = max(obj.end_time, start)  # a late press on a short slider may land after it ended
    t = start
    while t < end:
        replay.move(t, obj.ball_at(t))
        t += FRAME_MS
    replay.move(end, obj.ball_at(end))
    replay.press(start, max(end, start + TAP_HOLD_MS))
    return end


def perform(bm: Beatmap, intents: dict[int, Intent]) -> Replay:
    windows = bm.difficulty.windows
    replay = Replay()
    cursor, free_at = rules.SPINNER_CENTER, 0.0  # hand position, and when its last action ends
    replay.move(0.0, cursor)
    for obj in bm.objects:
        intent = intents.get(obj.index)
        if intent is None or intent.action is Action.SKIP:
            continue
        move_start = max(free_at, intent.ready_at)
        if intent.action is Action.SPIN:
            start = max(obj.time, move_start)
            end = max(obj.end_time, start + 250)
            replay.move(move_start, cursor)
            _spin(replay, start, end)
            free_at, cursor = end, replay.frames[-1][1:]
            continue
        press = max(obj.time, move_start)
        if press > obj.time + windows.meh:
            continue  # decision came too late to land a hit
        replay.move(move_start, cursor)
        replay.move(press, obj.pos)
        if intent.action is Action.HOLD and obj.kind is Kind.SLIDER:
            free_at = _follow(replay, obj, press)
        else:
            replay.press(press, press + TAP_HOLD_MS)
            free_at = press
        cursor = replay.frames[-1][1:]
    return replay
