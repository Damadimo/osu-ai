"""osu!standard difficulty formulas as osu!lazer 2026.1005.0 computes them. Units: osu!pixels (512x384) and ms.

lazer keeps CS/AR/OD/HP as float32 and rounds through float32 in a few places; doing the same keeps
window, radius and preempt boundaries exact (OD 8.3 gives a 115.5 ms 50 window, AR 9.3 a 554 ms preempt).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

PLAYFIELD_W, PLAYFIELD_H = 512.0, 384.0
SPINNER_CENTER = (256.0, 192.0)
MISS_WINDOW_MS = 400.0  # a press on a hovered circle up to this early is a miss; earlier presses only shake it
FOLLOW_RADIUS_SCALE = 2.4  # how far from the slider ball tracking survives, relative to the circle radius
SLIDER_TAIL_LENIENCY_MS = 36.0  # the tail counts if the slider is tracked at any time in its last 36 ms
STACK_OFFSET = -6.4  # osu!px per stack level (times the circle scale), up and to the left

_F32 = struct.Struct("f")


def f32(x: float) -> float:
    """Round to float32, like a C# (float) cast."""
    return _F32.unpack(_F32.pack(x))[0]


def difficulty_range(value: float, low: float, mid: float, high: float) -> float:
    """Map a 0-10 difficulty setting through (0 -> low, 5 -> mid, 10 -> high), in lazer's operation order."""
    if value > 5:
        return mid + (high - mid) * ((value - 5) / 5)
    if value < 5:
        return mid + (mid - low) * ((value - 5) / 5)
    return mid


def circle_scale(cs: float) -> float:
    """Object scale: 1 at CS0-ish, times stable's 1.00041 gamefield rounding allowance."""
    inner = f32(1.0 - f32(0.7) * ((f32(cs) - 5) / 5))
    return f32(f32(inner / 2) * f32(1.00041))


def circle_radius(cs: float) -> float:
    return 64 * circle_scale(cs)


def preempt_ms(ar: float) -> float:
    """How long before its hit time an object appears (the approach ring's lifetime), truncated to whole ms."""
    return float(int(difficulty_range(f32(ar), 1800, 1200, 450)))


def fade_in_ms(ar: float) -> float:
    return 400 * min(1.0, preempt_ms(ar) / 450)


@dataclass(frozen=True)
class HitWindows:
    great: float  # +-ms for a 300
    ok: float  # +-ms for a 100
    meh: float  # +-ms for a 50
    miss: float = MISS_WINDOW_MS

    def result(self, offset_ms: float) -> int | None:
        """300/100/50, 0 for a miss (within 400 ms), None if a press this far off does nothing."""
        off = abs(offset_ms)
        if off <= self.great:
            return 300
        if off <= self.ok:
            return 100
        if off <= self.meh:
            return 50
        return 0 if off <= self.miss else None

    def judge(self, offset_ms: float) -> int:
        return self.result(offset_ms) or 0


def hit_windows(od: float) -> HitWindows:
    od = f32(od)
    return HitWindows(
        math.floor(difficulty_range(od, 80, 50, 20)) - 0.5,
        math.floor(difficulty_range(od, 140, 100, 60)) - 0.5,
        math.floor(difficulty_range(od, 200, 150, 100)) - 0.5,
    )


def spins_per_second_required(od: float) -> float:
    return difficulty_range(f32(od), 90, 150, 225) / 60


def spins_required(od: float, duration_ms: float) -> int:
    """Whole spins needed for a 300 on a spinner (0 means it is cleared without spinning)."""
    return int(spins_per_second_required(od) * (duration_ms / 1000) + 0.0001)


def bonus_spins(od: float, duration_ms: float) -> int:
    """Spins past spins_required + 2 that still award a bonus tick."""
    full = int(difficulty_range(f32(od), 250, 380, 430) / 60 * (duration_ms / 1000) + 0.0001)
    return max(0, full - spins_required(od, duration_ms) - 2)
