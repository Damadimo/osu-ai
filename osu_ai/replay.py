"""Replays: cursor keyframes plus key presses, the same information an .osr file holds."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from .curves import Point


@dataclass(frozen=True)
class Press:
    down: float
    up: float


@dataclass
class Replay:
    frames: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y), t ascending
    presses: list[Press] = field(default_factory=list)

    def move(self, t: float, pos: Point) -> None:
        if self.frames and t < self.frames[-1][0]:
            raise ValueError(f"frame at {t} is before the last frame at {self.frames[-1][0]}")
        self.frames.append((t, pos[0], pos[1]))

    def press(self, down: float, up: float) -> None:
        self.presses.append(Press(down, up))

    def finalize(self) -> None:
        self.presses.sort(key=lambda p: p.down)
        self._times = [f[0] for f in self.frames]
        held: list[list[float]] = []  # union of press intervals, for held_at()
        for p in self.presses:
            if held and p.down <= held[-1][1]:
                held[-1][1] = max(held[-1][1], p.up)
            else:
                held.append([p.down, p.up])
        self._held_starts = [h[0] for h in held]
        self._held = held

    def cursor_at(self, t: float) -> Point:
        i = bisect.bisect_right(self._times, t)
        if i == 0:
            return (256.0, 192.0) if not self.frames else self.frames[0][1:]
        if i == len(self.frames):
            return self.frames[-1][1:]
        (t0, x0, y0), (t1, x1, y1) = self.frames[i - 1], self.frames[i]
        f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)

    def held_at(self, t: float) -> bool:
        i = bisect.bisect_right(self._held_starts, t)
        return i > 0 and self._held[i - 1][1] >= t
