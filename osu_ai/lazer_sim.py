"""The model plays our lazer-accurate renderer in closed loop, as a rehearsal for the real game.

Each 60 fps tick renders what lazer would show at that moment given how the agent has played
so far (its hits vanish, its misses fade, its early clicks kill approach rings), runs the
model and the policy on it, and judges the resulting presses with the lazer rules. Nothing
here reads the beatmap to decide; the beatmap only draws the frames and judges.
"""

from __future__ import annotations

import heapq
import itertools
import random
import time
from collections import deque

import numpy as np

from . import judge
from .beatmap import Beatmap, Kind
from .lazer_frames import Look, Outcome, View, render
from .live import LiveJudge
from .policy import Event, Policy
from .replay import Replay

FRAME_MS = 1000 / 60


class LazerSim:
    def __init__(self, bm: Beatmap, agent, look: Look = Look(), jitter_ms: float = 0.0, seed: int = 0,
                 record: bool = False, lead_ms: float = 0.0, background: np.ndarray | None = None):
        """lead_ms: decide every press this long before it happens, as the real game requires
        (perfectly calibrated latency: the press still lands when planned). background: what lazer
        shows behind the playfield at the agent's input size (lazer_frames.backdrop), or black."""
        self.bm, self.agent, self.look, self.background = bm, agent, look, background
        self.view = View(agent.input_size)
        self.judge = LiveJudge(bm)
        self.policy = Policy(lead_ms=lead_ms)
        self.replay = Replay()
        self.held: dict[int, float] = {}
        self.cursor = self.policy.cursor
        self.pending: list[tuple[float, int, Event]] = []  # input not yet due, by time
        self._seq = itertools.count()
        self.by_index = {o.index: o for o in bm.objects}
        self.outcome = Outcome(spin_rate={o.index: 1.6 for o in bm.objects if o.kind is Kind.SPINNER})
        self.rng = random.Random(seed)
        self.jitter_ms = jitter_ms
        self.origin = bm.objects[0].time - bm.difficulty.preempt - 500
        self.end = bm.objects[-1].end_time + 500
        self.infer_ms: deque[float] = deque(maxlen=600)
        self.record = record
        self.frames: list[np.ndarray] = []
        self.times: list[float] = []

    def _record_move(self, t: float, pos) -> None:
        if not self.replay.frames or t >= self.replay.frames[-1][0]:
            self.replay.move(t, pos)

    def apply(self, events: list[Event]) -> None:
        """Queue the policy's events; presses decided ahead (lead) take effect when due."""
        for e in events:
            heapq.heappush(self.pending, (e.t, next(self._seq), e))

    def _feed(self, t: float) -> None:
        """Hand the judge every input due by t, in time order, with its key."""
        while self.pending and self.pending[0][0] <= t:
            _, _, e = heapq.heappop(self.pending)
            if e.kind == "up":
                if e.key in self.held:
                    self.replay.press(self.held.pop(e.key), e.t)
                    self.judge.on_release(e.t, e.key, self.cursor)
                continue
            self.cursor = (e.x, e.y)
            self._record_move(e.t, self.cursor)
            if e.kind == "down":
                self.held[e.key] = e.t
                self.judge.on_press(e.t, self.cursor, e.key)

    def _sync_outcome(self, t: float) -> None:
        """Mirror the judge's verdicts into what the renderer shows."""
        for i, (j, offset) in self.judge.head.items():
            if i not in self.outcome.judged:
                o = self.by_index[i]
                self.outcome.judged[i] = (o.time + offset, True) if j > 0 else (t, False)

    def step(self, t: float, dt_ms: float) -> None:
        frame = np.clip(render(self.bm, t, self.outcome, self.look, self.view, self.background) + 0.5, 0, 255).astype(np.uint8)
        if self.record:
            self.frames.append(frame)
            self.times.append(t)
        start = time.perf_counter()
        dets, modes = self.agent.see(frame)
        self.infer_ms.append((time.perf_counter() - start) * 1000)
        self.apply(self.policy.step(t, dt_ms, dets, modes))
        self._feed(t)
        self.judge.update(t, self.cursor)
        self._sync_outcome(t)

    def run(self) -> judge.Score:
        self.agent.reset()
        t, prev = self.origin, self.origin - FRAME_MS
        while t <= self.end:
            self.step(t, t - prev)
            prev = t
            t += FRAME_MS + (self.rng.gauss(0, self.jitter_ms) if self.jitter_ms else 0.0)
        if self.policy.key_down is not None:
            self.apply([self.policy.release(t)])
        self._feed(float("inf"))
        return judge.score(self.bm, self.replay)
