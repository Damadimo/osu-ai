"""The vision model plays the simulator: each 60 fps tick it sees the last 4 frames and acts.

The policy (policy.py) aims at the model's soonest detection and presses when that
detection's predicted time-to-hit runs out. Code renders frames, applies the policy's
events to a replay, and judges the result; it never reads the beatmap to decide.
"""

from __future__ import annotations

import time
from collections import deque

import mlx.core as mx
import numpy as np

from . import frames, judge
from .beatmap import Beatmap, Kind
from .framedata import FRAME_MS, HISTORY, SCALE, small_gray
from .live import LiveJudge
from .policy import Event, Policy
from .replay import Replay
from .vision import OsuNet, decode, decode_state

MIN_SCORE = 0.35


class VisionAgent:
    def __init__(self, weights: str, width: int = 48):
        self.model = OsuNet(width)
        self.model.load_weights(weights)
        self.model.eval()
        mx.eval(self.model.parameters())
        self.history: deque[np.ndarray] = deque(maxlen=HISTORY)

    def see(self, frame: np.ndarray):
        """Push a frame. Returns (detections, modes):
        detections [(x, y, ms until hit, score)] in osu!px, soonest first;
        modes (P(hold), P(spin), ball x, ball y)."""
        self.history.append(frame)
        while len(self.history) < HISTORY:
            self.history.appendleft(frame)
        x = np.stack(self.history, axis=-1)[None].astype(np.float32) / 255.0
        heat, off, dt, ball, modes = self.model(mx.array(x))
        dets = []
        for px, py, ms, score in decode(heat, off, dt, threshold=MIN_SCORE)[0]:
            ox, oy = frames.to_playfield(px / SCALE, py / SCALE)
            dets.append((ox, oy, ms, score))
        p_hold, p_spin, bx, by = decode_state(ball, modes)[0]
        return dets, (p_hold, p_spin, *frames.to_playfield(bx / SCALE, by / SCALE))


class VisionGame:
    """One simulated run: model + policy + live judge, stepped at 60 fps of game time."""

    def __init__(self, bm: Beatmap, agent: VisionAgent):
        self.bm, self.agent = bm, agent
        self.colors = frames.combo_colors(bm)
        self.judge = LiveJudge(bm)
        self.by_index = {o.index: o for o in bm.objects}
        self.policy = Policy()
        self.replay = Replay()
        self.held: dict[int, float] = {}  # key -> when it went down
        self.dets: list = []
        self.infer_ms: deque[float] = deque(maxlen=120)
        self.origin = bm.objects[0].time - bm.difficulty.preempt - 500
        self.end = bm.objects[-1].end_time + 500
        self.on_press = None
        self.modes = (0.0, 0.0, 256.0, 192.0)

    @property
    def cursor(self):
        return self.policy.cursor

    @property
    def plan(self):
        return self.policy.plan

    def hit_set(self) -> set[int]:
        """Circles that were hit vanish from the screen; a slider stays until its end."""
        return {i for i, (j, _) in self.judge.head.items() if j > 0 and self.by_index[i].kind is Kind.CIRCLE}

    def _record(self, t: float, pos) -> None:
        """Cursor frames must go forward in time; a click scheduled a few ms ahead may already be recorded."""
        if not self.replay.frames or t >= self.replay.frames[-1][0]:
            self.replay.move(t, pos)

    def apply(self, events: list[Event]) -> None:
        for e in events:
            if e.kind == "up":
                self.replay.press(self.held.pop(e.key), e.t)
                continue
            self._record(e.t, (e.x, e.y))
            if e.kind == "down":
                self.held[e.key] = e.t
                self.judge.on_press(e.t, (e.x, e.y))
                if self.on_press:
                    self.on_press()

    def step(self, t: float, dt_ms: float) -> None:
        frame = frames.render(self.bm, t, self.colors, done=self.hit_set())
        start = time.perf_counter()
        self.dets, self.modes = self.agent.see(small_gray(frame))
        self.infer_ms.append((time.perf_counter() - start) * 1000)
        self.apply(self.policy.step(t, dt_ms, self.dets, self.modes))
        self.judge.update(t, self.cursor, bool(self.held))

    def finish(self, t: float) -> None:
        if self.policy.key_down is not None:
            self.apply([self.policy.release(t)])

    def run_offline(self) -> judge.Score:
        """As fast as the hardware allows, in 60 fps game time."""
        t = self.origin
        while t <= self.end:
            self.step(t, FRAME_MS)
            t += FRAME_MS
        self.finish(t)
        return judge.score(self.bm, self.replay)
