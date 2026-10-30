"""A live, wall-clock game: a pilot's plans drive the cursor and keys as they arrive.

The hands only execute plans: they glide to the object the model named and press
at the moment the model said. LiveJudge runs judge.py's engine as the game goes, so
the live numbers match the final score recomputed from the recorded replay.
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from collections import Counter, deque
from dataclasses import dataclass, field

from . import judge, rules
from .beatmap import Beatmap, HitObject, Kind
from .hands import SPIN_RADIUS, SPIN_RPS, TAP_HOLD_MS, Action
from .pilot import Plan, build_request
from .replay import Replay
from .systemone import SystemOneError

LATE_TOLERANCE_MS = 150  # a plan whose moment passed longer ago than this is dropped


class LiveJudge:
    """judge.Engine fed as the game runs, plus what viewers draw (heads, final results, popups).

    Fed the same moments as a replay holds (on_press/on_release at key changes, update at every cursor
    frame, in time order), its numbers equal judge.score on that replay. update(t, cursor, held) closes
    the moment t; held=False releases the keys still down, a collection keeps only those keys.
    """

    def __init__(self, bm: Beatmap):
        self.objects = bm.objects
        self.popups: deque[tuple[float, HitObject, int]] = deque(maxlen=40)
        self.engine = judge.Engine(bm, on_object=self._on_object)
        self.head = self.engine.heads  # index -> (judgment, offset ms) of hit circles and slider heads
        self.results = self.engine.judgments  # index -> final 300/100/50/0
        self._next = 0  # first object not yet finalized (objects are in time order)
        self._presses = 0

    def on_press(self, t: float, cursor, key: int | None = None) -> None:
        """A key goes down; without a key, presses alternate between the two like a replay's."""
        key = self._presses % 2 if key is None else key
        self._presses += 1
        self.engine.press(t, cursor, key)

    def on_release(self, t: float, key: int, cursor=None) -> None:
        self.engine.release(t, self.engine.cursor if cursor is None else cursor, key)

    def update(self, t: float, cursor, held=None) -> None:
        self.engine.move(t, cursor)
        if held is not None:
            keep = (set(self.engine.keys) if held else set()) if isinstance(held, bool) else set(held)
            for key in sorted(self.engine.keys - keep):
                self.engine.release(t, cursor, key)
        self.engine.frame()

    def finish(self) -> judge.Score:
        self.engine.finish()
        return self.engine.result()

    def _on_object(self, t: float, o: HitObject, j: int) -> None:
        self.popups.append((t, o, j))
        while self._next < len(self.objects) and self.objects[self._next].index in self.results:
            self._next += 1

    def spins(self, o: HitObject) -> float:
        return self.engine.spinner_rotation(o.index) / 360

    @property
    def combo(self) -> int:
        return self.engine.scoring.combo

    @property
    def max_combo(self) -> int:
        return self.engine.scoring.max_combo

    @property
    def accuracy(self) -> float | None:
        return self.engine.scoring.accuracy if self.engine.scoring.judged else None

    @property
    def health(self) -> float:
        return self.engine.health.value


class Hands:
    """Executes plans: glide to the named object, press at the named time."""

    def __init__(self, bm: Beatmap, judge: LiveJudge, on_press=None):
        self.bm, self.judge, self.on_press = bm, judge, on_press
        self.by_index = {o.index: o for o in bm.objects}
        self.replay = Replay()
        self.cursor = rules.SPINNER_CENTER
        self.plans: dict[int, Plan] = {}
        self.pressed: set[int] = set()
        self.busy: tuple[str, HitObject, float, float] | None = None  # (kind, object, start, end)
        self.last_t: float | None = None

    def offer(self, plans: list[Plan]) -> None:
        for p in plans:
            old = self.plans.get(p.index)
            if p.index not in self.pressed and (old is None or p.asked_at >= old.asked_at):
                self.plans[p.index] = p

    def held(self, t: float) -> bool:
        return self.busy is not None and self.busy[2] <= t <= self.busy[3] or any(
            p.down <= t <= p.up for p in self.replay.presses[-2:])

    def next_plan(self, t: float) -> Plan | None:
        live = [p for i, p in self.plans.items()
                if i not in self.pressed and p.press_at is not None and p.press_at >= t - LATE_TOLERANCE_MS]
        return min(live, key=lambda p: p.press_at, default=None)

    def step(self, t: float) -> None:
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        if self.busy and t > self.busy[3]:
            self.busy = None
        if self.busy:
            kind, obj, start, _ = self.busy
            if kind == "hold":
                self.cursor = obj.ball_at(t)
            else:
                a = 2 * math.pi * SPIN_RPS * (t - start) / 1000
                cx, cy = rules.SPINNER_CENTER
                self.cursor = (cx + SPIN_RADIUS * math.cos(a), cy + SPIN_RADIUS * math.sin(a))
        else:
            plan = self.next_plan(t)
            if plan is not None:
                obj = self.by_index[plan.index]
                target = rules.SPINNER_CENTER if plan.action is Action.SPIN else obj.pos
                remaining = plan.press_at - t
                if remaining <= 0:
                    self.cursor = target
                    self._press(plan, obj, t)
                else:
                    f = min(1.0, dt / remaining)
                    self.cursor = (self.cursor[0] + (target[0] - self.cursor[0]) * f,
                                   self.cursor[1] + (target[1] - self.cursor[1]) * f)
        self.replay.move(t, self.cursor)

    def _press(self, plan: Plan, obj: HitObject, t: float) -> None:
        self.pressed.add(obj.index)
        if plan.action is Action.SPIN:
            end = max(obj.end_time, t + 250)
            self.busy = ("spin", obj, t, end)
        elif plan.action is Action.HOLD and obj.kind is Kind.SLIDER:
            end = max(obj.end_time, t + TAP_HOLD_MS)
            self.busy = ("hold", obj, t, end)
        else:
            end = t + TAP_HOLD_MS
        self.replay.press(t, end)
        self.judge.on_press(t, self.cursor)
        if self.on_press:
            self.on_press()


@dataclass
class ModelStats:
    sent: int = 0
    ok: int = 0
    in_flight: int = 0
    errors: Counter = field(default_factory=Counter)
    latencies: deque = field(default_factory=lambda: deque(maxlen=60))
    all_latencies: list = field(default_factory=list)
    reads: list[Plan] = field(default_factory=list)  # every plan, graded after the fact
    tokens: int = 0

    def latency(self) -> tuple[float, float] | None:
        lat = sorted(self.latencies)
        return (statistics.median(lat), lat[int(0.95 * (len(lat) - 1))]) if lat else None


class Clock:
    def __init__(self, origin_ms: float):
        self.origin_ms, self.start = origin_ms, time.perf_counter()

    def now(self) -> float:
        return self.origin_ms + (time.perf_counter() - self.start) * 1000


class Session:
    """One live run: the pilot thinks on a timer, the hands act every frame."""

    def __init__(self, bm: Beatmap, pilot, *, tick_ms: float = 150, max_in_flight: int = 4,
                 timeout_s: float = 2.0, on_press=None):
        self.bm, self.pilot = bm, pilot
        self.tick_ms, self.max_in_flight, self.timeout_s = tick_ms, max_in_flight, timeout_s
        self.judge = LiveJudge(bm)
        self.hands = Hands(bm, self.judge, on_press)
        self.stats = ModelStats()
        self.origin = bm.objects[0].time - bm.difficulty.preempt - 1500
        self.end = bm.objects[-1].end_time + 1000
        self.clock = Clock(self.origin)
        self.finished = False
        self._seed = 0

    def askable(self, t: float) -> list[HitObject]:
        preempt, meh = self.bm.difficulty.preempt, self.bm.difficulty.windows.meh
        out = []
        for o in self.bm.objects[self.judge._next:]:
            if o.time - preempt > t:
                break
            done = o.index in self.hands.pressed or o.index in self.judge.results
            deadline = o.end_time if o.kind is Kind.SPINNER else o.time + meh
            if not done and t <= deadline:
                out.append(o)
        return out

    async def _ask(self, t: float, pending: list[HitObject]) -> None:
        self._seed += 1
        req = build_request(self.bm, t, pending, self._seed)
        self.stats.sent += 1
        self.stats.in_flight += 1
        started = self.clock.now()
        try:
            plans, decision = await asyncio.wait_for(self.pilot.plan(req), self.timeout_s)
            latency = self.clock.now() - started
            self.stats.ok += 1
            self.stats.latencies.append(latency)
            self.stats.all_latencies.append(latency)
            if decision is not None:
                self.stats.tokens += decision.usage.get("input_tokens", 0)
            self.stats.reads.extend(plans)
            self.hands.offer(plans)
        except SystemOneError as exc:
            self.stats.errors[f"HTTP {exc.status}"] += 1
        except TimeoutError:
            self.stats.errors["timeout"] += 1
        except Exception as exc:  # keep playing; a failed request is a missed decision
            self.stats.errors[type(exc).__name__] += 1
        finally:
            self.stats.in_flight -= 1

    async def think(self) -> None:
        tasks = set()
        while not self.finished:
            t = self.clock.now()
            if self.stats.in_flight < self.max_in_flight:
                pending = self.askable(t)
                if pending:
                    task = asyncio.create_task(self._ask(t, pending))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
            await asyncio.sleep(self.tick_ms / 1000)

    def step(self) -> float:
        t = self.clock.now()
        self.hands.step(t)
        self.judge.update(t, self.hands.cursor, self.hands.held(t))
        if t > self.end:
            self.finished = True
        return t
