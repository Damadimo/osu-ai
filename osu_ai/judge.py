"""Judge a replay as osu!lazer 2026.1005.0 does with default mods (no Classic).

One state machine, Engine, applies the rules in time order: which circle a press goes to and note
lock (StartTimeOrderedHitPolicy), slider tracking (SliderInputManager: the cursor must come within
1x the radius to start tracking and stay within 2.4x, the head-key restriction, ticks waiting for the
head, the 36 ms tail window), spinner spin history, and lazer's score and health processors for every
judgement. score() drives it from a replay; live.LiveJudge drives the same engine frame by frame, so
both agree on the same input.

lazer updates ~480 times a second, so between inputs the engine also runs the frames where something
can change: just after each hit window closes, at slider tick/repeat/tail times, at spinner starts and
ends, and every 2 ms while a slider or spinner is in play, with the cursor moving linearly between inputs
(as lazer interpolates replay frames). Within one frame lazer updates the latest-starting object first.
With step_ms=None it judges only at the inputs it is given: replay playback judges every replay frame
plus wherever lazer's own clock lands in between, so the oracle tests feed those frames in too.
Keys are 0 and 1 (lazer's left and right buttons); a replay's presses alternate between them.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import lazer_rules as lz
from . import rules
from .beatmap import Beatmap, HitObject, Kind
from .replay import Replay
from .rules import f32

STEP_MS = 2.0  # frame spacing while a slider or spinner is in play (lazer updates at ~480 Hz)


@dataclass
class Judgement:
    t: float  # when it was judged, ms
    index: int  # beatmap object index
    kind: str  # what was judged, as a lazer class name ("HitCircle", "SliderTick", ...)
    result: str  # lazer HitResult name ("Great", "LargeTickMiss", ...)
    offset: float | None = None  # judgement time minus hit time, for hit circles and slider heads
    combo: int = 0  # combo after this judgement
    health: float = 1.0  # health after this judgement


@dataclass
class Score:
    judgments: list[int]  # per object: 300/100/50/0 (a slider's is its head's, a spinner's its own)
    max_combo: int
    head_offsets: list[float] = field(default_factory=list)  # ms, press minus hit time of hit circles/heads
    accuracy: float = 1.0  # lazer's accuracy: heads/circles/spinners 300, ticks/repeats 30, tails 150
    total_score: int = 0  # lazer's standardised score
    rank: str = "X"
    failed_at: float | None = None  # when health first ran out (lazer fails there; we play on, as NoFail)
    min_health: float = 1.0
    statistics: Counter = field(default_factory=Counter)  # lazer HitResult name -> count
    events: list[Judgement] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.failed_at is None

    @property
    def counts(self) -> Counter:
        return Counter(self.judgments)

    def summary(self) -> str:
        c, s = self.counts, self.statistics
        ticks = f"{s[lz.TICK_HIT]}/{s[lz.TICK_HIT] + s[lz.TICK_MISS]}"
        ends = f"{s[lz.TAIL_HIT]}/{sum(e.kind == lz.TAIL for e in self.events)}"
        outcome = "pass" if self.passed else f"FAIL at {self.failed_at / 1000:.1f}s"
        return (f"accuracy {self.accuracy:6.2%}  300:{c[300]} 100:{c[100]} 50:{c[50]} miss:{c[0]}  "
                f"max combo {self.max_combo}  ticks {ticks} ends {ends}  rank {self.rank}  "
                f"{outcome}, min HP {self.min_health:.0%}")


class _Circle:
    """A hit circle or slider head: what a press can hit, and what note lock orders."""

    __slots__ = ("obj", "i", "time", "x", "y", "result", "offset", "hit_key", "slider", "blocker")

    def __init__(self, obj: HitObject, i: int, slider: _Slider | None = None):
        self.obj, self.i, self.slider = obj, i, slider
        self.time, self.x, self.y = obj.time, obj.x, obj.y
        self.result: str | None = None
        self.offset: float | None = None
        self.hit_key: int | None = None  # lazer's HitAction: the first key that pressed it while it waited
        self.blocker = -1  # the latest circle-like strictly before this one


class _Nested:
    __slots__ = ("time", "kind", "result")

    def __init__(self, time: float, kind: str):
        self.time, self.kind = time, kind
        self.result: str | None = None


class _Slider:
    __slots__ = ("obj", "head", "ticks", "next", "tail", "tracking", "any_key_after", "last_keys", "result")

    def __init__(self, obj: HitObject):
        self.obj = obj
        nested = [_Nested(n.time, lz.NESTED_KIND[n.kind]) for n in obj.nested]
        self.ticks, self.tail = nested[:-1], nested[-1]  # ticks and repeats, in time order
        self.next = 0  # ticks[:next] are judged
        self.tracking = False
        self.any_key_after: float | None = None  # from then on either key tracks, not just the head's
        self.last_keys: frozenset = frozenset()
        self.result: str | None = None
        self.head: _Circle


class _Spinner:
    __slots__ = ("obj", "required", "ticks", "awarded", "turned", "turned_at_spin", "spin_max", "spins",
                 "last_angle", "result")

    def __init__(self, obj: HitObject, od: float):
        duration = obj.end_time - obj.time
        self.obj = obj
        self.required = rules.spins_required(od, duration)
        self.ticks = [lz.SPIN_TICK] * (self.required + 2) + [lz.SPIN_BONUS] * rules.bonus_spins(od, duration)
        self.awarded = 0
        self.turned = self.turned_at_spin = self.spin_max = 0.0  # SpinnerSpinHistory, in degrees
        self.spins = 0
        self.last_angle: float | None = None
        self.result: str | None = None

    def turn(self, delta: float) -> None:
        """A spin counts once it reaches 360 degrees one way; turning back never subtracts.
        In float32 like lazer's SpinnerSpinHistory, which matters when a spin ends right on 360."""
        if delta == 0:
            return
        self.turned = f32(self.turned + delta)
        current = f32(self.turned - self.turned_at_spin)
        self.spin_max = max(self.spin_max, abs(current))
        while self.spin_max >= 360:
            self.spins += 1
            self.turned_at_spin = f32(self.turned_at_spin + ((current > 0) - (current < 0)) * 360)
            current = f32(self.turned - self.turned_at_spin)
            self.spin_max = abs(current)

    @property
    def rotation(self) -> float:
        return 360 * self.spins + self.spin_max

    def progress(self) -> float:
        return 1.0 if self.required == 0 else min(max(f32(self.rotation) / 360 / self.required, 0.0), 1.0)


class Engine:
    """lazer's gameplay rules for one beatmap. Feed inputs in time order: move(), press(), release(),
    then frame() to close that moment; finish() runs out the map. on_object(t, obj, judgment) is called
    as each object gets its final 300/100/50/0."""

    def __init__(self, bm: Beatmap, on_object=None, step_ms: float | None = STEP_MS):
        d = bm.difficulty
        self.step_ms = step_ms
        self.bm, self.objects = bm, bm.objects
        self.on_object = on_object
        self.windows, self.preempt = d.windows, d.preempt
        radius = f32(d.radius)
        self.r2, self.follow2 = radius * radius, f32(radius * f32(rules.FOLLOW_RADIUS_SCALE)) ** 2
        self.health = lz.Health(bm.objects, bm.breaks, d.hp)
        self.scoring = lz.Scoring(bm.objects)
        self.events: list[Judgement] = []
        self.heads: dict[int, tuple[int, float]] = {}  # object index -> (300/100/50/0, offset) of circles/heads
        self.judgments: dict[int, int] = {}  # object index -> final 300/100/50/0

        self.circles: list[_Circle] = []
        self.sliders: list[_Slider] = []
        self.spinners: list[_Spinner] = []
        for o in bm.objects:
            if o.kind is Kind.SPINNER:
                self.spinners.append(_Spinner(o, d.od))
                continue
            slider = _Slider(o) if o.kind is Kind.SLIDER else None
            circle = _Circle(o, len(self.circles), slider)
            self.circles.append(circle)
            if slider is not None:
                slider.head = circle
                self.sliders.append(slider)
        times = [c.time for c in self.circles]
        for c in self.circles:
            c.blocker = bisect.bisect_left(times, c.time) - 1
        self._alive_until = [t - self.preempt for t in times]  # a circle takes presses from time - preempt
        self._first_open = 0  # circles[:_first_open] are all judged
        self._miss_next = 0  # next circle whose window may close
        self._active_sliders: list[_Slider] = []
        self._active_spinners: list[_Spinner] = []
        self._next_slider = self._next_spinner = 0
        self._schedule = self._frame_times() if step_ms is not None else []
        self._sched_i = 0

        self.t = -math.inf
        self.cursor = rules.SPINNER_CENTER
        self.keys: set[int] = set()
        self._framed = True

    def _frame_times(self) -> list[float]:
        meh = self.windows.meh
        points = [math.nextafter(c.time + meh, math.inf) for c in self.circles]
        grids = []
        for s in self.sliders:
            o = s.obj
            points += [n.time for n in s.ticks]
            points += [s.tail.time - rules.SLIDER_TAIL_LENIENCY_MS, s.tail.time, o.end_time]
        for sp in self.spinners:
            o = sp.obj
            points += [o.time - self.step_ms, o.time, math.nextafter(o.end_time, -math.inf), o.end_time]
        step = self.step_ms
        for o in self.objects:
            if o.kind is not Kind.CIRCLE:
                grids.append(np.arange(math.ceil(o.time / step) * step, o.end_time, step))
        return np.unique(np.concatenate([np.array(points, dtype=float), *grids])).tolist()

    # ---- inputs ----

    def move(self, t: float, cursor) -> None:
        """Advance to time t (running the frames in between) with the cursor at `cursor` by then."""
        if t > self.t:
            if not self._framed:
                self._frame(self.t)
            t0, (x0, y0), (x1, y1) = self.t, self.cursor, cursor
            sched, i = self._schedule, self._sched_i
            while i < len(sched) and sched[i] < t:
                s = sched[i]
                i += 1
                if s <= t0:
                    continue
                f = (s - t0) / (t - t0) if t0 > -math.inf else 1.0
                self.cursor = (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)
                self._frame(s)
            self._sched_i = i
            self.t, self._framed = t, False
        self.cursor = (cursor[0], cursor[1])

    def press(self, t: float, cursor, key: int) -> None:
        """A key goes down: the earliest hovered circle still waiting takes the press."""
        self.move(t, cursor)
        if key in self.keys:
            return
        self.keys.add(key)
        t, (cx, cy) = self.t, self.cursor
        last = bisect.bisect_right(self._alive_until, t)
        for c in self.circles[self._first_open:last]:
            if c.result is not None or (cx - c.x) ** 2 + (cy - c.y) ** 2 > self.r2:
                continue
            judgment = self.windows.result(t - c.time)
            blocker = self.circles[c.blocker] if c.blocker >= 0 else None
            locked = blocker is not None and blocker.result is None and t < blocker.time
            if judgment is not None and not locked:
                self._judge_circle(c, t, lz.FROM_JUDGMENT[judgment])
            if c.hit_key is None:
                c.hit_key = key
            return  # a hovered waiting circle takes the press even when it only shakes

    def release(self, t: float, cursor, key: int) -> None:
        self.move(t, cursor)
        self.keys.discard(key)

    def frame(self) -> None:
        """Close the current moment: run its frame after all of its inputs."""
        if not self._framed:
            self._frame(self.t)
            self._framed = True

    def finish(self) -> None:
        """Run out the map. Two frames: a slider head that times out on the first lets its ticks and tail,
        which wait for it, be judged on the second."""
        if self.objects:
            end = max(max(o.end_time for o in self.objects), self.objects[-1].time + self.windows.meh) + STEP_MS
            for t in (end, end + STEP_MS):
                self.move(max(t, self.t), self.cursor)
                self.frame()
        self.frame()

    # ---- frames ----

    def _frame(self, t: float) -> None:
        horizon = t + self.preempt
        while self._next_slider < len(self.sliders) and self.sliders[self._next_slider].obj.time <= horizon:
            self._active_sliders.append(self.sliders[self._next_slider])
            self._next_slider += 1
        while self._next_spinner < len(self.spinners) and self.spinners[self._next_spinner].obj.time <= horizon:
            self._active_spinners.append(self.spinners[self._next_spinner])
            self._next_spinner += 1
        closing = []  # hit circles whose window closed unhit (slider heads are their slider's business)
        while self._miss_next < len(self.circles) and t - self.circles[self._miss_next].time > self.windows.meh:
            c = self.circles[self._miss_next]
            if c.result is None and c.slider is None:
                closing.append(c)
            self._miss_next += 1
        # lazer updates the latest-starting object first, each running all of its own checks in turn,
        # which orders same-frame judgements of overlapping objects (and so their combo and health)
        due = [*self._active_sliders, *self._active_spinners, *closing]
        if len(due) > 1:
            due.sort(key=lambda d: (d.obj.time, d.obj.index), reverse=True)
        for d in due:
            if isinstance(d, _Slider):
                self._slider_frame(d, t)
            elif isinstance(d, _Spinner):
                self._spinner_frame(d, t)
            elif d.result is None:
                self._judge_circle(d, t, lz.MISS)
        self._active_sliders = [s for s in self._active_sliders if s.result is None]
        self._active_spinners = [sp for sp in self._active_spinners if sp.result is None]

    def _slider_frame(self, s: _Slider, t: float) -> None:
        """One DrawableSlider update: tracking, then ticks, repeats and the tail (which wait for the head),
        then the head's own miss check, then the slider's result."""
        bx, by = s.obj.ball_at(t)
        d2 = (self.cursor[0] - bx) ** 2 + (self.cursor[1] - by) ** 2
        self._track(s, t, d2 <= (self.follow2 if s.tracking else self.r2))
        if s.head.result is None:
            if t - s.head.time > self.windows.meh:
                self._judge_circle(s.head, t, lz.MISS)
            return
        k = s.next
        while k < len(s.ticks) and s.ticks[k].time <= t:
            k += 1
        due = s.ticks[s.next:k]
        for n in [n for n in due if n.kind == lz.TICK] + [n for n in due if n.kind == lz.REPEAT]:
            self._judge_nested(s, n, t, s.tracking)
        s.next = k
        tail = s.tail
        if tail.result is None and t - tail.time >= -rules.SLIDER_TAIL_LENIENCY_MS and s.next == len(s.ticks):
            if s.tracking:
                self._judge_nested(s, tail, t, True)
            elif t >= tail.time:
                self._judge_nested(s, tail, t, False)
        if s.result is None and tail.result is not None and t >= s.obj.end_time:
            hit = any(lz.is_hit(n.result) for n in (s.head, *s.ticks, tail))
            s.result = lz.IGNORE_HIT if hit else lz.IGNORE_MISS
            self._emit(t, s.obj, lz.SLIDER, s.result)
            self._finalize(t, s.obj, lz.JUDGMENT[s.head.result])

    def _track(self, s: _Slider, t: float, in_range: bool) -> None:
        """SliderInputManager.updateTracking. A head hit with one key tracks only with that key until the
        other key, if it was down, has been let go."""
        key = s.head.hit_key
        if key is None:
            s.any_key_after = None
        elif s.any_key_after is None and 1 - key not in s.last_keys:
            s.any_key_after = t
        if key is not None and (s.any_key_after is None or t <= s.any_key_after):
            valid_key = key in self.keys
        else:
            valid_key = bool(self.keys)
        s.last_keys = frozenset(self.keys)
        s.tracking = (s.result is None or t <= s.obj.end_time) and in_range and valid_key

    def _post_head(self, s: _Slider, t: float) -> None:
        """PostProcessHeadJudgement: a late head hit settles the ticks it skipped over, all hit if the cursor
        was near the ball at each of them (else all missed), and starts tracking right away."""
        cx, cy = self.cursor
        bx, by = s.obj.ball_at(t)
        if (cx - bx) ** 2 + (cy - by) ** 2 > self.follow2:
            return
        passed = [n for n in (*s.ticks, s.tail) if n.result is None and n.time <= t]
        all_in = True
        for n in passed:
            nx, ny = s.obj.ball_at(n.time)
            if (cx - nx) ** 2 + (cy - ny) ** 2 > self.follow2:
                all_in = False
                break
        for n in passed:
            self._judge_nested(s, n, t, all_in)
        while s.next < len(s.ticks) and s.ticks[s.next].result is not None:
            s.next += 1
        self._track(s, t, all_in or (cx - bx) ** 2 + (cy - by) ** 2 <= self.r2)

    def _spinner_frame(self, sp: _Spinner, t: float) -> None:
        o = sp.obj
        cx, cy = rules.SPINNER_CENTER
        angle = f32(-math.degrees(math.atan2(f32(self.cursor[0] - cx), f32(self.cursor[1] - cy))))
        if sp.last_angle is not None and o.time <= t < o.end_time and self.keys:
            delta = f32(angle - sp.last_angle)
            sp.turn(f32(delta - 360) if delta > 180 else f32(delta + 360) if delta < -180 else delta)
        sp.last_angle = angle
        if t >= o.end_time:
            for kind in sp.ticks[sp.awarded:]:
                self._emit(t, o, kind, lz.IGNORE_MISS)
            sp.awarded = len(sp.ticks)
            p = sp.progress()
            sp.result = lz.GREAT if p >= 1 else lz.OK if p > 0.9 else lz.MEH if p > 0.75 else lz.MISS
            self._emit(t, o, lz.SPINNER, sp.result)
            self._finalize(t, o, lz.JUDGMENT[sp.result])
            return
        while sp.awarded < min(int(sp.rotation / 360), len(sp.ticks)):  # each full spin scores the next tick
            self._emit(t, o, sp.ticks[sp.awarded], lz.MAX_RESULT[sp.ticks[sp.awarded]])
            sp.awarded += 1

    # ---- judgements ----

    def _judge_circle(self, c: _Circle, t: float, result: str) -> None:
        """Judging a circle-like first misses every earlier one still waiting (StartTimeOrderedHitPolicy.HandleHit)."""
        for p in self.circles[self._first_open:c.i]:
            if p.result is None and p.time < c.time:
                self._set_circle(p, t, lz.MISS)
        self._set_circle(c, t, result)
        while self._first_open < len(self.circles) and self.circles[self._first_open].result is not None:
            self._first_open += 1
        if c.slider is not None and result != lz.MISS:
            self._post_head(c.slider, t)

    def _set_circle(self, c: _Circle, t: float, result: str) -> None:
        c.result, c.offset = result, t - c.time
        self.heads[c.obj.index] = (lz.JUDGMENT[result], c.offset)
        if c.slider is None:
            self._emit(t, c.obj, lz.CIRCLE, result, c.offset)
            self._finalize(t, c.obj, lz.JUDGMENT[result])
        else:
            self._emit(t, c.obj, lz.HEAD, result, c.offset)

    def _judge_nested(self, s: _Slider, n: _Nested, t: float, hit: bool) -> None:
        n.result = lz.MAX_RESULT[n.kind] if hit else lz.MIN_RESULT[n.kind]
        self._emit(t, s.obj, n.kind, n.result)

    def _emit(self, t: float, obj: HitObject, kind: str, result: str, offset: float | None = None) -> None:
        top = kind in (lz.CIRCLE, lz.SLIDER, lz.SPINNER)  # nested objects have no combo flags of their own
        self.health.apply(t, kind, result, top and obj.new_combo, top and obj.last_in_combo)
        self.scoring.apply(kind, result)
        self.events.append(Judgement(t, obj.index, kind, result, offset, self.scoring.combo, self.health.value))

    def _finalize(self, t: float, obj: HitObject, judgment: int) -> None:
        self.judgments[obj.index] = judgment
        if self.on_object is not None:
            self.on_object(t, obj, judgment)

    # ---- results ----

    def spinner_rotation(self, index: int) -> float:
        """Degrees counted so far on a spinner."""
        return next((sp.rotation for sp in self.spinners if sp.obj.index == index), 0.0)

    def result(self) -> Score:
        sc, hp = self.scoring, self.health
        failed = hp.failed_at is not None
        offsets = [c.offset for c in self.circles if c.result in (lz.GREAT, lz.OK, lz.MEH)]
        return Score(
            judgments=[self.judgments.get(o.index, 0) for o in self.objects],
            max_combo=sc.max_combo,
            head_offsets=offsets,
            accuracy=sc.accuracy,
            total_score=sc.total_score,
            rank=lz.rank(sc.accuracy, sc.statistics[lz.MISS], failed),
            failed_at=hp.failed_at,
            min_health=hp.lowest,
            statistics=Counter(sc.statistics),
            events=self.events,
        )


def replay_inputs(replay: Replay):
    """(time, cursor, keys released, keys pressed, keys of zero-length presses released) in time order.
    Press i uses key i % 2 (as osr.export writes them); pressing a key that is still down is no new press."""
    replay.finalize()
    held: dict[int, list[list[float]]] = {0: [], 1: []}
    for i, p in enumerate(replay.presses):
        spans = held[i % 2]
        if spans and p.down < spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], p.up)
        else:
            spans.append([p.down, max(p.up, p.down)])
    events: dict[float, list[list[int]]] = {t: [[], [], []] for t, _, _ in replay.frames}
    for key, spans in held.items():
        for down, up in spans:
            events.setdefault(down, [[], [], []])[1].append(key)
            events.setdefault(up, [[], [], []])[2 if up == down else 0].append(key)
    for t in sorted(events):
        early, downs, late = events[t]
        yield t, replay.cursor_at(t), early, downs, late


def score(bm: Beatmap, replay: Replay) -> Score:
    engine = Engine(bm)
    for t, cursor, early, downs, late in replay_inputs(replay):
        engine.move(t, cursor)
        for key in early:
            engine.release(t, cursor, key)
        for key in downs:
            engine.press(t, cursor, key)
        for key in late:
            engine.release(t, cursor, key)
        engine.frame()
    engine.finish()
    return engine.result()
