"""How the vision agent acts on what the model sees, in the simulator or the real game.

Each frame the model reports the upcoming circles (position, ms until hit) and whether a
slider or spinner is active. The policy turns that into hand events: cursor moves and key
presses at exact times on the frame clock. It never reads the beatmap; the environment
decides how events become a replay (simulator) or OS input (real game).

In the real game every action must leave `lead_ms` early to cover capture and input
latency, so a press is decided that far ahead of its moment. Until it fires, the cursor
stays on its target, and the circle it is aimed at is ignored as a new target.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

TAP_MS = 30
REFRACTORY_MS = 45  # no second click right after one (one press per circle)
COMMIT_MS = 150  # schedule the click once the soonest circle is this close
SAME_TARGET_PX = 32.0  # a detection this close to the planned click is the same circle
SAME_CIRCLE_MS = 70  # ...if it is also due within this long after a press there
SETTLE_MS = 10  # keep the cursor on a pressed circle this long after the press
MAX_DRIFT_MS = 35  # a planned click may slip this much past its first plan, then it fires anyway
FUSE_FRAMES = 6  # the click time is the mean of this many recent frames' estimates for the circle
TRACK_JUMP_MS = 60  # an estimate this far from that mean means the model changed its mind: start over
HOVER_PX = 40.0  # circles this close to a press are under the cursor when it lands
EARLY_GUARD_MS = 90  # never press where every circle under the cursor is due this much later (lazer: Miss)
SPIN_RPS, SPIN_RADIUS = 6.0, 60.0
CENTER = (256.0, 192.0)


@dataclass(frozen=True)
class Event:
    kind: str  # "move", "down" or "up"
    t: float  # ms on the frame clock
    x: float = 0.0  # osu!px, for "move" and "down"
    y: float = 0.0
    key: int = 0  # which of the two keys, for "down" and "up"


class Policy:
    """Aims at the soonest detection and presses when its predicted time-to-hit runs out."""

    def __init__(self, fuse_frames: int = FUSE_FRAMES, lead_ms: float = 0.0):
        self.lead_ms = lead_ms
        self.cursor = CENTER
        self.plan: tuple[float, float, float] | None = None  # (press time, x, y)
        self.plan_first = 0.0  # the press time first planned for the current target
        self.retarget = False  # the estimate jumped: re-anchor plan_first at the next commit
        self.track: deque[float] = deque(maxlen=fuse_frames)  # recent hit-time estimates of the soonest circle
        self.track_pos: tuple[float, float] | None = None
        self.pressed: deque[tuple[float, float, float]] = deque(maxlen=4)  # (time, x, y) of recent presses
        self.key_down: float | None = None  # when the held key went down
        self.key = 1  # the key pressed last; presses alternate so back-to-back taps stay distinct
        self.last_press = -1e9
        self.spin_start = 0.0
        self.holding = False

    def _handled(self, x: float, y: float, hit_at: float) -> bool:
        """A detection that is the circle we already pressed (or will press any moment)."""
        return any(math.dist((x, y), (px, py)) < SAME_TARGET_PX and hit_at < when + SAME_CIRCLE_MS
                   for when, px, py in self.pressed)

    def step(self, t: float, dt_ms: float, dets, modes) -> list[Event]:
        """One frame at time t. dets: [(x, y, ms until hit, score)] soonest first, in osu!px;
        modes: (P(hold), P(spin), ball x, ball y)."""
        p_hold, p_spin, bx, by = modes
        # Once following a slider, one unsure frame must not let go of it (that breaks the slider).
        holding = self.key_down is not None and (p_hold > 0.5 or (self.holding and p_hold > 0.25))
        self.holding = holding
        spinning = p_spin > 0.5
        settling = self.pressed and t < self.pressed[-1][0] + SETTLE_MS
        events = [Event("move", t, *self.cursor)]

        target = next((d for d in dets if not self._handled(d[0], d[1], t + d[2])), None)
        if target is not None and not spinning:
            tx, ty, ms, _ = target
            est = t + max(ms, 0.0)
            # A new circle, or (in a stack) the model switching which circle it times: old guesses go.
            if (self.track_pos is None or math.dist((tx, ty), self.track_pos) > SAME_TARGET_PX
                    or (self.track and abs(est - sum(self.track) / len(self.track)) > TRACK_JUMP_MS)):
                self.track.clear()
                self.retarget = True
                if self.plan and self.plan[0] - t >= dt_ms + self.lead_ms:
                    self.plan = None  # it was built on the guesses just dropped (unless it fires now)
            self.track.append(est)
            self.track_pos = (tx, ty)
            hit_at = sum(self.track) / len(self.track)  # each frame's guess is noisy; their mean much less so
            moving_on = self.plan and math.dist((tx, ty), self.plan[1:]) > SAME_TARGET_PX
            # Commit to a click time once the soonest circle is close; refresh it with every new frame,
            # unless a different circle shows up just before the committed click is due.
            if hit_at - t <= COMMIT_MS + self.lead_ms and not (moving_on and self.plan[0] - t < dt_ms + self.lead_ms):
                if self.plan is None or moving_on or self.retarget:
                    self.plan_first = max(hit_at, t)
                    self.retarget = False
                self.plan = (max(hit_at, t), tx, ty)
            if not holding and not settling:
                aim = self.plan[1:] if self.plan else (tx, ty)
                remaining = max((self.plan[0] if self.plan else t + ms) - t, dt_ms)
                f = min(1.0, dt_ms / remaining)
                self.cursor = (self.cursor[0] + (aim[0] - self.cursor[0]) * f,
                               self.cursor[1] + (aim[1] - self.cursor[1]) * f)
        # A stack can make the estimate recede forever (the next circle's ring takes over), so a
        # click never slips more than MAX_DRIFT_MS past the time first planned for it.
        due = self.plan and min(self.plan[0], self.plan_first + MAX_DRIFT_MS)
        if self.plan and due < t + dt_ms + self.lead_ms:
            _, px, py = self.plan
            when = max(due, t)
            under = [t + d[2] for d in dets if math.dist(d[:2], (px, py)) <= HOVER_PX]
            too_early = bool(under) and min(under) > when + EARLY_GUARD_MS
            if when - self.last_press >= REFRACTORY_MS and not too_early:
                events += self._press(when, (px, py))
            self.plan = None
        elif spinning:
            if self.key_down is None:
                events += self._press(t, self.cursor)
                self.spin_start = t
            a = 2 * math.pi * SPIN_RPS * (t - self.spin_start) / 1000
            self.cursor = (CENTER[0] + SPIN_RADIUS * math.cos(a), CENTER[1] + SPIN_RADIUS * math.sin(a))
        elif holding:
            self.cursor = (bx, by)  # follow the slider ball the model sees
        elif self.key_down is not None and t - self.key_down >= TAP_MS:
            events.append(self.release(t))
        return events

    def _press(self, when: float, pos) -> list[Event]:
        events = [self.release(when)] if self.key_down is not None else []
        self.track.clear()  # the next circle may sit on the same spot (a stack)
        self.pressed.append((when, *pos))
        self.key = 1 - self.key
        self.cursor = pos
        self.key_down = self.last_press = when
        return events + [Event("down", when, *pos, key=self.key)]

    def release(self, t: float) -> Event:
        """Let go of the held key (a tap is held at least TAP_MS)."""
        event = Event("up", max(t, self.key_down + TAP_MS), key=self.key)
        self.key_down = None
        return event
