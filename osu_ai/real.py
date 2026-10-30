"""Play the real osu!lazer: frames from OsuHands go through the model and policy into timed input.

Everything runs on the perf_counter_ns clock (mach time), which OsuHands shares. A frame's time
is when the WindowServer showed it (display_ns, else its pts), and the policy works in ms on
that frame clock. What it plans for frame-clock time T has to reach the game at T, so it is
sent at T - L, where L (latency_ms) is the whole capture + input delay; the policy decides
lead_ms >= L ahead so that there is still time. A press is a MOVE 6 ms before the key goes
down, because lazer applies keyboard input before mouse input within an update frame, and
the two keys alternate so that each press is a fresh one.

Safety: keys are released on any exit or error and whenever no frame has arrived for 500 ms;
the run stops when osu! loses focus, leaves the player, or the connection drops; and a press is
never sent onto a circle the model says is due well after it (that early, a press is a Miss).
Every frame, detection and sent event is recorded, for debugging and DAgger. The same loop
runs offline over recorded frames (dry run), with a stand-in for OsuHands.

The rest is what playing needs around that: finding osu!'s window, menus by keyboard with
every step confirmed in osu!'s runtime log, screenshots, footage capture, and reading back
how lazer judged a run.
"""

from __future__ import annotations

import gc
import json
import math
import statistics
import struct
import threading
import time
import zipfile
import zlib
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import numpy as np

from . import lazer
from .beatmap import Kind
from .geometry import VIRTUAL, OsuWindow
from .hands_client import (ALL_TAGS, CANCEL, KEY_DOWN, KEY_UP, KEY_X, KEY_Z, MOVE, RELEASE, Event, Frame,
                           HandsError, now_ns)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
CAPTURE = VIRTUAL  # 640x480 luma: 1 px per osu!px of the virtual screen
RECORD = (320, 240)  # stored frames (the model's largest input)
MOVE_LEAD_MS = 6.0  # a press's MOVE goes this long before its key: >= 1 ms input poll + 2.1 ms update frame
MOVE_GUARD_MS = 4.0  # no other move leaves this close before a press's MOVE (it could arrive after it)
RELEASE_GAP_MS = 2.0  # a key let go just before it is pressed again
MIN_HOLD_MS = 20.0
MOVE_HZ = 240.0  # cursor moves faster than lazer's 480 Hz update are wasted; 240 is plenty
WATCHDOG_MS = 500.0
NO_FRAMES_S = 60.0  # a still screen (a break) may send no frames at all; only give up after this long
STOP_POLL_MS = 100.0
HOVER_RADIUS = 40.0  # osu!px: a circle this close to a press is hovered (CS4 radius 36.5)
MAX_BALL_LEAD = 60.0  # osu!px: never extrapolate the slider ball further than this

# status of a recorded event
SENT, LATE, SKIPPED_EARLY, DROPPED_LATE, SUPPRESSED, RATE_LIMITED, NOT_HELD = range(7)
STATUS = ("sent", "late", "skipped_early", "dropped_late", "suppressed", "rate_limited", "not_held")


def capture_stops(hands) -> int:
    """How many "capture_stopped" events OsuHands has sent (the system ended a capture)."""
    return sum(isinstance(e, dict) and e.get("event") == "capture_stopped" for e in list(getattr(hands, "events", ())))


def downsample(pixels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Box-filter a (H, W) uint8 frame to (w, h) = size: each output pixel is the rounded mean of a k x k block."""
    h, w = pixels.shape
    k = w // size[0]
    if k < 1 or k * size[0] != w or k * size[1] != h:
        raise ValueError(f"cannot box-filter a {w}x{h} frame to {size[0]}x{size[1]}")
    if k == 1:
        return pixels
    s = pixels.reshape(size[1], k, size[0], k).sum(axis=(1, 3), dtype=np.uint32)
    return ((s + k * k // 2) // (k * k)).astype(np.uint8)


def save_npz(path: Path, frames: list[np.ndarray] | np.ndarray, **arrays) -> Path:
    """Like np.savez_compressed(path, frames=..., **arrays), but the frames are streamed into the
    file one by one: minutes of 320x240 footage never have to be stacked in memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for name, a in arrays.items():
            with z.open(f"{name}.npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(a), allow_pickle=False)
        shape = frames[0].shape if len(frames) else (RECORD[1], RECORD[0])
        with z.open("frames.npy", "w", force_zip64=True) as f:
            np.lib.format.write_array_header_1_0(f, {"descr": "|u1", "fortran_order": False,
                                                     "shape": (len(frames), *shape)})
            for frame in frames:
                if frame.shape != shape or frame.dtype != np.uint8:
                    raise ValueError(f"frame {frame.shape} {frame.dtype} in a stack of {shape} uint8")
                f.write(np.ascontiguousarray(frame).tobytes())
    return path


def save_recording(path: Path, frames: list[np.ndarray], pts_ms, meta: dict) -> Path:
    """A realdata.Recording file (frames, pts_ms, meta as JSON), written without stacking the frames."""
    return save_npz(path, frames, pts_ms=np.asarray(pts_ms, np.float64), meta=np.asarray(json.dumps(meta)))


@dataclass
class Sent:
    """One input event this run sent (or deliberately did not send)."""

    frame: int  # index of the frame whose step produced it
    kind: int  # MOVE, KEY_DOWN, KEY_UP (hands_client kinds)
    status: int  # SENT, LATE, ... (STATUS names)
    at_ns: int  # when it was scheduled to run (perf_counter ns); for immediate moves, when it was sent
    sent_ns: int  # when it left this process
    x: float = math.nan  # global points
    y: float = math.nan
    key: int = 0  # macOS keycode
    t_ms: float = math.nan  # the policy event's time on the frame clock
    ox: float = math.nan  # osu!px
    oy: float = math.nan


class DryHands:
    """Stands in for OsuHands in a dry run: keeps every input event instead of posting it."""

    def __init__(self):
        self.events: list[Event] = []
        self.frontmost = {"allowed": True}
        self.closed = None

    def send(self, *events: Event) -> None:
        self.events.extend(events)

    def cancel(self, tag: int = ALL_TAGS) -> None:
        self.events.append(Event(CANCEL, tag=tag))

    def release_all(self, at_ns: int = 0) -> None:
        self.events.append(Event(RELEASE, at_ns))


class RealGame:
    """One run of the agent in the real game (or a dry run over recorded frames).

    agent: has input_size (w, h) and see(frame (h, w) uint8) -> (dets, modes), with dets
    [(x, y, ms until hit, score)] in osu!px, soonest first, and modes (P(hold), P(spin), ball x,
    ball y) in osu!px. policy: policy.Policy (step(t_ms, dt_ms, dets, modes) -> events, cursor,
    key_down, release(t), lead_ms).

    Policy events become input: "move" sends the policy's cursor as the step left it, at once (at
    most MOVE_HZ; while following a slider, ahead of the ball by its velocity times the latency
    plus the frame's age); "down" at frame time T is a MOVE at T - L - 6 ms and the key (Z for
    key 0, X for key 1) at T - L; "up" is the key-up at T - L. Whatever is already due goes at
    once, a press keeping its MOVE 6 ms ahead; a press over max_late_ms late is dropped (a spinner's is not).
    """

    def __init__(self, hands, agent, policy, geometry: OsuWindow, latency_ms: float = 30.0, *,
                 fps: float = 60.0, min_gap_ms: float = 12.0, early_guard_ms: float | None = 100.0,
                 hover_radius: float = HOVER_RADIUS, max_late_ms: float = 100.0, record: bool = True,
                 stop: Callable[[], str | None] | None = None, log: Callable[[str], None] = print):
        lead = getattr(policy, "lead_ms", None)
        if lead is not None and lead < latency_ms:
            raise ValueError(f"policy.lead_ms = {lead} < latency_ms = {latency_ms}: every press would leave late; "
                             f"build the policy with lead_ms = latency + processing slack")
        self.hands, self.agent, self.policy, self.geometry = hands, agent, policy, geometry
        self.latency_ms, self.fps, self.min_gap_ms = latency_ms, fps, min_gap_ms
        self.early_guard_ms, self.hover_radius, self.max_late_ms = early_guard_ms, hover_radius, max_late_ms
        self.record, self.stop, self.log = record, stop, log
        self.input_size = tuple(agent.input_size)
        self.reason: str | None = None
        self.t0_ns: int | None = None  # frame-clock origin: display time of the first frame
        self.frame: Frame | None = None  # the frame being processed
        self.sent: list[Sent] = []
        self.rows: list[tuple] = []  # per frame: seq, pts, display, recv, arrived, step start, decided, idle, fed,
        #                              infer ms, dropped (times in perf_counter ns)
        self.small: list[np.ndarray] = []
        self.dets: list[list[tuple[float, float, float, float]]] = []
        self.modes: list[tuple[float, float, float, float]] = []
        self.cursor: list[tuple[float, float]] = []
        self._clock: Callable[[], int] = now_ns
        self._now = 0
        self._fed_ns: int | None = None
        self._recent: deque[tuple[float, list]] = deque(maxlen=2)  # (frame ms, dets) of the last fed frames
        self._balls: deque[tuple[float, float, float]] = deque(maxlen=4)  # (frame ms, x, y) while holding
        self._held: dict[int, dict] = {}  # keycode -> {"down": ns, "up": ns or None, "tag": int}
        self._pending: list[tuple[int, int, float, float]] = []  # presses in flight: (move ns, key ns, x, y)
        self._last_move_ns = 0
        self._tag = 0
        self._last_frame_ns = 0
        self._starved = False  # the watchdog released the keys
        self._done = threading.Event()
        self._next_stop_poll = 0
        self._stops = 0

    # running

    def run(self, max_s: float = 900.0, frames: Iterable[Frame] | None = None) -> str:
        """Play until something stops the run; returns why. Without `frames` this is live: the
        capture of the virtual screen is started here and stopped at the end. With `frames` it is a
        dry run: 'now' is each frame's arrival time and nothing waits.

        The garbage collector is off for the run: a collection can stall a frame for tens of ms,
        which would send presses late. A few minutes of play leaves little garbage behind."""
        gc.collect()
        gc.freeze()
        gc.disable()
        try:
            return self._run(max_s, frames)
        finally:
            gc.enable()
            gc.unfreeze()

    def _run(self, max_s: float, frames: Iterable[Frame] | None) -> str:
        live = frames is None
        it: Iterator[Frame] | None = None if live else iter(frames)
        if live:
            front = getattr(self.hands, "frontmost", None)
            if front is not None and front.get("allowed") is False:
                raise HandsError(f"osu! is not frontmost ({front.get('bundle_id')}): input would be dropped")
            self._clock = now_ns
        else:
            self._clock = lambda: self._now
        if hasattr(self.agent, "reset"):
            self.agent.reset()
        start = self._clock() if live else None
        self._last_frame_ns = start or 0
        self._stops = capture_stops(self.hands)
        last_seq = -1
        reason = None
        try:
            if live:
                self.hands.configure(clamp=self.geometry.rect)  # no stray move ever leaves osu!'s client area
                cap = self.hands.start_capture(self.geometry.capture_rect, size=CAPTURE, fps=self.fps, fmt="luma")
                self._last_frame_ns = self._clock()
                threading.Thread(target=self._watchdog, name="realgame-watchdog", daemon=True).start()
            while reason is None:
                if live:
                    f = self.hands.next_frame(after=last_seq, timeout=0.1)
                    if f is not None:
                        last_seq = f.seq
                        if f.config != cap.config:  # from before this capture (a screenshot, an older run)
                            f = None
                else:
                    f = next(it, None)
                    if f is None:
                        reason = "end of frames"
                        break
                    self._now = f.arrived_ns
                    if start is None:
                        start = f.arrived_ns
                reason = self._should_stop(start, max_s, live)
                if reason is None and f is not None:
                    self._step(f)
        except BaseException as e:
            reason = f"error: {e!r}"
            raise
        finally:
            self.reason = reason
            self._finish(live)
        return reason

    def _should_stop(self, start: int, max_s: float, live: bool) -> str | None:
        now = self._clock()
        if now - start > max_s * 1e9:
            return f"time limit ({max_s:g} s)"
        if getattr(self.hands, "closed", None):
            return f"OsuHands: {self.hands.closed}"
        front = getattr(self.hands, "frontmost", None)
        if front is not None and front.get("allowed") is False:
            return f"osu! lost focus to {front.get('bundle_id') or front.get('name')}"
        if live and now - self._last_frame_ns > NO_FRAMES_S * 1e9:
            return f"no frames for {NO_FRAMES_S:g} s"
        if live and capture_stops(self.hands) > self._stops:
            return "the system stopped the capture"
        if self.stop is not None and now >= self._next_stop_poll:
            self._next_stop_poll = now + int(STOP_POLL_MS * 1e6)
            return self.stop()
        return None

    def _watchdog(self) -> None:
        """Lets go of every key when frames stop coming (capture died, osu! hung, this process stalled)."""
        while not self._done.wait(0.05):
            if not self._starved and now_ns() - self._last_frame_ns > WATCHDOG_MS * 1e6:
                self._starved = True
                try:
                    self.hands.cancel(ALL_TAGS)
                    self.hands.release_all()
                except (HandsError, OSError):
                    pass
                self.log(f"watchdog: no frame for {WATCHDOG_MS:g} ms, released all keys")

    def _finish(self, live: bool) -> None:
        self._done.set()
        try:
            self.hands.cancel(ALL_TAGS)
            self.hands.release_all()
        except (HandsError, OSError) as e:
            self.log(f"could not release keys ({e}); OsuHands releases them when this process disconnects")
        if self.policy.key_down is not None:
            self.policy.release(self.policy.key_down)
        self._held.clear()
        if live:
            try:
                self.hands.stop_capture()
            except (HandsError, OSError):
                pass

    # one frame

    def _step(self, f: Frame) -> None:
        start = self._clock()
        self._last_frame_ns = start
        if self._starved:  # the watchdog let go of everything: start clean (again, in case it fired mid-step)
            self._starved = False
            self.hands.cancel(ALL_TAGS)
            self.hands.release_all()
            self._held.clear()
            self._pending.clear()
            if self.policy.key_down is not None:
                self.policy.release(self.policy.key_down)
        self.frame = f
        shown = f.display_ns or f.t_ns
        if self.t0_ns is None:
            self.t0_ns = shown
        t = (shown - self.t0_ns) / 1e6
        i = len(self.rows)
        feed = self._fed_ns is None or shown - self._fed_ns >= self.min_gap_ms * 1e6
        dets, modes, infer_ms, decided = [], (math.nan,) * 4, math.nan, 0
        if feed:
            x = downsample(f.pixels, self.input_size)
            t_inf = time.perf_counter_ns()
            dets, modes = self.agent.see(x)
            infer_ms = (time.perf_counter_ns() - t_inf) / 1e6
            dt = (shown - self._fed_ns) / 1e6 if self._fed_ns is not None else 1000 / self.fps
            self._fed_ns = shown
            self._recent.append((t, list(dets)))
            events = self.policy.step(t, dt, dets, modes)
            decided = self._clock()
            self._act(i, t, shown, events, modes)
        if self.record:
            self.small.append(downsample(f.pixels, RECORD) if f.pixels.shape[1] != RECORD[0] else f.pixels.copy())
            self.rows.append((f.seq, f.t_ns, f.display_ns, f.recv_ns, f.arrived_ns, start, decided, f.idle, feed,
                              infer_ms, f.dropped))
            self.dets.append([tuple(map(float, d)) for d in dets])
            self.modes.append(tuple(map(float, modes)))
            self.cursor.append(tuple(map(float, self.policy.cursor)) if feed else (math.nan, math.nan))

    def _to_ns(self, t_ms: float) -> int:
        return self.t0_ns + round(t_ms * 1e6)

    def _act(self, i: int, t: float, shown: int, events, modes) -> None:
        now = self._clock()
        self._pending = [p for p in self._pending if p[1] + RELEASE_GAP_MS * 1e6 >= now]
        for k in [k for k, h in self._held.items() if h["up"] is not None and h["up"] < now]:
            del self._held[k]
        moved = False
        for e in events:
            if e.kind == "down":
                self._press(i, e, now, spinning=modes[1] > 0.5)
            elif e.kind == "up":
                self._release(i, e, now)
            elif e.kind == "move":
                moved = True
            else:
                self.log(f"policy event of unknown kind {e.kind!r} ignored")
        if moved:
            self._move(i, t, shown, modes, now)

    def _next_tag(self) -> int:
        self._tag = self._tag % 0xFFFFFFF0 + 1
        return self._tag

    def _press(self, i: int, e, now: int, spinning: bool = False) -> None:
        """A spinner's press is decided for the frame it is seen in, so it is always L + the frame's
        age late; it holds the key for the whole spinner, so it is never dropped for that."""
        key = KEY_Z if e.key == 0 else KEY_X
        at = self._to_ns(e.t - self.latency_ms)
        px, py = self.geometry.post_point(e.x, e.y)
        rec = dict(frame=i, kind=KEY_DOWN, key=key, t_ms=e.t, ox=e.x, oy=e.y, x=px, y=py, at_ns=at, sent_ns=now)
        if self._too_early(e):
            self.sent.append(Sent(status=SKIPPED_EARLY, **rec))
            return
        if now - at > self.max_late_ms * 1e6 and not spinning:
            self.sent.append(Sent(status=DROPPED_LATE, **rec))
            return
        lead = int(MOVE_LEAD_MS * 1e6)
        key_at = max(at, now + lead)
        move_at = key_at - lead
        tag = self._next_tag()
        out = []
        held = self._held.get(key)
        if held is not None and (held["up"] is None or held["up"] > key_at - RELEASE_GAP_MS * 1e6):
            if held["up"] is not None and held["tag"]:
                out.append(Event(CANCEL, tag=held["tag"]))  # its scheduled release comes too late
            up_at = max(now, key_at - int(RELEASE_GAP_MS * 1e6))
            out.append(Event(KEY_UP, up_at, key, tag=tag))
            self.sent.append(Sent(i, KEY_UP, SENT, up_at, now, key=key, t_ms=e.t))
        out += [Event(MOVE, move_at, x=px, y=py, tag=tag), Event(KEY_DOWN, key_at, key, tag=tag)]
        self.hands.send(*out)
        self._held[key] = {"down": key_at, "up": None, "tag": 0}
        self._pending.append((move_at, key_at, px, py))
        status = SENT if key_at == at else LATE
        self.sent.append(Sent(i, MOVE, status, move_at, now, px, py, 0, e.t, e.x, e.y))
        self.sent.append(Sent(status=status, **{**rec, "at_ns": key_at}))

    def _release(self, i: int, e, now: int) -> None:
        key = KEY_Z if e.key == 0 else KEY_X
        held = self._held.get(key)
        at = self._to_ns(e.t - self.latency_ms)
        if held is None or held["up"] is not None:
            self.sent.append(Sent(i, KEY_UP, NOT_HELD, at, now, key=key, t_ms=e.t))
            return
        up_at = max(at, held["down"] + int(MIN_HOLD_MS * 1e6))
        tag = self._next_tag()
        self.hands.send(Event(KEY_UP, up_at, key, tag=tag))
        held.update(up=up_at, tag=tag)
        self.sent.append(Sent(i, KEY_UP, SENT if up_at >= now else LATE, up_at, now, key=key, t_ms=e.t))

    def _move(self, i: int, t: float, shown: int, modes, now: int) -> None:
        x, y = self.policy.cursor
        holding = getattr(self.policy, "holding", None)
        if holding is None:
            holding = self.policy.key_down is not None and modes[0] > 0.5
        if holding:
            x, y = self._lead_ball(t, x, y, now, shown)
        else:
            self._balls.clear()
        px, py = self.geometry.post_point(x, y)
        now = max(now, self._clock())  # the presses of this step took time to send
        status = SENT
        if now - self._last_move_ns < 1e9 / MOVE_HZ:
            status = RATE_LIMITED
        elif any(m - MOVE_GUARD_MS * 1e6 <= now <= k + RELEASE_GAP_MS * 1e6 and (px, py) != (wx, wy)
                 for m, k, wx, wy in self._pending):
            status = SUPPRESSED  # a press is about to land: the cursor must stay on its circle
        if status == SENT:
            self.hands.send(Event(MOVE, 0, x=px, y=py))
            self._last_move_ns = now
        self.sent.append(Sent(i, MOVE, status, now, now, px, py, 0, math.nan, x, y))

    def _lead_ball(self, t: float, bx: float, by: float, now: int, shown: int) -> tuple[float, float]:
        """Where the slider ball will be when a move sent now lands: the ball the frame shows,
        moved on by its velocity over the last frames for the latency plus the time since the frame."""
        if self._balls and math.dist(self._balls[-1][1:], (bx, by)) > 80:
            self._balls.clear()  # a new slider, or the ball estimate jumped
        self._balls.append((t, bx, by))
        t0, x0, y0 = self._balls[0]
        if t - t0 < 1:
            return bx, by
        lead = self.latency_ms + (now - shown) / 1e6
        dx, dy = (bx - x0) / (t - t0) * lead, (by - y0) / (t - t0) * lead
        d = math.hypot(dx, dy)
        if d > MAX_BALL_LEAD:
            dx, dy = dx * MAX_BALL_LEAD / d, dy * MAX_BALL_LEAD / d
        return bx + dx, by + dy

    def _too_early(self, e) -> bool:
        """Would this press land on a hovered circle long before that circle is due? Lazer gives a
        press to the earliest hovered circle, and judges one pressed more than 'meh' early a Miss."""
        if self.early_guard_ms is None:
            return False
        due = [t + ms for t, dets in self._recent for x, y, ms, *_ in dets
               if math.dist((x, y), (e.x, e.y)) <= self.hover_radius]
        return bool(due) and min(due) > e.t + self.early_guard_ms

    # results

    def summary(self) -> dict:
        fed = [r for r in self.rows if r[8]]
        arrive = [(r[5] - (r[2] or r[1])) / 1e6 for r in fed]
        decide = [(r[6] - (r[2] or r[1])) / 1e6 for r in fed]
        infer = [r[9] for r in fed]
        presses = [s for s in self.sent if s.kind == KEY_DOWN]
        late = [(s.at_ns - self._to_ns(s.t_ms - self.latency_ms)) / 1e6 for s in presses if s.status == LATE]

        def pct(v, q):
            return round(float(np.percentile(v, q)), 2) if v else None

        return {
            "reason": self.reason, "frames": len(self.rows), "fed": len(fed),
            "seq_gaps": int(sum(max(0, b[0] - a[0] - 1) for a, b in zip(self.rows, self.rows[1:]))),
            "shown_to_step_ms": {"p50": pct(arrive, 50), "p95": pct(arrive, 95)},
            "infer_ms": {"p50": pct(infer, 50), "p95": pct(infer, 95)},
            "shown_to_decision_ms": {"p50": pct(decide, 50), "p95": pct(decide, 95), "max": pct(decide, 100)},
            "presses": {STATUS[s]: sum(p.status == s for p in presses) for s in sorted({p.status for p in presses})},
            "late_by_ms": {"p50": pct(late, 50), "max": pct(late, 100)} if late else None,
            "moves": sum(s.kind == MOVE and s.status == SENT and math.isnan(s.t_ms) for s in self.sent),
        }

    def save(self, path: Path, meta: dict | None = None) -> Path:
        """Everything this run saw and did, for debugging and DAgger (np.load(path); meta is JSON)."""
        n = len(self.rows)

        def col(j: int, dtype) -> np.ndarray:
            return np.array([r[j] for r in self.rows], dtype)

        offsets = np.cumsum([0] + [len(d) for d in self.dets]).astype(np.int64)
        dets = np.array([d[:4] for ds in self.dets for d in ds], np.float32).reshape(-1, 4)
        sent = self.sent
        meta = {**(meta or {}), "latency_ms": self.latency_ms, "lead_ms": getattr(self.policy, "lead_ms", None),
                "fps": self.fps, "min_gap_ms": self.min_gap_ms, "early_guard_ms": self.early_guard_ms,
                "hover_radius": self.hover_radius, "input_size": list(self.input_size), "record_size": list(RECORD),
                "window": list(self.geometry.rect), "capture_rect": list(self.geometry.capture_rect),
                "t0_ns": self.t0_ns, "summary": self.summary(), "status_names": STATUS}
        return save_npz(
            path, self.small,
            seq=col(0, np.int64), pts_ns=col(1, np.int64), display_ns=col(2, np.int64), recv_ns=col(3, np.int64),
            arrived_ns=col(4, np.int64), step_ns=col(5, np.int64), decided_ns=col(6, np.int64), idle=col(7, bool),
            fed=col(8, bool), infer_ms=col(9, np.float32), dropped=col(10, np.int64),
            det_offsets=offsets, dets=dets, modes=np.array(self.modes, np.float32).reshape(n, 4),
            cursor=np.array(self.cursor, np.float32).reshape(n, 2),
            ev_frame=np.array([s.frame for s in sent], np.int32), ev_kind=np.array([s.kind for s in sent], np.int8),
            ev_status=np.array([s.status for s in sent], np.int8),
            ev_at_ns=np.array([s.at_ns for s in sent], np.int64),
            ev_sent_ns=np.array([s.sent_ns for s in sent], np.int64),
            ev_xy=np.array([(s.x, s.y) for s in sent], np.float64).reshape(-1, 2),
            ev_osu=np.array([(s.ox, s.oy) for s in sent], np.float32).reshape(-1, 2),
            ev_key=np.array([s.key for s in sent], np.int16), ev_t_ms=np.array([s.t_ms for s in sent], np.float64),
            meta=np.asarray(json.dumps(meta)),
        )


def recorded_frames(path: Path, delay_ms: float = 8.0) -> Iterator[Frame]:
    """Frames of a saved RealGame run or a realdata.Recording, for a dry run. A Recording has no
    arrival times, so frames 'arrive' delay_ms after they were shown."""
    d = np.load(path)
    frames = d["frames"]
    if "display_ns" in d.files:
        shown = np.where(d["display_ns"] > 0, d["display_ns"], d["pts_ns"])
        arrived, idle, seq = d["arrived_ns"], d["idle"], d["seq"]
    else:
        shown = np.round(d["pts_ms"] * 1e6).astype(np.int64)
        arrived = shown + int(delay_ms * 1e6)
        idle, seq = np.zeros(len(frames), bool), np.arange(len(frames))
    for i in range(len(frames)):
        s = int(shown[i])
        yield Frame(int(seq[i]), s, s, int(arrived[i]), int(arrived[i]), frames[i], 0, bool(idle[i]), 0)


class Footage:
    """Captures the virtual screen without playing: frames at RECORD size with their display times."""

    def __init__(self, hands, geometry: OsuWindow, fps: float = 60.0):
        self.hands, self.geometry, self.fps = hands, geometry, fps
        self.frames: list[np.ndarray] = []
        self.shown_ns: list[int] = []
        self.idle: list[bool] = []
        self.dropped = 0
        self.cap = None

    def start(self) -> None:
        self.cap = self.hands.start_capture(self.geometry.capture_rect, size=CAPTURE, fps=self.fps, fmt="luma")

    def run(self, until: Callable[[], str | None], max_s: float) -> str:
        """Record until until() gives a reason (checked every 100 ms) or max_s passes; returns why."""
        if self.cap is None:
            self.start()
        deadline = time.monotonic() + max_s
        last_seq, next_check, last_frame = -1, 0.0, time.monotonic()
        stops = capture_stops(self.hands)
        while True:
            f = self.hands.next_frame(after=last_seq, timeout=0.1)
            now = time.monotonic()
            if f is not None:
                last_seq = f.seq
                if f.config == self.cap.config:
                    self.frames.append(downsample(f.pixels, RECORD))
                    self.shown_ns.append(f.display_ns or f.t_ns)
                    self.idle.append(f.idle)
                    self.dropped = f.dropped
                    last_frame = now
            if now >= deadline:
                return f"time limit ({max_s:g} s)"
            if self.hands.closed:
                return f"OsuHands: {self.hands.closed}"
            if now - last_frame > NO_FRAMES_S:
                return f"no frames for {NO_FRAMES_S:g} s"
            if capture_stops(self.hands) > stops:
                return "the system stopped the capture"
            if now >= next_check:
                next_check = now + STOP_POLL_MS / 1000
                if reason := until():
                    return reason

    def stop(self) -> None:
        if self.cap is not None:
            self.hands.stop_capture()
            self.cap = None

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(frames (N, 240, 320) uint8, display times in ms on the perf_counter clock)."""
        frames = np.stack(self.frames) if self.frames else np.zeros((0, RECORD[1], RECORD[0]), np.uint8)
        return frames, self.pts_ms()

    def pts_ms(self) -> np.ndarray:
        return np.array(self.shown_ns, np.float64) / 1e6

    def save(self, path: Path, meta: dict) -> Path:
        """As a realdata.Recording (frames at 320x240, display time in ms, meta)."""
        return save_recording(path, self.frames, self.pts_ms(), meta)


# finding the window, screenshots

def locate(hands, client: tuple[int, int] | None | str = "auto", log: Callable[[str], None] = print) -> OsuWindow:
    """osu!'s client area, from CoreGraphics' window list (documented global points, no Screen
    Recording needed), cross-checked against ScreenCaptureKit's when that is allowed."""
    if client == "auto":
        client = lazer.windowed_size()
    cg = hands.osu_window(source="cg")
    sck = hands.osu_window(source="sck") if hands.permissions().get("screen_capture") else None
    if cg is None and sck is None:
        raise HandsError("no osu! window on screen: is osu! running, and not minimised?")
    if cg is not None and sck is not None and max(abs(a - b) for a, b in zip(cg["frame"], sck["frame"])) > 1:
        log(f"WARNING: window lists disagree: CoreGraphics {cg['frame']}, ScreenCaptureKit {sck['frame']}; "
            f"using CoreGraphics. Check a lazer_shot.py PNG shows exactly the client area.")
    return OsuWindow.from_window(cg or sck, client)


def grab(hands, rect, size: tuple[int, int], fmt: str = "bgra", n: int = 1, gap_s: float = 0.0,
         timeout: float = 3.0) -> list[np.ndarray]:
    """n frames of `rect` (global points) at `size` pixels, about gap_s apart (for menus, not gameplay)."""
    cap = hands.start_capture(rect, size=size, fps=30, fmt=fmt)
    out: list[np.ndarray] = []
    try:
        last, deadline = -1, time.monotonic() + timeout + n * gap_s
        while len(out) < n:
            f = hands.next_frame(after=last, timeout=0.5)
            if f is not None:
                last = f.seq
                if f.config == cap.config and (out or not f.idle):
                    out.append(f.pixels.copy())
                    if len(out) < n and gap_s:
                        time.sleep(gap_s)
            if time.monotonic() > deadline:
                raise HandsError(f"capture gave {len(out)} of {n} frames: is Screen Recording allowed for OsuHands?")
    finally:
        hands.stop_capture()
    return out


def save_png(path: Path, img: np.ndarray) -> Path:
    """Write a gray (H, W), RGB (H, W, 3) or BGRA (H, W, 4) uint8 image as a PNG (zlib only)."""
    if img.ndim == 2:
        color, rgb = 0, img
    elif img.shape[2] == 4:
        color, rgb = 2, img[..., 2::-1]
    else:
        color, rgb = 2, img
    h, w = rgb.shape[:2]
    rows = np.ascontiguousarray(rgb, dtype=np.uint8).reshape(h, -1)
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rows], axis=1).tobytes()

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return path


TOAST_RGB = ((61, 57, 70), (73, 69, 84))  # notification background (purple Background3), and hovered (Background2)
TOOLBAR_PT = 45.0  # the toolbar (log-in button included) is the top 40 pt of the client area


def find_toast(before: np.ndarray, after: list[np.ndarray]) -> tuple[float, float] | None:
    """Where to click the import toast, in client points, from BGRA shots of the client area at
    1 px per point: one from before the import and two from after it. The toast is the steady
    new block of notification purple in the top-right tray (300 pt wide, under the toolbar)."""
    h, w = before.shape[:2]
    x0, y0, y1 = max(0, w - 320), int(TOOLBAR_PT), min(h, 600)

    def rgb(img):
        return img[y0:y1, x0:, 2::-1].astype(np.int16)

    base, shots = rgb(before), [rgb(a) for a in after]
    purple = np.zeros(base.shape[:2], bool)
    for c in TOAST_RGB:
        purple |= (np.abs(shots[-1] - np.array(c, np.int16)) <= 10).all(axis=2)
    changed = np.all([np.abs(s - base).sum(axis=2) > 24 for s in shots], axis=0)
    steady = np.abs(shots[-1] - shots[0]).sum(axis=2) <= 12
    mask = purple & changed & steady
    rows = mask.sum(axis=1) >= 80
    top = int(np.argmax(rows)) if rows.any() else None
    if top is None:
        return None
    bottom = top
    while bottom + 1 < len(rows) and (rows[bottom + 1] or (bottom + 2 < len(rows) and rows[bottom + 2])):
        bottom += 1
    if bottom - top < 30:
        return None
    cols = np.nonzero(mask[top:bottom + 1].sum(axis=0) >= 0.3 * (bottom - top))[0]
    if len(cols) == 0 or cols[-1] - cols[0] < 150:
        return None
    left, right = cols[0], cols[-1]
    return x0 + left + 0.4 * (right - left), y0 + top + min(30.0, (bottom - top) / 2)


class Driver:
    """Gets osu! where we need it by keyboard and mouse, confirming every step in its runtime log.

    Only menu keys go out (never during play), clicks never land on the toolbar (where the log-in
    button is), and a failed step leaves a screenshot in runs/shots/ and raises.
    """

    def __init__(self, hands, tail: lazer.LogTail | None = None, geometry: OsuWindow | None = None,
                 shots: Path = RUNS / "shots", log: Callable[[str], None] = print):
        self.hands, self.log, self.shots = hands, log, shots
        self.tail = tail or lazer.LogTail()
        if not lazer.absolute_mouse(self.tail.data):
            raise RuntimeError("osu! would take our cursor moves as relative motion (no UseRelativeMode=false for "
                               "its MouseHandler in input.json): quit osu! and run scripts/lazer_setup.py first")
        self.geometry = geometry or locate(hands, log=log)
        self.hands.configure(clamp=self.geometry.rect)  # else OsuHands keeps whatever clamp an earlier client set
        self.typed = 40  # how much song select's search box may hold (it keeps its text between visits)

    # basics

    def front(self, timeout: float = 3.0) -> None:
        """Bring osu! to the front: OsuHands drops our input unless it is frontmost."""
        if (self.hands.frontmost or {}).get("allowed"):
            return
        lazer.activate()
        deadline = time.monotonic() + timeout
        while not (self.hands.frontmost or {}).get("allowed"):
            if time.monotonic() > deadline:
                self.fail(f"osu! is not frontmost ({(self.hands.frontmost or {}).get('bundle_id')})")
            time.sleep(0.05)

    def press(self, spec: str, gap_ms: float = 350.0, hold_ms: float = 60.0) -> None:
        self.front()
        end = self.hands.keys(spec, gap_ms=gap_ms, hold_ms=hold_ms)
        time.sleep(max(0.0, (end - now_ns()) / 1e9) + 0.05)

    def type(self, text: str, gap_ms: float = 70.0) -> None:
        self.press(" ".join("space" if c == " " else c for c in text.lower()), gap_ms=gap_ms, hold_ms=30.0)

    def click(self, cx: float, cy: float, allow_toolbar: bool = False) -> None:
        """Left-click at client point (cx, cy)."""
        if cy < TOOLBAR_PT and not allow_toolbar:
            raise ValueError(f"refusing to click the toolbar at y={cy:g} pt (its user button logs in)")
        self.front()
        x, y = self.geometry.client_point(cx, cy)
        self.hands.click(x, y)
        time.sleep(0.25)

    def park(self) -> None:
        """Cursor to the left of the window and still: the player loader only starts while the cursor
        is over it (not over its settings panel on the right) or idle for 1.5 s."""
        self.front()
        self.hands.move(*self.geometry.client_point(0.25 * self.geometry.width, 0.6 * self.geometry.height))
        time.sleep(0.1)

    def shot(self, name: str, scale: int = 2) -> Path | None:
        """Screenshot of the client area into runs/shots/ (None if capture is not possible)."""
        g = self.geometry
        try:
            img = grab(self.hands, g.rect, (int(g.width * scale), int(g.height * scale)))[0]
        except (HandsError, OSError) as e:
            self.log(f"screenshot failed: {e}")
            return None
        return save_png(self.shots / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}.png", img)

    def fail(self, message: str) -> NoReturn:
        path = self.shot("failure")
        raise RuntimeError(f"{message} (screen: {self.tail.screen}; screenshot: {path})")

    def wait_screen(self, names, timeout: float, since: int | None = None, what: str = "") -> str:
        got = self.tail.wait_screen(names, timeout, since)
        if got is None:
            self.fail(f"{what or 'waiting'}: no {names} within {timeout:g} s")
        return got

    # navigation

    def _back_until(self, targets: set[str], tries: int = 5) -> str:
        """Esc until the screen is one of `targets`, looking after every press: Esc twice on the
        main menu would open the exit dialog, and Esc in a play would pause it."""
        for _ in range(tries):
            self.tail.poll()
            if self.tail.screen == "Loader" or (self.tail.screen or "").startswith("Intro"):
                self.tail.wait(lambda: self.tail.screen == "MainMenu", 20.0)  # osu! is starting: never Esc its intro
            if self.tail.screen in targets:
                return self.tail.screen
            if self.tail.screen in lazer.PLAYERS:
                self.fail("in a player: leave it first (Driver.leave_player)")
            if self.tail.screen is None:
                self.fail("osu! is not running (its log shows no screen)")
            self.press("esc")
            self.tail.wait(lambda: self.tail.screen in targets, 3.0)
        self.fail(f"could not get back to {sorted(targets)}")

    def main_menu(self) -> None:
        self._back_until({"MainMenu"})

    def song_select(self) -> None:
        if self._back_until({"SoloSongSelect", "MainMenu"}) == "SoloSongSelect":
            return
        mark = self.tail.mark()
        for _ in range(3):  # logo -> Play -> Solo; stop as soon as song select is there
            self.press("p")
            if self.tail.wait_screen("SoloSongSelect", 0.6, since=mark):
                return
        self.wait_screen("SoloSongSelect", 8.0, since=mark, what="opening song select")

    def select(self, osu_path: Path, beatmap_id: int | None = None) -> str:
        """Filter song select down to this difficulty and check lazer selected it. Typing a
        BeatmapID matches lazer's online id offline; without one, title words and diff= are used."""
        self.song_select()
        time.sleep(0.5)  # song select slides in; keys typed before its search box is up are lost
        want = lazer.display_title(osu_path)
        meta = lazer.osu_metadata(osu_path)
        bid = meta.get("BeatmapID", "")
        bid = int(bid) if bid.isdigit() and int(bid) > 0 else beatmap_id
        queries = [str(bid)] if bid else []
        words = "".join(c if c.isalnum() and c.isascii() else " " for c in meta.get("Title", "").lower()).split()
        version = "".join(c if c.isalnum() and c.isascii() else " " for c in meta.get("Version", "").lower()).split()
        if words:
            queries.append(" ".join(words[:4] + ([f"diff={max(version, key=len)}"] if version else [])))
        for q in queries:
            self.press(" ".join(["backspace"] * (self.typed + 2)), gap_ms=15, hold_ms=10)
            self.type(q)
            self.typed = max(len(q), 10)
            if self.tail.wait(lambda: self.tail.beatmap == want, 4.0):
                time.sleep(1.0)  # the carousel settles; a later selection would log another line
                self.tail.poll()
                if self.tail.beatmap == want:
                    return want
        self.fail(f"song select did not select {want!r} (selected: {self.tail.beatmap!r})")

    def toggle_autoplay(self) -> None:
        """Mod select (F1), search "at", toggle it (Enter), close (F1). The search box starts focused and
        empty (ModSelectTextSearchStartsActive), so this always toggles exactly Autoplay."""
        if lazer.settings(data=self.tail.data).get("ModSelectTextSearchStartsActive", "True") != "True":
            self.fail("ModSelectTextSearchStartsActive is off: typing in mod select would toggle mods by hotkey")
        self.press("f1")
        time.sleep(0.4)
        self.type("at")
        self.press("enter")
        self.press("f1")
        time.sleep(0.4)

    def start(self, autoplay: str | None = None) -> str:
        """From song select: start the selected map (autoplay: "ctrl" for Ctrl+Enter) and return the
        player screen it entered."""
        if self.tail.screen != "SoloSongSelect":
            self.fail("start() needs song select")
        self.park()
        mark = self.tail.mark()
        self.press("ctrl+enter" if autoplay == "ctrl" else "enter")
        return self.wait_screen(lazer.PLAYERS, 30.0, since=mark, what="starting the map")

    def leave_player(self) -> None:
        """Out of a player: Esc leaves a replay at once, pauses a play (then Up selects Quit, Enter
        confirms), and on the fail screen Esc is Quit. Each step checks the screen first; a player
        that just started may ignore Esc, so the sequence is retried."""
        time.sleep(1.0)  # a player that has only just started may not pause yet
        for _ in range(3):
            self.tail.poll()
            if self.tail.screen not in lazer.PLAYERS:
                return
            mark = self.tail.mark()
            self.press("esc")
            if self.tail.wait(lambda: self.tail.screens_since(mark), 1.2):
                continue
            self.press("up")
            self.press("enter")
            self.tail.wait(lambda: self.tail.screens_since(mark), 2.0)
        self.tail.wait(lambda: self.tail.screen not in lazer.PLAYERS, 3.0)
        if self.tail.screen in lazer.PLAYERS:
            self.fail("could not leave the player")


# how lazer judged a run

def hit_offsets(bm, presses, radius_scale: float = 1.5,
                window_ms: float = 200.0) -> list[tuple[int, float, float, float]]:
    """Match key-downs (osr.Press, beatmap time) to circles and slider heads: for each object the
    press nearest in time within window_ms that is within radius_scale radii of it.
    Returns (object index, press - hit time in ms, cursor dx, cursor dy) per matched object."""
    r = bm.difficulty.radius * radius_scale
    used: set[int] = set()
    out = []
    for o in bm.objects:
        if o.kind is Kind.SPINNER:
            continue
        best = None
        for j, p in enumerate(presses):
            if j in used or abs(p.down - o.time) > window_ms or math.dist((p.x, p.y), o.pos) > r:
                continue
            if best is None or abs(p.down - o.time) < abs(presses[best].down - o.time):
                best = j
        if best is not None:
            used.add(best)
            p = presses[best]
            out.append((o.index, p.down - o.time, p.x - o.x, p.y - o.y))
    return out


def offset_report(bm, offsets, latency_ms: float) -> dict:
    """Summary of hit offsets: the median is the correction to add to latency_ms."""
    heads = sum(o.kind is not Kind.SPINNER for o in bm.objects)
    if not offsets:
        return {"objects": heads, "matched": 0}
    dt = [o[1] for o in offsets]
    w = bm.difficulty.windows
    return {
        "objects": heads, "matched": len(offsets),
        "offset_ms": {"median": round(statistics.median(dt), 1), "mean": round(statistics.fmean(dt), 1),
                      "sd": round(statistics.pstdev(dt), 1), "p10": round(float(np.percentile(dt, 10)), 1),
                      "p90": round(float(np.percentile(dt, 90)), 1)},
        "within": {"great": sum(abs(d) <= w.great for d in dt), "ok": sum(abs(d) <= w.ok for d in dt),
                   "meh": sum(abs(d) <= w.meh for d in dt)},
        "aim_osupx": {"median_dist": round(statistics.median(math.hypot(o[2], o[3]) for o in offsets), 2),
                      "mean_dx": round(statistics.fmean(o[2] for o in offsets), 2),
                      "mean_dy": round(statistics.fmean(o[3] for o in offsets), 2)},
        "suggested_latency_ms": round(latency_ms + statistics.median(dt), 1),
    }
