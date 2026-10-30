"""RealGame end to end without the game: synthetic frames (gray discs, shrinking white rings) go
through a fake agent that reads them, the real policy, and RealGame, into a stand-in for OsuHands
(dry run) or into hands_selftest (live path: real socket, scheduler and capture protocol).
Also the menu Driver against the selftest, the log tail, screenshots and the toast finder.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from osu_ai import geometry as g
from osu_ai import hands_client as hc
from osu_ai import lazer, real
from osu_ai.policy import Event as PEvent
from osu_ai.policy import Policy

WIN = g.OsuWindow(244, 107, 1024, 768)
BASE = 5_000_000_000_000  # an arbitrary perf_counter_ns origin
PERIOD = 1e9 / 60
PREEMPT, RADIUS, RING = 600.0, 36.5, 5.0
G0 = -200.0  # game time shown by the first frame
CIRCLES = ((1000, 100, 100), (1400, 420, 110), (1800, 250, 300), (2150, 60, 340), (2500, 440, 330))
SELFTEST = hc.ROOT / "hands_app" / "build" / "hands_selftest"


@lru_cache(maxsize=None)
def _dist(vx: float, vy: float) -> np.ndarray:
    yy, xx = np.mgrid[0:480, 0:640] + 0.5
    return np.hypot(xx - vx, yy - vy).astype(np.float32)


def render(game_ms: float, circles=CIRCLES) -> np.ndarray:
    """The 640x480 virtual screen: a gray disc per visible circle and its approach ring (white, RING px wide)."""
    img = np.zeros((480, 640), np.uint8)
    for t_hit, x, y in circles:
        if not t_hit - PREEMPT <= game_ms <= t_hit:
            continue
        d = _dist(*g.osu_to_virtual(x, y))
        img[d <= RADIUS] = 100
        ring = RADIUS * (1 + 3 * (t_hit - game_ms) / PREEMPT)
        img[(d >= ring) & (d <= ring + RING)] = 255
    return img


@lru_cache(maxsize=None)
def _frames(n_ms: float, period_ns: float, delay_ms: float, circles: tuple) -> tuple:
    out, k = [], 0
    while G0 + k * period_ns / 1e6 <= n_ms:
        shown = BASE + round(k * period_ns)
        arrived = shown + round(delay_ms * 1e6)
        out.append(hc.Frame(k, shown, shown, arrived, arrived, render(G0 + k * period_ns / 1e6, circles), 0, False, 0))
        k += 1
    return tuple(out)


def frames(n_ms: float = 2800, period_ns: float = PERIOD, delay_ms: float = 8.0, circles=CIRCLES) -> list:
    return list(_frames(n_ms, period_ns, delay_ms, tuple(circles)))


def hit_ns(t_hit: float, latency_ms: float) -> int:
    """When the key-down for a circle due at game time t_hit should go out."""
    return BASE + round((t_hit - G0 - latency_ms) * 1e6)


class RingAgent:
    """Finds the known circles in a frame and reads their time from the approach ring's radius:
    along rays from the centre, the middle of the first bright run is the middle of the ring."""

    def __init__(self, size=(320, 240), circles=CIRCLES):
        self.input_size, self.circles, self.calls = size, circles, 0

    def see(self, frame: np.ndarray):
        self.calls += 1
        assert frame.shape == (self.input_size[1], self.input_size[0]) and frame.dtype == np.uint8
        k = 640 / self.input_size[0]
        r = RADIUS / k
        dets = []
        for _, x, y in self.circles:
            u, v = g.osu_to_input(x, y, self.input_size)
            if frame[int(v), int(u)] < 60:
                continue
            found = []
            for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                s = np.arange(0.8 * r, 4.6 * r, 0.05)
                px, py = u + s * np.cos(a) - 0.5, v + s * np.sin(a) - 0.5
                ok = (px >= 0) & (py >= 0) & (px < frame.shape[1] - 1) & (py < frame.shape[0] - 1)
                if ok.sum() < 5:
                    continue
                px, py, s = px[ok], py[ok], s[ok]
                x0, y0 = px.astype(int), py.astype(int)
                fx, fy = px - x0, py - y0
                f = frame.astype(np.float32)
                val = (f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x0 + 1] * fx * (1 - fy)
                       + f[y0 + 1, x0] * (1 - fx) * fy + f[y0 + 1, x0 + 1] * fx * fy)
                bright = np.nonzero(val > 140)[0]
                if len(bright):
                    run = bright[: np.argmax(np.diff(bright, append=bright[-1] + 2) > 1) + 1]
                    found.append((s[run[0]] + s[run[-1]]) / 2)
            if found:
                ring = float(np.median(found)) * k - RING / 2  # inner edge of the ring
                dets.append((x, y, PREEMPT * (ring / RADIUS - 1) / 3, 1.0))
        return sorted(dets, key=lambda d: d[2]), (0.0, 0.0, 256.0, 192.0)


def run_dry(agent=None, policy=None, latency_ms=30.0, slack_ms=25.0, source=None, **kw):
    hands = real.DryHands()
    agent = agent or RingAgent()
    policy = policy or Policy(lead_ms=latency_ms + slack_ms)
    game = real.RealGame(hands, agent, policy, WIN, latency_ms, log=lambda m: None, **kw)
    reason = game.run(frames=source if source is not None else frames())
    return game, hands, reason


def kinds(hands, kind):
    return [e for e in hands.events if e.kind == kind]


# the frame -> model -> policy -> input pipeline, dry


def test_downsample_is_a_rounded_box_filter():
    a = np.arange(16, dtype=np.uint8).reshape(4, 4)
    assert real.downsample(a, (2, 2)).tolist() == [[3, 5], [11, 13]]  # means 2.5, 4.5, 10.5, 12.5 round up
    big = np.random.default_rng(0).integers(0, 256, (480, 640), dtype=np.uint8)
    small = real.downsample(big, (160, 120))
    assert small.shape == (120, 160) and abs(int(small[7, 9]) - big[28:32, 36:40].mean()) <= 0.5
    assert real.downsample(big, (640, 480)) is big
    with pytest.raises(ValueError):
        real.downsample(big, (300, 240))


def test_ring_agent_reads_time_from_the_picture():
    agent = RingAgent()
    for game_ms in (500, 700, 900, 990):
        dets, _ = agent.see(real.downsample(render(game_ms), (320, 240)))
        assert dets[0][:2] == (100, 100)
        assert dets[0][2] == pytest.approx(1000 - game_ms, abs=6)


def test_each_circle_is_pressed_once_at_its_time_minus_latency():
    game, hands, reason = run_dry(latency_ms=30.0)
    assert reason == "end of frames"
    downs = kinds(hands, hc.KEY_DOWN)
    assert len(downs) == len(CIRCLES)
    assert [e.key for e in downs] == [hc.KEY_Z, hc.KEY_X] * 2 + [hc.KEY_Z]  # alternating keys
    for e, (t_hit, x, y) in zip(downs, CIRCLES):
        assert abs(e.at_ns - hit_ns(t_hit, 30.0)) < 8e6, (t_hit, (e.at_ns - hit_ns(t_hit, 30.0)) / 1e6)
        move = next(m for m in hands.events if m.kind == hc.MOVE and m.tag == e.tag)
        assert e.at_ns - move.at_ns == 6_000_000  # the cursor is there an update frame before the key
        assert math.dist((move.x, move.y), WIN.osu_to_points(x, y)) <= 1.2
        assert (move.x - WIN.left) % 1 == 0 and (move.y - WIN.top) % 1 == 0  # whole points from the client origin
    s = game.summary()
    assert s["presses"] == {"sent": 5} and s["fed"] == s["frames"]
    assert hands.events[-2:] == [hc.Event(hc.CANCEL, tag=hc.ALL_TAGS), hc.Event(hc.RELEASE)]


def test_keys_go_up_after_they_went_down():
    _, hands, _ = run_dry()
    for key in (hc.KEY_Z, hc.KEY_X):
        seq = [(e.at_ns, e.kind) for e in hands.events if e.key == key and e.kind in (hc.KEY_DOWN, hc.KEY_UP)]
        seq.sort()
        assert [k for _, k in seq] == [hc.KEY_DOWN, hc.KEY_UP] * (len(seq) // 2)
        assert all(b[0] - a[0] >= real.MIN_HOLD_MS * 1e6 for a, b in zip(seq[::2], seq[1::2]))


def test_a_larger_latency_sends_everything_earlier():
    _, h30, _ = run_dry(latency_ms=30.0)
    _, h50, _ = run_dry(latency_ms=50.0)
    d30, d50 = kinds(h30, hc.KEY_DOWN), kinds(h50, hc.KEY_DOWN)
    assert len(d30) == len(d50) == 5
    for a, b in zip(d30, d50):
        assert (a.at_ns - b.at_ns) / 1e6 == pytest.approx(20, abs=8)


def test_presses_that_cannot_leave_in_time_go_at_once_with_the_move_first():
    game, hands, _ = run_dry(source=frames(delay_ms=45.0), latency_ms=30.0, slack_ms=10.0)
    downs = [s for s in game.sent if s.kind == hc.KEY_DOWN]
    assert {s.status for s in downs} == {real.LATE}
    for s in downs:
        move = next(m for m in game.sent if m.kind == hc.MOVE and m.frame == s.frame and m.t_ms == s.t_ms)
        assert move.at_ns == s.sent_ns and s.at_ns == move.at_ns + 6_000_000  # move now, key 6 ms later
        assert s.at_ns - hit_ns(next(t for t, x, y in CIRCLES if (x, y) == (s.ox, s.oy)), 30.0) < 40e6


def test_120_fps_feeds_the_model_frames_at_least_15_ms_apart():
    agent = RingAgent()
    game, hands, _ = run_dry(agent=agent, source=frames(period_ns=1e9 / 120))
    fed = [r[8] for r in game.rows]
    assert len(fed) > 300 and agent.calls == sum(fed) == pytest.approx(len(fed) / 2, abs=1)
    shown = [r[2] for r in game.rows if r[8]]
    assert min(np.diff(shown)) >= 15e6
    assert len(kinds(hands, hc.KEY_DOWN)) == 5


def test_160x120_agents_get_box_filtered_frames_too():
    agent = RingAgent(size=(160, 120))
    _, hands, _ = run_dry(agent=agent)
    downs = kinds(hands, hc.KEY_DOWN)
    assert len(downs) == 5
    for e, (t_hit, *_) in zip(downs, CIRCLES):
        assert abs(e.at_ns - hit_ns(t_hit, 30.0)) < 12e6


class Scripted:
    """A policy that replays fixed events at given frames (to test RealGame's side exactly)."""

    def __init__(self, script, lead_ms=60.0, holding=False):
        self.script, self.lead_ms, self.holding = script, lead_ms, holding
        self.cursor, self.key_down, self.n = (256.0, 192.0), None, 0

    def step(self, t, dt_ms, dets, modes):
        events = [PEvent("move", t, *self.cursor)]
        for e in self.script.get(self.n, []):
            e = e(t, modes) if callable(e) else e
            if e.kind == "down":
                self.key_down = e.t
            elif e.kind == "move":
                self.cursor = (e.x, e.y)
                continue
            events.append(e)
        self.n += 1
        return events

    def release(self, t):
        self.key_down = None
        return PEvent("up", t)


class Blind:
    input_size = (320, 240)

    def __init__(self, dets=(), modes=(0.0, 0.0, 256.0, 192.0)):
        self.dets, self.modes = list(dets), modes

    def see(self, frame):
        return self.dets, self.modes


def test_a_press_onto_a_circle_due_much_later_is_not_sent():
    # The model sees a circle at (100, 100) due 400 ms after frame 3; a press there 50 ms after frame 3
    # would be a Miss in lazer, so it is skipped. A press in empty space is fine.
    agent = Blind(dets=[(100.0, 100.0, 400.0, 0.9)])
    script = {3: [lambda t, m: PEvent("down", t + 50, 100, 100, key=0)],
              5: [lambda t, m: PEvent("down", t + 70, 300, 300, key=1)]}
    game, hands, _ = run_dry(agent=agent, policy=Scripted(script), source=frames(400))
    statuses = [(s.status, s.ox) for s in game.sent if s.kind == hc.KEY_DOWN]
    assert statuses == [(real.SKIPPED_EARLY, 100.0), (real.SENT, 300.0)]
    assert [e.key for e in kinds(hands, hc.KEY_DOWN)] == [hc.KEY_X]


def test_a_key_still_held_is_let_go_just_before_it_is_pressed_again():
    script = {2: [lambda t, m: PEvent("down", t + 80, 200, 200, key=0)],
              4: [lambda t, m: PEvent("down", t + 80, 210, 200, key=0)]}  # no "up" in between
    game, hands, _ = run_dry(agent=Blind(), policy=Scripted(script), source=frames(400))
    z = sorted((e.at_ns, e.kind) for e in hands.events if e.key == hc.KEY_Z and e.kind in (hc.KEY_DOWN, hc.KEY_UP))
    assert [k for _, k in z] == [hc.KEY_DOWN, hc.KEY_UP, hc.KEY_DOWN]
    assert z[2][0] - z[1][0] == real.RELEASE_GAP_MS * 1e6


def test_a_release_never_overtakes_a_late_press():
    script = {2: [lambda t, m: PEvent("down", t, 200, 200, key=0), lambda t, m: PEvent("up", t + 5, key=0)]}
    _, hands, _ = run_dry(agent=Blind(), policy=Scripted(script), source=frames(300, delay_ms=40))
    down, up = kinds(hands, hc.KEY_DOWN)[0], kinds(hands, hc.KEY_UP)[0]
    assert up.at_ns - down.at_ns >= real.MIN_HOLD_MS * 1e6


def test_far_too_late_presses_are_dropped():
    script = {2: [lambda t, m: PEvent("down", t - 200, 200, 200, key=0)]}
    game, hands, _ = run_dry(agent=Blind(), policy=Scripted(script), source=frames(300))
    assert [s.status for s in game.sent if s.kind == hc.KEY_DOWN] == [real.DROPPED_LATE]
    assert not kinds(hands, hc.KEY_DOWN)


def test_a_spinners_press_is_never_dropped_for_lateness():
    # The policy presses for a spinner in the frame that shows it, so with 120 ms of capture and inference
    # that press is always over max_late_ms behind; it still has to go out (the key is held for the whole
    # spinner), while a circle's press that late is dropped.
    script = {2: [lambda t, m: PEvent("down", t, 256, 192, key=0)]}
    spinning = Blind(modes=(0.0, 1.0, 256.0, 192.0))
    game, hands, _ = run_dry(agent=spinning, policy=Scripted(script), source=frames(300, delay_ms=120))
    assert [s.status for s in game.sent if s.kind == hc.KEY_DOWN] == [real.LATE]
    assert [e.key for e in kinds(hands, hc.KEY_DOWN)] == [hc.KEY_Z]
    game, hands, _ = run_dry(agent=Blind(), policy=Scripted(script), source=frames(300, delay_ms=120))
    assert [s.status for s in game.sent if s.kind == hc.KEY_DOWN] == [real.DROPPED_LATE]


def test_late_presses_report_how_late_their_key_went():
    game, _, _ = run_dry(source=frames(delay_ms=45.0), latency_ms=30.0, slack_ms=10.0)
    late = [(s.at_ns - game._to_ns(s.t_ms - 30.0)) / 1e6 for s in game.sent if s.kind == hc.KEY_DOWN]
    assert len(late) == 5 and min(late) > 20  # 45 ms old frames, 10 ms of slack, a 6 ms move lead
    assert game.summary()["late_by_ms"] == {"p50": round(float(np.median(late)), 2), "max": round(max(late), 2)}


class StaleCapture:
    """A live OsuHands whose capture only ever delivers frames of an older capture config."""

    def __init__(self):
        self.frontmost, self.closed, self.events, self.sent, self.stopped = {"allowed": True}, None, [], [], False

    def configure(self, **kw):
        pass

    def start_capture(self, rect, size, fps, fmt):
        return hc.Capture(tuple(rect), tuple(size), 2)

    def stop_capture(self):
        self.stopped = True

    def next_frame(self, after=-1, timeout=1.0):
        time.sleep(0.01)
        t = hc.now_ns()
        return hc.Frame(after + 1, t, t, t, t, np.zeros((768, 1024, 4), np.uint8), 1, False, 0)

    def send(self, *events):
        self.sent.extend(events)

    def cancel(self, tag=hc.ALL_TAGS):
        self.send(hc.Event(hc.CANCEL, tag=tag))

    def release_all(self, at_ns=0):
        self.send(hc.Event(hc.RELEASE, at_ns))


def test_frames_of_another_capture_are_ignored_and_the_run_still_ends():
    hands = StaleCapture()
    game = real.RealGame(hands, Blind(), Policy(lead_ms=60), WIN, 30, log=lambda m: None)
    assert game.run(max_s=0.3) == "time limit (0.3 s)"
    assert not game.rows and hands.stopped
    assert hands.sent[-2:] == [hc.Event(hc.CANCEL, tag=hc.ALL_TAGS), hc.Event(hc.RELEASE)]


def test_the_cursor_stays_put_while_a_press_is_landing():
    # Press at frame 2 lands ~45 ms later; frames 3..4 want the cursor elsewhere meanwhile.
    script = {2: [lambda t, m: PEvent("down", t + 75, 200, 200, key=0)],
              3: [PEvent("move", 0, 400, 300)], 4: [PEvent("move", 0, 400, 300)]}
    game, hands, _ = run_dry(agent=Blind(), policy=Scripted(script), source=frames(300, delay_ms=8))
    press = next(s for s in game.sent if s.kind == hc.KEY_DOWN)
    for s in game.sent:
        if s.kind == hc.MOVE and math.isnan(s.t_ms) and s.status == real.SENT:
            assert not (press.at_ns - 7e6 <= s.at_ns <= press.at_ns + 2e6)  # nothing moves it away in the window
    assert any(s.status == real.SUPPRESSED for s in game.sent)


def test_no_move_leaves_just_before_a_press_moves_the_cursor():
    # Frame 4 arrives 8 ms before the key (2 ms before the press's own MOVE): a move sent then could
    # reach osu! after that MOVE, so it is held back too.
    script = {2: [lambda t, m: PEvent("down", t + 75, 200, 200, key=0)], 4: [PEvent("move", 0, 400, 300)]}
    game, _, _ = run_dry(agent=Blind(), policy=Scripted(script), source=frames(300, delay_ms=3.5))
    press = next(s for s in game.sent if s.kind == hc.KEY_DOWN)
    late_moves = [s for s in game.sent if s.kind == hc.MOVE and math.isnan(s.t_ms) and s.frame == 4]
    assert [(s.status, round((press.at_ns - s.at_ns) / 1e6, 1)) for s in late_moves] == [(real.SUPPRESSED, 8.2)]


def test_slider_following_leads_the_ball_by_the_latency():
    # The ball moves right at 0.3 osu!px/ms; moves sent while holding are ahead of it by
    # 0.3 * (latency + time since the frame was shown).
    speed, latency, delay = 0.3, 30.0, 8.0

    class Ball(Scripted):
        def step(self, t, dt_ms, dets, modes):
            self.key_down = 0.0
            self.cursor = (100 + speed * t, 200.0)
            return [PEvent("move", t, *self.cursor)]

    game, hands, _ = run_dry(agent=Blind(), policy=Ball({}, holding=True), source=frames(400, delay_ms=delay))
    moves = [s for s in game.sent if s.kind == hc.MOVE and s.status == real.SENT]
    t0 = game.t0_ns
    for s in moves[3:]:
        frame_ms = (game.rows[s.frame][2] - t0) / 1e6
        assert s.ox == pytest.approx(100 + speed * (frame_ms + latency + delay), abs=0.01)


def test_an_error_releases_everything_and_propagates():
    class Broken(Blind):
        def see(self, frame):
            raise RuntimeError("model fell over")

    hands = real.DryHands()
    game = real.RealGame(hands, Broken(), Policy(lead_ms=60), WIN, 30, log=lambda m: None)
    with pytest.raises(RuntimeError, match="fell over"):
        game.run(frames=frames(100))
    assert hands.events[-2:] == [hc.Event(hc.CANCEL, tag=hc.ALL_TAGS), hc.Event(hc.RELEASE)]
    assert game.reason.startswith("error")


def test_stop_and_focus_loss_end_the_run():
    hands = real.DryHands()
    polls = []

    def stop():
        polls.append(1)
        return "left the player" if len(polls) > 3 else None

    game = real.RealGame(hands, Blind(), Policy(lead_ms=60), WIN, 30, log=lambda m: None, stop=stop)
    assert game.run(frames=frames(2000)) == "left the player"
    assert 18 <= len(game.rows) <= 24  # polled every 100 ms of frame arrivals
    hands2 = real.DryHands()
    game2 = real.RealGame(hands2, Blind(), Policy(lead_ms=60), WIN, 30, log=lambda m: None)
    src = frames(300)

    def lose_focus():
        for i, f in enumerate(src):
            if i == 5:
                hands2.frontmost = {"allowed": False, "bundle_id": "com.apple.finder"}
            yield f

    assert "lost focus" in game2.run(frames=lose_focus())
    assert len(game2.rows) == 5


def test_policy_lead_must_cover_the_latency():
    with pytest.raises(ValueError, match="lead_ms"):
        real.RealGame(real.DryHands(), Blind(), Policy(lead_ms=10), WIN, 30)


def test_record_round_trips_and_replays_dry(tmp_path):
    game, hands, _ = run_dry()
    path = game.save(tmp_path / "run.npz", {"map": "synthetic"})
    d = np.load(path)
    n = len(game.rows)
    assert d["frames"].shape == (n, 240, 320) and d["frames"].dtype == np.uint8
    assert d["display_ns"][0] == BASE and d["fed"].all() and d["det_offsets"][-1] == len(d["dets"])
    meta = json.loads(str(d["meta"]))
    assert meta["map"] == "synthetic" and meta["latency_ms"] == 30.0 and meta["window"] == [244, 107, 1024, 768]
    downs = d["ev_kind"] == hc.KEY_DOWN
    assert downs.sum() == 5 and (d["ev_status"][downs] == real.SENT).all()
    # the recorded 320x240 frames drive a second, identical dry run
    replayed = list(real.recorded_frames(path))
    assert len(replayed) == n and replayed[3].display_ns == d["display_ns"][3]
    _, hands2, _ = run_dry(source=replayed)
    assert [e.at_ns for e in kinds(hands2, hc.KEY_DOWN)] == [e.at_ns for e in kinds(hands, hc.KEY_DOWN)]


# live path, against hands_selftest (real socket server, scheduler and frame protocol)

live = pytest.mark.skipif(not SELFTEST.exists(), reason="run hands_app/build.sh first")


@pytest.fixture
def selftest():
    path = hc.ROOT / "run" / f"selftest-real-{os.getpid()}.sock"
    path.parent.mkdir(exist_ok=True)
    proc = subprocess.Popen([str(SELFTEST), "--socket", str(path)])
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                h = hc.Hands(path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                assert time.monotonic() < deadline
                time.sleep(0.02)
        yield h
        h.close()
    finally:
        proc.terminate()
        proc.wait(5)
        path.unlink(missing_ok=True)


@live
def test_live_press_is_posted_at_hit_time_minus_latency(selftest):
    h, latency = selftest, 30.0
    h.request("_fake_log")
    due = {}

    class Clocked(Blind):
        def see(self, frame):
            shown = game.frame.display_ns or game.frame.t_ns
            due.setdefault("ns", shown + 400_000_000)
            return [(256.0, 192.0, (due["ns"] - shown) / 1e6, 1.0)], self.modes

    game = real.RealGame(h, Clocked(), Policy(lead_ms=latency + 25), WIN, latency, log=lambda m: None,
                         stop=lambda: "done" if "ns" in due and hc.now_ns() > due["ns"] + 150e6 else None)
    assert game.run() == "done"
    log = h.request("_fake_log")["log"]
    downs = [e for e in log if e[0] == "down"]
    assert len(downs) == 1 and downs[0][1] == hc.KEY_Z
    assert abs(downs[0][4] - (due["ns"] - latency * 1e6)) < 3e6  # posted on time, to the ms
    before = [e for e in log if e[0] == "move" and e[4] <= downs[0][4]]
    x, y = WIN.post_point(256, 192)
    assert before[-1][2:4] == [x, y] and downs[0][4] - before[-1][4] >= 5.5e6
    assert any(e[0] == "up" and e[1] == hc.KEY_Z and e[4] > downs[0][4] for e in log)
    s = game.summary()
    assert s["fed"] >= 25 and s["presses"] == {"sent": 1}


@live
def test_live_watchdog_lets_go_when_frames_stop(selftest):
    h = selftest
    h.request("_fake_log")
    released = threading.Event()

    spinner = Blind(modes=(0.0, 1.0, 256.0, 192.0))  # the policy holds a key down to spin
    game = real.RealGame(h, spinner, Policy(lead_ms=55), WIN, 30,
                         log=lambda m: released.set() if "watchdog" in m else None,
                         stop=lambda: "seen" if released.is_set() else None)
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("reason", game.run()))
    t.start()
    time.sleep(0.4)
    h.request("capture_stop")  # frames stop arriving
    t.join(5)
    assert result["reason"] == "seen"
    log = h.request("_fake_log")["log"]
    downs = [e for e in log if e[0] == "down"]
    ups = [e for e in log if e[0] == "up" and e[1] == hc.KEY_Z]
    assert downs and ups and ups[0][4] > downs[0][4]
    assert h.stats()["input"]["held_keys"] == []


@live
def test_live_stops_when_the_system_ends_the_capture(selftest):
    game = real.RealGame(selftest, Blind(), Policy(lead_ms=55), WIN, 30, log=lambda m: None)
    threading.Timer(0.3, lambda: selftest.events.append({"event": "capture_stopped"})).start()
    assert game.run(max_s=5) == "the system stopped the capture"
    assert 10 <= len(game.rows) <= 40


@live
def test_idle_frames_never_carry_another_captures_pixels(selftest):
    h = selftest
    h.start_capture((0, 0, 64, 48), size=(64, 48), fps=60, fmt="bgra")
    shot = h.next_frame(timeout=1.0)
    assert shot is not None and shot.pixels.shape == (48, 64, 4)
    h.request("_fake_idle", every=1)  # from now on the screen "never changes"
    cap = h.start_capture((0, 0, 64, 48), size=(64, 48), fps=60, fmt="luma")
    deadline, seen = time.monotonic() + 0.3, []
    while time.monotonic() < deadline:
        if (f := h.next_frame(timeout=0.05)) is not None:
            seen.append(f)
    assert not [f for f in seen if f.config == cap.config]  # idle, with nothing of its own to repeat
    h.request("_fake_idle", every=0)


@live
def test_live_refuses_to_start_without_focus(selftest):
    selftest.request("_fake_front", bundle="com.apple.finder")
    time.sleep(0.05)
    game = real.RealGame(selftest, Blind(), Policy(lead_ms=55), WIN, 30, log=lambda m: None)
    with pytest.raises(hc.HandsError, match="frontmost"):
        game.run()


# menus, screenshots, the log


def write_log(data: Path, name: str, lines: list[str], kind: str = "runtime") -> Path:
    logs = data / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{name}.{kind}.log"
    with path.open("a") as f:
        for line in lines:
            f.write(f"2026-10-23 04:12:08 [verbose]: {line}\n")
    return path


MOUSE = "osu.Framework.Input.Handlers.Mouse.MouseHandler, osu.Framework"


def absolute_mouse(data: Path, relative: bool | None = False) -> None:
    """input.json as lazer writes it (relative=None: no mouse entry)."""
    data.mkdir(parents=True, exist_ok=True)
    handlers = [{"$type": "osu.Framework.Input.Handlers.Keyboard.KeyboardHandler, osu.Framework", "Enabled": True}]
    if relative is not None:
        handlers.append({"$type": MOUSE, "UseRelativeMode": relative, "Sensitivity": 1.0, "Enabled": True})
    (data / "input.json").write_text(json.dumps({"InputHandlers": handlers}))


def test_input_must_be_absolute(tmp_path):
    assert not lazer.absolute_mouse(tmp_path)  # no input.json: this build defaults to relative mode on macOS
    for relative, ok in ((True, False), (None, False), (False, True)):
        absolute_mouse(tmp_path, relative)
        assert lazer.absolute_mouse(tmp_path) is ok
    (tmp_path / "input.json").write_text("{not json")
    assert not lazer.absolute_mouse(tmp_path)
    absolute_mouse(tmp_path, True)
    with pytest.raises(RuntimeError, match="lazer_setup"):
        real.Driver(SimpleNamespace(), lazer.LogTail(tmp_path), WIN, log=lambda m: None)


def test_log_tail_follows_the_osu_screen_stack(tmp_path):
    write_log(tmp_path, "100", ["📺 OsuScreenStack#473(depth:1) entered Loader#461",
                                "📺 BackgroundScreenStack#584(depth:1) entered BackgroundScreenDefault#225",
                                "📺 OsuScreenStack#473(depth:3) entered MainMenu#324",
                                "Game-wide working beatmap updated to A - B (C) [D]"])
    tail = lazer.LogTail(tmp_path)
    assert tail.screen == "MainMenu" and tail.beatmap == "A - B (C) [D]"
    mark = tail.mark()
    path = write_log(tmp_path, "100", ["📺 OsuScreenStack#473(depth:4) entered SoloSongSelect#1",
                                       "📺 OsuScreenStack#473(depth:5) entered PlayerLoader#2"])
    with path.open("a") as f:
        f.write("2026-10-23 04:12:09 [verbose]: 📺 OsuScreenStack#473(depth:6) entered SoloPla")  # half a line
    assert tail.wait_screen("PlayerLoader", 1, since=mark) == "PlayerLoader"
    with path.open("a") as f:
        f.write("yer#3\n")
    assert tail.wait_screen("SoloPlayer", 1, since=mark) == "SoloPlayer"
    write_log(tmp_path, "100", ["📺 OsuScreenStack#473(depth:5) exit from SoloPlayer#3",
                                "📺 OsuScreenStack#473(depth:5) resume to PlayerLoader#2",
                                "📺 OsuScreenStack#473(depth:4) exit from PlayerLoader#2",
                                "📺 OsuScreenStack#473(depth:4) resume to SoloSongSelect#1"])
    tail.poll()
    assert tail.screen == "SoloSongSelect"
    assert ("exit from", "SoloPlayer") in tail.screens_since(mark)
    assert tail.wait_screen("MainMenu", 0.2, since=mark) is None
    assert tail.find(lazer.IMPORTED, mark) is None
    time.sleep(1.05)
    write_log(tmp_path, "200", ["📺 OsuScreenStack#9(depth:1) entered Loader#1"])  # osu! restarted
    assert tail.wait(lambda: tail.screen == "Loader", 2)
    write_log(tmp_path, "200", ["⚠️ Imported osu-ai playing A - B (C) [D]! Click to view."])
    m = tail.wait_for(lazer.IMPORTED, 1, since=mark)
    assert m and m.group(1) == "osu-ai" and m.group(2) == "A - B (C) [D]"


def test_display_title_matches_lazers_log(tmp_path):
    osu = tmp_path / "x.osu"
    osu.write_text("osu file format v14\n\n[Metadata]\nTitle:Cold Green Eyes ft. Roos Denayer\nArtist:Station Earth\n"
                   "Creator:Bearizm\nVersion:apple's Insane\nBeatmapID:827932\n\n[Difficulty]\nHPDrainRate:6\n")
    assert lazer.display_title(osu) == "Station Earth - Cold Green Eyes ft. Roos Denayer (Bearizm) [apple's Insane]"
    assert lazer.osu_metadata(osu)["BeatmapID"] == "827932"


def test_png_writer_round_trips(tmp_path):
    gray = (np.arange(48, dtype=np.uint8).reshape(6, 8) * 5)
    bgra = np.zeros((5, 7, 4), np.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2] = 10, 20, 200  # B, G, R
    for img, want in [(gray, np.repeat(gray[..., None], 3, axis=2)), (bgra, None)]:
        path = real.save_png(tmp_path / "x.png", img)
        got = pygame.surfarray.array3d(pygame.image.load(str(path))).transpose(1, 0, 2)
        if want is None:
            assert got.shape == (5, 7, 3) and (got == [200, 20, 10]).all()
        else:
            assert (got == want).all()


def toast_scene(rng, toast: bool, flash: float = 0.0, hovered: bool = False):
    """A BGRA main menu at 1 px per point: noisy background, a side flash, maybe a toast."""
    img = rng.integers(20, 200, (768, 1024, 4)).astype(np.uint8)
    img[..., 3] = 255
    img[:, 900:, :3] = np.clip(img[:, 900:, :3] + flash, 0, 255).astype(np.uint8)  # beat flash on the right edge
    if toast:
        x0, y0, x1, y1 = 1024 - 300, 40 + 23, 1024 - 20, 40 + 23 + 96
        img[y0:y1, x0:x1, :3] = (84, 69, 73) if hovered else (70, 57, 61)  # BGR of the notification purple
        text = rng.random((y1 - y0, x1 - x0)) < 0.15
        img[y0:y1, x0:x1][text] = (240, 240, 240, 255)
    return img


def scene(toast: bool, flash: float = 0.0, hovered: bool = False):
    return toast_scene(np.random.default_rng(1), toast, flash, hovered)


def test_find_toast_clicks_inside_the_new_notification():
    before = scene(False)
    for after in ([scene(True), scene(True)], [scene(True, 30), scene(True, 60)], [scene(True, hovered=True)] * 2):
        x, y = real.find_toast(before, after)
        assert 1024 - 300 + 40 < x < 1024 - 60 and 40 + 23 < y < 40 + 23 + 96  # on its text, off its close button
    assert real.find_toast(before, [scene(False, 40), scene(False, 40)]) is None  # a beat flash is no toast


@live
def test_driver_types_clicks_and_screenshots(selftest, tmp_path, monkeypatch):
    monkeypatch.setattr(lazer, "activate", lambda: None)  # never bring the real osu! forward from a test
    h = selftest
    absolute_mouse(tmp_path)
    write_log(tmp_path, "100", ["📺 OsuScreenStack#1(depth:3) entered SoloSongSelect#5",
                                "Game-wide working beatmap updated to "
                                "Station Earth - Cold Green Eyes ft. Roos Denayer (Bearizm) [apple's Insane]"])
    drv = real.Driver(h, lazer.LogTail(tmp_path), WIN, shots=tmp_path / "shots", log=lambda m: None)
    h.request("_fake_log")
    drv.press("p p")
    keys = [(e[0], e[1]) for e in h.request("_fake_log")["log"]]
    assert keys == [("down", 35), ("up", 35), ("down", 35), ("up", 35)]
    drv.press("ctrl+enter")
    assert [(e[0], e[1]) for e in h.request("_fake_log")["log"]] == [("down", 59), ("down", 36), ("up", 36), ("up", 59)]
    with pytest.raises(ValueError, match="toolbar"):
        drv.click(500, 20)
    drv.click(500, 400)
    log = h.request("_fake_log")["log"]
    assert [e[0] for e in log] == ["move", "mdown", "mup"] and log[1][2:4] == [244 + 500, 107 + 400]
    osu = tmp_path / "x.osu"
    osu.write_text("[Metadata]\nTitle:Cold Green Eyes ft. Roos Denayer\nArtist:Station Earth\nCreator:Bearizm\n"
                   "Version:apple's Insane\nBeatmapID:827932\n")
    assert drv.select(osu) == lazer.display_title(osu)
    typed = [e[1] for e in h.request("_fake_log")["log"] if e[0] == "down"]
    assert typed[-6:] == [hc.KEYCODES[c] for c in "827932"] and typed[0] == hc.KEYCODES["backspace"]
    shot = drv.shot("test")
    assert shot.exists() and shot.stat().st_size > 1000
    other = tmp_path / "y.osu"
    other.write_text("[Metadata]\nTitle:Other\nArtist:X\nCreator:Y\nVersion:Z\n")
    with pytest.raises(RuntimeError, match="did not select"):
        drv.select(other)
    assert list((tmp_path / "shots").glob("*failure.png"))


def test_hit_offsets_match_presses_to_circles():
    from osu_ai.beatmap import Kind
    from osu_ai.osr import Press

    objs = [SimpleNamespace(index=i, kind=Kind.CIRCLE, time=t, x=x, y=y, pos=(x, y))
            for i, (t, x, y) in enumerate(CIRCLES)]
    w = SimpleNamespace(great=34.5, ok=79.5, meh=124.5)
    bm = SimpleNamespace(objects=objs, difficulty=SimpleNamespace(radius=RADIUS, windows=w))
    presses = [Press(t + off, t + off + 40, x + 2, y - 1, i % 2) for i, ((t, x, y), off) in
               enumerate(zip(CIRCLES, [12, 18, 15, 60, 14]))]
    presses.append(Press(1405, 1440, 50, 50, 0))  # far from any circle: ignored
    offsets = real.hit_offsets(bm, presses)
    assert [o[0] for o in offsets] == [0, 1, 2, 3, 4] and [o[1] for o in offsets] == [12, 18, 15, 60, 14]
    rep = real.offset_report(bm, offsets, latency_ms=30)
    assert rep["offset_ms"]["median"] == 15 and rep["suggested_latency_ms"] == 45
    assert rep["within"] == {"great": 4, "ok": 5, "meh": 5} and rep["aim_osupx"]["mean_dx"] == 2


def test_keys_parse():
    assert hc.parse_keys("p p ctrl+enter F1 escape") == [(35,), (35,), (59, 36), (122,), (53,)]
    with pytest.raises(ValueError, match="unknown key"):
        hc.parse_keys("ctrl+nope")


@live
def test_locate_and_footage_against_the_selftest(selftest):
    win = real.locate(selftest, client=None)  # the selftest's osu! window, no Screen Recording: CoreGraphics only
    assert win.rect == (0, 32, 1512, 950)
    with pytest.raises(ValueError, match="WindowedSize"):
        real.locate(selftest, client=(1024, 768))
    shots = real.grab(selftest, WIN.rect, (1024, 768), n=2, gap_s=0.05)
    assert len(shots) == 2 and shots[0].shape == (768, 1024, 4)
    fp = real.Footage(selftest, WIN)
    t0 = time.monotonic()
    reason = fp.run(lambda: "enough" if time.monotonic() - t0 > 0.5 else None, max_s=5)
    fp.stop()
    frames_, ms = fp.arrays()
    assert reason == "enough" and frames_.shape[1:] == (240, 320) and len(frames_) >= 20
    assert (np.diff(ms) > 0).all() and abs(np.median(np.diff(ms)) - 1000 / 60) < 3


# the scripts' orchestration against a fake osu!: it watches what the selftest "posts" and writes
# the runtime-log lines the real game writes (docs/research/automation.md), so every menu step,
# wait and timeout of collect_real.py and play_real.py runs as it will live

TITLE = "Fake Artist - Fake Song (Mapper) [Hard]"
OSU_TEXT = """osu file format v14

[General]
AudioLeadIn: 0
Mode: 0

[Metadata]
Title:Fake Song
Artist:Fake Artist
Creator:Mapper
Version:Hard
BeatmapID:4242

[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:7
ApproachRate:9
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,500,4,2,0,100,1,0

[HitObjects]
""" + "".join(f"{x},{y},{t},{1 | (4 if i == 0 else 0)},0,0:0:0:0:\n" for i, (t, x, y) in
              enumerate([(600, 100, 100), (900, 300, 120), (1200, 420, 300), (1500, 200, 250)] * 1))


class FakeOsu(threading.Thread):
    def __init__(self, hands, data: Path, screen: str = "MainMenu", ctrl_works: bool = True, replay_s: float = 1.0,
                 play_s: float = 1.6, export: Path | None = None, fail: bool = False):
        super().__init__(daemon=True)
        self.hands, self.data, self.ctrl_works, self.replay_s, self.play_s, self.export = \
            hands, data, ctrl_works, replay_s, play_s, export
        self.fail, self.failed = fail, False
        self.screen, self.back = screen, []  # current screen, and the stack under it
        self.query, self.ctrl, self.mod_overlay, self.mod_text, self.autoplay = "", False, False, "", False
        self.paused, self.toast, self.p_count, self.esc_at_menu = False, False, 0, 0
        self.timers: list[tuple[float, callable]] = []
        self.inputs: list[list] = []
        self.stopped = threading.Event()
        self.log(f"📺 OsuScreenStack#1(depth:3) entered {screen}#100")

    def log(self, *lines: str) -> None:
        write_log(self.data, "100", list(lines))

    def push(self, name: str) -> None:
        self.back.append(self.screen)
        self.log(f"📺 OsuScreenStack#1(depth:{len(self.back) + 3}) suspended {self.screen}#1 (waiting on {name}#2)",
                 f"📺 OsuScreenStack#1(depth:{len(self.back) + 3}) entered {name}#2")
        self.screen = name

    def pop(self, to: str | None = None) -> None:
        while True:
            prev = self.back.pop()
            self.log(f"📺 OsuScreenStack#1(depth:{len(self.back) + 3}) exit from {self.screen}#2",
                     f"📺 OsuScreenStack#1(depth:{len(self.back) + 3}) resume to {prev}#1")
            self.screen = prev
            if to is None or prev == to:
                return

    def later(self, s: float, fn) -> None:
        self.timers.append((time.monotonic() + s, fn))

    def run(self) -> None:
        while not self.stopped.wait(0.02):
            now = time.monotonic()
            for due, fn in [t for t in self.timers if t[0] <= now]:
                self.timers.remove((due, fn))
                fn()
            try:
                events = self.hands.request("_fake_log")["log"]
            except hc.HandsError:
                return
            for e in events:
                self.inputs.append(e)
                self.on(e)

    def player_done(self, player: str) -> None:
        if self.screen != player:
            return
        if player == "SoloPlayer":  # offline, lazer says so on a pass and on a fail
            self.log("No token, skipping score submission")
            if self.fail:
                self.failed = True  # the fail screen: still SoloPlayer
                return
        self.later(0.8, lambda: self.push("SoloResultsScreen"))

    def start_player(self, player: str, after: float) -> None:
        self.push("PlayerLoader" if player == "SoloPlayer" or self.screen == "SoloSongSelect" else "ReplayPlayerLoader")
        self.later(0.3, lambda: self.push(player))
        self.later(0.3 + after, lambda: self.player_done(player))

    def write_export(self) -> None:
        if self.export is not None:
            dest = self.data / "exports" / "Guest playing fake.osr"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.export.read_bytes())

    def on(self, e: list) -> None:
        kind, code = e[0], e[1]
        if kind == "mdown":
            cx, cy = e[2] - WIN.left, e[3] - WIN.top
            if self.toast and self.screen == "MainMenu" and 1024 - 300 <= cx <= 1024 - 20 and 63 <= cy <= 63 + 90:
                self.toast = False
                self.log("Beginning PresentScore with score osu-ai playing " + TITLE)
                self.push("SoloResultsScreen")
            return
        if kind == "down" and code == 59:
            self.ctrl = True
        elif kind == "up" and code == 59:
            self.ctrl = False
        if kind != "down":
            return
        key = next(k for k, v in hc.KEYCODES.items() if v == code)
        if self.mod_overlay:
            if key == "f1":
                self.mod_overlay, self.mod_text = False, ""
            elif key == "enter" and self.mod_text == "at":
                self.autoplay = not self.autoplay
            elif len(key) == 1:
                self.mod_text += key
            return
        s = self.screen
        if s == "MainMenu":
            if key == "p":
                self.p_count += 1
                if self.p_count >= 2:
                    self.p_count = 0
                    self.push("SoloSongSelect")
            elif key == "esc":
                self.esc_at_menu += 1
        elif s == "SoloSongSelect":
            if key == "backspace":
                self.query = self.query[:-1]
            elif key == "space" or len(key) == 1:
                self.query += " " if key == "space" else key
                if self.query in ("4242", "fake song diff=hard"):
                    self.log(f"Game-wide working beatmap updated to {TITLE}")
            elif key == "f1":
                self.mod_overlay = True
            elif key == "esc":
                if self.query:
                    self.query = ""
                else:
                    self.pop()
            elif key == "enter":
                auto = self.autoplay or (self.ctrl and self.ctrl_works)
                self.start_player("ReplayPlayer" if auto else "SoloPlayer", self.play_s)
        elif s == "SoloResultsScreen":
            if key == "esc":
                self.pop(to="MainMenu" if "MainMenu" == self.back[-1] or "SoloSongSelect" not in self.back
                         else "SoloSongSelect")
            elif key == "f1":
                self.log("Beginning PresentScore with score osu-ai playing " + TITLE)
                self.pop(to="MainMenu")
                self.start_player("ReplayPlayer", self.replay_s)
            elif key == "f2":
                self.write_export()
        elif s == "ReplayPlayer" and key == "esc":
            self.pop(to="MainMenu" if "SoloSongSelect" not in self.back else "SoloSongSelect")
        elif s == "SoloPlayer" and self.failed:
            if key == "f2":
                self.write_export()
            elif key == "esc":  # on the fail screen Esc is Quit
                self.failed = False
                self.pop(to="SoloSongSelect")
        elif s == "SoloPlayer":
            if key == "esc":
                self.paused = True
            elif key == "enter" and self.paused:
                self.paused = False
                self.pop(to="SoloSongSelect")


@pytest.fixture
def fake_osu(selftest, tmp_path, monkeypatch):
    monkeypatch.setattr(lazer, "DATA", tmp_path / "osu")
    monkeypatch.setattr(lazer, "activate", lambda: None)
    absolute_mouse(tmp_path / "osu")
    osu = tmp_path / "fake.osu"
    osu.write_text(OSU_TEXT)
    stored = lazer.stored_path(osu)
    stored.parent.mkdir(parents=True)
    stored.write_bytes(osu.read_bytes())
    games: list[FakeOsu] = []

    def start(**kw) -> tuple[FakeOsu, real.Driver]:
        game = FakeOsu(selftest, tmp_path / "osu", **kw)
        games.append(game)
        game.start()
        drv = real.Driver(selftest, lazer.LogTail(), WIN, shots=tmp_path / "shots", log=print)
        return game, drv

    def imported(paths):
        from osu_ai import osr as osr_mod
        r = osr_mod.read(paths[0])  # lazer would refuse what osrparse cannot read
        assert r.game_version == 30000019 and r.beatmap_hash == lazer_md5(osu)
        games[-1].toast = True
        games[-1].log(f"⚠️ Imported {r.username} playing {TITLE}! Click to view.")

    monkeypatch.setattr(lazer, "import_files", imported)
    yield SimpleNamespace(start=start, osu=osu, hands=selftest, tmp=tmp_path)
    for g_ in games:
        g_.stopped.set()
        g_.join(2)


def lazer_md5(path: Path) -> str:
    import hashlib
    return hashlib.md5(path.read_bytes()).hexdigest()


def scripts():
    import sys
    sys.path.insert(0, str(hc.ROOT / "scripts"))
    import collect_real
    import play_real
    return collect_real, play_real


@live
def test_collect_replay_end_to_end(fake_osu, monkeypatch):
    collect_real, _ = scripts()
    monkeypatch.setattr(collect_real, "OUT", fake_osu.tmp / "real")
    monkeypatch.setattr(collect_real, "ROOT", fake_osu.tmp)
    game, drv = fake_osu.start()
    args = SimpleNamespace(fps=60.0, toast_offsets=list(collect_real.TOAST_OFFSETS))
    out = fake_osu.tmp / "real" / "replay" / "fake-s0.npz"
    saved = collect_real.record_replay(drv, fake_osu.osu, 0, out, args)
    from osu_ai.realdata import Recording
    rec = Recording.load(saved)
    assert rec.frames.shape[1:] == (240, 320) and len(rec.frames) > 30
    assert (np.diff(rec.pts_ms) > 0).all()
    m = rec.meta
    assert m["mode"] == "replay" and m["title"] == TITLE and m["beatmap_id"] == 4242 and m["replay_seed"] == 0
    assert m["stop"].startswith("lazer left ReplayPlayer") and m["window"] == [244, 107, 1024, 768]
    assert m["outcome"] and m["presses"] and m["toast_risk"] in (True, False)
    assert drv.tail.screen == "MainMenu" and game.esc_at_menu == 0  # never Esc on the main menu (exit dialog)
    keys = [e[1] for e in game.inputs if e[0] == "down"]
    assert hc.KEYCODES["f1"] in keys and hc.KEY_Z not in keys and hc.KEY_X not in keys


@live
def test_collect_replay_says_at_once_why_lazer_refused_the_replay(fake_osu, monkeypatch):
    collect_real, _ = scripts()
    monkeypatch.setattr(collect_real, "OUT", fake_osu.tmp / "real")
    monkeypatch.setattr(collect_real, "ROOT", fake_osu.tmp)
    _, drv = fake_osu.start()

    def refused(paths):  # lazer explains a failed import in its database log only
        name = Path(paths[0]).name
        why = f"Score '{name}' failed to import: no corresponding beatmap with the hash 'c0ffee' could be found."
        write_log(fake_osu.tmp / "osu", "100", [f"[?????] Beginning import from {name}...", why], kind="database")

    monkeypatch.setattr(lazer, "import_files", refused)
    args = SimpleNamespace(fps=60.0, toast_offsets=list(collect_real.TOAST_OFFSETS))
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="no corresponding beatmap"):
        collect_real.record_replay(drv, fake_osu.osu, 0, fake_osu.tmp / "real" / "replay" / "x.npz", args)
    assert time.monotonic() - t0 < 10


@live
def test_play_real_presses_nothing_into_a_play_that_is_still_running(fake_osu):
    _, play_real = scripts()
    game, drv = fake_osu.start(screen="SoloSongSelect")
    game.push("SoloPlayer")
    assert drv.tail.wait_screen("SoloPlayer", 2.0)
    mark = drv.tail.mark()
    assert play_real.export_replay(drv, "no frames for 60 s", mark) is None
    downs = [e[1] for e in game.inputs if e[0] == "down"]
    assert hc.KEYCODES["f2"] not in downs and downs == [hc.KEYCODES[k] for k in ("esc", "up", "enter")]
    assert drv.tail.screen == "SoloSongSelect"


@live
def test_collect_autoplay_falls_back_to_the_mod_when_ctrl_does_not_register(fake_osu, monkeypatch):
    collect_real, _ = scripts()
    monkeypatch.setattr(collect_real, "ROOT", fake_osu.tmp)
    game, drv = fake_osu.start(ctrl_works=False)
    state = {"via": "ctrl", "mod_on": False}
    out = fake_osu.tmp / "real" / "autoplay" / "fake.npz"
    saved = collect_real.record_autoplay(drv, fake_osu.osu, out, SimpleNamespace(fps=60.0), state)
    assert state == {"via": "mods", "mod_on": True} and game.autoplay
    from osu_ai.realdata import Recording
    rec = Recording.load(saved)
    assert rec.meta["mode"] == "autoplay" and rec.meta["autoplay_via"] == "mods" and len(rec.frames) > 30
    assert drv.tail.screen == "SoloSongSelect" and game.esc_at_menu == 0
    # the unplayed SoloPlayer was left through the pause menu: Esc, Up (Quit), Enter
    downs = [e[1] for e in game.inputs if e[0] == "down"]
    i = downs.index(hc.KEYCODES["esc"])
    assert downs[i:i + 3] == [hc.KEYCODES["esc"], hc.KEYCODES["up"], hc.KEYCODES["enter"]]


@live
@pytest.mark.parametrize("fail", [False, True], ids=["pass", "fail"])
def test_play_real_end_to_end(fake_osu, fail):
    _, play_real = scripts()
    from osu_ai import osr as osr_mod
    from osu_ai.beatmap import parse

    bm = parse(fake_osu.osu)
    # what lazer will export: presses 12 ms late on every circle
    frames_ = [(o.time - 300, o.x, o.y, 0) for o in bm.objects[:1]]
    for o in bm.objects:
        frames_ += [(o.time + 12, o.x + 1, o.y, 1), (o.time + 60, o.x + 1, o.y, 0)]
    export = fake_osu.tmp / "lazer-export.osr"
    export.write_bytes(osr_mod.encode(bm.md5, frames_, "Guest", (4, 0, 0, 0, 0, 0), 1000000, 4, True))
    game, drv = fake_osu.start(screen="MainMenu", play_s=2.0, export=export, fail=fail)
    due = {}

    class Clocked(Blind):  # sees one circle at the first object's position, due 600 ms after the run starts
        def see(self, frame):
            g_ = run_holder["game"]
            shown = g_.frame.display_ns or g_.frame.t_ns
            due.setdefault("ns", shown + 600_000_000)
            return [(100.0, 100.0, (due["ns"] - shown) / 1e6, 1.0)] if shown < due["ns"] else [], self.modes

    run_holder = {}
    real_game = real.RealGame

    def spy(*a, **kw):
        run_holder["game"] = real_game(*a, **kw)
        return run_holder["game"]

    args = play_real.parser().parse_args([str(fake_osu.osu), "--latency-ms", "30"])
    import unittest.mock as mock
    with mock.patch.object(real, "RealGame", spy):
        report = play_real.play(drv, Clocked(), fake_osu.osu, args, out=fake_osu.tmp / "play")
    assert report["run"]["reason"] == "play over: No token, skipping score submission"
    assert report["run"]["presses"] == {"sent": 1}
    assert report["lazer"]["count_300"] == 4 and report["lazer"]["mods"] == 0
    o = report["offsets"]
    assert o["matched"] == 4 and o["offset_ms"]["median"] == 12 and o["suggested_latency_ms"] == 42
    assert list((fake_osu.tmp / "play").glob("*.npz")) and list((fake_osu.tmp / "play").glob("*.json"))
    zx = [e for e in game.inputs if e[0] == "down" and e[1] in (hc.KEY_Z, hc.KEY_X)]
    assert len(zx) == 1 and abs(zx[0][4] - (due["ns"] - 30e6)) < 3e6
    assert game.esc_at_menu == 0
    assert drv.tail.screen == ("SoloSongSelect" if fail else "SoloResultsScreen")  # a fail screen is quit with Esc


@live
def test_lazer_shot_moves_clicks_types_and_saves_a_png(selftest, tmp_path, monkeypatch, capsys):
    import sys
    sys.path.insert(0, str(hc.ROOT / "scripts"))
    import lazer_shot

    monkeypatch.setattr(lazer, "DATA", tmp_path / "osu")
    write_log(tmp_path / "osu", "100", ["📺 OsuScreenStack#1(depth:3) entered MainMenu#2"])
    absolute_mouse(tmp_path / "osu")
    monkeypatch.setattr(lazer, "activate", lambda: None)
    allowed = {"screen_capture": True, "accessibility": True, "post_event": True}
    monkeypatch.setattr(selftest, "permissions", lambda: allowed)
    monkeypatch.setattr(selftest, "osu_window", lambda source="sck", **kw: {"frame": [244, 79, 1024, 796]})
    monkeypatch.setattr(selftest, "close", lambda: None)  # the fixture closes it
    monkeypatch.setattr(lazer_shot.Hands, "launch", classmethod(lambda cls, **kw: selftest))
    (tmp_path / "osu" / "framework.ini").write_text("WindowMode = Windowed\nWindowedSize = 1024x768\n")
    selftest.request("_fake_log")
    out = tmp_path / "shot.png"
    monkeypatch.setattr(sys, "argv", ["lazer_shot.py", "--out", str(out), "--move", "100,600", "--click", "500,300",
                                      "--keys", "p p", "--wait-ms", "10"])
    lazer_shot.main()
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["png"] == str(out) and report["screen"] == "MainMenu" and report["png_px_per_pt"] == 2
    assert report["client_rect_pt"] == [244, 107, 1024, 768]
    got = pygame.image.load(str(out))
    assert got.get_size() == (2048, 1536)
    log = selftest.request("_fake_log")["log"]
    assert [e[0] for e in log] == ["move", "move", "mdown", "mup", "down", "up", "down", "up"]
    assert log[0][2:4] == [244 + 100, 107 + 600] and log[2][2:4] == [244 + 500, 107 + 300]
    monkeypatch.setattr(sys, "argv", ["lazer_shot.py", "--click", "500,20"])
    with pytest.raises(ValueError, match="toolbar"):
        lazer_shot.main()
    monkeypatch.setattr(sys, "argv", ["lazer_shot.py", "--keys", "ctrl+nope"])
    with pytest.raises(ValueError, match="unknown key"):
        lazer_shot.main()
