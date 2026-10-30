"""The OsuHands protocol, end to end, against hands_selftest.

hands_selftest is the real socket server and real-time input scheduler with fake frames and
a dry-run input poster: it never reads the screen or posts events, so no permissions are
needed. Build it with hands_app/build.sh.
"""

from __future__ import annotations

import os
import subprocess
import time

import numpy as np
import pytest

from osu_ai import hands_client as hc

SELFTEST = hc.ROOT / "hands_app" / "build" / "hands_selftest"

pytestmark = pytest.mark.skipif(not SELFTEST.exists(), reason="run hands_app/build.sh first")


def connect(path, timeout: float = 5.0) -> hc.Hands:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return hc.Hands(path)
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


@pytest.fixture
def server():
    path = hc.ROOT / "run" / f"selftest-{os.getpid()}.sock"
    path.parent.mkdir(exist_ok=True)
    proc = subprocess.Popen([str(SELFTEST), "--socket", str(path)])
    try:
        yield path
    finally:
        proc.terminate()
        proc.wait(5)
        path.unlink(missing_ok=True)


@pytest.fixture
def hands(server):
    h = connect(server)
    yield h
    h.close()


def posted(h: hc.Hands, after_s: float = 0.0) -> list:
    time.sleep(after_s)
    return h.request("_fake_log")["log"]


def test_same_clock_as_perf_counter(hands):
    p = hands.ping()
    assert abs(p["offset_ns"]) <= p["rtt_ns"]
    assert hands.hello["clock"] == "mach_absolute_time"
    assert hands.status()["frontmost"]["allowed"]


def test_frames_arrive_with_pattern_timestamps_and_mapping(hands):
    cap = hands.start_capture((100, 50, 512, 384), size=(64, 48), fps=120)
    assert cap.size == (64, 48) and cap.rect == (100, 50, 512, 384)
    assert cap.to_points(*cap.to_pixels(300.0, 200.0)) == pytest.approx((300.0, 200.0))
    first = hands.next_frame(timeout=2)
    time.sleep(0.25)
    last = hands.latest()
    assert first is not None and last.seq > first.seq + 10
    f = last
    assert f.pixels.shape == (48, 64) and f.pixels.dtype == np.uint8 and not f.idle
    flat = f.pixels.reshape(-1)
    assert int.from_bytes(flat[:8].tobytes(), "little") == f.seq
    y, x = np.mgrid[0:48, 0:64]
    assert np.array_equal(f.pixels.reshape(-1)[8:], ((x + 2 * y + f.seq) & 255).astype(np.uint8).reshape(-1)[8:])
    assert f.t_ns < f.recv_ns <= f.arrived_ns <= hc.now_ns()
    assert 3.5 < (f.recv_ns - f.t_ns) / 1e6 < 4.5
    assert f.config == cap.config
    hands.stop_capture()


def test_bgra_and_idle_frames(hands):
    hands.request("_fake_idle", every=3)
    hands.start_capture((0, 0, 320, 240), size=(32, 24), fmt="bgra", fps=120)
    time.sleep(0.2)
    frames = hands.recent()
    assert any(f.idle for f in frames) and any(not f.idle for f in frames)
    for f in frames:
        assert f.pixels.shape == (24, 32, 4)
    idle = next(f for f in frames if f.idle)
    before = [f for f in frames if f.seq < idle.seq]
    if before:
        assert idle.pixels is before[-1].pixels


def test_update_capture_changes_config(hands):
    cap = hands.start_capture((0, 0, 400, 300), size=(40, 30))
    cap2 = hands.update_capture(size=(80, 60))
    assert cap2.config != cap.config and cap2.size == (80, 60)
    deadline = time.monotonic() + 2
    while (f := hands.next_frame(timeout=1)) is not None and f.config != cap2.config:
        assert time.monotonic() < deadline
    assert f.pixels.shape == (60, 80)


def test_scheduled_input_runs_on_time_and_in_order(hands):
    posted(hands)
    t0 = hc.now_ns() + 30_000_000
    hands.send(hc.Event(hc.MOVE, t0, x=10, y=20),
               hc.Event(hc.KEY_DOWN, t0 + 5_000_000, hc.KEY_Z),
               hc.Event(hc.KEY_UP, t0 + 25_000_000, hc.KEY_Z),
               hc.Event(hc.MOVE, t0 + 5_000_000, x=11, y=21))  # same time as KEY_DOWN, sent after it
    log = posted(hands, 0.1)
    assert [e[0] for e in log] == ["move", "down", "move", "up"]
    targets = [t0, t0 + 5_000_000, t0 + 5_000_000, t0 + 25_000_000]
    late_us = [(e[4] - t) / 1000 for e, t in zip(log, targets)]
    assert all(0 <= late < 2000 for late in late_us), late_us
    stats = hands.stats()["input"]
    assert stats["executed"] >= 4 and stats["timed"] >= 4


def test_cancel_by_tag(hands):
    posted(hands)
    t = hc.now_ns() + 150_000_000
    hands.tap(hc.KEY_X, t, tag=7)
    hands.move(5, 5, t, tag=8)
    hands.cancel(7)
    log = posted(hands, 0.25)
    assert [e[0] for e in log] == ["move"]


def test_input_needs_osu_frontmost_and_focus_loss_releases_keys(hands):
    posted(hands)
    hands.key_down(hc.KEY_Z)
    time.sleep(0.02)
    hands.request("_fake_front", bundle="com.apple.finder")
    time.sleep(0.05)
    hands.key_down(hc.KEY_X)
    hands.move(1, 1)
    log = posted(hands, 0.05)
    assert [(e[0], e[1]) for e in log] == [("down", hc.KEY_Z), ("up", hc.KEY_Z)]
    assert hands.frontmost["allowed"] is False
    assert hands.stats()["input"]["blocked"] >= 2
    hands.request("_fake_front", bundle=hc.OSU_BUNDLE)
    hands.tap(hc.KEY_X)
    assert [e[0] for e in posted(hands, 0.1)] == ["down", "up"]


def test_focus_loss_cancels_pending_input(hands):
    posted(hands)
    hands.tap(hc.KEY_X, hc.now_ns() + 150_000_000)
    hands.request("_fake_front", bundle="com.apple.finder")
    hands.request("_fake_front", bundle=hc.OSU_BUNDLE)
    assert posted(hands, 0.3) == []


def test_disconnect_releases_held_keys(server):
    h = connect(server)
    posted(h)
    h.key_down(hc.KEY_Z)
    h.key_down(hc.KEY_X, hc.now_ns() + 500_000_000)  # pending, must be cancelled
    time.sleep(0.05)
    h.close()
    time.sleep(0.1)
    h2 = connect(server)
    log = posted(h2, 0.6)
    assert [(e[0], e[1]) for e in log] == [("down", hc.KEY_Z), ("up", hc.KEY_Z)]
    h2.close()


def test_new_client_replaces_old(server):
    a = connect(server)
    b = connect(server)
    deadline = time.monotonic() + 2
    while a.closed is None:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert b.ping()["ok"]
    b.close()


def test_errors_are_reported(hands):
    with pytest.raises(hc.HandsError, match="unknown command"):
        hands.request("nope")
    with pytest.raises(hc.HandsError, match="size"):
        hands.start_capture((0, 0, 100, 100), size=(15, 15))
    with pytest.raises(hc.HandsError, match="not running"):
        hands.update_capture(fps=60)
    assert hands.configure(clamp=(0, 0, 100, 100))["clamp"] == [0, 0, 100, 100]
    assert hands.configure(clamp=None)["clamp"] is None
