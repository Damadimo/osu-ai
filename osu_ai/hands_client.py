"""Client for OsuHands.app: ScreenCaptureKit frames in, synthetic mouse moves and Z/X keys out.

OsuHands (hands_app/) holds the macOS permissions (Screen Recording, Accessibility), so
this process needs none. It speaks the binary protocol in hands_app/Sources/Common/Wire.swift
over run/hands.sock. Every timestamp is on the time.perf_counter_ns() clock, which on macOS
is mach_absolute_time in ns: frame times and scheduled input share the agent's clock with no
offset. Positions are global screen points (top-left of the main display, y down). Input only
reaches the game while osu! is the frontmost app, and held keys are released on disconnect.
"""

from __future__ import annotations

import itertools
import json
import socket
import struct
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "hands_app" / "OsuHands.app"
SOCKET = ROOT / "run" / "hands.sock"
OSU_BUNDLE = "sh.ppy.osu.lazer"
KEY_Z, KEY_X = 6, 7  # macOS virtual keycodes kVK_ANSI_Z, kVK_ANSI_X

JSON, INPUT, FRAME = 1, 2, 3  # message types
MOVE, KEY_DOWN, KEY_UP, CANCEL, RELEASE = 1, 2, 3, 4, 5  # input record kinds
MOUSE_DOWN, MOUSE_UP = 6, 7  # key field = button: 0 left, 1 right
ALL_TAGS = 0xFFFFFFFF
LUMA, BGRA = 1, 4
IDLE = 1  # frame flag: screen unchanged, no pixels sent

_HEAD = struct.Struct("<IB")  # length, type
_FRAME = struct.Struct("<BHIQqqqqHHI")  # rest of the 56-byte frame header
_RECORD = struct.Struct("<BxHIqdd")  # one 32-byte input record

now_ns = time.perf_counter_ns

# macOS virtual keycodes (Carbon HIToolbox Events.h, ANSI positions) for driving menus with keys().
KEYCODES = {
    **dict(zip("asdfhgzxcv", range(10))), "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28, "0": 29,
    "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42,
    ",": 43, "/": 44, "n": 45, "m": 46, ".": 47, "`": 50,
    "enter": 36, "tab": 48, "space": 49, "backspace": 51, "esc": 53, "cmd": 55, "shift": 56, "alt": 58, "ctrl": 59,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100, "f9": 101, "f10": 109,
    "f11": 103, "f12": 111, "home": 115, "pageup": 116, "end": 119, "pagedown": 121,
    "left": 123, "right": 124, "down": 125, "up": 126,
}
_KEY_ALIASES = {"return": "enter", "escape": "esc", "delete": "backspace", "control": "ctrl", "option": "alt",
                "command": "cmd"}


def parse_keys(spec: str) -> list[tuple[int, ...]]:
    """"p p p" or "ctrl+enter" -> one tuple of keycodes per space-separated chord (modifiers first)."""
    chords = []
    for token in spec.split():
        names = [_KEY_ALIASES.get(n, n) for n in token.lower().split("+")]
        unknown = [n for n in names if n not in KEYCODES]
        if unknown or not all(names):
            raise ValueError(f"unknown key {unknown or token!r} in {spec!r}; known: {' '.join(KEYCODES)}")
        chords.append(tuple(KEYCODES[n] for n in names))
    return chords


class HandsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Frame:
    seq: int  # capture counter; a gap means the helper replaced frames we had not read yet
    t_ns: int  # presentation time ScreenCaptureKit stamped on the frame
    display_ns: int  # when the WindowServer displayed it (0 if unknown)
    recv_ns: int  # when the helper received it
    arrived_ns: int  # when this client finished reading it
    pixels: np.ndarray  # (H, W) uint8 luma, or (H, W, 4) BGRA
    config: int  # capture config id; it changes with every start/update_capture
    idle: bool  # screen unchanged: pixels are the previous frame's
    dropped: int  # frames replaced before sending, total for this capture

    @property
    def age_ms(self) -> float:
        return (now_ns() - self.t_ns) / 1e6


@dataclass(frozen=True)
class Capture:
    """Maps frame pixels to global screen points for one capture config.

    Pixel coordinates are continuous: pixel (i, j) covers [i, i+1) x [j, j+1).
    """

    rect: tuple[float, float, float, float]  # captured area in global points (x, y, w, h)
    size: tuple[int, int]  # frame width, height in pixels
    config: int

    def to_points(self, px: float, py: float) -> tuple[float, float]:
        x, y, w, h = self.rect
        return x + px * w / self.size[0], y + py * h / self.size[1]

    def to_pixels(self, x: float, y: float) -> tuple[float, float]:
        rx, ry, w, h = self.rect
        return (x - rx) * self.size[0] / w, (y - ry) * self.size[1] / h


@dataclass(frozen=True)
class Event:
    """One input action. at_ns is a perf_counter_ns() time; 0 means now."""

    kind: int
    at_ns: int = 0
    key: int = 0
    x: float = 0.0
    y: float = 0.0
    tag: int = 0  # cancel(tag) drops every pending event with this tag

    def pack(self) -> bytes:
        return _RECORD.pack(self.kind, self.key, self.tag, self.at_ns, self.x, self.y)


class Hands:
    """One connection to OsuHands. Thread-safe; a background thread reads frames and replies."""

    def __init__(self, path: Path | str = SOCKET, ring: int = 8, timeout: float = 5.0):
        self.path = Path(path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._sock.connect(str(self.path))
        except OSError:
            self._sock.close()
            raise
        self._send_lock = threading.Lock()
        self._cond = threading.Condition()
        self._ids = itertools.count(1)
        self._replies: dict[int, dict] = {}
        self._latest: Frame | None = None
        self._ring: deque[Frame] = deque(maxlen=max(ring, 1))
        self.capture: Capture | None = None
        self.hello: dict | None = None
        self.frontmost: dict | None = None  # last focus event: {"bundle_id", "allowed", ...}
        self.events: deque[dict] = deque(maxlen=100)
        self.closed: str | None = None
        self._reader = threading.Thread(target=self._read_loop, name="hands-reader", daemon=True)
        self._reader.start()
        with self._cond:
            self._cond.wait_for(lambda: self.hello is not None or self.closed is not None, timeout)
        if self.hello is None:
            self.close()
            raise HandsError(f"no hello from OsuHands on {self.path}")

    @classmethod
    def launch(cls, app: Path | str = APP, path: Path | str = SOCKET, wait_s: float = 10.0, **kw) -> Hands:
        """Connect, starting OsuHands first if needed.

        `open` makes the app its own responsible process, so macOS asks for (and remembers)
        permissions as com.osujev.hands rather than as this terminal; -g keeps osu! in front.
        """
        try:
            return cls(path, **kw)
        except (FileNotFoundError, ConnectionRefusedError):
            pass
        if not Path(app).exists():
            raise HandsError(f"{app} is missing: run hands_app/build.sh")
        cmd = ["open", "-g", str(app)]
        if Path(path) != SOCKET:
            cmd += ["--args", "--socket", str(path)]
        subprocess.run(cmd, check=True)
        deadline = time.monotonic() + wait_s
        while True:
            try:
                return cls(path, **kw)
            except (FileNotFoundError, ConnectionRefusedError) as e:
                if time.monotonic() > deadline:
                    raise HandsError(f"OsuHands did not start: see {Path(path).parent / 'hands.log'}") from e
                time.sleep(0.05)

    def close(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
        if self._reader.is_alive() and self._reader is not threading.current_thread():
            self._reader.join(1.0)

    def __enter__(self) -> Hands:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # control

    def request(self, cmd: str, timeout: float = 10.0, **args) -> dict:
        rid = next(self._ids)
        self._send(JSON, json.dumps({"id": rid, "cmd": cmd, **args}).encode())
        with self._cond:
            self._cond.wait_for(lambda: rid in self._replies or self.closed is not None, timeout)
            reply = self._replies.pop(rid, None)
        if reply is None:
            raise HandsError(f"{cmd}: {self.closed or f'no reply in {timeout} s'}")
        if not reply.get("ok"):
            raise HandsError(f"{cmd}: {reply.get('error')}")
        return reply

    def ping(self) -> dict:
        """Round trip, and a clock check: offset_ns should be within rtt_ns / 2 of zero."""
        t0 = now_ns()
        reply = self.request("ping")
        t1 = now_ns()
        return {**reply, "rtt_ns": t1 - t0, "offset_ns": reply["now_ns"] - (t0 + t1) // 2}

    def status(self) -> dict:
        return self.request("status")

    def permissions(self) -> dict:
        """{"screen_capture": bool, "accessibility": bool, "post_event": bool}"""
        return self.status()["permissions"]

    def request_permissions(self, screen: bool = True, accessibility: bool = True) -> dict:
        """Show the macOS prompts (only the first time per app identity; then use open_settings)."""
        return self.request("request_permissions", screen=screen, accessibility=accessibility)

    def open_settings(self, pane: str = "screen") -> None:
        """Open System Settings at Privacy > Screen Recording ("screen") or Accessibility."""
        self.request("open_settings", pane=pane)

    def overlay(self, show: bool | None = None) -> bool:
        """The cursor and Z/X keys OsuHands draws above osu! for people watching (on by default,
        shown while osu! is frontmost). Capture leaves it out, so the model never sees it."""
        return self.request("overlay", **({} if show is None else {"show": show}))["show"]

    def displays(self) -> list[dict]:
        return self.request("displays")["displays"]

    def windows(self, source: str = "sck", all: bool = False) -> list[dict]:
        """On-screen windows. "sck" needs Screen Recording; "cg" does not (but has no titles)."""
        return self.request("windows", timeout=30.0, source=source, all=all)["windows"]

    def osu_window(self, source: str = "sck", bundle: str = OSU_BUNDLE) -> dict | None:
        """osu!'s main window: the largest normal-layer window it owns."""
        wins = [w for w in self.windows(source) if w["bundle_id"] == bundle and w["layer"] == 0]
        return max(wins, key=lambda w: w["frame"][2] * w["frame"][3], default=None)

    def configure(self, target_bundle: str | None = None, clamp: tuple | None | bool = False) -> dict:
        """Set the app that must be frontmost for input, and/or a rect [x, y, w, h] that cursor moves
        are clamped into (None removes it). Clamp to osu!'s window to keep off hot corners."""
        args: dict = {}
        if target_bundle is not None:
            args["target_bundle"] = target_bundle
        if clamp is not False:
            args["clamp"] = None if clamp is None else list(clamp)
        return self.request("config", **args)

    def stats(self, reset: bool = False) -> dict:
        return self.request("stats", reset=reset)

    def quit(self) -> None:
        try:
            self.request("quit", timeout=2.0)
        finally:
            self.close()

    # capture

    def start_capture(self, rect: tuple[float, float, float, float], size: tuple[int, int] = (640, 480),
                      fps: float = 120.0, fmt: str = "luma", window: int | None = None,
                      queue_depth: int = 3, resolution: str = "auto") -> Capture:
        """Stream `rect` (global points) scaled to `size` pixels. Keep rect and size the same aspect
        ratio: the image is stretched, never letterboxed. fps=0 means every display refresh.
        window: capture only that window (ignores overlays such as notifications)."""
        reply = self.request("capture_start", timeout=30.0, rect=list(rect), size=list(size), fps=fps, format=fmt,
                             window=window, queue_depth=queue_depth, resolution=resolution)
        return self._set_capture(reply)

    def update_capture(self, **args) -> Capture:
        """Change rect/size/fps/... of the running capture (format/window/display changes restart it).
        Frames from before the change can still arrive: check frame.config."""
        if "rect" in args:
            args["rect"] = list(args["rect"])
        if "size" in args:
            args["size"] = list(args["size"])
        return self._set_capture(self.request("capture_update", timeout=30.0, **args))

    def stop_capture(self) -> None:
        self.request("capture_stop", timeout=30.0)
        self.capture = None

    def _set_capture(self, reply: dict) -> Capture:
        cfg = reply["config"]
        self.capture = Capture(tuple(reply["rect"]), tuple(cfg["size"]), reply["config_id"])
        return self.capture

    def latest(self) -> Frame | None:
        return self._latest

    def recent(self) -> list[Frame]:
        """The last `ring` frames, oldest first."""
        with self._cond:
            return list(self._ring)

    def next_frame(self, after: int | None = None, timeout: float = 1.0) -> Frame | None:
        """Block until a frame with seq > `after` (default: newer than the latest) arrives."""
        with self._cond:
            if after is None:
                after = self._latest.seq if self._latest else -1
            self._cond.wait_for(lambda: (self._latest is not None and self._latest.seq > after)
                                or self.closed is not None, timeout)
            frame = self._latest
        return frame if frame is not None and frame.seq > after else None

    # input

    def send(self, *events: Event) -> None:
        """Send events in one message. They run in time order; equal times in this order."""
        if events:
            self._send(INPUT, b"".join(e.pack() for e in events))

    def move(self, x: float, y: float, at_ns: int = 0, tag: int = 0) -> None:
        self.send(Event(MOVE, at_ns, x=x, y=y, tag=tag))

    def key_down(self, key: int = KEY_Z, at_ns: int = 0, tag: int = 0) -> None:
        self.send(Event(KEY_DOWN, at_ns, key, tag=tag))

    def key_up(self, key: int = KEY_Z, at_ns: int = 0, tag: int = 0) -> None:
        self.send(Event(KEY_UP, at_ns, key, tag=tag))

    def tap(self, key: int = KEY_Z, at_ns: int = 0, hold_ms: float = 30.0, tag: int = 0) -> None:
        at_ns = at_ns or now_ns()
        self.send(Event(KEY_DOWN, at_ns, key, tag=tag), Event(KEY_UP, at_ns + int(hold_ms * 1e6), key, tag=tag))

    def click(self, x: float, y: float, button: int = 0, at_ns: int = 0, hold_ms: float = 60.0) -> None:
        """Move there, then press and release a mouse button (for menus, not gameplay)."""
        at_ns = at_ns or now_ns()
        self.send(Event(MOVE, at_ns, x=x, y=y), Event(MOUSE_DOWN, at_ns + 30_000_000, button, x=x, y=y),
                  Event(MOUSE_UP, at_ns + 30_000_000 + int(hold_ms * 1e6), button, x=x, y=y))

    def keys(self, spec: str, at_ns: int = 0, gap_ms: float = 300.0, hold_ms: float = 50.0) -> int:
        """Press keys for menus: "p p p", "8 2 7 9 3 2", "ctrl+enter". Chords are gap_ms apart; within
        one, modifiers go down first and up last. Returns the perf_counter_ns time the last key is up.
        Key-downs only reach the game while osu! is frontmost (the helper drops them otherwise)."""
        step, hold, gap = 20_000_000, int(hold_ms * 1e6), int(gap_ms * 1e6)
        t = at_ns or now_ns()
        events = []
        for i, (*mods, key) in enumerate(parse_keys(spec)):
            if i:
                t += gap
            for m in mods:
                events.append(Event(KEY_DOWN, t, m))
                t += step
            events += [Event(KEY_DOWN, t, key), Event(KEY_UP, t + hold, key)]
            t += hold
            for m in reversed(mods):
                t += step
                events.append(Event(KEY_UP, t, m))
        self.send(*events)
        return t

    def cancel(self, tag: int = ALL_TAGS) -> None:
        """Drop pending (not yet executed) events with this tag; ALL_TAGS drops everything."""
        self.send(Event(CANCEL, tag=tag))

    def release_all(self, at_ns: int = 0) -> None:
        self.send(Event(RELEASE, at_ns))

    # wire

    def _send(self, kind: int, payload: bytes) -> None:
        if self.closed is not None:
            raise HandsError(self.closed)
        with self._send_lock:
            self._sock.sendall(_HEAD.pack(len(payload) + 1, kind) + payload)

    def _fill(self, buf) -> None:
        view = memoryview(buf).cast("B")
        while view:
            n = self._sock.recv_into(view)
            if not n:
                raise ConnectionError("OsuHands closed the connection")
            view = view[n:]

    def _read_loop(self) -> None:
        head, fhead = bytearray(_HEAD.size), bytearray(_FRAME.size)
        try:
            while True:
                self._fill(head)
                n, kind = _HEAD.unpack(head)
                if kind == FRAME:
                    self._fill(fhead)
                    self._on_frame(*_FRAME.unpack(fhead), size=n - 1 - _FRAME.size)
                else:
                    body = bytearray(n - 1)
                    self._fill(body)
                    if kind == JSON:
                        self._on_json(json.loads(body))
        except Exception as e:  # noqa: BLE001 - any failure ends the connection
            reason = f"OsuHands connection lost: {e!r}"
        with self._cond:
            self.closed = reason
            self._cond.notify_all()

    def _on_frame(self, fmt, flags, config, seq, t, shown, recv, sent, w, h, dropped, size: int) -> None:
        idle = bool(flags & IDLE)
        if idle:
            prev = self._latest
            if prev is None or prev.config != config:  # an idle frame repeats this capture's pixels, not another's
                return
            pixels = prev.pixels
        else:
            bpp = 4 if fmt == BGRA else 1
            if size != w * h * bpp:
                raise ValueError(f"frame of {size} bytes for {w}x{h}x{bpp}")
            pixels = np.empty(size, np.uint8)
            self._fill(pixels)
            pixels = pixels.reshape((h, w, 4) if bpp == 4 else (h, w))
        frame = Frame(seq, t, shown, recv, now_ns(), pixels, config, idle, dropped)
        with self._cond:
            self._latest = frame
            self._ring.append(frame)
            self._cond.notify_all()

    def _on_json(self, msg: dict) -> None:
        with self._cond:
            if "id" in msg:
                self._replies[msg["id"]] = msg
            else:
                self.events.append(msg)
                if msg.get("event") == "hello":
                    self.hello = msg
                    self.frontmost = msg.get("frontmost")
                elif msg.get("event") == "frontmost":
                    self.frontmost = msg
            self._cond.notify_all()


def main() -> None:
    """python -m osu_ai.hands_client [status | grant | windows | probe]

    status: launch OsuHands if needed, print permissions, focus and the clock check.
    grant: show the macOS prompts for what is missing and open System Settings at that pane
      (after enabling Screen Recording, let System Settings quit and reopen OsuHands).
    windows: list on-screen windows.
    probe: capture osu!'s window (or the main display) for 3 s and print frame pacing and
      latency. Never moves the mouse or presses keys.
    """
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    with Hands.launch() as h:
        perms = h.permissions()
        if cmd == "grant":
            missing = [k for k in ("screen_capture", "accessibility") if not perms[k]]
            if missing:
                h.request_permissions(screen="screen_capture" in missing, accessibility="accessibility" in missing)
                h.open_settings("screen" if "screen_capture" in missing else "accessibility")
            perms = h.permissions()
        elif cmd == "windows":
            for w in h.windows("sck" if perms["screen_capture"] else "cg"):
                print(f"{w['id']:>6} {w['bundle_id'] or w['app']:<32} {w['frame']} {w['title']!r}")
        elif cmd == "probe":
            win = h.osu_window()
            x, y, w, hh = win["frame"] if win else next(d for d in h.displays() if d["main"])["frame"]
            size = (640, 2 * round(320 * hh / w))  # same aspect as the window
            h.start_capture((x, y, w, hh), size=size, fps=120)
            frames, end = [], time.monotonic() + 3
            while time.monotonic() < end:
                if (f := h.next_frame(timeout=1.0)) is not None:
                    frames.append(f)
            h.stop_capture()
            full = [f for f in frames if not f.idle]
            if len(full) < 2:
                print(f"only {len(full)} frames: is the screen changing, and is Screen Recording allowed?")
                return
            gap = np.diff([f.t_ns for f in full]) / 1e6
            lag = np.array([(f.recv_ns - (f.display_ns or f.t_ns)) / 1e6 for f in full])
            age = np.array([(f.arrived_ns - (f.display_ns or f.t_ns)) / 1e6 for f in full])
            print(f"{'osu! window' if win else 'main display'} {[x, y, w, hh]} -> {size}: {len(full)} frames, "
                  f"{len(frames) - len(full)} idle, {full[-1].dropped} dropped in 3 s")
            print(f"frame interval ms p50 {np.median(gap):.2f} p95 {np.percentile(gap, 95):.2f}; "
                  f"display->helper ms p50 {np.median(lag):.2f} p95 {np.percentile(lag, 95):.2f}; "
                  f"display->python ms p50 {np.median(age):.2f}; pts-display ms {(full[-1].t_ns - full[-1].display_ns) / 1e6:.2f}")
            out = SOCKET.parent / "probe.npy"
            np.save(out, full[-1].pixels)
            print(f"last frame saved to {out}")
        p = h.ping()
        print(f"OsuHands pid {h.hello['pid']}: {perms}; frontmost {h.frontmost}")
        print(f"ping {p['rtt_ns'] / 1e3:.0f} us, clock offset {p['offset_ns'] / 1e3:.1f} us (same clock if |offset| < ping / 2)")


if __name__ == "__main__":
    main()
