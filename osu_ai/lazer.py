"""Drive the real osu!lazer client on this Mac: launch it, import files, and follow its screens.

The game is launched through Launch Services with updates and crash reporting turned off,
so its version (and our calibration) can't drift. Which screen it is on comes from its own
runtime log ("... OsuScreenStack#473(depth:3) entered SoloPlayer#123"), never from pixels:
the current screen is the top of the OsuScreenStack, the last screen it entered or resumed
to (the background and overlay stacks log too, and are ignored). Offline only: nothing here
logs in, and a logged-out client never submits scores.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path

DATA = Path.home() / "Library/Application Support/osu"
APP = "/Applications/osu!.app"
BUNDLE = "sh.ppy.osu.lazer"
ENV = {"OSU_EXTERNAL_UPDATE_PROVIDER": "1", "OSU_DISABLE_ERROR_REPORTING": "1", "OSU_WEBSOCKET_SERVER": "1"}
_SCREEN = re.compile(r"OsuScreenStack#\d+\(depth:\d+\) entered (\w+)#")
_STACK = re.compile(r"OsuScreenStack#\d+\(depth:(\d+)\) (entered|resume to|exit from|suspended) (\w+#|\[empty\])")
_BEATMAP = re.compile(r"Game-wide working beatmap updated to (.+)$")
IMPORTED = re.compile(r"⚠️ Imported (.+?) playing (.+)! Click to view\.")
# A failed import says why only in the database log (the runtime log just stops at "... is initialising").
IMPORT_FAILED = re.compile(r"failed to import|Failed to parse headers|Could not import")
# A solo play is over (passed, failed or quit): logged offline, where nothing is ever submitted.
PLAY_OVER = re.compile(r"No token, skipping score submission|Score submission cancelled|No hits registered, skipping"
                       r"|Zero score, skipping")
PLAYERS = ("SoloPlayer", "ReplayPlayer")


def running() -> bool:
    return subprocess.run(["pgrep", "-f", "osu!.app/Contents/MacOS"], capture_output=True).returncode == 0


def launch(wait_s: float = 60.0) -> None:
    """Start osu! (if needed) and wait until it reaches the main menu."""
    if running():
        activate()
        return
    log_before = set(DATA.glob("logs/*.runtime.log"))
    env = [a for k, v in ENV.items() for a in ("--env", f"{k}={v}")]
    subprocess.run(["open", *env, "-a", APP], check=True)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        new = set(DATA.glob("logs/*.runtime.log")) - log_before
        if new and "MainMenu" in screens(max(new)):
            activate()
            return
        time.sleep(0.5)
    raise TimeoutError("osu! did not reach the main menu")


def activate() -> None:
    """Bring a running osu! to the front. Never starts it: AppleScript would, without launch()'s environment."""
    if running():
        subprocess.run(["osascript", "-e", f'tell application id "{BUNDLE}" to activate'], capture_output=True)


def quit(wait_s: float = 20.0) -> None:
    """Ask twice: the first request opens the exit dialog, the second confirms it."""
    for _ in range(2):
        subprocess.run(["osascript", "-e", 'quit app "osu!"'], capture_output=True)
        time.sleep(1.2)
    deadline = time.monotonic() + wait_s
    while running() and time.monotonic() < deadline:
        time.sleep(0.5)


def import_files(paths: list[Path]) -> None:
    """Import .osz/.osk/.osr files. osu! deletes imported archives, so it gets copies."""
    staging = Path(tempfile.mkdtemp(prefix="osu-import-"))
    copies = []
    for p in paths:
        copies.append(staging / Path(p).name)
        shutil.copy(p, copies[-1])
    subprocess.run(["open", "-a", APP, *map(str, copies)], check=True)


def runtime_log() -> Path:
    return max(DATA.glob("logs/*.runtime.log"), key=lambda p: p.stat().st_mtime)


def screens(log: Path | None = None) -> list[str]:
    """Every screen the game entered this session, in order."""
    text = (log or runtime_log()).read_text(errors="replace")
    return _SCREEN.findall(text)


def screen() -> str | None:
    """The screen the game is on now."""
    return LogTail().screen


def wait_for_screen(name: str, timeout_s: float = 30.0, after: int | None = None) -> bool:
    """Wait until the game enters `name` (after the first `after` screen entries, if given)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        found = screens()
        if name in found[after or 0:]:
            return True
        time.sleep(0.1)
    return False


def selected_beatmap() -> str | None:
    found = [m.group(1) for line in runtime_log().read_text(errors="replace").splitlines()
             if (m := _BEATMAP.search(line))]
    return found[-1] if found else None


def settings(name: str = "game.ini", data: Path | None = None) -> dict[str, str]:
    """One of osu!'s ini files as a dict. Read only: osu! rewrites them whenever it exits."""
    path = (data or DATA) / name
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.lstrip().startswith("#"):
                out[key.strip()] = value.strip()
    return out


def absolute_mouse(data: Path | None = None) -> bool:
    """Whether osu! reads the cursor as absolute window positions, as our posted moves need: input.json
    must turn MouseHandler.UseRelativeMode off. Without it this build runs in relative mode on macOS
    (docs/research/cross-check.md §1e), and SDL then warps and holds the real cursor."""
    try:
        handlers = json.loads(((data or DATA) / "input.json").read_text())["InputHandlers"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    mouse = [h for h in handlers if isinstance(h, dict) and ".Mouse.MouseHandler," in str(h.get("$type"))]
    return bool(mouse) and all(h.get("UseRelativeMode") is False for h in mouse)


def windowed_size(data: Path | None = None) -> tuple[int, int] | None:
    """The client area in points (framework.ini WindowedSize) if osu! runs Windowed, else None."""
    fw = settings("framework.ini", data)
    w, _, h = fw.get("WindowedSize", "").partition("x")
    if fw.get("WindowMode") != "Windowed" or not (w.isdigit() and h.isdigit()):
        return None
    return int(w), int(h)


_VISUAL = ("Skin", "DimLevel", "BlurLevel", "ShowStoryboard", "BeatmapSkins", "BeatmapColours", "HitLighting",
           "StarFountains", "HUDVisibilityMode", "GameplayCursorSize", "AutoCursorSize", "LightenDuringBreaks",
           "FadePlayfieldWhenHealthLow", "IncreaseFirstObjectVisibility", "Scaling", "UIScale",
           "ComboColourNormalisationAmount", "ReplaySettingsOverlay", "GameplayLeaderboard", "KeyOverlay", "AudioOffset")


def visual_settings(data: Path | None = None) -> dict[str, str]:
    """The settings that change what captured frames look like (and nothing about the account)."""
    game, fw = settings("game.ini", data), settings("framework.ini", data)
    out = {k: game[k] for k in _VISUAL if k in game}
    out.update({f"framework.{k}": fw[k] for k in ("WindowMode", "WindowedSize", "FrameSync") if k in fw})
    return out


def osu_section(path: Path, name: str) -> dict[str, str]:
    """A key: value section of a .osu file ("General", "Metadata", ...)."""
    out: dict[str, str] = {}
    section = None
    for raw in Path(path).read_bytes().decode("utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif section == name and ":" in line:
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out


def osu_metadata(path: Path) -> dict[str, str]:
    return osu_section(path, "Metadata")


def player_length_s(bm, osu_path: Path) -> float:
    """How long lazer's player runs for this map (a beatmap.Beatmap): from min(0, first object - 2000,
    first object - AudioLeadIn) to the end of the last object (MasterGameplayClockContainer)."""
    first = bm.objects[0].time
    lead_in = float(osu_section(osu_path, "General").get("AudioLeadIn") or 0)
    start = min(0.0, first - 2000, first - lead_in if lead_in > 0 else 0.0)
    return (max(o.end_time for o in bm.objects) - start) / 1000


def display_title(path: Path) -> str:
    """How lazer names this difficulty in its log: "Artist - Title (Creator) [Version]"."""
    m = osu_metadata(path)
    creator = f" ({m['Creator']})" if m.get("Creator") else ""
    version = f" [{m['Version']}]" if m.get("Version") else ""
    return f"{m.get('Artist') or 'unknown artist'} - {m.get('Title') or 'unknown title'}{creator}{version}".strip()


def stored_path(osu_file: Path, data: Path | None = None) -> Path:
    """Where lazer keeps this exact file once its set is imported (files/ is keyed by SHA-256)."""
    h = hashlib.sha256(Path(osu_file).read_bytes()).hexdigest()
    return (data or DATA) / "files" / h[0] / h[:2] / h


def is_imported(osu_file: Path, data: Path | None = None) -> bool:
    return stored_path(osu_file, data).exists()


def exports(since: float = 0.0, data: Path | None = None) -> list[Path]:
    """Replays exported with F2 (to exports/), modified at or after `since` (time.time()), oldest first."""
    found = [p for p in ((data or DATA) / "exports").glob("*.osr") if p.stat().st_mtime >= since]
    return sorted(found, key=lambda p: p.stat().st_mtime)


class LogTail:
    """Follows osu!'s newest runtime log (or another kind: "database", ...) as it grows, and the
    next one when osu! restarts.

    Every complete line is kept, so callers take a mark() before acting and then wait for
    what the game logs after it. `screen` is the top of the OsuScreenStack.
    """

    def __init__(self, data: Path | None = None, from_start: bool = True, kind: str = "runtime"):
        self.data = data or DATA
        self.kind = kind
        self.lines: list[str] = []
        self.log_start = 0  # index in lines where the current log (osu! process) begins
        self.path: Path | None = None
        self.screen: str | None = None
        self.beatmap: str | None = None  # the game-wide working beatmap, as lazer names it
        self._pos = 0
        self._partial = b""
        self._next_check = 0.0
        self.poll()
        if not from_start:
            self.lines.clear()
            self.log_start = 0

    def _newest(self) -> Path | None:
        logs = list((self.data / "logs").glob(f"*.{self.kind}.log"))
        return max(logs, key=lambda p: (p.stat().st_mtime, p.name)) if logs else None

    def poll(self) -> int:
        """Read what was appended since the last poll; returns the number of new lines."""
        now = time.monotonic()
        if self.path is None or now >= self._next_check:
            self._next_check = now + 1.0
            newest = self._newest()
            if newest is not None and newest != self.path:
                self.path, self._pos, self._partial = newest, 0, b""
                self.screen, self.log_start = None, len(self.lines)  # a new log is a new osu! process
        if self.path is None:
            return 0
        try:
            with self.path.open("rb") as f:
                f.seek(self._pos)
                chunk = f.read()
        except FileNotFoundError:
            self.path = None
            return 0
        self._pos += len(chunk)
        *done, self._partial = (self._partial + chunk).split(b"\n")
        for raw in done:
            line = raw.decode("utf-8", errors="replace").rstrip("\r")
            self.lines.append(line)
            if m := _STACK.search(line):
                depth, event, name = int(m.group(1)), m.group(2), m.group(3).rstrip("#")
                if event in ("entered", "resume to"):
                    self.screen = None if name == "[empty]" else name
                elif event == "exit from" and depth == 0:
                    self.screen = None  # the game is shutting down
            elif m := _BEATMAP.search(line):
                self.beatmap = m.group(1)
        return len(done)

    def mark(self) -> int:
        self.poll()
        return len(self.lines)

    def since(self, mark: int) -> list[str]:
        self.poll()
        return self.lines[mark:]

    def find(self, pattern: re.Pattern | str, since: int = 0) -> re.Match | None:
        """The first match of `pattern` in a line logged after `since`."""
        rx = re.compile(pattern) if isinstance(pattern, str) else pattern
        return next((m for line in self.since(since) if (m := rx.search(line))), None)

    def screens_since(self, since: int) -> list[tuple[str, str]]:
        """(event, screen) for every OsuScreenStack change after `since`: entered, resume to, exit from, suspended."""
        return [(m.group(2), m.group(3).rstrip("#")) for line in self.since(since) if (m := _STACK.search(line))]

    def wait(self, check: Callable[[], object], timeout: float, every: float = 0.05):
        """Poll until check() returns something truthy (returned), or None after `timeout` s."""
        deadline = time.monotonic() + timeout
        while True:
            self.poll()
            if result := check():
                return result
            if time.monotonic() >= deadline:
                return None
            time.sleep(every)

    def wait_for(self, pattern: re.Pattern | str, timeout: float, since: int = 0) -> re.Match | None:
        return self.wait(lambda: self.find(pattern, since), timeout)

    def wait_screen(self, names: str | Iterable[str], timeout: float, since: int | None = None) -> str | None:
        """Wait until the current screen is one of `names`; with `since`, it must also have been
        entered or resumed to after that mark. Returns the screen, or None on timeout."""
        names = {names} if isinstance(names, str) else set(names)

        def check():
            if self.screen not in names:
                return None
            if since is not None and not any(e in ("entered", "resume to") and s == self.screen
                                             for e, s in self.screens_since(since)):
                return None
            return self.screen

        return self.wait(check, timeout)
