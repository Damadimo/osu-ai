"""Record labeled footage of the real osu!lazer: lazer plays replays (or Autoplay) while OsuHands captures.

  uv run scripts/collect_real.py replay                        # each imported map in data/lazer_osu/, seed 0
  uv run scripts/collect_real.py replay --targets --seeds 0 1  # the maps in data/real_maps.json, two replays each
  uv run scripts/collect_real.py replay --maps data/lazer_osu/827932.osu
  uv run scripts/collect_real.py autoplay --targets            # lazer's Autoplay instead of our replays

replay mode, per map and seed: synth_replay.human_replay (timing and aim noise, some misses) ->
.osr (osr.export, judged by our lazer-rules judge for the header and the labels) -> import into
lazer -> click the "Imported ... Click to view" toast (found by comparing screenshots, else at a
fixed offset from the window's top-right) -> results screen -> F1 plays the replay -> capture the
640x480 virtual screen at 60 fps until lazer leaves the replay (or the map is over) -> Esc.
autoplay mode: song select, select the map (checked against osu!'s log), Ctrl+Enter (if Ctrl does
not register, the Autoplay mod through mod select), capture until the results screen.

Each run is saved as a realdata.Recording (320x240 frames, display time in ms on the perf_counter
clock, meta) in runs/real/<mode>/. Runs already on disk are skipped, so stopping and starting
again resumes. A step that times out stops with a screenshot in runs/shots/ (or, with
--keep-going, moves on to the next map). osu! must be running, Windowed (scripts/lazer_setup.py),
logged out, and in a menu. Replays are imported locally and never submitted anywhere.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

from osu_ai import beatmap, judge, lazer, osr, real
from osu_ai.hands_client import Hands, HandsError, now_ns
from osu_ai.synth_replay import human_replay

ROOT = Path(__file__).resolve().parent.parent
MAPS = ROOT / "data" / "lazer_osu"
TARGETS = ROOT / "data" / "real_maps.json"
OUT = real.RUNS / "real"
USERNAME = "osu-ai"
TOAST_OFFSETS = ((-160.0, 93.0), (-160.0, 53.0))  # from the client's top-right: toolbar shown, toolbar hidden


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def target_ids() -> dict[str, int]:
    """md5 -> BeatmapID for the maps in data/real_maps.json (some .osu files carry no BeatmapID)."""
    return {m["osu_md5"]: m["beatmap_id"] for m in json.loads(TARGETS.read_text())} if TARGETS.exists() else {}


def pick_maps(args) -> list[Path]:
    if args.maps:
        maps = [Path(p) for p in args.maps]
    else:
        maps = sorted(MAPS.glob("*.osu"))
        if args.targets:
            want = set(target_ids())
            maps = [p for p in maps if md5(p) in want]
    out = []
    for p in maps:
        if not lazer.is_imported(p):
            print(f"skip {p.name}: its set is not imported in lazer")
            continue
        try:
            bm = beatmap.parse(p)
        except ValueError as e:
            print(f"skip {p.name}: {e}")
            continue
        if len(bm.objects) < args.min_objects:
            print(f"skip {p.name}: {len(bm.objects)} objects")
            continue
        out.append(p)
    return out


def first_gameplay(tail: lazer.LogTail, mark: int) -> bool:
    """No player screen before `mark` in this osu! process: with the HUD hidden, lazer shows a
    one-time "score overlay disabled" toast over the top-right of the playfield during the first."""
    return not any("OsuScreenStack" in line and f"entered {p}#" in line
                   for line in tail.lines[tail.log_start:mark] for p in lazer.PLAYERS)


def base_meta(mode: str, path: Path, bm: beatmap.Beatmap, drv: real.Driver, fps: float) -> dict:
    g = drv.geometry
    return {
        "mode": mode, "map": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path), "md5": bm.md5,
        "beatmap_id": bm.beatmap_id or target_ids().get(bm.md5), "title": lazer.display_title(path),
        "osu_format": general_version(path), "window": list(g.rect), "capture_rect": list(g.capture_rect),
        "capture_size": list(real.CAPTURE), "size": list(real.RECORD), "fps": fps, "lazer": lazer.visual_settings(),
        "clock": "pts_ms: display time of each frame, ms on the perf_counter (mach) clock; "
                 "game time = pts_ms + c, find c with realdata.align",
    }


def general_version(path: Path) -> int:
    first = path.read_bytes().decode("utf-8-sig", errors="replace").lstrip().split("\n", 1)[0]
    return int(first.rsplit("v", 1)[1]) if first.startswith("osu file format v") else 14


def capture_play(drv: real.Driver, footage: real.Footage, player: str, budget_s: float) -> tuple[str, float]:
    """Record from the moment `player` was entered until lazer leaves it (or the budget runs out),
    then get back out. Returns (why recording stopped, perf_counter ms when the player was seen)."""
    tail = drv.tail
    entered_ms = now_ns() / 1e6
    mark = tail.mark()

    def left() -> str | None:
        changes = tail.screens_since(mark)
        return f"lazer left {player}: {changes[0][0]} {changes[0][1]}" if changes else None

    reason = footage.run(left, max_s=budget_s)
    footage.stop()
    if tail.screen in lazer.PLAYERS:
        drv.leave_player()
    return reason, entered_ms


def record_replay(drv: real.Driver, path: Path, seed: int, out: Path, args) -> Path:
    tail, hands, g = drv.tail, drv.hands, drv.geometry
    bm = beatmap.parse(path)
    replay = human_replay(bm, random.Random(seed))
    score = judge.score(bm, replay)
    osr_path = OUT / "replays" / f"{path.stem}-s{seed}-{time.strftime('%Y%m%d-%H%M%S')}.osr"
    osr.export(bm, replay, score, osr_path, USERNAME)

    drv.main_menu()  # the toast is clicked over the main menu, where that spot is empty
    drv.park()
    client = (g.rect, (int(g.width), int(g.height)))
    before = real.grab(hands, *client)[0]
    db = lazer.LogTail(kind="database", from_start=False)  # where lazer says why an import failed
    mark = tail.mark()
    lazer.import_files([osr_path])

    def outcome():
        return tail.find(lazer.IMPORTED, mark) or next(
            (line for line in db.since(0) if osr_path.name in line and lazer.IMPORT_FAILED.search(line)), None)

    imported = tail.wait(outcome, 30.0)
    if not isinstance(imported, re.Match):
        drv.fail(f"lazer did not import {osr_path.name}: {imported or 'nothing in its logs within 30 s'}")
    if imported.group(2) != lazer.display_title(path):
        drv.fail(f"lazer imported the replay for {imported.group(2)!r}, expected {lazer.display_title(path)!r}")
    time.sleep(1.0)  # the toast slides in and flashes
    after = real.grab(hands, *client, n=2, gap_s=0.15)
    found = real.find_toast(before, after)
    spots = ([found] if found else []) + [(g.width + dx, dy) for dx, dy in args.toast_offsets]
    first_click = tail.mark()

    def opened():  # lazer logs "Beginning PresentScore" as soon as the toast is clicked
        return tail.find("Beginning PresentScore", first_click) or tail.wait_screen("SoloResultsScreen", 0, first_click)

    for cx, cy in spots:
        drv.click(cx, cy)
        if tail.wait(opened, 2.5):
            break
        drv.log(f"no score opened after clicking ({cx:.0f}, {cy:.0f})")
    if not tail.wait(opened, 5.0):
        drv.fail("clicking the import toast did not open the score (is it still on screen?)")
    drv.wait_screen("SoloResultsScreen", 15.0, since=first_click, what="opening the imported score")
    time.sleep(1.5)

    drv.park()  # the replay loader only starts with the cursor over it, or idle
    footage = real.Footage(hands, g, args.fps)
    footage.start()
    mark = tail.mark()
    drv.press("f1")
    drv.wait_screen("ReplayPlayer", 30.0, since=mark, what="starting the replay (F1 on the results screen)")
    toast_risk = first_gameplay(tail, mark) and lazer.visual_settings().get("HUDVisibilityMode") == "Never"
    reason, entered_ms = capture_play(drv, footage, "ReplayPlayer", lazer.player_length_s(bm, path) + 15)
    drv.main_menu()

    judged = {str(e.index): [e.t, e.result != "Miss"] for e in score.events
              if e.kind in ("HitCircle", "SliderHeadCircle")}
    meta = base_meta("replay", path, bm, drv, args.fps) | {
        "replay": str(osr_path.relative_to(ROOT)), "replay_seed": seed, "replay_username": USERNAME,
        "presses": [[round(p.down, 3), round(p.up, 3)] for p in replay.presses],
        "outcome": judged,  # our judge's view of the replay: object index -> [judged at ms, hit]
        "sim": {"accuracy": score.accuracy, "counts": {str(k): v for k, v in score.counts.items()},
                "max_combo": score.max_combo, "rank": getattr(score, "rank", None)},
        "player_entered_ms": entered_ms, "stop": reason, "dropped": footage.dropped, "toast_risk": toast_risk,
    }
    return footage.save(out, meta)


def record_autoplay(drv: real.Driver, path: Path, out: Path, args, state: dict) -> Path:
    tail, hands, g = drv.tail, drv.hands, drv.geometry
    bm = beatmap.parse(path)
    for _ in range(2):
        drv.select(path, target_ids().get(bm.md5))  # (again after leaving a player: the selection must be ours)
        footage = real.Footage(hands, g, args.fps)
        footage.start()
        mark = tail.mark()
        player = drv.start(autoplay="ctrl" if state["via"] == "ctrl" and not state["mod_on"] else None)
        if player == "ReplayPlayer":
            break
        footage.stop()  # Ctrl did not register: this is a real play with nobody playing it
        drv.leave_player()
        drv.song_select()
        if state["mod_on"]:
            drv.fail("Autoplay is selected but the map started without it")
        drv.log("Ctrl+Enter started a normal play; switching to the Autoplay mod")
        drv.toggle_autoplay()
        state.update(via="mods", mod_on=True)
    else:
        drv.fail("could not start the map with Autoplay")
    toast_risk = first_gameplay(tail, mark) and lazer.visual_settings().get("HUDVisibilityMode") == "Never"
    reason, entered_ms = capture_play(drv, footage, "ReplayPlayer", lazer.player_length_s(bm, path) + 15)
    drv.song_select()
    meta = base_meta("autoplay", path, bm, drv, args.fps) | {
        "autoplay_via": state["via"], "player_entered_ms": entered_ms, "stop": reason, "dropped": footage.dropped,
        "toast_risk": toast_risk,
    }
    return footage.save(out, meta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["replay", "autoplay"])
    ap.add_argument("--maps", nargs="*", help=".osu files (default: every imported map in data/lazer_osu/)")
    ap.add_argument("--targets", action="store_true", help="only the maps listed in data/real_maps.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="replay mode: one replay per seed and map")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--min-objects", type=int, default=30)
    ap.add_argument("--toast-offsets", type=lambda s: tuple(map(float, s.split(","))), nargs="+",
                    default=list(TOAST_OFFSETS), help="fallback toast click points dx,dy from the client's top-right")
    ap.add_argument("--keep-going", action="store_true", help="after a failure, go back to the menu and continue")
    args = ap.parse_args()

    maps = pick_maps(args)
    seeds = args.seeds if args.mode == "replay" else [0]
    todo = [(m, s, OUT / args.mode / (f"{m.stem}-s{s}.npz" if args.mode == "replay" else f"{m.stem}.npz"))
            for m in maps for s in seeds]
    todo = [t for t in todo if not t[2].exists()]
    print(f"{len(todo)} recordings to make ({len(maps)} maps); already done ones are skipped")
    if not todo:
        return
    tail = lazer.LogTail()
    if tail.screen is None:
        sys.exit("osu! is not running: start it (uv run python -c 'from osu_ai import lazer; lazer.launch()')")
    if tail.screen in lazer.PLAYERS:
        sys.exit(f"osu! is in {tail.screen}: get it to a menu first")
    with Hands.launch() as hands:
        perms = hands.permissions()
        if not all(perms.values()):
            sys.exit(f"OsuHands permissions {perms}: uv run python -m osu_ai.hands_client grant")
        try:
            drv = real.Driver(hands, tail)
        except (RuntimeError, ValueError, HandsError) as e:
            sys.exit(str(e))
        drv.front()
        state = {"via": "ctrl", "mod_on": False}
        failures = 0
        try:
            for path, seed, out in todo:
                t0 = time.monotonic()
                try:
                    if args.mode == "replay":
                        saved = record_replay(drv, path, seed, out, args)
                    else:
                        saved = record_autoplay(drv, path, out, args, state)
                except (RuntimeError, HandsError, TimeoutError) as e:
                    failures += 1
                    print(f"FAILED {path.name} seed {seed}: {e}")
                    hands.release_all()
                    if not args.keep_going:
                        sys.exit(1)
                    if drv.tail.screen in lazer.PLAYERS:
                        drv.leave_player()
                    drv.main_menu()
                    continue
                print(f"{saved.relative_to(ROOT)}: {time.monotonic() - t0:.0f} s")
        finally:
            with contextlib.suppress(HandsError, OSError):
                hands.release_all()
                if state["mod_on"] and drv.tail.screen == "SoloSongSelect":
                    drv.toggle_autoplay()  # leave song select as we found it
        print(f"done, {failures} failed")


if __name__ == "__main__":
    main()
