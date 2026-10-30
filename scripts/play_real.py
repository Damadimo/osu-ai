"""Let the model play one map in the real osu!lazer, then read back how lazer judged it.

  uv run --extra train scripts/play_real.py data/lazer_osu/827932.osu
  uv run --extra train scripts/play_real.py data/lazer_osu/827932.osu --latency-ms 42
  uv run --extra train scripts/play_real.py data/lazer_osu/827932.osu --dry runs/real/play/<run>.npz

From the main menu or song select: select the map (checked against osu!'s log), press Enter with
the cursor parked over the loader, wait for SoloPlayer and its 750 ms intro zoom, then RealGame
plays until lazer leaves the player (results) or the map is over (a fail leaves the fail screen
up). F2 then exports lazer's own replay of the run: its key-downs against the beatmap give the hit
offsets, whose median is the correction to add to --latency-ms, and its header holds lazer's
judgement. The run (frames, detections, events sent) and the report go to runs/real/play/.

--dry replays the frames of a saved run (or a realdata recording) through the model and policy
offline, with no osu! and no OsuHands, and prints what would have been sent.
Offline only: osu! stays logged out and nothing is submitted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from osu_ai import beatmap, lazer, osr, real
from osu_ai.geometry import OsuWindow
from osu_ai.hands_client import KEY_DOWN, Hands, HandsError
from osu_ai.policy import Policy

ROOT = Path(__file__).resolve().parent.parent
OUT = real.RUNS / "real" / "play"
TARGETS = ROOT / "data" / "real_maps.json"


def target_id(bm: beatmap.Beatmap) -> int | None:
    if bm.beatmap_id or not TARGETS.exists():
        return bm.beatmap_id
    return next((m["beatmap_id"] for m in json.loads(TARGETS.read_text()) if m["osu_md5"] == bm.md5), None)


def load_agent(args):
    from osu_ai.agent import LazerAgent  # MLX: needs --extra train

    agent = LazerAgent(str(args.weights), width=args.width, input_size=args.input)
    blank = np.zeros((args.input[1], args.input[0]), np.uint8)
    t0 = time.perf_counter()
    for _ in range(3):  # the first calls compile kernels: never let that happen mid-map
        agent.see(blank)
    agent.reset()
    print(f"agent {args.weights} at {args.input[0]}x{args.input[1]} ready in {time.perf_counter() - t0:.1f} s")
    return agent


def export_replay(drv: real.Driver, reason: str, mark: int) -> Path | None:
    """F2 on the results screen (or the fail screen) writes lazer's replay to exports/. Nothing is
    pressed while the play may still be running (the run stopped for another reason): then the
    player is left through the pause menu instead."""
    tail = drv.tail
    if "focus" in reason:
        drv.log(f"not exporting a replay: the run ended with {reason!r} (osu! is paused)")
        return None

    def over():
        return tail.find(lazer.PLAY_OVER, mark)

    if tail.screen == "SoloPlayer" and tail.wait(over, 2.0):
        tail.wait(lambda: tail.screen != "SoloPlayer", 6.0)  # a pass goes on to results in a moment; a fail stays
    if tail.screen == "SoloPlayer" and not over():
        drv.log(f"not exporting a replay: the play is still running (run ended with {reason!r}); leaving it")
        drv.leave_player()
        return None
    if tail.screen not in ("SoloResultsScreen", "SoloPlayer"):
        drv.log(f"not exporting a replay: screen {tail.screen}, run ended with {reason!r}")
        return None
    time.sleep(1.5)  # let the screen settle (results animate in; the fail screen fades in)
    since = time.time() - 1
    drv.press("f2")
    path = tail.wait(lambda: (lazer.exports(since) or [None])[-1], 15.0, every=0.25)
    sizes, deadline = [-1], time.monotonic() + 10
    while path is not None and time.monotonic() < deadline:  # lazer may still be writing it
        time.sleep(0.3)
        sizes.append(path.stat().st_size)
        if sizes[-1] > 0 and sizes[-1] == sizes[-2]:
            break
    if tail.screen == "SoloPlayer":
        drv.leave_player()  # on the fail screen, Esc is Quit
    if path is None:
        drv.log("F2 exported nothing (a play that was neither passed nor failed has no replay)")
    return path


def lazer_result(path: Path) -> dict:
    r = osr.read(path)
    info = osr.score_info(path) or {}
    return {"replay": str(path), "count_300": r.count_300, "count_100": r.count_100, "count_50": r.count_50,
            "count_miss": r.count_miss, "max_combo": r.max_combo, "score": r.score, "mods": int(r.mods),
            **{k: info[k] for k in ("statistics", "maximum_statistics", "rank", "accuracy", "total_score")
               if k in info}}


def dry(args, bm: beatmap.Beatmap) -> None:
    agent = load_agent(args)
    hands = real.DryHands()
    game = real.RealGame(hands, agent, Policy(lead_ms=args.latency_ms + args.slack_ms), OsuWindow(244, 107, 1024, 768),
                         args.latency_ms, fps=args.fps, early_guard_ms=bm.difficulty.windows.meh - 15,
                         hover_radius=bm.difficulty.radius + 4, log=print)
    reason = game.run(frames=real.recorded_frames(args.dry))
    downs = [s for s in game.sent if s.kind == KEY_DOWN]
    print(json.dumps({"reason": reason, **game.summary()}, indent=1))
    for s in downs[:20]:
        print(f"press {real.STATUS[s.status]:<14} at frame ms {s.t_ms:9.1f} -> osu!px ({s.ox:6.1f}, {s.oy:6.1f})")


def play(drv: real.Driver, agent, path: Path, args, out: Path = OUT) -> dict:
    """Select, start, play and judge one map; returns the report (also written next to the run)."""
    tail = drv.tail
    bm = beatmap.parse(path)
    guard = args.guard_ms if args.guard_ms is not None else bm.difficulty.windows.meh - 15
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{path.stem}"
    drv.front()
    for _ in range(2):
        print(f"selecting {drv.select(path, target_id(bm))}")
        player = drv.start()
        if player == "SoloPlayer":
            break
        drv.log(f"the map started in {player}: Autoplay is selected; turning it off")
        drv.leave_player()
        drv.song_select()
        drv.toggle_autoplay()
    else:
        drv.fail("the map keeps starting with Autoplay")
    mark = tail.mark()
    time.sleep(0.75)  # intro zoom: the playfield grows from 0.7 to 1 over 750 ms

    def stop() -> str | None:
        """The play is over once lazer leaves the player, or logs that it is not submitting the
        score (offline it says so on a pass, a fail and a quit; a fail stays on SoloPlayer)."""
        if changes := tail.screens_since(mark):
            return f"lazer left the player: {changes[0][0]} {changes[0][1]}"
        if over := tail.find(lazer.PLAY_OVER, mark):
            return f"play over: {over.group(0)}"
        return None

    policy = Policy(lead_ms=args.latency_ms + args.slack_ms)
    game = real.RealGame(drv.hands, agent, policy, drv.geometry, args.latency_ms, fps=args.fps,
                         early_guard_ms=guard, hover_radius=bm.difficulty.radius + 4, stop=stop, log=drv.log)
    meta = {"map": str(path), "md5": bm.md5, "title": lazer.display_title(path), "weights": str(args.weights),
            "slack_ms": args.slack_ms, "lazer": lazer.visual_settings()}
    reason = "not started"
    try:
        reason = game.run(max_s=lazer.player_length_s(bm, path) + 30)  # normally lazer's log ends it first
    finally:
        saved = game.save(out / f"{stamp}.npz", {**meta, "stop": reason})
        print(f"run saved to {saved}")
    report = {"map": meta["title"], "latency_ms": args.latency_ms, "run": game.summary()}
    print(json.dumps(report["run"], indent=1))
    exported = export_replay(drv, reason, mark)
    if exported is not None:
        report["lazer"] = lazer_result(exported)
        offsets = real.hit_offsets(bm, osr.presses(osr.read(exported), osr.time_offset(bm)))
        report["offsets"] = real.offset_report(bm, offsets, args.latency_ms)
        report["per_object"] = [[i, round(dt, 1), round(dx, 1), round(dy, 1)] for i, dt, dx, dy in offsets]
    (out / f"{stamp}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_object", "run")}, indent=1))
    if report.get("offsets", {}).get("matched"):
        o = report["offsets"]
        print(f"hit offsets: median {o['offset_ms']['median']:+.1f} ms over {o['matched']} objects -> "
              f"next time try --latency-ms {o['suggested_latency_ms']:g}")
    return report


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", type=Path)
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "osunet-lazer.safetensors")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--input", type=lambda s: tuple(int(v) for v in s.split("x")), default=(320, 240))
    ap.add_argument("--latency-ms", type=float, default=30.0,
                    help="L: from a frame being shown to lazer judging input sent then; events leave L early")
    ap.add_argument("--slack-ms", type=float, default=30.0,
                    help="the policy decides latency + this far ahead; must cover shown_to_decision_ms p95 "
                         f"+ {real.MOVE_LEAD_MS:g} ms (a press's move goes that long before its key), else presses go late")
    ap.add_argument("--fps", type=float, default=60.0, help="capture rate (120: the model still sees >= 15 ms steps)")
    ap.add_argument("--guard-ms", type=float,
                    help="never press a hovered circle due later than this (default meh - 15)")
    ap.add_argument("--dry", type=Path, help="a saved run or recording: replay it offline instead of playing")
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.dry:
        return dry(args, beatmap.parse(args.map))
    if not lazer.is_imported(args.map):
        sys.exit(f"{args.map}: this exact file is not in lazer's store (import its set first)")
    tail = lazer.LogTail()
    if tail.screen is None:
        sys.exit("osu! is not running: start it (uv run python -c 'from osu_ai import lazer; lazer.launch()')")
    if tail.screen in lazer.PLAYERS:
        sys.exit(f"osu! is in {tail.screen}: get it to a menu first")
    agent = load_agent(args)
    with Hands.launch() as hands:
        perms = hands.permissions()
        if not all(perms.values()):
            sys.exit(f"OsuHands permissions {perms}: uv run python -m osu_ai.hands_client grant")
        try:
            drv = real.Driver(hands, tail)
        except (RuntimeError, ValueError, HandsError) as e:
            sys.exit(str(e))
        play(drv, agent, args.map, args)


if __name__ == "__main__":
    main()
