"""Look at osu! and nudge it one step at a time: a screenshot, optionally after a click or some keys.

  uv run scripts/lazer_shot.py                          # PNG of osu!'s client area, 2 px per point
  uv run scripts/lazer_shot.py --keys 'p p p'           # press P three times (350 ms apart), then the PNG
  uv run scripts/lazer_shot.py --keys 'ctrl+enter'      # a chord: modifiers down first, up last
  uv run scripts/lazer_shot.py --click 512,300          # left-click client point (512, 300), then the PNG
  uv run scripts/lazer_shot.py --move 100,600           # only move the cursor there (check where osu! draws it)
  uv run scripts/lazer_shot.py --display                # the whole main display instead of osu!

Click coordinates are client points: (0, 0) is the top-left of osu!'s drawable area (under the
title bar). The PNG has --scale px per point (2 by default), so something at PNG pixel (px, py)
is at --click px/2,py/2. Clicks on the toolbar band (top 45 pt, where the log-in button is) are
refused. Key names: letters, digits, enter, esc, space, tab, backspace, up/down/left/right,
f1..f12, and ctrl/shift/alt/cmd in chords. Prints one JSON line: where the PNG went and what
osu!'s log says (screen, selected beatmap). Connecting replaces any other OsuHands client.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from osu_ai import lazer, real
from osu_ai.hands_client import Hands, HandsError, parse_keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="PNG path (default runs/shots/<time>-shot.png)")
    ap.add_argument("--keys", help="space-separated keys or chords, e.g. 'p p p' or 'ctrl+enter'")
    ap.add_argument("--gap-ms", type=float, default=350.0, help="between keys")
    ap.add_argument("--click", help="x,y in client points")
    ap.add_argument("--move", help="x,y in client points: move the cursor there, no click")
    ap.add_argument("--wait-ms", type=float, default=700.0, help="after clicking/typing, before the screenshot")
    ap.add_argument("--display", action="store_true", help="screenshot the main display, not osu!")
    ap.add_argument("--whole", action="store_true", help="the whole window frame, title bar included")
    ap.add_argument("--scale", type=float, default=2.0, help="PNG pixels per point")
    args = ap.parse_args()
    if args.keys:
        parse_keys(args.keys)  # fail on a typo before touching anything
    click, move = (tuple(float(v) for v in a.split(",")) if a else None for a in (args.click, args.move))
    if any(p is not None and len(p) != 2 for p in (click, move)):
        sys.exit("--click and --move take x,y")

    tail = lazer.LogTail()
    with Hands.launch() as hands:
        perms = hands.permissions()
        if not perms["screen_capture"]:
            sys.exit("OsuHands has no Screen Recording permission: uv run python -m osu_ai.hands_client grant")
        try:
            win = None if args.display and not (args.keys or click or move) else real.locate(hands)
        except (HandsError, ValueError) as e:
            sys.exit(str(e))
        if args.keys or click or move:
            if not (perms["accessibility"] and perms["post_event"]):
                sys.exit("OsuHands cannot post input (Accessibility): uv run python -m osu_ai.hands_client grant")
            try:
                drv = real.Driver(hands, tail, win)
            except RuntimeError as e:
                sys.exit(str(e))
            if move:
                drv.front()
                hands.move(*win.client_point(*move))
            if click:
                drv.click(*click)
            if args.keys:
                drv.press(args.keys, gap_ms=args.gap_ms)
            time.sleep(args.wait_ms / 1000)
        if args.display:
            rect = next(d for d in hands.displays() if d["main"])["frame"]
        elif args.whole:
            rect = next(w for w in hands.windows("cg") if w["bundle_id"] == lazer.BUNDLE and w["layer"] == 0)["frame"]
        else:
            rect = win.rect
        size = tuple(min(4096, 2 * round(v * args.scale / 2)) for v in rect[2:])
        try:
            img = real.grab(hands, rect, size)[0]
        except HandsError as e:
            sys.exit(f"screenshot failed: {e}")
        out = args.out or real.RUNS / "shots" / f"{time.strftime('%Y%m%d-%H%M%S')}-shot.png"
        real.save_png(out, img)
        tail.poll()
        print(json.dumps({"png": str(out), "rect_pt": list(rect), "png_px_per_pt": size[0] / rect[2],
                          "client_rect_pt": list(win.rect) if win else None, "screen": tail.screen,
                          "beatmap": tail.beatmap, "frontmost": (hands.frontmost or {}).get("bundle_id")}))


if __name__ == "__main__":
    main()
