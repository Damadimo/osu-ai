"""Watch the vision model play osu! live, with the song.

  uv run --extra train scripts/watch_vision.py data/maps/75.osu

The model sees the last 4 frames (160x120 grayscale) every tick and decides where
to aim, when to click, and whether to hold or spin. Overlays show what it detects
(dots with its predicted ms-until-hit) and its committed click.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pygame

from osu_ai import beatmap, frames, game, judge, osr
from osu_ai.vision_live import VisionAgent, VisionGame

ZOOM = 1.5
PANEL = 300
W, H = int(frames.W * ZOOM) + PANEL, int(frames.H * ZOOM)
TEXT, DIM = (230, 230, 240), (140, 140, 160)
JUDGMENT = {300: ("300", (120, 200, 255)), 100: ("100", (110, 220, 140)), 50: ("50", (245, 190, 80)), 0: ("miss", (240, 90, 90))}


def screen_pos(p) -> tuple[int, int]:
    x, y = frames.to_screen(p)
    return int(x * ZOOM), int(y * ZOOM)


def click_sound() -> pygame.mixer.Sound:
    t = np.arange(int(44100 * 0.03)) / 44100
    wave = (np.sin(2 * np.pi * 1900 * t) * np.exp(-t * 140) * 9000).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([wave, wave]))


def draw(screen, g: VisionGame, t: float, fonts) -> None:
    big, font, small = fonts
    frame = frames.render(g.bm, t, g.colors, done=g.hit_set())
    screen.fill((10, 10, 14))
    screen.blit(pygame.transform.smoothscale(frame, (int(frames.W * ZOOM), int(frames.H * ZOOM))), (0, 0))
    for x, y, ms, score in g.dets[:6]:  # what the model sees
        p = screen_pos((x, y))
        pygame.draw.circle(screen, (255, 255, 255), p, 4)
        screen.blit(small.render(f"{ms:.0f}", True, (255, 255, 255)), (p[0] + 6, p[1] - 22))
    if g.plan:
        p = screen_pos(g.plan[1:])
        pygame.draw.circle(screen, (255, 230, 90), p, int(g.bm.difficulty.radius * ZOOM) + 6, 2)
    for when, o, j in g.judge.popups:
        age = t - when
        if 0 <= age < 600:
            label, color = JUDGMENT[j]
            pos = o.path.end if o.path is not None else o.pos
            x, y = screen_pos(pos)
            surf = font.render(label, True, color)
            screen.blit(surf, surf.get_rect(center=(x, y - 24 - age / 30)))
    pygame.draw.circle(screen, (255, 230, 90), screen_pos(g.cursor), 10)
    pygame.draw.circle(screen, (40, 30, 0), screen_pos(g.cursor), 10, 2)

    x0, y = int(frames.W * ZOOM) + 20, 16
    acc = g.judge.accuracy
    counts = {v: list(g.judge.results.values()).count(v) for v in (300, 100, 50, 0)}
    rows = [
        (big, "--" if acc is None else f"{acc:.1%}", TEXT),
        (font, f"combo {g.judge.combo}", TEXT),
        (small, f"300 {counts[300]}   100 {counts[100]}   50 {counts[50]}   miss {counts[0]}", DIM),
        (small, "", DIM),
        (font, "the model", TEXT),
        (small, f"{statistics.median(g.infer_ms):.1f} ms per frame" if g.infer_ms else "", DIM),
        (small, f"sees {len(g.dets)} upcoming circles", DIM),
        (small, f"hold {g.modes[0]:.0%}   spin {g.modes[1]:.0%}", DIM),
        (small, "click in " + (f"{g.plan[0] - t:.0f} ms" if g.plan else "--"), (255, 230, 90)),
    ]
    for f, text, color in rows:
        screen.blit(f.render(text, True, color), (x0, y))
        y += f.get_linesize() + 4
    screen.blit(small.render("Esc to stop", True, DIM), (x0, H - 30))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map")
    parser.add_argument("--weights", default="models/osunet.safetensors")
    parser.add_argument("--start", type=float)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()

    bm = game.segment(beatmap.parse(args.map), args.start, args.seconds)
    g = VisionGame(bm, VisionAgent(args.weights))
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":  # the renderer's offscreen default; this window must be real
        del os.environ["SDL_VIDEODRIVER"]
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("a small vision model plays osu!")
    fonts = (pygame.font.SysFont("helveticaneue", 40, bold=True), pygame.font.SysFont("helveticaneue", 20, bold=True),
             pygame.font.SysFont("helveticaneue", 16))
    g.on_press = click_sound().play
    audio = next((p for p in (Path("data/audio") / f"{Path(args.map).stem}{e}" for e in (".mp3", ".ogg")) if p.exists()), None)
    origin = max(g.origin, 0.0)
    if audio:
        pygame.mixer.music.load(str(audio))
        pygame.mixer.music.set_volume(0.6)
        pygame.mixer.music.play(start=origin / 1000)
    wall0, prev, running = time.perf_counter(), origin, True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        t = origin + (time.perf_counter() - wall0) * 1000
        g.step(t, max(t - prev, 1.0))
        prev = t
        draw(screen, g, t, fonts)
        pygame.display.flip()
        if t > g.end:
            running = False
        await asyncio.sleep(0.002)
    g.finish(prev)
    pygame.mixer.music.stop()
    pygame.quit()
    score = judge.score(bm, g.replay)
    print(f"{bm.name}: {score.summary()}")
    out = Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-vision-{Path(args.map).stem}.osr"
    print(f"replay: {osr.export(bm, g.replay, score, out, 'osunet')}")


if __name__ == "__main__":
    asyncio.run(main())
