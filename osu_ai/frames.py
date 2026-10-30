"""Render a moment of a beatmap to an image, the way the player sees it.

Used to give image-capable models (Jev 1.13+/1.14) a real screenshot instead of
a text description. Offscreen pygame; no window needed.
"""

from __future__ import annotations

import io
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from . import rules
from .beatmap import Beatmap, Kind

W, H = 640, 480
MARGIN_X, MARGIN_Y = 64, 48  # 512x384 playfield centered, osu!pixels = screen pixels
COMBO_COLORS = [(255, 102, 171), (84, 190, 255), (255, 209, 84), (128, 232, 146)]

_font: pygame.font.Font | None = None


def _get_font() -> pygame.font.Font:
    global _font
    if _font is None:
        pygame.font.init()
        _font = pygame.font.SysFont("helveticaneue", 22, bold=True)
    return _font


def to_screen(p) -> tuple[int, int]:
    return int(p[0] + MARGIN_X), int(p[1] + MARGIN_Y)


def to_playfield(x: float, y: float) -> tuple[float, float]:
    """Screen pixel -> osu!pixel."""
    return x - MARGIN_X, y - MARGIN_Y


def combo_colors(bm: Beatmap) -> dict[int, tuple[int, int, int]]:
    colors, k = {}, -1
    for o in bm.objects:
        k += o.new_combo
        colors[o.index] = COMBO_COLORS[k % len(COMBO_COLORS)]
    return colors


def render(bm: Beatmap, t: float, colors: dict | None = None, cursor=None, done: set[int] = frozenset()) -> pygame.Surface:
    colors = colors or combo_colors(bm)
    preempt, r = bm.difficulty.preempt, int(bm.difficulty.radius)
    surf = pygame.Surface((W, H))
    surf.fill((12, 12, 18))
    font = _get_font()
    visible = [o for o in bm.objects
               if o.time - preempt <= t <= max(o.end_time, o.time) and o.index not in done]
    for o in reversed(visible):
        color = colors[o.index]
        if o.kind is Kind.SPINNER:
            c = to_screen(rules.SPINNER_CENTER)
            pygame.draw.circle(surf, (90, 90, 120), c, 170, 5)
            left = max(0.0, min(1.0, (o.end_time - t) / max(1, o.end_time - o.time)))
            pygame.draw.circle(surf, (210, 210, 240), c, int(15 + 155 * left), 2)
            continue
        if o.kind is Kind.SLIDER:
            pts = [to_screen(p) for p in o.path.points[::2]] + [to_screen(o.path.end)]
            body = tuple(int(v * 0.35) for v in color)
            for p in pts:
                pygame.draw.circle(surf, (230, 230, 230), p, r)
            for p in pts:
                pygame.draw.circle(surf, body, p, r - 4)
            pygame.draw.circle(surf, (230, 230, 230), to_screen(o.path.end), r, 3)
            if t >= o.time:
                pygame.draw.circle(surf, (250, 250, 250), to_screen(o.ball_at(t)), int(r * 0.85))
                continue
        pos = to_screen(o.pos)
        pygame.draw.circle(surf, color, pos, r)
        pygame.draw.circle(surf, (240, 240, 240), pos, r, 3)
        num = font.render(str(o.combo_number), True, (255, 255, 255))
        surf.blit(num, num.get_rect(center=pos))
        if t < o.time:
            ring = r * (1 + 3 * (o.time - t) / preempt)
            pygame.draw.circle(surf, color, pos, int(ring), 3)
    if cursor is not None:
        pygame.draw.circle(surf, (255, 230, 90), to_screen(cursor), 8)
    return surf


def png_bytes(surf: pygame.Surface) -> bytes:
    buf = io.BytesIO()
    pygame.image.save(surf, buf, "frame.png")
    return buf.getvalue()


def jpg_bytes(surf: pygame.Surface) -> bytes:
    buf = io.BytesIO()
    pygame.image.save(surf, buf, "frame.jpg")
    return buf.getvalue()
