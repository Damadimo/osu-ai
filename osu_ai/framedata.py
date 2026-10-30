"""Frame datasets for the vision model: 60 fps clips of a map, labeled from the beatmap.

Each frame is a small grayscale render of what the player sees, assuming perfect
play (circles vanish once hit). Labels come straight from the .osu file: every
upcoming object's position and how many ms until it must be hit, plus whether a
slider or spinner is active.
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from . import frames, rules
from .beatmap import Beatmap, Kind

FPS = 60
FRAME_MS = 1000 / FPS
SIZE = (160, 120)  # model input, width x height
SCALE = SIZE[0] / frames.W  # screen pixels per rendered pixel
HISTORY = 4  # frames per model input


def small_gray(surf: pygame.Surface) -> np.ndarray:
    small = pygame.transform.smoothscale(surf, SIZE)
    return pygame.surfarray.array3d(small).mean(axis=2).T.astype(np.uint8)  # (H, W)


def to_input_px(p) -> tuple[float, float]:
    """osu!pixel -> model-input pixel."""
    sx, sy = frames.to_screen(p)
    return sx * SCALE, sy * SCALE


def labels_at(bm: Beatmap, t: float):
    preempt = bm.difficulty.preempt
    objs, slider, spinner = [], (0.0, 0.0, 0.0), 0.0
    for o in bm.objects:
        if o.time - preempt > t:
            break
        if o.kind is Kind.SPINNER:
            if o.time <= t <= o.end_time:
                spinner = 1.0
            continue
        if o.time >= t:
            x, y = to_input_px(o.pos)
            objs.append((x, y, o.time - t))
        elif o.kind is Kind.SLIDER and t <= o.end_time:
            x, y = to_input_px(o.ball_at(t))
            slider = (1.0, x, y)
    return objs, slider, spinner


def render_clips(bm: Beatmap, n_clips: int, clip_frames: int, rng: random.Random) -> dict:
    colors = frames.combo_colors(bm)
    start, end = bm.objects[0].time - bm.difficulty.preempt, bm.objects[-1].end_time
    imgs, ts, clip_ids, offsets, objs, sliders, spinners = [], [], [], [0], [], [], []
    for c in range(n_clips):
        t0 = rng.uniform(start, max(start, end - clip_frames * FRAME_MS))
        for k in range(clip_frames):
            t = t0 + k * FRAME_MS
            imgs.append(small_gray(frames.render(bm, t, colors)))
            o, s, sp = labels_at(bm, t)
            objs.extend(o)
            offsets.append(len(objs))
            sliders.append(s)
            spinners.append(sp)
            ts.append(t)
            clip_ids.append(c)
    return {
        "frames": np.stack(imgs), "t": np.array(ts, np.float32), "clip": np.array(clip_ids, np.int32),
        "obj_offsets": np.array(offsets, np.int64), "objs": np.array(objs, np.float32).reshape(-1, 3),
        "slider": np.array(sliders, np.float32), "spinner": np.array(spinners, np.float32),
        "preempt": np.float32(bm.difficulty.preempt), "radius_px": np.float32(bm.difficulty.radius * SCALE),
    }
