"""The vision model as a player: frames in, what it sees out, in osu!pixels.

Frames are 320x240 grayscale views of the 640x480 virtual screen (playfield at (64, 56)),
whether they come from our lazer renderer or from the real game through OsuHands. The agent
keeps the last 4 frames and reports every upcoming circle it detects (position, ms until it
must be hit, confidence) plus whether a slider or spinner is active and where the ball is.
"""

from __future__ import annotations

from collections import deque

import mlx.core as mx
import numpy as np

from .framedata import HISTORY
from .lazer_frames import MARGIN, VIRTUAL
from .vision import OsuNet, decode, decode_next, decode_state

MIN_SCORE = 0.35


class LazerAgent:
    def __init__(self, weights: str, width: int = 48, input_size: tuple[int, int] = (320, 240)):
        mx.set_cache_limit(256 << 20)  # one small frame at a time: never let MLX hoard freed buffers
        self.input_size = input_size
        self.stride = 8 if input_size[0] >= 320 else 4
        keys = mx.load(weights)
        self.model = OsuNet(width, stride=self.stride, next_heads=any(k.startswith("nxt.") for k in keys))
        self.model.load_weights(weights)
        self.model.eval()
        mx.eval(self.model.parameters())
        self.history: deque[np.ndarray] = deque(maxlen=HISTORY)
        self.f = input_size[0] / VIRTUAL[0]  # input px per osu!px

    def to_osu(self, px: float, py: float) -> tuple[float, float]:
        return px / self.f - MARGIN[0], py / self.f - MARGIN[1]

    def reset(self) -> None:
        self.history.clear()

    def see(self, frame: np.ndarray):
        """Push a (H, W) uint8 frame. Returns (detections, modes):
        detections [(x, y, ms until hit, score)] in osu!px, soonest first;
        modes (P(hold), P(spin), ball x, ball y)."""
        self.history.append(frame)
        while len(self.history) < HISTORY:
            self.history.appendleft(frame)
        x = np.stack(self.history, axis=-1)[None].astype(np.float32) / 255.0
        out = self.model(mx.array(x))
        heat, off, dt, ball, modes = out[:5]
        found = decode(heat, off, dt, threshold=MIN_SCORE, stride=self.stride)[0]
        if len(out) > 5:  # the next circle and the one after it, when the model is sure of them
            found = decode_next(out[5], threshold=MIN_SCORE, stride=self.stride)[0] or found
        dets = [(*self.to_osu(px, py), ms, score) for px, py, ms, score in found]
        p_hold, p_spin, bx, by = decode_state(ball, modes, stride=self.stride)[0]
        return dets, (p_hold, p_spin, *self.to_osu(bx, by))
