"""Real osu!lazer footage: captured frames, aligned to game time and labeled from the beatmap.

A recording is the stream of grayscale frames OsuHands captured over the 640x480 virtual
screen (playfield at (64, 56)), stored at the model's 320x240, with each frame's display
time on the capture clock. Game time per frame comes from the footage itself: our renderer
(lazer_frames.py) draws what lazer shows at any game time, so the clock offset is the one
that makes rendered frames match the captured ones best. Labels then come straight from the
beatmap, as in the simulator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import judge, osr
from .beatmap import Beatmap, Kind
from .lazer_frames import Look, Outcome, View, render
from .replay import Replay


@dataclass
class Recording:
    frames: np.ndarray  # (N, H, W) uint8 grayscale of the 640x480 osu!px region
    pts_ms: np.ndarray  # (N,) float64 presentation time of each frame, ms on the capture clock
    meta: dict

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, frames=self.frames, pts_ms=self.pts_ms, meta=json.dumps(self.meta))
        return path

    @classmethod
    def load(cls, path: Path) -> Recording:
        d = np.load(path)
        return cls(d["frames"], d["pts_ms"], json.loads(str(d["meta"])))

    @property
    def view(self) -> View:
        h, w = self.frames.shape[1:]
        return View((w, h))


def _match(rec: Recording, bm: Beatmap, outcome: Outcome, look: Look, idx: np.ndarray, c: float) -> float:
    """Mean correlation between captured frames and frames rendered at capture time + c."""
    scores = []
    for i in idx:
        real = rec.frames[i].astype(np.float32)
        fake = render(bm, rec.pts_ms[i] + c, outcome, look, rec.view)
        if fake.std() < 1 or real.std() < 1:
            continue
        scores.append(float(np.corrcoef(real.ravel(), fake.ravel())[0, 1]))
    return float(np.mean(scores)) if scores else -1.0


def align(rec: Recording, bm: Beatmap, outcome: Outcome, look: Look = Look(), guess_ms: float | None = None,
          search_ms: float = 20000, frames_used: int = 120) -> tuple[float, dict]:
    """Clock offset c (game time = capture time + c) that best explains the footage.

    Coarse search over +-search_ms around the guess (default: the first object appears with the
    first busy frame), then 1 ms steps; returns c and the match quality."""
    busy = np.nonzero(rec.frames.reshape(len(rec.frames), -1).std(axis=1) > 3)[0]
    if len(busy) < 10:
        raise ValueError("recording has no gameplay frames")
    if guess_ms is None:
        guess_ms = bm.objects[0].time - bm.difficulty.preempt - rec.pts_ms[busy[0]]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(busy, size=min(frames_used, len(busy)), replace=False))
    coarse_idx = idx[:: max(1, len(idx) // 30)]
    grid = guess_ms + np.arange(-search_ms, search_ms + 1, 25.0)
    coarse = [_match(rec, bm, outcome, look, coarse_idx, c) for c in grid]
    best = float(grid[int(np.argmax(coarse))])
    for step, span in ((5.0, 50.0), (1.0, 8.0), (0.25, 1.5)):
        cands = best + np.arange(-span, span + step / 2, step)
        scores = [_match(rec, bm, outcome, look, idx, c) for c in cands]
        best = float(cands[int(np.argmax(scores))])
    return best, {"match": _match(rec, bm, outcome, look, idx, best), "coarse_best": float(max(coarse))}




def replay_from_osr(path: Path, bm: Beatmap) -> Replay:
    """An .osr (ours or lazer's own export) as a Replay in beatmap time: cursor frames and key
    presses (lazer shifts replays of format v4-and-older maps by 24 ms, osr.time_offset)."""
    decoded = osr.read(path)
    offset = osr.time_offset(bm)
    out, t, last = Replay(), float(offset), None
    for e in decoded.replay_data:
        t += e.time_delta
        if e.time_delta == -12345 or (last is not None and t < last):
            continue
        last = t
        out.move(t, (e.x, e.y))
    for p in osr.presses(decoded, offset):
        out.press(p.down, p.up)
    return out


def outcome_from_score(bm: Beatmap, score: judge.Score) -> Outcome:
    """When each circle and slider head was judged, and whether it was hit, from a judged play."""
    judged = {e.index: (e.t, e.result != "Miss") for e in score.events
              if e.kind in ("HitCircle", "SliderHeadCircle")}
    return Outcome(judged, {o.index: 1.6 for o in bm.objects if o.kind is Kind.SPINNER})
