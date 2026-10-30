"""Small vision model for osu!: last 4 frames in, every upcoming circle out.

CenterNet-style heads on a stride-4 grid (40x30 cells for a 160x120 input):
  heat  - logit that an upcoming hit object is centered in this cell
  off   - sub-cell position of that center
  dt    - time until it must be hit, in seconds
  ball  - logit that the active slider's ball is in this cell (follow it while holding)
  hold, spin - whole-screen logits: keep the key down for a slider / spin a spinner
  next, next2 - (optional) the next circle to hit and the one after it, each with its own
          heat, sub-cell position and dt, regressed over the 3x3 cells around it. In streams
          and stacks neighbouring circles share cells, so "every upcoming circle" can't tell
          which one is due; these heads can, and the policy aims with them.
Aim comes from the next circle's position; clicking happens at its dt.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .framedata import HISTORY, SIZE

STRIDE = 4
GRID = (SIZE[1] // STRIDE, SIZE[0] // STRIDE)  # (rows, cols) = (30, 40)
NEXT_CH = 4  # per next-head: heat logit, off x, off y, dt


def conv(cin: int, cout: int, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm(cout),
        nn.ReLU(),
    )


class Res(nn.Module):
    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        self.a, self.b = conv(ch, ch, dilation=dilation), conv(ch, ch, dilation=dilation)

    def __call__(self, x):
        return x + self.b(self.a(x))


class OsuNet(nn.Module):
    """stride 4: 160x120 input (the first simulator models). stride 8: 320x240 input, where the
    approach ring is about 2 px thick instead of 1; the extra stride-2 stem keeps the same
    40x30 output grid and nearly the same compute."""

    def __init__(self, width: int = 48, stride: int = 4, next_heads: bool = False):
        super().__init__()
        w = width
        self.stride = stride
        self.next_heads = next_heads
        if stride == 8:
            self.stem = nn.Sequential(conv(HISTORY, w // 2, stride=2), conv(w // 2, w, stride=2))  # 60x80
        else:
            self.stem = nn.Sequential(conv(HISTORY, w), conv(w, w, stride=2))  # 60x80
        self.down1 = nn.Sequential(conv(w, 2 * w, stride=2), Res(2 * w))  # 30x40
        self.down2 = nn.Sequential(conv(2 * w, 4 * w, stride=2), Res(4 * w), Res(4 * w, 2), Res(4 * w, 4))  # 15x20
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse = nn.Sequential(conv(6 * w, 2 * w), Res(2 * w))
        self.heat = nn.Conv2d(2 * w, 1, 1)
        self.off = nn.Conv2d(2 * w, 2, 1)
        self.dt = nn.Conv2d(2 * w, 1, 1)
        self.ball = nn.Conv2d(2 * w, 1, 1)
        self.modes = nn.Linear(4 * w, 2)  # hold, spin
        if next_heads:
            self.nxt = nn.Sequential(conv(2 * w, 2 * w), nn.Conv2d(2 * w, 2 * NEXT_CH, 1))

    def __call__(self, x):
        s1 = self.down1(self.stem(x))
        s2 = self.down2(s1)
        f = self.fuse(mx.concatenate([s1, self.up(s2)], axis=-1))
        state = self.modes(s2.mean(axis=(1, 2)))
        out = self.heat(f)[..., 0], self.off(f), self.dt(f)[..., 0], self.ball(f)[..., 0], state
        if self.next_heads:  # (B, rows, cols, 2, NEXT_CH): [heat logit, off x, off y, dt] for next, next2
            nxt = self.nxt(f)
            return *out, nxt.reshape(*nxt.shape[:3], 2, NEXT_CH)
        return out


def _gaussian(cx: float, cy: float, grid: tuple[int, int] = GRID) -> np.ndarray:
    rows, cols = grid
    yy, xx = np.mgrid[0:rows, 0:cols]
    return np.exp(-((xx - cx + 0.5) ** 2 + (yy - cy + 0.5) ** 2) / 2.0)


def targets(objs_per_frame: list[np.ndarray], max_dt_ms: float, sliders=None, spinners=None,
            stride: int = STRIDE, grid: tuple[int, int] = GRID) -> dict[str, np.ndarray]:
    """CenterNet targets: gaussian heat, and offset/dt at each object's center cell; slider ball and states.
    Positions are in input pixels; `stride` input pixels per grid cell."""
    rows, cols = grid
    b = len(objs_per_frame)
    sliders = np.zeros((b, 3), np.float32) if sliders is None else np.asarray(sliders, np.float32)
    spinners = np.zeros(b, np.float32) if spinners is None else np.asarray(spinners, np.float32)
    ball = np.zeros((b, rows, cols), np.float32)
    for i, (active, bx, by) in enumerate(sliders):
        if active:
            cx, cy = bx / stride, by / stride
            ix, iy = int(np.clip(cx, 0, cols - 1)), int(np.clip(cy, 0, rows - 1))
            ball[i] = _gaussian(cx, cy, grid)
            ball[i, iy, ix] = 1.0
    heat = np.zeros((b, rows, cols), np.float32)
    off = np.zeros((b, rows, cols, 2), np.float32)
    dt = np.zeros((b, rows, cols), np.float32)
    mask = np.zeros((b, rows, cols), np.float32)
    best = np.full((b, rows, cols), np.inf, np.float32)
    yy, xx = np.mgrid[0:rows, 0:cols]
    for i, objs in enumerate(objs_per_frame):
        for x, y, ms in objs:
            if ms > max_dt_ms:
                continue
            cx, cy = x / stride, y / stride
            ix, iy = int(cx), int(cy)
            if not (0 <= ix < cols and 0 <= iy < rows):
                continue
            g = np.exp(-((xx - cx + 0.5) ** 2 + (yy - cy + 0.5) ** 2) / (2 * 1.0 ** 2))
            heat[i] = np.maximum(heat[i], g)
            heat[i, iy, ix] = 1.0
            if ms < best[i, iy, ix]:  # stacked objects: the soonest one owns the cell
                best[i, iy, ix] = ms
                off[i, iy, ix] = (cx - ix, cy - iy)
                dt[i, iy, ix] = ms / 1000
                mask[i, iy, ix] = 1.0
    state = np.stack([sliders[:, 0], spinners], axis=1)
    out = {"heat": heat, "off": off, "dt": dt, "mask": mask, "ball": ball, "state": state}
    # The next two circles: heat as above; position and dt regressed on the 3x3 cells around each,
    # weighted by the gaussian, so a peak one cell off still reads the right circle.
    nheat = np.zeros((b, rows, cols, 2), np.float32)
    nreg = np.zeros((b, rows, cols, 2, 3), np.float32)  # off x, off y (from each cell's corner), dt s
    nw = np.zeros((b, rows, cols, 2), np.float32)
    for i, objs in enumerate(objs_per_frame):
        order = sorted((o for o in objs if o[2] <= max_dt_ms), key=lambda o: o[2])[:2]
        for k, (x, y, ms) in enumerate(order):
            cx, cy = x / stride, y / stride
            ix, iy = int(cx), int(cy)
            if not (0 <= ix < cols and 0 <= iy < rows):
                continue
            g = np.exp(-((xx - cx + 0.5) ** 2 + (yy - cy + 0.5) ** 2) / 2.0)
            nheat[i, :, :, k] = g
            nheat[i, iy, ix, k] = 1.0
            near = (np.abs(xx - ix) <= 1) & (np.abs(yy - iy) <= 1)
            nw[i, :, :, k] = np.where(near, g, 0.0)
            nreg[i, :, :, k, 0] = cx - xx
            nreg[i, :, :, k, 1] = cy - yy
            nreg[i, :, :, k, 2] = ms / 1000
    out.update({"nheat": nheat, "nreg": nreg, "nw": nw})
    return out


def _focal(logits, target):
    """CenterNet focal loss, summed."""
    p = mx.clip(mx.sigmoid(logits), 1e-4, 1 - 1e-4)
    pos = target == 1.0
    pos_loss = -((1 - p) ** 2) * mx.log(p) * pos
    neg_loss = -((1 - target) ** 4) * (p ** 2) * mx.log(1 - p) * (1 - pos)
    return pos_loss.sum() + neg_loss.sum()


def loss_fn(model: OsuNet, x, t):
    out = model(x)
    heat, off, dt, ball, state = out[:5]
    n = mx.maximum(t["mask"].sum(), 1.0)
    focal = _focal(heat, t["heat"]) / n
    n_ball = mx.maximum((t["ball"] == 1.0).sum(), 1.0)
    l_ball = _focal(ball, t["ball"]) / n_ball
    l_state = nn.losses.binary_cross_entropy(state, t["state"], with_logits=True, reduction="mean")
    m = t["mask"]
    l_off = (mx.abs(off - t["off"]).sum(axis=-1) * m).sum() / n
    l_dt = (mx.abs(dt - t["dt"]) * t.get("dt_w", m)).sum() / n  # dt_w: optional per-object weights
    total = focal + l_off + 10 * l_dt + l_ball + l_state
    if len(out) > 5:
        nxt = out[5]
        n_next = mx.maximum((t["nheat"] == 1.0).sum(), 1.0)
        l_nheat = (_focal(nxt[..., 0, 0], t["nheat"][..., 0]) + _focal(nxt[..., 1, 0], t["nheat"][..., 1])) / n_next
        w = t["nw"]
        wn = mx.maximum(w.sum(), 1e-6)
        l_noff = (mx.abs(nxt[..., 1:3] - t["nreg"][..., :2]).sum(axis=-1) * w).sum() / wn
        near = 1 + 3 * (t["nreg"][..., 2] < 0.35)  # the clicks happen on the last ~300 ms
        l_ndt = (mx.abs(nxt[..., 3] - t["nreg"][..., 2]) * w * near).sum() / wn
        total = total + l_nheat + l_noff + 10 * l_ndt
        return total, (focal + l_nheat, l_off + l_noff, l_dt + l_ndt, l_ball, l_state)
    return total, (focal, l_off, l_dt, l_ball, l_state)


def decode_state(ball, state, stride: int = STRIDE) -> list[tuple[float, float, float, float]]:
    """Per frame: (P(hold), P(spin), ball x px, ball y px)."""
    probs = np.array(mx.sigmoid(state))
    b = np.array(ball)
    out = []
    for i in range(b.shape[0]):
        iy, ix = np.unravel_index(np.argmax(b[i]), b[i].shape)
        out.append((float(probs[i, 0]), float(probs[i, 1]), (ix + 0.5) * stride, (iy + 0.5) * stride))
    return out


def decode(heat, off, dt, threshold: float = 0.3, k: int = 12,
           stride: int = STRIDE) -> list[list[tuple[float, float, float, float]]]:
    """Peaks -> [(x px, y px, dt ms, score)] per frame, soonest first."""
    p = mx.sigmoid(heat)
    pooled = nn.MaxPool2d(3, stride=1, padding=1)(p[..., None])[..., 0]
    peaks = np.array((p == pooled) & (p > threshold))
    p, off, dt = np.array(p), np.array(off), np.array(dt)
    out = []
    for i in range(p.shape[0]):
        ys, xs = np.nonzero(peaks[i])
        order = np.argsort(-p[i, ys, xs])[:k]
        dets = [((xs[j] + off[i, ys[j], xs[j], 0]) * stride, (ys[j] + off[i, ys[j], xs[j], 1]) * stride,
                 float(dt[i, ys[j], xs[j]] * 1000), float(p[i, ys[j], xs[j]])) for j in order]
        out.append(sorted(dets, key=lambda d: d[2]))
    return out


def decode_next(nxt, threshold: float = 0.3, stride: int = STRIDE) -> list[list[tuple[float, float, float, float]]]:
    """Per frame: [(x px, y px, dt ms, score)] for the next circle and the one after it (if seen),
    each averaged over the 3x3 cells around its peak."""
    nxt = np.array(nxt)
    rows, cols = nxt.shape[1:3]
    out = []
    for i in range(nxt.shape[0]):
        dets = []
        for k in range(2):
            p = 1 / (1 + np.exp(-np.clip(nxt[i, :, :, k, 0], -60, 60)))
            iy, ix = np.unravel_index(np.argmax(p), p.shape)
            if p[iy, ix] < threshold:
                continue
            y0, y1, x0, x1 = max(iy - 1, 0), min(iy + 2, rows), max(ix - 1, 0), min(ix + 2, cols)
            w = p[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            reg = nxt[i, y0:y1, x0:x1, k]
            cx = ((xx + reg[..., 1]) * w).sum() / w.sum()
            cy = ((yy + reg[..., 2]) * w).sum() / w.sum()
            ms = (reg[..., 3] * w).sum() / w.sum() * 1000
            dets.append((float(cx * stride), float(cy * stride), float(ms), float(p[iy, ix])))
        out.append(dets)
    return out
