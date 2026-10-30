"""Render osu!lazer frames as the model sees them: our minimal skin on a black playfield.

This reproduces what the real client draws with the 'osu-jev minimal' skin
(scripts/make_skin.py) and our settings (background dim 100%, storyboard off, HUD hidden),
using lazer 2026.1005.0's own timings (docs/research/visuals.md, skins.md):

  hit circle  gray-160 disc of radius 54u, fading in over F ms from T-P, tinted 0.765 until
              T-400 and back to full by T-300
  approach    white ring, radii 54u..63u times k = 4 -> 1 over P; alpha 0.9 ramping over
              min(2F, P); fades over 50 ms after T and vanishes the moment the circle is hit
  slider      legacy body (gray-128 border, dark gradient core), head = gray ring with a black
              core, white ball of radius 46u from T to the end, white reverse arrows
  spinner     gray ring growing with spin progress, white approach ring shrinking to the end
  after       hit: 60 ms OutQuad fade (hit animations off) or 240 ms fade growing to x1.4 (on);
              miss: 100 ms linear fade

u is 1/64 of the circle radius. Frames cover the 640x480 virtual screen with the playfield at
(64, 56), rendered straight at the output size with area anti-aliasing. What happened to each
object (hit or missed, and when) comes from an Outcome, so frames show imperfect play as
lazer would.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .beatmap import Beatmap, HitObject, Kind

MARGIN = (64.0, 56.0)  # playfield origin in the 640x480 virtual screen (lazer's +8 shift)
VIRTUAL = (640, 480)

CIRCLE_GRAY, BORDER_GRAY, CORE_PEAK = 160.0, 128.0, 44.6
DIM = 195 / 255  # "not yet hittable" tint
DISC_U, HEAD_CORE_U, BALL_U = 54.0, 28.0, 46.0
RING_U = (54.0, 63.0)
BODY_EDGE_U, BORDER_U, SHADOW_U = 64.0, 52.0, 59.0  # body radius; inner edge of border; shadow starts
SPIN_RING = ((121.0, 127.0), (151.0, 159.0))  # at progress 0 and 1 (osu!px)
SPIN_APPROACH = (116.0, 126.0)  # times 1.1625 -> 0.0625 over the spinner


@dataclass(frozen=True)
class Look:
    """Client settings that change how objects look (osu! ruleset settings, in client.realm)."""

    hit_animations: bool = False
    snaking: bool = False


@dataclass
class Outcome:
    """How each object was played: what decides how lazer draws it after the fact."""

    judged: dict[int, tuple[float, bool]] = field(default_factory=dict)  # circle-like -> (time, hit)
    spin_rate: dict[int, float] = field(default_factory=dict)  # spinner -> progress per spinner length

    def state(self, o: HitObject, t: float) -> tuple[float, bool] | None:
        """(judgement time, hit) if the object's head was judged at or before t."""
        j = self.judged.get(o.index)
        return j if j is not None and j[0] <= t else None


def perfect(bm: Beatmap) -> Outcome:
    return Outcome({o.index: (o.time, True) for o in bm.objects if o.kind is not Kind.SPINNER})


def sample_outcome(bm: Beatmap, rng: random.Random, hit_rate: float = 0.95, timing_sd: float = 12.0) -> Outcome:
    """Imperfect but plausible play: timing noise, some misses (unclicked, or clicked far off),
    and lazer's note lock (hitting a later circle force-misses earlier unjudged ones)."""
    w = bm.difficulty.windows
    judged: dict[int, tuple[float, bool]] = {}
    for o in bm.objects:
        if o.kind is Kind.SPINNER:
            continue
        if rng.random() < hit_rate:
            judged[o.index] = (o.time + max(-w.meh, min(w.meh, rng.gauss(0, timing_sd))), True)
        elif rng.random() < 0.7:
            judged[o.index] = (o.time + w.meh, False)  # never clicked: misses once it can't be hit
        else:
            off = rng.uniform(w.meh, min(w.meh + 150, 399))
            judged[o.index] = (o.time + (off if rng.random() < 0.5 else -off), False)
    first_later_hit = math.inf  # note lock: a hit force-misses every earlier object still waiting
    for i in sorted(judged, reverse=True):
        t, hit = judged[i]
        if t > first_later_hit:
            judged[i] = (first_later_hit, False)
        elif hit:
            first_later_hit = t
    spin = {o.index: rng.uniform(1.2, 2.5) for o in bm.objects if o.kind is Kind.SPINNER}
    return Outcome(judged, spin)


# ---- drawing primitives (output pixels; pixel i covers [i, i+1)) ----

@dataclass(frozen=True)
class View:
    """Where the virtual screen lands in the output image. zoom/shift (output px) model small
    errors in our screen calibration; the defaults are exact."""

    size: tuple[int, int] = (320, 240)
    zoom: float = 1.0
    shift: tuple[float, float] = (0.0, 0.0)

    @property
    def f(self) -> float:
        return self.size[0] / VIRTUAL[0] * self.zoom  # output px per osu!px

    def to_px(self, p) -> tuple[float, float]:
        w, h = self.size
        return ((p[0] + MARGIN[0] - VIRTUAL[0] / 2) * self.f + w / 2 + self.shift[0],
                (p[1] + MARGIN[1] - VIRTUAL[1] / 2) * self.f + h / 2 + self.shift[1])


class Canvas:
    """The objects' layer over black (img) and how much of whatever is behind still shows through
    each pixel (trans), so the frame over a background B is img + trans * B."""

    def __init__(self, view: View):
        self.view = view
        self.w, self.h = view.size
        self.f = view.f
        self.to_px = view.to_px
        self.img = np.zeros((self.h, self.w), np.float32)
        self.trans = np.ones((self.h, self.w), np.float32)

    def _box(self, cx: float, cy: float, r: float):
        x0, x1 = max(int(cx - r - 1), 0), min(int(cx + r + 2), self.w)
        y0, y1 = max(int(cy - r - 1), 0), min(int(cy + r + 2), self.h)
        if x0 >= x1 or y0 >= y1:
            return None
        ys = np.arange(y0, y1, dtype=np.float32) + 0.5 - cy
        xs = np.arange(x0, x1, dtype=np.float32) + 0.5 - cx
        return (slice(y0, y1), slice(x0, x1)), np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)

    def blend(self, region, gray, alpha) -> None:
        """Alpha-composite a gray layer onto the canvas inside `region`."""
        dst = self.img[region]
        dst += (gray - dst) * alpha
        self.trans[region] *= 1 - alpha

    def disc(self, c, r: float, gray: float, alpha: float) -> None:
        if alpha <= 0 or r <= 0:
            return
        found = self._box(*c, r)
        if found:
            region, d = found
            self.blend(region, gray, alpha * np.clip(r - d + 0.5, 0, 1))

    def ring(self, c, r0: float, r1: float, gray: float, alpha: float) -> None:
        if alpha <= 0 or r1 <= 0:
            return
        found = self._box(*c, r1)
        if found:
            region, d = found
            cover = np.clip(r1 - d + 0.5, 0, 1) - np.clip(r0 - d + 0.5, 0, 1)
            self.blend(region, gray, alpha * cover)

    def polygon(self, pts: np.ndarray, gray: float, alpha: float) -> None:
        """Convex polygon (output px), anti-aliased by 4x4 supersampling of its bounding box."""
        if alpha <= 0:
            return
        x0, y0 = np.floor(pts.min(axis=0)).astype(int)
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.w), min(y1, self.h)
        if x0 >= x1 or y0 >= y1:
            return
        ss = 4
        ys = y0 + (np.arange((y1 - y0) * ss) + 0.5) / ss
        xs = x0 + (np.arange((x1 - x0) * ss) + 0.5) / ss
        inside = np.ones((len(ys), len(xs)), bool)
        n = len(pts)
        (ax, ay), (bx, by), (qx, qy) = pts[0], pts[1], pts[2]
        sign = np.sign((bx - ax) * (qy - ay) - (by - ay) * (qx - ax))
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            edge = (b[0] - a[0]) * (ys[:, None] - a[1]) - (b[1] - a[1]) * (xs[None, :] - a[0])
            inside &= edge * sign >= 0
        cover = inside.reshape(y1 - y0, ss, x1 - x0, ss).mean(axis=(1, 3))
        self.blend((slice(y0, y1), slice(x0, x1)), gray, alpha * cover)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _out_quad(x: float) -> float:
    x = _clamp01(x)
    return x * (2 - x)


def _dim(o: HitObject, t: float) -> float:
    """lazer tints objects 195/255 until they could be hit (T-400), back to white by T-300."""
    return DIM + (1 - DIM) * _clamp01((t - (o.time - 400)) / 100)


@dataclass(frozen=True)
class _Head:
    alpha: float  # whole-piece alpha (fade-in times hit/miss fade)
    scale: float  # hit-animation growth
    approach: float  # approach ring alpha


def _head_state(o: HitObject, t: float, preempt: float, fade: float, outcome: Outcome, look: Look) -> _Head | None:
    if t < o.time - preempt:
        return None
    alpha, scale = _clamp01((t - (o.time - preempt)) / fade), 1.0
    approach = 0.9 * _clamp01((t - o.time + preempt) / min(2 * fade, preempt))
    if t > o.time:
        approach *= _clamp01(1 - (t - o.time) / 50)
    judged = outcome.state(o, t)
    if judged is not None:
        tj, hit = judged
        u = t - tj
        if hit:
            approach = 0.0
            grow = _clamp01(u / 240)
            scale = 1 + 0.4 * _out_quad(grow)
            alpha *= 1 - _out_quad(grow)  # the legacy piece fades over 240 ms either way
            if not look.hit_animations:
                alpha *= (1 - _clamp01(u / 60)) ** 2  # ...but the whole object is gone in 60 ms
        else:
            fade_out = _clamp01(1 - u / 100)
            alpha *= fade_out
            approach *= fade_out
        if alpha <= 0.002 and approach <= 0.002:
            return None
    return _Head(alpha, scale, approach)


def _draw_head(cv: Canvas, o: HitObject, t: float, head: _Head, s: float, slider: bool) -> None:
    c = cv.to_px(o.pos)
    k = s * cv.f * head.scale  # output px per u
    gray = CIRCLE_GRAY * _dim(o, t)
    if slider:  # one opaque texture: a gray ring around a black core, so nothing shows through the seam
        found = cv._box(*c, DISC_U * k)
        if found and head.alpha > 0:
            region, d = found
            outer = np.clip(DISC_U * k - d + 0.5, 0, 1)
            core = np.clip(HEAD_CORE_U * k - d + 0.5, 0, 1)
            cv.blend(region, gray * (outer - core) / np.maximum(outer, 1e-6), head.alpha * outer)
    else:
        cv.disc(c, DISC_U * k, gray, head.alpha)


def _draw_approach(cv: Canvas, o: HitObject, t: float, head: _Head, preempt: float, s: float) -> None:
    if head.approach <= 0:
        return
    k = 4 - 3 * _clamp01((t - o.time + preempt) / preempt)
    unit = s * cv.f * k
    cv.ring(cv.to_px(o.pos), RING_U[0] * unit, RING_U[1] * unit, 255.0, head.approach)


def _path_points(o: HitObject, lo: float = 0.0, hi: float = 1.0, step_px: float = 2.0) -> np.ndarray:
    """Points along the slider path between progress lo and hi (osu!px)."""
    n = max(2, int((hi - lo) * o.path.length / step_px) + 2)
    return np.array([o.path.at(lo + (hi - lo) * i / (n - 1)) for i in range(n)], np.float32)


def _segment_distance(px: np.ndarray, py: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance from each pixel to a polyline, chunked to bound memory."""
    best = np.full(px.shape, np.inf, np.float32)
    a, b = pts[:-1], pts[1:]
    ab = b - a
    len2 = np.maximum((ab ** 2).sum(axis=1), 1e-9)
    for i in range(0, len(a), 64):
        sa, sab, sl = a[i:i + 64], ab[i:i + 64], len2[i:i + 64]
        dx = px[..., None] - sa[:, 0]
        dy = py[..., None] - sa[:, 1]
        u = np.clip((dx * sab[:, 0] + dy * sab[:, 1]) / sl, 0, 1)
        d = np.sqrt((dx - u * sab[:, 0]) ** 2 + (dy - u * sab[:, 1]) ** 2).min(axis=-1)
        np.minimum(best, d, out=best)
    return best


_BODY_CACHE: dict = {}


def _draw_body(cv: Canvas, o: HitObject, t: float, alpha: float, s: float, lo: float, hi: float) -> None:
    if alpha <= 0 or hi <= lo:
        return
    key = (o.index, o.time, o.x, o.y, o.end_time, cv.view, lo, hi)
    shape = _BODY_CACHE.get(key)
    if shape is None:
        shape = _body_shape(cv, o, s, lo, hi)
        if lo == 0.0 and hi == 1.0:  # the full body never changes; cache it
            if len(_BODY_CACHE) > 400:  # bounded: training draws many views, so old shapes rarely repeat
                _BODY_CACHE.clear()
            _BODY_CACHE[key] = shape
    if shape is not None:
        region, gray, layer_alpha = shape
        cv.blend(region, gray * _dim(o, t), alpha * layer_alpha)


def _body_shape(cv: Canvas, o: HitObject, s: float, lo: float, hi: float):
    """The legacy slider body over its bounding box: (region, gray at full tint, alpha)."""
    pts = _path_points(o, lo, hi)
    x0, y0 = cv.to_px((0.0, 0.0))
    vp = pts * cv.f + np.array([x0, y0], np.float32)  # output px
    r_edge = BODY_EDGE_U * s * cv.f
    x0, y0 = np.floor(vp.min(axis=0) - r_edge - 1).astype(int)
    x1, y1 = np.ceil(vp.max(axis=0) + r_edge + 2).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, cv.w), min(y1, cv.h)
    if x0 >= x1 or y0 >= y1:
        return None
    ys = np.arange(y0, y1, dtype=np.float32) + 0.5
    xs = np.arange(x0, x1, dtype=np.float32) + 0.5
    px, py = np.meshgrid(xs, ys)
    d = _segment_distance(px, py, vp) / (s * cv.f)  # in u
    k = 1 / (s * cv.f)  # one output pixel in u, for anti-aliasing
    edge = np.clip((BODY_EDGE_U - d) / k + 0.5, 0, 1)
    in_border = np.clip((d - BORDER_U) / k + 0.5, 0, 1)  # 1 outside the core
    in_shadow = np.clip((d - SHADOW_U) / k + 0.5, 0, 1)  # 1 in the outer shadow band
    core_gray = CORE_PEAK / 0.7 * np.clip(1 - d / BORDER_U, 0, 1)
    gray = np.where(in_shadow > 0.5, 0.0, np.where(in_border > 0.5, BORDER_GRAY, core_gray))
    shadow = 0.25 * np.clip((BODY_EDGE_U - d) / (BODY_EDGE_U - SHADOW_U), 0, 1)
    layer_alpha = (1 - in_border) * 0.7 + in_border * (1 - in_shadow) + in_shadow * shadow
    return (slice(y0, y1), slice(x0, x1)), gray.astype(np.float32), (edge * layer_alpha).astype(np.float32)


def _draw_arrow(cv: Canvas, at, toward, s: float, scale: float, gray: float, alpha: float) -> None:
    """White triangle pointing from `at` toward `toward` (back along the slider)."""
    if alpha <= 0:
        return
    c = np.array(cv.to_px(at), np.float32)
    d = np.array(toward, np.float32) - np.array(at, np.float32)
    n = np.linalg.norm(d)
    d = d / n if n > 1e-6 else np.array([1.0, 0.0], np.float32)
    perp = np.array([-d[1], d[0]], np.float32)
    k = s * cv.f * scale
    tip, base, half = 21.8 * k, -17.9 * k, 21.8 * k  # the reverse-arrow texture, in u
    pts = np.stack([c + d * tip, c + d * base + perp * half, c + d * base - perp * half])
    cv.polygon(pts, gray, alpha)


def _draw_slider(cv: Canvas, o: HitObject, t: float, preempt: float, fade: float, s: float,
                 outcome: Outcome, look: Look) -> _Head | None:
    if t < o.time - preempt or t > o.end_time + 240:
        return None
    alpha = _clamp01((t - (o.time - preempt)) / fade)
    if t > o.end_time:
        alpha *= _clamp01(1 - (t - o.end_time) / 240)
    span = (o.end_time - o.time) / o.slides
    lo, hi = 0.0, 1.0
    head_judged = outcome.state(o, t)
    if look.snaking:
        hi = _clamp01((t - o.time + preempt) / (preempt / 3))
        if head_judged and head_judged[1] and t >= o.time + span * (o.slides - 1):
            p = _clamp01((t - o.time - span * (o.slides - 1)) / span) if span > 0 else 1.0
            if (o.slides - 1) % 2 == 0:
                lo = p
            else:
                hi = min(hi, 1 - p)
    _draw_body(cv, o, t, alpha, s, lo, hi)

    end, start = o.path.end, o.path.at(0)
    near_end, near_start = o.path.at(0.98), o.path.at(0.02)
    for k in range(o.slides - 1):
        t_k = o.time + (k + 1) * span
        at, toward = (end, near_end) if k % 2 == 0 else (start, near_start)
        if k == 0:
            appear = o.time - preempt + (preempt / 3 if look.snaking else 0)
            a = _clamp01((t - appear) / 150)
        else:
            appear = t_k - 2 * span
            a = _clamp01((t - appear) / min(span, 150)) if span > 0 else 1.0
        if t < appear:
            continue
        scale = 1 + 0.3 * (1 - _out_quad(((t - appear) % 300) / 300))  # idle pulse 1.3 -> 1
        if t >= t_k:
            u = _clamp01((t - t_k) / min(300, span)) if span > 0 else 1.0
            scale, a = 1 + 0.4 * _out_quad(u), a * (1 - _out_quad(u))
        _draw_arrow(cv, at, toward, s, scale, 255.0 * _dim(o, t), a * alpha)

    head = _head_state(o, t, preempt, fade, outcome, look)
    if head is not None:
        _draw_head(cv, o, t, head, s, slider=True)
    if o.time <= t < o.end_time:
        cv.disc(cv.to_px(o.ball_at(t)), BALL_U * s * cv.f, 255.0, 1.0)
    return head


def _draw_spinner(cv: Canvas, o: HitObject, t: float, fade: float, outcome: Outcome) -> None:
    if t < o.time - fade or t > o.end_time + 240:
        return
    alpha = _clamp01((t - (o.time - fade)) / fade)
    if t > o.end_time:
        alpha *= _clamp01(1 - (t - o.end_time) / 240)
    length = max(o.end_time - o.time, 1.0)
    progress = _clamp01(outcome.spin_rate.get(o.index, 1.7) * (t - o.time) / length)
    g = _out_quad(progress)
    r0 = SPIN_RING[0][0] + (SPIN_RING[1][0] - SPIN_RING[0][0]) * g
    r1 = SPIN_RING[0][1] + (SPIN_RING[1][1] - SPIN_RING[0][1]) * g
    c = cv.to_px(o.pos)
    cv.ring(c, r0 * cv.f, r1 * cv.f, BORDER_GRAY, alpha)
    if o.time <= t <= o.end_time:
        k = 1.1625 - 1.1 * _clamp01((t - o.time) / length)
        cv.ring(c, SPIN_APPROACH[0] * k * cv.f, SPIN_APPROACH[1] * k * cv.f, 255.0, alpha)


LUMA_709 = (0.2126, 0.7152, 0.0722)  # the capture's luma: BT.709 weights on gamma-encoded RGB
BG_ZOOM = 1 + 100 / 1024  # lazer enlarges every background by this about the centre (BackgroundScreen.cs)


def luma(rgb: np.ndarray, weights=LUMA_709) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (H, W) float32 luma, 0-255."""
    return rgb.astype(np.float32) @ np.asarray(weights, np.float32)


def _gauss_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k = (k / k.sum()).astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(img, pad, mode="edge")
        n = img.shape[axis]
        img = sum(w * (p[i:i + n] if axis == 0 else p[:, i:i + n]) for i, w in enumerate(k))
    return img.astype(np.float32)


def backdrop(gray: np.ndarray, view: View = View(), dim: float = 1.0, zoom: float = BG_ZOOM,
             offset: tuple[float, float] = (0.0, 0.0), blur_px: float = 0.0) -> np.ndarray:
    """The map background as lazer shows it behind the playfield, at the view's size: (H, W) float32.

    gray: the background's luma, scaled to cover the 640x480 virtual screen with its aspect kept
    (scripts/prep_backgrounds.py). lazer fills the window with it, enlarges it by BG_ZOOM about the
    centre and multiplies it by (1 - dim) in gamma-encoded values; objects then blend over it.
    offset (virtual px) and extra zoom model parallax; blur_px is a gaussian blur in output px."""
    hb, wb = gray.shape
    s = min(wb / VIRTUAL[0], hb / VIRTUAL[1])  # background px per virtual px
    w, h = view.size
    step = s / (view.f * zoom)  # background px per output px
    m = max(1, math.ceil(step - 1e-6))  # supersample so the lookup never shrinks, then average: area sampling
    f = view.f * m

    def axis(n_out, shift, off, n_bg):
        v = (np.arange(n_out, dtype=np.float32) + 0.5 - n_out / 2 - shift * m) / f  # virtual px from centre
        b = (v - off) / zoom * s + n_bg / 2 - 0.5  # background pixel coordinate
        b = np.clip(b, 0, n_bg - 1)
        i0 = np.minimum(b.astype(np.int64), n_bg - 2) if n_bg > 1 else np.zeros(n_out, np.int64)
        return i0, np.clip(b - i0, 0, 1).astype(np.float32)

    x0, wx = axis(w * m, view.shift[0], offset[0], wb)
    y0, wy = axis(h * m, view.shift[1], offset[1], hb)
    x1, y1 = np.minimum(x0 + 1, wb - 1), np.minimum(y0 + 1, hb - 1)
    top = gray[y0][:, x0] * (1 - wx) + gray[y0][:, x1] * wx
    bottom = gray[y1][:, x0] * (1 - wx) + gray[y1][:, x1] * wx
    out = top * (1 - wy)[:, None] + bottom * wy[:, None]
    if m > 1:
        out = out.reshape(h, m, w, m).mean(axis=(1, 3))
    if blur_px > 0.05:
        out = _gauss_blur(out, blur_px)
    return (out * (1 - dim)).astype(np.float32)


def render(bm: Beatmap, t: float, outcome: Outcome | None = None, look: Look = Look(),
           view: View = View(), background: np.ndarray | None = None) -> np.ndarray:
    """The frame at game time t: (H, W) float32 gray, 0-255. background: what lazer shows behind
    the playfield at the view's size (see backdrop()); None is black, dim 100%."""
    outcome = outcome if outcome is not None else perfect(bm)
    d = bm.difficulty
    preempt, s = d.preempt, d.radius / 64
    fade = 400 * min(1.0, preempt / 450)
    cv = Canvas(view)
    live = [o for o in bm.objects if o.time - preempt <= t <= o.end_time + 800]
    for o in live:  # spinners sit below everything
        if o.kind is Kind.SPINNER:
            _draw_spinner(cv, o, t, fade, outcome)
    heads = []
    for o in reversed(live):  # earlier objects are drawn on top
        if o.kind is Kind.SLIDER:
            head = _draw_slider(cv, o, t, preempt, fade, s, outcome, look)
        elif o.kind is Kind.CIRCLE:
            head = _head_state(o, t, preempt, fade, outcome, look)
            if head is not None:
                _draw_head(cv, o, t, head, s, slider=False)
        else:
            continue
        if head is not None:
            heads.append((o, head))
    for o, head in heads:  # approach circles are in a layer above all hit objects
        _draw_approach(cv, o, t, head, preempt, s)
    return cv.img if background is None else cv.img + cv.trans * background


def labels(bm: Beatmap, t: float, outcome: Outcome, view: View = View()):
    """What the model should report at time t, in output pixels:
    objs [(x, y, ms until hit)] for every circle or slider head still waiting to be judged
    (late ones have negative ms); slider (active, ball x, ball y); spinner flag."""
    preempt = bm.difficulty.preempt
    objs, slider, spinner = [], (0.0, 0.0, 0.0), 0.0
    for o in bm.objects:
        if o.time - preempt > t:
            break
        if o.kind is Kind.SPINNER:
            if o.time <= t <= o.end_time:
                spinner = 1.0
            continue
        if o.kind is Kind.SLIDER and o.time <= t <= o.end_time:
            slider = (1.0, *view.to_px(o.ball_at(t)))
        if outcome.state(o, t) is None and t <= o.time + bm.difficulty.windows.meh:
            objs.append((*view.to_px(o.pos), o.time - t))
    return objs, slider, spinner
