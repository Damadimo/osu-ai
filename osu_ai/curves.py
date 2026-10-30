"""Slider paths built the way osu!lazer builds them, so the ball is where lazer puts it.

A port of ConvertHitObjectParser's path splitting, SliderPath (segments, length rules, catmull
optimisation) and osu-framework's PathApproximator (adaptive bezier subdivision, circular arcs,
catmull). Legacy (.osu before lazer's v128) quirks are kept: a perfect curve needs exactly 3 points
(else it is a bezier, and collinear points make it linear), catmull paths never split at repeated
points, a repeated final point stays in its segment, and a path whose last two points coincide is
not extended to its declared length.
"""

from __future__ import annotations

import bisect
import math

from .rules import f32

Point = tuple[float, float]

FIRST_LAZER_VERSION = 128
BEZIER_TOLERANCE = 0.25
CATMULL_DETAIL = 50
CIRCULAR_ARC_TOLERANCE = 0.1
LINEAR, PERFECT, CATMULL, BEZIER = "L", "P", "C", "B"  # "B<n>" is a B-spline of degree n


def _path_type(token: str) -> str:
    kind = token[0]
    if kind == "B":
        return f"B{token[1:]}" if token[1:].isdigit() and int(token[1:]) > 0 else BEZIER
    return kind if kind in (LINEAR, PERFECT) else CATMULL


def _is_linear(a: Point, b: Point, c: Point) -> bool:
    return abs((b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1])) <= 1e-3


def _segments(kind: str, points: list[Point], end_point: Point | None, version: int) -> list[list]:
    """ConvertHitObjectParser.convertPoints: [point, segment type or None] lists, split where a point repeats."""
    if kind == PERFECT:
        count = len(points) + (end_point is not None)
        if version < FIRST_LAZER_VERSION:
            if count != 3:
                kind = BEZIER
            elif _is_linear(points[0], points[1], end_point if end_point is not None else points[2]):
                kind = LINEAR
        elif count > 3:
            kind = BEZIER
    vertices = [[p, None] for p in points]
    vertices[0][1] = kind
    out, start, end = [], 0, 0
    while (end := end + 1) < len(vertices):
        if vertices[end][0] != vertices[end - 1][0]:
            continue
        if kind == CATMULL and end > 1 and version < FIRST_LAZER_VERSION:
            continue
        if end == len(vertices) - 1:
            continue
        vertices[end - 1][1] = kind
        out.append(vertices[start:end])
        start = end + 1
    if start < end:
        out.append(vertices[start:end])
    return out


def control_points(spec: str, head: Point, version: int = 14) -> list[tuple[Point, str | None]]:
    """The control points of a .osu path ("B|x:y|x:y..."), each with the type of the segment it starts."""
    points: list[Point] = []
    starts: list[tuple[str, int]] = []
    for token in spec.split("|"):
        if not token:
            continue
        if token[0].isalpha():
            starts.append((_path_type(token), len(points)))
            if not points:
                points.append(head)
        elif ":" in token:
            x, y = (f32(float(v)) for v in token.split(":")[:2])
            points.append((x, y) if version >= FIRST_LAZER_VERSION else (float(int(x)), float(int(y))))
    out: list[tuple[Point, str | None]] = []
    for i, (kind, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(points)
        if end <= start:
            continue
        end_point = points[end] if end < len(points) else None
        for seg in _segments(kind, points[start:end], end_point, version):
            out.extend((p, t) for p, t in seg)
    return out


# ---- PathApproximator ----

def _subdivide(cp: list[Point]) -> tuple[list[Point], list[Point]]:
    n = len(cp)
    mid = list(cp)
    left, right = [cp[0]] * n, [cp[0]] * n
    for i in range(n):
        left[i] = mid[0]
        right[n - i - 1] = mid[n - i - 1]
        for j in range(n - i - 1):
            mid[j] = ((mid[j][0] + mid[j + 1][0]) / 2, (mid[j][1] + mid[j + 1][1]) / 2)
    return left, right


def _flat_enough(cp: list[Point]) -> bool:
    limit = BEZIER_TOLERANCE * BEZIER_TOLERANCE * 4
    for i in range(1, len(cp) - 1):
        dx = cp[i - 1][0] - 2 * cp[i][0] + cp[i + 1][0]
        dy = cp[i - 1][1] - 2 * cp[i][1] + cp[i + 1][1]
        if dx * dx + dy * dy > limit:
            return False
    return True


def _approximate(cp: list[Point], out: list[Point]) -> None:
    n = len(cp)
    left, right = _subdivide(cp)
    lr = left + right[1:]
    out.append(cp[0])
    for i in range(1, n - 1):
        a, b, c = lr[2 * i - 1], lr[2 * i], lr[2 * i + 1]
        out.append((0.25 * (a[0] + 2 * b[0] + c[0]), 0.25 * (a[1] + 2 * b[1] + c[1])))


def _bspline_pieces(points: list[Point], degree: int) -> list[list[Point]]:
    """Bezier pieces of a clamped uniform B-spline (Boehm's knot insertion), in path order."""
    count = len(points) - 1
    if degree == count:
        return [list(points)]
    pts = list(points)
    pieces = []
    for i in range(count - degree):
        sub = [pts[i]] + [pts[i]] * degree
        for j in range(degree - 1):
            sub[j + 1] = pts[i + 1]
            for k in range(1, degree - j):
                w = min(k, count - degree - i)
                a, b = pts[i + k], pts[i + k + 1]
                pts[i + k] = ((w * a[0] + b[0]) / (w + 1), (w * a[1] + b[1]) / (w + 1))
        sub[degree] = pts[i + 1]
        pieces.append(sub)
    pieces.append(pts[count - degree:])
    return pieces


def bspline(points: list[Point], degree: int) -> list[Point]:
    """Adaptive piecewise-linear approximation (a bezier when degree >= len(points) - 1)."""
    if len(points) < 2:
        return list(points)
    degree = min(degree, len(points) - 1)
    out: list[Point] = []
    stack = _bspline_pieces(points, degree)[::-1]
    while stack:
        parent = stack.pop()
        if _flat_enough(parent):
            _approximate(parent, out)
            continue
        left, right = _subdivide(parent)
        stack.append(right)
        stack.append(left)
    out.append(points[-1])
    return out


def _arc(a: Point, b: Point, c: Point) -> tuple[Point, float, float, float, float] | None:
    """(centre, radius, start angle, signed direction, angle range) of the circle through a, b, c."""
    if _is_linear(a, b, c):
        return None
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    sa, sb, sc = (p[0] ** 2 + p[1] ** 2 for p in (a, b, c))
    centre = ((sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / d,
              (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / d)
    radius = math.dist(a, centre)
    start = math.atan2(a[1] - centre[1], a[0] - centre[0])
    end = math.atan2(c[1] - centre[1], c[0] - centre[0])
    while end < start:
        end += 2 * math.pi
    direction, span = 1.0, end - start
    if (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1]) < 0:
        direction, span = -1.0, 2 * math.pi - span
    return centre, radius, start, direction, span


def _arc_points(radius: float, span: float) -> int:
    r, tolerance = f32(radius), f32(CIRCULAR_ARC_TOLERANCE)  # lazer does this step in float32
    if 2 * r <= tolerance:
        return 2
    return max(2, math.ceil(span / (2 * math.acos(f32(1 - f32(tolerance / r))))))


def circular_arc(a: Point, b: Point, c: Point) -> list[Point] | None:
    arc = _arc(a, b, c)
    if arc is None:
        return None
    (cx, cy), r, start, direction, span = arc
    n = _arc_points(r, span)
    if n >= 1000:  # SliderPath gives up on huge arcs
        return None
    thetas = (start + direction * (i / (n - 1)) * span for i in range(n))
    return [(cx + math.cos(theta) * r, cy + math.sin(theta) * r) for theta in thetas]


def _catmull_point(v1: Point, v2: Point, v3: Point, v4: Point, t: float) -> Point:
    t2, t3 = t * t, t * t * t
    return tuple(0.5 * (2 * v2[k] + (-v1[k] + v3[k]) * t + (2 * v1[k] - 5 * v2[k] + 4 * v3[k] - v4[k]) * t2
                        + (-v1[k] + 3 * v2[k] - 3 * v3[k] + v4[k]) * t3) for k in (0, 1))


def catmull(points: list[Point]) -> list[Point]:
    """Every piece is emitted as a (start, end) pair, as osu-framework does: 100 points per knot."""
    out: list[Point] = []
    n = len(points)
    for i in range(n - 1):
        v1 = points[i - 1] if i > 0 else points[i]
        v2 = points[i]
        v3 = points[i + 1]
        v4 = points[i + 2] if i < n - 2 else (2 * v3[0] - v2[0], 2 * v3[1] - v2[1])
        for c in range(CATMULL_DETAIL):
            out.append(_catmull_point(v1, v2, v3, v4, c / CATMULL_DETAIL))
            out.append(_catmull_point(v1, v2, v3, v4, (c + 1) / CATMULL_DETAIL))
    return out


def _optimise_catmull(path: list[Point]) -> tuple[list[Point], float]:
    """Keep only vertices 6 px apart (and knot ends), returning the removed length, like osu!stable's draw step."""
    out: list[Point] = []
    start, removed, removed_total = None, 0.0, 0.0
    for i, p in enumerate(path):
        if start is None:
            out.append(p)
            start = p
            continue
        from_start = math.dist(start, p)
        removed += math.dist(path[i - 1], p)
        if from_start > 6 or (i + 1) % (CATMULL_DETAIL * 2) == 0 or i == len(path) - 1:
            out.append(p)
            removed_total += removed - from_start
            start, removed = None, 0.0
    return out, removed_total


def _sub_path(points: list[Point], kind: str) -> tuple[list[Point], float]:
    if kind == LINEAR:
        return list(points), 0.0
    if kind == PERFECT and len(points) == 3:
        arc = circular_arc(*points)
        if arc:
            return arc, 0.0
    if kind == CATMULL:
        return _optimise_catmull(catmull(points))
    degree = int(kind[1:]) if kind[1:] else len(points)
    return bspline(points, degree), 0.0


class SliderPath:
    """Ball position along a slider as a function of progress (0 = head, 1 = end of one slide)."""

    def __init__(self, curve_type: str, points: list[Point], length: float | None = None, version: int = 14):
        controls = []
        for seg in _segments(_path_type(curve_type), list(points), None, version):
            controls.extend((p, t) for p, t in seg)
        self._build(controls, length)

    @classmethod
    def from_osu(cls, spec: str, head: Point, length: float | None, version: int = 14) -> SliderPath:
        path = object.__new__(cls)
        path._build(control_points(spec, head, version), length)
        return path

    def _build(self, controls: list[tuple[Point, str | None]], expected: float | None) -> None:
        path: list[Point] = []
        optimised, start = 0.0, 0
        for i in range(len(controls)):
            if controls[i][1] is None and i < len(controls) - 1:
                continue
            seg = [p for p, _ in controls[start:i + 1]]
            if len(seg) == 1:
                path.append(seg[0])
            else:
                sub, removed = _sub_path(seg, controls[start][1] or LINEAR)
                optimised += removed
                path.extend(sub[1:] if path and sub and path[-1] == sub[0] else sub)
            start = i
        length = optimised
        cum = [0.0]
        for a, b in zip(path, path[1:]):
            length += math.dist(a, b)
            cum.append(length)
        if expected is not None and expected > 0 and length != expected:
            if len(path) >= 2 and path[-1] == path[-2] and expected > length:
                cum.append(length)  # osu!stable never extends a path ending on a repeated point
            else:
                cum.pop()
                end = len(path) - 1
                if length > expected:
                    while cum and cum[-1] >= expected:
                        cum.pop()
                        path.pop()
                        end -= 1
                if end <= 0:
                    cum.append(0.0)
                else:
                    (ax, ay), (bx, by) = path[end - 1], path[end]
                    seg = math.hypot(bx - ax, by - ay)
                    if seg > 0:
                        f = (expected - cum[-1]) / seg
                        path[end] = (ax + (bx - ax) * f, ay + (by - ay) * f)
                    cum.append(expected)
        if not path:
            path, cum = [(0.0, 0.0)], [0.0]
        self.points, self.cum = path, cum
        self.length = cum[-1]

    def shifted(self, dx: float, dy: float) -> SliderPath:
        """The same path moved by (dx, dy), e.g. for a stacking offset."""
        out = object.__new__(SliderPath)
        out.points = [(x + dx, y + dy) for x, y in self.points]
        out.cum, out.length = self.cum, self.length
        return out

    def at(self, progress: float) -> Point:
        d = min(max(progress, 0.0), 1.0) * self.length
        cum, pts = self.cum, self.points
        i = bisect.bisect_left(cum, d)
        if i <= 0:
            return pts[0]
        if i >= len(pts):
            return pts[-1]
        d0, d1 = cum[i - 1], cum[i]
        if abs(d1 - d0) <= 1e-7:
            return pts[i - 1]
        w = (d - d0) / (d1 - d0)
        (ax, ay), (bx, by) = pts[i - 1], pts[i]
        return (ax + (bx - ax) * w, ay + (by - ay) * w)

    @property
    def end(self) -> Point:
        return self.at(1.0)
