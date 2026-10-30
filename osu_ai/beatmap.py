"""Parse .osu files (osu!standard) into hit objects exactly as osu!lazer 2026.1005.0 loads them.

Follows LegacyBeatmapDecoder, ConvertHitObjectParser and OsuBeatmapProcessor: format v<5 maps are
shifted 24 ms, coordinates are truncated to integers, settings are clamped, objects are stably
sorted by time, a new combo starts after every break and spinner, slider ticks/repeats/tail come
from lazer's event generator (tail at the true end), and stacking uses lazer's thresholds.
"""

from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from . import rules
from .curves import FIRST_LAZER_VERSION, Point, SliderPath
from .rules import f32

EARLY_VERSION_OFFSET_MS = 24.0  # lazer shifts v4-and-older maps to undo an old client-side offset
STACK_DISTANCE = 3.0
MAX_TICK_LENGTH = 100000.0


class Kind(Enum):
    CIRCLE = "circle"
    SLIDER = "slider"
    SPINNER = "spinner"


class Nested(NamedTuple):
    """A slider tick, repeat or tail: judged by whether the slider is tracked when its time comes."""

    time: float
    kind: str  # "tick", "repeat" or "tail"


@dataclass
class HitObject:
    index: int
    kind: Kind
    x: float
    y: float
    time: float  # when it must be hit (slider head / spinner start)
    end_time: float  # == time for circles
    new_combo: bool  # lazer's NewCombo: restarts numbering and the health bonus (spinners keep their flag)
    combo_number: int = 1
    slides: int = 1
    path: SliderPath | None = None
    nested: list[Nested] = field(default_factory=list)  # slider ticks, repeats and tail, in time order
    last_in_combo: bool = False  # lazer's LastInCombo: hitting it earns the end-of-combo health bonus

    @property
    def pos(self) -> Point:
        return (self.x, self.y)

    @property
    def checkpoints(self) -> list[tuple[float, bool]]:
        """Slider ticks, repeats and tail as (time, is_tail); the tail is at the slider's true end."""
        return [(n.time, n.kind == "tail") for n in self.nested]

    def ball_at(self, t: float) -> Point:
        """Slider ball position at time t (the object's position for circles/spinners)."""
        duration = self.end_time - self.time
        if self.path is None or t <= self.time or duration <= 0:
            return self.pos
        progress = min((t - self.time) / duration, 1.0) * self.slides
        p = progress % 1
        return self.path.at(1 - p if int(progress) % 2 else p)


@dataclass(frozen=True)
class Difficulty:
    hp: float
    cs: float
    od: float
    ar: float
    slider_multiplier: float
    tick_rate: float

    @property
    def scale(self) -> float:
        return rules.circle_scale(self.cs)

    @property
    def radius(self) -> float:
        return rules.circle_radius(self.cs)

    @property
    def preempt(self) -> float:
        return rules.preempt_ms(self.ar)

    @property
    def fade_in(self) -> float:
        return rules.fade_in_ms(self.ar)

    @property
    def windows(self) -> rules.HitWindows:
        return rules.hit_windows(self.od)


@dataclass
class Beatmap:
    title: str
    artist: str
    version: str
    beatmap_id: int | None
    md5: str
    difficulty: Difficulty
    objects: list[HitObject]
    breaks: list[tuple[float, float]] = field(default_factory=list)  # (start, end) ms, sorted
    format_version: int = 14
    stack_leniency: float = 0.7

    @property
    def name(self) -> str:
        return f"{self.artist} - {self.title} [{self.version}]"


def _sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], [])
        elif current is not None:
            current.append(line)
    return sections


def _key_values(lines: list[str]) -> dict[str, str]:
    return {k.strip(): v.strip() for k, _, v in (ln.partition(":") for ln in lines)}


def _format_version(text: str) -> int:
    first = text.lstrip().split("\n", 1)[0].strip()
    return int(first.rsplit("v", 1)[1]) if first.startswith("osu file format v") else 14


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def _difficulty(kv: dict[str, str]) -> Difficulty:
    od = float(kv.get("OverallDifficulty", 5))
    return Difficulty(
        hp=_clamp(float(kv.get("HPDrainRate", 5)), 0, 10),
        cs=_clamp(float(kv.get("CircleSize", 5)), 0, 10),
        od=_clamp(od, 0, 10),
        ar=_clamp(float(kv.get("ApproachRate", od)), 0, 10),  # old maps tie AR to OD
        slider_multiplier=_clamp(float(kv.get("SliderMultiplier", 1.4)), 0.4, 3.6),
        tick_rate=_clamp(float(kv.get("SliderTickRate", 1)), 0.5, 8),
    )


class _Timing:
    """Beat length (red lines) and slider velocity / tick generation (green lines, reset by red lines),
    resolved like lazer's decoder: at one time the first red line and the last green line win."""

    def __init__(self, lines: list[str], offset: float):
        beats: dict[float, float] = {}
        diffs: dict[float, tuple[float, bool]] = {}
        group: list[tuple[float, float, bool]] = []  # consecutive lines at one time

        def flush() -> None:
            if not group:
                return
            reds = [g for g in group if g[2]]
            greens = [g for g in group if not g[2]]
            if reds:
                beats[group[0][0]] = _clamp(reds[0][1], 6, 60000)
            _, beat, _ = greens[-1] if greens else reds[0]
            sv = 100.0 / -beat if beat < 0 else 1.0
            diffs[group[0][0]] = (_clamp(sv, 0.1, 10.0), not math.isnan(beat))
            group.clear()

        for line in lines:
            parts = line.split(",")
            try:
                t, beat = float(parts[0]) + offset, float(parts[1])
            except (ValueError, IndexError):
                continue
            red = len(parts) < 7 or parts[6][:1] == "1"
            if red and math.isnan(beat):
                continue
            if group and group[0][0] != t:
                flush()
            group.append((t, beat, red))
        flush()
        self._beat_times = sorted(beats)
        self._beats = [beats[t] for t in self._beat_times]
        self._diff_times = sorted(diffs)
        self._diffs = [diffs[t] for t in self._diff_times]

    def beat_length(self, t: float) -> float:
        if not self._beats:
            return 1000.0
        return self._beats[max(bisect.bisect_right(self._beat_times, t) - 1, 0)]

    def difficulty(self, t: float) -> tuple[float, bool]:
        """(slider velocity multiplier, whether ticks are generated) at time t."""
        i = bisect.bisect_right(self._diff_times, t) - 1
        return self._diffs[i] if i >= 0 else (1.0, True)


def _slider_events(start: float, spans: int, distance: float, diff: Difficulty, timing: _Timing,
                   version: int) -> tuple[float, list[Nested]]:
    """(end time, ticks/repeats/tail) as lazer's Slider + SliderEventGenerator compute them."""
    beat = timing.beat_length(start)
    sv, ticks_on = timing.difficulty(start)
    velocity = 100 * diff.slider_multiplier / (beat * (_clamp(f32(100.0 / sv), 10, 1000) / 100.0))
    end = start + spans * distance / velocity
    span = (end - start) / spans
    tick_dist = velocity * beat / diff.tick_rate * (1 / sv if version < 8 else 1) if ticks_on else math.inf
    length = min(MAX_TICK_LENGTH, distance)
    tick_dist = _clamp(tick_dist, 0.0, length)
    min_from_end = velocity * 10
    nested: list[Nested] = []
    for k in range(spans):
        span_start = start + k * span
        if tick_dist != 0:
            ticks = []
            d = tick_dist
            while d <= length and d < length - min_from_end:
                p = d / length
                ticks.append(Nested(span_start + (1 - p if k % 2 else p) * span, "tick"))
                d += tick_dist
            nested += ticks[::-1] if k % 2 else ticks
        if k < spans - 1:
            nested.append(Nested(start + (k + 1) * span, "repeat"))
    nested.append(Nested(start + spans * span, "tail"))
    return end, nested


def _coord(s: str, version: int) -> float:
    v = f32(float(s))
    return v if version >= FIRST_LAZER_VERSION else float(int(v))


def _hit_object(parts: list[str], offset: float, version: int, diff: Difficulty, timing: _Timing,
                last: HitObject | None) -> HitObject | None:
    x, y = _coord(parts[0], version), _coord(parts[1], version)
    t, type_bits = float(parts[2]) + offset, int(parts[3])
    # The first object, and the first after a spinner, start a new combo (spinners keep their own bit).
    new_combo = bool(type_bits & 4) or last is None or last.kind is Kind.SPINNER
    if type_bits & 1:
        return HitObject(0, Kind.CIRCLE, x, y, t, t, new_combo)
    if type_bits & 2:
        repeats = int(parts[6])
        if repeats > 9000:
            raise ValueError("repeat count is way too high")
        repeats = max(0, repeats - 1)
        length = max(0.0, float(parts[7])) if len(parts) > 7 and parts[7] else 0.0
        path = SliderPath.from_osu(parts[5], (x, y), length or None, version)
        if abs(path.length) <= 1e-7:
            repeats = 0  # lazer drops repeats of zero-length sliders
        end, nested = _slider_events(t, repeats + 1, path.length, diff, timing, version)
        return HitObject(0, Kind.SLIDER, x, y, t, end, new_combo, slides=repeats + 1, path=path, nested=nested)
    if type_bits & 8:
        end = t + max(0.0, float(parts[5]) + offset - t)
        return HitObject(0, Kind.SPINNER, *rules.SPINNER_CENTER, t, end, bool(type_bits & 4))
    return None  # osu!mania holds


def _hit_objects(lines: list[str], offset: float, version: int, diff: Difficulty, timing: _Timing) -> list[HitObject]:
    objects: list[HitObject] = []
    for line in lines:
        try:
            obj = _hit_object(line.split(","), offset, version, diff, timing, objects[-1] if objects else None)
        except (ValueError, IndexError):
            continue  # lazer skips a line it cannot parse
        if obj is not None:
            objects.append(obj)
    objects.sort(key=lambda o: o.time)  # stable, like lazer's OrderBy
    for i, o in enumerate(objects):
        o.index = i
    return objects


def _breaks(lines: list[str], offset: float) -> list[tuple[float, float]]:
    out = []
    for line in lines:
        parts = line.split(",")
        if parts[0].strip() in ("2", "Break") and len(parts) > 2:
            start = float(parts[1]) + offset
            out.append((start, max(start, float(parts[2]) + offset)))
    return sorted(out)


def _apply_combos(objects: list[HitObject], breaks: list[tuple[float, float]]) -> None:
    """New combos after breaks (LegacyBeatmapDecoder), then numbering and LastInCombo (OsuBeatmapProcessor)."""
    b = 0
    for o in objects:
        while b < len(breaks) and breaks[b][1] < o.time:
            o.new_combo = True
            b += 1
    prev, number = None, 0
    for o in objects:
        if o.kind is not Kind.SPINNER and (prev is None or prev.kind is Kind.SPINNER):
            o.new_combo = True
        if o.kind is not Kind.SPINNER and o.new_combo:
            number = 1
            if prev is not None:
                prev.last_in_combo = True
        else:
            number += 1
        o.combo_number = number
        prev = o


def _stack_heights(objects: list[HitObject], threshold: float, version: int) -> list[int]:
    """osu!'s stacking (lazer OsuBeatmapProcessor): objects at (almost) the same spot close in
    time are drawn and judged a little up-left of each other, one level per object."""
    heights = [0] * len(objects)
    pos = [o.pos for o in objects]
    dist = math.dist
    if version < 6:
        for i, cur in enumerate(objects):
            if heights[i] != 0 and cur.kind is not Kind.SLIDER:
                continue
            start, slider_stack = cur.end_time, 0
            path_end = cur.path.at(1.0) if cur.path is not None else pos[i]
            for j in range(i + 1, len(objects)):
                if objects[j].time - threshold > start:
                    break
                if dist(pos[j], pos[i]) < STACK_DISTANCE:
                    heights[i] += 1
                    start = objects[j].time
                elif dist(pos[j], path_end) < STACK_DISTANCE:
                    slider_stack += 1  # under a slider's end: stack down-right instead
                    heights[j] -= slider_stack
                    start = objects[j].time
        return heights
    end_pos = [o.ball_at(o.end_time) for o in objects]
    for i in range(len(objects) - 1, 0, -1):
        if heights[i] != 0 or objects[i].kind is Kind.SPINNER:
            continue
        cur, n = i, i
        if objects[i].kind is Kind.CIRCLE:
            while (n := n - 1) >= 0:
                other = objects[n]
                if other.kind is Kind.SPINNER:
                    continue
                if int(objects[cur].time) - int(other.end_time) > threshold:
                    break
                if other.kind is Kind.SLIDER and dist(end_pos[n], pos[cur]) < STACK_DISTANCE:
                    # Circles stacked on a slider's end go down-right of it rather than up-left.
                    offset = heights[cur] - heights[n] + 1
                    for j in range(n + 1, i + 1):
                        if dist(end_pos[n], pos[j]) < STACK_DISTANCE:
                            heights[j] -= offset
                    break
                if dist(pos[n], pos[cur]) < STACK_DISTANCE:
                    heights[n] = heights[cur] + 1
                    cur = n
        else:
            while (n := n - 1) >= 0:
                other = objects[n]
                if other.kind is Kind.SPINNER:
                    continue
                if objects[cur].time - other.time > threshold:
                    break
                if dist(end_pos[n], pos[cur]) < STACK_DISTANCE:
                    heights[n] = heights[cur] + 1
                    cur = n
    return heights


def parse(path: str | Path) -> Beatmap:
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    version = _format_version(text)
    offset = EARLY_VERSION_OFFSET_MS if version < 5 else 0.0
    sec = _sections(text)
    general = _key_values(sec.get("General", []))
    if general.get("Mode", "0") != "0":
        raise ValueError(f"{path}: not an osu!standard map (Mode {general['Mode']})")
    meta = _key_values(sec.get("Metadata", []))
    diff = _difficulty(_key_values(sec.get("Difficulty", [])))
    timing = _Timing(sec.get("TimingPoints", []), offset)
    objects = _hit_objects(sec.get("HitObjects", []), offset, version, diff, timing)
    breaks = _breaks(sec.get("Events", []), offset)
    _apply_combos(objects, breaks)

    leniency = f32(float(general.get("StackLeniency", 0.7)))
    threshold = f32(int(diff.preempt) * leniency)
    scale = diff.scale
    for obj, height in zip(objects, _stack_heights(objects, threshold, version)):
        if height and obj.kind is not Kind.SPINNER:
            shift = f32(f32(height * scale) * f32(rules.STACK_OFFSET))
            obj.x, obj.y = obj.x + shift, obj.y + shift
            if obj.path is not None:
                obj.path = obj.path.shifted(shift, shift)

    return Beatmap(
        title=meta.get("Title", "?"),
        artist=meta.get("Artist", "?"),
        version=meta.get("Version", "?"),
        beatmap_id=int(meta["BeatmapID"]) if meta.get("BeatmapID", "").isdigit() else None,
        md5=hashlib.md5(raw).hexdigest(),
        difficulty=diff,
        objects=objects,
        breaks=breaks,
        format_version=version,
        stack_leniency=leniency,
    )
