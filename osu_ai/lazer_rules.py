"""What a judgement is worth in osu!lazer 2026.1005.0 with default mods: accuracy, combo, score, rank, health.

Ports of ScoreProcessor/OsuScoreProcessor and OsuHealthProcessor/DrainingHealthProcessor. Judgements
use lazer's names: results ("Great", "LargeTickMiss", ...) and the class of what was judged
("SliderTick", ...), so a judgement log reads like lazer's own and diffs cleanly against the oracle.
"""

from __future__ import annotations

import math
from collections import Counter

from .beatmap import HitObject, Kind
from .rules import difficulty_range, f32

GREAT, OK, MEH, MISS = "Great", "Ok", "Meh", "Miss"
TICK_HIT, TICK_MISS = "LargeTickHit", "LargeTickMiss"
TAIL_HIT = "SliderTailHit"
IGNORE_HIT, IGNORE_MISS = "IgnoreHit", "IgnoreMiss"
SMALL_BONUS, LARGE_BONUS = "SmallBonus", "LargeBonus"

CIRCLE, HEAD, TICK, REPEAT, TAIL = "HitCircle", "SliderHeadCircle", "SliderTick", "SliderRepeat", "SliderTailCircle"
SLIDER, SPINNER, SPIN_TICK, SPIN_BONUS = "Slider", "Spinner", "SpinnerTick", "SpinnerBonusTick"
NESTED_KIND = {"tick": TICK, "repeat": REPEAT, "tail": TAIL}

MAX_RESULT = {CIRCLE: GREAT, HEAD: GREAT, SPINNER: GREAT, TICK: TICK_HIT, REPEAT: TICK_HIT, TAIL: TAIL_HIT,
              SLIDER: IGNORE_HIT, SPIN_TICK: SMALL_BONUS, SPIN_BONUS: LARGE_BONUS}
MIN_RESULT = {kind: {GREAT: MISS, TICK_HIT: TICK_MISS}.get(best, IGNORE_MISS) for kind, best in MAX_RESULT.items()}
BASE_SCORE = {GREAT: 300, OK: 100, MEH: 50, TICK_HIT: 30, TAIL_HIT: 150, SMALL_BONUS: 10, LARGE_BONUS: 50}
JUDGMENT = {GREAT: 300, OK: 100, MEH: 50, MISS: 0}  # a circle-like result as the classic number
FROM_JUDGMENT = {300: GREAT, 100: OK, 50: MEH, 0: MISS}

_MISSES = {MISS, TICK_MISS, IGNORE_MISS}
_AFFECTS_COMBO = {GREAT, OK, MEH, MISS, TICK_HIT, TICK_MISS, TAIL_HIT}
_BONUS = {SMALL_BONUS, LARGE_BONUS}
_SCORABLE = {MISS, MEH, OK, GREAT, TICK_MISS, TICK_HIT, SMALL_BONUS, LARGE_BONUS, TAIL_HIT}
COMBO_EXPONENT = 0.5
FAIL_HEALTH = 1e-7  # Precision.AlmostBigger(0, health)


def is_hit(result: str) -> bool:
    return result not in _MISSES


def affects_accuracy(result: str) -> bool:
    return result in _SCORABLE and result not in _BONUS


def rank(accuracy: float, misses: int, failed: bool = False) -> str:
    if failed:
        return "F"
    grade = ("X" if accuracy == 1 else "S" if accuracy >= 0.95 else "A" if accuracy >= 0.9
             else "B" if accuracy >= 0.8 else "C" if accuracy >= 0.7 else "D")
    return "A" if grade in ("S", "X") and misses else grade  # a single Miss (not a tick miss) costs the S


def perfect_play(objects: list[HitObject]) -> list[tuple[float, str, str]]:
    """(time, kind, result) of an autoplay in lazer's simulation order: each object's nested judgements,
    then its own. Spinner ticks are left out: they are bonus, which neither drain nor max combo counts."""
    out = []
    for o in objects:
        if o.kind is Kind.CIRCLE:
            out.append((o.time, CIRCLE, GREAT))
        elif o.kind is Kind.SLIDER:
            out.append((o.time, HEAD, GREAT))
            out += [(n.time, NESTED_KIND[n.kind], MAX_RESULT[NESTED_KIND[n.kind]]) for n in o.nested]
            out.append((o.end_time, SLIDER, IGNORE_HIT))
        else:
            out.append((o.end_time, SPINNER, GREAT))
    return out


# ---- health ----

def miss_health(hp: float) -> float:
    return difficulty_range(f32(hp), -0.03, -0.125, -0.2)


def tick_miss_health(hp: float) -> float:
    return difficulty_range(f32(hp), -0.02, -0.075, -0.14)


def health_increase(kind: str, result: str, hp: float) -> float:
    """OsuHealthProcessor.getHealthIncreaseFor, before the end-of-combo bonus."""
    if result == TICK_MISS:
        return tick_miss_health(hp)
    if result == MISS:
        return miss_health(hp)
    if result in (TICK_HIT, TAIL_HIT):
        return 0.015 if kind == TICK else 0.02
    return {MEH: 0.002, OK: 0.011, GREAT: 0.03, SMALL_BONUS: 0.0085, LARGE_BONUS: 0.01}.get(result, 0.0)


def drain_rate(objects: list[HitObject], breaks: list[tuple[float, float]], hp: float) -> float:
    """DrainingHealthProcessor.ComputeDrainRate: the health per ms that a perfect play drains down to
    DR(HP; 0.99, 0.9, 0.4) at its lowest, found by binary search."""
    increases = [(t, health_increase(kind, result, hp)) for t, kind, result in perfect_play(objects)]
    if len(increases) <= 1:
        return 0.0
    target = min(max(difficulty_range(f32(hp), 0.99, 0.9, 0.4), 0.0), 1.0)
    start = objects[0].time
    adjustment, rate = 1, 1.0
    while adjustment > 0:
        current = lowest = 1.0
        last, b = start, 0
        for t, amount in increases:
            while b < len(breaks) and breaks[b][1] <= t:
                last = t  # no drain across a break
                b += 1
            current -= (t - last) * rate
            lowest = min(lowest, current)
            current = min(1.0, current + amount)
            last = t
            if lowest < 0:
                break
        if abs(lowest - target) <= 0.01:
            break
        adjustment *= 2
        if adjustment >= 2 ** 31:  # C#'s int overflows here and ends the search
            adjustment -= 2 ** 32
        rate += 1.0 / adjustment * ((lowest > target) - (lowest < target))
    return rate


def no_drain_periods(objects: list[HitObject], breaks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Each break stops drain from the end of the last object before it to the start of the first after it."""
    out = []
    for start, end in breaks:
        before = [o.end_time for o in objects if o.end_time <= start]
        after = [o.time for o in objects if o.time >= end]
        out.append((before[-1] if before else -math.inf, after[0] if after else math.inf))
    return out


def _union(periods: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Overlapping periods merged: lazer only asks whether a moment is in any of them, so time covered
    by two (breaks with no object between them) must not count twice."""
    out: list[tuple[float, float]] = []
    for lo, hi in sorted(periods):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


class Health:
    """OsuHealthProcessor in live play: drain between judgements, each judgement's change including the
    end-of-combo bonus, and the fail check (health below 1e-7 right after a non-bonus judgement).
    Play goes on after a fail, as with NoFail; failed_at is when lazer would have failed."""

    PERFECT, GOOD, NONE = 2, 1, 0

    def __init__(self, objects: list[HitObject], breaks: list[tuple[float, float]], hp: float):
        self.hp = hp
        self.rate = drain_rate(objects, breaks, hp)
        self.start = objects[0].time if objects else 0.0
        self.end = objects[-1].end_time if objects else 0.0
        self.pauses = _union(no_drain_periods(objects, breaks))
        self.value = self.lowest = 1.0
        self.failed_at: float | None = None
        self._t = -math.inf
        self._combo_result = self.PERFECT

    def drain_to(self, t: float) -> None:
        a, b = max(self._t, self.start), min(t, self.end)
        self._t = max(self._t, t)
        if b <= a or not self.rate:
            return
        span = b - a - sum(max(0.0, min(b, hi) - max(a, lo)) for lo, hi in self.pauses)
        self.value = max(0.0, self.value - self.rate * span)
        self.lowest = min(self.lowest, self.value)

    def increase(self, kind: str, result: str, new_combo: bool = False, last_in_combo: bool = False) -> float:
        """The change a judgement makes; updates the combo's worst result for the end-of-combo bonus."""
        if new_combo:
            self._combo_result = self.PERFECT
        if result in (TICK_MISS, OK):
            self._combo_result = min(self._combo_result, self.GOOD)
        elif result in (MEH, MISS):
            self._combo_result = self.NONE
        if kind == TAIL and not is_hit(result):
            self._combo_result = min(self._combo_result, self.GOOD)
        change = health_increase(kind, result, self.hp)
        if last_in_combo and is_hit(result):
            change += (0.03, 0.05, 0.07)[self._combo_result]
        return change

    def apply(self, t: float, kind: str, result: str, new_combo: bool = False, last_in_combo: bool = False) -> None:
        self.drain_to(t)
        self.value = min(max(self.value + self.increase(kind, result, new_combo, last_in_combo), 0.0), 1.0)
        self.lowest = min(self.lowest, self.value)
        bonus = MAX_RESULT[kind] in _BONUS
        if self.failed_at is None and not bonus and result != IGNORE_HIT and self.value < FAIL_HEALTH:
            self.failed_at = t


# ---- score ----

class Scoring:
    """ScoreProcessor: combo, statistics, accuracy and standardised score as judgements arrive."""

    def __init__(self, objects: list[HitObject]):
        self.combo = self.max_combo = 0
        self.statistics: Counter = Counter()
        self._base = self._max_base = 0
        self._accuracy_judged = 0
        self._combo_portion = self._bonus = 0.0
        combo, self._max_combo_portion, self._max_accuracy_judged = 0, 0.0, 0
        for _, kind, result in perfect_play(objects):
            if result in _AFFECTS_COMBO:
                combo += 1
            if affects_accuracy(result):
                self._max_accuracy_judged += 1
            if result in _SCORABLE:
                self._max_combo_portion += BASE_SCORE[result] * combo ** COMBO_EXPONENT

    def apply(self, kind: str, result: str) -> None:
        self.statistics[result] += 1
        if result in _AFFECTS_COMBO:
            self.combo = self.combo + 1 if is_hit(result) else 0
            self.max_combo = max(self.max_combo, self.combo)
        best = MAX_RESULT[kind]
        if affects_accuracy(best):
            self._max_base += BASE_SCORE[best]
            self._accuracy_judged += 1
        if affects_accuracy(result):
            self._base += BASE_SCORE.get(result, 0)
        if result in _BONUS:
            self._bonus += BASE_SCORE[result]
        elif result in _SCORABLE:
            self._combo_portion += BASE_SCORE[best] * self.combo ** COMBO_EXPONENT

    @property
    def judged(self) -> bool:
        return self._accuracy_judged > 0

    @property
    def accuracy(self) -> float:
        return self._base / self._max_base if self._max_base else 1.0

    @property
    def total_score(self) -> int:
        """lazer's standardised score (no mods, so no multiplier)."""
        acc = self.accuracy
        combo = self._combo_portion / self._max_combo_portion if self._max_combo_portion else 1.0
        judged = self._accuracy_judged / self._max_accuracy_judged if self._max_accuracy_judged else 1.0
        return round(500000 * acc * combo + 500000 * acc ** 5 * judged + self._bonus)
