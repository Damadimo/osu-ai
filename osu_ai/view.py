"""What a player sees at a moment in time, described in words for a System One model.

Descriptions are visual ("a numbered circle with a track"), not the object's
type, and approach rings are bucketed words rather than millisecond counts:
Jev reads meaning well and numbers poorly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .beatmap import Beatmap, HitObject, Kind
from .curves import Point

_REGIONS = {
    ("top", "left"): "top left", ("top", "center"): "top", ("top", "right"): "top right",
    ("middle", "left"): "left", ("middle", "center"): "center", ("middle", "right"): "right",
    ("bottom", "left"): "bottom left", ("bottom", "center"): "bottom", ("bottom", "right"): "bottom right",
}


def region(p: Point) -> str:
    col = "left" if p[0] < 512 / 3 else "right" if p[0] > 1024 / 3 else "center"
    row = "top" if p[1] < 384 / 3 else "bottom" if p[1] > 768 / 3 else "middle"
    return _REGIONS[(row, col)]


def ring(remaining: float) -> str:
    """Approach ring state from the fraction of approach time left."""
    if remaining <= 0:
        return "ring gone, being played now"
    if remaining < 0.2:
        return "ring almost touching the circle"
    if remaining < 0.4:
        return "ring mostly closed"
    if remaining < 0.6:
        return "ring about half closed"
    if remaining < 0.8:
        return "ring large"
    return "ring very large, just appeared"


def looks_like(obj: HitObject) -> str:
    if obj.kind is Kind.SPINNER:
        return "a big spinner disc covering the middle of the screen"
    if obj.kind is Kind.SLIDER:
        arrow = ", with a reverse arrow at the end" if obj.slides > 1 else ""
        return f"circle numbered {obj.combo_number} with a track leading to the {region(obj.path.end)}{arrow}"
    return f"circle numbered {obj.combo_number}"


@dataclass
class Snapshot:
    t: float
    new: HitObject  # the object that just appeared
    visible: list[HitObject]
    labels: dict[int, str]  # object index -> label shown to the model

    def label(self, obj: HitObject) -> str:
        return self.labels[obj.index]

    def state(self, preempt: float) -> dict:
        return {
            "game": "osu!, a rhythm game: click each circle when its shrinking approach ring touches it",
            "objects_on_screen": {
                self.labels[o.index]: {
                    "looks_like": looks_like(o),
                    "where": region(o.pos),
                    "approach": "spinning" if o.kind is Kind.SPINNER and o.time <= self.t
                    else ring((o.time - self.t) / preempt),
                }
                for o in self.visible
            },
        }


def snapshot(bm: Beatmap, new: HitObject, t: float, seed: int = 0) -> Snapshot:
    preempt = bm.difficulty.preempt
    visible = [o for o in bm.objects if o.time - preempt <= t <= o.end_time]
    if new not in visible:
        visible.append(new)
    # Shuffled letters so the label order never hints at the hit order.
    letters = random.Random(seed * 100_003 + new.index).sample("ABCDEFGHJKMNPQRSTUVWXYZ", k=min(len(visible), 23))
    labels = {o.index: letters[i] for i, o in enumerate(visible[:23])}
    visible = [o for o in visible if o.index in labels]
    return Snapshot(t, new, visible, labels)
