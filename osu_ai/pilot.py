"""Real-time protocol: the model decides when and how to hit every object.

Each tick the model sees the screen as a player would: where objects are, how
big each approach ring is, and how fast rings shrink. For every object still to
be hit it picks when that ring will touch its circle, in 50 ms buckets counted
from the moment of the snapshot, and how to play it. The hands press exactly
then, on that object. Code does no timing: a misread is a mistimed click or a
miss. (Spinners have no ring; the model only decides whether to spin them.)
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from .beatmap import Beatmap, HitObject, Kind
from .brains import ACTION_CRITERIA
from .hands import CORRECT_ACTION, Action
from .systemone import Decision, SystemOneClient
from .view import looks_like, region

BUCKET_MS = 50
HORIZON_MS = 600
LATER = "later"
WHEN_CRITERIA: dict[str, str | None] = {f"{a}-{a + BUCKET_MS}ms": None for a in range(0, HORIZON_MS, BUCKET_MS)}
WHEN_CRITERIA[LATER] = f"more than {HORIZON_MS} ms from now"


def ring_size(obj: HitObject, t: float, preempt: float) -> float:
    """Approach ring size relative to its circle: 4x when it appears, 1x at the hit time."""
    return 1 + 3 * max(obj.time - t, 0) / preempt


@dataclass(frozen=True)
class Plan:
    index: int
    action: Action
    press_at: float | None  # game ms, or None when the model said "later"
    bucket: str
    asked_at: float
    truth_ms: float  # real time from the snapshot to the hit, to grade the read
    horizon_ms: float = HORIZON_MS  # beyond this the model answers "later"

    @property
    def read_correct(self) -> bool:
        if self.bucket == LATER:
            return self.truth_ms >= self.horizon_ms
        if self.bucket == "spinner":
            return True
        lo, hi = (int(x) for x in self.bucket.removesuffix("ms").split("-"))
        return lo <= self.truth_ms < hi


@dataclass
class Request:
    t: float
    state: dict
    questions: dict
    labels: dict[str, HitObject]


def build_request(bm: Beatmap, t: float, pending: list[HitObject], seed: int) -> Request:
    preempt = bm.difficulty.preempt
    letters = random.Random(seed).sample("ABCDEFGHJKMNPQRSTUVWXYZ", k=min(len(pending), 23))
    labels = dict(zip(letters, pending))
    objects = {}
    for label, o in labels.items():
        objects[label] = {"looks_like": looks_like(o), "where": region(o.pos)}
        if o.kind is not Kind.SPINNER:
            objects[label]["ring_size"] = f"{ring_size(o, t, preempt):.2f}x the circle"
    state = {
        "game": "osu!: click each circle at the moment its shrinking approach ring touches it",
        "approach_rings": f"every ring appears at 4.00x its circle's size and shrinks at a steady speed "
                          f"to 1.00x over {preempt / 1000:.2f} seconds; it touches the circle at 1.00x",
        "objects_on_screen": objects,
    }
    questions = {}
    for label, o in labels.items():
        if o.kind is not Kind.SPINNER:
            questions[f"when_{label}"] = {
                "type": "choice",
                "instructions": f"How long from now until the approach ring of object `{label}` touches its circle?",
                "criteria": WHEN_CRITERIA,
            }
        questions[f"how_{label}"] = {
            "type": "choice",
            "instructions": f"How should object `{label}` be played?",
            "criteria": ACTION_CRITERIA,
        }
    return Request(t, state, questions, labels)


def read_plans(req: Request, answers: dict) -> list[Plan]:
    plans = []
    for label, o in req.labels.items():
        action = Action(answers[f"how_{label}"]["choice"])
        if o.kind is Kind.SPINNER:
            plans.append(Plan(o.index, action, o.time, "spinner", req.t, o.time - req.t))
            continue
        bucket = answers[f"when_{label}"]["choice"]
        press_at = None if bucket == LATER else req.t + int(bucket.split("-")[0]) + BUCKET_MS / 2
        plans.append(Plan(o.index, action, press_at, bucket, req.t, o.time - req.t))
    return plans


class ModelPilot:
    """Any System One backend: Jev via Vercel/TypeSafe, or a local Laya server."""

    def __init__(self, client: SystemOneClient, name: str):
        self.client, self.name = client, name

    async def plan(self, req: Request) -> tuple[list[Plan], Decision]:
        decision = await self.client.decide(req.state, req.questions)
        return read_plans(req, decision.answers), decision


class OraclePilot:
    """Correct reads after a fixed delay. Only for checking the harness itself."""

    def __init__(self, latency_ms: float):
        self.latency_ms, self.name = latency_ms, f"oracle+{latency_ms:g}ms"

    async def plan(self, req: Request) -> tuple[list[Plan], None]:
        await asyncio.sleep(self.latency_ms / 1000)
        plans = []
        for o in req.labels.values():
            truth = o.time - req.t
            if o.kind is Kind.SPINNER:
                plans.append(Plan(o.index, Action.SPIN, o.time, "spinner", req.t, truth))
            elif truth >= HORIZON_MS:
                plans.append(Plan(o.index, CORRECT_ACTION[o.kind], None, LATER, req.t, truth))
            else:
                lo = int(max(truth, 0) // BUCKET_MS) * BUCKET_MS
                plans.append(Plan(o.index, CORRECT_ACTION[o.kind], req.t + lo + BUCKET_MS / 2,
                                  f"{lo}-{lo + BUCKET_MS}ms", req.t, truth))
        return plans, None
