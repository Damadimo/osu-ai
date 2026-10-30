"""Brains decide what to do with each object as it appears on screen."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from .beatmap import Beatmap, Kind
from .hands import CORRECT_ACTION, Action
from .systemone import SystemOneClient
from .view import Snapshot


@dataclass
class Answer:
    action: Action
    latency_ms: float = 0.0
    confidence: float | None = None
    telemetry: dict = field(default_factory=dict)


class Brain(Protocol):
    name: str

    async def decide(self, snap: Snapshot, bm: Beatmap) -> Answer: ...


class AutoBrain:
    """Knows the right answer instantly. Upper bound for the pipeline."""

    name = "auto"

    async def decide(self, snap: Snapshot, bm: Beatmap) -> Answer:
        return Answer(CORRECT_ACTION[snap.new.kind])


class DelayedAutoBrain:
    """Right answers, but each one arrives after a latency drawn from `latencies_ms`."""

    def __init__(self, latencies_ms: list[float], seed: int = 0):
        self.name = f"auto+{int(sum(latencies_ms) / len(latencies_ms))}ms"
        self._latencies = latencies_ms
        self._rng = random.Random(seed)

    async def decide(self, snap: Snapshot, bm: Beatmap) -> Answer:
        return Answer(CORRECT_ACTION[snap.new.kind], latency_ms=self._rng.choice(self._latencies))


ACTION_CRITERIA = {
    "tap": "a plain numbered circle with nothing attached: click it once",
    "hold": "a numbered circle with a track attached: press it and follow the ball along the track",
    "spin": "a spinner: hold and spin the cursor around the middle of the screen",
}


class JevBrain:
    """Asks a System One model (Jev or a local clone) how to play each new object."""

    def __init__(self, client: SystemOneClient, name: str = "jev"):
        self.client = client
        self.name = name

    async def decide(self, snap: Snapshot, bm: Beatmap) -> Answer:
        new_label = snap.label(snap.new)
        questions = {
            "action": {
                "type": "choice",
                "instructions": f"How should object `{new_label}` be played?",
                "criteria": ACTION_CRITERIA,
            }
        }
        # Telemetry only: can the model read which object comes first?
        pending = [o for o in snap.visible if o.time >= snap.t and o.kind is not Kind.SPINNER]
        if len(pending) > 1:
            questions["first"] = {
                "type": "choice",
                "instructions": "Which object has to be hit first? It is the one whose approach ring is closest "
                                "to touching its circle; among equal rings, the lower number goes first.",
                "criteria": {snap.label(o): None for o in pending},
            }
        decision = await self.client.decide(snap.state(bm.difficulty.preempt), questions)
        action_answer = decision.answers["action"]
        telemetry = {"usage": decision.usage}
        if "first" in decision.answers:
            truth = snap.label(min(pending, key=lambda o: (o.time, o.index)))
            telemetry["first_correct"] = decision.answers["first"]["choice"] == truth
        return Answer(
            Action(action_answer["choice"]),
            latency_ms=decision.latency_ms,
            confidence=action_answer.get("confidence"),
            telemetry=telemetry,
        )
