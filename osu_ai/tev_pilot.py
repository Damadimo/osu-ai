"""Live pilot backed by the local fine-tuned decision model (osu_ai.tev).

Each snapshot is one screen description shared by every question, so the model
reads that prefix once and then answers each object's question from a copy of
the cached prefix. Inference runs on a worker thread so the window stays smooth.
"""

from __future__ import annotations

import asyncio
import copy
import random
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from .dataset import ACTION_OPTIONS, HORIZON_MS, STEP_MS, TIMING_OPTIONS, screen_state
from .hands import Action
from .pilot import Plan, Request
from .tev import TevModel


class TevPilot:
    def __init__(self, bm, model: str, adapter: str | None, name: str = "tev"):
        self.bm = bm
        self.tev = TevModel(model, adapter)
        self.name = name
        self._pool = ThreadPoolExecutor(max_workers=1)  # MLX work stays on one thread
        self._actions: dict[int, Action] = {}
        self._rng = random.Random(0)

    def _answer_all(self, prompts: list[list[int]], options: list[list[dict]]) -> list[str]:
        """Answer questions that share a long prompt prefix with one prefix pass."""
        n = min(len(p) for p in prompts)
        common = next((i for i in range(n) if any(p[i] != prompts[0][i] for p in prompts)), n)
        common = max(common - 1, 1)  # leave at least one token to feed per question
        cache = make_prompt_cache(self.tev.model)
        self.tev.model(mx.array(prompts[0][:common])[None], cache=cache)
        mx.eval([c.state for c in cache])
        answers = []
        for tokens, opts in zip(prompts, options):
            c = copy.deepcopy(cache)
            logits = self.tev.model(mx.array(tokens[common:])[None], cache=c)[0, -1]
            ids = mx.array([self.tev._letter(o["label"]) for o in opts])
            best = int(mx.argmax(logits[ids]).item())
            answers.append(opts[best]["key"])
        return answers

    def _plan_sync(self, req: Request, bm) -> list[Plan]:
        pending = list(req.labels.values())
        state, labels = screen_state(bm, req.t, self._rng, pending[0])
        records, targets = [], []
        for o in pending:
            if o.index not in labels:
                continue
            letter = labels[o.index]
            if o.index not in self._actions:
                records.append({"state": state, "options": ACTION_OPTIONS,
                                "question": f"How should object {letter} be played?"})
                targets.append(("action", o))
            if o.kind.value != "spinner":
                records.append({"state": state, "options": TIMING_OPTIONS,
                                "question": f"How long from now until the ring of object {letter} touches its circle?"})
                targets.append(("timing", o))
        if not records:
            return []
        keys = self._answer_all([self.tev.prompt_tokens(r) for r in records], [r["options"] for r in records])
        timing = {}
        for (kind, o), key in zip(targets, keys):
            if kind == "action":
                self._actions[o.index] = Action(key)
            else:
                timing[o.index] = key
        plans = []
        for o in pending:
            action = self._actions.get(o.index, Action.TAP)
            truth = o.time - req.t
            if o.kind.value == "spinner":
                plans.append(Plan(o.index, action, o.time, "spinner", req.t, truth))
            elif o.index in timing:
                key = timing[o.index]
                press = None if key == "later" else req.t + int(key.split("-")[0]) + STEP_MS / 2
                plans.append(Plan(o.index, action, press, key, req.t, truth, HORIZON_MS))
        return plans

    async def plan(self, req: Request) -> tuple[list[Plan], None]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self._plan_sync, req, self.bm), None
