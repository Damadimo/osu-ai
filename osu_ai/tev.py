"""A Jev-style decision model running locally with MLX (Qwen3.5 + optional LoRA adapter).

One forward pass per question: render the chat prompt, read the next-token
logits for the option letters, softmax them into a choice with probabilities.
"""

from __future__ import annotations

import time

import mlx.core as mx
from mlx_lm import load

from .dataset import SYSTEM, user_message


class TevModel:
    def __init__(self, model: str, adapter: str | None = None):
        self.model, self.tok = load(model, adapter_path=adapter)
        self.name = model.split("/")[-1] + ("+osu" if adapter else "")
        self._letter_ids: dict[str, int] = {}

    def _letter(self, letter: str) -> int:
        if letter not in self._letter_ids:
            ids = self.tok.encode(letter, add_special_tokens=False)
            assert len(ids) == 1, f"{letter!r} is not a single token"
            self._letter_ids[letter] = ids[0]
        return self._letter_ids[letter]

    def prompt_tokens(self, record: dict) -> list[int]:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(record)}]
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                            return_dict=False)

    def decide(self, record: dict) -> tuple[str, dict[str, float], float]:
        """Returns (chosen option key, {key: probability}, latency ms)."""
        start = time.perf_counter()
        tokens = mx.array(self.prompt_tokens(record))[None]
        logits = self.model(tokens)[0, -1]
        ids = mx.array([self._letter(o["label"]) for o in record["options"]])
        probs = mx.softmax(logits[ids].astype(mx.float32)).tolist()
        latency = (time.perf_counter() - start) * 1000
        by_key = {o["key"]: p for o, p in zip(record["options"], probs)}
        return max(by_key, key=by_key.get), by_key, latency
