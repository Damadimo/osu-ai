"""Client for System One decision APIs (TypeSafe request format).

Vercel AI Gateway, TypeSafe itself, and local Jev-compatible servers (Laya,
Kev) all serve POST {base_url}/v1/systemone with the same body, so a backend
is just a base URL, a model name, and the env var that holds its key.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(frozen=True)
class Backend:
    base_url: str
    model: str
    api_key_env: str | None = None  # None for local servers


BACKENDS = {
    "vercel": Backend("https://ai-gateway.vercel.sh/typesafe", "typesafe-ai/jev", "AI_GATEWAY_API_KEY"),
    "typesafe": Backend("https://api.typesafe.ai", "jev-latest", "TYPESAFE_API_KEY"),
    # Local Laya server: LAYA_HOST=127.0.0.1 LAYA_DEVICE=mps uv run laya-serve
    "laya": Backend("http://127.0.0.1:8000", "english"),
    "laya-typed": Backend("http://127.0.0.1:8000", "typed-decisions"),
}


@dataclass
class Decision:
    answers: dict[str, Any]
    latency_ms: float
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


class SystemOneError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status in (429, 500, 502, 503, 529)


class SystemOneClient:
    """Async client; one instance keeps one warm connection pool."""

    def __init__(self, backend: Backend, timeout_s: float = 3.0):
        headers = {"Content-Type": "application/json"}
        if backend.api_key_env:
            key = os.environ.get(backend.api_key_env)
            if not key:
                raise SystemOneError(f"{backend.api_key_env} is not set; add it to .env")
            headers["Authorization"] = f"Bearer {key}"
        self.backend = backend
        self._http = httpx.AsyncClient(
            base_url=backend.base_url, headers=headers, timeout=timeout_s, http2=True
        )

    async def decide(self, state: Any, questions: dict[str, dict]) -> Decision:
        body = {"model": self.backend.model, "state": state, "questions": questions}
        start = time.perf_counter()
        resp = await self._http.post("/v1/systemone", json=body)
        latency_ms = (time.perf_counter() - start) * 1000
        if resp.status_code != 200:
            raise SystemOneError(f"HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code)
        data = resp.json()
        return Decision(data["answers"], latency_ms, data.get("model", ""), data.get("usage", {}))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> SystemOneClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
