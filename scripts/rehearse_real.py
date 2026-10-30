"""Rehearse a real-game run end to end: the real play loop (RealGame + Policy + model) against a
stand-in osu! with realistic delays.

  uv run --extra train scripts/rehearse_real.py data/lazer_osu/827932.osu [--latency-ms 25 --display-ms 15 ...]

The stand-in shows, at each 60 Hz display time, what lazer would draw `display_ms` earlier given
the play so far (lazer_frames), delivers it `capture_ms` later, and applies the input RealGame
sends the way OsuHands and lazer do: at its scheduled time, `input_ms` later, quantised to 480 Hz
update frames with the keyboard before the mouse, judged by the lazer rules (judge.Engine).
latency_ms is what RealGame assumes; the true total delay is display_ms + input_ms.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import math

import numpy as np

from osu_ai import beatmap, judge, real
from osu_ai import hands_client as hc
from osu_ai.agent import LazerAgent
from osu_ai.beatmap import Kind
from osu_ai.geometry import OsuWindow
from osu_ai.lazer_frames import Look, Outcome, View, render
from osu_ai.policy import Policy

WIN = OsuWindow(244, 107, 1024, 768)
BASE = 7_000_000_000_000
UPDATE_MS = 1000 / 480


class Game:
    """The stand-in osu!: the judge engine plus what it has drawn, advanced as input arrives."""

    def __init__(self, bm, g0: float, input_ms: float):
        self.bm, self.g0, self.input_ms = bm, g0, input_ms
        self.engine = judge.Engine(bm)
        self.cursor = (256.0, 192.0)
        self.pending: list = []  # (exec ns, order, Event) waiting for their time
        self.order = itertools.count()
        self.held: set[int] = set()
        self.frame_k = 0  # next update frame to run
        self.outcome = Outcome(spin_rate={o.index: 1.6 for o in bm.objects if o.kind is Kind.SPINNER})

    def game_ms(self, ns: float) -> float:
        return (ns - BASE) / 1e6 + self.g0

    def submit(self, now_ns: int, events) -> None:
        for e in events:
            if e.kind == hc.CANCEL:
                self.pending = [p for p in self.pending if not (e.tag == hc.ALL_TAGS or p[2].tag == e.tag)]
                heapq.heapify(self.pending)
                continue
            at = e.at_ns if e.at_ns > 0 else now_ns
            heapq.heappush(self.pending, (max(at, now_ns), next(self.order), e))

    def advance(self, g: float) -> None:
        """Run update frames up to game time g, applying input that has arrived by then."""
        while (self.frame_k + 1) * UPDATE_MS <= g:
            t = (self.frame_k + 1) * UPDATE_MS
            arrived = []
            while self.pending and self.game_ms(self.pending[0][0]) + self.input_ms <= t:
                arrived.append(heapq.heappop(self.pending)[2])
            for e in arrived:  # keyboard first, at the previous frame's cursor
                if e.kind == hc.KEY_DOWN and e.key not in self.held:
                    self.held.add(e.key)
                    self.engine.press(t, self.cursor, e.key)
                elif e.kind == hc.KEY_UP and e.key in self.held:
                    self.held.discard(e.key)
                    self.engine.release(t, self.cursor, e.key)
                elif e.kind == hc.RELEASE:
                    for k in sorted(self.held):
                        self.engine.release(t, self.cursor, k)
                    self.held.clear()
            for e in arrived:
                if e.kind == hc.MOVE:
                    px = WIN.left + math.floor(e.x - WIN.left)
                    py = WIN.top + math.floor(e.y - WIN.top)
                    self.cursor = WIN.points_to_osu(px, py)
            self.engine.move(t, self.cursor)
            self.engine.frame()
            self.frame_k += 1
        for i, (j, off) in self.engine.heads.items():
            if i not in self.outcome.judged:
                o = self.bm.objects[i]
                self.outcome.judged[i] = (o.time + off, True) if j > 0 else (g, False)


class Hands(real.DryHands):
    def __init__(self, game: Game):
        super().__init__()
        self.game, self.real_game = game, None

    def send(self, *events):
        self.game.submit(self.real_game._now, events)
        super().send(*events)

    def cancel(self, tag=hc.ALL_TAGS):
        self.send(hc.Event(hc.CANCEL, tag=tag))

    def release_all(self, at_ns=0):
        self.send(hc.Event(hc.RELEASE, at_ns))


def rehearse(path: str, weights: str, latency_ms: float, display_ms: float, input_ms: float, capture_ms: float,
             infer_ms: float, slack_ms: float, jitter_ms: float, seed: int = 0):
    bm = beatmap.parse(path)
    g0 = bm.objects[0].time - bm.difficulty.preempt - 1500
    end_g = max(o.end_time for o in bm.objects) + 1500
    game = Game(bm, g0, input_ms)
    hands = Hands(game)
    agent = LazerAgent(weights)
    w = bm.difficulty.windows
    rg = real.RealGame(hands, agent, Policy(lead_ms=latency_ms + slack_ms), WIN, latency_ms,
                       early_guard_ms=w.meh - 15, log=lambda m: None)
    hands.real_game = rg
    rng = np.random.default_rng(seed)
    view, look = View((640, 480)), Look()

    def frames():
        k = 0
        while True:
            shown = BASE + round((k * 1000 / 60 + rng.normal(0, jitter_ms)) * 1e6)
            g = game.game_ms(shown) - display_ms
            if g > end_g:
                return
            game.advance(g)
            img = np.clip(render(bm, g, game.outcome, look, view) + 0.5, 0, 255).astype(np.uint8)
            arrived = shown + round((capture_ms + infer_ms) * 1e6)
            yield hc.Frame(k, shown, shown, arrived, arrived, img, 0, False, 0)
            k += 1

    rg.run(frames=frames())
    game.advance(end_g + 1000)
    game.engine.finish()
    return bm, game.engine.result(), rg.summary()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("maps", nargs="+")
    ap.add_argument("--weights", default="models/osunet-lazer-next.safetensors")
    ap.add_argument("--latency-ms", type=float, default=25.0, help="what RealGame assumes")
    ap.add_argument("--display-ms", type=float, default=15.0, help="lazer draw -> display")
    ap.add_argument("--input-ms", type=float, default=10.0, help="posted event -> lazer")
    ap.add_argument("--capture-ms", type=float, default=6.0)
    ap.add_argument("--infer-ms", type=float, default=4.0)
    ap.add_argument("--slack-ms", type=float, default=20.0)
    ap.add_argument("--jitter-ms", type=float, default=1.0)
    args = ap.parse_args()
    for path in args.maps:
        bm, score, s = rehearse(path, args.weights, args.latency_ms, args.display_ms, args.input_ms,
                                args.capture_ms, args.infer_ms, args.slack_ms, args.jitter_ms)
        print(f"{bm.name[:55]}: {score.summary()}")
        print(f"   presses {s['presses']}  late {s['late_by_ms']}  decision p95 {s['shown_to_decision_ms']['p95']} ms")


if __name__ == "__main__":
    main()
