"""pygame window that shows a live Session: the map, the model's cursor, and its reads."""

from __future__ import annotations

import asyncio
import math
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from . import rules
from .beatmap import HitObject, Kind
from .live import Clock, Session
from .pilot import LATER

W, H = 1200, 720
SCALE = 1.5
PX, PY = 40, 96  # playfield top-left on screen
PANEL_X = PX + int(512 * SCALE) + 36

BG = (16, 16, 22)
FIELD = (26, 26, 36)
TEXT = (225, 225, 235)
DIM = (140, 140, 160)
GOOD, BAD, WARN = (110, 220, 140), (240, 90, 90), (245, 190, 80)
COMBO_COLORS = [(255, 102, 171), (84, 190, 255), (255, 209, 84), (128, 232, 146)]
JUDGMENT_STYLE = {300: ("300", (120, 200, 255)), 100: ("100", GOOD), 50: ("50", WARN), 0: ("miss", BAD)}


def to_screen(p) -> tuple[int, int]:
    return int(PX + p[0] * SCALE), int(PY + p[1] * SCALE)


def _click_sound() -> pygame.mixer.Sound:
    n = int(44100 * 0.03)
    t = np.arange(n) / 44100
    wave = (np.sin(2 * np.pi * 1900 * t) * np.exp(-t * 140) * 9000).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([wave, wave]))


class Viewer:
    def __init__(self, session: Session, title: str, audio: Path | None):
        self.s, self.title, self.audio = session, title, audio
        bm = session.bm
        self.radius = bm.difficulty.radius * SCALE
        self.preempt = bm.difficulty.preempt
        self.colors: dict[int, tuple[int, int, int]] = {}
        k = -1
        for o in bm.objects:
            k += o.new_combo
            self.colors[o.index] = COMBO_COLORS[k % len(COMBO_COLORS)]
        self.bodies: dict[int, tuple[pygame.Surface, tuple[int, int]]] = {}
        self.trail: deque = deque(maxlen=14)

    # ---------- drawing helpers ----------

    def _text(self, font, text, color, pos, center=False):
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=pos) if center else surf.get_rect(topleft=pos)
        self.screen.blit(surf, rect)
        return rect

    def _slider_body(self, o: HitObject):
        if o.index not in self.bodies:
            pts = [to_screen(p) for p in o.path.points]
            r = int(self.radius)
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x0, y0 = min(xs) - r - 4, min(ys) - r - 4
            surf = pygame.Surface((max(xs) - x0 + r + 5, max(ys) - y0 + r + 5), pygame.SRCALPHA)
            dense = []
            for a, b in zip(pts, pts[1:]):
                steps = max(1, int(math.dist(a, b) / 3))
                dense += [(a[0] + (b[0] - a[0]) * i / steps - x0, a[1] + (b[1] - a[1]) * i / steps - y0)
                          for i in range(steps)]
            dense.append((pts[-1][0] - x0, pts[-1][1] - y0))
            body = tuple(int(c * 0.35) for c in self.colors[o.index])
            for p in dense:
                pygame.draw.circle(surf, (235, 235, 235, 255), p, r)
            for p in dense:
                pygame.draw.circle(surf, (*body, 255), p, r - 4)
            self.bodies[o.index] = (surf, (x0, y0))
        return self.bodies[o.index]

    def _plan_tag(self, o: HitObject, t: float, pos):
        plan = self.s.hands.plans.get(o.index)
        if o.index in self.s.hands.pressed or o.kind is Kind.SPINNER:
            return
        if plan is None:
            label, color = "?", DIM
        elif plan.bucket == LATER:
            label, color = "later", DIM
        else:
            ok = plan.read_correct
            label = f"{plan.bucket.removesuffix('ms')} {'ok' if ok else 'x'}"
            color = GOOD if ok else BAD
        self._text(self.small, label, color, (pos[0] + self.radius * 0.8, pos[1] + self.radius * 0.8))

    def _circle(self, o: HitObject, pos, t: float, number: bool = True):
        color = self.colors[o.index]
        r = int(self.radius)
        pygame.draw.circle(self.screen, color, pos, r)
        pygame.draw.circle(self.screen, (240, 240, 240), pos, r, 3)
        if number:
            self._text(self.font, str(o.combo_number), (255, 255, 255), pos, center=True)
        if t < o.time:
            ring = r * (1 + 3 * (o.time - t) / self.preempt)
            pygame.draw.circle(self.screen, color, pos, int(ring), 3)

    def draw(self, t: float) -> None:
        s = self.s
        scr = self.screen
        scr.fill(BG)
        pygame.draw.rect(scr, FIELD, (PX - 16, PY - 16, 512 * SCALE + 32, 384 * SCALE + 32), border_radius=10)

        visible = []
        for o in s.bm.objects[s.judge._next:]:
            if o.time - self.preempt > t:
                break
            if o.index in s.judge.results:
                continue
            visible.append(o)
        for o in reversed(visible):
            pos = to_screen(o.pos)
            if o.kind is Kind.SPINNER:
                c = to_screen(rules.SPINNER_CENTER)
                pygame.draw.circle(scr, (60, 60, 80), c, 200, 6)
                if t >= o.time:
                    left = max(0.0, (o.end_time - t) / (o.end_time - o.time))
                    pygame.draw.circle(scr, (200, 200, 240), c, int(20 + 180 * left), 2)
                    need = rules.spins_per_second_required(s.bm.difficulty.od) * (o.end_time - o.time) / 1000
                    self._text(self.font, f"spin {s.judge.spins(o):.1f}/{need:.1f}", TEXT, (c[0], c[1] + 230), True)
                continue
            if o.kind is Kind.SLIDER:
                surf, off = self._slider_body(o)
                scr.blit(surf, off)
                end = to_screen(o.path.end)
                pygame.draw.circle(scr, (240, 240, 240), end, int(self.radius), 3)
                if o.slides > 1:
                    self._text(self.small, "<<", (255, 255, 255), end, center=True)
                if t >= o.time:
                    ball = to_screen(o.ball_at(t))
                    pygame.draw.circle(scr, (250, 250, 250), ball, int(self.radius * 0.85))
                    if s.hands.busy and s.hands.busy[1] is o:
                        pygame.draw.circle(scr, (250, 250, 250), ball, int(self.radius * rules.FOLLOW_RADIUS_SCALE), 2)
                    continue
            self._circle(o, pos, t)
            self._plan_tag(o, t, pos)
        for o in list(self.bodies):
            if all(v.index != o for v in visible):
                del self.bodies[o]

        for when, o, j in s.judge.popups:
            age = t - when
            if 0 <= age < 600:
                text, color = JUDGMENT_STYLE[j]
                x, y = to_screen(o.pos if o.kind is not Kind.SLIDER else o.path.end)
                self._text(self.font, text, color, (x, y - 20 - age / 30), center=True)

        cur = to_screen(s.hands.cursor)
        self.trail.append(cur)
        for i, p in enumerate(self.trail):
            pygame.draw.circle(scr, (255, 220, 80), p, 2 + i // 3)
        pygame.draw.circle(scr, (255, 230, 90), cur, 10)
        pygame.draw.circle(scr, (40, 30, 0), cur, 10, 2)
        self._hud(t)

    def _hud(self, t: float) -> None:
        s, j = self.s, self.s.judge
        self._text(self.big, f"{s.pilot.name} is playing", TEXT, (PX - 16, 18))
        self._text(self.small, self.title, DIM, (PX - 16, 58))
        c = {v: list(j.results.values()).count(v) for v in (300, 100, 50, 0)}
        acc = j.accuracy
        x = PANEL_X
        self._text(self.huge, "--" if acc is None else f"{acc:.1%}", TEXT, (x, 14))
        self._text(self.font, f"combo {j.combo}   300:{c[300]}  100:{c[100]}  50:{c[50]}  miss:{c[0]}", DIM, (x, 64))

        st = s.stats
        y = 110
        self._text(self.font, "the model", TEXT, (x, y)); y += 30
        lat = st.latency()
        rows = [
            ("latency", "--" if lat is None else f"{lat[0]:.0f} ms typical, {lat[1]:.0f} ms p95"),
            ("answered", f"{st.ok} of {st.sent} requests  ({st.in_flight} in flight)"),
        ]
        for err, n in st.errors.most_common(3):
            rows.append(("refused" if err.startswith("HTTP") else "failed", f"{n}x {err}"))
        graded = [p for p in st.reads if p.bucket not in ("spinner",)]
        if graded:
            timing = sum(p.read_correct for p in graded) / len(graded)
            actions = sum(p.action.value == ("hold" if s.hands.by_index[p.index].kind is Kind.SLIDER else "tap")
                          for p in graded) / len(graded)
            rows.append(("timing reads", f"{timing:.0%} in the right 50 ms bucket"))
            rows.append(("action reads", f"{actions:.0%} right (tap / hold / spin)"))
        for k, v in rows:
            self._text(self.small, k, DIM, (x, y))
            self._text(self.small, v, TEXT, (x + 110, y)); y += 24

        y += 16
        self._text(self.font, "latest reads", TEXT, (x, y)); y += 30
        for p in reversed(st.reads[-12:]):
            o = s.hands.by_index[p.index]
            if p.bucket == "spinner":
                line, color = f"#{o.combo_number:<3} spinner: {p.action.value}", DIM
            else:
                said = p.bucket.removesuffix("ms")
                line = f"#{o.combo_number:<3} said {said:>9}  real {p.truth_ms:4.0f} ms  {p.action.value}"
                color = GOOD if p.read_correct else BAD
            self._text(self.mono, line, color, (x, y)); y += 21
        self._text(self.small, "Esc to stop", DIM, (x, H - 34))

    # ---------- main loop ----------

    async def run(self) -> None:
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption(f"{self.s.pilot.name} plays osu!")
        self.huge = pygame.font.SysFont("helveticaneue", 44, bold=True)
        self.big = pygame.font.SysFont("helveticaneue", 30, bold=True)
        self.font = pygame.font.SysFont("helveticaneue", 20, bold=True)
        self.small = pygame.font.SysFont("helveticaneue", 16)
        self.mono = pygame.font.SysFont("menlo", 14)
        click = _click_sound()
        self.s.hands.on_press = click.play
        if self.audio:
            pygame.mixer.music.load(str(self.audio))
            pygame.mixer.music.set_volume(0.6)
            pygame.mixer.music.play(start=max(self.s.origin, 0) / 1000)
        self.s.clock = Clock(self.s.origin)  # start the game clock with the music
        thinker = asyncio.create_task(self.s.think())
        try:
            while not self.s.finished:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        self.s.finished = True
                t = self.s.step()
                self.draw(t)
                pygame.display.flip()
                await asyncio.sleep(0.004)
        finally:
            thinker.cancel()
            pygame.mixer.music.stop()
            pygame.quit()
