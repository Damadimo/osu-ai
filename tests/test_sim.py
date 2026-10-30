import asyncio
import math
from pathlib import Path

import pytest

from osu_ai import beatmap, game, judge
from osu_ai.brains import AutoBrain, DelayedAutoBrain
from osu_ai.curves import SliderPath
from osu_ai.hands import Action, Intent, perform
from osu_ai.replay import Replay

MAPS = sorted(Path(__file__).parent.parent.glob("data/maps/*.osu"))
FREEDOM_DIVE = Path(__file__).parent.parent / "data/maps/129891.osu"


def test_perfect_circle_slider_is_an_arc():
    path = SliderPath("P", [(0, 0), (50, 50), (100, 0)], math.pi * 50)
    assert path.end == pytest.approx((100, 0), abs=0.5)
    assert path.at(0.5) == pytest.approx((50, 50), abs=0.5)


def test_linear_slider_is_cut_and_extended_to_its_length():
    assert SliderPath("L", [(0, 0), (100, 0)], 60).end == pytest.approx((60, 0))
    assert SliderPath("L", [(0, 0), (100, 0)], 150).end == pytest.approx((150, 0))


@pytest.mark.skipif(not FREEDOM_DIVE.exists(), reason="download data/maps/129891.osu")
def test_freedom_dive_matches_known_object_and_combo_counts():
    bm = beatmap.parse(FREEDOM_DIVE)
    assert len(bm.objects) == 1983
    sliders = [o for o in bm.objects if o.kind is beatmap.Kind.SLIDER]
    # osu! max combo counts each slider's head plus every tick, repeat and tail.
    assert len(bm.objects) + sum(len(o.checkpoints) for o in sliders) == 2385


@pytest.mark.parametrize("path", MAPS, ids=lambda p: p.stem)
def test_perfect_brain_scores_100_percent(path):
    run = asyncio.run(game.run(beatmap.parse(path), AutoBrain(), realtime=False))
    assert run.score.accuracy == 1.0


@pytest.mark.skipif(not FREEDOM_DIVE.exists(), reason="download data/maps/129891.osu")
def test_decisions_that_arrive_after_the_hit_window_miss():
    bm = beatmap.parse(FREEDOM_DIVE)
    too_late = bm.difficulty.preempt + bm.difficulty.windows.meh + 1
    run = asyncio.run(game.run(bm, DelayedAutoBrain([too_late]), realtime=False))
    # Spinners last long enough to finish even when started late; everything else misses.
    for obj, judgment in zip(bm.objects, run.score.judgments):
        assert judgment == 0 or obj.kind is beatmap.Kind.SPINNER


@pytest.mark.skipif(not FREEDOM_DIVE.exists(), reason="download data/maps/129891.osu")
def test_timing_offsets_map_to_hit_windows():
    bm = beatmap.parse(FREEDOM_DIVE)  # OD8: 300 within 32 ms, 100 within 76 ms
    circles = [o for o in bm.objects if o.kind is beatmap.Kind.CIRCLE][:20]
    for offset, expected in [(20, 300), (-50, 100), (100, 50)]:
        replay = Replay()
        for o in circles:
            replay.move(o.time + offset, o.pos)
            replay.press(o.time + offset, o.time + offset + 20)
        sub = beatmap.Beatmap(**{**bm.__dict__, "objects": circles})
        assert set(judge.score(sub, replay).judgments) == {expected}


@pytest.mark.skipif(not FREEDOM_DIVE.exists(), reason="download data/maps/129891.osu")
def test_tapping_a_slider_instead_of_holding_loses_accuracy():
    bm = beatmap.parse(FREEDOM_DIVE)
    sliders = [o for o in bm.objects if o.kind is beatmap.Kind.SLIDER][:20]
    sub = beatmap.Beatmap(**{**bm.__dict__, "objects": sliders})
    tapped = judge.score(sub, perform(sub, {o.index: Intent(Action.TAP, 0) for o in sliders}))
    held = judge.score(sub, perform(sub, {o.index: Intent(Action.HOLD, 0) for o in sliders}))
    # lazer keeps the head's 300 for a tapped slider; the missed ticks and tail still cost a lot
    assert held.accuracy == 1.0 and tapped.accuracy < 0.7
