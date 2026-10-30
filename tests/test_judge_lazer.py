"""Judging checked rule by rule, and against osu!lazer's own gameplay code.

Each scenario is a tiny map plus replay frames (time, x, y, keys held: bit 0 = left, bit 1 = right).
The engine runs on exactly those frames (step_ms=None). The oracle tests play the same frames in lazer
(tools/lazer_oracle `replay`, a headless ReplayPlayer), add the few frames lazer judged in between (where
its clock landed), and require identical judgements.
"""

import bisect
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from osu_ai import beatmap, judge
from osu_ai import lazer_rules as lz
from osu_ai.live import LiveJudge
from osu_ai.synth_replay import human_replay

ROOT = Path(__file__).parent.parent
ORACLE_DIR = ROOT / "tools/lazer_oracle"
ORACLE_DLL = ORACLE_DIR / "bin/Release/net10.0/oracle.dll"
OSU_APP = Path("/Applications/osu!.app/Contents/MacOS/osu.Game.dll")
DOTNET = shutil.which("dotnet") or "/opt/homebrew/bin/dotnet"
TUTORIAL = ROOT / "data/maps/2116202.osu"
FREEDOM_DIVE = ROOT / "data/maps/129891.osu"

R = 36.49502944946289  # CS4 radius
SLIDER = "100,192,1000,6,0,L|300:192,1,200"  # 0.28 px/ms: tick at 1500 (x=240), tail at 1714.29 (x=300)
SLIDER_END = 1000 + 200 / 0.28


def osu_text(objects: list[str], od: float = 8, tick_rate: float = 1, hp: float = 5) -> str:
    return "\n".join([
        "osu file format v14", "", "[General]", "StackLeniency: 0.7", "Mode: 0", "",
        "[Metadata]", "Title:judge test", "Artist:osu-ai", "Version:test", "",
        "[Difficulty]", f"HPDrainRate:{hp}", "CircleSize:4", f"OverallDifficulty:{od}", "ApproachRate:9",
        "SliderMultiplier:1.4", f"SliderTickRate:{tick_rate}", "",
        "[TimingPoints]", "0,500,4,2,0,100,1,0", "", "[HitObjects]", *objects, "",
    ])


def ball(t: float) -> tuple[float, float]:
    return (100 + 0.28 * min(max(t - 1000, 0), SLIDER_END - 1000), 192.0)


def follow(t0: float, t1: float, keys: int, offset=lambda t: (0.0, 0.0), step: float = 10.0) -> list[list[float]]:
    """Frames on the SLIDER ball (plus an offset) every `step` ms."""
    out, t = [], t0
    while t < t1:
        (bx, by), (dx, dy) = ball(t), offset(t)
        out.append([t, bx + dx, by + dy, keys])
        t += step
    return out


def spin(t0: float, t1: float, rotations: float, keys: int = 1, back_and_forth: bool = False) -> list[list[float]]:
    out, n = [], int((t1 - t0) / 10)
    for i in range(n + 1):
        f = i / n
        a = math.radians(170 * math.sin(f * 40)) if back_and_forth else 2 * math.pi * rotations * f
        out.append([t0 + (t1 - t0) * f, 256 + 60 * math.sin(a), 192 - 60 * math.cos(a), keys])
    return out


def at(t: float, x: float, y: float, keys: int = 0) -> list[float]:
    return [t, x, y, keys]


def slider_results(head: str, tick: str, tail: str, ticks: int = 1) -> list[tuple]:
    return [(0, lz.HEAD, head)] + [(0, lz.TICK, tick)] * ticks + [(0, lz.TAIL, tail), (0, lz.SLIDER, lz.IGNORE_HIT)]


def spinner_results(spins: int, result: str) -> list[tuple]:
    """A 3 s spinner at OD5: 7 spins required, so 9 small ticks, then 10 bonus ticks."""
    small = min(spins, 9)
    return ([(0, lz.SPIN_TICK, lz.SMALL_BONUS)] * small + [(0, lz.SPIN_TICK, lz.IGNORE_MISS)] * (9 - small)
            + [(0, lz.SPIN_BONUS, lz.IGNORE_MISS)] * 10 + [(0, lz.SPINNER, result)])


def wander(t: float) -> tuple[float, float]:
    """2R off the ball (kept), then 3R (lost), back to 2R (not regained), then 0.5R (regained)."""
    for start, end, k in ((1200, 1300, 2), (1300, 1350, 3), (1350, 1600, 2), (1600, 1e9, 0.5)):
        if start <= t < end:
            return 0.0, k * R
    return 0.0, 0.0


HEAD_ELSEWHERE = [at(690, 400, 50), at(700, 400, 50, 2), at(990, 100, 192, 2)]  # right key held from before
SPINNER_MAP = (["256,192,1000,12,0,4000"], {"od": 5})
END = at(2500, 300, 192)

# name -> (objects, map settings, frames, expected [(object index, kind, result)] in order)
SCENARIOS = {
    "note lock: an early press on the next circle only shakes it": (
        ["100,100,1000,5,0", "300,100,1100,1,0"], {},
        [at(900, 300, 100), at(950, 300, 100, 1), at(970, 300, 100), at(995, 100, 100), at(1000, 100, 100, 2),
         at(1020, 100, 100), at(1095, 300, 100), at(1100, 300, 100, 1), at(1120, 300, 100), at(1500, 300, 100)],
        [(0, lz.CIRCLE, lz.GREAT), (1, lz.CIRCLE, lz.GREAT)]),
    "hitting a later circle after the earlier one's time misses the earlier one": (
        ["100,100,1000,5,0", "300,100,1100,1,0"], {},
        [at(900, 300, 100), at(1050, 300, 100, 1), at(1070, 300, 100), at(1500, 300, 100)],
        [(0, lz.CIRCLE, lz.MISS), (1, lz.CIRCLE, lz.OK)]),
    "a press 120-400 ms early is a miss, earlier only shakes": (
        ["100,100,1000,5,0", "100,100,2000,1,0"], {},
        [at(500, 100, 100), at(550, 100, 100, 1), at(570, 100, 100), at(1010, 100, 100, 2), at(1030, 100, 100),
         at(1800, 100, 100, 1), at(1820, 100, 100), at(2500, 100, 100)],
        [(0, lz.CIRCLE, lz.GREAT), (1, lz.CIRCLE, lz.MISS)]),
    "simultaneous circles do not lock each other": (
        ["100,100,1000,5,0", "300,100,1000,1,0"], {},
        [at(980, 300, 100), at(990, 300, 100, 1), at(1000, 100, 100), at(1010, 100, 100, 2), at(1030, 100, 100),
         at(1500, 100, 100)],
        [(1, lz.CIRCLE, lz.GREAT), (0, lz.CIRCLE, lz.GREAT)]),
    "a followed slider gets head, tick and tail": (
        [SLIDER], {}, [at(990, 100, 192), *follow(1000, SLIDER_END + 20, 1), END],
        slider_results(lz.GREAT, lz.TICK_HIT, lz.TAIL_HIT)),
    "releasing before the last 36 ms loses only the tail": (
        [SLIDER], {},
        [at(990, 100, 192), *follow(1000, SLIDER_END - 60, 1), *follow(SLIDER_END - 60, SLIDER_END, 0), END],
        slider_results(lz.GREAT, lz.TICK_HIT, lz.IGNORE_MISS)),
    "releasing inside the last 36 ms keeps the tail": (
        [SLIDER], {},
        [at(990, 100, 192), *follow(1000, SLIDER_END - 20, 1), *follow(SLIDER_END - 20, SLIDER_END, 0), END],
        slider_results(lz.GREAT, lz.TICK_HIT, lz.TAIL_HIT)),
    "tracking survives up to 2.4x but must come back within 1x": (
        [SLIDER], {}, [at(990, 100, 192), *follow(1000, SLIDER_END + 20, 1, wander), END],
        slider_results(lz.GREAT, lz.TICK_MISS, lz.TAIL_HIT)),
    "a head hit while the other key is held tracks only the head's key": (
        [SLIDER], {}, [*HEAD_ELSEWHERE, *follow(1000, 1100, 3), *follow(1100, SLIDER_END + 20, 2), END],
        slider_results(lz.GREAT, lz.TICK_MISS, lz.IGNORE_MISS)),
    "once the other key is let go, either key tracks": (
        [SLIDER], {},
        [*HEAD_ELSEWHERE, *follow(1000, 1050, 3), *follow(1050, 1150, 1), *follow(1150, 1200, 3),
         *follow(1200, SLIDER_END + 20, 2), END],
        slider_results(lz.GREAT, lz.TICK_HIT, lz.TAIL_HIT)),
    "a missed head still lets a held key track the ball": (
        [SLIDER], {},
        [at(690, 400, 50), at(700, 400, 50, 1), at(1140, 400, 50, 1), *follow(1150, SLIDER_END + 20, 1), END],
        slider_results(lz.MISS, lz.TICK_HIT, lz.TAIL_HIT)),
    "a late head hit settles the ticks it skipped": (
        [SLIDER], {"od": 0, "tick_rate": 8},
        [at(1140, 100, 192), at(1150, 100, 192, 1), *follow(1160, SLIDER_END + 20, 1), END],
        slider_results(lz.MEH, lz.TICK_HIT, lz.TAIL_HIT, ticks=11)),
    "a head shaken by one key and hit with the other still tracks": (
        [SLIDER], {},
        [at(490, 100, 192), at(500, 100, 192, 1), at(520, 100, 192), at(990, 100, 192),
         *follow(1000, SLIDER_END + 20, 2), END],
        slider_results(lz.GREAT, lz.TICK_HIT, lz.TAIL_HIT)),
    "a circle inside a slider that times out on the tick's frame is judged first (later objects update first)": (
        [SLIDER, "400,300,1379,1,0"], {}, [at(990, 100, 192), *follow(1000, SLIDER_END + 20, 1), END],
        [(0, lz.HEAD, lz.GREAT), (1, lz.CIRCLE, lz.MISS), (0, lz.TICK, lz.TICK_HIT), (0, lz.TAIL, lz.TAIL_HIT),
         (0, lz.SLIDER, lz.IGNORE_HIT)]),
    "spinning 8 times clears a 7-spin spinner": (
        *SPINNER_MAP, [at(990, 256, 132), *spin(1000, 3990, 8.2), at(4100, 256, 132)], spinner_results(8, lz.GREAT)),
    "6.5 spins of 7 is a 100": (
        *SPINNER_MAP, [at(990, 256, 132), *spin(1000, 3990, 6.5), at(4100, 256, 132)], spinner_results(6, lz.OK)),
    "turning back and forth never completes a spin": (
        *SPINNER_MAP, [at(990, 256, 132), *spin(1000, 3990, 0, back_and_forth=True), at(4100, 256, 132)],
        spinner_results(0, lz.MISS)),
}


def judge_frames(bm: beatmap.Beatmap, frames: list[list[float]], step_ms: float | None = None) -> judge.Score:
    engine = judge.Engine(bm, step_ms=step_ms)
    held: set[int] = set()
    for t, x, y, mask in frames:
        now = {k for k in (0, 1) if int(mask) >> k & 1}
        engine.move(t, (x, y))
        for key in sorted(held - now):
            engine.release(t, (x, y), key)
        for key in sorted(now - held):
            engine.press(t, (x, y), key)
        held = now
        engine.frame()
    engine.finish()
    return engine.result()


@pytest.fixture(scope="module")
def scenario_maps(tmp_path_factory) -> dict[str, Path]:
    out = {}
    for i, (name, (objects, settings, _, _)) in enumerate(SCENARIOS.items()):
        path = tmp_path_factory.mktemp("maps") / f"scenario{i}.osu"
        path.write_text(osu_text(objects, **settings))
        out[name] = path
    return out


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario(name, scenario_maps):
    _, _, frames, expected = SCENARIOS[name]
    score = judge_frames(beatmap.parse(scenario_maps[name]), frames)
    assert [(e.index, e.kind, e.result) for e in score.events] == expected


def test_note_lock_judges_at_the_hit(scenario_maps):
    name = "hitting a later circle after the earlier one's time misses the earlier one"
    score = judge_frames(beatmap.parse(scenario_maps[name]), SCENARIOS[name][2])
    assert [e.t for e in score.events] == [1050, 1050]
    assert score.events[1].offset == -50 and score.head_offsets == [-50]
    assert score.judgments == [0, 100] and score.rank == "D"


def test_slider_accuracy_weights_ticks_and_tails(scenario_maps):
    """Head 300, tick 30, tail 150: dropping the tail of a fully tracked slider leaves 330/480."""
    name = "releasing before the last 36 ms loses only the tail"
    score = judge_frames(beatmap.parse(scenario_maps[name]), SCENARIOS[name][2])
    assert score.accuracy == pytest.approx(330 / 480)
    assert score.max_combo == 2 and score.judgments == [300] and score.rank == "D"
    assert score.statistics[lz.TICK_HIT] == 1 and score.statistics[lz.IGNORE_MISS] == 1


def test_live_judge_matches_score_on_the_same_replay():
    if not FREEDOM_DIVE.exists():
        pytest.skip("download data/maps/129891.osu")
    bm = beatmap.parse(FREEDOM_DIVE)
    bm = beatmap.Beatmap(**{**bm.__dict__, "objects": bm.objects[:300]})
    replay = human_replay(bm, random.Random(3), timing_sd=30, aim_sd=0.5, miss_rate=0.1)
    expected = judge.score(bm, replay)
    live = LiveJudge(bm)
    for t, cursor, early, downs, late in judge.replay_inputs(replay):
        for key in early:
            live.on_release(t, key, cursor)
        for key in downs:
            live.on_press(t, cursor, key)
        for key in late:
            live.on_release(t, key, cursor)
        live.update(t, cursor)
    got = live.finish()
    assert [(e.t, e.index, e.kind, e.result, e.combo) for e in got.events] == \
           [(e.t, e.index, e.kind, e.result, e.combo) for e in expected.events]
    assert (got.accuracy, got.max_combo, got.total_score, got.judgments) == \
           (expected.accuracy, expected.max_combo, expected.total_score, expected.judgments)
    assert live.accuracy == expected.accuracy and live.max_combo == expected.max_combo
    assert set(live.results) == {o.index for o in bm.objects} and live._next == len(bm.objects)


def test_health_runs_out_without_input_but_not_with_perfect_play():
    if not FREEDOM_DIVE.exists():
        pytest.skip("download data/maps/129891.osu")
    from osu_ai.hands import CORRECT_ACTION, Intent, perform
    from osu_ai.replay import Replay

    bm = beatmap.parse(FREEDOM_DIVE)
    idle = judge.score(bm, Replay())
    assert not idle.passed and idle.rank == "F" and idle.min_health == 0
    assert idle.failed_at < bm.objects[40].time  # misses drain it within the first few combos
    perfect = judge.score(bm, perform(bm, {o.index: Intent(CORRECT_ACTION[o.kind], 0) for o in bm.objects}))
    assert perfect.passed and perfect.rank == "X" and perfect.total_score == 1000880
    assert perfect.max_combo == len(bm.objects) + sum(len(o.nested) for o in bm.objects)


# ---- against lazer itself ----

def _dense(frames: list[list[float]], gap: float = 2.0) -> list[list[float]]:
    """Frames at most `gap` ms apart (cursor interpolated, keys held): lazer and the engine judge the same moments."""
    out = [frames[0]]
    for a, b in zip(frames, frames[1:]):
        steps = math.ceil((b[0] - a[0]) / gap)
        for i in range(1, steps):
            f = i / steps
            out.append([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3]])
        out.append(b)
    return out


def _with_lazer_frames(frames: list[list[float]], extra: list[float]) -> list[list[float]]:
    """The frames lazer also judged between replay frames (its headless clock lands there every 33 ms, at a phase
    that varies run to run): cursor interpolated, keys of the replay frame before, as its replay handler does."""
    times, out = [f[0] for f in frames], list(frames)
    for t in extra:
        i = bisect.bisect_right(times, t)
        if i == 0:
            out.append([t, frames[0][1], frames[0][2], 0])
        elif i == len(frames):
            out.append([t, *frames[-1][1:]])
        else:
            a, b = frames[i - 1], frames[i]
            f = (t - a[0]) / (b[0] - a[0])
            out.append([t, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3]])
    return sorted(out, key=lambda f: f[0])


def _replay_frames(replay) -> list[list[float]]:
    keys, out = set(), []
    for t, (x, y), early, downs, late in judge.replay_inputs(replay):
        keys = (keys - set(early) | set(downs)) - set(late)
        out.append([t, x, y, sum(1 << k for k in keys)])
    return out


def _chaos_frames(bm: beatmap.Beatmap, rng: random.Random) -> list[list[float]]:
    """Adversarial play: early/late/missing presses, either or both keys, held keys, cursor excursions
    around 1x/2.4x the radius on sliders, early releases, and spinners that change direction."""
    events = []  # (t, order, value): order 0 = cursor (x, y), 1 = key (key, down)
    r, meh, cursor, free = bm.difficulty.radius, bm.difficulty.windows.meh, (256.0, 192.0), bm.objects[0].time - 600
    for o in bm.objects:
        if o.kind is beatmap.Kind.SPINNER:
            start, end, key = o.time + rng.uniform(-200, 300), o.end_time + rng.uniform(-400, 200), rng.randrange(2)
            rps, radius, a, direction, t = rng.choice([0.5, 3, 8]), rng.uniform(5, 120), 0.0, 1, start
            events.append((start, 1, (key, True)))
            while t < end:
                direction = -direction if rng.random() < 0.02 else direction
                a += direction * 2 * math.pi * rps * 0.008
                events.append((t, 0, (256 + radius * math.cos(a), 192 + radius * math.sin(a))))
                t += 8
            events.append((t, 1, (key, False)))
            free = t
            continue
        press = o.time + rng.choice([rng.gauss(0, 20), rng.uniform(-500, -150), rng.uniform(meh - 20, meh + 60)])
        aim = (o.x + rng.gauss(0, 0.6 * r), o.y + rng.gauss(0, 0.6 * r))
        t0 = max(free, press - 150)
        for f in (i / 8 for i in range(1, 9)):
            x, y = cursor[0] + (aim[0] - cursor[0]) * f, cursor[1] + (aim[1] - cursor[1]) * f
            events.append((t0 + (press - t0) * f, 0, (x, y)))
        key, cursor = rng.randrange(2), aim
        if rng.random() < 0.1:
            events += [(press - 100, 1, (1 - key, True)), (press + rng.uniform(0, 600), 1, (1 - key, False))]
        events.append((press, 1, (key, True)))
        release = press + rng.uniform(15, 90)
        if o.kind is beatmap.Kind.SLIDER:
            release, drift, t = o.end_time + rng.choice([rng.uniform(-80, 40), rng.uniform(-400, 0)]), 0.0, press
            while t < release:
                drift = rng.choice([0.0, 1.5 * r, 2.2 * r, 2.8 * r]) if rng.random() < 0.03 else drift
                (bx, by), a = o.ball_at(t), rng.uniform(0, 2 * math.pi)
                events.append((t, 0, (bx + drift * math.cos(a), by + drift * math.sin(a))))
                t += 8
            release = max(release, press + 10)
        events.append((release, 1, (key, False)))
        free = release if o.kind is beatmap.Kind.SLIDER else press
    events.sort(key=lambda e: (e[0], e[1]))
    frames, keys, cur = [], set(), (256.0, 192.0)
    for t, order, value in events:
        if order == 0:
            cur = value
        else:
            (keys.add if value[1] else keys.discard)(value[0])
        if frames and frames[-1][0] == t:
            frames[-1] = [t, cur[0], cur[1], sum(1 << k for k in keys)]
        else:
            frames.append([t, cur[0], cur[1], sum(1 << k for k in keys)])
    return frames


@pytest.fixture(scope="module")
def lazer_replays(tmp_path_factory):
    """Runs (map, frames) pairs through lazer's ReplayPlayer in one oracle process."""
    if not OSU_APP.exists() or not Path(DOTNET).exists():
        pytest.skip("needs osu!lazer in /Applications and the .NET SDK")
    sources = [*ORACLE_DIR.glob("*.cs"), ORACLE_DIR / "oracle.csproj"]
    if not ORACLE_DLL.exists() or max(p.stat().st_mtime for p in sources) > ORACLE_DLL.stat().st_mtime:
        subprocess.run([DOTNET, "build", "-c", "Release"], cwd=ORACLE_DIR, check=True, capture_output=True)
    folder = tmp_path_factory.mktemp("replays")

    def run(jobs: list[tuple[Path, list[list[float]]]]) -> list[dict]:
        args = []
        for i, (path, frames) in enumerate(jobs):
            end = max(o.end_time for o in beatmap.parse(path).objects) + 3000
            frames_file = folder / f"frames{i}.json"
            frames_file.write_text(json.dumps({"frames": frames + [[end, frames[-1][1], frames[-1][2], 0]]}))
            args += [str(path), str(frames_file)]
        out = subprocess.run([DOTNET, str(ORACLE_DLL), "replay", *args], check=True, capture_output=True, text=True,
                             timeout=1800).stdout
        return [json.loads(line) for line in out.splitlines()]

    return run


def _assert_same(lazer: dict, ours: judge.Score, label: str) -> None:
    assert lazer["completed"], label
    theirs = [(j[1], j[2], j[3], j[4]) for j in lazer["judgements"]]
    assert [(e.index, e.kind, e.result, e.combo) for e in ours.events] == theirs, label
    assert ours.accuracy == pytest.approx(lazer["accuracy"], abs=1e-12), label
    assert (ours.max_combo, ours.total_score) == (lazer["max_combo"], lazer["total_score"]), label
    for e, j in zip(ours.events, lazer["judgements"]):
        # lazer reports a judgement no later than its object's time plus hit window (ours: when it was made),
        # and a spin may complete a frame apart (float32 angles)
        assert j[0] <= e.t + 10 and abs(e.health - j[5]) < 1e-3, label


def test_scenarios_match_lazer(lazer_replays, scenario_maps):
    jobs = [(scenario_maps[name], _dense(frames)) for name, (_, _, frames, _) in SCENARIOS.items()]
    for name, (path, frames), lazer in zip(SCENARIOS, jobs, lazer_replays(jobs)):
        _assert_same(lazer, judge_frames(beatmap.parse(path), _with_lazer_frames(frames, lazer["extra_frames"])), name)


def test_random_replays_match_lazer(lazer_replays):
    """Human-like play and adversarial play on a real map: every judgement, combo, accuracy and score."""
    if not TUTORIAL.exists():
        pytest.skip("download data/maps/2116202.osu")
    bm = beatmap.parse(TUTORIAL)
    human = human_replay(bm, random.Random(1), timing_sd=25, aim_sd=0.5, miss_rate=0.08)
    jobs = [(TUTORIAL, _dense(_replay_frames(human))), (TUTORIAL, _dense(_chaos_frames(bm, random.Random(2))))]
    for (path, frames), lazer, label in zip(jobs, lazer_replays(jobs), ["human", "chaos"]):
        _assert_same(lazer, judge_frames(bm, _with_lazer_frames(frames, lazer["extra_frames"])), label)
