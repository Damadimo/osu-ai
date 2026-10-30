"""Difficulty formulas, beatmap loading and health checked against osu!lazer itself.

The oracle (tools/lazer_oracle) loads the installed osu!lazer assemblies read-only; tests that need it
are skipped when dotnet or /Applications/osu!.app is missing.
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from osu_ai import beatmap, rules
from osu_ai import lazer_rules as lz
from osu_ai.beatmap import Kind

ROOT = Path(__file__).parent.parent
ORACLE_DIR = ROOT / "tools/lazer_oracle"
ORACLE_DLL = ORACLE_DIR / "bin/Release/net10.0/oracle.dll"
OSU_APP = Path("/Applications/osu!.app/Contents/MacOS/osu.Game.dll")
DOTNET = shutil.which("dotnet") or "/opt/homebrew/bin/dotnet"
MAPS = sorted(ROOT.glob("data/maps/*.osu")) + sorted(ROOT.glob("data/lazer_osu/*.osu"))
NESTED = {"SliderTick": "tick", "SliderRepeat": "repeat", "SliderTailCircle": "tail"}
KIND = {"HitCircle": Kind.CIRCLE, "Slider": Kind.SLIDER, "Spinner": Kind.SPINNER}


@pytest.fixture(scope="session")
def oracle():
    if not OSU_APP.exists() or not Path(DOTNET).exists():
        pytest.skip("needs osu!lazer in /Applications and the .NET SDK")
    sources = [*ORACLE_DIR.glob("*.cs"), ORACLE_DIR / "oracle.csproj"]
    if not ORACLE_DLL.exists() or max(p.stat().st_mtime for p in sources) > ORACLE_DLL.stat().st_mtime:
        subprocess.run([DOTNET, "build", "-c", "Release"], cwd=ORACLE_DIR, check=True, capture_output=True)

    def run(*args: str) -> str:
        return subprocess.run([DOTNET, str(ORACLE_DLL), *args], check=True, capture_output=True, text=True).stdout

    return run


@pytest.fixture(scope="session")
def lazer_maps(oracle) -> dict[str, dict]:
    if not MAPS:
        pytest.skip("no maps in data/maps or data/lazer_osu")
    dumps = [json.loads(line) for line in oracle("maps", *map(str, MAPS)).splitlines()]
    return {d["file"]: d for d in dumps if "error" not in d}


def test_known_values():
    w = rules.hit_windows(8.3)
    assert (w.great, w.ok, w.meh, w.miss) == (29.5, 72.5, 115.5, 400)
    assert (rules.hit_windows(0.1).meh, rules.hit_windows(10).great) == (197.5, 19.5)
    assert (rules.preempt_ms(9.3), rules.preempt_ms(9.6), rules.preempt_ms(8)) == (554, 509, 750)
    assert rules.circle_radius(4) == pytest.approx(36.495, abs=1e-3)
    assert rules.fade_in_ms(10) == 400 and rules.fade_in_ms(5) == 400
    assert w.result(-29.5) == 300 and w.result(30) == 100 and w.result(-115.5) == 50
    assert w.result(116) == 0 and w.result(-400) == 0 and w.result(401) is None
    assert rules.spins_required(8, 1000) == 3 and rules.spins_required(5, 3000) == 7
    assert (lz.miss_health(6), lz.tick_miss_health(7)) == pytest.approx((-0.14, -0.101))


def test_difficulty_tables_match_lazer(oracle):
    for row in json.loads(oracle("tables")):
        v, w = row["value"], rules.hit_windows(row["value"])
        assert (w.great, w.ok, w.meh, w.miss) == (row["great"], row["ok"], row["meh"], row["miss"]), v
        ours = (rules.circle_scale(v), rules.circle_radius(v), rules.preempt_ms(v))
        assert ours == (row["scale"], row["radius"], row["preempt"]), v


@pytest.mark.parametrize("path", MAPS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_beatmap_matches_lazer(path, lazer_maps):
    """Times, stacked positions, slider ticks/repeats/tail, combo flags, breaks, max combo and drain rate."""
    lazer = lazer_maps[str(path)]
    bm = beatmap.parse(path)
    assert bm.format_version == lazer["version"]
    assert [list(b) for b in bm.breaks] == [list(b) for b in lazer["breaks"]]
    assert len(bm.objects) == len(lazer["objects"])
    assert len(bm.objects) + sum(len(o.nested) for o in bm.objects) == lazer["max_combo"]
    assert lz.drain_rate(bm.objects, bm.breaks, bm.difficulty.hp) == lazer["drain_rate"]
    for o, t in zip(bm.objects, lazer["objects"]):
        assert (o.kind, o.time, o.new_combo, o.last_in_combo) == (KIND[t["type"]], t["t"], t["nc"], t["last_in_combo"])
        assert o.end_time == pytest.approx(t["end"], abs=1e-4)
        assert math.dist(o.pos, t["spos"]) < 0.01
        if o.kind is Kind.SLIDER:
            assert o.path.length == pytest.approx(t["extra"]["distance"], abs=1e-3)
            assert math.dist(o.ball_at(o.end_time), t["extra"]["end_pos"]) < 0.05
            nested = [n for n in t["nested"] if n["type"] in NESTED]
            assert [n.kind for n in o.nested] == [NESTED[n["type"]] for n in nested]
            for ours, theirs in zip(o.nested, nested):
                assert ours.time == pytest.approx(theirs["t"], abs=1e-4)
                assert math.dist(o.ball_at(ours.time), theirs["pos"]) < 0.05


BASIC = [lz.GREAT, lz.GREAT, lz.OK, lz.GREAT, lz.MEH, lz.GREAT, lz.MISS]


def _scripted(bm):
    """The oracle's `health` script: a fixed mix of results for every judgement in the map, in lazer's order."""
    k = 0
    for o in bm.objects:
        duration = o.end_time - o.time
        if o.kind is Kind.SLIDER:
            nested = [lz.HEAD] + [lz.NESTED_KIND[n.kind] for n in o.nested]
        elif o.kind is Kind.SPINNER:
            nested = ([lz.SPIN_TICK] * (rules.spins_required(bm.difficulty.od, duration) + 2)
                      + [lz.SPIN_BONUS] * rules.bonus_spins(bm.difficulty.od, duration))
        else:
            nested = []
        any_hit = False
        for kind in nested:
            k += 1
            if lz.MAX_RESULT[kind] == lz.GREAT:
                result = BASIC[k % 7]
            else:
                result = lz.MIN_RESULT[kind] if k % 5 == 0 else lz.MAX_RESULT[kind]
            any_hit |= lz.is_hit(result)
            yield o.index, kind, result, False, False
        k += 1
        kind = {Kind.CIRCLE: lz.CIRCLE, Kind.SLIDER: lz.SLIDER, Kind.SPINNER: lz.SPINNER}[o.kind]
        result = (lz.IGNORE_HIT if any_hit else lz.IGNORE_MISS) if kind == lz.SLIDER else BASIC[k % 7]
        yield o.index, kind, result, o.new_combo, o.last_in_combo


def test_health_changes_match_lazer(oracle):
    """Every judgement's health change, including the end-of-combo bonus, for scripted results on every map."""
    for line in oracle("health", *map(str, MAPS)).splitlines():
        lazer = json.loads(line)
        bm = beatmap.parse(lazer["file"])
        health = lz.Health(bm.objects, bm.breaks, bm.difficulty.hp)
        ours = list(_scripted(bm))
        assert len(ours) == len(lazer["log"])
        for (index, kind, result, nc, lic), entry in zip(ours, lazer["log"]):
            assert [index, kind, result] == entry[:3]
            assert health.increase(kind, result, nc, lic) == pytest.approx(entry[3], abs=1e-12)


def test_back_to_back_breaks_pause_drain_once():
    """Two breaks with no object between them give the same no-drain period twice: health must not rise."""
    times = [1000, 1500, 2000, 30000, 30500, 31000]
    objects = [beatmap.HitObject(i, Kind.CIRCLE, 100, 100, t, t, i in (0, 3)) for i, t in enumerate(times)]
    health = lz.Health(objects, [(3000.0, 10000.0), (12000.0, 25000.0)], 6)
    health.apply(2000, lz.CIRCLE, lz.MISS)
    before = health.value
    health.drain_to(30000)
    assert health.value == before < 1


def test_fail_conditions_match_lazer(oracle):
    path = ROOT / "data/maps/129891.osu"
    if not path.exists():
        pytest.skip("download data/maps/129891.osu")
    bm = beatmap.parse(path)
    circle = next(o for o in bm.objects if o.kind is Kind.CIRCLE)
    slider = next(o for o in bm.objects if o.kind is Kind.SLIDER)
    cases = [(0, lz.CIRCLE, lz.GREAT, circle), (0, lz.CIRCLE, lz.MEH, circle), (0, lz.TAIL, lz.IGNORE_MISS, None),
             (0, lz.SLIDER, lz.IGNORE_HIT, slider), (0.05, lz.CIRCLE, lz.MISS, circle),
             (0.2, lz.CIRCLE, lz.MISS, circle), (0, lz.SPIN_TICK, lz.SMALL_BONUS, None)]
    lines = oracle("failtest", str(path)).splitlines()
    assert len(lines) == len(cases)
    for line, (start, kind, result, obj) in zip(lines, cases):
        health = lz.Health(bm.objects, bm.breaks, bm.difficulty.hp)
        health.value, health._t = start, math.inf  # no drain: just this judgement
        health.apply(math.inf, kind, result, obj is not None and obj.new_combo, obj is not None and obj.last_in_combo)
        expected_health, expected_failed = line.split("health after ")[1].split(" failed=")
        assert f"{health.value:.4f}" == expected_health
        assert (health.failed_at is not None) == (expected_failed.strip() == "True")


def test_v3_maps_are_shifted_24ms():
    path = ROOT / "data/maps/75.osu"
    if not path.exists():
        pytest.skip("download data/maps/75.osu")
    lines = [line.strip() for line in path.read_text().splitlines()]
    first = lines[lines.index("[HitObjects]") + 1]
    bm = beatmap.parse(path)
    assert bm.format_version == 3 and bm.objects[0].time == float(first.split(",")[2]) + 24


def test_legacy_path_quirks():
    """lazer keeps a repeated final control point in its bezier, never extends a path ending on a repeated
    point, and treats a legacy perfect curve that lacks exactly 3 points as a bezier."""
    from osu_ai.curves import SliderPath

    doubled_end = SliderPath.from_osu("B|50:100|100:0|100:0", (0, 0), None)
    single_end = SliderPath.from_osu("B|50:100|100:0", (0, 0), None)
    assert math.dist(doubled_end.at(0.5), single_end.at(0.5)) > 1
    flat = SliderPath.from_osu("P|50:0|50:0", (0, 0), 80.0)  # collinear perfect curve: linear, and not extended
    assert flat.length == pytest.approx(50) and flat.end == pytest.approx((50, 0))
    assert SliderPath.from_osu("L|50:0", (0, 0), 80.0).end == pytest.approx((80, 0))
    four = SliderPath.from_osu("P|50:50|100:0|150:50", (0, 0), None)
    assert four.points == SliderPath.from_osu("B|50:50|100:0|150:50", (0, 0), None).points
