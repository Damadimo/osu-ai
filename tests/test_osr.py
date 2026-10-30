"""The .osr writer: lazer's header layout, osrparse decoding, and frames that replay the run exactly."""

from __future__ import annotations

import io
import json
import lzma
import random
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from osrparse import GameMode, Mod

from osu_ai import beatmap, judge, osr
from osu_ai.replay import Replay

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / "data" / "lazer_osu" / "827932.osu"  # Cold Green Eyes [apple's Insane], v14
MD5 = "c29b4567c8e831ae1ba140ae5caeda16"


def sample_replay() -> Replay:
    r = Replay()
    t = -40.0  # replays may start before 0 (first object earlier than its approach time)
    while t <= 2000:
        r.move(t, (100 + 0.1 * t, 50 + 0.05 * t))
        t += 8
    for down in (300.4, 520.6, 700, 980):  # taps
        r.press(down, down + 45)
    r.press(1200, 1700)  # a slider hold...
    r.press(1450, 1500)  # ...with a tap during it: must still be a fresh press on the free key
    r.press(1800, 1830)
    return r


def lazer_decode(raw: bytes) -> dict:
    """Mirror of LegacyScoreDecoder.Parse's header logic (lazer 2026.1005.0)."""
    f = io.BytesIO(raw)

    def u(fmt):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    def string():
        if u("<B") == 0:
            return None
        n = shift = 0
        while True:
            b = u("<B")
            n |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                break
        return f.read(n).decode()

    d = {"ruleset": u("<B"), "version": u("<i"), "md5": string(), "user": string(), "replay_md5": string(),
         "counts": [u("<H") for _ in range(6)], "score": u("<i"), "combo": u("<H"), "perfect": u("<?"),
         "mods": u("<i"), "hp": string(), "ticks": u("<q")}
    n = u("<i")
    d["replay"] = f.read(n)
    if d["version"] >= 20140721:
        d["online_id"] = u("<q")
    if d["version"] >= 30000001:
        n = u("<i")
        d["score_info"] = f.read(n) if n > 0 else None
    d["trailing"] = f.read()
    return d


def write(tmp_path, replay=None, score=None) -> tuple[Path, Replay]:
    replay = replay or sample_replay()
    score = score or judge.Score([300, 100, 50, 0, 300], 2)
    path = osr.export(SimpleNamespace(md5=MD5), replay, score, tmp_path / "x.osr", "osu-ai")
    return path, replay


def test_header_is_what_lazer_reads_without_adding_classic(tmp_path):
    path, _ = write(tmp_path)
    d = lazer_decode(path.read_bytes())
    assert d["ruleset"] == 0 and d["version"] == osr.VERSION >= 30000000  # below that lazer adds Classic
    assert d["md5"] == MD5 and d["user"] == "osu-ai"
    assert d["counts"] == [2, 1, 1, 0, 0, 1] and d["score"] == 750 and d["combo"] == 2 and not d["perfect"]
    assert d["mods"] == 0 and d["online_id"] == 0 and d["score_info"] is None and d["trailing"] == b""
    assert d["ticks"] > osr._EPOCH_TICKS + 1_700_000_000 * 10_000_000  # a sane UTC date
    rep = d["replay"]
    size = struct.unpack_from("<q", rep, 5)[0]
    text = osr._unlzma(rep)  # raw LZMA1 up to the header's size, like SharpCompress in lazer
    assert size == len(text) > 0
    assert text.decode("ascii").endswith(",-12345|0|0|0")


def test_osrparse_decodes_it(tmp_path):
    path, replay = write(tmp_path)
    r = osr.read(path)
    assert r.mode is GameMode.STD and r.game_version == osr.VERSION and r.mods == Mod.NoMod
    assert r.beatmap_hash == MD5 and r.username == "osu-ai" and r.replay_id == 0
    assert (r.count_300, r.count_100, r.count_50, r.count_miss) == (2, 1, 1, 1)
    assert len(r.replay_data) == len(osr.frames(replay))


def test_frames_every_16_ms_plus_exact_key_events(tmp_path):
    path, replay = write(tmp_path)
    t, times, prev_keys, downs = 0, [], 0, []
    for e in osr.read(path).replay_data:
        t += e.time_delta
        times.append(t)
        x, y = replay.cursor_at(t)
        assert (e.x, e.y) == pytest.approx((x, y), abs=1e-3)
        keys = int(e.keys)
        for mask in osr.KEYS:
            if keys & mask and not prev_keys & mask:
                downs.append((t, mask))
        prev_keys = keys
    assert times[0] == -40 and times[-1] == 2000
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert min(gaps) >= 1 and max(gaps) <= osr.SAMPLE_MS
    for p in replay.presses:
        assert round(p.down) in times and round(p.up) in times
    # every press shows up as a fresh key-down on alternating keys, the tap during the hold included
    assert [t for t, _ in downs] == [300, 521, 700, 980, 1200, 1450, 1800]
    assert [m for _, m in downs] == [osr.KEYS[i % 2] for i in range(7)]


def test_presses_read_back(tmp_path):
    path, replay = write(tmp_path)
    got = osr.presses(osr.read(path))
    assert [(p.down, p.up) for p in got] == [(300, 345), (521, 566), (700, 745), (980, 1025), (1200, 1700),
                                              (1450, 1500), (1800, 1830)]
    assert [p.key for p in got] == [0, 1, 0, 1, 0, 1, 0]
    p = got[4]
    assert (p.x, p.y) == pytest.approx(replay.cursor_at(1200), abs=1e-3)


def test_key_alternation_matches_the_judge(tmp_path):
    r = Replay()
    r.move(0, (10, 10))
    r.move(1000, (20, 20))
    r.press(100, 900)  # key 0
    r.press(200, 800)  # key 1
    r.press(300, 350)  # key 0 again, still held: no new press (judge.replay_inputs merges it too)
    r.press(820, 830)  # key 1, released at 800: a new press
    r.press(900, 950)  # key 0, which goes up at 900 exactly: lazer must see it go up first
    path, _ = write(tmp_path, r)
    got = osr.presses(osr.read(path))
    assert [(p.down, p.up, p.key) for p in got] == [(100, 899, 0), (200, 800, 1), (820, 830, 1), (900, 950, 0)]
    from osu_ai.judge import replay_inputs
    downs = [(t, k) for t, _, _, d, _ in replay_inputs(r) for k in d]
    assert downs == [(100, 0), (200, 1), (820, 1), (900, 0)]


def test_score_info_block_of_lazer_exports(tmp_path):
    path, _ = write(tmp_path)
    assert osr.score_info(path) is None
    info = {"statistics": {"great": 3, "miss": 1}, "rank": "B"}
    raw = path.read_bytes()[:-4]
    block = osr._lzma(json.dumps(info).encode())
    lazer_like = tmp_path / "lazer.osr"
    lazer_like.write_bytes(raw + struct.pack("<i", len(block)) + block)
    assert osr.score_info(lazer_like) == info
    assert osr.read(lazer_like).count_300 == 2  # osrparse ignores the block


def test_old_format_maps_are_written_24_ms_early_like_lazers_encoder(tmp_path):
    # lazer shifts v4-and-older maps' objects by +24 ms (beatmap.py does too) and adds the same 24 ms to
    # every replay frame time when decoding (LegacyScoreDecoder.beatmapOffset); its encoder subtracts it.
    replay = sample_replay()
    old = SimpleNamespace(md5=MD5, format_version=3)
    path = osr.export(old, replay, judge.Score([300], 1), tmp_path / "v3.osr", "osu-ai")
    assert osr.time_offset(old) == 24 and osr.time_offset(SimpleNamespace(format_version=5)) == 0
    t, downs, prev = 24, [], 0  # lazer's decoder: lastTime starts at beatmapOffset
    for e in osr.read(path).replay_data:
        t += e.time_delta
        downs += [t for mask in osr.KEYS if int(e.keys) & mask and not prev & mask]
        prev = int(e.keys)
    assert downs == [round(p.down) for p in sorted(replay.presses, key=lambda p: p.down)]
    got = osr.presses(osr.read(path), osr.time_offset(old))
    assert [p.down for p in got] == downs
    assert [p.down for p in osr.presses(osr.read(path))] == [d - 24 for d in downs]


def test_python_lzma_accepts_the_known_size_header():
    data = b"0|1|2|0," * 500
    assert lzma.decompress(osr._lzma(data), format=lzma.FORMAT_AUTO) == data


@pytest.mark.skipif(not MAP.exists(), reason="data/lazer_osu/827932.osu missing")
def test_human_replay_round_trip(tmp_path):
    from osu_ai.synth_replay import human_replay

    bm = beatmap.parse(MAP)
    assert bm.md5 == MD5
    replay = human_replay(bm, random.Random(3))
    score = judge.score(bm, replay)
    path = osr.export(bm, replay, score, tmp_path / "h.osr", "osu-ai")
    r = osr.read(path)
    assert r.beatmap_hash == bm.md5 and r.count_300 == score.counts[300]
    got = osr.presses(r)
    want = sorted(replay.presses, key=lambda p: p.down)
    assert len(got) == len(want)
    assert all(g.down == round(w.down) and g.key == i % 2 for i, (g, w) in enumerate(zip(got, want)))
