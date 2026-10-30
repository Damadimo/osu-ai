"""Write .osr replays that osu!lazer imports and plays back exactly; read the ones it exports.

lazer decodes the legacy format (Scoring/Legacy/LegacyScoreDecoder.cs). We write version
30000019, its own LegacyScoreEncoder.LATEST_VERSION, so the decoder expects the score-info
block after the online id: we write it empty. A version below 30000000 would make lazer add
the Classic mod, which changes judgement. Mods are none and the online id is 0, so an import
never looks anything up online. The LZMA header carries the real uncompressed size, as lazer's
own encoder writes it. Cursor frames come every 16 ms (lazer records at 60 Hz) plus one at
every exact key event. Press i uses key i % 2, the contract judge.replay_inputs relies on, so
that the simulator judges exactly what lazer will play back. For maps in format v4 and older,
lazer's object times carry a 24 ms shift (as ours do) and its decoder adds the same 24 ms to
replay times, so frames are written 24 ms early and read back 24 ms later, as lazer does.

For watching locally only: never submit a generated replay anywhere.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import lzma
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from osrparse import Key
from osrparse import Replay as OsrReplay

from .beatmap import EARLY_VERSION_OFFSET_MS, Beatmap
from .judge import Score
from .replay import Replay as Recording

VERSION = 30000019  # lazer's LegacyScoreEncoder.LATEST_VERSION
SAMPLE_MS = 16  # cursor frames between key events (lazer records at 60 Hz)
KEYS = (int(Key.K1 | Key.M1), int(Key.K2 | Key.M2))  # left, right
LEFT_BITS, RIGHT_BITS = 1 | 4, 2 | 8  # lazer: M1 or K1 is left, M2 or K2 is right
MAX_COORD = 131072.0  # lazer clamps replay coordinates to this
_EPOCH_TICKS = 621355968000000000  # .NET ticks at the Unix epoch


@dataclass(frozen=True)
class Press:
    """One key press read back from a replay: times in ms of beatmap time, cursor in osu!px."""

    down: float
    up: float
    x: float
    y: float
    key: int  # 0 left, 1 right


def time_offset(bm) -> int:
    """ms that lazer adds to every replay frame time of this map (LegacyScoreDecoder's beatmapOffset):
    24 for format v4 and older, whose object times carry the same shift; its encoder subtracts it."""
    return int(EARLY_VERSION_OFFSET_MS) if getattr(bm, "format_version", 14) < 5 else 0


def frames(replay: Recording) -> list[tuple[int, float, float, int]]:
    """(time ms, x, y, key bits) every SAMPLE_MS and at every key change; times strictly increase.
    A press while its key is still down is no new press (the two merge, as in judge.replay_inputs);
    one that starts the moment its key goes up gets that key up 1 ms earlier, so that lazer sees
    it go up and down again. Every press is held at least 1 ms."""
    replay.finalize()
    spans: list[list[int]] = []  # [down, up, key index]
    last: list[list[int] | None] = [None, None]
    held_until = [-math.inf, -math.inf]  # unrounded end of each key's last span, as the judge keeps it
    for i, p in enumerate(replay.presses):  # finalize() sorted them by down
        down, k = round(p.down), i % 2
        up = max(round(p.up), down + 1)
        prev = last[k]
        if p.down < held_until[k]:
            held_until[k] = max(held_until[k], p.up)
            prev[1] = max(prev[1], up)  # still held: no new press
            continue
        held_until[k] = max(p.up, p.down)
        if prev is not None and down <= prev[1]:
            if down - 1 > prev[0]:
                prev[1] = down - 1
            else:
                down, up = prev[1] + 1, max(up, prev[1] + 2)
        span = [down, up, k]
        spans.append(span)
        last[k] = span
    starts = [f[0] for f in replay.frames] + [s[0] for s in spans]
    ends = [f[0] for f in replay.frames] + [s[1] for s in spans]
    if not starts:
        return []
    t0, t1 = math.floor(min(starts)), math.ceil(max(ends))
    times = set(range(t0, t1 + 1, SAMPLE_MS)) | {t1}
    times |= {s[0] for s in spans} | {s[1] for s in spans}
    out = []
    for t in sorted(times):
        bits = 0
        for down, up, k in spans:
            if down <= t < up:
                bits |= KEYS[k]
        x, y = replay.cursor_at(t)
        x, y = max(-MAX_COORD, min(MAX_COORD, x)), max(-MAX_COORD, min(MAX_COORD, y))
        if (x, y) == (256.0, -500.0):  # lazer drops leading frames at exactly this spot
            y = -499.999
        out.append((t, x, y, bits))
    return out


def _uleb(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _string(s: str) -> bytes:
    b = s.encode()
    return b"\x0b" + _uleb(len(b)) + b


def _lzma(data: bytes) -> bytes:
    """LZMA 'alone' stream with the real size in its header (Python writes -1 for unknown)."""
    c = bytearray(lzma.compress(data, format=lzma.FORMAT_ALONE,
                                filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 21}]))
    c[5:13] = struct.pack("<q", len(data))
    return bytes(c)


def _num(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def encode(beatmap_md5: str, replay_frames: list[tuple[int, float, float, int]], username: str,
           counts: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0), total: int = 0, max_combo: int = 0,
           perfect: bool = False, when: datetime.datetime | None = None) -> bytes:
    """An .osr file. counts: n300, n100, n50, geki, katu, miss (lazer shows them; it never re-judges on import)."""
    parts, prev = [], 0
    for t, x, y, keys in replay_frames:
        t = round(t)  # stable's format (and osrparse) only has whole-ms deltas
        parts.append(f"{t - prev}|{_num(x)}|{_num(y)}|{int(keys)}")
        prev = t
    data = _lzma(",".join(parts + ["-12345|0|0|0"]).encode("ascii"))
    when = when or datetime.datetime.now(datetime.timezone.utc)
    ticks = _EPOCH_TICKS + round(when.timestamp() * 10_000_000)
    replay_md5 = hashlib.md5(f"{username}-{beatmap_md5}-{ticks}".encode()).hexdigest()
    return b"".join([
        struct.pack("<Bi", 0, VERSION), _string(beatmap_md5), _string(username), _string(replay_md5),
        struct.pack("<6H", *counts), struct.pack("<iH?i", total, max_combo, perfect, 0),
        _string(""), struct.pack("<q", ticks), struct.pack("<i", len(data)), data,
        struct.pack("<q", 0),  # online score id: 0 means none, so nothing is looked up online
        struct.pack("<i", 0),  # score-info block: empty
    ])


def export(bm: Beatmap, replay: Recording, score: Score, path: Path, username: str) -> Path:
    c = score.counts
    total = getattr(score, "total_score", 0) or sum(score.judgments)  # lazer's standardised score if judged so
    shift = time_offset(bm)
    written = [(t - shift, x, y, keys) for t, x, y, keys in frames(replay)]
    raw = encode(bm.md5, written, username, tuple(min(n, 0xFFFF) for n in (c[300], c[100], c[50], 0, 0, c[0])),
                 min(int(total), 2**31 - 1), min(score.max_combo, 0xFFFF), c[0] == 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def read(path: Path) -> OsrReplay:
    return OsrReplay.from_path(str(path))


def presses(replay: OsrReplay, offset_ms: float = 0.0) -> list[Press]:
    """Key presses in a decoded replay (lazer's own exports included), in time order. Like lazer,
    frames that go back in time are skipped; times are beatmap time once offset_ms is
    time_offset(bm) of the replay's map."""
    out: list[Press] = []
    open_: dict[int, list[float]] = {}  # key -> [down, x, y]
    t, last = float(offset_ms), None
    for e in replay.replay_data:
        t += e.time_delta
        if last is not None and t < last:
            continue
        last = t
        bits = int(e.keys)
        for k, mask in ((0, LEFT_BITS), (1, RIGHT_BITS)):
            if bits & mask and k not in open_:
                open_[k] = [t, e.x, e.y]
            elif not bits & mask and k in open_:
                down, x, y = open_.pop(k)
                out.append(Press(down, t, x, y, k))
    out += [Press(down, last, x, y, k) for k, (down, x, y) in open_.items()]
    return sorted(out, key=lambda p: p.down)


def score_info(path: Path) -> dict | None:
    """The JSON score-info block lazer writes into its own .osr files (statistics, rank, mods...),
    or None if the file has none (ours, or a stable replay)."""
    raw = Path(path).read_bytes()
    pos = 0

    def take(fmt: str):
        nonlocal pos
        v = struct.unpack_from(fmt, raw, pos)
        pos += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v

    def string() -> None:
        nonlocal pos
        if take("<B") == 0x0B:
            n = shift = 0
            while True:
                b = take("<B")
                n |= (b & 0x7F) << shift
                shift += 7
                if not b & 0x80:
                    break
            pos += n

    take("<B")
    version = take("<i")
    for _ in range(3):
        string()
    take("<6HiH?i")
    string()
    take("<q")
    n = take("<i")
    pos += n  # replay data
    take("<q")
    if version < 30000001 or pos + 4 > len(raw):
        return None
    n = take("<i")
    if n <= 0:
        return None
    return json.loads(_unlzma(raw[pos:pos + n]))


def _unlzma(block: bytes) -> bytes:
    """Decode an LZMA 'alone' block the way lazer does: raw LZMA1 up to the size in its header."""
    props, size = block[0], struct.unpack_from("<q", block, 5)[0]
    lc, lp, pb = props % 9, (props // 9) % 5, props // 45
    dict_size = struct.unpack_from("<I", block, 1)[0]
    dec = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[
        {"id": lzma.FILTER_LZMA1, "dict_size": dict_size, "lc": lc, "lp": lp, "pb": pb}])
    return dec.decompress(block[13:], max_length=size if size >= 0 else -1)
