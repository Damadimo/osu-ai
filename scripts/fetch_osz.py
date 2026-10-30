"""Download and check real beatmap sets (.osz) from the osu.direct mirror. No osu! login is involved.

  uv run scripts/fetch_osz.py download                  # every set listed in data/real_maps.json
  uv run scripts/fetch_osz.py download --sets 371128 127712
  uv run scripts/fetch_osz.py verify                    # check every data/osz/*.osz
  uv run scripts/fetch_osz.py search --min 2 --max 6 --per-bin 40 --out data/corpus_sets.json

osu.direct limits (https://osu.direct/api/docs): 120 API requests/60 s, burst 25/5 s,
200 downloads per UTC hour, 8 GiB per UTC day, at most 5 parallel downloads per IP.
This script runs one request at a time, waits between requests, and stops on 429/403/503.
The SR values from osu.direct search match the osu!lazer 2026.1005.0 difficulty calculator.

osu!lazer deletes an .osz after importing it, so import a copy, or run `download` again afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import httpx

API = "https://osu.direct/api"
UA = {"User-Agent": "osu-ai/0.1"}  # catboy.best rejects UAs it dislikes; keep it short and stable
OSZ_DIR = Path("data/osz")
AUDIO_EXT = (".mp3", ".ogg", ".wav")


def check_osz(path: Path) -> list[str]:
    """Problems found in an .osz file (empty list = OK)."""
    problems = []
    try:
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                problems.append("CRC error")
            names = {n.lower(): n for n in z.namelist()}
            osus = [n for n in z.namelist() if n.lower().endswith(".osu")]
            if not osus:
                problems.append("no .osu files")
            for n in osus:
                text = z.read(n).decode("utf-8-sig", errors="replace")
                audio = next((l.split(":", 1)[1].strip() for l in text.splitlines() if l.startswith("AudioFilename")), None)
                if not audio or audio.lower() not in names:
                    problems.append(f"{n}: audio {audio!r} missing")
    except zipfile.BadZipFile:
        problems.append("not a zip file")
    return problems


def stop(resp: httpx.Response) -> None:
    body = resp.read()[:200]  # works for streamed and normal responses
    sys.exit(f"stopping: HTTP {resp.status_code} {body!r} Retry-After={resp.headers.get('Retry-After')}")


def download(http: httpx.Client, set_id: int, no_video: bool = True) -> None:
    out = OSZ_DIR / f"{set_id}.osz"
    if out.exists() and not check_osz(out):
        print(f"{set_id}: already present")
        return
    url = f"{API}/d/{set_id}" + ("?noVideo" if no_video else "")
    tmp = out.with_suffix(".osz.part")
    with http.stream("GET", url) as resp:
        if resp.status_code != 200:
            stop(resp)
        with tmp.open("wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
        budget = resp.history[0].headers.get("X-Download-Budget-Remaining") if resp.history else None
    problems = check_osz(tmp)
    if problems:
        tmp.unlink()
        sys.exit(f"{set_id}: bad download: {problems}")
    tmp.replace(out)
    print(f"{set_id}: {out.stat().st_size / 1e6:.1f} MB, hourly downloads left: {budget}")


def search(http: httpx.Client, lo: float, hi: float, per_bin: int, bin_width: float, ar: tuple[float, float],
           od: tuple[float, float], length: tuple[int, int]) -> list[dict]:
    """Most-played ranked osu!standard sets with a difficulty in each SR bin. One entry per set."""
    seen: set[int] = set()
    rows = []
    b = lo
    while b < hi - 1e-9:
        b_hi = min(hi, round(b + bin_width, 3))
        # Meilisearch filter inside [...]; array fields match per element, so re-check each difficulty below.
        q = (f"[beatmaps.mode_int = 0 AND beatmaps.difficulty_rating {b} TO {b_hi} AND beatmaps.ar {ar[0]} TO {ar[1]}"
             f" AND beatmaps.accuracy {od[0]} TO {od[1]} AND beatmaps.hit_length {length[0]} TO {length[1]}]")
        got, offset = 0, 0
        while got < per_bin:
            params = {"q": q, "mode": 0, "status": 1, "amount": 100, "offset": offset, "sort": "play_count:desc"}
            resp = http.get(f"{API}/v2/search", params=params)
            if resp.status_code != 200:
                stop(resp)
            sets = resp.json()
            for s in sets:
                diffs = [d for d in s["beatmaps"] if d.get("mode_int") == 0 and b <= d["difficulty_rating"] < b_hi
                         and ar[0] <= d["ar"] <= ar[1] and od[0] <= d["accuracy"] <= od[1]
                         and length[0] <= d["hit_length"] <= length[1]]
                if not diffs or s["id"] in seen or s.get("nsfw"):
                    continue
                seen.add(s["id"])
                d = min(diffs, key=lambda d: abs(d["difficulty_rating"] - (b + b_hi) / 2))
                rows.append(dict(set_id=s["id"], beatmap_id=d["id"], artist=s["artist"], title=s["title"], version=d["version"],
                                 sr=d["difficulty_rating"], bpm=d["bpm"], ar=d["ar"], od=d["accuracy"], cs=d["cs"], hp=d["drain"],
                                 length_s=d["hit_length"], n_circles=d["count_circles"], n_sliders=d["count_sliders"],
                                 n_spinners=d["count_spinners"], video=s.get("video"), storyboard=s.get("storyboard"),
                                 set_play_count=s.get("play_count"), sr_bin=[b, b_hi]))
                got += 1
                if got >= per_bin:
                    break
            if len(sets) < 100:
                break
            offset += 100
            time.sleep(1.2)
        print(f"{b:.2f}-{b_hi:.2f}*: {got} sets (filter matched {resp.headers.get('totalhits')})")
        b = b_hi
        time.sleep(1.2)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--sets", type=int, nargs="*")
    d.add_argument("--maps", type=Path, default=Path("data/real_maps.json"))
    d.add_argument("--with-video", action="store_true")
    sub.add_parser("verify")
    s = sub.add_parser("search")
    s.add_argument("--min", type=float, default=2.0)
    s.add_argument("--max", type=float, default=6.0)
    s.add_argument("--bin", type=float, default=0.5)
    s.add_argument("--per-bin", type=int, default=20)
    s.add_argument("--ar", type=float, nargs=2, default=(0, 10))
    s.add_argument("--od", type=float, nargs=2, default=(0, 10))
    s.add_argument("--length", type=int, nargs=2, default=(60, 240), help="drain seconds")
    s.add_argument("--out", type=Path, default=Path("data/corpus_sets.json"))
    args = ap.parse_args()

    OSZ_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "verify":
        for p in sorted(OSZ_DIR.glob("*.osz")):
            print(p.name, check_osz(p) or "ok")
        return
    with httpx.Client(headers=UA, timeout=120, follow_redirects=True) as http:
        if args.cmd == "download":
            ids = args.sets or list(dict.fromkeys(m["set_id"] for m in json.loads(args.maps.read_text())))
            for i, sid in enumerate(ids):
                download(http, sid, no_video=not args.with_video)
                if i + 1 < len(ids):
                    time.sleep(3)
        else:
            rows = search(http, args.min, args.max, args.per_bin, args.bin, tuple(args.ar), tuple(args.od), tuple(args.length))
            args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
            print(f"{len(rows)} sets -> {args.out}")


if __name__ == "__main__":
    main()
