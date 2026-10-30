"""Train the vision model on osu!lazer-accurate frames rendered on the fly (lazer_frames.py).

  uv run --extra train scripts/train_lazer.py --steps 30000 --out models/osunet-lazer.safetensors

Worker processes render 4-frame stacks at 320x240 from random maps, times and plays (perfect
or imperfect: timing noise, misses, note lock), with both settings of lazer's hit animations
and snaking sliders, small calibration errors (zoom, shift), brightness/gamma/blur/noise and
jittered frame gaps. The maps of the real-game tests are held out (TEST_SETS) and so are the
simulator demo maps, so validation on them is a fair test of maps never seen.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import itertools
import math
import multiprocessing as mp
import random
import statistics
import time
import zipfile
from collections import deque
from pathlib import Path

import numpy as np

TEST_SETS = {371128, 127712, 128931, 513590, 10435, 190390, 163112, 1}  # real-game test maps + DISCO
SIZE = (320, 240)
STRIDE = 8
GRID = (SIZE[1] // STRIDE, SIZE[0] // STRIDE)
MAX_DT_MS = 1800
FRAME_MS = 1000 / 60


def _set_id(text: str) -> int | None:
    for line in text.splitlines()[:80]:
        if line.startswith("BeatmapSetID:"):
            v = line.split(":", 1)[1].strip()
            return int(v) if v.lstrip("-").isdigit() else None
    return None


def map_sources() -> list[tuple[str, str]]:
    """(name, .osu text) for every usable osu!standard map: loose files and the .osz sets."""
    out, seen = [], set()
    for path in sorted(glob.glob("data/corpus/*.osu") + glob.glob("data/lazer_osu/*.osu")):
        raw = Path(path).read_bytes()
        key = hashlib.md5(raw).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append((path, raw.decode("utf-8-sig", errors="replace")))
    for path in sorted(glob.glob("data/osz/*.osz")):
        try:
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name.lower().endswith(".osu"):
                        raw = z.read(name)
                        key = hashlib.md5(raw).hexdigest()
                        if key not in seen:
                            seen.add(key)
                            out.append((f"{path}:{name}", raw.decode("utf-8-sig", errors="replace")))
        except zipfile.BadZipFile:
            continue
    return out


def split(sources: list[tuple[str, str]]) -> tuple[list, list, list]:
    """train / val (5% of non-test maps, by name hash) / test (the held-out sets)."""
    train, val, test = [], [], []
    for name, text in sources:
        if "Mode: 1" in text or "Mode: 2" in text or "Mode: 3" in text:
            continue
        if _set_id(text) in TEST_SETS or name.startswith("data/maps/"):
            test.append((name, text))
        elif int(hashlib.md5(name.encode()).hexdigest(), 16) % 20 == 0:
            val.append((name, text))
        else:
            train.append((name, text))
    return train, val, test


# ---- worker side (no MLX here) ----

_MAPS: list = []
_HARD: list = []  # per map: hit times of circles in streams or stacks (where the model struggles)
HARD_FRAC = 0.5  # share of samples taken just before such a circle


def _hard_times(bm) -> list[float]:
    """Circles/slider heads within 130 ms of a neighbour, or stacked within 12 osu!px of the previous."""
    from osu_ai.beatmap import Kind

    objs = [o for o in bm.objects if o.kind is not Kind.SPINNER]
    out = []
    for a, b in zip(objs, objs[1:]):
        if b.time - a.end_time < 130 or abs(b.x - a.x) + abs(b.y - a.y) < 12:
            out += [a.time, b.time]
    return sorted(set(out))


MAPS_PER_WORKER = 160  # parsed maps are big; each worker keeps its own random share


def _init_worker(sources: list[tuple[str, str]]) -> None:
    import os
    import tempfile

    from osu_ai import beatmap

    global _MAPS, _HARD
    _MAPS, _HARD = [], []
    tmp = Path(tempfile.mkdtemp())
    sources = random.Random(os.getpid()).sample(sources, min(len(sources), MAPS_PER_WORKER))
    for i, (name, text) in enumerate(sources):
        path = tmp / f"{i}.osu"
        path.write_text(text, encoding="utf-8")
        try:
            bm = beatmap.parse(path)
        except Exception:
            continue
        finally:
            os.unlink(path)
        if len(bm.objects) >= 30:
            _MAPS.append(bm)
            _HARD.append(_hard_times(bm))


def _augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brightness, gamma, blur and noise, the same for all 4 frames of a stack."""
    x = x / 255.0
    if rng.random() < 0.8:
        x = np.clip(x * rng.uniform(0.85, 1.15), 0, 1) ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.4:
        k = rng.uniform(0.2, 0.6)  # cheap separable 3-tap blur
        x = x * (1 - k) + k / 2 * (np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0))
        x = x * (1 - k) + k / 2 * (np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1))
    if rng.random() < 0.5:
        x = x + rng.normal(0, rng.uniform(0.002, 0.012), x.shape)
    return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)


def sample_stack(seed: int, fixed_look=None):
    """One training example: (frames (H, W, 4) uint8, objs (n, 3), slider (3,), spinner, radius px)."""
    from osu_ai.lazer_frames import Look, View, labels, perfect, render, sample_outcome

    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    k = rng.randrange(len(_MAPS))
    bm, hard = _MAPS[k], _HARD[k]
    if rng.random() < 0.35:
        outcome = perfect(bm)
    else:
        outcome = sample_outcome(bm, rng, hit_rate=rng.uniform(0.75, 1.0), timing_sd=rng.uniform(4, 30))
    look = fixed_look or Look(hit_animations=rng.random() < 0.5, snaking=rng.random() < 0.5)
    view = View(SIZE)
    if rng.random() < 0.6:
        view = View(SIZE, zoom=rng.uniform(0.985, 1.015), shift=(rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)))
    preempt = bm.difficulty.preempt
    start, end = bm.objects[0].time - preempt, bm.objects[-1].end_time + 300
    if hard and rng.random() < HARD_FRAC:
        t = rng.choice(hard) - rng.uniform(-40, 300)  # the last few hundred ms before a hard circle
    else:
        t = rng.uniform(start, end)
    times = [t]
    dropped = rng.randrange(3) if rng.random() < 0.1 else -1  # now and then the capture skips a frame
    for k in range(3):
        gap = min(22.0, max(12.0, rng.gauss(FRAME_MS, 1.0)))
        times.append(times[-1] - (2 * gap if k == dropped else gap))
    frames = np.stack([render(bm, tt, outcome, look, view) for tt in reversed(times)], axis=-1)
    objs, slider, spinner = labels(bm, t, outcome, view)
    radius_px = bm.difficulty.radius * view.f
    return _augment(frames, nrng), np.array(objs, np.float32).reshape(-1, 3), np.array(slider, np.float32), spinner, radius_px


_REAL: list = []  # (frames memmap, game time per frame, beatmap, outcome, valid end indices)


def _init_real(items: list[dict]) -> None:
    """Load aligned recordings (prepared by prepare_real) for sampling in this worker."""
    from osu_ai import beatmap
    from osu_ai.lazer_frames import Outcome

    global _REAL
    _REAL = []
    for it in items:
        frames = np.load(it["frames"], mmap_mode="r")
        game_ms = np.load(it["times"])
        bm = beatmap.parse(it["map"])
        outcome = Outcome({int(k): (v[0], bool(v[1])) for k, v in it["judged"].items()},
                          {int(k): v for k, v in it.get("spin_rate", {}).items()})
        gaps = np.diff(game_ms)
        ok = (gaps > 10) & (gaps < 25)
        valid = [k for k in range(3, len(game_ms)) if ok[k - 3:k].all()]
        if valid:
            _REAL.append((frames, game_ms, bm, outcome, np.array(valid)))


def sample_real(seed: int):
    """A 4-frame stack of real footage, labeled from the beatmap at the last frame's game time."""
    from osu_ai.lazer_frames import View, labels

    rng = np.random.default_rng(seed)
    frames, game_ms, bm, outcome, valid = _REAL[rng.integers(len(_REAL))]
    k = int(valid[rng.integers(len(valid))])
    x = np.moveaxis(np.asarray(frames[k - 3:k + 1]), 0, -1)
    view = View((x.shape[1], x.shape[0]))
    objs, slider, spinner = labels(bm, float(game_ms[k]), outcome, view)
    if rng.random() < 0.5:  # mild: real frames already have the real pipeline's look
        x = _augment(x, rng)
    return x, np.array(objs, np.float32).reshape(-1, 3), np.array(slider, np.float32), spinner, bm.difficulty.radius * view.f


def _init_both(sources, real_items) -> None:
    _init_worker(sources)
    _init_real(real_items)


def make_batch(args: tuple[int, int]):
    seed, n, real_frac = (*args, 0.0) if len(args) == 2 else args
    n_real = int(round(n * real_frac)) if _REAL else 0
    items = [sample_real(seed * 1000 + i) for i in range(n_real)]
    items += [sample_stack(seed * 1000 + i) for i in range(n_real, n)]
    x = np.stack([it[0] for it in items])
    return x, [it[1] for it in items], np.stack([it[2] for it in items]), np.array([it[3] for it in items], np.float32), [it[4] for it in items]


# ---- trainer ----

def evaluate(model, batches) -> dict:
    """How well the detection the policy would aim at (the next circle) matches the truth."""
    import mlx.core as mx

    from osu_ai.vision import decode, decode_next, decode_state

    model.eval()
    found, total, aim, near, timing, hold_ok, spin_ok, ball = 0, 0, [], [], [], [], [], []
    for x, objs, sliders, spinners, radii in batches:
        out = model(mx.array(x.astype(np.float32) / 255.0))
        heat, off, dt, ballmap, state = out[:5]
        dets = decode(heat, off, dt, threshold=0.35, stride=STRIDE)
        if len(out) > 5:
            dets = [n or d for n, d in zip(decode_next(out[5], threshold=0.35, stride=STRIDE), dets)]
        for (ph, ps, bx, by), sl, sp in zip(decode_state(ballmap, state, stride=STRIDE), sliders, spinners):
            hold_ok.append((ph > 0.5) == bool(sl[0]))
            spin_ok.append((ps > 0.5) == bool(sp))
            if sl[0]:
                ball.append(math.dist((bx, by), (sl[1], sl[2])))
        for d, o, r in zip(dets, objs, radii):
            o = o[o[:, 2] <= MAX_DT_MS]
            if len(o) == 0:
                continue
            total += 1
            nx, ny, ndt = o[np.argmin(o[:, 2])]
            if not d:
                continue
            px, py, pdt, _ = d[0]
            dist = math.dist((px, py), (nx, ny))
            aim.append(dist)
            if dist <= r:
                found += 1
                timing.append(abs(pdt - ndt))
                if ndt < 300:
                    near.append(abs(pdt - ndt))
    model.train()
    q = lambda v, f: float(np.percentile(v, f)) if v else float("nan")
    return {"found": found / max(total, 1), "aim_px": q(aim, 50), "timing_ms": q(timing, 50),
            "near_ms": q(near, 50), "near_p90": q(near, 90), "near_25": sum(e <= 25 for e in near) / max(len(near), 1),
            "hold": statistics.mean(hold_ok) if hold_ok else float("nan"),
            "spin": statistics.mean(spin_ok) if spin_ok else float("nan"), "ball_px": q(ball, 50)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("models/osunet-lazer.safetensors"))
    parser.add_argument("--init", help="start from these weights (missing/new layers start fresh)")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--real", type=Path, help="prepared real footage (data/real/prepared.json)")
    parser.add_argument("--real-frac", type=float, default=0.5, help="share of each batch from real footage")
    parser.add_argument("--start-step", type=int, default=0, help="resume the schedule here (with --init)")
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from osu_ai.vision import OsuNet, loss_fn, targets

    mx.set_cache_limit(2 << 30)  # MLX keeps freed buffers; unbounded, that grows until macOS kills us

    train, val, test = split(map_sources())
    print(f"maps: {len(train)} train, {len(val)} val, {len(test)} held-out test", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(2, initializer=_init_worker, initargs=(val,)) as pool:
        val_batches = pool.map(make_batch, [(10_000_000 + i, 100) for i in range(15)])
    with ctx.Pool(2, initializer=_init_worker, initargs=(test,)) as pool:
        test_batches = pool.map(make_batch, [(20_000_000 + i, 100) for i in range(15)])

    model = OsuNet(args.width, stride=STRIDE, next_heads=True)
    if args.init:  # reuse every layer whose shape still fits (e.g. all but the new stem)
        have = dict(tree_flatten(model.parameters()))
        fits = [(k, v) for k, v in mx.load(args.init).items() if k in have and have[k].shape == v.shape]
        model.load_weights(fits, strict=False)
        print(f"init: {len(fits)}/{len(have)} tensors from {args.init}", flush=True)
    mx.eval(model.parameters())
    print(f"OsuNet width {args.width} stride {STRIDE}: "
          f"{sum(v.size for _, v in tree_flatten(model.parameters())) / 1e6:.2f}M parameters", flush=True)
    total = args.start_step + args.steps
    decay = optim.cosine_decay(args.lr, total, 1e-5)
    opt = optim.AdamW(learning_rate=lambda step: decay(step + args.start_step), weight_decay=1e-4)
    step_fn = nn.value_and_grad(model, loss_fn)

    real_train, real_val = [], []
    if args.real:
        import json

        prepared = json.loads(args.real.read_text())
        real_val = [it for it in prepared if it.get("split") == "val"]
        real_train = [it for it in prepared if it.get("split") != "val"]
        print(f"real footage: {len(real_train)} train recordings, {len(real_val)} val", flush=True)
        if real_val:
            with ctx.Pool(2, initializer=_init_both, initargs=([], real_val)) as p:
                real_val_batches = p.map(make_batch, [(30_000_000 + i, 100, 1.0) for i in range(15)])
    pool = ctx.Pool(args.workers, initializer=_init_both, initargs=(train, real_train))
    frac = args.real_frac if real_train else 0.0
    seeds = itertools.count(1 + args.start_step * 7)
    # Bounded prefetch: Pool.imap over an endless generator submits every job at once and queues
    # results without limit; the backlog grew to ~100 GiB (compressed, so RSS looked small).
    inflight = deque(pool.apply_async(make_batch, ((next(seeds), args.batch, frac),)) for _ in range(2 * args.workers))

    def next_batch():
        batch = inflight.popleft().get()
        inflight.append(pool.apply_async(make_batch, ((next(seeds), args.batch, frac),)))
        return batch
    start, waited = time.perf_counter(), 0.0
    for step in range(1, args.steps + 1):
        w0 = time.perf_counter()
        x, objs, sliders, spinners, _ = next_batch()
        waited += time.perf_counter() - w0
        t = targets(objs, MAX_DT_MS, sliders, spinners, stride=STRIDE, grid=GRID)
        near = (t["dt"] < 0.35) & (t["mask"] > 0)
        t["dt_w"] = t["mask"] * (1 + 3 * near)  # the clicks happen on the last ~300 ms
        t = {k: mx.array(v) for k, v in t.items()}
        (loss, parts), grads = step_fn(model, mx.array(x.astype(np.float32) / 255.0), t)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if step % 100 == 0:
            el = time.perf_counter() - start
            focal, l_off, l_dt, l_ball, l_state = (p.item() for p in parts)
            print(f"step {args.start_step + step}: loss {loss.item():.3f} (heat {focal:.3f} off {l_off:.3f} "
                  f"dt {l_dt * 1000:.0f} ms ball {l_ball:.3f} state {l_state:.3f})  {step / el:.2f} steps/s, "
                  f"waiting on data {waited / el:.0%}, mlx {mx.get_active_memory() / 2**30:.1f}+"
                  f"{mx.get_cache_memory() / 2**30:.1f} GiB", flush=True)
        if step % args.save_every == 0:
            args.out.parent.mkdir(exist_ok=True)
            model.save_weights(str(args.out))
            args.out.with_suffix(".step").write_text(str(args.start_step + step))  # for resuming
        if step % args.eval_every == 0 or step == args.steps:
            evals = [("val", val_batches), ("TEST", test_batches)]
            if real_val:
                evals.append(("REAL", real_val_batches))
            for name, b in evals:
                m = evaluate(model, b)
                print(f"  {name} step {args.start_step + step}: found {m['found']:.1%}, aim {m['aim_px']:.2f}px, timing {m['timing_ms']:.1f}ms; "
                      f"<300ms: median {m['near_ms']:.1f} p90 {m['near_p90']:.1f} within25 {m['near_25']:.0%}; "
                      f"hold {m['hold']:.1%} spin {m['spin']:.1%} ball {m['ball_px']:.2f}px", flush=True)
            args.out.parent.mkdir(exist_ok=True)
            model.save_weights(str(args.out))
    pool.terminate()


if __name__ == "__main__":
    main()
