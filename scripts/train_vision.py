"""Train the osu! vision model on rendered clips and report timing/aim on held-out maps.

  uv run --extra train scripts/train_vision.py --steps 6000
"""

from __future__ import annotations

import argparse
import glob
import math
import random
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from osu_ai.framedata import HISTORY
from osu_ai.vision import OsuNet, decode, decode_state, loss_fn, targets

MAX_DT_MS = 1800  # every object that can be on screen


class Clips:
    """Holds a set of rendered maps in memory and samples 4-frame stacks."""

    def __init__(self, files: list[str]):
        self.maps = []
        for f in files:
            d = np.load(f)
            clip = d["clip"]
            # A stack is valid when its 4 frames come from the same clip.
            ok = np.nonzero(np.r_[np.zeros(HISTORY - 1, bool), clip[HISTORY - 1:] == clip[:-HISTORY + 1]])[0]
            self.maps.append((d["frames"], d["obj_offsets"], d["objs"], ok, float(d["radius_px"]), d["slider"], d["spinner"]))

    def sample(self, rng: np.random.Generator, n: int):
        xs, objs, radii, sliders, spinners = [], [], [], [], []
        for _ in range(n):
            frames, offs, all_objs, ok, radius, slider, spinner = self.maps[rng.integers(len(self.maps))]
            i = ok[rng.integers(len(ok))]
            xs.append(frames[i - HISTORY + 1:i + 1])
            objs.append(all_objs[offs[i]:offs[i + 1]])
            radii.append(radius)
            sliders.append(slider[i])
            spinners.append(spinner[i])
        x = np.stack(xs).transpose(0, 2, 3, 1).astype(np.float32) / 255.0  # (B, H, W, 4)
        return x, objs, radii, sliders, spinners


def evaluate(model: OsuNet, val: Clips, n: int = 1500) -> dict:
    model.eval()
    rng = np.random.default_rng(123)
    found, timing_err, aim_err, near_err = 0, [], [], []
    hold_right, spin_right, ball_err = [], [], []
    total = 0
    for _ in range(n // 100):
        x, objs, radii, sliders, spinners = val.sample(rng, 100)
        heat, off, dt, ball, state = model(mx.array(x))
        dets = decode(heat, off, dt)
        for (ph, ps, bx, by), sl, sp in zip(decode_state(ball, state), sliders, spinners):
            hold_right.append((ph > 0.5) == bool(sl[0]))
            spin_right.append((ps > 0.5) == bool(sp))
            if sl[0]:
                ball_err.append(math.dist((bx, by), (sl[1], sl[2])))
        for d, o, r in zip(dets, objs, radii):
            o = o[o[:, 2] <= MAX_DT_MS]
            if len(o) == 0:
                continue
            total += 1
            nx, ny, ndt = o[np.argmin(o[:, 2])]  # the true next object
            if not d:
                continue
            px, py, pdt, _ = d[0]  # the model's soonest detection
            dist = math.dist((px, py), (nx, ny))
            aim_err.append(dist)
            if dist <= r:
                found += 1
                timing_err.append(abs(pdt - ndt))
                if ndt < 300:
                    near_err.append(abs(pdt - ndt))
    model.train()
    q = lambda v, f: statistics.quantiles(v, n=20)[int(f * 20) - 1] if len(v) > 20 else float("nan")
    return {
        "next_found": found / max(total, 1),
        "aim_px_median": statistics.median(aim_err) if aim_err else float("nan"),
        "timing_ms_median": statistics.median(timing_err) if timing_err else float("nan"),
        "near_ms_median": statistics.median(near_err) if near_err else float("nan"),
        "near_ms_p90": q(near_err, 0.9),
        "near_within_25": sum(e <= 25 for e in near_err) / max(len(near_err), 1),
        "hold_acc": statistics.mean(hold_right), "spin_acc": statistics.mean(spin_right),
        "ball_px_median": statistics.median(ball_err) if ball_err else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--train-maps", type=int, default=120, help="maps held in memory at once")
    parser.add_argument("--out", type=Path, default=Path("models/osunet.safetensors"))
    parser.add_argument("--init", help="start from these weights (new heads start fresh)")
    args = parser.parse_args()

    train_files = sorted(glob.glob("data/frames/train/*.npz"))
    val = Clips(sorted(glob.glob("data/frames/val/*.npz")))
    rng = np.random.default_rng(0)
    model = OsuNet(args.width)
    if args.init:
        model.load_weights(args.init, strict=False)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"OsuNet width {args.width}: {n_params / 1e6:.2f}M parameters")
    schedule = optim.cosine_decay(args.lr, args.steps, 1e-5)
    opt = optim.AdamW(learning_rate=schedule, weight_decay=1e-4)
    step_fn = nn.value_and_grad(model, loss_fn)

    def train_step(x, t):
        (loss, parts), grads = step_fn(model, x, t)
        opt.update(model, grads)
        return loss, parts

    train = None
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        if train is None or step % 1500 == 1:  # rotate which maps are in memory
            random.Random(step).shuffle(train_files)
            train = Clips(train_files[:args.train_maps])
        x, objs, _, sliders, spinners = train.sample(rng, args.batch)
        t = {k: mx.array(v) for k, v in targets(objs, MAX_DT_MS, sliders, spinners).items()}
        loss, (focal, l_off, l_dt, l_ball, l_state) = train_step(mx.array(x), t)
        mx.eval(model.parameters(), opt.state, loss)
        if step % 100 == 0:
            rate = step / (time.perf_counter() - start)
            print(f"step {step}: loss {loss.item():.3f} (heat {focal.item():.3f} off {l_off.item():.3f} "
                  f"dt {l_dt.item() * 1000:.0f} ms ball {l_ball.item():.3f} state {l_state.item():.3f})  {rate:.1f} steps/s",
                  flush=True)
        if step % 1000 == 0 or step == args.steps:
            m = evaluate(model, val)
            print(f"  VAL step {step}: next circle found {m['next_found']:.0%}, aim error {m['aim_px_median']:.1f} px, "
                  f"timing error {m['timing_ms_median']:.0f} ms; due within 300 ms: median {m['near_ms_median']:.0f} ms, "
                  f"p90 {m['near_ms_p90']:.0f} ms, within ±25 ms {m['near_within_25']:.0%}; hold right {m['hold_acc']:.0%}, "
                  f"spin right {m['spin_acc']:.0%}, ball error {m['ball_px_median']:.1f} px", flush=True)
            args.out.parent.mkdir(exist_ok=True)
            model.save_weights(str(args.out))


if __name__ == "__main__":
    main()
