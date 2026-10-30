"""Put real captured frames next to our renderer's frames at the same game time.

  uv run scripts/compare_real.py data/real/<name>.npz --map data/lazer_osu/<id>.osu [--out runs/compare.png]

Aligns the recording to game time (realdata.align), then writes a PNG with rows of
(real | rendered | difference x4) at a few moments, and prints how well they match
overall and how the gray levels compare (gain/gamma between the real pipeline and ours).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from osu_ai import beatmap
from osu_ai.lazer_frames import Look, perfect, render
from osu_ai.realdata import Recording, align


def save_png(gray: np.ndarray, path: Path) -> None:
    import pygame

    img = np.clip(gray, 0, 255).astype(np.uint8)
    pygame.image.save(pygame.surfarray.make_surface(np.stack([img.T] * 3, axis=-1)), str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--map", required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/compare.png"))
    parser.add_argument("--hit-animations", action="store_true")
    parser.add_argument("--snaking", action="store_true")
    parser.add_argument("--offset", type=float, help="skip alignment and use this clock offset (ms)")
    parser.add_argument("--rows", type=int, default=6)
    args = parser.parse_args()

    rec = Recording.load(args.recording)
    bm = beatmap.parse(args.map)
    look = Look(args.hit_animations, args.snaking)
    outcome = perfect(bm)  # autoplay footage; for replays, pass the replay's outcome instead
    if args.offset is None:
        c, info = align(rec, bm, outcome, look)
        print(f"offset {c:.2f} ms, match {info['match']:.4f}")
    else:
        c = args.offset
    busy = np.nonzero(rec.frames.reshape(len(rec.frames), -1).std(axis=1) > 3)[0]
    picks = busy[np.linspace(0, len(busy) - 1, args.rows + 2).astype(int)[1:-1]]
    rows, real_all, fake_all = [], [], []
    for i in picks:
        real = rec.frames[i].astype(np.float32)
        fake = render(bm, rec.pts_ms[i] + c, outcome, look, rec.view)
        real_all.append(real.ravel())
        fake_all.append(fake.ravel())
        rows.append(np.concatenate([real, fake, np.clip(128 + 4 * (real - fake), 0, 255)], axis=1))
    real_all, fake_all = np.concatenate(real_all), np.concatenate(fake_all)
    lit = fake_all > 8
    gain = float(np.median(real_all[lit] / np.maximum(fake_all[lit], 1)))
    print(f"gray levels: real/rendered median ratio {gain:.3f} on lit pixels; "
          f"mean abs diff {np.abs(real_all - fake_all).mean():.2f}")
    for level in (44, 128, 160, 255):
        near = np.abs(fake_all - level) < 6
        if near.sum() > 50:
            print(f"  rendered ~{level}: real median {np.median(real_all[near]):.1f} ({near.sum()} px)")
    args.out.parent.mkdir(exist_ok=True)
    save_png(np.concatenate(rows, axis=0), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
