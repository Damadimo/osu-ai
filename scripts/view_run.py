"""Turn a recorded real-game run into a video of what the model saw and did.

  uv run --with "imageio[ffmpeg]" scripts/view_run.py runs/real/play/<run>.npz [--out runs/view.mp4]

Each frame is the 320x240 view the model got, scaled up, with the next circle (yellow) and the
one after it (cyan) labelled with the model's ms-until-hit, the cursor it commanded (red), and
a flash where a key press went out. Pass --start/--seconds to cut a segment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from osu_ai.lazer_frames import MARGIN, VIRTUAL

SCALE = 2
KEY_DOWN = 2


def to_px(x: float, y: float, w: int) -> tuple[int, int]:
    f = w / VIRTUAL[0] * SCALE
    return int((x + MARGIN[0]) * f), int((y + MARGIN[1]) * f)


def main() -> None:
    import imageio.v2 as imageio
    import pygame

    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the run")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()

    d = np.load(args.run)
    meta = json.loads(str(d["meta"]))
    frames, shown = d["frames"], d["display_ns"]
    shown = np.where(shown > 0, shown, d["pts_ns"])
    t_s = (shown - shown[0]) / 1e9
    det_off, dets, cursor = d["det_offsets"], d["dets"], d["cursor"]
    presses = d["ev_at_ns"][(d["ev_kind"] == KEY_DOWN) & (d["ev_status"] <= 1)]
    h, w = frames.shape[1:]
    lo = args.start
    hi = lo + args.seconds if args.seconds else t_s[-1] + 1
    pygame.font.init()
    font = pygame.font.SysFont("helveticaneue", 18, bold=True)
    out = args.out or args.run.with_suffix(".mp4")
    writer = imageio.get_writer(out, fps=60, macro_block_size=8)
    for i in np.nonzero((t_s >= lo) & (t_s < hi))[0]:
        gray = np.repeat(np.repeat(frames[i], SCALE, axis=0), SCALE, axis=1)
        surf = pygame.surfarray.make_surface(np.stack([gray.T] * 3, axis=-1))
        for k, (x, y, ms, score) in enumerate(dets[det_off[i]:det_off[i + 1]][:2]):
            color = (255, 220, 60) if k == 0 else (80, 220, 255)
            p = to_px(x, y, w)
            pygame.draw.circle(surf, color, p, 6, 2)
            surf.blit(font.render(f"{ms:.0f}", True, color), (p[0] + 9, p[1] - 22))
        if not np.isnan(cursor[i]).any():
            pygame.draw.circle(surf, (255, 70, 70), to_px(*cursor[i], w), 5)
        if np.any(np.abs(presses - shown[i]) < 25e6):
            pygame.draw.rect(surf, (255, 70, 70), surf.get_rect(), 6)
        surf.blit(font.render(f"{t_s[i]:6.2f}s  {meta.get('title', '')}", True, (200, 200, 200)), (8, 6))
        writer.append_data(np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)))
    writer.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
