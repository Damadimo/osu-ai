"""Generate the 'osu-jev minimal' osu! skin: plain gray discs and white rings on black.

  uv run --with pillow scripts/make_skin.py data/skin/osu-jev-minimal

Every texture is @2x (2 px per unscaled osu!px). Circles are a gray disc with a white
approach ring; slider heads are a gray ring with a black core; the ball is a white disc.
Numbers, follow points, judgements, hit lighting, kiai flashes and the cursor are blank,
so the model sees only what it needs and our renderer (lazer_frames.py) can match it.
lazer deletes the .osk when it imports it, so import a copy.""" 
import os
import sys
import zipfile

import numpy as np
from PIL import Image

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
SS = 4  # supersampling for anti-aliased edges


def save(name, rgba):
    Image.fromarray(rgba, "RGBA").save(os.path.join(OUT, name))


def blank(name, size=4):
    save(name, np.zeros((size, size, 4), np.uint8))


def radial(name, size, layers):
    """layers: list of (r_in_px, r_out_px, gray, alpha) drawn in order (later on top), radii in output px."""
    n = size * SS
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / SS
    out = np.zeros((n, n, 4), np.float32)
    out[..., :3] = layers[-1][2]  # transparent pixels carry the outer layer's gray (no dark AA fringe)
    for r_in, r_out, gray, alpha in layers:
        m = (r >= r_in) & (r < r_out)
        out[m] = (gray, gray, gray, alpha)
    img = Image.fromarray(out.astype(np.uint8), "RGBA").resize((size, size), Image.Resampling.BOX)
    save(name, np.asarray(img))


def arrow(name, size=128, gray=255):
    """Filled right-pointing triangle (lazer rotates it to face back along the slider)."""
    n = size * SS
    yy, xx = np.mgrid[0:n, 0:n] / SS
    # triangle with tip at (0.84 s, 0.5 s), base at x = 0.22 s, half-height 0.34 s
    s = size
    x0, x1, h = 0.22 * s, 0.84 * s, 0.34 * s
    inside = (xx >= x0) & (xx <= x1) & (np.abs(yy - 0.5 * s) <= h * (x1 - xx) / (x1 - x0))
    out = np.zeros((n, n, 4), np.uint8)
    out[..., :3] = gray
    out[inside, 3] = 255
    img = Image.fromarray(out, "RGBA").resize((size, size), Image.Resampling.BOX)
    save(name, np.asarray(img))


# --- visible elements (sizes in @2x px; osu!px = px / 2) ---
radial("hitcircleoverlay@2x.png", 256, [(0, 108, 160, 255)])                       # disc r=54, gray 160
radial("sliderstartcircleoverlay@2x.png", 256, [(0, 56, 0, 255), (56, 108, 160, 255)])  # ring 28..54 gray 160, black core
radial("approachcircle@2x.png", 256, [(108, 126, 255, 255)])                       # ring 54..63, white
radial("sliderb@2x.png", 256, [(0, 92, 255, 255)])                                 # disc r=46, white
arrow("reversearrow@2x.png", 128, 255)                                             # 64x64 osu!px triangle
radial("spinner-top@2x.png", 1024, [(484, 508, 128, 255)])                         # ring, gray 128
radial("spinner-approachcircle@2x.png", 512, [(232, 252, 255, 255)])               # ring, white

# --- must exist but be invisible ---
for n in ["hitcircle", "sliderstartcircle", "sliderendcircle", "sliderendcircleoverlay",
          "sliderfollowcircle", "sliderscorepoint", "followpoint", "lighting", "star2",
          "cursor", "cursormiddle", "cursortrail", "cursor-smoke",
          "hit0", "hit50", "hit100", "hit300", "slidertickmiss", "sliderendmiss",
          "spinner-bottom", "spinner-middle", "spinner-middle2", "spinner-glow",
          "spinner-spin", "spinner-clear", "spinner-rpm"]:
    blank(f"{n}@2x.png")
for i in range(10):
    blank(f"default-{i}@2x.png")
    blank(f"score-{i}@2x.png")
for n in ["comma", "dot", "percent", "x"]:
    blank(f"score-{n}@2x.png")

INI = """[General]
Name: osu-jev minimal
Author: osu-jev
Version: 2.7
AllowSliderBallTint: 0
CursorRotate: 0
CursorExpand: 0
CursorCentre: 1
CursorTrailRotate: 0
HitCircleOverlayAboveNumber: 1
SpinnerFrequencyModulate: 0

[Colours]
Combo1: 255,255,255
SliderBorder: 128,128,128
SliderTrackOverride: 0,0,0
SliderBall: 255,255,255
StarBreakAdditive: 0,0,0

[Fonts]
HitCirclePrefix: default
HitCircleOverlap: 0
ScorePrefix: score
ScoreOverlap: 0
ComboPrefix: score
ComboOverlap: 0
"""
with open(os.path.join(OUT, "skin.ini"), "w", newline="\r\n") as f:
    f.write(INI)

osk = OUT.rstrip("/") + ".osk"
with zipfile.ZipFile(osk, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(OUT)):
        z.write(os.path.join(OUT, fn), fn)
print(osk, len(os.listdir(OUT)), "files")
