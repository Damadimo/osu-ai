from pathlib import Path

import numpy as np
import pytest

from osu_ai import beatmap
from osu_ai.lazer_frames import View, backdrop, luma, perfect, render

MAP = Path(__file__).parent.parent / "data/lazer_osu/827932.osu"


def marker_image(h: int, w: int, at) -> np.ndarray:
    """A black image covering the virtual screen with one bright 2x2 block at virtual point `at`."""
    img = np.zeros((h, w), np.float32)
    s = min(w / 640, h / 480)
    x, y = int((at[0] - 320) * s + w / 2), int((at[1] - 240) * s + h / 2)
    img[y - 1:y + 1, x - 1:x + 1] = 255
    return img


def peak(img: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(img > img.max() * 0.5)
    return xs.mean() + 0.5, ys.mean() + 0.5


@pytest.mark.parametrize("shape", [(960, 1280), (480, 853), (600, 640)])
@pytest.mark.parametrize("view", [View((320, 240)), View((320, 240), zoom=1.012, shift=(1.3, -0.8)), View((640, 480))])
def test_background_lines_up_with_the_playfield(shape, view):
    for p in [(0, 0), (512, 384), (256, 192)]:  # osu!px (0, 0) is virtual (64, 56)
        bg = backdrop(marker_image(*shape, at=(p[0] + 64, p[1] + 56)), view, dim=0.0, zoom=1.0)
        assert peak(bg) == pytest.approx(view.to_px(p), abs=0.75)


def test_lazer_zoom_scales_about_the_centre():
    view = View((640, 480))
    bg = backdrop(marker_image(480, 640, at=(320 + 100, 240 - 50)), view, dim=0.0)  # default zoom: lazer's
    assert peak(bg) == pytest.approx((320 + 100 * (1 + 100 / 1024), 240 - 50 * (1 + 100 / 1024)), abs=0.75)


def test_dim_darkens_linearly():
    gray = np.full((480, 640), 200, np.float32)
    assert backdrop(gray, View(), dim=0.75) == pytest.approx(np.full((240, 320), 50.0), abs=1e-3)


def test_luma_uses_709_weights():
    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[..., 1] = 255
    assert luma(rgb)[0, 0] == pytest.approx(0.7152 * 255, abs=1e-3)


@pytest.mark.skipif(not MAP.exists(), reason="needs data/lazer_osu/827932.osu")
def test_objects_composite_over_the_background():
    bm = beatmap.parse(MAP)
    o = bm.objects[0]
    t, view = o.time - 100, View((320, 240))
    black = render(bm, t, perfect(bm), view=view)
    assert np.array_equal(render(bm, t, perfect(bm), view=view, background=np.zeros((240, 320), np.float32)), black)
    over = render(bm, t, perfect(bm), view=view, background=np.full((240, 320), 60.0, np.float32))
    over2 = render(bm, t, perfect(bm), view=view, background=np.full((240, 320), 120.0, np.float32))
    trans = (over2 - over) / 60.0  # how much background shows through each pixel
    assert trans.min() >= -1e-4 and trans.max() <= 1 + 1e-4
    assert np.allclose(over, black + trans * 60.0, atol=1e-3)
    assert (trans > 0.999).mean() > 0.5  # most of the screen is empty
    cx, cy = (int(v) for v in view.to_px(o.pos))
    assert trans[cy, cx] < 0.05  # a nearly faded-in disc hides it
    assert ((black < 1) & (trans < 0.5)).any()  # black-drawn parts (slider head core) also hide it


@pytest.mark.skipif(not MAP.exists(), reason="needs data/lazer_osu/827932.osu")
def test_a_faded_in_slider_head_hides_the_background_everywhere():
    from osu_ai.beatmap import Kind

    bm = beatmap.parse(MAP)
    o = next(o for o in bm.objects if o.kind is Kind.SLIDER)
    view = View((320, 240))
    t = o.time - 100  # fully faded in, not yet hit
    lo = render(bm, t, perfect(bm), view=view, background=np.full((240, 320), 60.0, np.float32))
    hi = render(bm, t, perfect(bm), view=view, background=np.full((240, 320), 120.0, np.float32))
    trans = (hi - lo) / 60.0
    cx, cy = view.to_px(o.pos)
    r = 54 / 64 * bm.difficulty.radius * view.f
    ys, xs = np.mgrid[0:240, 0:320] + 0.5
    inside = np.hypot(xs - cx, ys - cy) < r - 1
    assert trans[inside].max() < 1e-3


def test_background_keeps_its_fine_detail():
    """Area sampling (as the real capture averages) keeps most contrast at 0.3 cycles per output px."""
    view = View((320, 240))
    s = 480 / (240 * (1 + 100 / 1024))  # background px per output px at lazer's zoom
    x = np.arange(854, dtype=np.float32)
    img = np.tile(128 + 100 * np.sin(2 * np.pi * 0.3 / s * x), (480, 1)).astype(np.float32)
    row = backdrop(img, view, dim=0.0)[120, 40:280]
    amplitude = (np.percentile(row, 99) - np.percentile(row, 1)) / 2
    assert amplitude > 75  # box-then-bilinear gave 60; ideal area sampling 84
