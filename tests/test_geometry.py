"""Screen geometry against the values measured and derived in docs/research (cross-check.md §1a)."""

from __future__ import annotations

import pytest

from osu_ai import geometry as g

PINNED = g.OsuWindow.from_frame([244, 79, 1024, 796], client=(1024, 768))  # this Mac's measured window


def test_title_bar_is_cut_off_the_frame():
    assert PINNED.rect == (244, 107, 1024, 768)
    assert PINNED.scale == pytest.approx(1.6)


def test_measured_mapping_osu_to_global_points():
    # cross-check.md: osu!px -> global pt = (346.4 + 1.6x, 196.6 + 1.6y)
    for x, y in [(0, 0), (256, 192), (512, 384), (100.5, 37.25)]:
        assert PINNED.osu_to_points(x, y) == pytest.approx((346.4 + 1.6 * x, 196.6 + 1.6 * y))


def test_virtual_screen_is_the_whole_pinned_client_area():
    assert PINNED.capture_rect == pytest.approx((244, 107, 1024, 768))
    assert PINNED.virtual_to_points(0, 0) == pytest.approx((244, 107))
    assert PINNED.virtual_to_points(640, 480) == pytest.approx((1268, 875))
    assert PINNED.osu_to_points(-64, -56) == pytest.approx((244, 107))


def test_window_centre_is_osu_256_184():
    assert PINNED.points_to_osu(244 + 512, 107 + 384) == pytest.approx((256, 184))


@pytest.mark.parametrize("win", [PINNED, g.OsuWindow(0, 32, 1512, 950), g.OsuWindow(10.5, 3, 800, 800)])
def test_round_trips(win):
    for x, y in [(0, 0), (512, 384), (33.3, 291.7), (-40, 420)]:
        assert win.points_to_osu(*win.osu_to_points(x, y)) == pytest.approx((x, y))
        vx, vy = g.osu_to_virtual(x, y)
        assert win.points_to_virtual(*win.virtual_to_points(vx, vy)) == pytest.approx((vx, vy))


def test_non_4_3_client_letterboxes_the_virtual_screen():
    win = g.OsuWindow(0, 32, 1512, 950)  # Borderless on the notched panel (geometry.md §4)
    s = 950 / 480
    assert win.scale == pytest.approx(s)
    x, y, w, h = win.capture_rect
    assert (w, h) == pytest.approx((640 * s, 950))
    assert x == pytest.approx((1512 - 640 * s) / 2) and y == pytest.approx(32)
    # geometry.md §2: 3024x1900 px client -> osu (0,0) at (498.667, 221.667) px, i.e. half that in points
    tall = g.OsuWindow(0, 0, 1512, 950)
    assert tall.osu_to_points(0, 0) == pytest.approx((498.667 / 2, 221.667 / 2), abs=1e-3)


def test_model_input_pixels():
    for size in [(320, 240), (160, 120), (640, 480)]:
        k = 640 / size[0]
        assert g.input_to_osu(64 / k, 56 / k, size) == pytest.approx((0, 0))
        assert g.input_to_osu(*g.osu_to_input(123.4, 56.7, size), size) == pytest.approx((123.4, 56.7))
    assert g.osu_to_input(256, 192, (320, 240)) == pytest.approx((160, 124))  # playfield centre sits 8 osu!px low
    assert g.osu_to_input(512, 384, (160, 120)) == pytest.approx((144, 110))


def test_post_point_is_whole_points_from_the_client_origin():
    win = g.OsuWindow(244.5, 107.25, 1024, 768)
    px, py = win.post_point(100.2, 50.9)
    assert (px - win.left) == round(px - win.left) and (py - win.top) == round(py - win.top)
    exact = win.osu_to_points(100.2, 50.9)
    assert abs(px - exact[0]) <= 0.5 and abs(py - exact[1]) <= 0.5
    assert PINNED.post_point(216, 100) == (692, 357)


def test_frames_that_are_not_a_titled_client_are_refused():
    with pytest.raises(ValueError, match="WindowedSize"):
        g.OsuWindow.from_frame([0, 32, 1512, 950], client=(1024, 768))
    assert g.OsuWindow.from_frame([0, 32, 1512, 950]).rect == (0, 32, 1512, 950)
    assert g.OsuWindow.from_window({"frame": [244, 107, 1024, 768]}, client=(1024, 768)).rect == (244, 107, 1024, 768)


def test_client_point():
    assert PINNED.client_point(0, 0) == (244, 107)
    assert PINNED.client_point(1024, 768) == (1268, 875)
