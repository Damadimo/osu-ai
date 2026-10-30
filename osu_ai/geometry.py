"""Where osu!lazer draws the playfield on this Mac, in global screen points.

lazer centers a 640x480 "virtual screen" in its client area at s = min(W/640, H/480) points
per osu!px, with the 512x384 playfield at (64, 56) inside it: stable's layout plus lazer's
+8 osu!px storyboard shift. So for a W x H client area at (left, top):

    X = left + W/2 + (x - 256) * s        Y = top + H/2 + (y - 184) * s

For the pinned 1024x768 window, s = 1.6 and the virtual screen is the whole client area.
Our model's frames are that virtual screen scaled to the input size, so every conversion
here is linear. Pixel coordinates are continuous (pixel (i, j) covers [i, i+1) x [j, j+1)),
and global points are CGEvent coordinates (main display's top-left, y down).
"""

from __future__ import annotations

from dataclasses import dataclass

VIRTUAL = (640, 480)  # the virtual screen, in osu!px
MARGIN = (64.0, 56.0)  # the playfield's origin inside it
TITLE_BAR_MAX = 64.0  # a frame taller than the client area by more than this is not just a title bar


def osu_to_virtual(x: float, y: float) -> tuple[float, float]:
    return x + MARGIN[0], y + MARGIN[1]


def virtual_to_osu(vx: float, vy: float) -> tuple[float, float]:
    return vx - MARGIN[0], vy - MARGIN[1]


def input_to_osu(u: float, v: float, size: tuple[int, int]) -> tuple[float, float]:
    """Model-input pixel (u, v) at input size (w, h) -> osu!px."""
    return virtual_to_osu(u * VIRTUAL[0] / size[0], v * VIRTUAL[1] / size[1])


def osu_to_input(x: float, y: float, size: tuple[int, int]) -> tuple[float, float]:
    vx, vy = osu_to_virtual(x, y)
    return vx * size[0] / VIRTUAL[0], vy * size[1] / VIRTUAL[1]


@dataclass(frozen=True)
class OsuWindow:
    """osu!'s client area (the drawable part of its window) in global points."""

    left: float
    top: float
    width: float
    height: float

    @classmethod
    def from_frame(cls, frame, client: tuple[float, float] | None = None) -> OsuWindow:
        """frame: a window frame [x, y, w, h] as OsuHands reports it. A titled macOS window's frame
        includes the title bar, so with the known client size (framework.ini WindowedSize) the client
        area is the bottom `client` part. Without it, the whole frame is taken as the client area."""
        x, y, w, h = map(float, frame)
        if client is None:
            return cls(x, y, w, h)
        cw, ch = map(float, client)
        if abs(w - cw) > 1 or not -1 <= h - ch <= TITLE_BAR_MAX:
            raise ValueError(f"window frame {[x, y, w, h]} does not hold a {cw:g}x{ch:g} client area "
                             f"under a title bar: is osu! Windowed at its framework.ini WindowedSize?")
        return cls(x + (w - cw) / 2, y + h - ch, cw, ch)

    @classmethod
    def from_window(cls, win: dict, client: tuple[float, float] | None = None) -> OsuWindow:
        """win: an entry of Hands.windows() / Hands.osu_window()."""
        return cls.from_frame(win["frame"], client)

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.width, self.height

    @property
    def scale(self) -> float:
        """Points per osu!px."""
        return min(self.width / VIRTUAL[0], self.height / VIRTUAL[1])

    def virtual_to_points(self, vx: float, vy: float) -> tuple[float, float]:
        s = self.scale
        return (self.left + self.width / 2 + (vx - VIRTUAL[0] / 2) * s,
                self.top + self.height / 2 + (vy - VIRTUAL[1] / 2) * s)

    def points_to_virtual(self, px: float, py: float) -> tuple[float, float]:
        s = self.scale
        return ((px - self.left - self.width / 2) / s + VIRTUAL[0] / 2,
                (py - self.top - self.height / 2) / s + VIRTUAL[1] / 2)

    def osu_to_points(self, x: float, y: float) -> tuple[float, float]:
        return self.virtual_to_points(*osu_to_virtual(x, y))

    def points_to_osu(self, px: float, py: float) -> tuple[float, float]:
        return virtual_to_osu(*self.points_to_virtual(px, py))

    def post_point(self, x: float, y: float) -> tuple[float, float]:
        """osu!px -> the point to post a cursor move to. SDL2 truncates the cursor to whole points
        from the client's top-left, so post whole points there and the game sees exactly them."""
        px, py = self.osu_to_points(x, y)
        return self.left + round(px - self.left), self.top + round(py - self.top)

    def client_point(self, cx: float, cy: float) -> tuple[float, float]:
        """A point given relative to the client area's top-left -> global points."""
        return self.left + cx, self.top + cy

    @property
    def capture_rect(self) -> tuple[float, float, float, float]:
        """The virtual screen in global points: capture this at 640x480 for 1 px per osu!px."""
        x0, y0 = self.virtual_to_points(0, 0)
        return x0, y0, VIRTUAL[0] * self.scale, VIRTUAL[1] * self.scale
