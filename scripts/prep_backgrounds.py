"""Extract map backgrounds for training: data/osz/*.osz -> data/backgrounds/{train,test}/*.npy.

  uv run --with pillow scripts/prep_backgrounds.py
  uv run --with pillow scripts/prep_backgrounds.py --covers   # also the corpus maps' set covers

--covers downloads assets.ppy.sh/beatmaps/<set>/covers/fullsize.jpg (the set's background, a
public image; no login) for every set in data/corpus/*.osu, one request per second, cached in
data/covers/. Test sets are never fetched. Images in data/extra_backgrounds/ (such as lazer's
own default backgrounds) are added to train.

Every distinct background named in a map's [Events] is decoded to RGB (transparent pixels over
black, as lazer draws them) and scaled to cover 640x480, the 4:3 virtual screen at twice the
model's input, keeping its aspect ratio so training can still crop, zoom and shift it. Sets in
train_lazer.TEST_SETS go to test/, the rest to train/. index.json maps (set id, file) to the .npy.
"""

from __future__ import annotations

import glob
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_lazer import TEST_SETS  # noqa: E402

from osu_ai.backgrounds import background_name, set_id  # noqa: E402

OUT = Path("data/backgrounds")
COVER = (640, 480)


def to_srgb(img: Image.Image) -> Image.Image:
    """Convert through the image's ICC profile when it has one (lazer's decoder does)."""
    icc = img.info.get("icc_profile")
    if not icc:
        return img
    from PIL import ImageCms

    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        mode = "RGBA" if "A" in img.mode else "RGB"
        return ImageCms.profileToProfile(img, src, ImageCms.createProfile("sRGB"), outputMode=mode,
                                         renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
    except Exception:
        return img


def cover(img: Image.Image, size=COVER) -> np.ndarray:
    """RGB uint8 scaled to cover `size` (aspect kept). Transparent parts go to black as lazer draws
    them: colour times alpha squared (it blends the image into a cleared framebuffer, then blends
    that framebuffer onto the screen)."""
    if img.mode == "CMYK" and not img.info.get("icc_profile"):
        raise ValueError("CMYK without a colour profile: no reliable conversion")
    img = to_srgb(img)
    if img.mode in ("RGBA", "LA", "P", "PA"):
        rgba = np.asarray(img.convert("RGBA"), np.float32)
        img = Image.fromarray((rgba[..., :3] * (rgba[..., 3:] / 255) ** 2 + 0.5).astype(np.uint8))
    img = img.convert("RGB")
    scale = max(size[0] / img.width, size[1] / img.height)
    w, h = max(size[0], round(img.width * scale)), max(size[1], round(img.height * scale))
    return np.asarray(img.resize((w, h), Image.LANCZOS), np.uint8)


def fetch_covers(skip: set[int]) -> list[Path]:
    """Set covers for the loose corpus maps, cached in data/covers/<set>.jpg."""
    import time
    import urllib.error
    import urllib.request

    cache = Path("data/covers")
    cache.mkdir(parents=True, exist_ok=True)
    sets = set()
    for p in glob.glob("data/corpus/*.osu"):
        sid = set_id(Path(p).read_text(encoding="utf-8", errors="replace"))
        if sid and sid > 0 and sid not in skip and sid not in TEST_SETS:
            sets.add(sid)
    out = []
    for sid in sorted(sets):
        path = cache / f"{sid}.jpg"
        if not path.exists() and not path.with_suffix(".missing").exists():
            url = f"https://assets.ppy.sh/beatmaps/{sid}/covers/fullsize.jpg"
            req = urllib.request.Request(url, headers={"User-Agent": "osu-ai/0.1 (local research)"})
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    path.write_bytes(r.read())
            except urllib.error.HTTPError as e:
                if e.code in (403, 429, 503):
                    print(f"stopping: HTTP {e.code} from {url}")
                    break
                path.with_suffix(".missing").touch()
            except OSError as e:
                print(f"{url}: {e}")
            time.sleep(1.0)
        if path.exists():
            out.append(path)
    return out


def main() -> None:
    index, seen = [], set()
    for split in ("train", "test"):
        (OUT / split).mkdir(parents=True, exist_ok=True)
    for path in sorted(glob.glob("data/osz/*.osz")):
        try:
            z = zipfile.ZipFile(path)
        except zipfile.BadZipFile:
            continue
        files = {n.lower(): n for n in z.namelist()}
        for name in z.namelist():
            if not name.lower().endswith(".osu"):
                continue
            text = z.read(name).decode("utf-8-sig", errors="replace")
            bg, sid = background_name(text), set_id(text) or int(Path(path).stem)
            if bg is None or (sid, bg.lower()) in seen:
                continue
            seen.add((sid, bg.lower()))
            member = files.get(bg.replace("\\", "/").lower())
            if member is None:
                print(f"{path}: {bg!r} is not in the set")
                continue
            try:
                arr = cover(Image.open(io.BytesIO(z.read(member))))
            except Exception as e:  # truncated or odd images: skip, there are plenty
                print(f"{path}: {bg!r}: {e}")
                continue
            split = "test" if sid in TEST_SETS else "train"
            out = OUT / split / f"{sid}_{len([i for i in index if i['set'] == sid])}.npy"
            np.save(out, arr)
            index.append({"set": sid, "file": bg, "split": split, "npy": str(out), "shape": list(arr.shape)})
    for path in sorted(glob.glob("data/extra_backgrounds/*")):  # e.g. lazer's bg1 (maps without a background)
        try:
            arr = cover(Image.open(path))
        except Exception as e:
            print(f"{path}: {e}")
            continue
        out = OUT / "train" / f"extra_{Path(path).stem}.npy"
        np.save(out, arr)
        index.append({"set": 0, "file": Path(path).name, "split": "train", "npy": str(out), "shape": list(arr.shape)})
    if "--covers" in sys.argv:
        for path in fetch_covers({i["set"] for i in index}):
            try:
                arr = cover(Image.open(path))
            except Exception as e:
                print(f"{path}: {e}")
                continue
            out = OUT / "train" / f"{path.stem}_cover.npy"
            np.save(out, arr)
            index.append({"set": int(path.stem), "file": "covers/fullsize.jpg", "split": "train", "npy": str(out),
                          "shape": list(arr.shape)})
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    n = {s: sum(i["split"] == s for i in index) for s in ("train", "test")}
    print(f"{len(index)} backgrounds: {n['train']} train, {n['test']} test -> {OUT}")


if __name__ == "__main__":
    main()
