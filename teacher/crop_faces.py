"""Faces: crop the aligned squares listed in data/faces/landmarks.csv and save them as 512 x 512 images in
data/faces/photo/v_<source stem>_<k>.png. Skips faces that are too small, too cut by the image edge or too tilted.
Also writes data/faces/crops.csv (name, file, side).

    uv run --python 3.11 --with pillow --with numpy python teacher/crop_faces.py
"""
import csv
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
FACES = ROOT / "data" / "faces"
MIN_SIDE = 150      # in pixels of the original photo
T = 512


def crop(img: Image.Image, cx: float, cy: float, side: float, angle: float) -> Image.Image:
    """Square of size `side` centred on (cx, cy), rotated by `angle` degrees, resampled to T x T."""
    a = math.radians(angle)
    e = side / T
    c, s = math.cos(a) * e, math.sin(a) * e
    # for each output pixel (x, y): source point = centre + R(angle) * e * (x - T/2, y - T/2)
    m = (c, -s, cx - c * T / 2 + s * T / 2, s, c, cy - s * T / 2 - c * T / 2)
    return img.transform((T, T), Image.AFFINE, m, resample=Image.BICUBIC)


def main() -> None:
    (FACES / "photo").mkdir(parents=True, exist_ok=True)
    tally, seen = Counter(), Counter()
    with (FACES / "crops.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "file", "side"])
        current, img = None, None
        for r in csv.DictReader((FACES / "landmarks.csv").open()):
            cx, cy, side, angle = (float(r[k]) for k in ("cx", "cy", "side", "angle"))
            if side < MIN_SIDE:
                tally["too_small"] += 1
                continue
            if r["file"] != current:
                current, img = r["file"], ImageOps.exif_transpose(Image.open(ROOT / r["file"])).convert("RGB")
            width, height = img.size
            d = side / 2
            outside = max(0, d - cx, d - cy, cx + d - width, cy + d - height) / side
            if outside > 0.12 or abs(angle) > 50:
                tally["cut_by_edge" if outside > 0.12 else "too_tilted"] += 1
                continue
            base = Path(r["file"]).stem
            seen[base] += 1
            name = f"v_{base}_{seen[base]}"
            # pad with a blurred stretched copy rather than black when the square slightly overflows
            margin = int(side)
            large = Image.new("RGB", (width + 2 * margin, height + 2 * margin))
            large.paste(img.resize((width + 2 * margin, height + 2 * margin), Image.BILINEAR).filter(ImageFilter.GaussianBlur(20)))
            large.paste(img, (margin, margin))
            crop(large, cx + margin, cy + margin, side, angle).save(FACES / "photo" / f"{name}.png")
            w.writerow([name, r["file"], f"{side:.0f}"])
            tally["kept"] += 1
    print(dict(tally))


if __name__ == "__main__":
    main()
