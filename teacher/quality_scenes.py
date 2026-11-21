"""Scenes: automatic quality control of the photo -> styled pairs.

For every pair, measures how faithful the styled image is to the photo:
  edges            : similarity of the edge maps (1 = identical). Low = the structure moved.
  layout           : similarity of the large light/dark masses (image reduced to 32 x 32).
  worst_tile       : worst edge similarity among the 16 tiles of the image (local defect).
  added            : largest excess of edges in one tile (invented element: object, person, game UI).
  brightness       : mean brightness difference (styled - photo), from -1 to 1.
  colour           : mean chroma difference on the image reduced to 16 x 16.
  banner           : excess of fine edges in a strip of the top third (invented text or game UI).
  photo_brightness : mean brightness of the photo (0 = black, 1 = white).
  diff             : mean pixel difference (very low = the style was not applied).

Output: data/scenes/quality.csv (one row per pair). Thresholds are applied by split_scenes.py.

    uv run --python 3.12 --with pillow --with numpy python teacher/quality_scenes.py
"""
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SCENES = Path(__file__).resolve().parent.parent / "data" / "scenes"
T = 128
TILES = 4


def gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize((T, T), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1)), dtype=np.float32) / 255


def edges(g: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(g)
    e = np.hypot(gx, gy)
    k = np.exp(-np.arange(-4, 5) ** 2 / 8.0)  # Gaussian blur (sigma 2) to tolerate a slight shift
    k /= k.sum()
    e = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, e)
    e = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, e)
    return e.astype(np.float32)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def fine_edges(img: Image.Image) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy) > 0.12


def banner(p: Image.Image, s: Image.Image) -> float:
    a, b = fine_edges(p), fine_edges(s)
    worst = 0.0
    for y in range(0, 80, 8):
        for x in range(0, 256 - 96 + 1, 16):
            worst = max(worst, float(b[y:y + 24, x:x + 96].mean() - a[y:y + 24, x:x + 96].mean()))
    return worst


def measure(name: str) -> dict:
    p = Image.open(SCENES / "photo" / f"{name}.png").convert("RGB")
    s = Image.open(SCENES / "styled" / f"{name}.png").convert("RGB")
    gp, gs = gray(p), gray(s)
    ep, es = edges(gp), edges(gs)
    c = T // TILES
    mean_ep = float(ep.mean()) + 1e-6
    worst, added = 1.0, 0.0
    for i in range(TILES):
        for j in range(TILES):
            a, b = ep[i * c:(i + 1) * c, j * c:(j + 1) * c], es[i * c:(i + 1) * c, j * c:(j + 1) * c]
            added = max(added, float(b.mean() - a.mean()) / mean_ep)
            if max(a.mean(), b.mean()) > 0.35 * mean_ep:  # tile without edges: nothing to compare
                worst = min(worst, corr(a, b))
    small = lambda im, n: np.asarray(im.resize((n, n), Image.BOX).convert("YCbCr"), dtype=np.float32) / 255
    pp, ss = small(p, 16), small(s, 16)
    return {
        "name": name,
        "edges": round(corr(ep, es), 4),
        "layout": round(corr(small(p, 32)[..., 0], small(s, 32)[..., 0]), 4),
        "worst_tile": round(worst, 4),
        "added": round(added, 4),
        "brightness": round(float(gs.mean() - gp.mean()), 4),
        "colour": round(float(np.hypot(pp[..., 1] - ss[..., 1], pp[..., 2] - ss[..., 2]).mean()), 4),
        "banner": round(banner(p, s), 4),
        "photo_brightness": round(float(gp.mean()), 4),
        "diff": round(float(np.abs(np.asarray(p.resize((T, T)), dtype=np.float32) - np.asarray(s.resize((T, T)), dtype=np.float32)).mean() / 255), 4),
    }


def main() -> None:
    names = sorted(f.stem for f in (SCENES / "styled").glob("*.png") if (SCENES / "photo" / f.name).exists())
    with Pool() as pool:
        rows = pool.map(measure, names, chunksize=32)
    with (SCENES / "quality.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} pairs measured -> {SCENES / 'quality.csv'}")
    for k in list(rows[0])[1:]:
        v = np.array([r[k] for r in rows])
        print(f"{k:16s} " + "  ".join(f"p{q:02d}={np.percentile(v, q):.3f}" for q in (1, 5, 25, 50, 75, 95, 99)))


if __name__ == "__main__":
    main()
