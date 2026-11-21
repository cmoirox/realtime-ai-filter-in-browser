"""Scenes: build the photo -> styled pairs with the FLUX.2 klein 4B teacher (mflux), loading the model once.

Input: every photo in data/sources/<origin>/<category>/*.jpg.
Output: data/scenes/photo/<name>.png and data/scenes/styled/<name>.png at 512 x 512, with <name> = <origin>_<category>_<stem>,
        and data/scenes/log.csv (one row per photo, processed or skipped).
Resumable: a pair already made is not redone. The order is shuffled (fixed seed) so that every category is
represented even if the run is stopped early. Roughly 10-20 s per 512 x 512 image on an Apple M5 Pro.

    caffeinate -i uv run --python 3.12 --with mflux --with pillow python teacher/restyle_scenes.py [--limit N]
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import SCENE_PROMPT, SEED, SIZE, STEPS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "sources"
SCENES = ROOT / "data" / "scenes"
MODEL = ROOT / "models" / "flux2-klein-4b-q8"


def is_grayscale_or_sepia(img: Image.Image) -> bool:
    """Black and white or sepia: very low mean saturation, or a single hue."""
    h, s, _ = img.resize((64, 64)).convert("HSV").split()
    sat = sum(s.getdata()) / (64 * 64 * 255)
    if sat < 0.08:
        return True
    hues = [t for t, ss in zip(h.getdata(), s.getdata()) if ss > 40]
    if len(hues) > 500:
        mean = sum(hues) / len(hues)
        spread = (sum((t - mean) ** 2 for t in hues) / len(hues)) ** 0.5
        return spread < 6 and sat < 0.35  # everything in the same hue: sepia
    return False


def load_teacher():
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2KleinEdit

    return Flux2KleinEdit(model_path=str(MODEL), model_config=ModelConfig.flux2_klein_4b())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="stop after N new pairs")
    args = ap.parse_args()

    photos = sorted(p for p in SOURCES.glob("*/*/*.jpg"))
    random.Random(2026).shuffle(photos)
    (SCENES / "photo").mkdir(parents=True, exist_ok=True)
    (SCENES / "styled").mkdir(parents=True, exist_ok=True)
    log = SCENES / "log.csv"
    done = set()
    if log.exists():
        done = {r["name"] for r in csv.DictReader(log.open())}

    model = load_teacher()

    made = 0
    with log.open("a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["name", "origin", "category", "status", "seconds"])
        for photo in photos:
            origin, category = photo.parts[-3], photo.parts[-2]
            name = f"{origin}_{category}_{photo.stem}"
            if name in done:
                continue
            try:
                img = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
            except Exception:
                w.writerow([name, origin, category, "unreadable", ""])
                continue
            if min(img.size) < 400:
                w.writerow([name, origin, category, "too_small", ""])
                continue
            if is_grayscale_or_sepia(img):
                w.writerow([name, origin, category, "grayscale", ""])
                continue
            source = SCENES / "photo" / f"{name}.png"
            ImageOps.fit(img, (SIZE, SIZE), Image.LANCZOS).save(source)
            start = time.time()
            out = model.generate_image(
                seed=SEED, prompt=SCENE_PROMPT, num_inference_steps=STEPS,
                height=SIZE, width=SIZE, image_paths=[str(source)],
            )
            out.image.convert("RGB").save(SCENES / "styled" / f"{name}.png")
            seconds = time.time() - start
            w.writerow([name, origin, category, "ok", f"{seconds:.1f}"])
            f.flush()
            made += 1
            if made % 25 == 0 or made <= 3:
                print(f"{made} pairs · last {seconds:.1f} s · {name}", flush=True)
            if args.limit and made >= args.limit:
                break
    print(f"done: {made} new pairs", flush=True)


if __name__ == "__main__":
    main()
