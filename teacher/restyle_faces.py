"""Faces: restyle every aligned face crop with the FLUX.2 klein 4B teacher, using a prompt that NAMES skin, hair,
eyes, facial hair, glasses and headwear (descriptions from describe_faces.py); otherwise the teacher lightens skin
tones and changes hair. Roughly 10-20 s per 512 x 512 image on an Apple M5 Pro.

    caffeinate -i uv run --python 3.12 --with mflux --with pillow python teacher/restyle_faces.py [--profiles] [--no-description] [--limit N] [--trial DIR NAME ...]

Default: data/faces/photo -> data/faces/styled/<name>.png, faces taken from data/faces/descriptions.csv, FairFace
ethnicity labels (data/faces/fairface.csv) used to correct the described skin tone. Resumable, shuffled order.
--profiles        : side views, data/faces/profiles/photo -> data/faces/profiles/styled, only the crops listed in
                    profiles/selected.csv whose description is not a front or back view.
--no-description  : plain FACE_PROMPT for every photo in data/faces/photo (no descriptions needed).
--trial DIR NAME… : restyle only these names into DIR, logging every prompt.
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import FACE_PROMPT, SEED, SIZE, STEPS, describe_for_prompt, face_prompt, parse_description  # noqa: E402
from restyle_scenes import is_grayscale_or_sepia, load_teacher  # noqa: E402

FACES = Path(__file__).resolve().parent.parent / "data" / "faces"


def jobs(args) -> tuple[Path, Path, list[tuple[str, str]]]:
    """(photo folder, output folder, [(name, prompt description)]) in processing order."""
    if args.no_description:
        names = [p.stem for p in sorted((FACES / "photo").glob("*.png"))]
        random.Random(2026).shuffle(names)
        return FACES / "photo", FACES / "styled", [(n, None) for n in names]
    base = FACES / "profiles" if args.profiles else FACES
    descs = {r[0]: r[1] for r in csv.reader((base / "descriptions.csv").open()) if len(r) > 1}
    ethnicity = {r[0]: r[1] for r in csv.reader((base / ("selected.csv" if args.profiles else "fairface.csv")).open())}
    if args.profiles:    # only the selected side views
        descs = {n: d for n, d in descs.items() if n in ethnicity and "front" not in parse_description(d).get("view of the head", "")
                 and "behind" not in parse_description(d).get("view of the head", "")}
    if args.trial:
        names = args.trial[1:]
    else:
        names = sorted(descs)
        random.Random(2027).shuffle(names)
    return base / "photo", base / "styled", [(n, describe_for_prompt(descs[n], ethnicity.get(n, ""))) for n in names if n in descs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", action="store_true", help="restyle the selected side-view crops")
    ap.add_argument("--no-description", action="store_true", help="plain face prompt, no per-face description")
    ap.add_argument("--limit", type=int, help="stop after N new faces")
    ap.add_argument("--trial", nargs="+", metavar="DIR NAME", help="output folder then face names to restyle")
    args = ap.parse_args()

    photos, dest, todo = jobs(args)
    if args.trial:
        dest = Path(args.trial[0])
    dest.mkdir(exist_ok=True, parents=True)
    model = load_teacher()
    made = 0
    for name, desc in todo:
        p = photos / f"{name}.png"
        if (dest / p.name).exists() or not p.exists() or is_grayscale_or_sepia(Image.open(p)):
            continue
        start = time.time()
        prompt = FACE_PROMPT if desc is None else face_prompt(desc)
        model.generate_image(seed=SEED, prompt=prompt, num_inference_steps=STEPS, height=SIZE, width=SIZE,
                             image_paths=[str(p)]).image.convert("RGB").save(dest / p.name)
        made += 1
        if made % 25 == 0 or made <= 3 or args.trial:
            print(f"{made} faces · last {time.time() - start:.1f} s · {name}" + (f" · {desc}" if desc else ""), flush=True)
        if args.limit and made >= args.limit:
            break
    print(f"done: {made} new faces", flush=True)


if __name__ == "__main__":
    main()
