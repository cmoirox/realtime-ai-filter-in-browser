"""Profiles: simple check of the restyled profile pairs (MediaPipe landmarks are useless on a true profile, so only:
large masses kept and not brightened too much), then add them to the face set: hard links in data/faces/photo and
data/faces/styled, names appended to data/splits/faces_train.txt, 36 held out in data/splits/faces_profiles_holdout.txt
(never seen in training). Run after split_faces.py (which rewrites faces_train.txt).

    uv run --python 3.11 --with pillow --with numpy python teacher/add_profiles.py
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality_scenes import corr, gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FACES = ROOT / "data" / "faces"
PROFILES = FACES / "profiles"
SPLITS = ROOT / "data" / "splits"


def small(im: Image.Image) -> np.ndarray:
    return np.asarray(im.resize((32, 32), Image.BOX).convert("L"), dtype=np.float32)


def main() -> None:
    kept, rejected = [], 0
    for f in sorted((PROFILES / "styled").glob("p_*.png")):
        p, s = Image.open(PROFILES / "photo" / f.name).convert("RGB"), Image.open(f).convert("RGB")
        if corr(small(p), small(s)) < 0.35 or float(gray(s).mean() - gray(p).mean()) > 0.30:
            rejected += 1
            continue
        kept.append(f.stem)
        for src, dest in ((PROFILES / "photo" / f.name, FACES / "photo" / f.name), (f, FACES / "styled" / f.name)):
            if not dest.exists():
                os.link(src, dest)
    random.Random(11).shuffle(kept)
    holdout, train = sorted(kept[:36]), kept[36:]
    (SPLITS / "faces_profiles_holdout.txt").write_text("\n".join(holdout) + "\n")
    split = SPLITS / "faces_train.txt"
    names = [n for n in split.read_text().split() if not n.startswith("p_")] + sorted(train)
    split.write_text("\n".join(names) + "\n")
    print(f"{len(kept)} profiles kept, {rejected} rejected; train: {len(names)} examples including {len(train)} profiles")


if __name__ == "__main__":
    main()
