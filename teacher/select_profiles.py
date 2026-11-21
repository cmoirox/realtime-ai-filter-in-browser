"""Profiles: drop unusable crops (black and white, too dark, black borders, too blurry) and keep the same number per
ethnicity, half true profiles ("segmentation" square), half strongly turned heads ("landmarks" square, most turned first).
Output: data/faces/profiles/selected.csv (same columns as crops.csv).

    uv run --python 3.11 --with opencv-python-headless --with numpy python teacher/select_profiles.py [count per ethnicity]
"""
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROFILES = Path(__file__).resolve().parent.parent / "data" / "faces" / "profiles"
PER_ETHNICITY = int(sys.argv[1]) if len(sys.argv) > 1 else 110


def main() -> None:
    rows = list(csv.DictReader((PROFILES / "crops.csv").open()))
    good, tally, sharpness = [], Counter(), {}
    for r in rows:
        img = cv2.imread(str(PROFILES / "photo" / f"{r['name']}.png"))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if hsv[..., 1].mean() < 28:
            tally["gray"] += 1
        elif hsv[..., 2].mean() < 88:        # too dark: the automatic description goes wrong (woman described as bearded)
            tally["dark"] += 1
        elif (img.max(2) == 0).mean() > 0.03:
            tally["black border"] += 1
        else:
            sharpness[r["name"]] = cv2.Laplacian(cv2.cvtColor(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
            good.append(r)
    threshold = np.percentile(list(sharpness.values()), 20)
    tally["blurry"] = sum(sharpness[r["name"]] < threshold for r in good)
    good = [r for r in good if sharpness[r["name"]] >= threshold]
    groups = defaultdict(list)
    for r in good:
        groups[(r["ethnicity"], r["square"])].append(r)
    rnd, selected = random.Random(7), []
    for (eth, mode), g in sorted(groups.items()):
        if mode == "segmentation":
            rnd.shuffle(g)
        else:
            g.sort(key=lambda r: -abs(float(r["yaw"])))
        selected += g[:PER_ETHNICITY if eth == "White" else PER_ETHNICITY // 2]     # "White" doubled: only source of blond, red and white hair
    with (PROFILES / "selected.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(selected)
    print(dict(tally), "| selected:", len(selected), dict(Counter(r["ethnicity"] for r in selected)), dict(Counter(r["square"] for r in selected)),
          dict(Counter(r["gender"] for r in selected)))


if __name__ == "__main__":
    main()
