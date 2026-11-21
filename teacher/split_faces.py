"""Faces: apply the quality-control thresholds (quality_faces.py) and split the face set.
Writes data/faces/verdict.csv (name, reason) and data/splits/faces_{train,val,test}.txt.
Profiles are added to faces_train.txt afterwards by add_profiles.py.

    python3 teacher/split_faces.py
"""
import csv
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACES = ROOT / "data" / "faces"
SPLITS = ROOT / "data" / "splits"
N_TEST = 24


def reason(r: dict) -> str:
    if r["photo_ok"] == "0":
        return "not_a_face"
    if r["found"] == "0":
        return "face_lost"
    if float(r["offset"]) > 0.05 or not 0.88 <= float(r["scale"]) <= 1.15 or float(r["angle"]) > 8:
        return "face_moved"
    if float(r["brightness"]) > 0.30:
        return "too_bright"
    if float(r["layout"]) < 0.35:
        return "structure_changed"
    return ""


def main() -> None:
    rows = [{"name": r["name"], "reason": reason(r)} for r in csv.DictReader((FACES / "quality.csv").open())]
    with (FACES / "verdict.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "reason"])
        w.writeheader()
        w.writerows(rows)
    kept = sorted(r["name"] for r in rows if not r["reason"])
    random.Random(2026).shuffle(kept)
    n_val = (len(kept) - N_TEST) // 12
    splits = {"test": kept[:N_TEST], "val": kept[N_TEST:N_TEST + n_val], "train": kept[N_TEST + n_val:]}
    SPLITS.mkdir(parents=True, exist_ok=True)
    for split, names in splits.items():
        (SPLITS / f"faces_{split}.txt").write_text("\n".join(sorted(names)) + "\n")
    print(f"{len(rows)} pairs, {len(kept)} kept:", {k: len(v) for k, v in splits.items()})
    print("rejected:", dict(Counter(r["reason"] for r in rows if r["reason"])))


if __name__ == "__main__":
    main()
