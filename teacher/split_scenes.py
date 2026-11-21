"""Scenes: apply the quality-control thresholds and split into train / val / test.

Reads data/scenes/quality.csv (quality_scenes.py), people.csv (detect_invented_people.py) and log.csv (restyle_scenes.py).
Writes data/scenes/verdict.csv (verdict and reason per pair) and data/splits/scenes_{train,val,test}.txt.
Thresholds were tuned by eye on contact sheets of the worst cases of each measure.

    python3 teacher/split_scenes.py
"""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "data" / "scenes"
SPLITS = ROOT / "data" / "splits"
N_TEST = 20


def reason(q: dict, invented: int) -> str:
    if invented >= 1:
        return "invented_person"
    if q["banner"] >= 0.30:
        return "invented_text"
    if q["brightness"] > 0.30:
        return "too_bright"       # night turned into daylight
    if q["edges"] < 0.55 or q["layout"] < 0.45:
        return "structure_changed"
    return ""


def main() -> None:
    people = {r["name"]: int(r["invented"]) for r in csv.DictReader((SCENES / "people.csv").open())}
    category = {r["name"]: r["category"] for r in csv.DictReader((SCENES / "log.csv").open())}
    rows = []
    for r in csv.DictReader((SCENES / "quality.csv").open()):
        q = {k: (v if k == "name" else float(v)) for k, v in r.items()}
        m = reason(q, people[q["name"]])
        rows.append({"name": q["name"], "category": category[q["name"]], "verdict": "rejected" if m else "kept", "reason": m})
    with (SCENES / "verdict.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    kept = sorted(r["name"] for r in rows if r["verdict"] == "kept")
    rnd = random.Random(2026)
    rnd.shuffle(kept)
    by_category = defaultdict(list)
    for n in kept:
        by_category[category[n]].append(n)
    test = []                                   # 20 test images, cycling through the categories
    while len(test) < N_TEST:
        for c in sorted(by_category):
            if len(test) < N_TEST and by_category[c]:
                test.append(by_category[c].pop())
    rest = [n for n in kept if n not in set(test)]
    n_val = len(rest) // 10
    SPLITS.mkdir(parents=True, exist_ok=True)
    for split, names in (("test", test), ("val", rest[:n_val]), ("train", rest[n_val:])):
        (SPLITS / f"scenes_{split}.txt").write_text("\n".join(sorted(names)) + "\n")
        print(f"{split:6s} {len(names)}")

    print(f"\n{len(rows)} pairs, {len(kept)} kept")
    for m, n in Counter(r["reason"] for r in rows if r["reason"]).most_common():
        print(f"  rejected {m:18s} {n}")
    total, ok = Counter(r["category"] for r in rows), Counter(r["category"] for r in rows if r["verdict"] == "kept")
    print("\nkept per category:")
    for c in sorted(total):
        print(f"  {c:22s} {ok[c]:4d} / {total[c]:4d}  ({100 * ok[c] // total[c]} %)")


if __name__ == "__main__":
    main()
