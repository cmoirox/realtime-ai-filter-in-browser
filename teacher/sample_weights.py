"""Faces: per-example sampling weights. Shows more often the open mouths (base value = mouth openness from
data/faces/mouths.csv), beards and moustaches, bald heads, darker skin tones, glasses and profiles (bonuses read from
the descriptions of describe_faces.py).

Output: data/faces/sample_weights.csv with columns `name,value`, one row per described face. The student trainer turns
`value` into a sampling frequency with a cap of 2: an example at 0.4 is seen 6 times more often, at 0.12 2.5 times.

    python3 teacher/sample_weights.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import parse_description  # noqa: E402

FACES = Path(__file__).resolve().parent.parent / "data" / "faces"


def main() -> None:
    mouth = {r["name"]: float(r["openness"]) for r in csv.DictReader((FACES / "mouths.csv").open())}
    descs = {}
    for f in (FACES / "descriptions.csv", FACES / "profiles" / "descriptions.csv"):
        if f.exists():
            descs.update({r[0]: r[1] for r in csv.reader(f.open()) if len(r) > 1})
    with (FACES / "sample_weights.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "value"])
        for name, text in sorted(descs.items()):
            d = parse_description(text)
            v = mouth.get(name, 0.0)
            if d.get("person") == "man" and (not d.get("beard", "no").startswith("no") or d.get("moustache", "no").startswith("yes")):
                v += 0.12
            if d.get("hair", "") in ("bald", "balding"):
                v += 0.10
            if d.get("skin", "") in ("brown", "dark brown", "very dark", "dark"):
                v += 0.08
            if not d.get("glasses", "no").startswith("no"):
                v += 0.05
            if name.startswith("p_"):
                v += 0.08
            w.writerow([name, f"{v:.3f}"])


if __name__ == "__main__":
    main()
