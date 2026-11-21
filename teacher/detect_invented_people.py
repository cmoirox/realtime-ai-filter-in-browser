"""Scenes: flag people invented by the teacher.

A person detector (YOLOv8n) runs on the photo and on its styled version. A person seen in the styled
image with nobody at the same place in the photo counts as "invented".
Output: data/scenes/people.csv (name, n_photo, n_styled, invented).

    uv run --python 3.12 --with ultralytics python teacher/detect_invented_people.py
"""
import csv
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "data" / "scenes"
BATCH = 64


def boxes(res, threshold: float, min_height: float = 0.0):
    out = []
    for b, c, k in zip(res.boxes.xyxyn.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
        if int(k) == 0 and c >= threshold and (b[3] - b[1]) >= min_height:
            out.append(b)
    return out


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def inside(a, b) -> bool:
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def main() -> None:
    model = YOLO(str(ROOT / "models" / "yolov8n.pt"))
    names = sorted(f.stem for f in (SCENES / "styled").glob("*.png"))
    with (SCENES / "people.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "n_photo", "n_styled", "invented"])
        for i in range(0, len(names), BATCH):
            batch = names[i:i + BATCH]
            rp = model.predict([str(SCENES / "photo" / f"{n}.png") for n in batch], conf=0.08, imgsz=512, device="mps", verbose=False)
            rs = model.predict([str(SCENES / "styled" / f"{n}.png") for n in batch], conf=0.08, imgsz=512, device="mps", verbose=False)
            for n, a, b in zip(batch, rp, rs):
                loose = boxes(a, 0.08)                 # photo: low threshold, so as not to miss a real person
                clear = boxes(b, 0.45, 0.08)           # styled: high threshold, clearly visible people only
                invented = sum(1 for s in clear if not any(iou(s, p) >= 0.15 or inside(s, p) for p in loose))
                w.writerow([n, len(boxes(a, 0.45, 0.08)), len(clear), invented])
            print(f"{min(i + BATCH, len(names))}/{len(names)}", flush=True)


if __name__ == "__main__":
    main()
