"""Faces: quality control of the photo face -> styled face pairs.

For every pair, finds the face in the photo and in the styled image (MediaPipe) and measures:
  photo_ok   : 1 if a face is found in the photo (0 = not actually a face)
  found      : 1 if a face is found in the styled image (0 = face lost)
  offset     : displacement of the eye midpoint and of the mouth, as a fraction of the image side
  scale      : ratio of the eye distances (styled / photo)
  angle      : difference in eye-line tilt, in degrees
  edges, layout, brightness : as in quality_scenes.py
Output: data/faces/quality.csv. Thresholds are applied by split_faces.py.

    uv run --python 3.11 --with mediapipe --with pillow --with numpy python teacher/quality_faces.py
"""
import csv
import math
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_faces import LANDMARKER, LEFT_EYE, MOUTH, RIGHT_EYE  # noqa: E402
from quality_scenes import corr, edges, gray  # noqa: E402

FACES = Path(__file__).resolve().parent.parent / "data" / "faces"


def points(det, img: Image.Image):
    a = np.ascontiguousarray(np.asarray(img))
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=a))
    if not res.face_landmarks:
        return None
    # the largest face
    v = max(res.face_landmarks, key=lambda f: abs(f[RIGHT_EYE[1]].x - f[LEFT_EYE[0]].x))
    pts = np.array([[q.x, q.y] for q in v])
    return pts[LEFT_EYE].mean(0), pts[RIGHT_EYE].mean(0), pts[MOUTH].mean(0)


def small(im: Image.Image) -> np.ndarray:
    return np.asarray(im.resize((32, 32), Image.BOX).convert("L"), dtype=np.float32)


def main() -> None:
    names = sorted(f.stem for f in (FACES / "styled").glob("*.png"))
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER)),
        num_faces=3, min_face_detection_confidence=0.3)
    rows = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as det:
        for i, name in enumerate(names):
            p = Image.open(FACES / "photo" / f"{name}.png").convert("RGB")
            s = Image.open(FACES / "styled" / f"{name}.png").convert("RGB")
            gp, gs = gray(p), gray(s)
            r = {"name": name, "photo_ok": 0, "found": 0, "offset": "", "scale": "", "angle": "",
                 "edges": round(corr(edges(gp), edges(gs)), 4), "layout": round(corr(small(p), small(s)), 4),
                 "brightness": round(float(gs.mean() - gp.mean()), 4)}
            a, b = points(det, p), points(det, s)
            r["photo_ok"], r["found"] = int(a is not None), int(b is not None)
            if a is not None and b is not None:
                eyes_a, eyes_b = (a[0] + a[1]) / 2, (b[0] + b[1]) / 2
                r["offset"] = round(float(max(np.linalg.norm(eyes_b - eyes_a), np.linalg.norm(b[2] - a[2]))), 4)
                r["scale"] = round(float(np.linalg.norm(b[1] - b[0]) / max(np.linalg.norm(a[1] - a[0]), 1e-6)), 4)
                tilt = lambda t: math.degrees(math.atan2(t[1][1] - t[0][1], t[1][0] - t[0][0]))
                r["angle"] = round(abs(tilt(b) - tilt(a)), 2)
            rows.append(r)
            if i % 500 == 0:
                print(f"{i}/{len(names)}", flush=True)
    with (FACES / "quality.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} pairs; face found in the photo: {sum(r['photo_ok'] for r in rows)}, in the styled image: {sum(r['found'] for r in rows)}")
    for k in ("offset", "scale", "angle", "edges", "layout", "brightness"):
        v = np.array([r[k] for r in rows if r[k] != ""], dtype=float)
        print(f"{k:10s} " + "  ".join(f"p{q:02d}={np.percentile(v, q):.3f}" for q in (1, 5, 25, 50, 75, 95, 99)))


if __name__ == "__main__":
    main()
