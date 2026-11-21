"""Faces: find the faces in the scene source photos with MediaPipe Face Landmarker (the same detector as the app)
and record, for each one, the aligned square to crop.

Square convention (the app must use the same one): centred slightly below the midpoint of the eyes,
side = 4 x the eye distance (at least 3.6 x the eyes-to-mouth distance), rotated so the eyes are horizontal.
Output: data/faces/landmarks.csv (file, cx, cy, side, angle in degrees).

    uv run --python 3.11 --with mediapipe --with pillow --with numpy python teacher/detect_faces.py [folder ...]
"""
import csv
import math
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LANDMARKER = ROOT / "models" / "face_landmarker.task"
LEFT_EYE, RIGHT_EYE = [33, 133], [362, 263]      # corners of each eye
MOUTH = [61, 291]


def aligned_square(pts: np.ndarray) -> tuple:
    """(cx, cy, side, angle in degrees) of the aligned square for 478 face landmarks in pixels."""
    l, r, m = pts[LEFT_EYE].mean(0), pts[RIGHT_EYE].mean(0), pts[MOUTH].mean(0)
    eyes = (l + r) / 2
    eye_dist, eye_mouth = np.linalg.norm(r - l), np.linalg.norm(m - eyes)
    side = max(4.0 * eye_dist, 3.6 * eye_mouth)
    centre = eyes + 0.25 * (m - eyes)
    return centre[0], centre[1], side, math.degrees(math.atan2(r[1] - l[1], r[0] - l[0]))


def aligned_square_rad(pts: np.ndarray) -> np.ndarray:
    """Same square as aligned_square, as an array [cx, cy, side, angle in radians] (for square_matrix)."""
    l, r, m = pts[LEFT_EYE].mean(0), pts[RIGHT_EYE].mean(0), pts[MOUTH].mean(0)
    eyes = (l + r) / 2
    eye_dist, eye_mouth = np.linalg.norm(r - l), np.linalg.norm(m - eyes)
    c = eyes + 0.25 * (m - eyes)
    return np.array([c[0], c[1], max(4.0 * eye_dist, 3.6 * eye_mouth), math.atan2(r[1] - l[1], r[0] - l[0])])


def square_matrix(q, n: int) -> np.ndarray:
    """2x3 affine: pixel of the n x n square -> pixel of the image (use with cv2.WARP_INVERSE_MAP). Angle in radians."""
    cx, cy, side, a = q
    e = side / n
    c, s = math.cos(a) * e, math.sin(a) * e
    return np.array([[c, -s, cx - c * n / 2 + s * n / 2], [s, c, cy - s * n / 2 - c * n / 2]], np.float32)


def main() -> None:
    from PIL import Image, ImageOps

    folders = sys.argv[1:] or [str(ROOT / "data" / "sources")]
    photos = sorted(p for d in folders for p in Path(d).resolve().rglob("*.jpg"))
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER)),
        num_faces=8, min_face_detection_confidence=0.5)
    out = ROOT / "data" / "faces" / "landmarks.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as det, out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "cx", "cy", "side", "angle"])
        for i, p in enumerate(photos):
            try:
                img = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
            except Exception:
                continue
            h, w_ = img.shape[:2]
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
            for face in res.face_landmarks:
                pts = np.array([[q.x * w_, q.y * h] for q in face])
                cx, cy, side, angle = aligned_square(pts)
                w.writerow([str(p.relative_to(ROOT)), f"{cx:.1f}", f"{cy:.1f}", f"{side:.1f}", f"{angle:.1f}"])
                n += 1
            if i % 500 == 0:
                print(f"{i}/{len(photos)} photos, {n} faces", flush=True)
    print(f"{len(photos)} photos, {n} faces -> {out}")


if __name__ == "__main__":
    main()
