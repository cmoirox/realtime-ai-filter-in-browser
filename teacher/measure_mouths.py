"""Faces: mouth openness of every face (gap between the lips / mouth width), measured on the photo of each styled pair.
Used to show open mouths more often during training (they are rare, and poorly learned); see sample_weights.py.
Output: data/faces/mouths.csv (name, openness).

    uv run --python 3.11 --with mediapipe --with pillow --with numpy python teacher/measure_mouths.py
"""
import csv
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FACES = ROOT / "data" / "faces"


def main() -> None:
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "models" / "face_landmarker.task")), num_faces=1)
    with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as det, (FACES / "mouths.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "openness"])
        for p in sorted((FACES / "styled").glob("*.png")):
            im = np.ascontiguousarray(np.asarray(Image.open(FACES / "photo" / p.name).convert("RGB")))
            r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=im))
            o = 0.0
            if r.face_landmarks:
                v = r.face_landmarks[0]
                o = abs(v[14].y - v[13].y) / max(abs(v[291].x - v[61].x), 1e-6)
            w.writerow([p.stem, f"{o:.3f}"])


if __name__ == "__main__":
    main()
