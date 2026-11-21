"""Faces: add more varied faces (darker skin tones, other ethnicities, older people) taken from FairFace
(CC BY 4.0, "validation" split, padding 1.25). Nothing is downloaded: put the parquet file in
data/sources/fairface/1.25/validation-*.parquet first.
Same aligned square as the rest of the face set. Output: data/faces/photo/f_<row>.png and data/faces/fairface.csv
(name, ethnicity, age, gender), which keeps the FairFace labels for attribution and for restyle_faces.py.

    uv run --python 3.11 --with mediapipe --with opencv-python-headless --with numpy --with pyarrow python teacher/add_fairface.py
"""
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_faces import LANDMARKER, aligned_square_rad, square_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FAIRFACE = ROOT / "data" / "sources" / "fairface" / "1.25"
FACES = ROOT / "data" / "faces"
ETHNICITIES = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino", "Southeast Asian"]
AGES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
QUOTAS = {2: 320, 1: 160, 4: 140, 5: 120, 6: 90, 0: 90, 3: 150}   # "White": only 60 and over (grey and white hair)
MIN_SIDE, OVERFLOW = 200, 0.12


def main() -> None:
    t = pq.read_table(next(FAIRFACE.glob("validation-*.parquet")))
    det = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER)), num_faces=1))
    (FACES / "photo").mkdir(parents=True, exist_ok=True)
    taken = {k: 0 for k in QUOTAS}
    with (FACES / "fairface.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "ethnicity", "age", "gender"])
        for i in range(t.num_rows):
            if all(taken[k] >= q for k, q in QUOTAS.items()):
                break
            row = t.slice(i, 1).to_pylist()[0]
            o, a = row["race"], row["age"]
            # skip babies and "White" faces under 60, and thin out the (over-represented) 20-39 age range
            if taken[o] >= QUOTAS[o] or a == 0 or (o == 3 and a < 7) or (a >= 3 and a <= 4 and taken[o] % 3 == 0 and i % 2 == 0):
                continue
            img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
            H, W = img.shape[:2]
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img[..., ::-1])))
            if not res.face_landmarks:
                continue
            q = aligned_square_rad(np.array([[p.x * W, p.y * H] for p in res.face_landmarks[0]]))
            c = q[2] / 2
            if q[2] < MIN_SIDE or q[0] - c < -OVERFLOW * q[2] or q[1] - c < -OVERFLOW * q[2] or q[0] + c > W + OVERFLOW * q[2] or q[1] + c > H + OVERFLOW * q[2]:
                continue
            out = cv2.warpAffine(img, square_matrix(q, 512), (512, 512), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(str(FACES / "photo" / f"f_{i}.png"), out)
            w.writerow([f"f_{i}", ETHNICITIES[o], AGES[a], "man" if row["gender"] == 0 else "woman"])
            taken[o] += 1
    print({ETHNICITIES[k]: v for k, v in taken.items()})


if __name__ == "__main__":
    main()
