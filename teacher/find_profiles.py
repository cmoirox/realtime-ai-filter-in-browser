"""Profiles: scan FairFace (parquet files in data/sources/fairface/1.25) and record for each photo whether MediaPipe
finds the face, how far the head is turned (yaw, in degrees) and the aligned square (angle in radians).
Output: data/faces/profiles/candidates.csv (split, row, ethnicity, age, gender, found, yaw, pitch, cx, cy, side, angle).
Resumable.

    uv run --python 3.11 --with mediapipe --with opencv-python-headless --with numpy --with pyarrow python teacher/find_profiles.py
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_faces import LANDMARKER, aligned_square_rad  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FAIRFACE = ROOT / "data" / "sources" / "fairface" / "1.25"
PROFILES = ROOT / "data" / "faces" / "profiles"


def main() -> None:
    det = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER)), num_faces=1,
        output_facial_transformation_matrixes=True))
    PROFILES.mkdir(parents=True, exist_ok=True)
    out = PROFILES / "candidates.csv"
    done = set()
    if out.exists():
        done = {(r["split"], r["row"]) for r in csv.DictReader(out.open())}
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["split", "row", "ethnicity", "age", "gender", "found", "yaw", "pitch", "cx", "cy", "side", "angle"])
        for file in sorted(FAIRFACE.glob("*.parquet")):
            split = file.name.split("-of-")[0]
            row = 0
            for batch in pq.ParquetFile(file).iter_batches(batch_size=256):
                for r in batch.to_pylist():
                    row += 1
                    if (split, str(row - 1)) in done:
                        continue
                    img = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
                    H, W = img.shape[:2]
                    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img[..., ::-1])))
                    base = [split, row - 1, r["race"], r["age"], r["gender"]]
                    if not res.face_landmarks:
                        w.writerow(base + [0, "", "", "", "", "", ""])
                        continue
                    m = np.array(res.facial_transformation_matrixes[0])[:3, :3]
                    yaw = math.degrees(math.atan2(m[0, 2], m[2, 2]))
                    pitch = math.degrees(math.asin(max(-1, min(1, -m[1, 2]))))
                    q = aligned_square_rad(np.array([[p.x * W, p.y * H] for p in res.face_landmarks[0]]))
                    w.writerow(base + [1, f"{yaw:.1f}", f"{pitch:.1f}", f"{q[0]:.1f}", f"{q[1]:.1f}", f"{q[2]:.1f}", f"{q[3]:.4f}"])
                if row % 2048 == 0:
                    print(split, row, flush=True)
                    f.flush()
    print("done")


if __name__ == "__main__":
    main()
