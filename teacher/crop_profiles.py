"""Profiles: crop to 512 x 512 the turned heads of FairFace found by find_profiles.py.
- face found by MediaPipe and turned by at least MIN_YAW degrees: the usual square (from the face landmarks);
- face lost (true profile): square derived from the "face skin" region of the multiclass selfie segmentation, with a
  rule fitted on the turned faces MediaPipe still finds (side = k x sqrt(area), centre shifted towards the front).
Outputs: data/faces/profiles/photo/p_<split initial><row>.png and data/faces/profiles/crops.csv
(name, ethnicity, age, gender, square, yaw), filtered next by select_profiles.py.

    uv run --python 3.11 --with mediapipe --with opencv-python-headless --with numpy --with pyarrow --with torch python teacher/crop_profiles.py
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
from detect_faces import square_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FAIRFACE = ROOT / "data" / "sources" / "fairface" / "1.25"
FACES = ROOT / "data" / "faces"
PROFILES = FACES / "profiles"
SEGMENTER = ROOT / "models" / "selfie_multiclass_256x256.tflite"
MIN_YAW, MIN_SIDE, OVERFLOW = 40, 170, 0.15
ETHNICITIES = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino", "Southeast Asian"]
AGES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]


def regions(seg, img):
    """Area and centre of the face skin, and facing direction (+1: looks towards image right) from the hair position."""
    cat = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img[..., ::-1]))).category_mask.numpy_view()
    cat = np.squeeze(np.array(cat))
    n, _, stats, cent = cv2.connectedComponentsWithStats((cat == 3).astype(np.uint8))
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[k, cv2.CC_STAT_AREA])
    ys, xs = np.nonzero(cat == 1)
    facing = 0.0
    if len(xs) > 0.08 * area:
        facing = (cent[k][0] - xs.mean()) / math.sqrt(area)
    return area, cent[k], facing


def main() -> None:
    (PROFILES / "photo").mkdir(parents=True, exist_ok=True)
    already = {r["name"] for r in csv.DictReader((FACES / "fairface.csv").open())}
    cand = [r for r in csv.DictReader((PROFILES / "candidates.csv").open())
            if r["found"] == "0" or abs(float(r["yaw"])) >= MIN_YAW]
    # FairFace validation faces already in the frontal set (add_fairface.py) are not reused
    cand = [r for r in cand if not (r["split"].startswith("validation") and f"f_{r['row']}" in already)]
    wanted = {(r["split"], int(r["row"])): r for r in cand}
    seg = mp.tasks.vision.ImageSegmenter.create_from_options(mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(SEGMENTER)), output_category_mask=True))

    # pass 1: images and regions
    items = []
    for file in sorted(FAIRFACE.glob("*.parquet")):
        split = file.name.split("-of-")[0]
        row = -1
        for batch in pq.ParquetFile(file).iter_batches(batch_size=256, columns=["image"]):
            for r in batch.to_pylist():
                row += 1
                c = wanted.get((split, row))
                if c is None:
                    continue
                img = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
                z = regions(seg, img)
                if z is not None:
                    items.append((c, img, z))
    print(len(cand), "candidates,", len(items), "with face skin", flush=True)

    # rule fitted on faces that are found and well turned (yaw >= 50)
    ks, dxs, dys, agree = [], [], [], []
    for c, img, (area, cen, facing) in items:
        if c["found"] == "1" and abs(float(c["yaw"])) >= 50 and abs(facing) > 0.15:
            r = math.sqrt(area)
            s = 1.0 if facing > 0 else -1.0
            ks.append(float(c["side"]) / r)
            dxs.append(s * (float(c["cx"]) - cen[0]) / r)
            dys.append((float(c["cy"]) - cen[1]) / r)
            agree.append((float(c["yaw"]) > 0) == (facing > 0))
    K, DX, DY = np.median(ks), np.median(dxs), np.median(dys)
    print(f"rule: side = {K:.2f} x sqrt(area) (std {np.std(ks):.2f}), centre shifted {DX:.2f} forwards, {DY:.2f} down; "
          f"hair direction agrees with yaw: {np.mean(agree):.2f} over {len(ks)}", flush=True)

    tally = {"landmarks": 0, "segmentation": 0, "small": 0, "edge": 0, "direction": 0}
    with (PROFILES / "crops.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "ethnicity", "age", "gender", "square", "yaw"])
        for c, img, (area, cen, facing) in items:
            H, W = img.shape[:2]
            if c["found"] == "1":
                q, mode = np.array([float(c[k]) for k in ("cx", "cy", "side", "angle")]), "landmarks"
            else:
                if abs(facing) < 0.15:
                    tally["direction"] += 1
                    continue
                r, s = math.sqrt(area), (1.0 if facing > 0 else -1.0)
                q, mode = np.array([cen[0] + s * DX * r, cen[1] + DY * r, K * r, 0.0]), "segmentation"
            d = q[2] / 2
            if q[2] < MIN_SIDE:
                tally["small"] += 1
                continue
            if max(d - q[0], d - q[1], q[0] + d - W, q[1] + d - H) > OVERFLOW * q[2]:
                tally["edge"] += 1
                continue
            name = f"p_{c['split'][0]}{c['row']}"
            out = cv2.warpAffine(img, square_matrix(q, 512), (512, 512), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(str(PROFILES / "photo" / f"{name}.png"), out)
            w.writerow([name, ETHNICITIES[int(c["ethnicity"])], AGES[int(c["age"])], "man" if c["gender"] == "0" else "woman", mode,
                        c["yaw"] or ("90" if facing > 0 else "-90")])
            tally[mode] += 1
    print(tally)


if __name__ == "__main__":
    main()
