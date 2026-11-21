"""Restyles every head in a photo, with the same pipeline as the web demo (web/src/main.ts):
aligned head square -> head model -> soft edge, clipped to the person's silhouette (zones model) -> pasted back.

    uv run --python 3.11 --with torch --with mediapipe --with opencv-python-headless --with numpy \
        python tools/restyle_photo.py photo.jpg [more.jpg ...] --out out_dir [--side-by-side]

Needs weights/head_g32.pt, weights/zones_m.pt (GitHub Release) and models/face_landmarker.task
(or web/public/models/face_landmarker.task after `npm run models`).
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "student"))
from model import convert_legacy_state_dict, create  # noqa: E402

LEFT_EYE, RIGHT_EYE, MOUTH = [33, 133], [362, 263], [61, 291]
N, NZ = 256, 160
RED = (150, 220)  # occluder threshold on the zones output (0-255), as in the web demo


def aligned_square(pts: np.ndarray) -> tuple:
    """Same convention as teacher/detect_faces.py and the web demo: (cx, cy, side, angle in radians)."""
    l, r, m = pts[LEFT_EYE].mean(0), pts[RIGHT_EYE].mean(0), pts[MOUTH].mean(0)
    eyes = (l + r) / 2
    c = eyes + 0.25 * (m - eyes)
    side = max(4.0 * np.linalg.norm(r - l), 3.6 * np.linalg.norm(m - eyes))
    return c[0], c[1], side, math.atan2(r[1] - l[1], r[0] - l[0])


def square_matrix(q: tuple, n: int) -> np.ndarray:
    """2x3 affine: pixel of the n x n square -> pixel of the image."""
    cx, cy, side, a = q
    e = side / n
    c, s = math.cos(a) * e, math.sin(a) * e
    return np.array([[c, -s, cx - c * n / 2 + s * n / 2], [s, c, cy - s * n / 2 - c * n / 2]], np.float32)


def soft_edge(n: int, radius: float = 0.45, fade: float = 0.13) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    u, v = np.abs((x + 0.5) / n - 0.5), np.abs((y + 0.5) / n - 0.5)
    a = np.clip((radius - (u ** 3 + v ** 3) ** (1 / 3)) / fade, 0, 1)
    return a * a * (3 - 2 * a)


def run(model: torch.nn.Module, rgb: np.ndarray, dev: str) -> np.ndarray:
    """uint8 RGB HxWx3 -> float RGB in [0, 255]."""
    x = torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(dev) / 127.5 - 1
    with torch.no_grad():
        return model(x)[0].clamp(-1, 1).add(1).mul(127.5).permute(1, 2, 0).cpu().numpy()


def load(size: str, path: Path, dev: str) -> torch.nn.Module:
    m = create(size)
    m.load_state_dict(convert_legacy_state_dict(torch.load(path, map_location="cpu", weights_only=True)))
    return m.eval().to(dev)


def restyle(img_bgr: np.ndarray, det, head, zones, dev: str) -> tuple[np.ndarray, int]:
    h, w = img_bgr.shape[:2]
    rgb = np.ascontiguousarray(img_bgr[..., ::-1])
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    out = img_bgr.astype(np.float32)
    edge = soft_edge(N)
    for face in res.face_landmarks:
        q = aligned_square(np.array([[p.x * w, p.y * h] for p in face]))
        M = square_matrix(q, N)
        crop = cv2.warpAffine(rgb, M, (N, N), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP, borderValue=(119, 119, 119))
        styled = run(head, crop, dev)
        alpha = edge.copy()
        if zones is not None:
            z = run(zones, cv2.resize(crop, (NZ, NZ), interpolation=cv2.INTER_AREA), dev)
            red = cv2.erode(z[..., 0], np.ones((3, 3), np.uint8))  # only solid occluders, no speckles
            red = np.clip((red - RED[0]) / (RED[1] - RED[0]), 0, 1)
            person = np.clip((np.maximum(z[..., 1], z[..., 2]) - 24) * 5 / 255, 0, 1) * (1 - red)
            alpha *= cv2.resize(person, (N, N), interpolation=cv2.INTER_LINEAR)
        excess = styled[..., 2] - styled[..., 1]  # bluish fringes at the crop border (RGB order here)
        alpha *= np.clip(1 - (excess - 12) / 22, 0, 1)
        rgba = np.dstack([styled[..., ::-1] * alpha[..., None], alpha]).astype(np.float32)
        warped = cv2.warpAffine(rgba, M, (w, h), flags=cv2.INTER_LINEAR)
        out = warped[..., :3] + out * (1 - warped[..., 3:4])
    return out.clip(0, 255).astype(np.uint8), len(res.face_landmarks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", default="weights/head_g32.pt")
    ap.add_argument("--zones", default="weights/zones_m.pt", help="'' to paste the whole square")
    ap.add_argument("--max-side", type=int, default=1280, help="downscale larger photos first")
    ap.add_argument("--side-by-side", action="store_true", help="write original | restyled")
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    head = load("g32", ROOT / args.head, dev)
    zones = load("m", ROOT / args.zones, dev) if args.zones else None
    task = next(p for p in (ROOT / "models/face_landmarker.task", ROOT / "web/public/models/face_landmarker.task") if p.exists())
    det = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(task)), num_faces=10, min_face_detection_confidence=0.5))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in args.photos:
        img = cv2.imread(p)
        if img is None:
            continue
        k = args.max_side / max(img.shape[:2])
        if k < 1:
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        styled, n = restyle(img, det, head, zones, dev)
        cv2.imwrite(str(out_dir / f"{Path(p).stem}.jpg"), np.hstack([img, styled]) if args.side_by_side else styled, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{Path(p).name}: {n} face(s)", flush=True)


if __name__ == "__main__":
    main()
