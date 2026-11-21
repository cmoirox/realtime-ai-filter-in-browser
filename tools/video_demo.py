"""Offline "whole head redrawn every frame" demo (no mask pasted over the face).

For each video frame: MediaPipe landmarks -> aligned square of the head (same convention as the face dataset,
smoothed over time) -> face model -> pasted back whole (hair, forehead, neck included) with a wide feathered
edge. Output: side-by-side video original | variant 1 | variant 2...

    uv run --python 3.11 --with torch --with mediapipe --with opencv-python-headless --with numpy \
        python tools/video_demo.py <video> <output.mp4> g32:weights/head_g32.pt:256 [xl:runs/faces_xl/ema_last.pt:384]
Each variant is size:weights:n (weights path relative to the repo root). Needs models/face_landmarker.task and
models/selfie_multiclass_256x256.tflite, and ffmpeg on the PATH.
Options (after the variants): --smooth 0.5 (square smoothing), --temporal 0.0 (blend with the previous drawing),
--radius 0.47, --feather 0.12, --zones 1 (paste only over the person), --max 300 (frames)
"""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "student"))
from model import convert_legacy_state_dict, create  # noqa: E402

LEFT_EYE, RIGHT_EYE, MOUTH = [33, 133], [362, 263], [61, 291]


def head_square(pts):
    """(cx, cy, side, angle) of the aligned head square from face-mesh landmarks."""
    l, r, m = pts[LEFT_EYE].mean(0), pts[RIGHT_EYE].mean(0), pts[MOUTH].mean(0)
    eyes = (l + r) / 2
    eye_dist, eye_mouth = np.linalg.norm(r - l), np.linalg.norm(m - eyes)
    c = eyes + 0.25 * (m - eyes)
    return np.array([c[0], c[1], max(4.0 * eye_dist, 3.6 * eye_mouth), math.atan2(r[1] - l[1], r[0] - l[0])])


def square_matrix(q, n):
    """2x3 affine: pixel of the n x n square -> pixel of the image."""
    cx, cy, side, a = q
    e = side / n
    c, s = math.cos(a) * e, math.sin(a) * e
    return np.array([[c, -s, cx - c * n / 2 + s * n / 2], [s, c, cy - s * n / 2 - c * n / 2]], np.float32)


def background_filter(img):
    """Approximate background: edge-preserving smoothing + more vivid colours (close to the web app's filter)."""
    f = cv2.bilateralFilter(img, 9, 60, 9)
    f = cv2.bilateralFilter(f, 9, 60, 9)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def edge_alpha(n, radius, feather):
    """Opacity of the pasted square: 1 in the centre, wide fade to the edge (shape between circle and square)."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    u, v = (x + 0.5) / n - 0.5, (y + 0.5) / n - 0.5
    d = (np.abs(u) ** 4 + np.abs(v) ** 4) ** 0.25
    a = np.clip((radius - d) / feather, 0, 1)
    return a * a * (3 - 2 * a)


def main():
    args = sys.argv[1:]
    opt = {"--smooth": 0.5, "--temporal": 0.0, "--radius": 0.47, "--max": 100000, "--feather": 0.12, "--zones": 1}
    while len(args) > 2 and args[-2] in opt:
        opt[args[-2]] = float(args[-1])
        args = args[:-2]
    video, output, variants = args[0], args[1], args[2:]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    models = []
    for v in variants:
        name, weights, n = v.split(":")
        size = name.split("+")[0]
        g = create(size, full_res_skip=False) if name.startswith("l") else create(size)
        g.load_state_dict(convert_legacy_state_dict(torch.load(ROOT / weights, map_location="cpu")))
        models.append((name, g.eval().to(dev), int(n)))
    det = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=4, min_face_detection_confidence=0.5, min_tracking_confidence=0.5))
    seg = mp.tasks.vision.ImageSegmenter.create_from_options(mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODELS / "selfie_multiclass_256x256.tflite")),
        output_category_mask=True)) if opt["--zones"] else None
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    H, W = int(cap.get(4)), int(cap.get(3))
    writer = cv2.VideoWriter(output + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * (1 + len(models)), H))
    tracks = []      # [smoothed square, {variant: previous drawing}]
    k = 0
    while k < opt["--max"]:
        ok, img = cap.read()
        if not ok:
            break
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img[..., ::-1])), int(k * 1000 / fps))
        new_tracks = []
        for face in res.face_landmarks:
            q = head_square(np.array([[p.x * W, p.y * H] for p in face]))
            prev = next((s for s in tracks if np.hypot(*(s[0][:2] - q[:2])) < 0.4 * q[2]), None)
            if prev is not None:
                # adaptive smoothing: strong when the head barely moves, weak when it moves fast (no trailing)
                v = np.hypot(*(prev[0][:2] - q[:2])) / q[2]
                a = min(1.0, (1 - opt["--smooth"]) + v * 12)
                q = prev[0] + a * (q - prev[0])
                new_tracks.append([q, prev[1]])
            else:
                new_tracks.append([q, {}])
        tracks = new_tracks
        background = background_filter(img)
        person = None
        if seg is not None and tracks:
            # person (hair, skin, clothes, held objects): only this is pasted, not the background drawn by the model
            cat = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img[..., ::-1]))).category_mask.numpy_view()
            person = cv2.resize((cat > 0).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            person = cv2.GaussianBlur(cv2.dilate(person, np.ones((5, 5), np.uint8)), (0, 0), 3)
        views = [img]
        for name, g, n in models:
            view = background.copy()
            for q, mem in tracks:
                M = square_matrix(q, n)
                crop = cv2.warpAffine(img, M, (n, n), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                x = torch.from_numpy(crop[..., ::-1].copy()).permute(2, 0, 1).float()[None].to(dev) / 127.5 - 1
                with torch.no_grad():
                    y = g(x)[0].clamp(-1, 1).add(1).mul(127.5).permute(1, 2, 0).cpu().numpy()[..., ::-1]
                if opt["--temporal"] > 0 and name in mem:
                    y = opt["--temporal"] * mem[name] + (1 - opt["--temporal"]) * y
                mem[name] = y
                a = edge_alpha(n, opt["--radius"], opt["--feather"])
                rgba = np.dstack([y * a[..., None], a]).astype(np.float32)
                w = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR)
                if person is not None:
                    w = w * person[..., None]
                view = (w[..., :3] + view * (1 - w[..., 3:4])).clip(0, 255).astype(np.uint8)
            views.append(view)
        writer.write(np.hstack(views))
        k += 1
        if k % 60 == 0:
            print(k, flush=True)
    writer.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", output + ".tmp.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", output], check=True)
    Path(output + ".tmp.mp4").unlink()
    print(output)


if __name__ == "__main__":
    main()
