"""Small zones model (replaces MediaPipe segmentation on the phone, where it is unreliable and too slow).

Teacher: MediaPipe "selfie multiclass" labels in data/zones/{scenes,faces}. Student: the same small network as the
others (student/model.py), RGB input 160 x 160 in [-1, 1], 3-channel output in [-1, 1]:
  red   = what can be in front of a face (body skin: hands, arms; clothes; held objects)
  green = hair
  blue  = skin (face and body)
This is exactly the "zones" texture the web app consumes.

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --python 3.11 --with torch --with pillow --with numpy python -u student/train_zones.py [--size m] [--epochs 40]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import convert_legacy_state_dict, create  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
C = 192  # size kept in memory
N = 160  # size seen by the model
PASTE = 0.0  # fraction of images that get a hand pasted over the face (--paste)


def load():
    cache = DATA / "cache" / f"zones_{C}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["y"]
    xs, ys = [], []
    for dataset in ("scenes", "faces"):
        for f in sorted((DATA / "zones" / dataset).glob("*.png")):
            p = DATA / dataset / "photo" / f.name
            if not p.exists():
                continue
            xs.append(np.asarray(Image.open(p).convert("RGB").resize((C, C), Image.BILINEAR)))
            m = np.asarray(Image.open(f).resize((C, C), Image.NEAREST))
            # MediaPipe classes: 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 accessories
            ys.append(np.stack([np.isin(m, (2, 4, 5)), m == 1, np.isin(m, (2, 3))], -1).astype(np.uint8))
    x, y = np.stack(xs), np.stack(ys)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y=y)
    return x, y


def batch(x, y, idx, device, train=True):
    xb = torch.from_numpy(x[idx]).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1
    yb = torch.from_numpy(y[idx]).to(device).permute(0, 3, 1, 2).float()
    if not train:
        return F.interpolate(xb, size=(N, N), mode="bilinear", antialias=True), F.interpolate(yb, size=(N, N), mode="bilinear")
    # random crop (whole image to close-up), mirror, exposure, tint, noise
    b = xb.shape[0]
    xs, ys = [], []
    for k in range(b):
        t = random.randint(int(C * 0.55), C)
        i, j = random.randint(0, C - t), random.randint(0, C - t)
        xi, yi = xb[k:k + 1, :, i:i + t, j:j + t], yb[k:k + 1, :, i:i + t, j:j + t]
        if random.random() < 0.5:
            xi, yi = xi.flip(3), yi.flip(3)
        xs.append(F.interpolate(xi, size=(N, N), mode="bilinear", antialias=True))
        ys.append(F.interpolate(yi, size=(N, N), mode="bilinear"))
    xb, yb = torch.cat(xs), torch.cat(ys)
    # paste: body skin (hands, arms) from another image of the batch is placed over this image's face.
    # Photos with a hand in front of the face are rare; without this the model never sees a hand on skin.
    if PASTE:
        for k in range(b):
            if random.random() > PASTE:
                continue
            d = (k + random.randint(1, b - 1)) % b
            m = (yb[d, 0] > 0.5) & (yb[d, 2] > 0.5)                       # donor's body skin
            target = (yb[k, 2] > 0.5) & (yb[k, 0] < 0.5)                  # receiver's face skin
            if m.sum() < 0.01 * N * N or target.sum() < 0.02 * N * N:
                continue
            cm = torch.nonzero(m).float().mean(0)
            cc = torch.nonzero(target).float().mean(0)
            spread = target.sum().float().sqrt() * 0.6
            dy = int(cc[0] - cm[0] + random.uniform(-1, 1) * spread)
            dx = int(cc[1] - cm[1] + random.uniform(-1, 1) * spread)
            mf = torch.roll(m, (dy, dx), (0, 1)).float()[None]
            xd = torch.roll(xb[d], (dy, dx), (1, 2))
            xb[k] = xb[k] * (1 - mf) + xd * mf
            yb[k] = torch.stack([torch.maximum(yb[k, 0], mf[0]), yb[k, 1] * (1 - mf[0]), torch.maximum(yb[k, 2], mf[0])])
    gain = torch.empty(b, 1, 1, 1, device=device).uniform_(0.6, 1.4)
    tint = torch.empty(b, 3, 1, 1, device=device).uniform_(0.88, 1.12)
    xb = ((xb + 1) * gain * tint - 1 + torch.randn_like(xb) * 0.03).clamp(-1, 1)
    return xb, yb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="m")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", help="output folder in runs/ (default: zones_<size>)")
    ap.add_argument("--init", help="initial weights")
    ap.add_argument("--paste", type=float, default=0.0, help="fraction of images with a hand pasted over the face")
    args = ap.parse_args()
    global PASTE
    PASTE = args.paste
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    x, y = load()
    n = len(x)
    order = np.random.RandomState(0).permutation(n)
    val, train = order[:300], order[300:]
    print(f"{len(train)} training images, {len(val)} validation images", flush=True)
    folder = ROOT / "runs" / (args.name or f"zones_{args.size}")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({"size": args.size, "n": N, "epochs": args.epochs}))
    g = create(args.size).to(device)
    if args.init:
        g.load_state_dict(convert_legacy_state_dict(torch.load(args.init, map_location=device)))
    opt = torch.optim.AdamW(g.parameters(), lr=2e-3, weight_decay=1e-4)
    total_steps = args.epochs * (len(train) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=total_steps)
    start = time.time()
    best = 0.0
    for ep in range(1, args.epochs + 1):
        g.train()
        np.random.shuffle(train)
        total = 0.0
        for k in range(0, len(train) - args.batch + 1, args.batch):
            xb, yb = batch(x, y, np.sort(train[k:k + args.batch]), device)
            s = g(xb)
            p = (s + 1) / 2
            # per-pixel error + Dice overlap (hands are small: without it the model ignores them)
            loss = F.binary_cross_entropy(p.clamp(1e-4, 1 - 1e-4), yb)
            inter = (p * yb).sum((2, 3))
            loss = loss + (1 - (2 * inter + 1) / (p.sum((2, 3)) + yb.sum((2, 3)) + 1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        g.eval()
        inter = np.zeros(3)
        union = np.zeros(3)
        with torch.no_grad():
            for k in range(0, len(val), 50):
                xb, yb = batch(x, y, np.sort(val[k:k + 50]), device, False)
                p = (g(xb) > 0).float()
                t = (yb > 0.5).float()
                inter += (p * t).sum((0, 2, 3)).cpu().numpy()
                union += ((p + t) > 0).float().sum((0, 2, 3)).cpu().numpy()
        iou = inter / np.maximum(union, 1)
        print(f"epoch {ep}/{args.epochs}  loss {total / (len(train) // args.batch):.4f}  IoU front {iou[0]:.3f} hair {iou[1]:.3f} skin {iou[2]:.3f}  ({(time.time() - start) / 60:.0f} min)", flush=True)
        torch.save(g.state_dict(), folder / "last.pt")
        if iou.mean() > best:
            best = iou.mean()
            torch.save(g.state_dict(), folder / "best.pt")
    (folder / "done").touch()
    print("done", flush=True)


if __name__ == "__main__":
    main()
