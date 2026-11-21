"""Trains a student model to imitate the teacher (photo -> styled pairs).

Losses: L1 + perceptual (VGG16) + adversarial (conditional PatchGAN, hinge) + video consistency
(the output for a frame shifted by a few pixels and re-noised must be the original output, shifted).
Data: data/{scenes,faces}/{photo,styled} + data/splits, cached at 384 x 384 in data/cache/.
Outputs in runs/<name>/: last.pt (resume), ema_last.pt, best.pt, a contact sheet per epoch, log.csv, config.json.

    PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i uv run --python 3.11 --with torch --with torchvision --with pillow --with numpy \
        python student/train.py --size m --epochs 60
"""
import argparse
import copy
import csv
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import create  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
C = 384          # size of the cached images
N = 256          # training size (the phone's input size)
MARGIN = 8       # maximum shift for video consistency


# ---------- data ----------

def _load(arg: tuple) -> np.ndarray:
    folder, name, c = arg   # size passed explicitly: worker processes do not see --cache
    return np.stack([np.asarray(Image.open(Path(folder) / d / f"{name}.png").convert("RGB").resize((c, c), Image.LANCZOS)) for d in ("photo", "styled")])


def load_split(split: str, dataset: str = "scenes") -> torch.Tensor:
    """uint8 tensor [n, 2, C, C, 3] (photo, styled) for a split (train, val, test).
    dataset = "scenes" (whole scenes) or "faces" (cropped, aligned heads)."""
    f = CACHE / f"{dataset}_{split}_{C}.npy"
    if not f.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        names = (DATA / "splits" / f"{dataset}_{split}.txt").read_text().split()
        with Pool() as p:
            np.save(f, np.stack(p.map(_load, [(str(DATA / dataset), n, C) for n in names], chunksize=16)))
    return torch.from_numpy(np.load(f))


def augmented_batch(data: torch.Tensor, idx, dev, min_crop: int = 176) -> tuple:
    """Returns (photo, styled) in [-1, 1], shape [b, 3, N + 2*MARGIN, N + 2*MARGIN]: random crop and mirror,
    plus a soft mouth mask (same shape, 1 channel)."""
    g = N + 2 * MARGIN
    xs, ys, ms = [], [], []
    grid = torch.arange(g, device=dev).float()
    for i in idx:
        pair = data[i].to(dev).permute(0, 3, 1, 2).float()          # [2, 3, C, C]
        s = random.randint(min_crop, C)                              # crop: from the whole scene to a close-up
        x0, y0 = random.randint(0, C - s), random.randint(0, C - s)
        pair = pair[:, :, y0:y0 + s, x0:x0 + s]
        pair = F.interpolate(pair, size=(g, g), mode="bilinear", antialias=True, align_corners=False)
        if random.random() < 0.5:
            pair = pair.flip(-1)
        xs.append(pair[0])
        ys.append(pair[1])
        # mouth area of an aligned face: centred at (0.5, 0.70) of the square, symmetric so mirroring does not matter
        cx, cy, sig = (0.5 * C - x0) / s * g, (0.70 * C - y0) / s * g, 0.10 * C / s * g
        ms.append(torch.exp(-((grid[None, :] - cx) ** 2 + (grid[:, None] - cy) ** 2) / (2 * sig * sig))[None])
    return torch.stack(xs) / 127.5 - 1, torch.stack(ys) / 127.5 - 1, torch.stack(ms)


def camera(x: torch.Tensor) -> torch.Tensor:
    """Mimics phone-camera variation: exposure, contrast, noise. Applied to the photo only."""
    b = x.shape[0]
    gain = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
    offset = torch.empty(b, 1, 1, 1, device=x.device).uniform_(-0.08, 0.08)
    noise = torch.randn_like(x) * torch.empty(b, 1, 1, 1, device=x.device).uniform_(0, 0.04)
    return ((x + 1) * gain - 1 + offset + noise).clamp(-1, 1)


# ---------- losses ----------

class Perceptual(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16
        v = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        self.slices = nn.ModuleList([v[:4], v[4:9], v[9:16], v[16:23]])   # relu1_2, relu2_2, relu3_3, relu4_3
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, a, b):
        a, b = ((a + 1) / 2 - self.mean) / self.std, ((b + 1) / 2 - self.mean) / self.std
        loss = 0.0
        for t in self.slices:
            a, b = t(a), t(b)
            loss = loss + F.l1_loss(a, b)
        return loss


def discriminator() -> nn.Sequential:
    """Conditional PatchGAN: sees the photo and the styled image stacked (6 channels)."""
    sn = nn.utils.spectral_norm
    return nn.Sequential(
        sn(nn.Conv2d(6, 64, 4, 2, 1)), nn.LeakyReLU(0.2, True),
        sn(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2, True),
        sn(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2, True),
        sn(nn.Conv2d(256, 256, 4, 1, 1)), nn.LeakyReLU(0.2, True),
        nn.Conv2d(256, 1, 4, 1, 1),
    )


# ---------- contact sheet ----------

def contact_sheet(g: nn.Module, data: torch.Tensor, dev, path: Path, n: int = 8) -> None:
    g.eval()
    rows = []
    with torch.no_grad():
        for i in range(n):
            pair = F.interpolate(data[i].to(dev).permute(0, 3, 1, 2).float(), size=(N, N), mode="bilinear", antialias=True) / 127.5 - 1
            out = g(pair[:1])
            rows.append(torch.cat([pair[0], pair[1], out[0]], dim=2))
    im = torch.cat(rows, dim=1).clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(im).save(path, quality=90)


def validate(g: nn.Module, data: torch.Tensor, dev, vgg) -> tuple:
    g.eval()
    l1 = perc = 0.0
    with torch.no_grad():
        for i in range(0, len(data), 32):
            pair = data[i:i + 32].to(dev).permute(0, 1, 4, 2, 3).float()
            x = F.interpolate(pair[:, 0], size=(N, N), mode="bilinear", antialias=True) / 127.5 - 1
            y = F.interpolate(pair[:, 1], size=(N, N), mode="bilinear", antialias=True) / 127.5 - 1
            out = g(x)
            l1 += F.l1_loss(out, y).item() * len(x)
            perc += vgg(out, y).item() * len(x)
    return l1 / len(data), perc / len(data)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------- training ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="m")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--w-gan", type=float, default=0.1)
    ap.add_argument("--w-perc", type=float, default=1.0)
    ap.add_argument("--w-l1", type=float, default=2.0)
    ap.add_argument("--w-video", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=3, help="epochs without the adversarial loss")
    ap.add_argument("--dataset", default="scenes", choices=["scenes", "faces"])
    ap.add_argument("--name", help="output folder in runs/ (default: the size)")
    ap.add_argument("--no-full-res-skip", action="store_true", help="no full-resolution skip connection")
    ap.add_argument("--min-crop", type=int, default=176, help="smallest random crop (out of --cache)")
    ap.add_argument("--w-mouth", type=float, default=0.0, help="weight of an extra L1 term on the mouth area (faces dataset)")
    ap.add_argument("--sample-weights", help="CSV (name, value): shows high-value examples more often (e.g. open mouths)")
    ap.add_argument("--sample-weights-cap", type=float, default=1.0, help="maximum value counted, in multiples of 0.4 (2 = up to 11x more often)")
    ap.add_argument("--n", type=int, default=256, help="training size (256 for real-time models, 384 for the large face model)")
    ap.add_argument("--cache", type=int, default=384, help="size of the images kept in memory (512 for the large model)")
    ap.add_argument("--iterations", type=int, help="stop after N iterations (speed test)")
    args = ap.parse_args()
    global N, C
    N, C = args.n, args.cache

    dev = pick_device()
    torch.manual_seed(0)
    random.seed(0)
    folder = ROOT / "runs" / (args.name or args.size)
    folder.mkdir(parents=True, exist_ok=True)
    train, val = load_split("train", args.dataset), load_split("val", args.dataset)
    (folder / "config.json").write_text(json.dumps(vars(args), indent=1))
    print(f"{len(train)} training pairs, {len(val)} validation pairs", flush=True)

    g, d, vgg = create(args.size, not args.no_full_res_skip).to(dev), discriminator().to(dev), Perceptual().to(dev)
    ema = copy.deepcopy(g)
    og = torch.optim.Adam(g.parameters(), args.lr, betas=(0.5, 0.999))
    od = torch.optim.Adam(d.parameters(), args.lr, betas=(0.5, 0.999))
    per_epoch = len(train) // args.batch
    sampling = None
    if args.sample_weights:   # weighted sampling: a wide-open mouth (0.4 and above) is seen 6x more often than a closed one
        names = (DATA / "splits" / f"{args.dataset}_train.txt").read_text().split()
        value = {r["name"]: float(r["value"]) for r in csv.DictReader(open(args.sample_weights))}
        sampling = torch.tensor([1 + 5 * min(args.sample_weights_cap, value.get(n, 0) / 0.4) for n in names])
    total = args.epochs * per_epoch
    lr_at = lambda it: 0.5 * (1 + np.cos(np.pi * it / total)) * 0.95 + 0.05   # cosine decay
    start, it, best = 0, 0, 1e9
    resume = folder / "last.pt"
    if resume.exists() and not args.iterations:
        e = torch.load(resume, map_location=dev, weights_only=False)  # our own file: also holds numpy scalars
        g.load_state_dict(e["g"]); d.load_state_dict(e["d"]); ema.load_state_dict(e["ema"])
        og.load_state_dict(e["og"]); od.load_state_dict(e["od"])
        start, it, best = e["epoch"], e["it"], e["best"]
        print(f"resuming at epoch {start + 1}", flush=True)
    log = folder / "log.csv"
    if not log.exists():
        log.write_text("epoch,l1,perceptual,video,gan_g,gan_d,val_l1,val_perceptual,minutes\n")

    t0 = time.time()
    for epoch in range(start, args.epochs):
        g.train()
        order = (torch.multinomial(sampling, len(train), replacement=True) if sampling is not None else torch.randperm(len(train))).tolist()
        running = np.zeros(5)
        gan_on = epoch >= args.warmup
        for k in range(per_epoch):
            for o in (og, od):
                o.param_groups[0]["lr"] = args.lr * lr_at(it)
            gx, gy, gm = augmented_batch(train, order[k * args.batch:(k + 1) * args.batch], dev, args.min_crop)
            dx, dy = random.randint(-MARGIN, MARGIN), random.randint(-MARGIN, MARGIN)
            crop = lambda t, ox, oy: t[..., MARGIN + oy:MARGIN + oy + N, MARGIN + ox:MARGIN + ox + N]
            x1, y1 = camera(crop(gx, 0, 0)), crop(gy, 0, 0)
            x2 = camera(crop(gx, dx, dy))                        # same scene, shifted and noised differently
            out1, out2 = g(x1), g(x2)

            if gan_on:                                           # discriminator learns real vs fake
                real, fake = d(torch.cat([x1, y1], 1)), d(torch.cat([x1, out1.detach()], 1))
                loss_d = F.relu(1 - real).mean() + F.relu(1 + fake).mean()
                od.zero_grad(set_to_none=True)
                loss_d.backward()
                od.step()
            else:
                loss_d = torch.zeros((), device=dev)

            l1 = F.l1_loss(out1, y1)
            if args.w_mouth:                                     # extra error counted on the mouth area (aligned faces)
                m1 = crop(gm, 0, 0)
                l1 = l1 + args.w_mouth * ((out1 - y1).abs() * m1).sum() / (3 * m1.sum())
            perc = vgg(out1, y1)
            # area shared by both views: out2[i, j] must equal out1[i + dy, j + dx]
            ya, yb, xa, xb = max(0, -dy), N - max(0, dy), max(0, -dx), N - max(0, dx)
            video = F.l1_loss(out2[..., ya:yb, xa:xb], out1[..., ya + dy:yb + dy, xa + dx:xb + dx])
            gan_g = -d(torch.cat([x1, out1], 1)).mean() if gan_on else torch.zeros((), device=dev)
            loss_g = args.w_l1 * l1 + args.w_perc * perc + args.w_video * video + args.w_gan * gan_g
            og.zero_grad(set_to_none=True)
            loss_g.backward()
            og.step()

            with torch.no_grad():                                # exponential moving average of the weights (more stable)
                for pe, pg in zip(ema.parameters(), g.parameters()):
                    pe.lerp_(pg, 0.002)
                for be, bg in zip(ema.buffers(), g.buffers()):
                    be.copy_(bg)
            running += [l1.item(), perc.item(), video.item(), gan_g.item(), loss_d.item()]
            it += 1
            if args.iterations and it >= args.iterations:
                print(f"{it} iterations in {time.time() - t0:.0f} s, i.e. {(time.time() - t0) / it:.3f} s per iteration; "
                      f"one epoch = {per_epoch} iterations", flush=True)
                return

        running /= per_epoch
        vl1, vperc = validate(ema, val, dev, vgg)
        minutes = (time.time() - t0) / 60
        with log.open("a") as f:
            csv.writer(f).writerow([epoch + 1, *[f"{v:.4f}" for v in running], f"{vl1:.4f}", f"{vperc:.4f}", f"{minutes:.1f}"])
        print(f"epoch {epoch + 1}/{args.epochs}  l1 {running[0]:.4f}  perc {running[1]:.3f}  video {running[2]:.4f}  "
              f"gan {running[3]:.3f}/{running[4]:.3f}  | val l1 {vl1:.4f} perc {vperc:.3f}  ({minutes:.0f} min)", flush=True)
        contact_sheet(ema, val, dev, folder / f"epoch_{epoch + 1:03d}.jpg")
        if vperc < best:
            best = vperc
            torch.save(ema.state_dict(), folder / "best.pt")
        torch.save(ema.state_dict(), folder / "ema_last.pt")
        torch.save({"g": g.state_dict(), "d": d.state_dict(), "ema": ema.state_dict(), "og": og.state_dict(), "od": od.state_dict(),
                    "epoch": epoch + 1, "it": it, "best": best}, resume)
    print("done", flush=True)


if __name__ == "__main__":
    main()
