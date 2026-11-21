"""Student model: a small fully-convolutional generator, photo -> styled image.

Deliberately plain operations so it runs in the browser (LiteRT.js WebGPU): 3x3 convolutions, ReLU,
nearest-neighbour x2 upsampling, additions. BatchNorm only exists during training: at export time it is
folded into the convolutions.
Input and output: RGB image in [-1, 1], shape [1, 3, N, N] on the PyTorch side.

    uv run --python 3.11 --with torch python student/model.py    # prints parameter counts
"""
import torch
from torch import nn

SMALL_SIZES = {  # name: (base width, number of residual blocks)
    "s": (24, 4),
    "m": (32, 5),
    "l": (48, 6),
}


def conv(cin: int, cout: int, stride: int = 1, relu: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.a = conv(c, c)
        self.b = conv(c, c, relu=False)

    def forward(self, x):
        return x + self.b(self.a(x))


class Student(nn.Module):
    def __init__(self, width: int = 32, blocks: int = 5, full_res_skip: bool = True):
        super().__init__()
        c = width
        self.full_res_skip = full_res_skip         # False: no full-resolution skip (copies less of the photo through)
        self.stem = conv(3, c // 2)                 # full resolution
        self.down1 = conv(c // 2, c, stride=2)      # 1/2
        self.down2 = conv(c, 2 * c, stride=2)       # 1/4
        self.blocks = nn.Sequential(*[ResBlock(2 * c) for _ in range(blocks)])
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.up1 = conv(2 * c, c)                   # 1/2
        self.up2 = conv(c, c // 2)                  # full resolution
        self.head = nn.Conv2d(c // 2, 3, 3, 1, 1)

    def forward(self, x):
        e0 = self.stem(x)
        e1 = self.down1(e0)
        y = self.blocks(self.down2(e1))
        y = self.up1(self.upsample(y)) + e1         # skips keep fine details
        y = self.up2(self.upsample(y))
        if self.full_res_skip:
            y = y + e0
        return torch.tanh(self.head(y))


class Large(nn.Module):
    """Larger variant: same operations as Student, but one level deeper (works at 1/8 resolution), wider,
    more blocks, and no full-resolution skip. The narrow configs (g24-g48) are fast enough to run every frame."""

    def __init__(self, width: int = 64, blocks: int = 8):
        super().__init__()
        c = width
        self.stem = conv(3, c // 2)                 # full resolution
        self.down1 = conv(c // 2, c, stride=2)      # 1/2
        self.down2 = conv(c, 2 * c, stride=2)       # 1/4
        self.down3 = conv(2 * c, 4 * c, stride=2)   # 1/8
        self.blocks = nn.Sequential(*[ResBlock(4 * c) for _ in range(blocks)])
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.up1 = conv(4 * c, 2 * c)               # 1/4
        self.up2 = conv(2 * c, c)                   # 1/2
        self.up3 = conv(c, c // 2)                  # full resolution
        self.refine = conv(c // 2, c // 2)
        self.head = nn.Conv2d(c // 2, 3, 3, 1, 1)

    def forward(self, x):
        e1 = self.down1(self.stem(x))
        e2 = self.down2(e1)
        y = self.blocks(self.down3(e2))
        y = self.up1(self.upsample(y)) + e2
        y = self.up2(self.upsample(y)) + e1
        y = self.refine(self.up3(self.upsample(y)))
        return torch.tanh(self.head(y))


# g24 to g48: same shape as xl but narrower, cheap enough to redraw every frame on a phone
LARGE_SIZES = {"xl": (64, 8), "g24": (24, 6), "g32": (32, 6), "g40": (40, 6), "g48": (48, 8)}


def create(size: str, full_res_skip: bool = True) -> nn.Module:
    if size in LARGE_SIZES:
        return Large(*LARGE_SIZES[size])
    width, blocks = SMALL_SIZES[size]
    return Student(width, blocks, full_res_skip)


# Top-level module names of the original (French) checkpoints -> current names.
_LEGACY_PREFIXES = {
    "entree": "stem", "bas1": "down1", "bas2": "down2", "bas3": "down3", "blocs": "blocks",
    "haut1": "up1", "haut2": "up2", "haut3": "up3", "fin": "refine", "sortie": "head",
}


def is_legacy_state_dict(sd: dict) -> bool:
    return any(k.split(".", 1)[0] in _LEGACY_PREFIXES for k in sd)


def convert_legacy_state_dict(sd: dict) -> dict:
    """Renames the keys of a checkpoint saved with the original module names; current keys pass through unchanged."""
    out = {}
    for k, v in sd.items():
        first, _, rest = k.partition(".")
        out[f"{_LEGACY_PREFIXES.get(first, first)}.{rest}" if rest else _LEGACY_PREFIXES.get(first, first)] = v
    return out


def fuse_batchnorm(model: nn.Module) -> nn.Module:
    """Copy of the model in eval mode with BatchNorm folded into the convolutions (for export)."""
    import copy
    m = copy.deepcopy(model).eval()
    for seq in [x for x in m.modules() if isinstance(x, nn.Sequential)]:
        if len(seq) >= 2 and isinstance(seq[0], nn.Conv2d) and isinstance(seq[1], nn.BatchNorm2d):
            seq[0] = torch.nn.utils.fusion.fuse_conv_bn_eval(seq[0], seq[1])
            seq[1] = nn.Identity()
    return m


if __name__ == "__main__":
    for s in [*SMALL_SIZES, *LARGE_SIZES]:
        m = create(s)
        n = sum(p.numel() for p in m.parameters())
        print(f"{s}: {n / 1e6:.2f} M parameters, {n * 4 / 1e6:.1f} MB ({n * 2 / 1e6:.1f} in half precision)")
