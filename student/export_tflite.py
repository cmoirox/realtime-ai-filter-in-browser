"""Exports a student model to .tflite (NHWC input and output [1, N, N, 3], values in [-1, 1]) and checks that the
exported file produces the same image as PyTorch.

    uv run --python 3.11 --with litert-torch --with pillow python student/export_tflite.py --size m [--weights x.pt] [--n 256]

Checkpoints saved with the original module names are converted on the fly (see model.convert_legacy_state_dict).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import convert_legacy_state_dict, create, fuse_batchnorm, is_legacy_state_dict  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class ToNHWC(nn.Module):
    """Wraps the model so input and output are NHWC, as the web page expects."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="m")
    ap.add_argument("--weights")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--output")
    ap.add_argument("--no-full-res-skip", action="store_true")
    args = ap.parse_args()

    import litert_torch
    from ai_edge_litert.interpreter import Interpreter

    m = create(args.size, not args.no_full_res_skip)
    if args.weights:
        sd = torch.load(args.weights, map_location="cpu")
        if is_legacy_state_dict(sd):
            print("legacy checkpoint: renaming keys")
            sd = convert_legacy_state_dict(sd)
        m.load_state_dict(sd)
    else:  # random weights: give BatchNorm non-trivial statistics so the fusion is actually tested
        m.train()
        with torch.no_grad():
            for _ in range(5):
                m(torch.randn(8, 3, 64, 64))
    m.eval()
    wrapped = ToNHWC(fuse_batchnorm(m)).eval()
    x = torch.rand(1, args.n, args.n, 3) * 2 - 1
    with torch.no_grad():
        ref = ToNHWC(m)(x).numpy()
    output = Path(args.output or ROOT / "exports" / f"student_{args.size}_{args.n}.tflite")
    output.parent.mkdir(parents=True, exist_ok=True)
    litert_torch.convert(wrapped, (x,)).export(str(output))

    it = Interpreter(model_path=str(output))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], x.numpy())
    it.invoke()
    out = it.get_tensor(o["index"])
    diff = np.abs(out - ref)
    print(f"{output.name}: {output.stat().st_size / 1e6:.2f} MB, input {list(i['shape'])}, output {list(o['shape'])}")
    print(f"difference from PyTorch: mean {diff.mean():.2e}, max {diff.max():.2e} (on a range of 2)")
    ops = sorted({d["op_name"] for d in it._get_ops_details()})
    print("ops:", ", ".join(ops))
    if diff.max() > 1e-3:
        sys.exit("FAILED: the exported file does not reproduce PyTorch")


if __name__ == "__main__":
    main()
