# Released weights

These files are attached to the GitHub Release, not committed (`.gitignore` excludes `weights/*.pt` and
`weights/*.tflite`). Download them into this folder.

| File | Size | Model | Params | Input | Output |
|---|---|---|---|---|---|
| `head_g32.pt` | 7.9 MB | `create("g32")` (`Large`, width 32, 6 blocks) | 1,969,987 | `[1, 3, N, N]` | `[1, 3, N, N]` |
| `head_g32_256.tflite` | 7.9 MB | same, BatchNorm folded, fp32 | 1,968,083 | `[1, 256, 256, 3]` | `[1, 256, 256, 3]` |
| `zones_m.pt` | 1.7 MB | `create("m")` (`Student`, width 32, 5 blocks, full-res skip) | 417,187 | `[1, 3, N, N]` | `[1, 3, N, N]` |
| `zones_m_160.tflite` | 1.7 MB | same, BatchNorm folded, fp32 | 416,387 | `[1, 160, 160, 3]` | `[1, 160, 160, 3]` |

All inputs and outputs are RGB in [-1, 1] (`pixel / 127.5 - 1`). The `.pt` files are PyTorch `state_dict`s for
`student/model.py` (any square size divisible by 8 works); the `.tflite` files are NHWC and fixed-size, as used by the
web app (LiteRT.js, WebGPU). Ops: `CONV_2D`, `ADD`, `PAD`, `RESIZE_NEAREST_NEIGHBOR`, `TANH`.

- **head_g32**: whole-head stylizer. Input is the aligned square of a head (see `tools/video_demo.py` for the
  crop convention); output is the styled head. Recipe: `student/recipes/head_g32.sh`.
- **zones_m**: segmentation of the same kind of crop or a whole frame. Output channel > 0 means "inside":
  - red: occluders that can be in front of a face (body skin such as hands and arms, clothes, held objects)
  - green: hair
  - blue: skin (face and body)

  Recipe: `student/recipes/zones_m.sh`.

Load in PyTorch:

```python
import torch
from model import create  # student/model.py
g = create("g32").eval()
g.load_state_dict(torch.load("weights/head_g32.pt", map_location="cpu"))
```

Re-export: `uv run --python 3.11 --with litert-torch --with pillow python student/export_tflite.py --size g32 --weights weights/head_g32.pt --n 256 --output weights/head_g32_256.tflite`
(and `--size m --weights weights/zones_m.pt --n 160` for zones). The script checks the `.tflite` against PyTorch.

## SHA-256

```
d1a15a5e15bfd6c82c01aec557ea67514a8b2c9dab112c3408d5c3b698939b51  head_g32.pt
34c789a654787e5d7286f7f7ec5028855acac337cddf7f10e4b60be24b3b2dec  head_g32_256.tflite
7fe203749a34a05d9c83d7d748beb2b18f921481f05ae871f7e542b8b26dc1db  zones_m.pt
54b79e7c2e1f63c49190a5cdd000f1974b03c446726eccd35322d82ccb333568  zones_m_160.tflite
```
