"""Label the photos with the MediaPipe "selfie multiclass" segmentation model (the teacher of the small zone model):
for each photo, a 256 x 256 image whose pixels are
0 background, 1 hair, 2 body skin (hands, neck, arms), 3 face skin, 4 clothes, 5 other (held objects).

Inputs: data/scenes/photo and data/faces/photo (512 x 512). Outputs: data/zones/{scenes,faces}/<name>.png.
Resumable. Nothing is downloaded (model expected in models/).

    uv run --python 3.11 --with mediapipe --with pillow --with numpy python teacher/label_zones.py
"""
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    opts = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "models" / "selfie_multiclass_256x256.tflite")),
        output_category_mask=True)
    with mp.tasks.vision.ImageSegmenter.create_from_options(opts) as seg:
        for subset in ("scenes", "faces"):
            out = ROOT / "data" / "zones" / subset
            out.mkdir(parents=True, exist_ok=True)
            photos = sorted((ROOT / "data" / subset / "photo").glob("*.png"))
            for k, f in enumerate(photos):
                target = out / f.name
                if target.exists():
                    continue
                im = np.array(Image.open(f).convert("RGB"))
                m = np.squeeze(seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=im)).category_mask.numpy_view())
                Image.fromarray(m.astype(np.uint8)).resize((256, 256), Image.NEAREST).save(target)
                if k % 500 == 0:
                    print(subset, k, len(photos), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
