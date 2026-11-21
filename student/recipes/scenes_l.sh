#!/bin/bash
# Whole-scene model: size l at 256 x 256 on the scenes set, without the full-resolution skip (copies less of the photo
# through). Resumable: run the same command again.
#   mkdir -p runs && nohup caffeinate -i student/recipes/scenes_l.sh > runs/scenes_l.log 2>&1 &
cd "$(dirname "$0")/../.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run --python 3.11 --with torch --with torchvision --with pillow --with numpy python -u student/train.py \
   --size l --dataset scenes --name scenes_l --no-full-res-skip --w-l1 1.0 --w-gan 0.5 --epochs 60 \
   && uv run --python 3.11 --with litert-torch --with pillow python student/export_tflite.py \
      --size l --no-full-res-skip --weights runs/scenes_l/ema_last.pt --n 256 --output exports/scenes_l_256.tflite
