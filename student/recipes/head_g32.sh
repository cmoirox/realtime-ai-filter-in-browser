#!/bin/bash
# Real-time whole-head model (released as weights/head_g32.pt / head_g32_256.tflite): g32 at 256 x 256 on the faces set,
# open mouths / beards / bald heads / darker skin / glasses / profiles sampled more often (data/faces/sample_weights.csv,
# written by teacher/sample_weights.py), extra L1 on the mouth area. Resumable: run the same command again.
# The released weights were trained as 60 epochs, then resumed for 20 more after profile views were added to the
# faces set; with the full set already in place, one 80-epoch run is the equivalent.
#   mkdir -p runs && nohup caffeinate -i student/recipes/head_g32.sh > runs/head_g32.log 2>&1 &
cd "$(dirname "$0")/../.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run --python 3.11 --with torch --with torchvision --with pillow --with numpy python -u student/train.py \
   --size g32 --dataset faces --name head_g32 --n 256 --cache 384 --batch 16 --min-crop 344 --w-l1 1.0 --w-gan 0.5 \
   --sample-weights data/faces/sample_weights.csv --sample-weights-cap 2 --w-mouth 2 --epochs 80 \
   && uv run --python 3.11 --with litert-torch --with pillow python student/export_tflite.py \
      --size g32 --weights runs/head_g32/ema_last.pt --n 256 --output exports/head_g32_256.tflite
