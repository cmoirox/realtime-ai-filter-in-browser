#!/bin/bash
# Zones model (released as weights/zones_m.pt / zones_m_160.tflite): size m at 160 x 160, two stages.
#   1. 40 epochs from scratch on the MediaPipe labels (runs/zones_m).
#   2. 30 epochs starting from stage 1's best weights, with hands pasted over half of the faces (runs/zones_m_paste).
# The released weights are stage 2's last.pt. Re-running skips finished stages.
#   mkdir -p runs && nohup caffeinate -i student/recipes/zones_m.sh > runs/zones_m.log 2>&1 &
cd "$(dirname "$0")/../.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
Z="uv run --python 3.11 --with torch --with pillow --with numpy python -u student/train_zones.py"
[ -f runs/zones_m/done ] || $Z --size m --name zones_m --epochs 40 || exit 1
[ -f runs/zones_m_paste/done ] || $Z --size m --name zones_m_paste --init runs/zones_m/best.pt --paste 0.5 --epochs 30 || exit 1
uv run --python 3.11 --with litert-torch --with pillow python student/export_tflite.py \
   --size m --weights runs/zones_m_paste/last.pt --n 160 --output exports/zones_m_160.tflite
