# Train your own style

This pipeline works with any look the teacher can produce: claymation, anime, comic ink, oil paint, a different game look. You change one prompt, regenerate the pairs and retrain. Exact commands for every step are in [`teacher/README.md`](../teacher/README.md) and [`student/recipes/`](../student/recipes).

## What you need

- A machine with a decent GPU. Everything here was run on an Apple M5 Pro laptop with 24 GB of RAM: the teacher runs with mflux (MLX), training with PyTorch on MPS. The training scripts also run on CUDA. On non-Apple machines, run the teacher with `diffusers` instead of mflux. Only `teacher/restyle_*.py` need adapting.
- About 50 GB of disk for the teacher weights, source photos, pairs and caches.
- [uv](https://docs.astral.sh/uv/). Every script declares its dependencies in its usage line (`uv run --with … python …`), so there's nothing to install globally.

## Steps and timings

| Step | Scripts | Time (M5 Pro) |
|---|---|---|
| 1. Get source photos | `select_coco.py`, `download_wikimedia.py`, `add_fairface.py` | ~1 h of downloads |
| 2. Find and crop faces | `detect_faces.py`, `crop_faces.py` (+ the `*_profiles.py` scripts for side views) | minutes |
| 3. Describe faces | `describe_faces.py` (Qwen2.5-VL 7B) | ~30 faces/min, about 2.5 h for 4,400 faces |
| 4. Restyle with the teacher | `restyle_faces.py` (and `restyle_scenes.py` for whole scenes) | ~10 s per image, ~20 s with a per-face prompt, so 12–24 h for 4,400 faces. Resumable: run it overnight |
| 5. Quality control and splits | `quality_faces.py`, `split_faces.py`, `contact_sheet.py` | minutes, plus a look at the contact sheets |
| 6. Sample weights | `measure_mouths.py`, `sample_weights.py` | minutes |
| 7. Train the student | `student/recipes/head_g32.sh` | ~5 h for 80 epochs |
| 8. Export and check | `student/export_tflite.py` (called by the recipe) | seconds |
| 9. Try it | `tools/video_demo.py` on a clip, then copy the `.tflite` file to `web/public/models/head_g32_256.tflite` | – |

The zones model doesn't depend on the style, so you can keep the released `zones_m_160.tflite`.

## Writing the prompt

Edit `teacher/prompts.py`. What mattered most:

1. **Ask for structural fidelity explicitly**, in the same sentence as the style: same framing, head angle, gaze, expression, age, hair, facial hair, skin tone, clothes, background layout, and "do not add or remove anything". The student learns a pixel-aligned mapping, so anything the teacher moves turns into blur.
2. **Keep the per-face description** (`face_prompt(description)`). Without it, most diffusion editors pull everyone towards one default face.
3. **Try 20 images first.** Run `restyle_faces.py --trial <out_dir> <name> <name> …` on 20 representative photos, look at them side by side, adjust, and only then launch the full run. Check the hard cases: dark skin, grey hair, beards, glasses, profiles, open mouths.
4. **Watch for words the model overreacts to.** Here "olive skin" came out khaki and "pale skin" too pale. `describe_for_prompt` maps such words to softer ones.

## Training tips

- Start from the recipe as is: its loss weights and sample weights took several iterations to tune.
- Look at the per-epoch contact sheets (`runs/<name>/epoch_XXX.jpg`: photo | teacher | student) more than at the loss numbers. The adversarial term makes the curves noisy.
- If the look is simpler than this one (flat cartoon), try a smaller model first: `--size m` or `--size l` run 2–3× faster on the phone.
- Keep the input size at 256 and the aligned-square convention. The web demo depends on both.
