# How it works

The goal: point a phone at people and see them restyled live. The phone's browser has to do all of it: no server, no app, no per-use cost. Diffusion image editors produce the look we want, but they take seconds per image on a laptop GPU. The phone has roughly 50 ms per frame for everything. So the big model teaches a tiny one, offline and once.

## 1. Pick the runtime before training anything

The network architecture depends on what the browser can actually run. A pretrained stylization network (AnimeGANv3, 4.2 MB) was converted to ONNX, TFLite and TF.js and benchmarked on a real iPhone in Safari, including 10-minute endurance runs ([benchmarks.md](benchmarks.md)). LiteRT.js on WebGPU ran 16 fps at 256² and stayed stable for 10 minutes. ONNX Runtime Web (WebGPU and WASM), TF.js (WebGL) and LiteRT.js on WASM all ran at 1–2.5 fps.

Result: `.tflite`, 256 × 256 input, and only "boring" ops (3×3 conv, ReLU, nearest upsample, add, tanh) that every backend handles.

## 2. The teacher: diffusion image editing

[FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) in edit mode (photo + text prompt → edited photo), 4 steps, 512², run locally with mflux on Apple silicon. It takes about 10 s per image, or ~20 s with a per-image prompt. An earlier setup (FLUX.1 Kontext dev + a style LoRA) gave similar looks but was much slower and non-commercial, so it was dropped.

The prompt asks for the style **and** for strict structural fidelity: same framing, head angle, gaze, expression, age, hair, facial hair, skin tone, clothes and background layout. The student can only learn a pixel-aligned mapping if the teacher keeps things where they were.

### Describing each face

With a style-only prompt, the teacher made people look alike. It lightened dark skin, turned white hair olive or blond, and removed moustaches. Fix: a local vision-language model (Qwen2.5-VL 7B, 4-bit, MLX, ~30 faces/min) fills a fixed form for each face:

```
person: man; hair colour: grey; hair: short; skin: dark brown; eyes: brown; moustache: yes (grey); beard: no; glasses: no; headwear: no
```

`teacher/prompts.py` turns this form into a sentence appended to the prompt ("The character must have exactly these colours and features, as in the photo: …"). A few rules came from side-by-side tests:

- Light skin is not named, because the teacher already defaults to it.
- "olive" and "tan" become "lightly tanned warm", because "olive" came out khaki.
- Facial hair is named with "exactly the same facial hair as in the photo, covering the same areas".
- FairFace's own group label acts as a guard rail against faces the describer rated too light.

### Data

- Everyday-scene photos from COCO (selected by object categories: dining, office, street, transport…) and Wikimedia Commons (category crawl, with per-file license and author recorded).
- Faces found with MediaPipe Face Landmarker and cut into **aligned squares**: centered a quarter of the way from the eye midpoint to the mouth, with side = max(4 × eye distance, 3.6 × eye–mouth distance), rotated so that the eyes are level. **The browser uses exactly the same convention.** The student only knows faces framed this way.
- 997 FairFace faces added to rebalance skin tones and ages, plus ~740 profile and three-quarter views. In full profile, MediaPipe loses the face, so those crops are derived from the skin zone of a segmentation instead.

### Automatic quality control

Every pair is measured, then filtered with thresholds tuned by eye on contact sheets of the worst cases:

- faces: the face must be found again in the output, landmarks may not move more than 5 % of the crop, scale must stay within 0.88–1.15, tilt within 8°, no strong lightening, and coarse structure correlation ≥ 0.35;
- scenes: edge correlation, worst local tile, invented edges (objects or HUD text the teacher hallucinated), and invented people (YOLOv8n run on both images).

## 3. The student

`student/model.py`, `Large(width=32, blocks=6)`, ~2 M parameters:

```
3×3 conv → stride-2 conv (½) → stride-2 conv (¼) → stride-2 conv (⅛) → 6 residual blocks
→ upsample + conv (¼) + skip → upsample + conv (½) + skip → upsample + conv → conv → 3×3 conv → tanh
```

BatchNorm during training, fused into the convolutions at export. There is no full-resolution skip, so the network can't just copy the photo through.

### Losses

| Loss | Weight | Why |
|---|---|---|
| L1 | 1 | colour and layout |
| VGG16 perceptual (relu1_2 … relu4_3) | 1 | texture and sharpness |
| Hinge PatchGAN, conditional, spectral norm | 0.5 (after 3 epochs) | crisp "rendered" materials instead of blur |
| Shift-consistency | 2 | the same frame shifted by up to 8 px and re-noised must give the shifted output, so there is no flicker in video |
| Extra L1 around the mouth | 2 | mouths were the blurriest part |

Phone-camera augmentations (exposure ±20 %, offset, noise), random crops and horizontal flips are applied. Rare cases (beards, bald heads, dark skin, glasses, profiles, open mouths) are oversampled. Training uses an EMA of the weights and keeps the checkpoint with the best validation perceptual loss.

### Export

`student/export_tflite.py` fuses BatchNorm, wraps the network to take NHWC input, converts it with `litert-torch`, and **re-runs the `.tflite` file against PyTorch on a random input**. The export fails if the max difference exceeds 1e-3. An earlier ONNX→TFLite conversion of the benchmark model silently drifted by ~3 %, hence this check.

## 4. The zones model

The person's silhouette matters for two reasons: so that a pasted head doesn't drag background along with it, and so that a hand in front of the face stays real. MediaPipe's multiclass segmenter is accurate, but it took 50–60 ms on the phone and misfired there. It became a teacher instead: it labelled ~7,900 photos, and the same small architecture (`Student(32, 5)`, 160²) learned to output three channels (occluders, hair, skin) with BCE + Dice loss. On the phone, the zones model runs in 8–13 ms.

## 5. In the browser (`web/src/main.ts`)

Every frame:

1. Take the center square of the camera (512²).
2. Run MediaPipe Face Landmarker (up to 5 faces), compute the aligned square of each face, and smooth it adaptively over time: heavily when the head is still, lightly when it moves, so there's no lag.
3. Run the zones model on the frame. Silhouette = max(hair, skin) minus occluders. Occluders are min-filtered over 4 neighbours, because speckled red on cheeks and beards used to punch holes in the face. Hair and skin are averaged with the previous frame.
4. For each face: warp the square to 256², run the head model, multiply by a soft super-elliptic edge and by the silhouette (warped into the square), drop bluish fringes, and warp back.
5. Optionally apply a "pixel" filter (160² nearest-neighbour, 24 levels per channel, 4×4 Bayer dithering). It hides the remaining seams and costs 2–8 ms.

All model calls go through a single queue, because the GPU runs one model at a time anyway.

## What didn't work

- **Full-frame student for the whole scene:** at the sizes that held 15 fps, the room only looked "filtered". At sizes that looked good, it ran at 8–13 fps.
- **Painting the face once and carrying it as a texture on the face mesh** (a Snapchat-style mask) gave visible seams on the forehead, frozen expressions, and many special cases for eyes, mouth and beards. Restyling the whole head every frame was simpler and looked better.
- **Re-tinting the output from the photo's colours** darkened faces that were simply backlit, eyes and teeth included. Naming the colours in the prompt worked; post-hoc colour transfer didn't.
