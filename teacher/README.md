# Teacher: building the training pairs

The teacher is **FLUX.2 klein 4B** (image edit, 4 steps, 8-bit weights) run locally with
[mflux](https://github.com/filipstrand/mflux). It restyles real photos into a stylized 3D
life-simulation game look; the resulting `photo -> styled` pairs train the small real-time
student network. Prompts and generation settings (seed 42, 4 steps, 512 x 512) live in
[`prompts.py`](prompts.py).

Two datasets are built: **scenes** (whole 512 x 512 photos) and **faces** (aligned 512 x 512
face squares, plus a set of side profiles). Everything is written under `data/` (see the data
layout in the repository README); third-party weights go in `models/`:

| file | used by |
| --- | --- |
| `models/flux2-klein-4b-q8/` (FLUX.2 klein 4B, mflux 8-bit) | `restyle_scenes.py`, `restyle_faces.py` |
| `models/face_landmarker.task` (MediaPipe) | face detection, cropping, quality, mouths |
| `models/selfie_multiclass_256x256.tflite` (MediaPipe) | `crop_profiles.py`, `label_zones.py` |
| `models/yolov8n.pt` (Ultralytics) | `detect_invented_people.py` |

The face describer (`mlx-community/Qwen2.5-VL-7B-Instruct-4bit`) is fetched by `mlx-vlm` on first use.
All commands run from the repository root. Every generation step is resumable: re-running it skips
what is already done. On macOS, prefix long runs with `caffeinate -i` to keep the machine awake.

Timing: the teacher takes roughly 10-20 s per 512 x 512 image on an Apple M5 Pro.

## Scenes

1. `python3 teacher/select_coco.py --download`
   Picks ~3,800 everyday city-life COCO photos (transport, meals, offices, kitchens, streets, groups,
   portraits...) from `data/coco/annotations/instances_{train,val}2017.json`, writes
   `data/coco/selection.csv` (with each image's license id) and downloads them to `data/sources/coco/<category>/`.
2. `python3 teacher/download_wikimedia.py`
   Adds ~800 freely licensed Wikimedia Commons photos for what COCO lacks (inside metros, buses and trams,
   cafes, shops, pedestrians) to `data/sources/wikimedia/<category>/`, with author and license of each file in
   `data/sources/wikimedia/credits.csv`.
3. `caffeinate -i uv run --python 3.12 --with mflux --with pillow python teacher/restyle_scenes.py [--limit N]`
   Skips unreadable, small (< 400 px) and black-and-white / sepia photos, center-crops to 512 x 512 and
   restyles with the teacher -> `data/scenes/{photo,styled}/`, `data/scenes/log.csv`. ~10-20 s per image.
4. `uv run --python 3.12 --with pillow --with numpy python teacher/quality_scenes.py`
   Measures how faithful each styled image is (edges, layout, brightness, invented text...) -> `data/scenes/quality.csv`.
5. `uv run --python 3.12 --with ultralytics python teacher/detect_invented_people.py`
   Runs YOLOv8n on both images to flag people the teacher invented -> `data/scenes/people.csv`.
6. `python3 teacher/split_scenes.py`
   Applies the thresholds -> `data/scenes/verdict.csv` and `data/splits/scenes_{train,val,test}.txt`
   (20 test images spread across categories, 10 % validation).

## Faces

7. `uv run --python 3.11 --with mediapipe --with pillow --with numpy python teacher/detect_faces.py`
   Finds faces in `data/sources/` with MediaPipe Face Landmarker and records the aligned square
   (side = 4 x eye distance, eyes horizontal) -> `data/faces/landmarks.csv`.
8. `uv run --python 3.11 --with pillow --with numpy python teacher/crop_faces.py`
   Crops the squares to 512 x 512 (drops small, cut-off and strongly tilted faces) -> `data/faces/photo/v_*.png`, `data/faces/crops.csv`.
9. `uv run --python 3.11 --with mediapipe --with opencv-python-headless --with numpy --with pyarrow python teacher/add_fairface.py`
   Adds ~1,000 more diverse faces from FairFace (CC BY 4.0; put the 1.25-padding validation parquet in
   `data/sources/fairface/1.25/` first) -> `data/faces/photo/f_*.png`, labels in `data/faces/fairface.csv`.
10. `uv run --python 3.12 --with mlx-vlm --with pillow python teacher/describe_faces.py`
    Describes skin, hair, eyes, facial hair, glasses and headwear of every face with Qwen2.5-VL -> `data/faces/descriptions.csv`.
11. `caffeinate -i uv run --python 3.12 --with mflux --with pillow python teacher/restyle_faces.py`
    Restyles every described face with a prompt that names those attributes -> `data/faces/styled/`.
    ~10-20 s per face. (`--no-description` uses the plain face prompt instead.)
12. `uv run --python 3.11 --with mediapipe --with pillow --with numpy python teacher/quality_faces.py`
    Re-detects the face in both images and measures offset, scale, tilt, layout and brightness -> `data/faces/quality.csv`.
13. `python3 teacher/split_faces.py`
    Applies the thresholds -> `data/faces/verdict.csv` and `data/splits/faces_{train,val,test}.txt` (24 test faces).

## Side profiles (optional, improves turned heads)

14. `uv run --python 3.11 --with mediapipe --with opencv-python-headless --with numpy --with pyarrow python teacher/find_profiles.py`
    Scans the FairFace parquet files and records head yaw and aligned square -> `data/faces/profiles/candidates.csv`.
15. `uv run --python 3.11 --with mediapipe --with opencv-python-headless --with numpy --with pyarrow --with torch python teacher/crop_profiles.py`
    Crops turned heads; true profiles (lost by the landmarker) get a square derived from the face-skin
    segmentation -> `data/faces/profiles/photo/p_*.png`, `data/faces/profiles/crops.csv`.
16. `uv run --python 3.11 --with opencv-python-headless --with numpy python teacher/select_profiles.py [count per ethnicity]`
    Drops gray, dark, black-bordered and blurry crops and balances ethnicities -> `data/faces/profiles/selected.csv`.
17. `uv run --python 3.12 --with mlx-vlm --with pillow python teacher/describe_faces.py --profiles`
    Describes the selected profiles, including which way the head is turned -> `data/faces/profiles/descriptions.csv`.
18. `caffeinate -i uv run --python 3.12 --with mflux --with pillow python teacher/restyle_faces.py --profiles`
    Restyles the side views (front and back views are skipped) -> `data/faces/profiles/styled/`.
19. `uv run --python 3.11 --with pillow --with numpy python teacher/add_profiles.py`
    Light quality check, hard-links the kept profiles into `data/faces/{photo,styled}/`, appends them to
    `data/splits/faces_train.txt` and holds 36 out in `data/splits/faces_profiles_holdout.txt`.
    Run it after step 13 (which rewrites `faces_train.txt`), and again whenever step 13 is re-run.

## Training extras

20. `uv run --python 3.11 --with mediapipe --with pillow --with numpy python teacher/measure_mouths.py`
    Mouth openness of every face pair -> `data/faces/mouths.csv`.
21. `python3 teacher/sample_weights.py`
    Per-face sampling weights (`name,value`): mouth openness plus bonuses for facial hair, bald heads,
    darker skin, glasses and profiles -> `data/faces/sample_weights.csv`.
22. `uv run --python 3.11 --with mediapipe --with pillow --with numpy python teacher/label_zones.py`
    Labels every photo with the MediaPipe multiclass segmenter (hair, body skin, face skin, clothes...)
    for the small zone model -> `data/zones/{scenes,faces}/*.png`.

## Reviewing

`uv run --python 3.12 --with pillow python teacher/contact_sheet.py <split or verdict.csv> [--reason R] [--n 20] [--shuffle]`
tiles `photo | styled` pairs into a JPEG in `data/contact_sheets/` for review by eye, e.g. `scenes_test`,
`faces_val`, or `data/scenes/verdict.csv --reason invented_person`.

## Licenses of the data

COCO images keep their original Flickr licenses (license id in `data/coco/selection.csv`), Wikimedia
Commons images their own free licenses (`data/sources/wikimedia/credits.csv`), and FairFace is CC BY 4.0.
Styled images are derived from these photos.
