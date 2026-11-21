# Contributing

Thanks for your interest! Contributions of all sizes are welcome: bug reports, measurements on new devices, docs, new styles and model improvements.

## Where to start

Look at the [open issues](../../issues), especially the ones labelled **good first issue** and **help wanted**. If you want to work on something bigger, open an issue first so we can agree on the approach.

Useful contributions that need no ML knowledge:

- **Device measurements.** Open the [live demo](https://cmoirox.github.io/realtime-ai-filter-in-browser/) and report the fps shown at the bottom, with your device, OS and browser. Slow devices are especially interesting.
- **Docs and translations** of the README.

## Setup

```bash
# web demo
cd web && npm install && npm run models && npm run dev

# Python tools (uv installs the dependencies listed in each script's usage line)
uv run --python 3.11 --with torch --with mediapipe --with opencv-python-headless --with numpy \
  python tools/restyle_photo.py some_photo.jpg --out out/ --side-by-side
```

Download the weights from the [latest release](../../releases/latest) into `weights/` for the Python tools.

## Pull requests

- Keep them focused: one change per PR, with a short description of *why*.
- Code, comments and docs in English.
- Web: run `npx tsc -p .` and `npm run build` in `web/`. If your change affects speed, give before/after fps on a real phone.
- Models: when you change the architecture, the training or the export, include a contact sheet (photo | teacher | student) on the held-out faces, and confirm that `student/export_tflite.py` passes its exactness check.
- New weights go into a GitHub Release, never into git.

## Privacy rules (important)

- **Never commit photos or videos of people** (datasets, test images, screenshots) unless they are public domain or CC0 / CC BY *and* not of private individuals in sensitive contexts. Credit them in `docs/images/CREDITS.md`.
- Don't add analytics, tracking or any network call that sends camera data. The demo must stay 100 % on-device.

## Code of conduct

Be kind and constructive. Harassment, or using this project to target, demean or deceive real people, is not welcome. The maintainer may close issues or PRs and block users who don't respect this.
