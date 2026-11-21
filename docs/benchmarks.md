# Benchmarks

Only measurements on a real phone count. Desktop numbers below only show that a backend loads and runs.

**Device:** iPhone 13-class (390 × 844 @3x, 4 cores reported), iOS 26.6.1, Safari 26.6.1, WebGPU available. Phone held in hand, rear camera, warm device.

## 1. Choosing a browser inference engine

Test model: AnimeGANv3 "Hayao" (4.2 MB), same weights in every format, fixed input shape. "ms" covers inference plus reading back the result. "fps" covers the whole loop: camera, conversion, inference, display.

| Engine | Res. | Duration | Frames | fps | ms median | ms p95 | fps first → last minute |
|---|---|---|---|---|---|---|---|
| **LiteRT.js · WebGPU** | 256 | 600 s | 9,865 | **16.4** | 44 | 52 | 16.7 → 16.3 |
| LiteRT.js · WebGPU | 384 | 30 s | 337 | 11.2 | 69 | 73 | – |
| ONNX Runtime Web · WASM (4 threads) | 256 | 30 s | 76 | 2.5 | 369 | 505 | – |
| LiteRT.js · WASM (no threads) | 256 | 30 s | 53 | 1.8 | 562 | 578 | – |
| ONNX Runtime Web · WebGPU | 256 | 16 s | 24 | 1.5 | 617 | 766 | – |
| TensorFlow.js · WebGL | 256 | 32 s | 34 | 1.1 | 553 | 2,144 | – |

LiteRT.js on WebGPU at 256² ran for 10 minutes without a crash and with negligible slowdown. The other engines were 6–15× slower. TF.js also needed 41 s to load the first time.

Desktop sanity check (MacBook Pro M5 Pro, headless Chrome, 256²): LiteRT.js WebGPU 57 fps (display-capped), ONNX Runtime WebGPU 20, ONNX Runtime WASM 8, TF.js WebGL 7, LiteRT.js WASM 2.

## 2. The released models on the phone

| Model | Input | Time per call |
|---|---|---|
| Head (`head_g32_256.tflite`, 7.9 MB) | 256² | 18–20 ms (26–27 ms when the pixel filter runs in the same frame) |
| Zones (`zones_m_160.tflite`, 1.7 MB) | 160² | 7.6–13 ms (MediaPipe multiclass segmenter: 50–60 ms) |
| MediaPipe Face Landmarker | 512² | 2.5 ms with no face, ~30 ms with a face |
| Pixel filter (160², 24 levels) | – | 2–8 ms |

Full pipeline (128 one-second samples):

| Situation | fps (median) |
|---|---|
| No face | 60 (display-capped) |
| One face | 14–16 |

Face landmark detection, not the restyling network, is the biggest cost per frame.

## 3. Earlier student sizes (for reference)

Smaller students from the same family, at 256² on the same phone:

| Student | Params | File | Time |
|---|---|---|---|
| `Student(24, 4)` (s) | ~0.2 M | 0.8 MB | – |
| `Student(32, 5)` (m) | ~0.4 M | 1.7 MB | ~18 ms as a full-frame scene model |
| `Student(48, 6)` (l) | ~1.1 M | 4.4 MB | ~21–25 ms |
| `Large(32, 6)` (g32) | ~2 M | 7.9 MB | 18–20 ms |
| `Large(64, 8)` (xl) | ~10 M | 41 MB | 112–214 ms |

The `Large` family (one extra downsampling level, no full-resolution skip) gives clearly better faces than the `Student` family at a similar phone cost. At 256 it costs about the same as `Student(48, 6)`, because most of its compute runs at 1/8 resolution.

## Open questions

- Some Android phones and Windows laptops (Chrome) ran at 6–8 fps. The cause hasn't been found yet. Suspects: a WASM fallback when `navigator.gpu` is missing, integrated GPUs, and pixel read-backs.
- Memory and heat were not measured, because Safari doesn't expose memory to the page.
