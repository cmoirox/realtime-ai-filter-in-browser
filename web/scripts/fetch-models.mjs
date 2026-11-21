// Downloads the model files the demo needs into public/models/:
//   - the released student weights (GitHub Release of this repository)
//   - MediaPipe Face Landmarker (Google, Apache-2.0)
// Usage: npm run models   (set RELEASE_URL to download from a fork's release)
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "public/models");
mkdirSync(out, { recursive: true });

const RELEASE_URL = process.env.RELEASE_URL ?? "https://github.com/cmoirox/realtime-ai-filter-in-browser/releases/download/v1.0.0";
const files = {
  "head_g32_256.tflite": `${RELEASE_URL}/head_g32_256.tflite`,
  "zones_m_160.tflite": `${RELEASE_URL}/zones_m_160.tflite`,
  "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
};

for (const [name, url] of Object.entries(files)) {
  const dest = join(out, name);
  if (existsSync(dest)) {
    console.log(`ok       ${name}`);
    continue;
  }
  const res = await fetch(url);
  if (!res.ok) {
    console.error(`FAILED   ${name}: ${res.status} ${url}`);
    process.exitCode = 1;
    continue;
  }
  writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
  console.log(`fetched  ${name}`);
}
