// Downloads the model files the demo needs into public/models/:
//   - the released student weights (GitHub Release of this repository)
//   - MediaPipe Face Landmarker (Google, Apache-2.0)
// Usage: npm run models   (set RELEASE_URL to download from a fork's release)
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "public/models");
mkdirSync(out, { recursive: true });

const RELEASE_URL = process.env.RELEASE_URL ?? "https://github.com/cmoirox/realtime-ai-filter-in-browser/releases/download/v1.0.0";
// Expected SHA-256 of each file: a tampered or truncated download is rejected (set SKIP_SHA256=1 for your own weights).
const files = {
  "head_g32_256.tflite": [`${RELEASE_URL}/head_g32_256.tflite`, "34c789a654787e5d7286f7f7ec5028855acac337cddf7f10e4b60be24b3b2dec"],
  "zones_m_160.tflite": [`${RELEASE_URL}/zones_m_160.tflite`, "54b79e7c2e1f63c49190a5cdd000f1974b03c446726eccd35322d82ccb333568"],
  "face_landmarker.task": [
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
  ],
};
const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");
const check = (name, buf, expected) => process.env.SKIP_SHA256 === "1" || sha256(buf) === expected || (console.error(`BAD HASH ${name}`), false);

for (const [name, [url, expected]] of Object.entries(files)) {
  const dest = join(out, name);
  if (existsSync(dest)) {
    if (!check(name, readFileSync(dest), expected)) process.exitCode = 1;
    else console.log(`ok       ${name}`);
    continue;
  }
  const res = await fetch(url);
  if (!res.ok) {
    console.error(`FAILED   ${name}: ${res.status} ${url}`);
    process.exitCode = 1;
    continue;
  }
  const buf = Buffer.from(await res.arrayBuffer());
  if (!check(name, buf, expected)) {
    process.exitCode = 1;
    continue;
  }
  writeFileSync(dest, buf);
  console.log(`fetched  ${name}`);
}
