// Copies the LiteRT.js and MediaPipe WebAssembly runtimes into public/ so they are served from our own origin,
// and the released weights from ../weights/ into public/models/ when present.
import { cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
cpSync(join(root, "node_modules/@litertjs/core/wasm"), join(root, "public/litert"), { recursive: true });
cpSync(join(root, "node_modules/@mediapipe/tasks-vision/wasm"), join(root, "public/mediapipe"), { recursive: true });

mkdirSync(join(root, "public/models"), { recursive: true });
for (const f of ["head_g32_256.tflite", "zones_m_160.tflite"]) {
  const src = join(root, "../weights", f);
  if (existsSync(src)) cpSync(src, join(root, "public/models", f));
}
const missing = ["head_g32_256.tflite", "zones_m_160.tflite", "face_landmarker.task"].filter((f) => !existsSync(join(root, "public/models", f)));
if (missing.length) console.warn(`missing in web/public/models: ${missing.join(", ")} — run \`npm run models\` (see README)`);
console.log("runtimes copied to public/litert and public/mediapipe");
