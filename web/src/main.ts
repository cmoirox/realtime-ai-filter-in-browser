// Real-time whole-head restyling in the browser.
//
// Per frame:
//   camera (center square, S x S)
//   -> MediaPipe Face Landmarker: one aligned square per face (same convention as the training crops)
//   -> zones model (160 x 160, whole frame): hair + skin = the person's silhouette, red = occluders (hands)
//   -> head model (256 x 256) on each aligned square
//   -> paste back with a soft edge, clipped to the silhouette
//   -> optional "pixel" effect (low resolution + few colours + ordered dithering) that hides small artifacts
//
// URL parameters: ?video=<url> (a video file instead of the camera), ?pixel=0 (disable the pixel effect),
// ?pixel=side,levels (e.g. 160,24), ?zones=0 (no silhouette clipping), ?faces=5 (max faces).
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const base = import.meta.env.BASE_URL;
const params = new URLSearchParams(location.search);
const S = 512; // working image side (center square of the camera)
const N = 256; // head model input/output side
const NZ = 160; // zones model input/output side
const MAX_FACES = Number(params.get("faces") ?? 5);
const USE_ZONES = params.get("zones") !== "0";
const PIXEL = (params.get("pixel") ?? "160,24").split(",").map(Number);

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const screen = $<HTMLCanvasElement>("screen");
const video = $<HTMLVideoElement>("video");
const welcome = $("welcome");
const message = $("message");
const toolbar = $("toolbar");
const stats = $("stats");

function canvas(w: number, h: number, read = false): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return [c, c.getContext("2d", { willReadFrequently: read })!];
}

// ---------- models (LiteRT.js, WebGPU with a WASM fallback) ----------

type Model = (input: Float32Array) => Promise<Float32Array>;
let litert: Promise<typeof import("@litertjs/core")> | null = null;
let backend = "?";

async function loadModel(name: string, n: number): Promise<Model> {
  litert ??= (async () => {
    const m = await import("@litertjs/core");
    await m.loadLiteRt(`${base}litert/`, { threads: false });
    return m;
  })();
  const { loadAndCompile, Tensor } = await litert;
  const accelerator = "gpu" in navigator ? "webgpu" : "wasm";
  backend = accelerator;
  const model = await loadAndCompile(`${base}models/${name}`, { accelerator });
  const shape = [1, n, n, 3];
  return async (input) => {
    const t = new Tensor(input, shape);
    const outputs = (await model.run(t)) as InstanceType<typeof Tensor>[];
    t.delete();
    let out = outputs[0];
    if (out.accelerator !== "wasm") out = await out.moveTo("wasm");
    const data = new Float32Array(out.toTypedArray() as Float32Array);
    out.delete();
    for (const other of outputs.slice(1)) other.delete();
    return data;
  };
}

// The GPU runs one model at a time: calls are queued.
let gpuQueue: Promise<unknown> = Promise.resolve();
function queued<T>(f: () => Promise<T>): Promise<T> {
  const p = gpuQueue.then(f);
  gpuQueue = p.catch(() => {});
  return p;
}

/** A model "workbench": input canvas -> model -> output canvas, both n x n, values mapped from/to [-1, 1]. */
type Bench = { n: number; input: HTMLCanvasElement; cInput: CanvasRenderingContext2D; output: HTMLCanvasElement; cOutput: CanvasRenderingContext2D; buffer: Float32Array; pixels: ImageData };
function bench(n: number): Bench {
  const [input, cInput] = canvas(n, n, true);
  const [output, cOutput] = canvas(n, n);
  return { n, input, cInput, output, cOutput, buffer: new Float32Array(n * n * 3), pixels: new ImageData(n, n) };
}
async function runBench(model: Model, b: Bench) {
  const d = b.cInput.getImageData(0, 0, b.n, b.n).data;
  const x = b.buffer;
  for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
    x[j] = d[i] / 127.5 - 1;
    x[j + 1] = d[i + 1] / 127.5 - 1;
    x[j + 2] = d[i + 2] / 127.5 - 1;
  }
  const y = await model(x);
  const p = b.pixels.data;
  for (let i = 0, j = 0; i < p.length; i += 4, j += 3) {
    p[i] = (y[j] + 1) * 127.5;
    p[i + 1] = (y[j + 1] + 1) * 127.5;
    p[i + 2] = (y[j + 2] + 1) * 127.5;
    p[i + 3] = 255;
  }
  b.cOutput.putImageData(b.pixels, 0, 0);
}

// ---------- faces ----------

const EYE_L = [33, 133];
const EYE_R = [362, 263];
const MOUTH = [61, 291];
type Square = { cx: number; cy: number; side: number; angle: number };

/** Aligned head square. MUST match teacher/detect_faces.py: the head model only knows faces framed this way. */
function alignedSquare(pts: { x: number; y: number }[]): Square {
  const mid = (ids: number[]) => ({ x: ((pts[ids[0]].x + pts[ids[1]].x) / 2) * S, y: ((pts[ids[0]].y + pts[ids[1]].y) / 2) * S });
  const l = mid(EYE_L);
  const r = mid(EYE_R);
  const m = mid(MOUTH);
  const eyes = { x: (l.x + r.x) / 2, y: (l.y + r.y) / 2 };
  const eyeDist = Math.hypot(r.x - l.x, r.y - l.y);
  const eyeMouth = Math.hypot(m.x - eyes.x, m.y - eyes.y);
  return {
    cx: eyes.x + 0.25 * (m.x - eyes.x),
    cy: eyes.y + 0.25 * (m.y - eyes.y),
    side: Math.max(4 * eyeDist, 3.6 * eyeMouth),
    angle: Math.atan2(r.y - l.y, r.x - l.x),
  };
}

/** Matches detections to last frame's faces and smooths them: strong when the head is still, weak when it moves fast (no lag). */
let tracked: Square[] = [];
function track(raw: Square[]) {
  tracked = raw.map((q) => {
    const old = tracked.find((s) => Math.hypot(s.cx - q.cx, s.cy - q.cy) < 0.4 * q.side);
    if (!old) return q;
    const speed = Math.hypot(old.cx - q.cx, old.cy - q.cy) / q.side;
    const a = Math.min(1, 0.5 + speed * 12);
    let da = q.angle - old.angle;
    da = Math.atan2(Math.sin(da), Math.cos(da));
    return { cx: old.cx + a * (q.cx - old.cx), cy: old.cy + a * (q.cy - old.cy), side: old.side + a * (q.side - old.side), angle: old.angle + a * da };
  });
}

// ---------- compositing ----------

const [work, cWork] = canvas(S, S, true); // frame being composed
const [paste, cPaste] = canvas(N, N, true); // one restyled head, with its alpha
const [silhouetteSq, cSilhouetteSq] = canvas(N, N); // the silhouette, in the head square's frame
const [zonesAlpha, cZonesAlpha] = canvas(NZ, NZ);

/** Opacity of a pasted head: 1 in the middle, wide soft fade towards the edge (between a circle and a square). */
function softEdge(radius = 0.45, fade = 0.13): HTMLCanvasElement {
  const [c, cc] = canvas(N, N);
  const im = cc.createImageData(N, N);
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) {
      const u = Math.abs((x + 0.5) / N - 0.5);
      const w = Math.abs((y + 0.5) / N - 0.5);
      const d = Math.pow(u ** 3 + w ** 3, 1 / 3);
      const a = Math.min(1, Math.max(0, (radius - d) / fade));
      im.data[4 * (y * N + x) + 3] = 255 * a * a * (3 - 2 * a);
    }
  cc.putImageData(im, 0, 0);
  return c;
}
const edge = softEdge();

// Zones output: red = in front of the face (hands, held objects), green = hair, blue = skin.
const RED = [150, 220]; // red below the first value: nothing in front; above the second: certainly in front
let zonesRgb: Uint8Array | null = null;
async function runZones(model: Model, b: Bench) {
  b.cInput.drawImage(work, 0, 0, NZ, NZ);
  await queued(() => runBench(model, b));
  const d = b.pixels.data;
  zonesRgb ??= new Uint8Array(NZ * NZ * 3);
  const z = zonesRgb;
  const im = cZonesAlpha.createImageData(NZ, NZ);
  for (let i = 0; i < NZ * NZ; i++) {
    // occluders: only solid, connected areas (min over the 4 neighbours removes speckles on cheeks and beards)
    const x = i % NZ;
    const y = (i / NZ) | 0;
    let r = d[4 * i];
    if (x > 0) r = Math.min(r, d[4 * (i - 1)]);
    if (x < NZ - 1) r = Math.min(r, d[4 * (i + 1)]);
    if (y > 0) r = Math.min(r, d[4 * (i - NZ)]);
    if (y < NZ - 1) r = Math.min(r, d[4 * (i + NZ)]);
    z[3 * i] = Math.max(0, Math.min(255, ((r - RED[0]) * 255) / (RED[1] - RED[0])));
    // hair and skin: averaged with the previous frame so the borders don't shimmer
    z[3 * i + 1] = (z[3 * i + 1] + d[4 * i + 1]) >> 1;
    z[3 * i + 2] = (z[3 * i + 2] + d[4 * i + 2]) >> 1;
    // silhouette = hair or skin, minus whatever is in front (a hand in front of the face stays real)
    const person = Math.max(0, Math.min(255, (Math.max(z[3 * i + 1], z[3 * i + 2]) - 24) * 5));
    im.data[4 * i + 3] = person * (1 - z[3 * i] / 255);
  }
  cZonesAlpha.putImageData(im, 0, 0);
}

/** The model sometimes leaks bluish/purple fringes at the crop border; skin and hair always have less blue than green. */
function removeBlue(threshold = 12, range = 22) {
  const im = cPaste.getImageData(0, 0, N, N);
  const d = im.data;
  for (let i = 0; i < d.length; i += 4) {
    const excess = d[i + 2] - d[i + 1];
    if (excess > threshold) d[i + 3] *= Math.max(0, 1 - (excess - threshold) / range);
  }
  cPaste.putImageData(im, 0, 0);
}

/** Into square space: transforms so that drawing the S x S working image lands the head square on N x N. */
function toSquare(c: CanvasRenderingContext2D, q: Square) {
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.translate(N / 2, N / 2);
  c.rotate(-q.angle);
  c.scale(N / q.side, N / q.side);
  c.translate(-q.cx, -q.cy);
}

async function restyleHead(model: Model, b: Bench, q: Square, source: HTMLCanvasElement) {
  b.cInput.setTransform(1, 0, 0, 1, 0, 0);
  b.cInput.fillStyle = "#777";
  b.cInput.fillRect(0, 0, N, N);
  toSquare(b.cInput, q);
  b.cInput.drawImage(source, 0, 0);
  await queued(() => runBench(model, b));
  cPaste.globalCompositeOperation = "copy";
  cPaste.drawImage(b.output, 0, 0);
  cPaste.globalCompositeOperation = "destination-in";
  cPaste.drawImage(edge, 0, 0);
  if (zonesRgb) {
    cSilhouetteSq.clearRect(0, 0, N, N);
    toSquare(cSilhouetteSq, q);
    cSilhouetteSq.imageSmoothingQuality = "high";
    cSilhouetteSq.drawImage(zonesAlpha, 0, 0, S, S);
    cSilhouetteSq.setTransform(1, 0, 0, 1, 0, 0);
    cPaste.drawImage(silhouetteSq, 0, 0);
  }
  cPaste.globalCompositeOperation = "source-over";
  removeBlue();
  cWork.save();
  cWork.translate(q.cx, q.cy);
  cWork.rotate(q.angle);
  cWork.imageSmoothingQuality = "high";
  cWork.drawImage(paste, -q.side / 2, -q.side / 2, q.side, q.side);
  cWork.restore();
}

// "Pixel" effect: low resolution with crisp pixels, few colours, 4 x 4 ordered dithering.
const [low, cLow] = canvas(PIXEL[0] || 1, PIXEL[0] || 1, true);
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
function pixelEffect() {
  const [n, levels = 24] = PIXEL;
  if (!n || !pixelOn) return;
  cLow.imageSmoothingEnabled = true;
  cLow.imageSmoothingQuality = "high";
  cLow.drawImage(work, 0, 0, n, n);
  if (levels > 1) {
    const im = cLow.getImageData(0, 0, n, n);
    const d = im.data;
    const step = 255 / (levels - 1);
    for (let y = 0, k = 0; y < n; y++)
      for (let x = 0; x < n; x++, k += 4) {
        const b = (BAYER[(y & 3) * 4 + (x & 3)] / 16 - 0.47) * step;
        d[k] = Math.round((d[k] + b) / step) * step;
        d[k + 1] = Math.round((d[k + 1] + b) / step) * step;
        d[k + 2] = Math.round((d[k + 2] + b) / step) * step;
      }
    cLow.putImageData(im, 0, 0);
  }
  cWork.imageSmoothingEnabled = false;
  cWork.drawImage(low, 0, 0, S, S);
  cWork.imageSmoothingEnabled = true;
}

// ---------- main loop ----------

let running = false;
let showOriginal = false;
let pixelOn = !!PIXEL[0];
let facing: "environment" | "user" = "environment";
let headModel: Model;
let zonesModel: Model | null = null;
let landmarker: FaceLandmarker;
const headBench = bench(N);
const zonesBench = bench(NZ);
const [original, cOriginal] = canvas(S, S);
const frameTimes: number[] = [];
let modelMs = 0;
let lastT = 0;

async function frame() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return;
  const side = Math.min(vw, vh);
  cWork.drawImage(video, (vw - side) / 2, (vh - side) / 2, side, side, 0, 0, S, S);
  cOriginal.drawImage(work, 0, 0);
  const t = Math.max(performance.now(), lastT + 1); // detectForVideo wants strictly increasing timestamps
  lastT = t;
  const res = landmarker.detectForVideo(work, t);
  track(res.faceLandmarks.map(alignedSquare));
  if (showOriginal) return;
  const t0 = performance.now();
  if (zonesModel && tracked.length) await runZones(zonesModel, zonesBench);
  for (const q of tracked) await restyleHead(headModel, headBench, q, original);
  modelMs += 0.1 * (performance.now() - t0 - modelMs);
  pixelEffect();
}

function draw() {
  const dpr = Math.min(2, devicePixelRatio || 1);
  const w = Math.round(innerWidth * dpr);
  const h = Math.round(innerHeight * dpr);
  if (screen.width !== w || screen.height !== h) {
    screen.width = w;
    screen.height = h;
  }
  const c = screen.getContext("2d")!;
  c.fillStyle = "#000";
  c.fillRect(0, 0, w, h);
  const side = Math.min(w, h);
  c.save();
  if (facing === "user") {
    c.translate(w, 0);
    c.scale(-1, 1); // selfie camera: show a mirror image
  }
  c.drawImage(work, (w - side) / 2, (h - side) / 2, side, side);
  c.restore();
}

async function loop() {
  while (running) {
    const t0 = performance.now();
    try {
      await frame();
    } catch (err) {
      message.textContent = String(err);
    }
    draw();
    frameTimes.push(performance.now() - t0);
    if (frameTimes.length > 30) frameTimes.shift();
    const avg = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
    stats.textContent = `${(1000 / avg).toFixed(0)} fps · ${tracked.length} face(s) · models ${modelMs.toFixed(0)} ms · ${backend}`;
    await new Promise((r) => requestAnimationFrame(r));
  }
}

async function startCamera() {
  const src = params.get("video");
  if (src) {
    video.src = src;
    video.loop = true;
    video.crossOrigin = "anonymous";
  } else {
    (video.srcObject as MediaStream | null)?.getTracks().forEach((tr) => tr.stop());
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
    });
  }
  await video.play();
}

async function start() {
  message.textContent = "Loading models…";
  try {
    const files = await FilesetResolver.forVisionTasks(`${base}mediapipe`);
    const options = (delegate: "GPU" | "CPU") => ({
      baseOptions: { modelAssetPath: `${base}models/face_landmarker.task`, delegate },
      runningMode: "VIDEO" as const,
      numFaces: MAX_FACES,
    });
    landmarker = await FaceLandmarker.createFromOptions(files, options("GPU")).catch(() => FaceLandmarker.createFromOptions(files, options("CPU")));
    headModel = await loadModel("head_g32_256.tflite", N);
    if (USE_ZONES) zonesModel = await loadModel("zones_m_160.tflite", NZ);
    message.textContent = "Starting camera…";
    await startCamera();
  } catch (err) {
    message.textContent = `Could not start: ${err}`;
    return;
  }
  message.textContent = "";
  welcome.hidden = true;
  toolbar.hidden = false;
  running = true;
  if ("wakeLock" in navigator) navigator.wakeLock.request("screen").catch(() => {});
  loop();
}

$("start").addEventListener("click", start);
$("flip").addEventListener("click", async () => {
  facing = facing === "environment" ? "user" : "environment";
  tracked = [];
  await startCamera().catch((err) => (message.textContent = String(err)));
});
$("pixel").addEventListener("click", (e) => {
  pixelOn = !pixelOn;
  (e.target as HTMLButtonElement).textContent = pixelOn ? "Pixel: on" : "Pixel: off";
});
$("pixel").textContent = pixelOn ? "Pixel: on" : "Pixel: off";
// Hold to compare with the raw camera.
const compare = $("compare");
for (const [ev, v] of [["pointerdown", true], ["pointerup", false], ["pointerleave", false]] as const) compare.addEventListener(ev, () => (showOriginal = v));
