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
// ?pixel=1 (turn on the pixel effect) or ?pixel=side,levels (e.g. 160,24), ?zones=0 (no silhouette clipping),
// ?faces=5 (max faces), ?eyelids=0 (don't draw closed eyelids).
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const base = import.meta.env.BASE_URL;
const params = new URLSearchParams(location.search);
const S = 512; // working image side (center square of the camera)
const N = 256; // head model input/output side
const NZ = 160; // zones model input/output side
const MAX_FACES = Number(params.get("faces") ?? 5);
const USE_ZONES = params.get("zones") !== "0";
const PIXEL_PARAM = params.get("pixel") ?? "0";
const PIXEL = (PIXEL_PARAM === "1" ? "160,24" : PIXEL_PARAM).split(",").map(Number);
const PIXEL_DEFAULT = PIXEL[0] ? PIXEL : [160, 24];
const EYELIDS = params.get("eyelids") !== "0";

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
/** A tracked face: smoothed square, landmarks (working-image pixels) and the expression signals used for eyelids. */
type Face = Square & {
  pts: Float32Array; // x, y per landmark
  blink: number[]; // MediaPipe eyeBlink score per eye (smoothed); [0] = TOUR_EYE[0]
  smile: number;
  mouthOpen: number;
  frontal: number; // 1 = facing the camera, 0 = full profile
  tClosedSince: number[]; // when each eye started being closed (0 = open)
  tLastClosed: number[];
};
type Detection = Square & { pts: Float32Array; blink: number[]; smile: number; mouthOpen: number; frontal: number };

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

/** Landmarks + blendshapes -> square, landmarks in pixels and expression signals. */
function detection(pts: { x: number; y: number; z: number }[], shapes?: { categoryName: string; score: number }[]): Detection {
  const blink = [0, 0];
  let smile = 0;
  for (const b of shapes ?? []) {
    if (b.categoryName === "eyeBlinkRight") blink[0] = b.score; // TOUR_EYE[0] is the person's right eye
    if (b.categoryName === "eyeBlinkLeft") blink[1] = b.score;
    if (b.categoryName === "mouthSmileLeft" || b.categoryName === "mouthSmileRight") smile += b.score / 2;
  }
  const xy = new Float32Array(pts.length * 2);
  for (let i = 0; i < pts.length; i++) {
    xy[2 * i] = pts[i].x * S;
    xy[2 * i + 1] = pts[i].y * S;
  }
  // facing the camera or not: normal of the plane (cheek to cheek) x (forehead to chin)
  const d = (i: number, j: number) => [pts[i].x - pts[j].x, pts[i].y - pts[j].y, pts[i].z - pts[j].z];
  const u = d(454, 234);
  const w = d(10, 152);
  const nx = u[1] * w[2] - u[2] * w[1];
  const ny = u[2] * w[0] - u[0] * w[2];
  const nz = u[0] * w[1] - u[1] * w[0];
  const dist = (i: number, j: number) => Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y);
  return {
    ...alignedSquare(pts),
    pts: xy,
    blink,
    smile,
    mouthOpen: dist(14, 13) / (dist(291, 61) || 1),
    frontal: Math.abs(nz) / (Math.hypot(nx, ny, nz) || 1),
  };
}

/** Matches detections to last frame's faces and smooths them: strong when the head is still, weak when it moves fast (no lag). */
let tracked: Face[] = [];
function track(raw: Detection[]) {
  tracked = raw.map((q) => {
    const old = tracked.find((s) => Math.hypot(s.cx - q.cx, s.cy - q.cy) < 0.4 * q.side);
    if (!old) return { ...q, tClosedSince: [0, 0], tLastClosed: [0, 0] };
    const speed = Math.hypot(old.cx - q.cx, old.cy - q.cy) / q.side;
    const a = Math.min(1, 0.5 + speed * 12);
    let da = q.angle - old.angle;
    da = Math.atan2(Math.sin(da), Math.cos(da));
    return {
      cx: old.cx + a * (q.cx - old.cx),
      cy: old.cy + a * (q.cy - old.cy),
      side: old.side + a * (q.side - old.side),
      angle: old.angle + a * da,
      pts: q.pts, // eyelids follow the raw landmarks: they must sit exactly on the eyes
      blink: [0, 1].map((k) => old.blink[k] + 0.7 * (q.blink[k] - old.blink[k])),
      smile: old.smile + 0.5 * (q.smile - old.smile),
      mouthOpen: old.mouthOpen + 0.6 * (q.mouthOpen - old.mouthOpen),
      frontal: q.frontal,
      tClosedSince: old.tClosedSince,
      tLastClosed: old.tLastClosed,
    };
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

// ---------- eyelids ----------
//
// The head model always draws open eyes (closed eyes are rare in the training photos), so a blink would just
// squash the drawn eye. Closed eyelids are drawn on top instead: a soft patch of the face's own (restyled) skin
// colour over the eye, and a dark lash line where the lids meet.

const TOUR_EYE = [
  [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
];
const BLINK = [0.4, 0.65, 5000]; // closure where the lid starts to appear, where it is full; ms after which eyes kept closed are reopened
const WINK = 0.12; // blink-score difference that allows a single closed eye (wink), when facing the camera

/** True when something (a hand…) is in front of this point of the working image, according to the zones model. */
function occluded(x: number, y: number): boolean {
  if (!zonesRgb) return false;
  const i = Math.min(NZ - 1, Math.max(0, Math.round((y / S) * NZ))) * NZ + Math.min(NZ - 1, Math.max(0, Math.round((x / S) * NZ)));
  return zonesRgb[3 * i] > 128;
}

/** Skin colour of the restyled face: the brightest of a few cheek / cheekbone / forehead points (glasses frames can cover some). */
function restyledSkin(f: Face): number[] {
  let best = [0, 0, 0];
  for (const i of [50, 280, 205, 425, 151]) {
    const x = Math.max(0, Math.min(S - 5, Math.round(f.pts[2 * i]) - 2));
    const y = Math.max(0, Math.min(S - 5, Math.round(f.pts[2 * i + 1]) - 2));
    const d = cWork.getImageData(x, y, 5, 5).data;
    const m = [0, 0, 0];
    for (let q = 0; q < d.length; q += 4) for (let k = 0; k < 3; k++) m[k] += d[q + k] / 25;
    if (m[0] + m[1] + m[2] > best[0] + best[1] + best[2]) best = m;
  }
  return best;
}

function eyelids(f: Face, t: number) {
  if (Math.max(f.blink[0], f.blink[1]) <= BLINK[0]) {
    for (let k = 0; k < 2; k++) if (t - f.tLastClosed[k] > 400) f.tClosedSince[k] = 0;
    return;
  }
  const P = (k: number, n: number) => [f.pts[2 * TOUR_EYE[k][n]], f.pts[2 * TOUR_EYE[k][n] + 1]];
  const width = (k: number) => Math.hypot(P(k, 0)[0] - P(k, 8)[0], P(k, 0)[1] - P(k, 8)[1]);
  // smiling or talking squints the eyes without closing them: then the lids must really touch
  const animated = f.smile > 0.3 || f.mouthOpen > 0.25;
  const gaps = [0, 0];
  const closure = [0, 1].map((k) => {
    // two signals: MediaPipe's blink score, and the real distance between the lids (the score alone is fooled by caps,
    // glasses, looking down)
    const gap = Math.hypot(P(k, 12)[0] - P(k, 4)[0], P(k, 12)[1] - P(k, 4)[1]) / (width(k) || 1);
    gaps[k] = gap;
    const c = Math.min(1, Math.max(0, Math.min((f.blink[k] - BLINK[0]) / (BLINK[1] - BLINK[0]), (0.3 - gap) / 0.08), (0.2 - gap) / 0.08));
    return animated ? Math.min(c, Math.max(0, (0.15 - gap) / 0.04)) : c;
  });
  // One eye closed without a clear wink is usually a measurement error (profile, glasses): close only if both are.
  const winkOk = f.frontal > 0.85 && !animated && Math.abs(gaps[0] - gaps[1]) > 0.08;
  if (Math.abs(f.blink[0] - f.blink[1]) < (winkOk ? WINK : 0.35)) closure[0] = closure[1] = Math.min(closure[0], closure[1]);
  let skin: number[] | null = null;
  for (let k = 0; k < 2; k++) {
    let c = closure[k];
    // A blink is short. "Closed" eyes that last are almost always someone looking down (reading, phone): reopen them.
    if (c >= 0.2) {
      f.tLastClosed[k] = t;
      if (!f.tClosedSince[k]) f.tClosedSince[k] = t;
    } else if (t - f.tLastClosed[k] > 400) f.tClosedSince[k] = 0;
    if (f.tClosedSince[k]) c *= Math.min(1, Math.max(0, 1 - (t - f.tClosedSince[k] - BLINK[2]) / 200));
    if (c <= 0) continue;
    let cx = 0;
    let cy = 0;
    for (const i of TOUR_EYE[k]) {
      cx += f.pts[2 * i] / TOUR_EYE[k].length;
      cy += f.pts[2 * i + 1] / TOUR_EYE[k].length;
    }
    if (occluded(cx, cy)) continue; // a hand in front of the eye
    skin ??= restyledSkin(f);
    const w = width(k);
    const tint = skin.map(Math.round).join(",");
    cWork.save();
    // the lid: a plain skin patch, wider than the real eye (the drawn eyes are bigger), solid in the middle, soft edge
    cWork.translate(cx, cy);
    cWork.rotate(f.angle);
    cWork.scale(w * 0.95, w * 0.6);
    const g = cWork.createRadialGradient(0, 0, 0.55, 0, 0, 1);
    g.addColorStop(0, `rgba(${tint},1)`);
    g.addColorStop(1, `rgba(${tint},0)`);
    cWork.fillStyle = g;
    cWork.globalAlpha = c;
    cWork.beginPath();
    cWork.arc(0, 0, 1, 0, 6.3);
    cWork.fill();
    cWork.setTransform(1, 0, 0, 1, 0, 0);
    // the lashes: a dark line along the lower lid, where the lids meet
    cWork.globalAlpha = c * 0.9;
    cWork.strokeStyle = "#3a2a22";
    cWork.lineWidth = Math.max(1.2, w * 0.075);
    cWork.lineCap = "round";
    cWork.lineJoin = "round";
    cWork.beginPath();
    for (let n = 0; n <= 8; n++) {
      const [x, y] = P(k, n);
      if (n) cWork.lineTo(x, y);
      else cWork.moveTo(x, y);
    }
    cWork.stroke();
    cWork.restore();
  }
}

// "Pixel" effect: low resolution with crisp pixels, few colours, 4 x 4 ordered dithering.
const [low, cLow] = canvas(PIXEL_DEFAULT[0], PIXEL_DEFAULT[0], true);
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
function pixelEffect() {
  const [n, levels = 24] = PIXEL_DEFAULT;
  if (!pixelOn) return;
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
const pixelOn = !!PIXEL[0];
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
  track(res.faceLandmarks.map((pts, i) => detection(pts, res.faceBlendshapes?.[i]?.categories)));
  if (showOriginal) return;
  const t0 = performance.now();
  if (zonesModel && tracked.length) await runZones(zonesModel, zonesBench);
  for (const f of tracked) {
    await restyleHead(headModel, headBench, f, original);
    if (EYELIDS) eyelids(f, t);
  }
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
      outputFaceBlendshapes: true, // eye-blink scores for the eyelids
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
// Hold to compare with the raw camera.
const compare = $("compare");
for (const [ev, v] of [["pointerdown", true], ["pointerup", false], ["pointerleave", false]] as const) compare.addEventListener(ev, () => (showOriginal = v));
