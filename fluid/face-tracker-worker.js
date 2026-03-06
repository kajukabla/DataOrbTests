const visionApiByUrl = new Map();
let landmarker = null;
let processing = false;
let activeDelegate = 'cpu';
let config = null;
let runtimeErrorStreak = 0;
let lastTimestampMs = 0;
const FACE_BLENDSHAPE_KEYS = [
  'jawOpen',
  'mouthPucker',
  'mouthFunnel',
  'mouthSmileLeft',
  'mouthSmileRight',
  'eyeBlinkLeft',
  'eyeBlinkRight',
  'browInnerUp',
  'browDownLeft',
  'browDownRight',
  'cheekPuff',
  'noseSneerLeft',
  'noseSneerRight',
];

function landmarkerOptions(modelURL, delegate) {
  return {
    baseOptions: {
      modelAssetPath: modelURL,
      delegate,
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.25,
    minFacePresenceConfidence: 0.25,
    minTrackingConfidence: 0.2,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  };
}

function extractBlendshapeScores(res) {
  const categories = res?.faceBlendshapes?.[0]?.categories;
  if (!categories?.length) return { scores: null, count: 0 };
  const bag = Object.create(null);
  for (const cat of categories) {
    if (!cat?.categoryName) continue;
    bag[cat.categoryName] = cat.score;
  }
  const out = Object.create(null);
  let used = 0;
  for (const key of FACE_BLENDSHAPE_KEYS) {
    const v = Number(bag[key]);
    if (!Number.isFinite(v)) continue;
    out[key] = Math.max(0, Math.min(1, v));
    used++;
  }
  return { scores: used > 0 ? out : null, count: categories.length };
}

function extractMatrix(res) {
  const raw = res?.facialTransformationMatrixes?.[0];
  if (!raw) return null;
  const src = Array.isArray(raw)
    ? raw
    : (raw.data || raw.matrix || raw.values || raw);
  if (!src || typeof src.length !== 'number' || src.length < 16) return null;
  const out = new Array(16);
  for (let i = 0; i < 16; i++) {
    const v = Number(src[i]);
    out[i] = Number.isFinite(v) ? v : 0;
  }
  return out;
}

async function ensureVisionBundle(bundleURL) {
  if (visionApiByUrl.has(bundleURL)) return visionApiByUrl.get(bundleURL);
  const mod = await import(bundleURL);
  const api = mod?.FaceLandmarker ? mod : mod?.default;
  if (!api || !api.FaceLandmarker || !api.FilesetResolver) {
    throw new Error('MediaPipe vision module did not expose FaceLandmarker');
  }
  visionApiByUrl.set(bundleURL, api);
  return api;
}

async function createLandmarker({ bundleURL, wasmURL, modelURL, preferGPU = false }) {
  config = { bundleURL, wasmURL, modelURL, preferGPU };
  const vision = await ensureVisionBundle(bundleURL);
  const fileset = await vision.FilesetResolver.forVisionTasks(wasmURL);

  async function tryCreate(delegate) {
    return vision.FaceLandmarker.createFromOptions(fileset, landmarkerOptions(modelURL, delegate));
  }
  const delegates = preferGPU ? ['GPU', 'CPU'] : ['CPU', 'GPU'];
  let lastErr = null;
  for (const delegate of delegates) {
    try {
      landmarker = await tryCreate(delegate);
      activeDelegate = delegate.toLowerCase();
      return;
    } catch (err) {
      lastErr = err;
      console.warn(`Face worker: ${delegate} delegate unavailable.`, err?.message || err);
    }
  }
  throw lastErr || new Error('Failed to initialize face landmarker');
}

function dispose() {
  try {
    landmarker?.close?.();
  } catch {}
  landmarker = null;
  runtimeErrorStreak = 0;
  lastTimestampMs = 0;
}

async function recoverWithDelegate(delegate) {
  if (!config) return false;
  try {
    try { landmarker?.close?.(); } catch {}
    landmarker = null;
    const vision = await ensureVisionBundle(config.bundleURL);
    const fileset = await vision.FilesetResolver.forVisionTasks(config.wasmURL);
    landmarker = await vision.FaceLandmarker.createFromOptions(fileset, landmarkerOptions(config.modelURL, delegate));
    activeDelegate = delegate.toLowerCase();
    runtimeErrorStreak = 0;
    self.postMessage({ type: 'delegate-fallback', delegate: activeDelegate });
    return true;
  } catch (err) {
    console.warn(`Face worker: recoverWithDelegate(${delegate}) failed`, err?.message || err);
    return false;
  }
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  if (msg.type === 'dispose') {
    dispose();
    return;
  }

  if (msg.type === 'init') {
    try {
      dispose();
      await createLandmarker(msg);
      self.postMessage({ type: 'init-ok', delegate: activeDelegate });
    } catch (err) {
      self.postMessage({ type: 'init-error', error: err?.message || String(err) });
    }
    return;
  }

  if (msg.type !== 'process') return;
  if (!landmarker) {
    try { msg.frame?.close?.(); } catch {}
    self.postMessage({ type: 'result', landmarks: null, inferenceMs: 0 });
    return;
  }
  if (processing) {
    try { msg.frame?.close?.(); } catch {}
    self.postMessage({ type: 'result', dropped: true, landmarks: null, inferenceMs: 0 });
    return;
  }

  processing = true;
  const t0 = performance.now();
  try {
    const frame = msg.frame;
    let ts = Number.isFinite(msg.timestampMs) ? msg.timestampMs : performance.now();
    if (!Number.isFinite(ts)) ts = performance.now();
    if (ts <= lastTimestampMs) ts = lastTimestampMs + 0.1;
    lastTimestampMs = ts;

    const res = landmarker.detectForVideo(frame, ts);
    try { frame?.close?.(); } catch {}
    const blend = extractBlendshapeScores(res);
    const matrix = extractMatrix(res);

    const lms = res?.faceLandmarks?.[0];
    if (!lms || !lms.length) {
      self.postMessage({
        type: 'result',
        landmarks: null,
        inferenceMs: performance.now() - t0,
        blendshapes: blend.scores,
        blendshapeCount: blend.count,
        matrix,
      });
      return;
    }

    const packed = new Float32Array(lms.length * 3);
    for (let i = 0; i < lms.length; i++) {
      const o = i * 3;
      packed[o] = lms[i].x;
      packed[o + 1] = lms[i].y;
      packed[o + 2] = lms[i].z;
    }

    self.postMessage({
      type: 'result',
      landmarks: packed.buffer,
      inferenceMs: performance.now() - t0,
      count: lms.length,
      blendshapes: blend.scores,
      blendshapeCount: blend.count,
      matrix,
    }, [packed.buffer]);
    runtimeErrorStreak = 0;
  } catch (err) {
    try { msg.frame?.close?.(); } catch {}
    runtimeErrorStreak++;
    if (runtimeErrorStreak >= 3 && activeDelegate === 'gpu') {
      const ok = await recoverWithDelegate('CPU');
      if (ok) {
        self.postMessage({
          type: 'result',
          landmarks: null,
          inferenceMs: performance.now() - t0,
          error: 'recovered-from-gpu-error',
        });
        processing = false;
        return;
      }
    }
    if (runtimeErrorStreak >= 7 && activeDelegate === 'cpu') {
      await recoverWithDelegate('CPU');
    }
    self.postMessage({
      type: 'result',
      landmarks: null,
      inferenceMs: performance.now() - t0,
      error: err?.message || String(err),
    });
  } finally {
    processing = false;
  }
};
