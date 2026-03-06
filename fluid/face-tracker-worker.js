let landmarker = null;
let processing = false;
let activeDelegate = 'cpu';
let config = null;
let runtimeErrorStreak = 0;
let lastTimestampMs = 0;

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
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  };
}

async function createLandmarker({ libURL, wasmURL, modelURL }) {
  config = { libURL, wasmURL, modelURL };
  const vision = await import(libURL);
  const fileset = await vision.FilesetResolver.forVisionTasks(wasmURL);

  async function tryCreate(delegate) {
    return vision.FaceLandmarker.createFromOptions(fileset, landmarkerOptions(modelURL, delegate));
  }

  try {
    landmarker = await tryCreate('GPU');
    activeDelegate = 'gpu';
    return;
  } catch (err) {
    // Fall through to CPU if GPU delegate is unavailable.
    console.warn('Face worker: GPU delegate unavailable, falling back to CPU.', err?.message || err);
  }
  landmarker = await tryCreate('CPU');
  activeDelegate = 'cpu';
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
    const vision = await import(config.libURL);
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

    const lms = res?.faceLandmarks?.[0];
    if (!lms || !lms.length) {
      self.postMessage({
        type: 'result',
        landmarks: null,
        inferenceMs: performance.now() - t0,
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
