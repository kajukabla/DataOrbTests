// ─── Bayesian Optimization Controller (PCA-based) ───────────────────────────
// Single GP in PCA space. Dual 1-5 ratings for motion + color, combined score.
// Mutate-best (70%) + EI exploration (30%).

import {
  gpFit, gpPredict, expectedImprovement,
  optimizeHyperparams, latinHypercube,
  fitPCA, toPCA, fromPCA
} from './gp.js';

// ─── Parameter Space ─────────────────────────────────────────────────────────

export const SLIDER_SPACE = {
  simSpeed:          { min: 0.1,  max: 1,     step: 0.01 },
  sizeRandomness:    { min: 0,    max: 1,     step: 0.01 },
  colorBlend:        { min: 0,    max: 1,     step: 0.01 },
  sheenStrength:     { min: 0,    max: 1,     step: 0.01 },
  clickSize:         { min: 0,    max: 1,     step: 0.01 },
  clickStrength:     { min: 0,    max: 1,     step: 0.01 },
  burstBehavior:     { min: 0,    max: 4,     step: 1    },
  burstCount:        { min: 0,    max: 16,    step: 1    },
  burstForce:        { min: 0,    max: 8,     step: 0.01 },
  burstForceRandomness:{ min: 0,  max: 1,     step: 0.01 },
  burstDyeIntensity: { min: 0,    max: 3.0,   step: 0.01 },
  burstSpeed:        { min: 0,    max: 10.0,  step: 0.01 },
  burstTravelSpeed:  { min: 0.25, max: 3.0,   step: 0.01 },
  burstDuration:     { min: 0.05, max: 32.0,  step: 0.01 },
  burstWidth:        { min: 0,    max: 12.0,  step: 0.01 },
  burstRadialAngle:  { min: 0,    max: 360,   step: 1    },
  noiseAmount:       { min: 0,    max: 1,     step: 0.01 },
  noiseType:         { min: 0,    max: 7,     step: 1    },
  noiseBehavior:     { min: 0,    max: 8,     step: 1    },
  noiseFrequency:    { min: 0,    max: 1,     step: 0.01 },
  noiseSpeed:        { min: 0,    max: 1,     step: 0.01 },
  noiseWarp:         { min: 0,    max: 1,     step: 0.01 },
  noiseSharpness:    { min: 0,    max: 1,     step: 0.01 },
  noiseAnisotropy:   { min: 0,    max: 1,     step: 0.01 },
  noiseBlend:        { min: 0,    max: 1,     step: 0.01 },
  curlStrength:      { min: 0,    max: 50,    step: 1    },
  maccormack:        { min: 0,    max: 1,     step: 0.01 },
  velDissipation:    { min: 0.99, max: 1.0,   step: 0.001 },
  dyeDissipation:    { min: 0.98, max: 1.0,   step: 0.001 },
  dyeSoftCap:        { min: 0,    max: 1,     step: 1    },
  dyeCeiling:        { min: 0.3,  max: 3.0,   step: 0.01 },
  pressureIters:     { min: 10,   max: 60,    step: 1    },
  pressureDecay:     { min: 0,    max: 1,     step: 0.01 },
  prismaticAmount:   { min: 0,    max: 20,    step: 0.1  },
  noiseDyeIntensity: { min: 0,    max: 1,     step: 0.01 },
  dyeNoiseAmount:    { min: 0,    max: 0.15,  step: 0.001 },
  tempAmount:        { min: 0,    max: 1,     step: 0.01 },
  tempBuoyancy:      { min: 0,    max: 3,     step: 0.01 },
  tempDissipation:   { min: 0.95, max: 1.0,   step: 0.001 },
  tempDyeHeat:       { min: 0,    max: 3,     step: 0.01 },
  tempEdgeCool:      { min: 0,    max: 3,     step: 0.01 },
  tempRadialMix:     { min: 0,    max: 1,     step: 0.01 },
  tempColorShift:    { min: 0,    max: 3,     step: 0.01 },
  moodAmount:        { min: 0,    max: 1,     step: 0.01 },
  moodSpeed:         { min: 0,    max: 1,     step: 0.01 },
  paletteIndex:      { min: -1,   max: 49,    step: 1 },
  metallic:          { min: 0,    max: 1,     step: 0.01 },
  roughness:         { min: 0,    max: 1,     step: 0.01 },
  baseColor_0:       { min: 0, max: 1, step: 0.01 },
  baseColor_1:       { min: 0, max: 1, step: 0.01 },
  baseColor_2:       { min: 0, max: 1, step: 0.01 },
  accentColor_0:     { min: 0, max: 1, step: 0.01 },
  accentColor_1:     { min: 0, max: 1, step: 0.01 },
  accentColor_2:     { min: 0, max: 1, step: 0.01 },
  tipColor_0:        { min: 0, max: 1, step: 0.01 },
  tipColor_1:        { min: 0, max: 1, step: 0.01 },
  tipColor_2:        { min: 0, max: 1, step: 0.01 },
  glitterColor_0:    { min: 0, max: 1, step: 0.01 },
  glitterColor_1:    { min: 0, max: 1, step: 0.01 },
  glitterColor_2:    { min: 0, max: 1, step: 0.01 },
  glitterAccent_0:   { min: 0, max: 1, step: 0.01 },
  glitterAccent_1:   { min: 0, max: 1, step: 0.01 },
  glitterAccent_2:   { min: 0, max: 1, step: 0.01 },
  glitterTip_0:      { min: 0, max: 1, step: 0.01 },
  glitterTip_1:      { min: 0, max: 1, step: 0.01 },
  glitterTip_2:      { min: 0, max: 1, step: 0.01 },
  sheenColor_0:      { min: 0, max: 1, step: 0.01 },
  sheenColor_1:      { min: 0, max: 1, step: 0.01 },
  sheenColor_2:      { min: 0, max: 1, step: 0.01 },
  bloomIntensity:    { min: 0, max: 1, step: 0.01 },
  bloomThreshold:    { min: 0, max: 2, step: 0.01 },
  bloomRadius:       { min: 0, max: 1, step: 0.01 },
  faceEffectorMode:  { min: 0, max: 2, step: 1    },
  faceMeshNoiseAmount: { min: 0, max: 6, step: 0.01 },
  faceMeshNoiseFreq: { min: 1, max: 60, step: 0.5 },
  faceMeshNoiseSpeed: { min: 0, max: 5, step: 0.01 },
  faceMeshNoiseDir: { min: 0, max: 1, step: 0.01 },
  faceMouthSimBoost: { min: 0, max: 2, step: 0.01 },
  faceDyeContribution: { min: 0, max: 4, step: 0.01 },
  faceDyeFill:       { min: 0, max: 5, step: 0.01 },
  faceEdgeBoost:     { min: 0, max: 3, step: 0.01 },
  faceFlowCarry:     { min: 0, max: 1.5, step: 0.01 },
  faceEyeOpenSize:   { min: 0, max: 2, step: 0.01 },
  faceMouthOpenSize: { min: 0, max: 2, step: 0.01 },
  faceMouthClosedSize: { min: 0, max: 2, step: 0.01 },
  faceMouthBoost:    { min: 0, max: 3, step: 0.01 },
  faceMaskDetail:    { min: 0.2, max: 1, step: 0.01 },
  faceStampSize:     { min: 0.5, max: 3, step: 0.01 },
  faceDebugMode:     { min: 0, max: 2, step: 1    },
  faceMeshEyeScale:  { min: 0.5, max: 3, step: 0.01 },
  faceDyeNoise:      { min: 0, max: 1, step: 1    },
  faceMeshThickness: { min: 0.5, max: 3, step: 0.01 },
  radialMask:        { min: 0.5, max: 1, step: 0.001 },
  boundaryMode:      { min: 0, max: 1, step: 1 },
  colormapMode:      { min: 0, max: 17, step: 1    },
  colorSource:       { min: 0, max: 8, step: 1    },
  colorGain:         { min: 0, max: 1, step: 0.01 },
  colormapCompress:  { min: 0, max: 1, step: 1    },
  particleCount:     { min: 65536, max: 4194304, step: 65536 },
  particleSize:      { min: 0.1, max: 4, step: 0.01 },
  glitterCap:        { min: 0.01, max: 3, step: 0.01 },
  streakGlow:        { min: 0, max: 20, step: 0.1 },
  densitySize:       { min: 0, max: 4, step: 0.1 },
  glitterFloor:      { min: 0.001, max: 0.2, step: 0.001 },
  glintBrightness:   { min: 0, max: 1, step: 0.01 },
  splatRadius:       { min: 0.0001, max: 0.01, step: 0.0001 },
  masterSpeed:       { min: 0, max: 1, step: 0.01 },
  noiseMapping:      { min: 0, max: 1, step: 1    },
  sphereMode:        { min: 0, max: 1, step: 1    },
  shadowExtend:      { min: 0, max: 1, step: 0.01 },
  sphereSize:        { min: 0.5, max: 2, step: 0.01 },
};

export const SLIDER_KEYS = Object.keys(SLIDER_SPACE);
export const D = SLIDER_KEYS.length;

const SLIDER_DEFAULTS = {
  particleCount: 4194304,
  particleSize: 0.9,
  dyeSoftCap: 1,
  dyeCeiling: 1.2,
  glitterCap: 1.0,
  streakGlow: 10.0,
  densitySize: 1.5,
  glitterFloor: 0.01,
  glintBrightness: 0.1,
  splatRadius: 0.0015,
  masterSpeed: 1.0,
  sphereSize: 1.0,
  shadowExtend: 0.5,
  simSpeed: 1.0,
  colorBlend: 0.5,
  clickSize: 0.5,
  clickStrength: 0.5,
  velDissipation: 0.998,
  dyeDissipation: 0.993,
  pressureIters: 30,
  pressureDecay: 0.8,
  curlStrength: 15,
  prismaticAmount: 20.0,
  bloomThreshold: 0.4,
  bloomRadius: 0.5,
  colorGain: 0.5,
  tempBuoyancy: 0.5,
  tempDissipation: 0.99,
  moodSpeed: 0.3,
  faceDyeContribution: 1.1,
  faceDyeFill: 1.8,
  faceEdgeBoost: 0.9,
  faceFlowCarry: 0.12,
  faceEyeOpenSize: 1.0,
  faceMouthOpenSize: 1.0,
  faceMouthClosedSize: 1.0,
  faceMouthBoost: 1.0,
  faceMaskDetail: 0.68,
  faceStampSize: 1.35,
  faceMeshEyeScale: 1.4,
  faceMeshNoiseAmount: 0.5,
  faceMeshNoiseFreq: 12.0,
  faceMeshNoiseSpeed: 1.0,
  faceMouthSimBoost: 0.3,
  faceMeshThickness: 1.0,
  radialMask: 1.0,
  boundaryMode: 0,
  noiseWarp: 0.35,
  noiseSharpness: 0.5,
  noiseAnisotropy: 0.5,
  noiseBlend: 0.5,
  burstForce: 0.8,
  burstForceRandomness: 0.25,
  burstDyeIntensity: 1.0,
  burstSpeed: 0.4,
  burstTravelSpeed: 1.2,
  burstDuration: 0.8,
  burstWidth: 0.35,
  burstRadialAngle: 45,
  sheenStrength: 1.5,
  metallic: 0.3,
  roughness: 0.4,
  paletteIndex: -1,
};

export const COLOR_KEYS = [
  'baseColor', 'accentColor', 'tipColor', 'glitterColor',
  'glitterAccent', 'glitterTip', 'sheenColor'
];

const NOISE_CONTROL_KEYS = ['noiseWarp', 'noiseSharpness', 'noiseAnisotropy', 'noiseBlend'];
const NOISE_PROFILE_BY_TYPE = [
  { noiseWarp: 0.22, noiseSharpness: 0.30, noiseAnisotropy: 0.25, noiseBlend: 0.24 },
  { noiseWarp: 0.76, noiseSharpness: 0.58, noiseAnisotropy: 0.40, noiseBlend: 0.52 },
  { noiseWarp: 0.42, noiseSharpness: 0.78, noiseAnisotropy: 0.30, noiseBlend: 0.48 },
  { noiseWarp: 0.38, noiseSharpness: 0.55, noiseAnisotropy: 0.20, noiseBlend: 0.74 },
  { noiseWarp: 0.52, noiseSharpness: 0.46, noiseAnisotropy: 0.86, noiseBlend: 0.40 },
  { noiseWarp: 0.50, noiseSharpness: 0.62, noiseAnisotropy: 0.72, noiseBlend: 0.44 },
  { noiseWarp: 0.66, noiseSharpness: 0.58, noiseAnisotropy: 0.60, noiseBlend: 0.64 },
  { noiseWarp: 0.78, noiseSharpness: 0.58, noiseAnisotropy: 0.70, noiseBlend: 0.62 },
];
const NOISE_ACTIVE_BY_TYPE = [
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: true, noiseBlend: true },
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: true, noiseBlend: true },
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: false, noiseBlend: true },
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: false, noiseBlend: true },
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: true, noiseBlend: true },
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: true, noiseBlend: true },
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: true, noiseBlend: true },
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: true, noiseBlend: true },
];

function clampNoiseType(raw) {
  const n = Math.round(Number.isFinite(raw) ? raw : 0);
  return Math.max(0, Math.min(NOISE_PROFILE_BY_TYPE.length - 1, n));
}

function snapClamp(v, s, fallback) {
  let out = Number.isFinite(v) ? v : fallback;
  out = Math.round(out / s.step) * s.step;
  if (out < s.min) out = s.min;
  if (out > s.max) out = s.max;
  return out;
}

function canonicalizeNoiseFields(params) {
  const noiseType = clampNoiseType(params.noiseType);
  params.noiseType = noiseType;
  const profile = NOISE_PROFILE_BY_TYPE[noiseType];
  const active = NOISE_ACTIVE_BY_TYPE[noiseType];
  for (const key of NOISE_CONTROL_KEYS) {
    const space = SLIDER_SPACE[key];
    const fallback = profile[key];
    const raw = active[key] ? params[key] : fallback;
    params[key] = snapClamp(raw, space, fallback);
  }
}

// ─── Color channel helpers ──────────────────────────────────────────────────

function getStateVal(state, key) {
  const m = key.match(/^(.+)_([012])$/);
  if (m && Array.isArray(state[m[1]])) return state[m[1]][parseInt(m[2])];
  return state[key];
}

function setStateVal(state, key, val) {
  const m = key.match(/^(.+)_([012])$/);
  if (m && Array.isArray(state[m[1]])) { state[m[1]][parseInt(m[2])] = val; return; }
  state[key] = val;
}

// ─── Normalization ───────────────────────────────────────────────────────────

export function stateToNormalized(state) {
  const canonical = {};
  for (let i = 0; i < D; i++) {
    const key = SLIDER_KEYS[i];
    canonical[key] = getStateVal(state, key);
  }
  canonicalizeNoiseFields(canonical);

  const x = new Float64Array(D);
  for (let i = 0; i < D; i++) {
    const key = SLIDER_KEYS[i];
    const s = SLIDER_SPACE[key];
    const val = canonical[key];
    if (val === undefined || val === null || Number.isNaN(val)) {
      x[i] = 0.5;
    } else {
      x[i] = (val - s.min) / (s.max - s.min);
      if (x[i] < 0) x[i] = 0;
      if (x[i] > 1) x[i] = 1;
    }
  }
  return x;
}

export function normalizedToState(x, state, lockedKeys) {
  for (let i = 0; i < D; i++) {
    const key = SLIDER_KEYS[i];
    if (lockedKeys && lockedKeys.has(key)) continue;
    const s = SLIDER_SPACE[key];
    let val = s.min + x[i] * (s.max - s.min);
    val = Math.round(val / s.step) * s.step;
    if (val < s.min) val = s.min;
    if (val > s.max) val = s.max;
    setStateVal(state, key, val);
  }
  const canonical = {};
  for (let i = 0; i < D; i++) {
    const key = SLIDER_KEYS[i];
    canonical[key] = getStateVal(state, key);
  }
  canonicalizeNoiseFields(canonical);
  for (const key of NOISE_CONTROL_KEYS) {
    if (lockedKeys && lockedKeys.has(key)) continue;
    setStateVal(state, key, canonical[key]);
  }
  if (!(lockedKeys && lockedKeys.has('noiseType'))) {
    setStateVal(state, 'noiseType', canonical.noiseType);
  }
  if (state.simSpeed < 0.2) state.simSpeed = 0.2;
}

function randomNormalized() {
  const x = new Float64Array(D);
  for (let i = 0; i < D; i++) x[i] = Math.random();
  return x;
}

// ─── Storage (localStorage-first, server fallback) ───────────────────────────

const RATINGS_KEY = 'dataorb-ratings-v2';
const PRESETS_KEY = 'dataorb-presets';
const MAX_RATINGS = 500;

function loadRatingsLocal() {
  try { return JSON.parse(localStorage.getItem(RATINGS_KEY) || '[]'); }
  catch { return []; }
}

function saveRatingsLocal(ratings) {
  if (ratings.length > MAX_RATINGS) ratings = ratings.slice(-MAX_RATINGS);
  try { localStorage.setItem(RATINGS_KEY, JSON.stringify(ratings)); }
  catch (e) { console.warn('BO: localStorage save failed:', e); }
}

// Also try server for backward compat
async function saveRatingsServer(ratings) {
  try {
    await fetch('/api/ratings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ratings }),
    });
  } catch {}
}

// ─── PCA Configuration ──────────────────────────────────────────────────────

const PCA_COMPONENTS = 8;

// ─── BOController ────────────────────────────────────────────────────────────

export class BOController {
  constructor() {
    this.ratings = [];
    this.examples = [];
    this.model = null;         // single GP in PCA space
    this.pca = null;           // PCA transform
    this.rateMode = false;
    this.boMorphMode = false;
    this._retrainCounter = 0;
    this._suggestionCount = 0;
    this.lockedKeys = new Set();
    // Pending ratings (1-5 scale, null = not yet rated)
    this._pendingMovement = null;
    this._pendingColor = null;
    this._state = null;
    this._syncAllUI = null;
    this._statePostNormalize = null;
  }

  setStatePostNormalize(fn) {
    this._statePostNormalize = typeof fn === 'function' ? fn : null;
  }

  _runStatePostNormalize(state, source) {
    if (!this._statePostNormalize || !state) return;
    try { this._statePostNormalize(state, source); }
    catch (e) { console.warn('BO: state post-normalize hook failed:', e); }
  }

  static async create() {
    const bo = new BOController();
    bo.ratings = loadRatingsLocal();
    await bo.refreshExamples();
    bo._fitPCA();

    if (bo.ratings.length >= 5) {
      console.log(`BO: ${bo.ratings.length} ratings loaded, will retrain after first frame`);
      setTimeout(() => bo.retrain(), 100);
    }
    return bo;
  }

  // ─── PCA ──────────────────────────────────────────────────────────────

  _fitPCA() {
    if (this.examples.length < 3) {
      console.log('BO: Too few presets for PCA, using raw space');
      this.pca = null;
      return;
    }

    // Build data matrix from all presets
    const N = this.examples.length;
    const data = new Float64Array(N * D);
    for (let i = 0; i < N; i++) {
      const x = stateToNormalized(this.examples[i].params);
      for (let j = 0; j < D; j++) data[i * D + j] = x[j];
    }

    this.pca = fitPCA(data, N, D, PCA_COMPONENTS);
    if (this.pca) {
      console.log(`BO: PCA fitted on ${N} presets → ${this.pca.nK} components`);
    }
  }

  // ─── Example Management ──────────────────────────────────────────────

  _getLocalPresets() {
    try { return JSON.parse(localStorage.getItem(PRESETS_KEY) || '[]'); }
    catch { return []; }
  }

  _saveLocalPresets(presets) {
    localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
  }

  async refreshExamples() {
    try {
      let res = await fetch('/api/examples').catch(() => null);
      if (!res || !res.ok) res = await fetch('data/examples.json').catch(() => null);
      if (res && res.ok) this.examples = await res.json();
      else this.examples = [];
    } catch { this.examples = []; }
    const local = this._getLocalPresets();
    const builtInNames = new Set(this.examples.map(e => e.name));
    for (const lp of local) {
      if (!builtInNames.has(lp.name)) this.examples.push(lp);
    }
  }

  async saveExample(name, state) {
    const params = {};
    for (const key of SLIDER_KEYS) params[key] = getStateVal(state, key);
    canonicalizeNoiseFields(params);
    let saved = false;
    let res = await fetch('/api/examples', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, params }),
    }).catch(() => null);
    if (!res || !res.ok) {
      res = await fetch('http://localhost:8081/api/examples', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, params }),
      }).catch(() => null);
    }
    if (res && res.ok) saved = true;
    const local = this._getLocalPresets();
    const idx = local.findIndex(e => e.name === name);
    const entry = { name, params };
    if (idx >= 0) local[idx] = entry; else local.push(entry);
    this._saveLocalPresets(local);
    if (!saved) console.log(`Preset "${name}" saved to localStorage`);
    await this.refreshExamples();
    this._fitPCA(); // refit with new preset
  }

  async deleteExample(name) {
    try {
      let res = await fetch('/api/examples', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      }).catch(() => null);
      if (!res || !res.ok) {
        await fetch('http://localhost:8081/api/examples', {
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        }).catch(() => null);
      }
    } catch {}
    const local = this._getLocalPresets();
    this._saveLocalPresets(local.filter(e => e.name !== name));
    await this.refreshExamples();
  }

  loadExample(example, state, syncAllUI) {
    const preserveKeys = new Set(['boundaryMode']);
    const preserved = {};
    for (const key of preserveKeys) preserved[key] = getStateVal(state, key);
    for (const key of SLIDER_KEYS) {
      if (preserveKeys.has(key)) continue;
      const s = SLIDER_SPACE[key];
      const def = SLIDER_DEFAULTS[key];
      setStateVal(state, key, def !== undefined ? def : s.min);
    }
    for (const key of SLIDER_KEYS) {
      if (preserveKeys.has(key)) continue;
      if (example.params[key] !== undefined) {
        setStateVal(state, key, example.params[key]);
      }
    }
    for (const key of preserveKeys) setStateVal(state, key, preserved[key]);
    this._runStatePostNormalize(state, 'load-example');
    if (syncAllUI) syncAllUI();
  }

  async listExamples() {
    await this.refreshExamples();
    return this.examples;
  }

  // ─── GP Training ──────────────────────────────────────────────────────

  retrain() {
    const startMs = performance.now();
    const N = this.ratings.length;

    if (N < 5) {
      console.log(`BO: Need 5+ ratings (have ${N})`);
      this.model = null;
      return;
    }

    // Use most recent ratings (cap for performance)
    const recent = N > 200 ? this.ratings.slice(-200) : this.ratings;
    const n = recent.length;

    // Combined score: average of motion + color, normalized to [0,1]
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const r = recent[i];
      const motion = (r.motionScore ?? 3) / 5;  // 1-5 → 0.2-1.0
      const color = (r.colorScore ?? 3) / 5;
      y[i] = (motion + color) / 2;
    }

    // Check if all same
    if (y.every(v => Math.abs(v - y[0]) < 1e-6)) {
      console.log('BO: All ratings identical, skipping GP');
      this.model = null;
      return;
    }

    // Build training data in PCA space (or raw if no PCA)
    const usePCA = this.pca !== null;
    const gD = usePCA ? this.pca.nK : D;
    const X = new Float64Array(n * gD);

    for (let i = 0; i < n; i++) {
      const norm = stateToNormalized(recent[i].params);
      if (usePCA) {
        const z = toPCA(this.pca, norm);
        for (let j = 0; j < gD; j++) X[i * gD + j] = z[j];
      } else {
        for (let j = 0; j < gD; j++) X[i * gD + j] = norm[j];
      }
    }

    const useARD = n >= 40;
    try {
      const hp = optimizeHyperparams(X, y, n, gD, useARD);
      this.model = gpFit(X, y, n, gD, hp.lengthScales, hp.sigmaF, hp.sigmaN, hp.mu);
      if (!this.model) {
        console.warn('BO: Cholesky failed');
        return;
      }
      this.model._usePCA = usePCA;
      console.log(`BO: GP trained — N=${n}, D=${gD}, PCA=${usePCA}, ARD=${useARD} (${(performance.now() - startMs).toFixed(0)}ms)`);
    } catch (e) {
      console.warn('BO: GP failed:', e);
      this.model = null;
    }
  }

  maybeRetrain() {
    this._retrainCounter++;
    const N = this.ratings.length;
    if (N === 5 || (N >= 5 && this._retrainCounter >= 5)) {
      this._retrainCounter = 0;
      setTimeout(() => this.retrain(), 0);
    }
  }

  // ─── Suggestions ──────────────────────────────────────────────────────

  getNextParams() {
    const N = this.ratings.length;

    // Phase 1: explore with example mutations
    if (N < 5 || !this.model) {
      return this._mutateBestPreset(0.3);
    }

    this._suggestionCount++;

    // 70% mutate best rated, 30% EI explore
    if (this._suggestionCount % 10 < 7) {
      return this._mutateBestRated(0.15);
    } else {
      return this._eiSuggestion();
    }
  }

  /** Mutate a random good preset with noise in PCA space. */
  _mutateBestPreset(sigma) {
    if (this.examples.length === 0) return randomNormalized();

    // Pick a random preset (biased toward variety)
    const ex = this.examples[Math.floor(Math.random() * this.examples.length)];
    const norm = stateToNormalized(ex.params);

    if (this.pca) {
      const z = toPCA(this.pca, norm);
      for (let c = 0; c < this.pca.nK; c++) {
        z[c] += gaussianRandom() * sigma;
      }
      console.log(`BO: Suggest — mutate preset "${ex.name}" (PCA σ=${sigma})`);
      return fromPCA(this.pca, z);
    } else {
      for (let i = 0; i < D; i++) {
        norm[i] += (Math.random() - 0.5) * sigma * 2;
        if (norm[i] < 0) norm[i] = 0;
        if (norm[i] > 1) norm[i] = 1;
      }
      return norm;
    }
  }

  /** Mutate the highest-rated config in PCA space. */
  _mutateBestRated(sigma) {
    // Find best rated
    let bestScore = -Infinity;
    let bestParams = null;
    for (const r of this.ratings) {
      const score = ((r.motionScore ?? 3) + (r.colorScore ?? 3)) / 2;
      if (score > bestScore) {
        bestScore = score;
        bestParams = r.params;
      }
    }

    if (!bestParams) return this._mutateBestPreset(sigma);

    const norm = stateToNormalized(bestParams);

    if (this.pca) {
      const z = toPCA(this.pca, norm);
      // Adaptive sigma: smaller as we get more ratings
      const adaptSigma = sigma / (1 + this.ratings.length / 50);
      for (let c = 0; c < this.pca.nK; c++) {
        z[c] += gaussianRandom() * adaptSigma;
      }
      console.log(`BO: Suggest — mutate best rated (score=${bestScore.toFixed(1)}, PCA σ=${adaptSigma.toFixed(3)})`);
      return fromPCA(this.pca, z);
    } else {
      for (let i = 0; i < D; i++) {
        norm[i] += (Math.random() - 0.5) * sigma * 2;
        if (norm[i] < 0) norm[i] = 0;
        if (norm[i] > 1) norm[i] = 1;
      }
      return norm;
    }
  }

  /** EI-based suggestion in PCA space. */
  _eiSuggestion() {
    if (!this.model) return this._mutateBestPreset(0.3);

    const usePCA = this.model._usePCA;
    const gD = usePCA ? this.pca.nK : D;

    // Find best observed score
    let fBest = -Infinity;
    for (const r of this.ratings) {
      const score = ((r.motionScore ?? 3) + (r.colorScore ?? 3)) / 10;
      if (score > fBest) fBest = score;
    }

    const xi = this.ratings.length < 20 ? 0.02 : 0.05;
    const nCandidates = 500;

    // Generate candidates: mix of LHS + mutations of best
    const lhsCandidates = latinHypercube(nCandidates / 2, gD);

    // Also generate mutation candidates around best
    const mutCandidates = new Float64Array((nCandidates / 2) * gD);
    for (let i = 0; i < nCandidates / 2; i++) {
      const bestNorm = this._getBestNormalized();
      if (usePCA) {
        const z = toPCA(this.pca, bestNorm);
        for (let j = 0; j < gD; j++) {
          mutCandidates[i * gD + j] = z[j] + gaussianRandom() * 0.2;
        }
      } else {
        for (let j = 0; j < gD; j++) {
          mutCandidates[i * gD + j] = bestNorm[j] + (Math.random() - 0.5) * 0.3;
          if (mutCandidates[i * gD + j] < 0) mutCandidates[i * gD + j] = 0;
          if (mutCandidates[i * gD + j] > 1) mutCandidates[i * gD + j] = 1;
        }
      }
    }

    let bestEI = -Infinity;
    let bestX = null;
    const xStar = new Float64Array(gD);

    // Evaluate LHS candidates
    for (let i = 0; i < nCandidates / 2; i++) {
      for (let j = 0; j < gD; j++) xStar[j] = lhsCandidates[i * gD + j];
      const { mean, variance } = gpPredict(this.model, xStar);
      const ei = expectedImprovement(mean, variance, fBest, xi);
      if (ei > bestEI) { bestEI = ei; bestX = new Float64Array(xStar); }
    }

    // Evaluate mutation candidates
    for (let i = 0; i < nCandidates / 2; i++) {
      for (let j = 0; j < gD; j++) xStar[j] = mutCandidates[i * gD + j];
      const { mean, variance } = gpPredict(this.model, xStar);
      const ei = expectedImprovement(mean, variance, fBest, xi);
      if (ei > bestEI) { bestEI = ei; bestX = new Float64Array(xStar); }
    }

    if (!bestX) return this._mutateBestPreset(0.3);

    // Convert from PCA/raw space to full normalized
    if (usePCA) {
      console.log(`BO: Suggest — EI (bestEI=${bestEI.toFixed(4)})`);
      return fromPCA(this.pca, bestX);
    } else {
      return bestX;
    }
  }

  _getBestNormalized() {
    let bestScore = -Infinity;
    let bestParams = null;
    for (const r of this.ratings) {
      const score = ((r.motionScore ?? 3) + (r.colorScore ?? 3)) / 2;
      if (score > bestScore) { bestScore = score; bestParams = r.params; }
    }
    if (bestParams) return stateToNormalized(bestParams);
    if (this.examples.length > 0) return stateToNormalized(this.examples[0].params);
    return randomNormalized();
  }

  getBestParams() {
    if (!this.model) return this._getBestNormalized();

    const usePCA = this.model._usePCA;
    const gD = usePCA ? this.pca.nK : D;
    const candidates = latinHypercube(2000, gD);
    let bestMean = -Infinity;
    let bestX = null;
    const xStar = new Float64Array(gD);

    for (let i = 0; i < 2000; i++) {
      for (let j = 0; j < gD; j++) xStar[j] = candidates[i * gD + j];
      const { mean } = gpPredict(this.model, xStar);
      if (mean > bestMean) { bestMean = mean; bestX = new Float64Array(xStar); }
    }

    if (!bestX) return this._getBestNormalized();
    return usePCA ? fromPCA(this.pca, bestX) : bestX;
  }

  // ─── Rating ────────────────────────────────────────────────────────────

  /**
   * Rate motion of current config. Score: 1-5.
   */
  rateMovement(score, state, syncAllUI) {
    if (this._pendingMovement !== null) return;
    score = Math.max(1, Math.min(5, Math.round(score)));
    this._pendingMovement = score;
    this._state = state;
    this._syncAllUI = syncAllUI;
    this.flashMovement(score);
    console.log(`BO: Motion rated ${score}/5`);
    this.updateOverlay();
    this._tryCommit();
  }

  /**
   * Rate color of current config. Score: 1-5.
   */
  rateColor(score, state, syncAllUI) {
    if (this._pendingColor !== null) return;
    score = Math.max(1, Math.min(5, Math.round(score)));
    this._pendingColor = score;
    this._state = state;
    this._syncAllUI = syncAllUI;
    this.flashColor(score);
    console.log(`BO: Color rated ${score}/5`);
    this.updateOverlay();
    this._tryCommit();
  }

  _tryCommit() {
    if (this._pendingMovement === null || this._pendingColor === null) return;

    try {
      const state = this._state;
      const syncAllUI = this._syncAllUI;

      const params = {};
      for (const key of SLIDER_KEYS) params[key] = getStateVal(state, key);
      canonicalizeNoiseFields(params);

      this.ratings.push({
        params,
        motionScore: this._pendingMovement,
        colorScore: this._pendingColor,
        timestamp: Date.now(),
      });

      console.log(`BO: Committed motion=${this._pendingMovement} color=${this._pendingColor} (${this.ratings.length} total)`);
      saveRatingsLocal(this.ratings);
      saveRatingsServer(this.ratings);
      this.maybeRetrain();

      this._pendingMovement = null;
      this._pendingColor = null;

      let nextX;
      try { nextX = this.getNextParams(); }
      catch (e) {
        console.warn('BO: getNextParams failed:', e);
        nextX = this._mutateBestPreset(0.3);
      }

      normalizedToState(nextX, state, this.lockedKeys);
      this._runStatePostNormalize(state, 'rating-advance');
      if (syncAllUI) syncAllUI();
      this.updateOverlay();

      setTimeout(() => this._resetFlash(), 600);
    } catch (e) {
      console.error('BO: commit crashed:', e);
      this._pendingMovement = null;
      this._pendingColor = null;
      normalizedToState(randomNormalized(), this._state, this.lockedKeys);
      this._runStatePostNormalize(this._state, 'commit-fallback');
      if (this._syncAllUI) this._syncAllUI();
    }
  }

  getMorphTarget() {
    if (!this.model) return this._mutateBestPreset(0.2);

    if (Math.random() < 0.7) {
      return this._mutateBestRated(0.1);
    } else {
      return this._eiSuggestion();
    }
  }

  // ─── UI ──────────────────────────────────────────────────────────────

  flashMovement(score) {
    const el = document.getElementById('boFlashMovement');
    if (!el) return;
    el.textContent = '★'.repeat(score) + '☆'.repeat(5 - score);
    el.style.color = score >= 4 ? '#4f4' : score >= 3 ? '#ff4' : '#f44';
    el.style.opacity = '1';
  }

  flashColor(score) {
    const el = document.getElementById('boFlashColor');
    if (!el) return;
    el.textContent = '★'.repeat(score) + '☆'.repeat(5 - score);
    el.style.color = score >= 4 ? '#4af' : score >= 3 ? '#ff4' : '#fa4';
    el.style.opacity = '1';
  }

  _resetFlash() {
    const m = document.getElementById('boFlashMovement');
    const c = document.getElementById('boFlashColor');
    if (m) { m.textContent = '☆☆☆☆☆'; m.style.color = '#555'; m.style.opacity = '0.3'; }
    if (c) { c.textContent = '☆☆☆☆☆'; c.style.color = '#555'; c.style.opacity = '0.3'; }
  }

  toggleLockColors() {
    const COLOR_PATTERN = /Color_|Accent_|Tip_|sheen.*_/i;
    const colorSliderKeys = SLIDER_KEYS.filter(k => COLOR_PATTERN.test(k));
    const allLocked = colorSliderKeys.every(k => this.lockedKeys.has(k));
    if (allLocked) {
      colorSliderKeys.forEach(k => this.lockedKeys.delete(k));
      console.log('BO: Colors UNLOCKED');
    } else {
      colorSliderKeys.forEach(k => this.lockedKeys.add(k));
      console.log('BO: Colors LOCKED');
    }
    this.updateOverlay();
    return !allLocked;
  }

  toggleLockMotion() {
    const motionKeys = SLIDER_KEYS.filter(k =>
      ['simSpeed', 'burstBehavior', 'burstCount', 'burstForce', 'burstForceRandomness', 'burstDyeIntensity', 'burstSpeed', 'burstTravelSpeed', 'burstDuration', 'burstWidth', 'burstRadialAngle',
        'noiseAmount', 'noiseType', 'noiseBehavior', 'noiseFrequency', 'noiseSpeed',
        'noiseWarp', 'noiseSharpness', 'noiseAnisotropy', 'noiseBlend',
        'curlStrength', 'dyeNoiseAmount'].includes(k));
    const allLocked = motionKeys.every(k => this.lockedKeys.has(k));
    if (allLocked) {
      motionKeys.forEach(k => this.lockedKeys.delete(k));
      console.log('BO: Motion UNLOCKED');
    } else {
      motionKeys.forEach(k => this.lockedKeys.add(k));
      console.log('BO: Motion LOCKED');
    }
    this.updateOverlay();
    return !allLocked;
  }

  updateOverlay() {
    const N = this.ratings.length;
    const countEl = document.getElementById('boCount');
    const phaseEl = document.getElementById('boPhase');
    if (countEl) countEl.textContent = `Ratings: ${N}`;
    if (phaseEl) {
      const pending = [];
      if (this._pendingMovement === null) pending.push('motion');
      if (this._pendingColor === null) pending.push('color');
      let phase = N < 5 ? 'exploring' : 'GP active';
      phase += this.pca ? ` | PCA ${this.pca.nK}D` : ` | raw ${D}D`;
      if (pending.length > 0 && this.rateMode) {
        phase += ' | rate: ' + pending.join(', ');
      }
      phaseEl.textContent = phase;
    }

    // Update touch rating button highlights
    this._updateTouchRatingUI();
  }

  _updateTouchRatingUI() {
    const motionBtns = document.querySelectorAll('.bo-motion-btn');
    const colorBtns = document.querySelectorAll('.bo-color-btn');
    motionBtns.forEach(btn => {
      btn.classList.toggle('rated', this._pendingMovement !== null);
      btn.classList.toggle('selected', parseInt(btn.dataset.score) === this._pendingMovement);
    });
    colorBtns.forEach(btn => {
      btn.classList.toggle('rated', this._pendingColor !== null);
      btn.classList.toggle('selected', parseInt(btn.dataset.score) === this._pendingColor);
    });
  }

  clearData() {
    this.ratings = [];
    this.model = null;
    this._retrainCounter = 0;
    this._pendingMovement = null;
    this._pendingColor = null;
    localStorage.removeItem(RATINGS_KEY);
    fetch('/api/ratings', { method: 'DELETE' }).catch(() => {});
    this.updateOverlay();
    this._resetFlash();
    console.log('BO: All rating data cleared');
  }

  // ─── Run Management ─────────────────────────────────────────────────

  async saveRun(name) {
    // Save to localStorage
    const runs = JSON.parse(localStorage.getItem('dataorb-runs') || '{}');
    runs[name] = this.ratings;
    localStorage.setItem('dataorb-runs', JSON.stringify(runs));
    // Try server too
    try {
      const res = await fetch('/api/runs/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      return res.ok;
    } catch { return true; } // localStorage save succeeded
  }

  async loadRun(name) {
    // Try localStorage first
    const runs = JSON.parse(localStorage.getItem('dataorb-runs') || '{}');
    if (runs[name]) {
      this.ratings = runs[name];
      this.model = null;
      this._retrainCounter = 0;
      this._pendingMovement = null;
      this._pendingColor = null;
      if (this.ratings.length >= 5) this.retrain();
      this.updateOverlay();
      this._resetFlash();
      return true;
    }
    // Try server
    try {
      const res = await fetch('/api/runs/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) return false;
      this.ratings = await res.json();
      this.model = null;
      this._retrainCounter = 0;
      if (this.ratings.length >= 5) this.retrain();
      this.updateOverlay();
      this._resetFlash();
      return true;
    } catch { return false; }
  }

  async listRuns() {
    const localRuns = Object.keys(JSON.parse(localStorage.getItem('dataorb-runs') || '{}'));
    try {
      const res = await fetch('/api/runs');
      const serverRuns = res.ok ? await res.json() : [];
      return [...new Set([...localRuns, ...serverRuns])];
    } catch { return localRuns; }
  }

  async deleteRun(name) {
    const runs = JSON.parse(localStorage.getItem('dataorb-runs') || '{}');
    delete runs[name];
    localStorage.setItem('dataorb-runs', JSON.stringify(runs));
    try {
      await fetch('/api/runs/delete', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
    } catch {}
    return true;
  }
}

// ─── Utility ─────────────────────────────────────────────────────────────────

/** Box-Muller transform for Gaussian random numbers. */
function gaussianRandom() {
  let u, v, s;
  do {
    u = Math.random() * 2 - 1;
    v = Math.random() * 2 - 1;
    s = u * u + v * v;
  } while (s >= 1 || s === 0);
  return u * Math.sqrt(-2 * Math.log(s) / s);
}
