// ─── Bayesian Optimization Controller ────────────────────────────────────────
// Dual-GP RLHF: separate models for movement and color preferences.
// Movement: ↑/↓ arrows   Color: →/← arrows
// Both must be rated before advancing to next generation.

import {
  gpFit, gpPredict, expectedImprovement,
  optimizeHyperparams, latinHypercube
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
  // Dye-coupled noise
  noiseDyeIntensity: { min: 0,    max: 1,     step: 0.01 },
  dyeNoiseAmount:    { min: 0,    max: 0.15,  step: 0.001 },
  // Temperature/buoyancy
  tempAmount:        { min: 0,    max: 1,     step: 0.01 },
  tempBuoyancy:      { min: 0,    max: 3,     step: 0.01 },
  tempDissipation:   { min: 0.95, max: 1.0,   step: 0.001 },
  tempDyeHeat:       { min: 0,    max: 3,     step: 0.01 },
  tempEdgeCool:      { min: 0,    max: 3,     step: 0.01 },
  tempRadialMix:     { min: 0,    max: 1,     step: 0.01 },
  tempColorShift:    { min: 0,    max: 3,     step: 0.01 },
  // Mood
  moodAmount:        { min: 0,    max: 1,     step: 0.01 },
  moodSpeed:         { min: 0,    max: 1,     step: 0.01 },
  paletteIndex:      { min: -1,   max: 49,    step: 1 },
  // Material properties
  metallic:          { min: 0,    max: 1,     step: 0.01 },
  roughness:         { min: 0,    max: 1,     step: 0.01 },
  // Color channels (7 colors × 3 channels = 21 dims)
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
  // Bloom
  bloomIntensity:    { min: 0, max: 1, step: 0.01 },
  bloomThreshold:    { min: 0, max: 2, step: 0.01 },
  bloomRadius:       { min: 0, max: 1, step: 0.01 },
  // Face tracking
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
  // Transfer function
  colormapMode:      { min: 0, max: 17, step: 1    },
  colorSource:       { min: 0, max: 8, step: 1    },
  colorGain:         { min: 0, max: 1, step: 0.01 },
  colormapCompress:  { min: 0, max: 1, step: 1    },
  // Particle overdraw cap
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

// Sensible defaults for params that shouldn't reset to min when loading old presets
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
  { noiseWarp: 0.22, noiseSharpness: 0.30, noiseAnisotropy: 0.25, noiseBlend: 0.24 }, // Classic
  { noiseWarp: 0.76, noiseSharpness: 0.58, noiseAnisotropy: 0.40, noiseBlend: 0.52 }, // Domain-Warped
  { noiseWarp: 0.42, noiseSharpness: 0.78, noiseAnisotropy: 0.30, noiseBlend: 0.48 }, // Ridged
  { noiseWarp: 0.38, noiseSharpness: 0.55, noiseAnisotropy: 0.20, noiseBlend: 0.74 }, // Voronoi
  { noiseWarp: 0.52, noiseSharpness: 0.46, noiseAnisotropy: 0.86, noiseBlend: 0.40 }, // Flow
  { noiseWarp: 0.50, noiseSharpness: 0.62, noiseAnisotropy: 0.72, noiseBlend: 0.44 }, // Gabor
  { noiseWarp: 0.66, noiseSharpness: 0.58, noiseAnisotropy: 0.60, noiseBlend: 0.64 }, // Hybrid
  { noiseWarp: 0.78, noiseSharpness: 0.58, noiseAnisotropy: 0.70, noiseBlend: 0.62 }, // Jupiter
];
const NOISE_ACTIVE_BY_TYPE = [
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: true, noiseBlend: true },  // Classic
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: true, noiseBlend: true },  // Domain-Warped
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: false, noiseBlend: true }, // Ridged
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: false, noiseBlend: true }, // Voronoi
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: true, noiseBlend: true },  // Flow
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: true, noiseBlend: true },  // Gabor
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: true, noiseBlend: true },  // Hybrid
  { noiseWarp: true, noiseSharpness: true, noiseAnisotropy: true, noiseBlend: true },  // Jupiter
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

// ─── Parameter Group Indices ────────────────────────────────────────────────
// Split SLIDER_KEYS into motion vs color for dual-GP training

const COLOR_PATTERN = /Color_|Accent_|Tip_|sheen.*_/i;

export const COLOR_INDICES = [];
export const MOTION_INDICES = [];
for (let i = 0; i < D; i++) {
  if (COLOR_PATTERN.test(SLIDER_KEYS[i])) {
    COLOR_INDICES.push(i);
  } else {
    MOTION_INDICES.push(i);
  }
}
export const D_MOTION = MOTION_INDICES.length;
export const D_COLOR = COLOR_INDICES.length;

// ─── Color channel helpers ──────────────────────────────────────────────────

/** Get state value for a key (handles color channels like baseColor_0). */
function getStateVal(state, key) {
  const m = key.match(/^(.+)_([012])$/);
  if (m && Array.isArray(state[m[1]])) return state[m[1]][parseInt(m[2])];
  return state[key];
}

/** Set state value for a key (handles color channels). */
function setStateVal(state, key, val) {
  const m = key.match(/^(.+)_([012])$/);
  if (m && Array.isArray(state[m[1]])) { state[m[1]][parseInt(m[2])] = val; return; }
  state[key] = val;
}

// ─── Normalization ───────────────────────────────────────────────────────────

/** Extract state → [0,1]^D normalized vector (Float64Array). */
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

/** Apply [0,1]^D normalized vector → state, snapping to step sizes.
 *  @param {Set} [lockedKeys] — keys to skip (preserve current state values)
 */
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
  // Keep noise control dimensions coherent with selected noise type.
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
  // Clamp critical params so the sim is always visually active
  if (state.simSpeed < 0.2) state.simSpeed = 0.2;
}

/** Generate a random normalized vector in [0,1]^D. */
function randomNormalized() {
  const x = new Float64Array(D);
  for (let i = 0; i < D; i++) x[i] = Math.random();
  return x;
}

// ─── Data Store ──────────────────────────────────────────────────────────────

const MAX_RATINGS = 2000;

async function loadRatings() {
  try {
    const res = await fetch('/api/ratings');
    if (!res.ok) return [];
    const arr = await res.json();
    return Array.isArray(arr) ? arr : [];
  } catch { return []; }
}

async function saveRatings(ratings) {
  // Cap at MAX_RATINGS (keep most recent)
  if (ratings.length > MAX_RATINGS) {
    ratings = ratings.slice(ratings.length - MAX_RATINGS);
  }
  try {
    await fetch('/api/ratings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ratings }),
    });
  } catch (e) { console.warn('BO: save failed:', e); }
}

/** Purge old-format or out-of-range ratings. */
function purgeStaleRatings(ratings) {
  const before = ratings.length;
  const valid = ratings.filter(r => {
    // Must have dual-rating format
    if (r.movementRating === undefined || r.colorRating === undefined) return false;
    // Check params are within current SLIDER_SPACE ranges (with 10% tolerance)
    const p = r.params;
    if (!p) return false;
    for (const key of SLIDER_KEYS) {
      if (p[key] === undefined) continue;
      const s = SLIDER_SPACE[key];
      const range = s.max - s.min;
      const tol = range * 0.1;
      if (p[key] < s.min - tol || p[key] > s.max + tol) return false;
    }
    return true;
  });
  const purged = before - valid.length;
  if (purged > 0) {
    console.log(`BO: Purged ${purged} stale ratings (${valid.length} remaining)`);
  }
  return valid;
}

// ─── Dual-GP Training Data ──────────────────────────────────────────────────

/** Extract motion-only training data from ratings. */
function ratingsToMotionData(ratings) {
  const N = ratings.length;
  const X = new Float64Array(N * D_MOTION);
  const y = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const r = ratings[i];
    const p = { ...r.params };
    canonicalizeNoiseFields(p);
    for (let j = 0; j < D_MOTION; j++) {
      const key = SLIDER_KEYS[MOTION_INDICES[j]];
      const s = SLIDER_SPACE[key];
      if (p[key] !== undefined) {
        X[i * D_MOTION + j] = (p[key] - s.min) / (s.max - s.min);
      } else {
        X[i * D_MOTION + j] = 0.5;
      }
    }
    y[i] = r.movementRating;
  }
  return { X, y, N, D: D_MOTION };
}

/** Extract color-only training data from ratings. */
function ratingsToColorData(ratings) {
  const N = ratings.length;
  const X = new Float64Array(N * D_COLOR);
  const y = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const r = ratings[i];
    const p = { ...r.params };
    canonicalizeNoiseFields(p);
    for (let j = 0; j < D_COLOR; j++) {
      const key = SLIDER_KEYS[COLOR_INDICES[j]];
      const s = SLIDER_SPACE[key];
      if (p[key] !== undefined) {
        X[i * D_COLOR + j] = (p[key] - s.min) / (s.max - s.min);
      } else {
        X[i * D_COLOR + j] = 0.5;
      }
    }
    y[i] = r.colorRating;
  }
  return { X, y, N, D: D_COLOR };
}

// ─── BOController Class ──────────────────────────────────────────────────────

export class BOController {
  constructor() {
    this.ratings = [];
    this.examples = [];
    this.motionModel = null;
    this.colorModel = null;
    this.rateMode = false;
    this.boMorphMode = false;
    this._retrainCounter = 0;
    this._morphCooldown = 0;
    this._suggestionCount = 0;
    this.lockedKeys = new Set();
    // Pending ratings for current generation
    this._pendingMovement = null;  // null = not yet rated
    this._pendingColor = null;
    this._state = null;            // ref set by rate calls
    this._syncAllUI = null;
    this._statePostNormalize = null;
  }

  setStatePostNormalize(fn) {
    this._statePostNormalize = typeof fn === 'function' ? fn : null;
  }

  _runStatePostNormalize(state, source) {
    if (!this._statePostNormalize || !state) return;
    try {
      this._statePostNormalize(state, source);
    } catch (e) {
      console.warn('BO: state post-normalize hook failed:', e);
    }
  }

  static async create() {
    const bo = new BOController();
    const raw = await loadRatings();
    bo.ratings = purgeStaleRatings(raw);
    // Save purged list if anything was removed
    if (bo.ratings.length !== raw.length && bo.ratings.length > 0) {
      saveRatings(bo.ratings);
    }
    await bo.refreshExamples();
    // Warm-start: inject examples as positive ratings when starting fresh
    if (bo.ratings.length === 0 && bo.examples.length > 0) {
      for (const ex of bo.examples) {
        bo.ratings.push({
          params: ex.params,
          movementRating: 1,
          colorRating: 1,
          timestamp: Date.now(),
        });
      }
      saveRatings(bo.ratings);
      console.log(`BO: Warm-started with ${bo.examples.length} example ratings`);
    }
    if (bo.ratings.length >= 10) {
      console.log(`BO: Deferring initial retrain (N=${bo.ratings.length}) — will train after first frame`);
    }
    return bo;
  }

  // ─── Example Management ──────────────────────────────────────────────

  async refreshExamples() {
    try {
      let res = await fetch('/api/examples').catch(() => null);
      if (!res || !res.ok) res = await fetch('data/examples.json').catch(() => null);
      if (res && res.ok) this.examples = await res.json();
    } catch { this.examples = []; }
  }

  async saveExample(name, state) {
    const params = {};
    for (const key of SLIDER_KEYS) params[key] = getStateVal(state, key);
    canonicalizeNoiseFields(params);
    let res = await fetch('/api/examples', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, params }),
    }).catch(() => null);
    if (!res || !res.ok) {
      // Current server doesn't support POST, try node server on 8081
      res = await fetch('http://localhost:8081/api/examples', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, params }),
      }).catch(() => null);
    }
    if (!res || !res.ok) {
      alert('Save failed — make sure the node server is running (node fluid/server.js).');
      return;
    }
    await this.refreshExamples();
  }

  async deleteExample(name) {
    try {
      await fetch('/api/examples', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
    } catch {
      await fetch('http://localhost:8081/api/examples', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
    }
    await this.refreshExamples();
  }

  loadExample(example, state, syncAllUI) {
    // Reset ALL slider params to sensible defaults before applying preset
    for (const key of SLIDER_KEYS) {
      const s = SLIDER_SPACE[key];
      const def = SLIDER_DEFAULTS[key];
      setStateVal(state, key, def !== undefined ? def : s.min);
    }
    // Apply saved preset values
    for (const key of SLIDER_KEYS) {
      if (example.params[key] !== undefined) {
        setStateVal(state, key, example.params[key]);
      }
    }
    this._runStatePostNormalize(state, 'load-example');
    if (syncAllUI) syncAllUI();
  }

  async listExamples() {
    await this.refreshExamples();
    return this.examples;
  }

  // ─── Dual-GP Retrain ──────────────────────────────────────────────────

  /** Rebuild both GP models from ratings. */
  retrain() {
    const startMs = performance.now();
    const MAX_TRAIN = 200;
    const recentRatings = this.ratings.length > MAX_TRAIN
      ? this.ratings.slice(-MAX_TRAIN)
      : this.ratings;
    const N = recentRatings.length;

    if (N < 5) {
      console.log(`BO: Retrain skipped — need 5+ ratings (have ${N})`);
      this.motionModel = null;
      this.colorModel = null;
      return;
    }

    // Train motion GP
    this.motionModel = this._trainGP(recentRatings, 'motion', ratingsToMotionData, D_MOTION);

    // Train color GP
    this.colorModel = this._trainGP(recentRatings, 'color', ratingsToColorData, D_COLOR);

    console.log(`BO: Dual retrain done (${(performance.now() - startMs).toFixed(0)}ms)`);
  }

  /** Train a single GP model for a parameter group. */
  _trainGP(ratings, label, dataFn, dims) {
    const { X, y, N, D: gD } = dataFn(ratings);

    // Check if all ratings are identical
    const allSame = y.every(v => v === y[0]);
    if (allSame) {
      console.log(`BO: ${label} GP skipped — all ratings identical`);
      return null;
    }

    const useARD = N >= 60;
    try {
      const hp = optimizeHyperparams(X, y, N, gD, useARD);
      const model = gpFit(X, y, N, gD, hp.lengthScales, hp.sigmaF, hp.sigmaN, hp.mu);
      if (!model) {
        console.warn(`BO: ${label} GP — Cholesky failed`);
        return null;
      }
      console.log(`BO: ${label} GP trained — N=${N}, D=${gD}, ARD=${useARD}, sigmaF=${hp.sigmaF.toFixed(3)}, sigmaN=${hp.sigmaN.toFixed(3)}`);
      return model;
    } catch (e) {
      console.warn(`BO: ${label} GP failed:`, e);
      return null;
    }
  }

  /** Retrain every 10 ratings, or on first reaching 10. */
  maybeRetrain() {
    this._retrainCounter++;
    const N = this.ratings.length;
    if (N === 10 || (N >= 10 && this._retrainCounter >= 10)) {
      this._retrainCounter = 0;
      console.log(`BO: Retrain triggered (N=${N})`);
      setTimeout(() => this.retrain(), 0);
    } else {
      console.log(`BO: Retrain skipped (N=${N}, counter=${this._retrainCounter}/10)`);
    }
  }

  // ─── Dual-GP Suggestions ──────────────────────────────────────────────

  /**
   * Get next parameter suggestion using dual GPs.
   * Generates motion and color suggestions independently, merges into full vector.
   */
  getNextParams() {
    const N = this.ratings.length;

    // Phase 1: random/example exploration
    if (N < 10 || (!this.motionModel && !this.colorModel)) {
      if (this.examples.length > 0 && N % 2 === 0) {
        const ex = this.examples[N % this.examples.length];
        const x = stateToNormalized(ex.params);
        for (let i = 0; i < D; i++) {
          x[i] += (Math.random() - 0.5) * 0.15;
          if (x[i] < 0) x[i] = 0;
          if (x[i] > 1) x[i] = 1;
        }
        console.log(`BO: Next suggestion — Phase 1 (example seed "${ex.name}", N=${N})`);
        return x;
      }
      console.log(`BO: Next suggestion — Phase 1 (random, N=${N})`);
      return randomNormalized();
    }

    // Force exploration every 3rd suggestion
    this._suggestionCount++;
    if (this._suggestionCount % 3 === 0) {
      console.log('BO: Next suggestion — Explore (random)');
      return randomNormalized();
    }

    const xi = N < 30 ? 0.01 : 0.05;
    const nCandidates = N < 30 ? 1000 : 500;

    // Generate motion and color suggestions independently, merge
    const fullX = randomNormalized(); // fallback base

    // Motion suggestion
    if (this.motionModel) {
      const motionX = this._argmaxEIPartial(this.motionModel, MOTION_INDICES, D_MOTION, nCandidates, xi, 'movementRating');
      for (let j = 0; j < D_MOTION; j++) {
        fullX[MOTION_INDICES[j]] = motionX[j];
      }
    }

    // Color suggestion
    if (this.colorModel) {
      const colorX = this._argmaxEIPartial(this.colorModel, COLOR_INDICES, D_COLOR, nCandidates, xi, 'colorRating');
      for (let j = 0; j < D_COLOR; j++) {
        fullX[COLOR_INDICES[j]] = colorX[j];
      }
    }

    console.log(`BO: Next suggestion — EI (motion=${!!this.motionModel}, color=${!!this.colorModel}, xi=${xi})`);
    return fullX;
  }

  /** Pure exploitation: return argmax of posterior mean (uses motion model for motion, color for color). */
  getBestParams() {
    const fullX = randomNormalized();

    if (this.motionModel) {
      const best = this._bestMeanPartial(this.motionModel, MOTION_INDICES, D_MOTION);
      for (let j = 0; j < D_MOTION; j++) fullX[MOTION_INDICES[j]] = best[j];
    }

    if (this.colorModel) {
      const best = this._bestMeanPartial(this.colorModel, COLOR_INDICES, D_COLOR);
      for (let j = 0; j < D_COLOR; j++) fullX[COLOR_INDICES[j]] = best[j];
    }

    return fullX;
  }

  // ─── Dual Rating ──────────────────────────────────────────────────────

  /**
   * Rate the movement of the current config.
   * @param {number} rating - 1 (good) or -1 (bad)
   * @param {object} state - fluid sim state
   * @param {function} syncAllUI - UI sync callback
   */
  rateMovement(rating, state, syncAllUI) {
    if (this._pendingMovement !== null) return; // already rated
    this._pendingMovement = rating;
    this._state = state;
    this._syncAllUI = syncAllUI;
    this.flashMovement(rating);
    console.log(`BO: Movement rated ${rating === 1 ? '↑' : '↓'}`);
    this.updateOverlay();
    this._tryCommit();
  }

  /**
   * Rate the color of the current config.
   * @param {number} rating - 1 (good) or -1 (bad)
   * @param {object} state - fluid sim state
   * @param {function} syncAllUI - UI sync callback
   */
  rateColor(rating, state, syncAllUI) {
    if (this._pendingColor !== null) return; // already rated
    this._pendingColor = rating;
    this._state = state;
    this._syncAllUI = syncAllUI;
    this.flashColor(rating);
    console.log(`BO: Color rated ${rating === 1 ? '→' : '←'}`);
    this.updateOverlay();
    this._tryCommit();
  }

  /** If both ratings are in, commit and advance. */
  _tryCommit() {
    if (this._pendingMovement === null || this._pendingColor === null) return;

    try {
      const state = this._state;
      const syncAllUI = this._syncAllUI;

      // Snapshot current params
      const params = {};
      for (const key of SLIDER_KEYS) params[key] = getStateVal(state, key);
      canonicalizeNoiseFields(params);

      // Store with dual ratings
      this.ratings.push({
        params,
        movementRating: this._pendingMovement,
        colorRating: this._pendingColor,
        timestamp: Date.now(),
      });

      const mStr = this._pendingMovement === 1 ? '↑' : '↓';
      const cStr = this._pendingColor === 1 ? '→' : '←';
      console.log(`BO: Rating committed ${mStr}${cStr} (${this.ratings.length} total)`);
      saveRatings(this.ratings);
      this.maybeRetrain();

      // Reset pending
      this._pendingMovement = null;
      this._pendingColor = null;

      // Get next suggestion
      let nextX;
      try {
        nextX = this.getNextParams();
      } catch (e) {
        console.warn('BO: getNextParams failed, using random:', e);
        nextX = randomNormalized();
      }

      normalizedToState(nextX, state, this.lockedKeys);
      this._runStatePostNormalize(state, 'rating-advance');
      if (syncAllUI) syncAllUI();
      this.updateOverlay();

      // Reset flash indicators after brief delay so user sees their ratings
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

  /**
   * Get morph target for BO-guided auto-morph.
   * 70% exploit (noisy best), 30% explore (EI).
   */
  getMorphTarget() {
    if (!this.motionModel && !this.colorModel) return randomNormalized();

    let x;
    if (Math.random() < 0.7) {
      x = this.getBestParams();
      for (let i = 0; i < D; i++) {
        x[i] += (Math.random() - 0.5) * 0.1;
        if (x[i] < 0) x[i] = 0;
        if (x[i] > 1) x[i] = 1;
      }
    } else {
      const xi = 0.01;
      const fullX = randomNormalized();
      if (this.motionModel) {
        const mx = this._argmaxEIPartial(this.motionModel, MOTION_INDICES, D_MOTION, 1000, xi, 'movementRating');
        for (let j = 0; j < D_MOTION; j++) fullX[MOTION_INDICES[j]] = mx[j];
      }
      if (this.colorModel) {
        const cx = this._argmaxEIPartial(this.colorModel, COLOR_INDICES, D_COLOR, 1000, xi, 'colorRating');
        for (let j = 0; j < D_COLOR; j++) fullX[COLOR_INDICES[j]] = cx[j];
      }
      x = fullX;
    }
    return x;
  }

  /** EI optimization over a subset of parameters. Returns partial normalized vector. */
  _argmaxEIPartial(model, indices, dims, nCandidates, xi, ratingKey) {
    // Find best observed value for this rating dimension
    let fBest = -Infinity;
    for (const r of this.ratings) {
      if (r[ratingKey] > fBest) fBest = r[ratingKey];
    }

    const candidates = latinHypercube(nCandidates, dims);
    let bestEI = -Infinity;
    let bestX = null;
    const xStar = new Float64Array(dims);

    for (let i = 0; i < nCandidates; i++) {
      for (let j = 0; j < dims; j++) xStar[j] = candidates[i * dims + j];
      const { mean, variance } = gpPredict(model, xStar);
      const ei = expectedImprovement(mean, variance, fBest, xi);
      if (ei > bestEI) {
        bestEI = ei;
        bestX = new Float64Array(xStar);
      }
    }

    return bestX || (() => { const x = new Float64Array(dims); for (let i = 0; i < dims; i++) x[i] = Math.random(); return x; })();
  }

  /** Argmax posterior mean over a subset of parameters. */
  _bestMeanPartial(model, indices, dims) {
    const candidates = latinHypercube(2000, dims);
    let bestMean = -Infinity;
    let bestX = null;
    const xStar = new Float64Array(dims);

    for (let i = 0; i < 2000; i++) {
      for (let j = 0; j < dims; j++) xStar[j] = candidates[i * dims + j];
      const { mean } = gpPredict(model, xStar);
      if (mean > bestMean) {
        bestMean = mean;
        bestX = new Float64Array(xStar);
      }
    }
    return bestX || (() => { const x = new Float64Array(dims); for (let i = 0; i < dims; i++) x[i] = Math.random(); return x; })();
  }

  // ─── UI ──────────────────────────────────────────────────────────────

  /** Flash the movement rating indicator. */
  flashMovement(rating) {
    const el = document.getElementById('boFlashMovement');
    if (!el) return;
    el.textContent = rating === 1 ? '\u2191' : '\u2193';
    el.style.color = rating === 1 ? '#4f4' : '#f44';
    el.style.opacity = '1';
  }

  /** Flash the color rating indicator. */
  flashColor(rating) {
    const el = document.getElementById('boFlashColor');
    if (!el) return;
    el.textContent = rating === 1 ? '\u2192' : '\u2190';
    el.style.color = rating === 1 ? '#4af' : '#fa4';
    el.style.opacity = '1';
  }

  /** Reset flash indicators for new generation. */
  _resetFlash() {
    const m = document.getElementById('boFlashMovement');
    const c = document.getElementById('boFlashColor');
    if (m) { m.textContent = '?'; m.style.color = '#555'; m.style.opacity = '0.3'; }
    if (c) { c.textContent = '?'; c.style.color = '#555'; c.style.opacity = '0.3'; }
  }

  /** Toggle locking color params (all 21 color channels). */
  toggleLockColors() {
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

  /** Toggle locking motion/dynamics params. */
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

  /** Update the overlay count and phase text. */
  updateOverlay() {
    const N = this.ratings.length;
    const countEl = document.getElementById('boCount');
    const phaseEl = document.getElementById('boPhase');
    if (countEl) countEl.textContent = `Ratings: ${N}`;
    if (phaseEl) {
      const pending = [];
      if (this._pendingMovement === null) pending.push('↑↓ movement');
      if (this._pendingColor === null) pending.push('←→ color');
      let phase = N < 10 ? 'exploring' : N < 60 ? 'EI active' : 'ARD active';
      phase += ` | Motion GP (D=${D_MOTION}) | Color GP (D=${D_COLOR})`;
      if (pending.length > 0 && this.rateMode) {
        phase += ' | rate: ' + pending.join(', ');
      }
      phaseEl.textContent = phase;
    }
  }

  /** Clear all ratings and models. */
  clearData() {
    this.ratings = [];
    this.motionModel = null;
    this.colorModel = null;
    this._retrainCounter = 0;
    this._pendingMovement = null;
    this._pendingColor = null;
    fetch('/api/ratings', { method: 'DELETE' }).catch(() => {});
    this.updateOverlay();
    this._resetFlash();
  }

  // ─── Run Management ─────────────────────────────────────────────────

  async saveRun(name) {
    const res = await fetch('/api/runs/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    return res.ok;
  }

  async loadRun(name) {
    const res = await fetch('/api/runs/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    if (!res.ok) return false;
    const raw = await res.json();
    this.ratings = purgeStaleRatings(raw);
    this.motionModel = null;
    this.colorModel = null;
    this._retrainCounter = 0;
    this._pendingMovement = null;
    this._pendingColor = null;
    if (this.ratings.length >= 10) {
      try { this.retrain(); } catch (e) { console.warn('BO: retrain after load failed:', e); }
    }
    this.updateOverlay();
    this._resetFlash();
    return true;
  }

  async listRuns() {
    try {
      const res = await fetch('/api/runs');
      return res.ok ? await res.json() : [];
    } catch { return []; }
  }

  async deleteRun(name) {
    const res = await fetch('/api/runs/delete', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    return res.ok;
  }
}
