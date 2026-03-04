// ─── Bayesian Optimization Controller ────────────────────────────────────────
// RLHF-style system: user rates configs (up=good, down=bad), GP learns
// preferences, Expected Improvement suggests better configurations.

import {
  gpFit, gpPredict, expectedImprovement,
  optimizeHyperparams, latinHypercube
} from './gp.js';

// ─── Parameter Space ─────────────────────────────────────────────────────────
// All rated params: sliders + color channels (7 colors × 3 = 21 channels)

export const SLIDER_SPACE = {
  simSpeed:          { min: 0,    max: 3,     step: 0.01 },
  particleSize:      { min: 0.3,  max: 4.0,   step: 0.1  },
  glintBrightness:   { min: 0.1,  max: 5.0,   step: 0.1  },
  sizeRandomness:    { min: 0,    max: 1,     step: 0.01 },
  prismaticAmount:   { min: 0,    max: 20,    step: 0.5  },
  colorBlend:        { min: 0,    max: 1,     step: 0.01 },
  sheenStrength:     { min: 0,    max: 1,     step: 0.01 },
  clickSize:         { min: 0,    max: 1,     step: 0.01 },
  clickStrength:     { min: 0,    max: 1,     step: 0.01 },
  injectorIntensity: { min: 0,    max: 1,     step: 0.01 },
  injectorSize:      { min: 0.5,  max: 1,     step: 0.01 },
  injectorCount:     { min: 0,    max: 8,     step: 1    },
  injectorSpeed:     { min: 0,    max: 1,     step: 0.01 },
  burstCount:        { min: 0,    max: 8,     step: 1    },
  noiseAmount:       { min: 0,    max: 1,     step: 0.01 },
  noiseFrequency:    { min: 0,    max: 1,     step: 0.01 },
  noiseSpeed:        { min: 0,    max: 1,     step: 0.01 },
  curlStrength:      { min: 0,    max: 50,    step: 1    },
  splatForce:        { min: 1000, max: 20000, step: 100  },
  velDissipation:    { min: 0.99, max: 1.0,   step: 0.001 },
  dyeDissipation:    { min: 0.98, max: 1.0,   step: 0.001 },
  pressureIters:     { min: 10,   max: 60,    step: 1    },
  pressureDecay:     { min: 0,    max: 1,     step: 0.01 },
  drawBotCount:      { min: 0,    max: 4,     step: 1    },
  drawBotSpeed:      { min: 0,    max: 1,     step: 0.01 },
  drawBotSize:       { min: 0,    max: 1,     step: 0.01 },
  drawBotForce:      { min: 0,    max: 1,     step: 0.01 },
  drawBotChaos:      { min: 0,    max: 1,     step: 0.01 },
  drawBotDrift:      { min: 0,    max: 1,     step: 0.01 },
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
};

export const SLIDER_KEYS = Object.keys(SLIDER_SPACE);
export const D = SLIDER_KEYS.length; // 50

export const COLOR_KEYS = [
  'baseColor', 'accentColor', 'tipColor', 'glitterColor',
  'glitterAccent', 'glitterTip', 'sheenColor'
];

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
  const x = new Float64Array(D);
  for (let i = 0; i < D; i++) {
    const key = SLIDER_KEYS[i];
    const s = SLIDER_SPACE[key];
    x[i] = (getStateVal(state, key) - s.min) / (s.max - s.min);
    if (x[i] < 0) x[i] = 0;
    if (x[i] > 1) x[i] = 1;
  }
  return x;
}

/** Apply [0,1]^D normalized vector → state, snapping to step sizes. */
export function normalizedToState(x, state) {
  for (let i = 0; i < D; i++) {
    const key = SLIDER_KEYS[i];
    const s = SLIDER_SPACE[key];
    let val = s.min + x[i] * (s.max - s.min);
    val = Math.round(val / s.step) * s.step;
    if (val < s.min) val = s.min;
    if (val > s.max) val = s.max;
    setStateVal(state, key, val);
  }
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

/**
 * Convert ratings array to GP training data.
 * Returns { X: Float64Array(N*D), y: Float64Array(N), N, D }
 */
function ratingsToTrainingData(ratings) {
  const N = ratings.length;
  const X = new Float64Array(N * D);
  const y = new Float64Array(N);

  for (let i = 0; i < N; i++) {
    const r = ratings[i];
    // Reconstruct normalized vector from stored params
    for (let j = 0; j < D; j++) {
      const key = SLIDER_KEYS[j];
      const s = SLIDER_SPACE[key];
      if (r.params[key] !== undefined) {
        X[i * D + j] = (r.params[key] - s.min) / (s.max - s.min);
      } else {
        X[i * D + j] = 0.5; // fallback
      }
    }
    y[i] = r.rating; // 0 or 1
  }

  return { X, y, N, D };
}

// ─── BOController Class ──────────────────────────────────────────────────────

export class BOController {
  constructor() {
    this.ratings = [];  // populated by create()
    this.examples = [];
    this.model = null;
    this.rateMode = false;
    this.boMorphMode = false;
    this._retrainCounter = 0;
    this._morphCooldown = 0;
  }

  static async create() {
    const bo = new BOController();
    bo.ratings = await loadRatings();
    await bo.refreshExamples();
    if (bo.ratings.length >= 10) {
      console.log(`BO: Deferring initial retrain (N=${bo.ratings.length}) — will train after first frame`);
    }
    return bo;
  }

  // ─── Example Management ──────────────────────────────────────────────

  async refreshExamples() {
    try {
      const res = await fetch('/api/examples');
      if (res.ok) this.examples = await res.json();
    } catch { this.examples = []; }
  }

  async saveExample(name, state) {
    const params = {};
    for (const key of SLIDER_KEYS) params[key] = getStateVal(state, key);
    await fetch('/api/examples', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, params }),
    });
    await this.refreshExamples();
  }

  async deleteExample(name) {
    await fetch('/api/examples', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    await this.refreshExamples();
  }

  loadExample(example, state, syncAllUI) {
    for (const key of SLIDER_KEYS) {
      if (example.params[key] !== undefined) {
        setStateVal(state, key, example.params[key]);
      }
    }
    if (syncAllUI) syncAllUI();
  }

  async listExamples() {
    await this.refreshExamples();
    return this.examples;
  }

  /** Rebuild GP model from all ratings (+ examples as synthetic rating:1). */
  retrain() {
    const startMs = performance.now();
    // Prepend examples as synthetic positive ratings
    const syntheticRatings = this.examples.map(ex => ({ params: ex.params, rating: 1 }));
    const allRatings = [...syntheticRatings, ...this.ratings];
    const { X, y, N } = ratingsToTrainingData(allRatings);
    if (N < 5) { console.log(`BO: Retrain skipped — need 5+ ratings (have ${N}, ${syntheticRatings.length} examples)`); this.model = null; return; }

    // Check if all ratings are identical
    const allSame = y.every(v => v === y[0]);
    if (allSame) { console.log(`BO: Retrain skipped — all ratings identical`); this.model = null; return; }

    const useARD = N >= 30;
    try {
      const hp = optimizeHyperparams(X, y, N, D, useARD);
      this.model = gpFit(X, y, N, D, hp.lengthScales, hp.sigmaF, hp.sigmaN, hp.mu);
      if (!this.model) console.warn('BO: gpFit returned null — Cholesky failed');
      else console.log(`BO: GP model trained — N=${N}, D=${D}, ARD=${useARD}, sigmaF=${hp.sigmaF.toFixed(3)}, sigmaN=${hp.sigmaN.toFixed(3)} (${(performance.now() - startMs).toFixed(0)}ms)`);
    } catch (e) {
      console.warn(`BO: retrain failed (${(performance.now() - startMs).toFixed(0)}ms):`, e);
      this.model = null;
    }
  }

  /** Retrain every 5 ratings, or on first reaching 10. */
  maybeRetrain() {
    this._retrainCounter++;
    const N = this.ratings.length;
    if (N === 10 || (N >= 10 && this._retrainCounter >= 5)) {
      this._retrainCounter = 0;
      console.log(`BO: Retrain triggered (N=${N})`);
      setTimeout(() => this.retrain(), 0);
    } else {
      console.log(`BO: Retrain skipped (N=${N}, counter=${this._retrainCounter}/5)`);
    }
  }

  /**
   * Get next parameter suggestion.
   * Phase 1 (<10 ratings): uniform random
   * Phase 2 (10-30): EI with isotropic kernel, xi=0.01
   * Phase 3 (30+): EI with ARD kernel, xi=0.001
   */
  getNextParams() {
    const N = this.ratings.length;

    if (N < 10 || !this.model) {
      console.log(`BO: Next suggestion — Phase 1 (random, N=${N})`);
      return randomNormalized();
    }

    const xi = N < 30 ? 0.01 : 0.001;
    console.log(`BO: Next suggestion — Phase ${N < 30 ? 2 : 3} (EI, xi=${xi})`);
    const nCandidates = N < 30 ? 1000 : 1500;

    return this._argmaxEI(nCandidates, xi);
  }

  /** Pure exploitation: return argmax of posterior mean. */
  getBestParams() {
    if (!this.model) return randomNormalized();

    const candidates = latinHypercube(2000, D);
    let bestMean = -Infinity;
    let bestX = null;
    const xStar = new Float64Array(D);

    for (let i = 0; i < 2000; i++) {
      for (let j = 0; j < D; j++) xStar[j] = candidates[i * D + j];
      const { mean } = gpPredict(this.model, xStar);
      if (mean > bestMean) {
        bestMean = mean;
        bestX = new Float64Array(xStar);
      }
    }
    return bestX || randomNormalized();
  }

  /**
   * Rate current configuration and advance to next.
   * @param {object} state - the fluid sim state object
   * @param {number} rating - 0 (bad) or 1 (good)
   * @param {function} syncAllUI - callback to sync UI after applying new params
   */
  rate(state, rating, syncAllUI) {
    // Snapshot current params (including color channels)
    const params = {};
    for (const key of SLIDER_KEYS) params[key] = getStateVal(state, key);

    this.ratings.push({ params, rating, timestamp: Date.now() });
    console.log(`BO: Rating ${rating === 1 ? '👍' : '👎'} recorded (${this.ratings.length} total)`);
    saveRatings(this.ratings);
    this.maybeRetrain();

    // Flash indicator
    this.flashRating(rating);

    // Apply next config (colors are now in the BO vector, no separate randomize)
    const nextX = this.getNextParams();
    normalizedToState(nextX, state);
    if (syncAllUI) syncAllUI();
    this.updateOverlay();
  }

  /**
   * Get morph target for BO-guided auto-morph.
   * 70% exploit (noisy best), 30% explore (EI).
   */
  getMorphTarget() {
    if (!this.model) return randomNormalized();

    let x;
    if (Math.random() < 0.7) {
      // Exploit: best + small noise
      x = this.getBestParams();
      for (let i = 0; i < D; i++) {
        x[i] += (Math.random() - 0.5) * 0.1;
        if (x[i] < 0) x[i] = 0;
        if (x[i] > 1) x[i] = 1;
      }
    } else {
      // Explore: EI
      x = this._argmaxEI(1000, 0.01);
    }
    return x;
  }

  /** Sample LHS candidates, return argmax EI. */
  _argmaxEI(nCandidates, xi) {
    if (!this.model) return randomNormalized();

    // Find best observed value
    const { y, N } = ratingsToTrainingData(this.ratings);
    let fBest = -Infinity;
    for (let i = 0; i < N; i++) if (y[i] > fBest) fBest = y[i];

    const candidates = latinHypercube(nCandidates, D);
    let bestEI = -Infinity;
    let bestX = null;
    const xStar = new Float64Array(D);

    for (let i = 0; i < nCandidates; i++) {
      for (let j = 0; j < D; j++) xStar[j] = candidates[i * D + j];
      const { mean, variance } = gpPredict(this.model, xStar);
      const ei = expectedImprovement(mean, variance, fBest, xi);
      if (ei > bestEI) {
        bestEI = ei;
        bestX = new Float64Array(xStar);
      }
    }
    return bestX || randomNormalized();
  }

  // ─── UI ──────────────────────────────────────────────────────────────

  /** Flash the up/down rating indicator. */
  flashRating(rating) {
    const el = document.getElementById('boFlash');
    if (!el) return;
    el.textContent = rating === 1 ? '\u2191' : '\u2193';
    el.style.color = rating === 1 ? '#4f4' : '#f44';
    el.style.opacity = '1';
    setTimeout(() => { el.style.opacity = '0'; }, 400);
  }

  /** Update the overlay count and phase text. */
  updateOverlay() {
    const N = this.ratings.length;
    const countEl = document.getElementById('boCount');
    const phaseEl = document.getElementById('boPhase');
    if (countEl) countEl.textContent = `Ratings: ${N}`;
    if (phaseEl) {
      if (N < 10) phaseEl.textContent = 'exploring';
      else if (N < 30) phaseEl.textContent = 'EI active';
      else phaseEl.textContent = 'ARD active';
    }
  }

  /** Clear all ratings and model. */
  clearData() {
    this.ratings = [];
    this.model = null;
    this._retrainCounter = 0;
    fetch('/api/ratings', { method: 'DELETE' }).catch(() => {});
    this.updateOverlay();
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
    this.ratings = await res.json();
    this.model = null;
    this._retrainCounter = 0;
    if (this.ratings.length >= 10) {
      try { this.retrain(); } catch (e) { console.warn('BO: retrain after load failed:', e); }
    }
    this.updateOverlay();
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
