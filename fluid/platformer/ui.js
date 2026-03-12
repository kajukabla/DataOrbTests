import { state, PALETTES, NOISE_TYPE_DEFS, applyPalette } from './state.js';

/* ── storage for sync ─────────────────────────────────────── */
const sliderRefs = [];
const selectRefs = [];
const colorRefs  = [];

/* ── color helpers (linear ↔ sRGB hex) ────────────────────── */
function linearToHex(c) {
  const toSRGB = v => v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
  const r = Math.round(Math.min(1, Math.max(0, toSRGB(c[0]))) * 255);
  const g = Math.round(Math.min(1, Math.max(0, toSRGB(c[1]))) * 255);
  const b = Math.round(Math.min(1, Math.max(0, toSRGB(c[2]))) * 255);
  return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
}

function hexToLinear(hex) {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const toLinear = v => v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  return [toLinear(r), toLinear(g), toLinear(b)];
}

/* ── core wiring functions ────────────────────────────────── */
function wireSlider(id, key, fmt) {
  const el    = document.getElementById(id);
  const valEl = document.getElementById(id + 'Val');
  if (!el) return;
  const format = fmt || (v => {
    const n = parseFloat(v);
    return n === Math.floor(n)
      ? n.toString()
      : n.toFixed(el.step && el.step.includes('.') ? el.step.split('.')[1].length : 2);
  });
  el.value = state[key];
  if (valEl) valEl.textContent = format(state[key]);
  el.addEventListener('input', () => {
    state[key] = parseFloat(el.value);
    if (valEl) valEl.textContent = format(state[key]);
  });
  sliderRefs.push({ el, valEl, key, format });
}

function wireSelect(id, key) {
  const el = document.getElementById(id);
  if (!el) return;
  el.value = state[key];
  el.addEventListener('change', () => {
    state[key] = parseInt(el.value);
  });
  selectRefs.push({ el, key });
}

function wireColor(id, key) {
  const el = document.getElementById(id);
  if (!el) return;
  el.value = linearToHex(state[key]);
  el.addEventListener('input', () => {
    state[key] = hexToLinear(el.value);
  });
  colorRefs.push({ el, key });
}

/* ── sync all UI controls to current state ────────────────── */
function syncAllUI() {
  for (const { el, valEl, key, format } of sliderRefs) {
    el.value = state[key];
    if (valEl) valEl.textContent = format(state[key]);
  }
  for (const { el, key } of selectRefs) {
    el.value = state[key];
  }
  for (const { el, key } of colorRefs) {
    el.value = linearToHex(state[key]);
  }
}

/* ── noise type profile switching ─────────────────────────── */
function applyNoiseProfile(typeIndex) {
  const def = NOISE_TYPE_DEFS[typeIndex];
  if (!def) return;
  const p = def.profile;
  state.noiseAmount     = p.amount;
  state.noiseBehavior   = p.behavior;
  state.noiseMapping    = p.mapping;
  state.noiseWarp       = p.warp;
  state.noiseSharpness  = p.sharpness;
  state.noiseAnisotropy = p.anisotropy;
  state.noiseBlend      = p.blend;
  state.noiseFrequency  = p.frequency;
  state.noiseSpeed      = p.speed;

  // Update labels
  const labels = def.labels || {};
  for (const [ctrlId, label] of Object.entries(labels)) {
    const labelEl = document.getElementById(ctrlId + 'Label');
    if (labelEl) labelEl.textContent = label;
  }

  // Show/hide controls
  const hideSet = new Set(def.hide || []);
  for (const name of ['noiseWarp', 'noiseSharpness', 'noiseAnisotropy', 'noiseBlend']) {
    const group = document.getElementById(name + 'Group');
    if (group) group.style.display = hideSet.has(name) ? 'none' : '';
  }

  syncAllUI();
}

/* ── preset system ────────────────────────────────────────── */
const PRESET_KEYS = [
  'shootForce', 'shootRadius', 'shootRate', 'dyeAmount', 'effectorStrength', 'repelForce', 'repelRadius', 'fluidGravity', 'platformBoundaries',
  'simSpeed', 'velDissipation', 'dyeDissipation', 'maccormack', 'pressureIters', 'pressureDecay',
  'curlStrength', 'dyeSoftCap', 'dyeCeiling',
  'baseColor', 'accentColor', 'glitterColor', 'glitterAccent', 'tipColor', 'glitterTip', 'sheenColor',
  'colorBlend', 'paletteIndex',
  'colormapMode', 'colorSource', 'colorGain',
  'noiseAmount', 'noiseType', 'noiseBehavior', 'noiseMapping', 'noiseFrequency', 'noiseSpeed',
  'noiseWarp', 'noiseSharpness', 'noiseAnisotropy', 'noiseBlend', 'noiseDyeIntensity', 'dyeNoiseAmount',
  'tempAmount', 'tempBuoyancy', 'tempDissipation', 'tempDyeHeat', 'tempEdgeCool', 'tempRadialMix', 'tempColorShift',
  'clickSize', 'clickStrength',
  'bloomIntensity', 'bloomThreshold', 'bloomRadius',
];

function savePreset(name) {
  const presets = JSON.parse(localStorage.getItem('platformer-presets') || '{}');
  const data = {};
  for (const key of PRESET_KEYS) {
    data[key] = Array.isArray(state[key]) ? [...state[key]] : state[key];
  }
  presets[name] = data;
  localStorage.setItem('platformer-presets', JSON.stringify(presets));
  refreshPresetList();
}

function loadPreset(name) {
  const presets = JSON.parse(localStorage.getItem('platformer-presets') || '{}');
  const data = presets[name];
  if (!data) return;
  for (const key of PRESET_KEYS) {
    if (key in data) {
      state[key] = Array.isArray(data[key]) ? [...data[key]] : data[key];
    }
  }
  syncAllUI();
}

function deletePreset(name) {
  const presets = JSON.parse(localStorage.getItem('platformer-presets') || '{}');
  delete presets[name];
  localStorage.setItem('platformer-presets', JSON.stringify(presets));
  refreshPresetList();
}

function refreshPresetList() {
  const select = document.getElementById('presetList');
  const presets = JSON.parse(localStorage.getItem('platformer-presets') || '{}');
  select.innerHTML = '<option value="">-- select --</option>';
  for (const name of Object.keys(presets).sort()) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  }
}

/* ── settings panel toggle ────────────────────────────────── */
function initPanelToggle() {
  const panel  = document.getElementById('settingsPanel');
  const toggle = document.getElementById('settingsToggle');
  toggle.addEventListener('click', () => panel.classList.toggle('open'));
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') panel.classList.toggle('open');
  });
}

/* ── main export ──────────────────────────────────────────── */
export function initUI() {
  initPanelToggle();

  // Game Controls
  wireSlider('shootForce', 'shootForce');
  wireSlider('shootRadius', 'shootRadius');
  wireSlider('shootRate', 'shootRate');
  wireSlider('dyeAmount', 'dyeAmount');
  wireSlider('effectorStrength', 'effectorStrength');
  wireSlider('repelForce', 'repelForce');
  wireSlider('repelRadius', 'repelRadius');
  wireSlider('fluidGravity', 'fluidGravity');
  // platformBoundaries checkbox
  const pbEl = document.getElementById('platformBoundaries');
  if (pbEl) {
    pbEl.checked = state.platformBoundaries;
    pbEl.addEventListener('change', () => { state.platformBoundaries = pbEl.checked; });
  }

  // Simulation
  wireSlider('simSpeed', 'simSpeed');
  wireSlider('velDissipation', 'velDissipation');
  wireSlider('dyeDissipation', 'dyeDissipation');
  wireSlider('maccormack', 'maccormack');
  wireSlider('pressureIters', 'pressureIters');
  wireSlider('pressureDecay', 'pressureDecay');
  wireSlider('curlStrength', 'curlStrength');
  wireSlider('dyeCeiling', 'dyeCeiling');
  wireSelect('dyeSoftCap', 'dyeSoftCap');

  // Colors
  wireColor('baseColor', 'baseColor');
  wireColor('accentColor', 'accentColor');
  wireColor('tipColor', 'tipColor');
  wireColor('glitterColor', 'glitterColor');
  wireColor('glitterAccent', 'glitterAccent');
  wireColor('glitterTip', 'glitterTip');
  wireColor('sheenColor', 'sheenColor');
  wireSlider('colorBlend', 'colorBlend');

  // Colormap
  wireSelect('colormapMode', 'colormapMode');
  wireSelect('colorSource', 'colorSource');
  wireSlider('colorGain', 'colorGain');

  // Noise
  wireSlider('noiseAmount', 'noiseAmount');
  wireSelect('noiseType', 'noiseType');
  wireSelect('noiseMapping', 'noiseMapping');
  wireSlider('noiseBehavior', 'noiseBehavior', v => {
    const n = Math.round(parseFloat(v));
    if (n === 0) return '0 None';
    if (n === 1) return '1 Mirror';
    return n + '-fold';
  });
  wireSlider('noiseFrequency', 'noiseFrequency');
  wireSlider('noiseSpeed', 'noiseSpeed');
  wireSlider('noiseWarp', 'noiseWarp');
  wireSlider('noiseSharpness', 'noiseSharpness');
  wireSlider('noiseAnisotropy', 'noiseAnisotropy');
  wireSlider('noiseBlend', 'noiseBlend');
  wireSlider('noiseDyeIntensity', 'noiseDyeIntensity');
  wireSlider('dyeNoiseAmount', 'dyeNoiseAmount');

  // Noise type profile switching
  document.getElementById('noiseType').addEventListener('change', () => {
    applyNoiseProfile(state.noiseType);
  });

  // Temperature
  wireSlider('tempAmount', 'tempAmount');
  wireSlider('tempBuoyancy', 'tempBuoyancy');
  wireSlider('tempDissipation', 'tempDissipation');
  wireSlider('tempDyeHeat', 'tempDyeHeat');
  wireSlider('tempEdgeCool', 'tempEdgeCool');
  wireSlider('tempRadialMix', 'tempRadialMix');
  wireSlider('tempColorShift', 'tempColorShift');

  // Interaction
  wireSlider('clickSize', 'clickSize');
  wireSlider('clickStrength', 'clickStrength');

  // Bloom
  wireSlider('bloomIntensity', 'bloomIntensity');
  wireSlider('bloomThreshold', 'bloomThreshold');
  wireSlider('bloomRadius', 'bloomRadius');

  // Palette (with special format)
  wireSlider('paletteIndex', 'paletteIndex', v => {
    const idx = Math.round(parseFloat(v));
    return idx < 0 ? 'Manual' : (PALETTES[idx]?.name || idx);
  });
  document.getElementById('paletteIndex').addEventListener('input', () => {
    const idx = Math.round(state.paletteIndex);
    if (idx >= 0) {
      applyPalette(idx);
      syncAllUI();
    }
  });

  // Presets
  refreshPresetList();
  document.getElementById('savePreset').addEventListener('click', () => {
    const name = document.getElementById('presetName').value.trim();
    if (name) savePreset(name);
  });
  document.getElementById('loadPreset').addEventListener('click', () => {
    const name = document.getElementById('presetList').value;
    if (name) loadPreset(name);
  });
  document.getElementById('deletePreset').addEventListener('click', () => {
    const name = document.getElementById('presetList').value;
    if (name) deletePreset(name);
  });

  // Double-click on preset list to load
  document.getElementById('presetList').addEventListener('dblclick', () => {
    const name = document.getElementById('presetList').value;
    if (name) loadPreset(name);
  });
}
