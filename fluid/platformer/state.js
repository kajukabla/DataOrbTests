// state.js — Game state, palettes, noise types, and level data

export const state = {
  // Game-specific
  // Jet controls
  jetForce: 4.0,
  jetSpeed: 0.8,
  jetDuration: 0.5,
  jetRadius: 0.10,
  jetDyeIntensity: 1.0,
  jetDrag: 2.2,
  effectorStrength: 1.0,
  repelRadius: 0.08,
  repelForce: 0.5,
  fluidGravity: 0,
  platformBoundaries: true,

  // Simulation
  simSpeed: 1.0,
  velDissipation: 0.998,
  dyeDissipation: 0.9995,
  maccormack: 0,
  pressureIters: 30,
  pressureDecay: 0.8,
  curlStrength: 5,
  dyeSoftCap: 1,
  dyeCeiling: 1.2,

  // Colors (linear 0-1 RGB)
  baseColor: [1.0, 0.55, 0.1],
  accentColor: [0.15, 0.3, 0.8],
  glitterColor: [1.0, 1.0, 1.0],
  glitterAccent: [0.3, 0.5, 1.0],
  tipColor: [1.0, 0.9, 0.5],
  glitterTip: [1.0, 0.95, 0.8],
  sheenColor: [1.0, 0.9, 0.7],
  colorBlend: 0.5,
  paletteIndex: -1,

  // Colormap
  colormapMode: 2,
  colorSource: 0,
  colorGain: 0.5,

  // Noise Field
  noiseAmount: 0.0,
  noiseType: 0,
  noiseBehavior: 0,
  noiseMapping: 0,
  noiseFrequency: 0.5,
  noiseSpeed: 0.5,
  noiseWarp: 0.35,
  noiseSharpness: 0.5,
  noiseAnisotropy: 0.5,
  noiseBlend: 0.5,
  noiseDyeIntensity: 0.0,
  dyeNoiseAmount: 0.0,

  // Temperature / Buoyancy
  tempAmount: 0,
  tempBuoyancy: 0.5,
  tempDissipation: 0.99,
  tempDyeHeat: 0,
  tempEdgeCool: 0,
  tempRadialMix: 0,
  tempColorShift: 0,

  // Bloom
  bloomIntensity: 0,
  bloomThreshold: 0.4,
  bloomRadius: 0.5,
};

export const PALETTES = [
  // Scientific colormaps (8-stop)
  { name: 'Viridis', colors: ['#440154','#443a83','#31688e','#21908c','#35b779','#6ece58','#b5de2b','#fde725'] },
  { name: 'Inferno', colors: ['#000004','#1b0c41','#4a0c6b','#781c6d','#a52c60','#cf4446','#ed6925','#fcffa4'] },
  { name: 'Magma', colors: ['#000004','#180f3d','#440f76','#721f81','#b5367a','#e55c30','#fba40a','#fcffa4'] },
  { name: 'Plasma', colors: ['#0d0887','#4b03a1','#7d03a8','#a82296','#cb4679','#e86825','#f89540','#f0f921'] },
  { name: 'Turbo', colors: ['#30123b','#4662d7','#36aaf9','#1ae4b6','#72fe5e','#c8ef34','#faba39','#e6550d'] },
  // Community palettes (5-stop)
  { name: 'Sunset Beach', colors: ['#69d2e7','#a7dbd8','#e0e4cc','#f38630','#fa6900'] },
  { name: 'Coral Reef', colors: ['#fe4365','#fc9d9a','#f9cdad','#c8c8a9','#83af9b'] },
  { name: 'Autumn Fire', colors: ['#ecd078','#d95b43','#c02942','#542437','#53777a'] },
  { name: 'Neon Garden', colors: ['#556270','#4ecdc4','#c7f464','#ff6b6b','#c44d58'] },
  { name: 'Deep Forest', colors: ['#e8ddcb','#cdb380','#036564','#033649','#031634'] },
  { name: 'Carnival', colors: ['#490a3d','#bd1550','#e97f02','#f8ca00','#8a9b0f'] },
  { name: 'Mint Fresh', colors: ['#594f4f','#547980','#45ada8','#9de0ad','#e5fcc2'] },
  { name: 'Desert Spice', colors: ['#00a0b0','#6a4a3c','#cc333f','#eb6841','#edc951'] },
  { name: 'Berry Blush', colors: ['#e94e77','#d68189','#c6a49a','#c6e5d9','#f4ead5'] },
  { name: 'Watermelon', colors: ['#3fb8af','#7fc7af','#dad8a7','#ff9e9d','#ff3d7f'] },
  { name: 'Ocean Deep', colors: ['#343838','#005f6b','#008c9e','#00b4cc','#00dffc'] },
  { name: 'Dusty Rose', colors: ['#413e4a','#73626e','#b38184','#f0b49e','#f7e4be'] },
  { name: 'Sunrise', colors: ['#ff4e50','#fc913a','#f9d423','#ede574','#e1f5c4'] },
  { name: 'Warm Sage', colors: ['#99b898','#fecea8','#ff847c','#e84a5f','#2a363b'] },
  { name: 'Teal Harvest', colors: ['#655643','#80bca3','#f6f7bd','#e6ac27','#bf4d28'] },
  { name: 'Electric Lime', colors: ['#00a8c6','#40c0cb','#f9f2e7','#aee239','#8fbe00'] },
  { name: 'Plum Night', colors: ['#351330','#424254','#64908a','#e8caa4','#cc2a41'] },
  { name: 'Tangerine', colors: ['#554236','#f77825','#d3ce3d','#f1efa5','#60b99a'] },
  { name: 'Crimson Gold', colors: ['#8c2318','#5e8c6a','#88a65e','#bfb35a','#f2c45a'] },
  { name: 'Candy', colors: ['#fad089','#ff9c5b','#f5634a','#ed303c','#3b8183'] },
  { name: 'Dusk', colors: ['#f8b195','#f67280','#c06c84','#6c5b7b','#355c7d'] },
  { name: 'Electric', colors: ['#d1e751','#ffffff','#000000','#4dbce9','#26ade4'] },
  { name: 'Emerald', colors: ['#1b676b','#519548','#88c425','#bef202','#eafde6'] },
  { name: 'Amber Spice', colors: ['#5e412f','#fcebb6','#78c0a8','#f07818','#f0a830'] },
  { name: 'Magenta Burst', colors: ['#bcbdac','#cfbe27','#f27435','#f02475','#3b2d38'] },
  { name: 'Dark Bloom', colors: ['#300030','#480048','#601848','#c04848','#f07241'] },
  { name: 'Pastel Dream', colors: ['#a8e6ce','#dcedc2','#ffd3b5','#ffaaa6','#ff8c94'] },
  { name: 'Noir Gold', colors: ['#3e4147','#fffedf','#dfba69','#5a2e2e','#2a2c31'] },
  { name: 'Neon Punk', colors: ['#fc354c','#29221f','#13747d','#0abfbc','#fcf7c5'] },
  { name: 'Citrus', colors: ['#cc0c39','#e6781e','#c8cf02','#f8fcc1','#1693a7'] },
  { name: 'Terracotta', colors: ['#a7c5bd','#e5ddcb','#eb7b59','#cf4647','#524656'] },
  { name: 'Cherry Blossom', colors: ['#5c323e','#a82743','#e15e32','#c0d23e','#e5f04c'] },
  { name: 'Earth Tone', colors: ['#fdf1cc','#c6d6b8','#987f69','#e3ad40','#fcd036'] },
  { name: 'Rainbow Pop', colors: ['#ff003c','#ff8a00','#fabe28','#88c100','#00c176'] },
  { name: 'Wine Country', colors: ['#d1313d','#e5625c','#f9bf76','#8eb2c5','#615375'] },
  { name: 'Deep Purple', colors: ['#111625','#341931','#571b3c','#7a1e48','#9d2053'] },
  { name: 'Volcano', colors: ['#395a4f','#432330','#853c43','#f25c5e','#ffa566'] },
  { name: 'Jungle', colors: ['#512b52','#635274','#7bb0a8','#a7dbab','#e4f5b1'] },
  { name: 'Firecracker', colors: ['#b5ac01','#ecba09','#e86e1c','#d41e45','#1b1521'] },
  { name: 'Cotton Candy', colors: ['#f1396d','#fd6081','#f3ffeb','#acc95f','#8f9924'] },
  { name: 'Sage Blush', colors: ['#b1e6d1','#77b1a9','#3d7b80','#270a33','#451a3e'] },
  { name: 'Amber Wave', colors: ['#ffab07','#e9d558','#72ad75','#0e8d94','#434d53'] },
  { name: 'Tropical', colors: ['#5cacc4','#8cd19d','#cee879','#fcb653','#ff5254'] },
  { name: 'Lava Flow', colors: ['#ccf390','#e0e05a','#f7c41f','#fc930a','#ff003d'] },
  { name: 'Moss Stone', colors: ['#f2e8c4','#98d9b6','#3ec9a7','#2b879e','#616668'] },
  { name: 'Twilight', colors: ['#2d1b33','#f36a71','#ee887a','#e4e391','#9abc8a'] },
  { name: 'Fire Dance', colors: ['#452e3c','#ff3d5a','#ffb969','#eaf27e','#3b8c88'] },
  { name: 'Aqua Terra', colors: ['#fb6900','#f63700','#004853','#007e80','#00b9bd'] },
];

export function hexToLinear(hex) {
  const r = parseInt(hex.slice(1,3), 16) / 255;
  const g = parseInt(hex.slice(3,5), 16) / 255;
  const b = parseInt(hex.slice(5,7), 16) / 255;
  const toLinear = v => v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  return [toLinear(r), toLinear(g), toLinear(b)];
}

export function applyPalette(idx) {
  if (idx < 0 || idx >= PALETTES.length) return;
  const pal = PALETTES[idx];
  const c = pal.colors.map(hexToLinear);
  if (c.length === 8) {
    state.baseColor = [...c[0]];
    state.accentColor = [...c[2]];
    state.tipColor = [...c[4]];
    state.glitterColor = [...c[6]];
    state.sheenColor = [...c[7]];
  } else {
    state.baseColor = [...c[0]];
    state.accentColor = [...c[1]];
    state.tipColor = [...c[2]];
    state.glitterColor = [...c[3]];
    state.sheenColor = [...c[4]];
  }
  state.glitterAccent = state.glitterColor.map(v => Math.min(1.0, v * 1.3 + 0.1));
  state.glitterTip = state.tipColor.map(v => Math.min(1.0, v * 1.2 + 0.15));
}

export const NOISE_TYPE_DEFS = [
  { name: 'Classic Curl', profile: { amount: 0.20, behavior: 0, mapping: 0, warp: 0.22, sharpness: 0.30, anisotropy: 0.25, blend: 0.24, frequency: 0.50, speed: 0.45 },
    labels: { noiseWarp: 'Octave Warp', noiseSharpness: 'Octave Contrast', noiseAnisotropy: 'Swirl Bias', noiseBlend: 'Micro Turbulence' } },
  { name: 'Domain-Warped', profile: { amount: 0.30, behavior: 0, mapping: 0, warp: 0.76, sharpness: 0.58, anisotropy: 0.40, blend: 0.52, frequency: 0.52, speed: 0.56 },
    labels: { noiseWarp: 'Primary Warp', noiseSharpness: 'Secondary Warp', noiseAnisotropy: 'Ribbon Flow', noiseBlend: 'Ridge Mix' } },
  { name: 'Ridged Fractal', profile: { amount: 0.24, behavior: 2, mapping: 0, warp: 0.42, sharpness: 0.78, anisotropy: 0.30, blend: 0.48, frequency: 0.56, speed: 0.46 },
    labels: { noiseWarp: 'Ridge Width', noiseSharpness: 'Ridge Exponent', noiseBlend: 'Detail Grain' }, hide: ['noiseAnisotropy'] },
  { name: 'Voronoi', profile: { amount: 0.32, behavior: 0, mapping: 0, warp: 0.38, sharpness: 0.55, anisotropy: 0.20, blend: 0.74, frequency: 0.66, speed: 0.42 },
    labels: { noiseWarp: 'Cell Scale', noiseSharpness: 'Seed Jitter', noiseBlend: 'Crack Contrast' }, hide: ['noiseAnisotropy'] },
  { name: 'Flow Rotated', profile: { amount: 0.28, behavior: 1, mapping: 0, warp: 0.52, sharpness: 0.46, anisotropy: 0.86, blend: 0.40, frequency: 0.52, speed: 0.58 },
    labels: { noiseWarp: 'Heading Rotation', noiseSharpness: 'Shear Strength', noiseAnisotropy: 'Stream Stretch', noiseBlend: 'Crossflow Mix' } },
  { name: 'Gabor Streaks', profile: { amount: 0.25, behavior: 3, mapping: 0, warp: 0.50, sharpness: 0.62, anisotropy: 0.72, blend: 0.44, frequency: 0.60, speed: 0.52 },
    labels: { noiseWarp: 'Band Frequency', noiseSharpness: 'Orientation Chaos', noiseAnisotropy: 'Crossbands', noiseBlend: 'Envelope Grain' } },
  { name: 'Hybrid Fractal', profile: { amount: 0.34, behavior: 0, mapping: 0, warp: 0.66, sharpness: 0.58, anisotropy: 0.60, blend: 0.64, frequency: 0.58, speed: 0.56 },
    labels: { noiseWarp: 'Flow Mix', noiseSharpness: 'Band Driver', noiseAnisotropy: 'Ridge Mix', noiseBlend: 'Cell Mix' } },
  { name: 'Jupiter Bands', profile: { amount: 0.44, behavior: 0, mapping: 1, warp: 0.78, sharpness: 0.58, anisotropy: 0.70, blend: 0.62, frequency: 0.55, speed: 0.60 },
    labels: { noiseWarp: 'Jet Shear', noiseSharpness: 'Storm Density', noiseAnisotropy: 'Vortex Strength', noiseBlend: 'Band Contrast' } },
];

// Build platforms dynamically based on screen size
export function buildPlatforms(canvasW, canvasH) {
  const groundY = canvasH - 60;
  return [
    // Ground — spans full width, thick enough to never leak
    { x: 0, y: groundY, w: canvasW, h: 60 },
    // Floating platforms — scaled proportionally
    { x: canvasW * 0.10, y: canvasH * 0.60, w: canvasW * 0.14, h: 16 },
    { x: canvasW * 0.36, y: canvasH * 0.50, w: canvasW * 0.16, h: 16 },
    { x: canvasW * 0.62, y: canvasH * 0.60, w: canvasW * 0.14, h: 16 },
    { x: canvasW * 0.22, y: canvasH * 0.38, w: canvasW * 0.12, h: 16 },
    { x: canvasW * 0.52, y: canvasH * 0.40, w: canvasW * 0.15, h: 16 },
    { x: canvasW * 0.04, y: canvasH * 0.27, w: canvasW * 0.12, h: 16 },
    { x: canvasW * 0.76, y: canvasH * 0.29, w: canvasW * 0.12, h: 16 },
    // Walls — full height
    { x: 0, y: 0, w: 10, h: canvasH },
    { x: canvasW - 10, y: 0, w: 10, h: canvasH },
    // Interior wall
    { x: canvasW * 0.45, y: canvasH * 0.62, w: 12, h: canvasH * 0.16 },
  ];
}

// Platforms array (rebuilt on resize via setPlatforms)
let _platforms = buildPlatforms(800, 600);
export function getPlatforms() { return _platforms; }
export function setPlatforms(p) { _platforms = p; }
