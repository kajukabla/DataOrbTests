// ─── Configuration ───────────────────────────────────────────────────────────
const SIM_RES = 512;
const WORKGROUP = 8;
const TEX_FMT = 'rgba16float';
const PARTICLE_STRIDE = 32;       // 8 floats × 4 bytes
const PARTICLE_WG = 256;
// Shared hard containment radius for all simulation layers.
// Slightly inset from 0.37 to prevent post-process/filter edge ridges.
const SIM_SPHERE_RADIUS = 0.3665;
const VISIBLE_SPHERE_RADIUS = SIM_SPHERE_RADIUS;

const state = {
  particleCount: 4194304,   // 4M default
  particleSize: 0.9,
  sizeRandomness: 0.3,
  glintBrightness: 0.1,
  prismaticAmount: 20.0,
  baseColor: [1.0, 0.55, 0.1],
  accentColor: [0.15, 0.3, 0.8],
  glitterColor: [1.0, 1.0, 1.0],
  glitterAccent: [0.3, 0.5, 1.0],
  tipColor: [1.0, 0.9, 0.5],
  glitterTip: [1.0, 0.95, 0.8],
  colorBlend: 0.5,
  sheenStrength: 1.5,
  metallic: 0.3,
  roughness: 0.4,
  clickSize: 0.5,
  clickStrength: 0.5,
  noiseAmount: 0.18,
  noiseFrequency: 0.5,
  noiseSpeed: 0.5,
  noiseType: 0,         // 0..7 (dropdown, includes Jupiter)
  noiseBehavior: 0,     // 0=none, 1=mirror, 2+=n-fold symmetry
  noiseMapping: 0,      // 0=normal cartesian, 1=radial outflow
  noiseWarp: 0.35,
  noiseSharpness: 0.5,
  noiseAnisotropy: 0.5,
  noiseBlend: 0.5,
  burstCount: 0,
  burstBehavior: 0,
  burstForce: 0.8,
  burstForceRandomness: 0.25,
  burstSpeed: 0.6,
  burstDuration: 0.8,
  burstWidth: 0.25,
  sheenColor: [1.0, 0.9, 0.7],
  splatForce: 6000,
  curlStrength: 15,
  pressureIters: 30,
  pressureDecay: 0.8,
  velDissipation: 0.998,
  dyeDissipation: 0.993,
  // Dye-coupled noise
  noiseDyeIntensity: 0.0,
  dyeNoiseAmount: 0.0,
  bloomIntensity: 0,
  bloomThreshold: 0.4,
  bloomRadius: 0.5,
  splatRadius: 0.0015,
  simSpeed: 1.0,
  masterSpeed: 1.0,
  autoMorph: false,
  // Temperature/buoyancy
  tempAmount: 0,
  tempBuoyancy: 0.5,
  tempDissipation: 0.99,
  tempDyeTint: 0,
  // Mood lighting
  moodAmount: 0,
  moodSpeed: 0.3,
  // Palette
  paletteIndex: -1,
  // Face effector
  faceEffectorMode: 0,
  faceDebugMode: 0,
};

const FACE_TRACKER_BUNDLE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
const FACE_TRACKER_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const FACE_TRACKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
let faceVisionModuleCache = null;

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

const FACE_IDX = {
  contour: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
  jaw: [323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93],
  lips: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78],
  mouthHole: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415],
  leftBrow: [70, 63, 105, 66, 107],
  rightBrow: [300, 293, 334, 296, 336],
  leftEye: [33, 160, 158, 133, 153, 144],
  rightEye: [263, 387, 385, 362, 380, 373],
  nose: [1, 4, 6, 168, 195, 5, 98, 327],
  cheeks: [50, 123, 117, 346, 352, 280],
};

function uniqueIndices(...groups) {
  const out = [];
  const seen = new Set();
  for (const group of groups) {
    for (const idx of group) {
      if (seen.has(idx)) continue;
      seen.add(idx);
      out.push(idx);
    }
  }
  return out;
}

const FACE_DENSE_INDICES = uniqueIndices(
  FACE_IDX.contour,
  FACE_IDX.lips,
  FACE_IDX.mouthHole,
  FACE_IDX.leftEye,
  FACE_IDX.rightEye,
  FACE_IDX.leftBrow,
  FACE_IDX.rightBrow,
  FACE_IDX.nose,
  FACE_IDX.cheeks
);

const FACE_DEBUG_PATHS = [
  { points: FACE_IDX.contour, closed: true },
  { points: FACE_IDX.lips, closed: true },
  { points: FACE_IDX.mouthHole, closed: true },
  { points: FACE_IDX.leftEye, closed: true },
  { points: FACE_IDX.rightEye, closed: true },
  { points: FACE_IDX.leftBrow, closed: false },
  { points: FACE_IDX.rightBrow, closed: false },
  { points: FACE_IDX.jaw, closed: false },
  { points: FACE_IDX.nose, closed: false },
];

// ─── Color Palettes ──────────────────────────────────────────────────────────
function hexToLinear(hex) {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const toL = c => c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  return [toL(r), toL(g), toL(b)];
}

const PALETTES = [
  // 5 Scientific colormaps (8-stop)
  { name: 'Viridis', colors: ['#440154','#443a83','#31688e','#21908c','#35b779','#6ece58','#b5de2b','#fde725'].map(hexToLinear) },
  { name: 'Inferno', colors: ['#000004','#1b0c41','#4a0c6b','#781c6d','#a52c60','#cf4446','#ed6925','#fcffa4'].map(hexToLinear) },
  { name: 'Magma', colors: ['#000004','#180f3d','#440f76','#721f81','#b5367a','#e55c30','#fba40a','#fcffa4'].map(hexToLinear) },
  { name: 'Plasma', colors: ['#0d0887','#4b03a1','#7d03a8','#a82296','#cb4679','#e86825','#f89540','#f0f921'].map(hexToLinear) },
  { name: 'Turbo', colors: ['#30123b','#4662d7','#36aaf9','#1ae4b6','#72fe5e','#c8ef34','#faba39','#e6550d'].map(hexToLinear) },
  // 45 Community palettes (5-stop)
  { name: 'Sunset Beach', colors: ['#69d2e7','#a7dbd8','#e0e4cc','#f38630','#fa6900'].map(hexToLinear) },
  { name: 'Coral Reef', colors: ['#fe4365','#fc9d9a','#f9cdad','#c8c8a9','#83af9b'].map(hexToLinear) },
  { name: 'Autumn Fire', colors: ['#ecd078','#d95b43','#c02942','#542437','#53777a'].map(hexToLinear) },
  { name: 'Neon Garden', colors: ['#556270','#4ecdc4','#c7f464','#ff6b6b','#c44d58'].map(hexToLinear) },
  { name: 'Deep Forest', colors: ['#e8ddcb','#cdb380','#036564','#033649','#031634'].map(hexToLinear) },
  { name: 'Carnival', colors: ['#490a3d','#bd1550','#e97f02','#f8ca00','#8a9b0f'].map(hexToLinear) },
  { name: 'Mint Fresh', colors: ['#594f4f','#547980','#45ada8','#9de0ad','#e5fcc2'].map(hexToLinear) },
  { name: 'Desert Spice', colors: ['#00a0b0','#6a4a3c','#cc333f','#eb6841','#edc951'].map(hexToLinear) },
  { name: 'Berry Blush', colors: ['#e94e77','#d68189','#c6a49a','#c6e5d9','#f4ead5'].map(hexToLinear) },
  { name: 'Watermelon', colors: ['#3fb8af','#7fc7af','#dad8a7','#ff9e9d','#ff3d7f'].map(hexToLinear) },
  { name: 'Ocean Deep', colors: ['#343838','#005f6b','#008c9e','#00b4cc','#00dffc'].map(hexToLinear) },
  { name: 'Dusty Rose', colors: ['#413e4a','#73626e','#b38184','#f0b49e','#f7e4be'].map(hexToLinear) },
  { name: 'Sunrise', colors: ['#ff4e50','#fc913a','#f9d423','#ede574','#e1f5c4'].map(hexToLinear) },
  { name: 'Warm Sage', colors: ['#99b898','#fecea8','#ff847c','#e84a5f','#2a363b'].map(hexToLinear) },
  { name: 'Teal Harvest', colors: ['#655643','#80bca3','#f6f7bd','#e6ac27','#bf4d28'].map(hexToLinear) },
  { name: 'Electric Lime', colors: ['#00a8c6','#40c0cb','#f9f2e7','#aee239','#8fbe00'].map(hexToLinear) },
  { name: 'Plum Night', colors: ['#351330','#424254','#64908a','#e8caa4','#cc2a41'].map(hexToLinear) },
  { name: 'Tangerine', colors: ['#554236','#f77825','#d3ce3d','#f1efa5','#60b99a'].map(hexToLinear) },
  { name: 'Crimson Gold', colors: ['#8c2318','#5e8c6a','#88a65e','#bfb35a','#f2c45a'].map(hexToLinear) },
  { name: 'Candy', colors: ['#fad089','#ff9c5b','#f5634a','#ed303c','#3b8183'].map(hexToLinear) },
  { name: 'Dusk', colors: ['#f8b195','#f67280','#c06c84','#6c5b7b','#355c7d'].map(hexToLinear) },
  { name: 'Electric', colors: ['#d1e751','#ffffff','#000000','#4dbce9','#26ade4'].map(hexToLinear) },
  { name: 'Emerald', colors: ['#1b676b','#519548','#88c425','#bef202','#eafde6'].map(hexToLinear) },
  { name: 'Amber Spice', colors: ['#5e412f','#fcebb6','#78c0a8','#f07818','#f0a830'].map(hexToLinear) },
  { name: 'Magenta Burst', colors: ['#bcbdac','#cfbe27','#f27435','#f02475','#3b2d38'].map(hexToLinear) },
  { name: 'Dark Bloom', colors: ['#300030','#480048','#601848','#c04848','#f07241'].map(hexToLinear) },
  { name: 'Pastel Dream', colors: ['#a8e6ce','#dcedc2','#ffd3b5','#ffaaa6','#ff8c94'].map(hexToLinear) },
  { name: 'Noir Gold', colors: ['#3e4147','#fffedf','#dfba69','#5a2e2e','#2a2c31'].map(hexToLinear) },
  { name: 'Neon Punk', colors: ['#fc354c','#29221f','#13747d','#0abfbc','#fcf7c5'].map(hexToLinear) },
  { name: 'Citrus', colors: ['#cc0c39','#e6781e','#c8cf02','#f8fcc1','#1693a7'].map(hexToLinear) },
  { name: 'Terracotta', colors: ['#a7c5bd','#e5ddcb','#eb7b59','#cf4647','#524656'].map(hexToLinear) },
  { name: 'Cherry Blossom', colors: ['#5c323e','#a82743','#e15e32','#c0d23e','#e5f04c'].map(hexToLinear) },
  { name: 'Earth Tone', colors: ['#fdf1cc','#c6d6b8','#987f69','#e3ad40','#fcd036'].map(hexToLinear) },
  { name: 'Rainbow Pop', colors: ['#ff003c','#ff8a00','#fabe28','#88c100','#00c176'].map(hexToLinear) },
  { name: 'Wine Country', colors: ['#d1313d','#e5625c','#f9bf76','#8eb2c5','#615375'].map(hexToLinear) },
  { name: 'Deep Purple', colors: ['#111625','#341931','#571b3c','#7a1e48','#9d2053'].map(hexToLinear) },
  { name: 'Volcano', colors: ['#395a4f','#432330','#853c43','#f25c5e','#ffa566'].map(hexToLinear) },
  { name: 'Jungle', colors: ['#512b52','#635274','#7bb0a8','#a7dbab','#e4f5b1'].map(hexToLinear) },
  { name: 'Firecracker', colors: ['#b5ac01','#ecba09','#e86e1c','#d41e45','#1b1521'].map(hexToLinear) },
  { name: 'Cotton Candy', colors: ['#f1396d','#fd6081','#f3ffeb','#acc95f','#8f9924'].map(hexToLinear) },
  { name: 'Sage Blush', colors: ['#b1e6d1','#77b1a9','#3d7b80','#270a33','#451a3e'].map(hexToLinear) },
  { name: 'Amber Wave', colors: ['#ffab07','#e9d558','#72ad75','#0e8d94','#434d53'].map(hexToLinear) },
  { name: 'Tropical', colors: ['#5cacc4','#8cd19d','#cee879','#fcb653','#ff5254'].map(hexToLinear) },
  { name: 'Lava Flow', colors: ['#ccf390','#e0e05a','#f7c41f','#fc930a','#ff003d'].map(hexToLinear) },
  { name: 'Moss Stone', colors: ['#f2e8c4','#98d9b6','#3ec9a7','#2b879e','#616668'].map(hexToLinear) },
  { name: 'Twilight', colors: ['#2d1b33','#f36a71','#ee887a','#e4e391','#9abc8a'].map(hexToLinear) },
  { name: 'Fire Dance', colors: ['#452e3c','#ff3d5a','#ffb969','#eaf27e','#3b8c88'].map(hexToLinear) },
  { name: 'Aqua Terra', colors: ['#fb6900','#f63700','#004853','#007e80','#00b9bd'].map(hexToLinear) },
];

function applyPalette(idx) {
  if (idx < 0 || idx >= PALETTES.length) return;
  const pal = PALETTES[idx];
  const c = pal.colors;
  if (c.length === 8) {
    // Scientific 8-stop: sample at 0,2,4,6,7
    state.baseColor = [...c[0]];
    state.accentColor = [...c[2]];
    state.tipColor = [...c[4]];
    state.glitterColor = [...c[6]];
    state.sheenColor = [...c[7]];
  } else {
    // Community 5-stop: direct mapping
    state.baseColor = [...c[0]];
    state.accentColor = [...c[1]];
    state.tipColor = [...c[2]];
    state.glitterColor = [...c[3]];
    state.sheenColor = [...c[4]];
  }
  // Derive glitterAccent/glitterTip as lighter variants
  state.glitterAccent = state.glitterColor.map(v => Math.min(1.0, v * 1.3 + 0.1));
  state.glitterTip = state.tipColor.map(v => Math.min(1.0, v * 1.2 + 0.15));
}

// ─── WGSL Shaders ────────────────────────────────────────────────────────────

const commonHeader = /* wgsl */`
struct Params {
  dt: f32,
  dx: f32,       // 1.0 / simRes
  simRes: f32,
  time: f32,
  splatX: f32,
  splatY: f32,
  splatDx: f32,
  splatDy: f32,
  splatR: f32,
  splatG: f32,
  splatB: f32,
  splatRadius: f32,
  curlStrength: f32,
  pressureDecay: f32,
  velDissipation: f32,
  dyeDissipation: f32,
};

@group(0) @binding(0) var<uniform> p: Params;

const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS: f32 = ${SIM_SPHERE_RADIUS};
`;

const MAX_SPLATS = 128;

const batchSplatShaderVel = /* wgsl */`
${commonHeader}

struct Splat {
  x: f32, y: f32, dx: f32, dy: f32,
  r: f32, g: f32, b: f32, radius: f32,
};

@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> splats: array<Splat>;
@group(0) @binding(4) var<uniform> splatMeta: vec4u;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / p.simRes;
  let toCenter = uv - SPHERE_CENTER;
  let dist = length(toCenter);
  if (dist > SPHERE_RADIUS) {
    textureStore(dst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }
  var vel = textureLoad(src, id.xy, 0).xy;
  let boundaryFade = 1.0 - smoothstep(SPHERE_RADIUS - 0.08, SPHERE_RADIUS - 0.02, dist);
  let count = min(splatMeta.x, ${MAX_SPLATS}u);
  for (var i = 0u; i < count; i++) {
    let s = splats[i];
    let diff = uv - vec2f(s.x, s.y);
    let dist2 = dot(diff, diff);
    let strength = exp(-dist2 / (2.0 * s.radius * s.radius));
    vel += strength * boundaryFade * vec2f(s.dx, s.dy);
  }
  if (dist > SPHERE_RADIUS - 0.04) {
    let normal = toCenter / max(dist, 1e-5);
    let outward = dot(vel, normal);
    if (outward > 0.0) {
      vel -= normal * outward;
    }
    let wallT = clamp((dist - (SPHERE_RADIUS - 0.04)) / 0.04, 0.0, 1.0);
    vel *= mix(1.0, 0.72, wallT);
  }
  textureStore(dst, id.xy, vec4f(vel, 0.0, 1.0));
}
`;

const batchSplatShaderDye = /* wgsl */`
${commonHeader}

struct Splat {
  x: f32, y: f32, dx: f32, dy: f32,
  r: f32, g: f32, b: f32, radius: f32,
};

@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> splats: array<Splat>;
@group(0) @binding(4) var<uniform> splatMeta: vec4u;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / p.simRes;
  let toCenter = uv - SPHERE_CENTER;
  let dist = length(toCenter);
  if (dist > SPHERE_RADIUS - 0.02) {
    textureStore(dst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }
  var dye = textureLoad(src, id.xy, 0);
  let boundaryFade = 1.0 - smoothstep(SPHERE_RADIUS - 0.08, SPHERE_RADIUS - 0.02, dist);
  let count = min(splatMeta.x, ${MAX_SPLATS}u);
  for (var i = 0u; i < count; i++) {
    let s = splats[i];
    let diff = uv - vec2f(s.x, s.y);
    let dist2 = dot(diff, diff);
    let strength = exp(-dist2 / (2.0 * s.radius * s.radius));
    let incoming = vec3f(s.r, s.g, s.b);
    if (incoming.x + incoming.y + incoming.z > 0.0) {
      dye = vec4f(mix(dye.rgb, incoming, min(strength * boundaryFade, 1.0)), 1.0);
    }
  }
  textureStore(dst, id.xy, dye);
}
`;

// Fused curl + vorticity with shared memory (eliminates 1 dispatch + pass barrier)
const fusedCurlVortShader = /* wgsl */`
${commonHeader}
@group(0) @binding(1) var velSrc: texture_2d<f32>;
@group(0) @binding(2) var curlDst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var velDst: texture_storage_2d<rgba16float, write>;

var<workgroup> vxTile: array<array<f32, 12>, 12>;
var<workgroup> vyTile: array<array<f32, 12>, 12>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(
  @builtin(global_invocation_id) id: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u
) {
  let res = u32(p.simRes);
  let lIdx = lid.y * ${WORKGROUP}u + lid.x;
  let baseX = wid.x * ${WORKGROUP}u;
  let baseY = wid.y * ${WORKGROUP}u;

  // Load 12x12 velocity tile (2-wide halo around 8x8 workgroup)
  for (var t = lIdx; t < 144u; t += ${WORKGROUP * WORKGROUP}u) {
    let tx = t % 12u;
    let ty = t / 12u;
    let gx = i32(baseX) + i32(tx) - 2;
    let gy = i32(baseY) + i32(ty) - 2;
    let cx = u32(clamp(gx, 0, i32(res) - 1));
    let cy = u32(clamp(gy, 0, i32(res) - 1));
    let v = textureLoad(velSrc, vec2u(cx, cy), 0).xy;
    vxTile[ty][tx] = v.x;
    vyTile[ty][tx] = v.y;
  }
  workgroupBarrier();

  if (id.x >= res || id.y >= res) { return; }

  let lx = lid.x;
  let ly = lid.y;

  // Compute curl + shear at center from vel tile (center at tile pos lx+2, ly+2)
  let curl = 0.5 * ((vyTile[ly+2][lx+3] - vyTile[ly+2][lx+1]) - (vxTile[ly+3][lx+2] - vxTile[ly+1][lx+2]));
  let shearX = 0.5 * (vxTile[ly+2][lx+3] - vxTile[ly+2][lx+1]);
  let shearY = 0.5 * (vyTile[ly+3][lx+2] - vyTile[ly+1][lx+2]);

  // Write curl texture (needed by particle update shader)
  textureStore(curlDst, id.xy, vec4f(curl, shearX, shearY, 1.0));

  // Vorticity confinement
  let uv = (vec2f(id.xy) + 0.5) / p.simRes;
  let dist = length(uv - SPHERE_CENTER);
  if (dist > SPHERE_RADIUS) {
    textureStore(velDst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }

  // Compute curl at 4 neighbors from shared vel tile for vorticity N vector
  let curlL = 0.5 * ((vyTile[ly+2][lx+2] - vyTile[ly+2][lx]) - (vxTile[ly+3][lx+1] - vxTile[ly+1][lx+1]));
  let curlR = 0.5 * ((vyTile[ly+2][lx+4] - vyTile[ly+2][lx+2]) - (vxTile[ly+3][lx+3] - vxTile[ly+1][lx+3]));
  let curlB = 0.5 * ((vyTile[ly+1][lx+3] - vyTile[ly+1][lx+1]) - (vxTile[ly+2][lx+2] - vxTile[ly][lx+2]));
  let curlT = 0.5 * ((vyTile[ly+3][lx+3] - vyTile[ly+3][lx+1]) - (vxTile[ly+4][lx+2] - vxTile[ly+2][lx+2]));

  let cL = abs(curlL);
  let cR = abs(curlR);
  let cB = abs(curlB);
  let cT = abs(curlT);

  var N = vec2f(cR - cL, cT - cB);
  let lenN = length(N);
  let vel = vec2f(vxTile[ly+2][lx+2], vyTile[ly+2][lx+2]);
  if (lenN < 1e-5) {
    textureStore(velDst, id.xy, vec4f(vel, 0.0, 1.0));
    return;
  }
  N = N / lenN;
  let force = vec2f(N.y, -N.x) * curl * p.curlStrength;
  var newVel = vel + force * p.dt;
  if (dist > SPHERE_RADIUS - 0.04) {
    let normal = (uv - SPHERE_CENTER) / max(dist, 1e-5);
    let outward = dot(newVel, normal);
    if (outward > 0.0) {
      newVel -= normal * outward;
    }
    let wallT = clamp((dist - (SPHERE_RADIUS - 0.04)) / 0.04, 0.0, 1.0);
    newVel *= mix(1.0, 0.75, wallT);
  }
  textureStore(velDst, id.xy, vec4f(newVel, 0.0, 1.0));
}
`;

// Fused divergence + clear pressure (eliminates 1 dispatch + pass barrier)
const fusedDivClearPressShader = /* wgsl */`
${commonHeader}
@group(0) @binding(1) var vel: texture_2d<f32>;
@group(0) @binding(2) var pressSrc: texture_2d<f32>;
@group(0) @binding(3) var divDst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var pressDst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  // Divergence
  let L = textureLoad(vel, vec2u(u32(max(i32(id.x)-1, 0)), id.y), 0).x;
  let R = textureLoad(vel, vec2u(min(id.x+1, res-1), id.y), 0).x;
  let B = textureLoad(vel, vec2u(id.x, u32(max(i32(id.y)-1, 0))), 0).y;
  let T = textureLoad(vel, vec2u(id.x, min(id.y+1, res-1)), 0).y;
  let div = 0.5 * ((R - L) + (T - B));
  textureStore(divDst, id.xy, vec4f(div, 0.0, 0.0, 1.0));
  // Clear pressure (decay)
  let pr = textureLoad(pressSrc, id.xy, 0).x * p.pressureDecay;
  textureStore(pressDst, id.xy, vec4f(pr, 0.0, 0.0, 1.0));
}
`;

// Shared memory Jacobi (shared mem reduces texture loads ~49%)
const jacobiShader = /* wgsl */`
${commonHeader}
@group(0) @binding(1) var pressure: texture_2d<f32>;
@group(0) @binding(2) var divTex: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

var<workgroup> pTile: array<array<f32, 10>, 10>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(
  @builtin(global_invocation_id) id: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u
) {
  let res = u32(p.simRes);
  let lIdx = lid.y * ${WORKGROUP}u + lid.x;
  let baseX = wid.x * ${WORKGROUP}u;
  let baseY = wid.y * ${WORKGROUP}u;

  // Load 10x10 pressure tile (1-wide halo around 8x8 workgroup)
  for (var t = lIdx; t < 100u; t += ${WORKGROUP * WORKGROUP}u) {
    let tx = t % 10u;
    let ty = t / 10u;
    let gx = i32(baseX) + i32(tx) - 1;
    let gy = i32(baseY) + i32(ty) - 1;
    let cx = u32(clamp(gx, 0, i32(res) - 1));
    let cy = u32(clamp(gy, 0, i32(res) - 1));
    pTile[ty][tx] = textureLoad(pressure, vec2u(cx, cy), 0).x;
  }
  workgroupBarrier();

  if (id.x >= res || id.y >= res) { return; }

  let lx = lid.x;
  let ly = lid.y;
  let pL = pTile[ly + 1][lx];
  let pR = pTile[ly + 1][lx + 2];
  let pB = pTile[ly][lx + 1];
  let pT = pTile[ly + 2][lx + 1];
  let d = textureLoad(divTex, id.xy, 0).x;

  let pNew = (pL + pR + pB + pT - d) * 0.25;
  textureStore(dst, id.xy, vec4f(pNew, 0.0, 0.0, 1.0));
}
`;

const gradSubShader = /* wgsl */`
${commonHeader}
@group(0) @binding(1) var velTex: texture_2d<f32>;
@group(0) @binding(2) var pressTex: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / p.simRes;
  let toCenter = uv - SPHERE_CENTER;
  let dist = length(toCenter);
  if (dist > SPHERE_RADIUS) {
    textureStore(dst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }
  let pL = textureLoad(pressTex, vec2u(u32(max(i32(id.x)-1, 0)), id.y), 0).x;
  let pR = textureLoad(pressTex, vec2u(min(id.x+1, res-1), id.y), 0).x;
  let pB = textureLoad(pressTex, vec2u(id.x, u32(max(i32(id.y)-1, 0))), 0).x;
  let pT = textureLoad(pressTex, vec2u(id.x, min(id.y+1, res-1)), 0).x;
  var vel = textureLoad(velTex, id.xy, 0).xy;
  vel -= 0.5 * vec2f(pR - pL, pT - pB);
  // Enforce no-penetration at sphere boundary
  if (dist > SPHERE_RADIUS - 0.02) {
    let normal = toCenter / dist;
    let outward = dot(vel, normal);
    if (outward > 0.0) {
      vel -= normal * outward;
    }
  }
  textureStore(dst, id.xy, vec4f(vel, 0.0, 1.0));
}
`;

const advectVelShader = /* wgsl */`
${commonHeader}
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var sampl: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / p.simRes;

  // Sphere boundary: hard kill outside simulation circle
  let toCenter = uv - SPHERE_CENTER;
  let dist = length(toCenter);
  if (dist > SPHERE_RADIUS) {
    textureStore(dst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }

  let vel = textureLoad(src, id.xy, 0).xy;
  let backUV = uv - p.dt * vel * p.dx;
  // Clamp backtrace inside sphere so we never sample from outside
  let backToCenter = backUV - SPHERE_CENTER;
  let backDist = length(backToCenter);
  var sampUV = backUV;
  if (backDist > SPHERE_RADIUS) {
    sampUV = SPHERE_CENTER + backToCenter / backDist * SPHERE_RADIUS;
  }
  let clamped = clamp(sampUV, vec2f(0.5 / p.simRes), vec2f(1.0 - 0.5 / p.simRes));
  var advected = textureSampleLevel(src, sampl, clamped, 0.0).xy;

  // Boundary interaction: reflect and repulse near edge
  let boundaryZone = SPHERE_RADIUS - 0.05;
  if (dist > boundaryZone) {
    let normal = toCenter / dist;
    let outward = dot(advected, normal);
    let proximity = (dist - boundaryZone) / (SPHERE_RADIUS - boundaryZone);
    if (outward > 0.0) {
      advected -= normal * outward * 2.0;
      advected *= mix(1.0, 0.7, proximity);
    }
    advected -= normal * proximity * proximity * 15.0 * p.dt;
  }

  textureStore(dst, id.xy, vec4f(advected * p.velDissipation, 0.0, 1.0));
}
`;

const advectDyeShader = /* wgsl */`
${commonHeader}
@group(0) @binding(1) var velTex: texture_2d<f32>;
@group(0) @binding(2) var dyeSrc: texture_2d<f32>;
@group(0) @binding(3) var sampl: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / p.simRes;

  // Hard kill dye outside sphere
  let toCenter = uv - SPHERE_CENTER;
  let dist = length(toCenter);
  if (dist > SPHERE_RADIUS) {
    textureStore(dst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }
  let edgeFade = smoothstep(SPHERE_RADIUS, SPHERE_RADIUS - 0.04, dist);

  let vel = textureLoad(velTex, id.xy, 0).xy;
  let backUV = uv - p.dt * vel * p.dx;
  // Clamp backtrace inside sphere
  let backToCenter = backUV - SPHERE_CENTER;
  let backDist = length(backToCenter);
  var sampUV = backUV;
  if (backDist > SPHERE_RADIUS) {
    sampUV = SPHERE_CENTER + backToCenter / backDist * SPHERE_RADIUS;
  }
  let clamped = clamp(sampUV, vec2f(0.5 / p.simRes), vec2f(1.0 - 0.5 / p.simRes));
  let advected = textureSampleLevel(dyeSrc, sampl, clamped, 0.0);
  // Ratio-preserving cap: scale all channels proportionally to keep hue intact
  var dye = advected.rgb * p.dyeDissipation * edgeFade;
  let maxC = max(dye.r, max(dye.g, dye.b));
  if (maxC > 1.2) {
    dye *= 1.2 / maxC;
  }
  // Aggressively remove grey component to keep gold tones sharp
  let minC = min(dye.r, min(dye.g, dye.b));
  dye -= vec3f(minC * 0.08);

  textureStore(dst, id.xy, vec4f(max(dye, vec3f(0.0)), 1.0));
}
`;

// ─── Curl Noise Compute Shader ──────────────────────────────────────────────
const curlNoiseShader = /* wgsl */`
struct NoiseParams {
  time: f32,
  amount: f32,
  simRes: f32,
  frequency: f32,
  speed: f32,
  noiseType: f32, // 0..7
  behavior: f32,  // 0=none, 1=mirror, 2+=n-fold symmetry
  mapping: f32,   // 0=cartesian, 1=radial outflow
  warp: f32,
  sharpness: f32,
  anisotropy: f32,
  blend: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
};

@group(0) @binding(0) var<uniform> np: NoiseParams;
@group(0) @binding(1) var velSrc: texture_2d<f32>;
@group(0) @binding(2) var velDst: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS = ${SIM_SPHERE_RADIUS};

fn hash(p: vec2f) -> f32 {
  var n = u32(dot(p, vec2f(127.1, 311.7)) * 43758.5453);
  n = n ^ (n >> 16u);
  n = n * 0x45d9f3bu;
  n = n ^ (n >> 16u);
  return f32(n & 0xFFFFu) / 65535.0;
}

fn vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash(i), hash(i + vec2f(1.0, 0.0)), u.x),
    mix(hash(i + vec2f(0.0, 1.0)), hash(i + vec2f(1.0, 1.0)), u.x),
    u.y
  );
}

fn rot2(p: vec2f, a: f32) -> vec2f {
  let c = cos(a);
  let s = sin(a);
  return vec2f(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn worleyLite(p: vec2f) -> f32 {
  let c = floor(p);
  let f = fract(p);
  let o0 = vec2f(0.0, 0.0);
  let o1 = vec2f(1.0, 0.0);
  let o2 = vec2f(0.0, 1.0);
  let o3 = vec2f(1.0, 1.0);
  let p0 = o0 + vec2f(hash(c + o0), hash(c + o0 + vec2f(17.0, 59.0))) - f;
  let p1 = o1 + vec2f(hash(c + o1), hash(c + o1 + vec2f(17.0, 59.0))) - f;
  let p2 = o2 + vec2f(hash(c + o2), hash(c + o2 + vec2f(17.0, 59.0))) - f;
  let p3 = o3 + vec2f(hash(c + o3), hash(c + o3 + vec2f(17.0, 59.0))) - f;
  let d = min(min(length(p0), length(p1)), min(length(p2), length(p3)));
  return 1.0 - clamp(d, 0.0, 1.0);
}

struct SymmetryFrame {
  uv: vec2f,
  basisX: vec2f,
  basisY: vec2f,
  handedness: f32,
};

fn makeSymmetryFrame(uvRaw: vec2f) -> SymmetryFrame {
  var frame: SymmetryFrame;
  frame.uv = uvRaw;
  frame.basisX = vec2f(1.0, 0.0);
  frame.basisY = vec2f(0.0, 1.0);
  frame.handedness = 1.0;

  if (np.behavior < 0.5) {
    return frame;
  }

  let p = uvRaw - SPHERE_CENTER;
  if (np.behavior < 1.5) {
    // Exact left/right mirror around the vertical axis.
    let sx = select(-1.0, 1.0, p.x >= 0.0);
    frame.uv = SPHERE_CENTER + vec2f(abs(p.x), p.y);
    frame.basisX = vec2f(sx, 0.0);
    frame.basisY = vec2f(0.0, 1.0);
    frame.handedness = sx;
    return frame;
  }

  // 2+: n-fold rotational symmetry by canonical wedge folding (single sample domain).
  let folds = max(2.0, min(8.0, floor(np.behavior + 0.5)));
  let sector = TAU / folds;
  let theta = atan2(p.y, p.x);
  let k = floor(theta / sector + 0.5); // nearest sector axis index
  let rotToCanonical = -k * sector;

  var q = rot2(p, rotToCanonical);
  var reflectY = 1.0;
  if (q.y < 0.0) {
    q.y = -q.y;
    reflectY = -1.0;
  }
  frame.uv = SPHERE_CENTER + q;

  let rotBack = -rotToCanonical;
  let c = cos(rotBack);
  let s = sin(rotBack);
  frame.basisX = vec2f(c, s);
  frame.basisY = vec2f(-s * reflectY, c * reflectY);
  frame.handedness = reflectY;
  return frame;
}

fn applyNoiseMapping(uvRaw: vec2f, t: f32) -> vec2f {
  if (np.mapping < 0.5) {
    return uvRaw;
  }

  // Radial outflow mapping: domain scrolls from the center toward the edge.
  let p = uvRaw - SPHERE_CENTER;
  let r = length(p) / max(SPHERE_RADIUS, 1e-5);
  let a = atan2(p.y, p.x);
  let a01 = (a + PI) / TAU;
  let radialScroll = r * (3.2 + np.frequency * 2.6) - t * (0.45 + np.speed * 1.8);
  let twist = (vnoise(vec2f(a01 * 8.0, r * 4.0 + t * 0.2)) - 0.5) * (0.35 + np.warp * 0.75);
  return vec2f(a01 * (1.0 + np.anisotropy * 3.0), radialScroll + twist);
}

fn scalarFieldCore(uvRaw: vec2f) -> f32 {
  let t = np.time * (0.1 + np.speed * 2.0);
  let uv = applyNoiseMapping(uvRaw, t);
  let baseScale = 2.0 + np.frequency * 14.0;
  var p = uv * baseScale;

  // Profile defaults by type.
  let ty = floor(np.noiseType + 0.5);
  var pWarp = 0.20;
  var pSharp = 0.25;
  var pAniso = 0.15;
  var pBlend = 0.20;
  if (ty < 0.5) { // Classic Curl
    pWarp = 0.20; pSharp = 0.20; pAniso = 0.10; pBlend = 0.20;
  } else if (ty < 1.5) { // Domain-Warped Curl
    pWarp = 0.95; pSharp = 0.35; pAniso = 0.30; pBlend = 0.40;
  } else if (ty < 2.5) { // Ridged Fractal
    pWarp = 0.40; pSharp = 1.00; pAniso = 0.35; pBlend = 0.35;
  } else if (ty < 3.5) { // Voronoi / Cell
    pWarp = 0.35; pSharp = 0.55; pAniso = 0.20; pBlend = 1.00;
  } else if (ty < 4.5) { // Flow / Rotated
    pWarp = 0.55; pSharp = 0.35; pAniso = 0.95; pBlend = 0.35;
  } else if (ty < 5.5) { // Gabor-like
    pWarp = 0.25; pSharp = 0.75; pAniso = 1.00; pBlend = 0.50;
  } else if (ty < 6.5) { // Hybrid
    pWarp = 0.80; pSharp = 0.70; pAniso = 0.65; pBlend = 0.75;
  } else { // Jupiter
    pWarp = 0.88; pSharp = 0.62; pAniso = 0.95; pBlend = 0.58;
  }

  // Shared sliders nudge profile defaults.
  let warpW = clamp(mix(pWarp, np.warp, 0.65), 0.0, 1.0);
  let sharpW = clamp(mix(pSharp, np.sharpness, 0.65), 0.0, 1.0);
  let anisoW = clamp(mix(pAniso, np.anisotropy, 0.65), 0.0, 1.0);
  let blendW = clamp(mix(pBlend, np.blend, 0.65), 0.0, 1.0);

  // Domain warp.
  let w = vec2f(
    vnoise(p * 0.65 + vec2f(t * 0.23, -t * 0.17)),
    vnoise(p * 0.65 + vec2f(-t * 0.19, t * 0.21))
  ) * 2.0 - 1.0;
  p += w * warpW * 2.4;

  // Multi-scale base field.
  let p2 = rot2(p * 2.0, 0.35 + anisoW * 1.2);
  let p3 = rot2(p * 4.0, -0.5 - anisoW * 0.9);
  let n1 = vnoise(p + vec2f(t * 0.30, t * 0.20));
  let n2 = vnoise(p2 + vec2f(-t * 0.50, t * 0.40));
  let n3 = vnoise(p3 + vec2f(t * 0.70, -t * 0.30));
  let fbm = n1 * 0.5 + n2 * 0.3 + n3 * 0.2;

  // Ridged multifractal feel.
  let ridged = pow(1.0 - abs(fbm * 2.0 - 1.0), 1.0 + sharpW * 2.0);

  // Banded/anisotropic streaking.
  let lat = (uv.y - 0.5) * (10.0 + np.frequency * 38.0);
  let jets = 0.5 + 0.5 * sin(lat + t * (0.6 + np.speed * 1.8) + (n1 - 0.5) * 3.2);
  let dirTheta = (vnoise(uv * 0.6 + vec2f(9.0, 4.0)) * 2.0 - 1.0) * PI;
  let dir = vec2f(cos(dirTheta), sin(dirTheta));
  let streak = 0.5 + 0.5 * sin(dot(uv - SPHERE_CENTER, dir) * (18.0 + np.frequency * 30.0) + t * 1.2);
  let banded = mix(fbm, fbm * mix(1.0, pow(jets, 1.7), anisoW), 0.6) * mix(1.0, streak, anisoW * 0.5);

  // Cellular breakup + micro detail.
  let cell = worleyLite(p * (1.4 + np.frequency * 1.6));
  let micro = (vnoise(p * 7.0 + vec2f(t * 0.9, -t * 0.8)) - 0.5) * 2.0;

  var s = mix(banded, ridged, sharpW);
  s += (cell - 0.5) * blendW * 0.8;
  s += micro * (0.15 + 0.12 * blendW);

  // Type accents.
  if (ty >= 6.5) {
    // Jupiter: banded gas giant flow with storm cells and a red-spot style vortex.
    let pPlanet = uvRaw - SPHERE_CENTER;
    let latNorm = pPlanet.y / max(SPHERE_RADIUS, 1e-5);
    let lon = (atan2(pPlanet.y, pPlanet.x) + PI) / TAU;
    let shear = (vnoise(vec2f(latNorm * 3.0 + 17.0, t * 0.15)) - 0.5) * (0.8 + np.warp);
    let bandPhase = latNorm * (18.0 + np.frequency * 22.0) + shear * 4.0 + t * (0.2 + np.speed * 1.2);
    let bands = 0.5 + 0.5 * sin(bandPhase);
    let storms = vnoise(vec2f(lon * 12.0 + t * 0.22, latNorm * 9.0 - t * 0.18));
    let grsCenter = vec2f(0.76, 0.44);
    let grsDelta = (uvRaw - grsCenter) / vec2f(0.09, 0.055);
    let grsMask = exp(-dot(grsDelta, grsDelta));
    let grsSwirl = 0.5 + 0.5 * sin(atan2(grsDelta.y, grsDelta.x) * 5.0 + t * 1.8);
    var jupiter = mix(bands, storms, 0.35 + 0.2 * blendW);
    jupiter = mix(jupiter, grsSwirl, grsMask * (0.55 + 0.3 * blendW));
    s = mix(s, jupiter, 0.72);
  } else if (ty >= 5.5) {
    let hybridCell = worleyLite(rot2(p * 1.8, 0.9) + vec2f(t * 0.2, -t * 0.14));
    s = mix(s, s + (hybridCell - 0.5) * 0.8, 0.35);
  } else if (ty >= 4.5 && ty < 5.5) {
    s = mix(s, streak, 0.45 + 0.3 * anisoW);
  } else if (ty >= 3.5 && ty < 4.5) {
    s = mix(s, cell, 0.55);
  }

  // Additional radial feel when radial mapping is enabled.
  if (np.mapping > 0.5) {
    let pRad = uvRaw - SPHERE_CENTER;
    let r = length(pRad) / max(SPHERE_RADIUS, 1e-5);
    let radialPulse = 0.5 + 0.5 * sin(r * (26.0 + np.frequency * 20.0) - t * (2.6 + np.speed * 4.2));
    s = mix(s, radialPulse, 0.38 + 0.22 * np.blend);
  }
  return s;
}

fn scalarField(uvRaw: vec2f) -> f32 {
  let frame = makeSymmetryFrame(uvRaw);
  return scalarFieldCore(frame.uv);
}

fn uvToPixel(uv: vec2f, res: f32) -> vec2u {
  let px = clamp(uv * res - vec2f(0.5), vec2f(0.0), vec2f(res - 1.0));
  return vec2u(px + vec2f(0.5));
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(np.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / np.simRes;

  let dist = length(uv - SPHERE_CENTER);
  if (dist > SPHERE_RADIUS) {
    textureStore(velDst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }
  let edgeFade = smoothstep(SPHERE_RADIUS, SPHERE_RADIUS - 0.08, dist);

  let frame = makeSymmetryFrame(uv);
  var vel = textureLoad(velSrc, id.xy, 0).xy;
  if (np.behavior >= 0.5) {
    let velCanonical = textureLoad(velSrc, uvToPixel(frame.uv, np.simRes), 0).xy;
    vel = frame.basisX * velCanonical.x + frame.basisY * velCanonical.y;
  }
  let eps = 1.0 / np.simRes;
  let sC = scalarFieldCore(frame.uv);
  let sX = scalarFieldCore(frame.uv + vec2f(eps, 0.0));
  let sY = scalarFieldCore(frame.uv + vec2f(0.0, eps));
  var curlCanonical = vec2f(sY - sC, -(sX - sC)) / eps;

  let uv2 = SPHERE_CENTER + (frame.uv - SPHERE_CENTER) * 1.91;
  let uv2X = SPHERE_CENTER + (frame.uv + vec2f(eps, 0.0) - SPHERE_CENTER) * 1.91;
  let uv2Y = SPHERE_CENTER + (frame.uv + vec2f(0.0, eps) - SPHERE_CENTER) * 1.91;
  let s2C = scalarFieldCore(uv2);
  let s2X = scalarFieldCore(uv2X);
  let s2Y = scalarFieldCore(uv2Y);
  curlCanonical += vec2f(s2Y - s2C, -(s2X - s2C)) / eps * (0.35 + np.blend * 0.25);

  // Curl vector is a 90°-rotated gradient, so reflections require det(J)*J.
  let curl = (frame.basisX * curlCanonical.x + frame.basisY * curlCanonical.y) * frame.handedness;
  let layerBoost = 1.0 + (np.sharpness + np.warp + np.anisotropy + np.blend) * 0.35;
  vel += curl * np.amount * (7.0 + 2.0 * layerBoost) * edgeFade;
  if (dist > SPHERE_RADIUS - 0.04) {
    let normal = (uv - SPHERE_CENTER) / max(dist, 1e-5);
    let outward = dot(vel, normal);
    if (outward > 0.0) {
      vel -= normal * outward;
    }
    let wallT = clamp((dist - (SPHERE_RADIUS - 0.04)) / 0.04, 0.0, 1.0);
    vel *= mix(1.0, 0.8, wallT);
  }
  textureStore(velDst, id.xy, vec4f(vel, 0.0, 1.0));
}
`;

// ─── Dye Noise Shader (injects dye where flow converges) ────────────────────
const dyeNoiseShader = /* wgsl */`
struct DyeNoiseParams {
  time: f32,
  amount: f32,
  simRes: f32,
  curlDyeAmount: f32,
  color: vec4f, // rgb=tint, a=noise symmetry behavior (0=none, 1=mirror, 2+=n-fold)
};

@group(0) @binding(0) var<uniform> dp: DyeNoiseParams;
@group(0) @binding(1) var velSrc: texture_2d<f32>;
@group(0) @binding(2) var dyeSrc: texture_2d<f32>;
@group(0) @binding(3) var dyeDst: texture_storage_2d<rgba16float, write>;

const TAU: f32 = 6.28318530718;
const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS = ${SIM_SPHERE_RADIUS};

fn rot2(p: vec2f, a: f32) -> vec2f {
  let c = cos(a);
  let s = sin(a);
  return vec2f(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn foldSymmetryUV(uvRaw: vec2f, behavior: f32) -> vec2f {
  if (behavior < 0.5) {
    return uvRaw;
  }
  let p = uvRaw - SPHERE_CENTER;
  if (behavior < 1.5) {
    return SPHERE_CENTER + vec2f(abs(p.x), p.y);
  }

  let folds = max(2.0, min(8.0, floor(behavior + 0.5)));
  let sector = TAU / folds;
  let theta = atan2(p.y, p.x);
  let k = floor(theta / sector + 0.5);
  var q = rot2(p, -k * sector);
  q.y = abs(q.y);
  return SPHERE_CENTER + q;
}

fn uvToPixel(uv: vec2f, res: f32) -> vec2u {
  let px = clamp(uv * res - vec2f(0.5), vec2f(0.0), vec2f(res - 1.0));
  return vec2u(px + vec2f(0.5));
}

fn sampleSymVel(uvRaw: vec2f, fallbackPx: vec2u, behavior: f32, res: f32) -> vec2f {
  if (behavior < 0.5) {
    return textureLoad(velSrc, fallbackPx, 0).xy;
  }
  return textureLoad(velSrc, uvToPixel(foldSymmetryUV(uvRaw, behavior), res), 0).xy;
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(dp.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / dp.simRes;
  let behavior = dp.color.a;
  let symUV = foldSymmetryUV(uv, behavior);
  let symPx = uvToPixel(symUV, dp.simRes);
  var dye = textureLoad(dyeSrc, id.xy, 0);
  if (behavior >= 0.5) {
    // Project dye to a canonical symmetry domain so mirrored sectors stay locked.
    dye = textureLoad(dyeSrc, symPx, 0);
  }

  // Fade out near sphere edge to prevent bleed
  let dist = length(uv - SPHERE_CENTER);
  if (dist > SPHERE_RADIUS) {
    textureStore(dyeDst, id.xy, dye);
    return;
  }
  let edgeFade = smoothstep(SPHERE_RADIUS, SPHERE_RADIUS - 0.06, dist);

  // Compute velocity divergence (negative = convergent flow = density buildup)
  let idR = vec2u(min(id.x + 1u, res - 1u), id.y);
  let idL = vec2u(max(id.x, 1u) - 1u, id.y);
  let idU = vec2u(id.x, min(id.y + 1u, res - 1u));
  let idD = vec2u(id.x, max(id.y, 1u) - 1u);
  let uvR = (vec2f(idR) + 0.5) / dp.simRes;
  let uvL = (vec2f(idL) + 0.5) / dp.simRes;
  let uvU = (vec2f(idU) + 0.5) / dp.simRes;
  let uvD = (vec2f(idD) + 0.5) / dp.simRes;
  let vC = sampleSymVel(uv, id.xy, behavior, dp.simRes);
  let vR = sampleSymVel(uvR, idR, behavior, dp.simRes);
  let vL = sampleSymVel(uvL, idL, behavior, dp.simRes);
  let vU = sampleSymVel(uvU, idU, behavior, dp.simRes);
  let vD = sampleSymVel(uvD, idD, behavior, dp.simRes);
  let div = (vR.x - vL.x + vU.y - vD.y) * 0.5;

  // Don't inject into already-bright areas — preserves contrast
  let existingBrightness = max(dye.r, max(dye.g, dye.b));
  let headroom = max(1.0 - existingBrightness, 0.0);
  if (headroom < 0.05) {
    textureStore(dyeDst, id.xy, dye);
    return;
  }

  // Symmetry-safe color phase: canonical position + speed only (no signed angle terms).
  let symP = (symUV - SPHERE_CENTER) / max(SPHERE_RADIUS, 1e-5);
  let colorPhase = dp.time * 0.05 + dot(symP, vec2f(6.0, 3.2)) + length(vC) * 1.15;
  // Cycle through R, G, B emphasis zones for real hue variety
  let cR = 0.3 + 0.7 * max(sin(colorPhase), 0.0);
  let cG = 0.3 + 0.7 * max(sin(colorPhase + 2.094), 0.0);
  let cB = 0.3 + 0.7 * max(sin(colorPhase + 4.189), 0.0);
  let spatialColor = vec3f(cR, cG, cB) * dp.color.rgb;

  // Inject dye where flow converges (negative divergence) — dyeNoiseAmount
  let amt = dp.amount * dp.amount;
  let convergence = max(-div, 0.0);
  let inject = convergence * amt * 0.5 * edgeFade * headroom;
  dye += vec4f(spatialColor * inject, 0.0);

  // Inject dye from curl noise velocity — proportional to local speed
  let velMag = length(vC);
  let curlDye = velMag * dp.curlDyeAmount * 0.003 * edgeFade * headroom;
  dye += vec4f(spatialColor * curlDye, 0.0);

  textureStore(dyeDst, id.xy, dye);
}
`;

// ─── Sphere Cleanup Shader (hard-zero vel+dye outside sphere every frame) ────
const sphereCleanupShader = /* wgsl */`
const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS = ${SIM_SPHERE_RADIUS};
const VISIBLE_RADIUS = ${VISIBLE_SPHERE_RADIUS};

struct CleanupParams { simRes: f32, pad1: f32, pad2: f32, pad3: f32, };
@group(0) @binding(0) var<uniform> cp: CleanupParams;
@group(0) @binding(1) var velSrc: texture_2d<f32>;
@group(0) @binding(2) var dyeSrc: texture_2d<f32>;
@group(0) @binding(3) var velDst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var dyeDst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(cp.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / cp.simRes;
  let dist = length(uv - SPHERE_CENTER);
  var vel = textureLoad(velSrc, id.xy, 0).xy;
  var dye = textureLoad(dyeSrc, id.xy, 0).rgb;

  if (dist > VISIBLE_RADIUS || dist > SPHERE_RADIUS) {
    textureStore(velDst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    textureStore(dyeDst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
  } else {
    if (dist > VISIBLE_RADIUS - 0.05) {
      let normal = (uv - SPHERE_CENTER) / max(dist, 1e-5);
      let outward = dot(vel, normal);
      if (outward > 0.0) {
        vel -= normal * outward;
      }
      let wallT = clamp((dist - (VISIBLE_RADIUS - 0.05)) / 0.05, 0.0, 1.0);
      vel *= mix(1.0, 0.7, wallT);
    }
    // Keep a narrow hard-clear band so RD/noise cannot form visible edge ridges.
    if (dist > VISIBLE_RADIUS - 0.006) {
      dye = vec3f(0.0, 0.0, 0.0);
    }
    textureStore(velDst, id.xy, vec4f(vel, 0.0, 1.0));
    textureStore(dyeDst, id.xy, vec4f(dye, 1.0));
  }
}
`;

// ─── Temperature/Buoyancy Shaders ────────────────────────────────────────────

const temperatureAdvectShader = /* wgsl */`
struct TempParams {
  dt: f32,
  dx: f32,
  simRes: f32,
  dissipation: f32,
};

const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS: f32 = ${SIM_SPHERE_RADIUS};

@group(0) @binding(0) var<uniform> tp: TempParams;
@group(0) @binding(1) var velTex: texture_2d<f32>;
@group(0) @binding(2) var tempSrc: texture_2d<f32>;
@group(0) @binding(3) var sampl: sampler;
@group(0) @binding(4) var tempDst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(tp.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / tp.simRes;
  let toCenter = uv - SPHERE_CENTER;
  let dist = length(toCenter);
  if (dist > SPHERE_RADIUS) {
    textureStore(tempDst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }
  let edgeFade = smoothstep(SPHERE_RADIUS, SPHERE_RADIUS - 0.04, dist);
  let vel = textureLoad(velTex, id.xy, 0).xy;
  let backUV = uv - tp.dt * vel * tp.dx;
  let backToCenter = backUV - SPHERE_CENTER;
  let backDist = length(backToCenter);
  var sampUV = backUV;
  if (backDist > SPHERE_RADIUS) {
    sampUV = SPHERE_CENTER + backToCenter / backDist * SPHERE_RADIUS;
  }
  let clamped = clamp(sampUV, vec2f(0.5 / tp.simRes), vec2f(1.0 - 0.5 / tp.simRes));
  let advected = textureSampleLevel(tempSrc, sampl, clamped, 0.0).r;
  let temp = advected * tp.dissipation * edgeFade;
  textureStore(tempDst, id.xy, vec4f(temp, 0.0, 0.0, 1.0));
}
`;

const buoyancyShader = /* wgsl */`
struct BuoyancyParams {
  simRes: f32,
  dt: f32,
  buoyancy: f32,
  pad: f32,
};

const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS: f32 = ${SIM_SPHERE_RADIUS};

@group(0) @binding(0) var<uniform> bp: BuoyancyParams;
@group(0) @binding(1) var tempTex: texture_2d<f32>;
@group(0) @binding(2) var velSrc: texture_2d<f32>;
@group(0) @binding(3) var velDst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(bp.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / bp.simRes;
  let toCenter = uv - SPHERE_CENTER;
  let dist = length(toCenter);
  if (dist > SPHERE_RADIUS) {
    textureStore(velDst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }
  let edgeFade = smoothstep(SPHERE_RADIUS, SPHERE_RADIUS - 0.04, dist);
  let temp = textureLoad(tempTex, id.xy, 0).r;
  var vel = textureLoad(velSrc, id.xy, 0).xy;
  vel.y += bp.buoyancy * (temp - 0.5) * bp.dt * edgeFade;
  vel *= edgeFade;
  if (dist > SPHERE_RADIUS - 0.04) {
    let normal = toCenter / max(dist, 1e-5);
    let outward = dot(vel, normal);
    if (outward > 0.0) {
      vel -= normal * outward;
    }
    let wallT = clamp((dist - (SPHERE_RADIUS - 0.04)) / 0.04, 0.0, 1.0);
    vel *= mix(1.0, 0.8, wallT);
  }
  textureStore(velDst, id.xy, vec4f(vel, 0.0, 1.0));
}
`;

const tempSplatShader = /* wgsl */`
${commonHeader}

struct Splat {
  x: f32, y: f32, dx: f32, dy: f32,
  r: f32, g: f32, b: f32, radius: f32,
};

@group(0) @binding(1) var tempSrc: texture_2d<f32>;
@group(0) @binding(2) var tempDst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> splats: array<Splat>;
@group(0) @binding(4) var<uniform> splatMeta: vec4u;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / p.simRes;
  let toCenter = uv - SPHERE_CENTER;
  let dist = length(toCenter);
  if (dist > SPHERE_RADIUS) {
    textureStore(tempDst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }
  let boundaryFade = 1.0 - smoothstep(SPHERE_RADIUS - 0.08, SPHERE_RADIUS - 0.02, dist);
  var temp = textureLoad(tempSrc, id.xy, 0).r;
  let count = min(splatMeta.x, ${MAX_SPLATS}u);
  for (var i = 0u; i < count; i++) {
    let s = splats[i];
    let diff = uv - vec2f(s.x, s.y);
    let dist2 = dot(diff, diff);
    let strength = exp(-dist2 / (2.0 * s.radius * s.radius));
    let dyeIntensity = (s.r + s.g + s.b) / 3.0;
    temp += strength * boundaryFade * dyeIntensity * 0.5;
  }
  temp = clamp(temp, 0.0, 1.0);
  textureStore(tempDst, id.xy, vec4f(temp, 0.0, 0.0, 1.0));
}
`;

// ─── Particle Compact Shader (GPU indirect draw — builds visible index list) ──
function makeParticleCompactShader(count) {
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> colors: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> visibleIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> counter: atomic<u32>;

@compute @workgroup_size(${PARTICLE_WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  if (idx >= ${count}u) { return; }
  let col = colors[idx];
  if (col.a > 0.0) {
    let slot = atomicAdd(&counter, 1u);
    visibleIndices[slot] = idx;
  }
}
`;
}

const particleFinalizeShader = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> drawArgs: array<u32>;

@compute @workgroup_size(1)
fn main() {
  drawArgs[0] = 4u;
  drawArgs[1] = atomicLoad(&counter);
  drawArgs[2] = 0u;
  drawArgs[3] = 0u;
}
`;

// ─── Particle Update Compute Shader (function — count baked in) ─────────────
function makeParticleUpdateShader(count, hdr) {
  return /* wgsl */`
${commonHeader}
@group(0) @binding(1) var velTex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(7) var curlTex: texture_2d<f32>;

struct Particle {
  posX: f32,
  posY: f32,
  normalX: f32,
  normalY: f32,
  normalZ: f32,
  angularVel: f32,
  life: f32,
  seed: f32,
};

@group(0) @binding(3) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(4) var dyeTex: texture_2d<f32>;

struct PParams {
  screen: vec4f,
  extra: vec4f,
  extra2: vec4f,
  extra3: vec4f,
};
@group(0) @binding(5) var<uniform> pp: PParams;
@group(0) @binding(6) var<storage, read_write> colors: array<vec4f>;

fn pcg(inp: u32) -> u32 {
  var state = inp * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn randF(seed: u32) -> f32 {
  return f32(pcg(seed)) / 4294967295.0;
}

fn oklabToLinear(lab: vec3f) -> vec3f {
  let l = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  let m = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  let s = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
  return max(vec3f(
    ${hdr ? '3.1272' : '4.0767416621'} * l*l*l + ${hdr ? '-2.2566' : '-3.3077115913'} * m*m*m + ${hdr ? '0.1294' : '0.2309699292'} * s*s*s,
    ${hdr ? '-1.0912' : '-1.2684380046'} * l*l*l + ${hdr ? '2.4138' : '2.6097574011'} * m*m*m + ${hdr ? '-0.3225' : '-0.3413193965'} * s*s*s,
    ${hdr ? '-0.0260' : '-0.0041960863'} * l*l*l + ${hdr ? '-0.5082' : '-0.7034186147'} * m*m*m + ${hdr ? '1.5341' : '1.7076147010'} * s*s*s
  ), vec3f(0.0));
}

fn grainTint(h: f32) -> vec3f {
  let boost = ${hdr ? '1.6' : '1.0'};
  let c0 = vec3f(1.0, 0.75, 0.3);
  let c1 = vec3f(1.0, 0.85, 0.45);
  let c2 = vec3f(1.0, 0.6, 0.25);
  let c3 = vec3f(1.0, 0.88, 0.55);
  let c4 = vec3f(0.85, 0.92, 1.0);
  let c5 = vec3f(1.0, 0.75, 0.88);
  let c6 = vec3f(0.75, 1.0, 0.82);
  let t0 = h / 0.6;
  var color = mix(c0, c1, clamp(t0, 0.0, 1.0));
  color = mix(color, c2, step(0.6, h));
  color = mix(color, c3, step(0.75, h));
  color = mix(color, c4, step(0.85, h));
  color = mix(color, c5, step(0.92, h));
  color = mix(color, c6, step(0.96, h));
  return color * boost;
}

@compute @workgroup_size(${PARTICLE_WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  if (idx >= ${count}u) { return; }

  var part = particles[idx];

  // Sample velocity at particle position
  var vel = textureSampleLevel(velTex, samp, vec2f(part.posX, part.posY), 0.0).xy;


  // Advect particle with fluid
  let dt = p.dt;
  var prePos = vec2f(part.posX, part.posY);
  let preToCenter = prePos - SPHERE_CENTER;
  let preDist = length(preToCenter);
  if (preDist > SPHERE_RADIUS) {
    let n = preToCenter / max(preDist, 1e-5);
    prePos = SPHERE_CENTER + n * (SPHERE_RADIUS - 0.001);
    part.posX = prePos.x;
    part.posY = prePos.y;
  }
  let nearDist = length(prePos - SPHERE_CENTER);
  if (nearDist > SPHERE_RADIUS - 0.03) {
    let n = (prePos - SPHERE_CENTER) / max(nearDist, 1e-5);
    let outward = dot(vel, n);
    if (outward > 0.0) {
      vel -= n * outward;
    }
  }
  part.posX += vel.x * dt * p.dx;
  part.posY += vel.y * dt * p.dx;

  // Read curl + shear from pre-computed curl texture (saves 2 velocity samples)
  let pos = vec2f(part.posX, part.posY);
  let curlData = textureSampleLevel(curlTex, samp, pos, 0.0);
  let curl = curlData.x * 2.0;
  let shearX = curlData.y;
  let shearY = curlData.z;

  // Update angular velocity with curl (damped for slower, smoother shimmer)
  part.angularVel = part.angularVel * 0.92 + curl * 1.0;

  // Rotate normal around Z-axis (spin)
  let spinAngle = part.angularVel * dt;
  let sa2 = spinAngle * spinAngle;
  let cosA = 1.0 - sa2 * 0.5;
  let sinA = spinAngle;
  let nx = part.normalX * cosA - part.normalY * sinA;
  let ny = part.normalX * sinA + part.normalY * cosA;

  // Shear-driven tilt (X/Y rotation) — gentle so shimmer is slow
  let tiltX = shearX * dt * 0.8;
  let tiltY = shearY * dt * 0.8;

  part.normalX = nx + tiltX;
  part.normalY = ny + tiltY;

  // Keep Z positive (facing camera) — render shader re-normalizes
  part.normalZ = max(abs(part.normalZ), 0.3);
  let nLen = length(vec3f(part.normalX, part.normalY, part.normalZ));
  if (nLen > 0.001) {
    part.normalX /= nLen;
    part.normalY /= nLen;
    part.normalZ /= nLen;
  }

  // Life decay
  part.life -= dt;

  // Boundary / respawn check
  let toCenter = vec2f(part.posX, part.posY) - SPHERE_CENTER;
  var dist = length(toCenter);

  if (dist > SPHERE_RADIUS) {
    let n = toCenter / max(dist, 1e-5);
    part.posX = SPHERE_CENTER.x + n.x * (SPHERE_RADIUS - 0.001);
    part.posY = SPHERE_CENTER.y + n.y * (SPHERE_RADIUS - 0.001);
    dist = SPHERE_RADIUS - 0.001;
  }

  if (part.life <= 0.0) {
    let timeU = u32(p.time * 1000.0);
    let h1 = pcg(idx + timeU * 7919u);
    let h2 = pcg(h1);
    let h3 = pcg(h2);
    let h4 = pcg(h3);
    let h5 = pcg(h4);
    let h6 = pcg(h5);

    // Random position inside sphere
    let angle = randF(h1) * 6.2831853;
    let radius = sqrt(randF(h2)) * SPHERE_RADIUS * 0.9;
    part.posX = 0.5 + cos(angle) * radius;
    part.posY = 0.5 + sin(angle) * radius;

    // Random hemisphere normal
    let phi = randF(h3) * 6.2831853;
    let cosTheta = randF(h4);
    let sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    part.normalX = sinTheta * cos(phi);
    part.normalY = sinTheta * sin(phi);
    part.normalZ = max(cosTheta, 0.3);
    let nL = length(vec3f(part.normalX, part.normalY, part.normalZ));
    part.normalX /= nL;
    part.normalY /= nL;
    part.normalZ /= nL;

    part.angularVel = (randF(h5) - 0.5) * 2.0;
    part.life = 2.0 + randF(h6) * 6.0;
  }

  particles[idx] = part;

  // ── Color computation (moved from vertex shader) ──
  let particleUV = vec2f(part.posX, part.posY);
  let centered = particleUV - vec2f(0.5, 0.5);
  if (length(centered) > ${VISIBLE_SPHERE_RADIUS}) {
    colors[idx] = vec4f(0.0);
    return;
  }

  let dye = textureSampleLevel(dyeTex, samp, particleUV, 0.0).rgb;
  let intensity = dot(dye, vec3f(0.3, 0.6, 0.1));
  let fluidGate = smoothstep(0.0, 0.15, intensity);
  if (fluidGate < 0.01) {
    colors[idx] = vec4f(0.0);
    return;
  }

  let n = normalize(vec3f(part.normalX, part.normalY, part.normalZ));
  let lightDir = normalize(vec3f(0.4, 0.6, 1.0));
  let viewDir = vec3f(0.0, 0.0, 1.0);
  let reflected = reflect(-lightDir, n);
  let specAngle = dot(reflected, viewDir);
  let glint = smoothstep(0.92, 1.0, specAngle);
  let ambient = max(dot(n, lightDir), 0.0) * 0.02;

  let seedU = u32(part.seed * 4294967295.0);
  let ch3 = f32(pcg(seedU + 269u)) / 4294967295.0;
  let ch4 = f32(pcg(seedU + 419u)) / 4294967295.0;
  var tint = grainTint(ch3);

  let pa = pp.extra.x;
  let hueAngle = ch4 * 6.283;
  let cosH = cos(hueAngle);
  let sinH = sin(hueAngle);
  let prismatic = vec3f(
    0.5 + 0.5 * cosH,
    0.5 + 0.5 * (cosH * (-0.5) - sinH * 0.866025),
    0.5 + 0.5 * (cosH * (-0.5) + sinH * 0.866025)
  );
  let prisThreshold = max(1.0 - pa * 0.05, 0.0);
  let prisMix = clamp(pa * 0.06, 0.0, 1.0);
  if (ch4 > prisThreshold) {
    tint = mix(tint, prismatic, prisMix);
  }

  // Glitter color: 3-stop gradient accent→base→tip in Oklab (pre-converted on CPU)
  let okBase = pp.extra.yzw;
  let okAccent = pp.extra2.xyz;
  let blend = pp.extra2.w;
  let gLo = mix(0.0, 0.35, blend);
  let gHi = mix(1.0, 0.4, blend);
  let densityT = smoothstep(gLo, gHi, intensity);
  let okTip = pp.extra3.yzw;
  let gt2 = min(densityT * 2.0, 1.0);
  let gt3 = max(densityT * 2.0 - 1.0, 0.0);
  let glitCol = oklabToLinear(mix(mix(okAccent, okBase, gt2), okTip, gt3));
  tint *= glitCol;

  let glintBoost = ${hdr ? '4.0' : '1.0'};
  let brightness = glint * pp.screen.w * glintBoost + ambient;
  colors[idx] = vec4f(tint * brightness * fluidGate, brightness * fluidGate);
}
`;
}

// ─── Particle Render Vertex Shader ───────────────────────────────────────────
const particleRenderVert = /* wgsl */`
struct Particle {
  posX: f32,
  posY: f32,
  normalX: f32,
  normalY: f32,
  normalZ: f32,
  angularVel: f32,
  life: f32,
  seed: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;

struct PParams {
  screen: vec4f,
  extra: vec4f,
  extra2: vec4f,
  extra3: vec4f,
};
@group(0) @binding(1) var<uniform> pp: PParams;
@group(0) @binding(2) var<storage, read> colors: array<vec4f>;
@group(0) @binding(3) var<storage, read> visibleIndices: array<u32>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) color: vec3f,
  @location(1) @interpolate(flat) alpha: f32,
  @location(2) localUV: vec2f,
};

@vertex
fn main(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32
) -> VSOut {
  let realIdx = visibleIndices[ii];
  let col = colors[realIdx];

  var out: VSOut;

  // Early cull: pre-computed alpha <= 0
  if (col.a <= 0.0) {
    out.pos = vec4f(2.0, 2.0, 0.0, 1.0);
    out.color = vec3f(0.0);
    out.alpha = 0.0;
    out.localUV = vec2f(0.0);
    return out;
  }

  let part = particles[realIdx];

  // Triangle-strip quad: 4 vertices
  var quadPos = array<vec2f, 4>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0,  1.0)
  );

  let qp = quadPos[vi];
  let localUV = qp * 0.5 + 0.5;

  let screenSize = pp.screen.xy;
  let basePixelSize = pp.screen.z;
  let sizeRand = pp.extra3.x;
  // Per-particle size variation: seed drives a 0.2..1.8 range scaled by randomness
  let sizeScale = mix(1.0, 0.2 + part.seed * 1.6, sizeRand);
  let pixelSize = basePixelSize * sizeScale;
  let aspect = screenSize.x / screenSize.y;
  let clipSize = vec2f(pixelSize * 2.0 / screenSize.x, pixelSize * 2.0 / screenSize.y);


  let rawClip = vec2f(part.posX, part.posY) * 2.0 - 1.0;
  let clipPos = vec2f(
    rawClip.x / max(aspect, 1.0),
    rawClip.y / max(1.0 / aspect, 1.0)
  );

  out.pos = vec4f(clipPos + qp * clipSize, 0.0, 1.0);
  out.color = col.rgb;
  out.alpha = col.a;
  out.localUV = localUV;
  return out;
}
`;

// ─── Particle Render Fragment Shader ─────────────────────────────────────────
const particleRenderFrag = /* wgsl */`
struct PParams {
  screen: vec4f,
  extra: vec4f,
  extra2: vec4f,
  extra3: vec4f,
};
@group(0) @binding(1) var<uniform> pp: PParams;

struct FSIn {
  @builtin(position) fragPos: vec4f,
  @location(0) @interpolate(flat) color: vec3f,
  @location(1) @interpolate(flat) alpha: f32,
  @location(2) localUV: vec2f,
};

@fragment
fn main(in: FSIn) -> @location(0) vec4f {
  // Sphere mask (safety for large particle sizes)
  let screenSize = pp.screen.xy;
  let aspect = screenSize.x / screenSize.y;
  let rawUV = vec2f(in.fragPos.x, screenSize.y - in.fragPos.y) / screenSize;
  // Convert to simulation UV (1:1 square)
  let uv = vec2f(
    (rawUV.x - 0.5) * max(aspect, 1.0) + 0.5,
    (rawUV.y - 0.5) * max(1.0 / aspect, 1.0) + 0.5
  );
  let centered = uv - vec2f(0.5, 0.5);
  let sphereDist = length(centered);
  let sphereRadius = ${VISIBLE_SPHERE_RADIUS};
  if (sphereDist > sphereRadius) { discard; }

  // Circular cutout + soft edge
  let d = length(in.localUV - vec2f(0.5));
  if (d > 0.5) { discard; }
  let edge = 1.0 - smoothstep(0.3, 0.5, d);
  return vec4f(in.color * edge, in.alpha * edge);
}
`;

const displayShaderVert = /* wgsl */`
@vertex
fn main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  // Fullscreen triangle
  let x = f32(i32(vi) / 2) * 4.0 - 1.0;
  let y = f32(i32(vi) % 2) * 4.0 - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}
`;

// ─── Bloom Shaders ──────────────────────────────────────────────────────────

const bloomDownsampleShader = /* wgsl */`
struct BloomParams {
  texelSize: vec2f,
  threshold: f32,
  knee: f32,
  intensity: f32,
  mipWeight: f32,
  flags: f32,       // bit 0 = apply threshold
  pad: f32,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var dstTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: BloomParams;

fn softThreshold(color: vec3f, threshold: f32, knee: f32) -> vec3f {
  let brightness = max(color.r, max(color.g, color.b));
  var soft = brightness - threshold + knee;
  soft = clamp(soft, 0.0, 2.0 * knee);
  soft = soft * soft / (4.0 * knee + 0.00001);
  let contribution = max(soft, brightness - threshold) / max(brightness, 0.00001);
  return color * max(contribution, 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dstSize = textureDimensions(dstTex);
  if (id.x >= dstSize.x || id.y >= dstSize.y) { return; }

  let texelSize = params.texelSize;
  let uv = (vec2f(id.xy) + 0.5) / vec2f(dstSize);

  // 13-tap box filter (Jimenez 2014)
  var color = textureSampleLevel(srcTex, samp, uv, 0.0).rgb * 0.125;

  color += textureSampleLevel(srcTex, samp, uv + vec2f(-texelSize.x, -texelSize.y), 0.0).rgb * 0.03125;
  color += textureSampleLevel(srcTex, samp, uv + vec2f( texelSize.x, -texelSize.y), 0.0).rgb * 0.03125;
  color += textureSampleLevel(srcTex, samp, uv + vec2f(-texelSize.x,  texelSize.y), 0.0).rgb * 0.03125;
  color += textureSampleLevel(srcTex, samp, uv + vec2f( texelSize.x,  texelSize.y), 0.0).rgb * 0.03125;

  color += textureSampleLevel(srcTex, samp, uv + vec2f(-texelSize.x, 0.0), 0.0).rgb * 0.0625;
  color += textureSampleLevel(srcTex, samp, uv + vec2f( texelSize.x, 0.0), 0.0).rgb * 0.0625;
  color += textureSampleLevel(srcTex, samp, uv + vec2f(0.0, -texelSize.y), 0.0).rgb * 0.0625;
  color += textureSampleLevel(srcTex, samp, uv + vec2f(0.0,  texelSize.y), 0.0).rgb * 0.0625;

  color += textureSampleLevel(srcTex, samp, uv + vec2f(-2.0 * texelSize.x, -2.0 * texelSize.y), 0.0).rgb * 0.03125;
  color += textureSampleLevel(srcTex, samp, uv + vec2f( 2.0 * texelSize.x, -2.0 * texelSize.y), 0.0).rgb * 0.03125;
  color += textureSampleLevel(srcTex, samp, uv + vec2f(-2.0 * texelSize.x,  2.0 * texelSize.y), 0.0).rgb * 0.03125;
  color += textureSampleLevel(srcTex, samp, uv + vec2f( 2.0 * texelSize.x,  2.0 * texelSize.y), 0.0).rgb * 0.03125;

  // Apply soft threshold on first downsample pass only
  if (params.flags > 0.5) {
    color = softThreshold(color, params.threshold, params.knee);
  }

  textureStore(dstTex, id.xy, vec4f(color, 1.0));
}
`;

const bloomUpsampleShader = /* wgsl */`
struct BloomParams {
  texelSize: vec2f,
  threshold: f32,
  knee: f32,
  intensity: f32,
  mipWeight: f32,
  flags: f32,
  pad: f32,
};

@group(0) @binding(0) var lowerTex: texture_2d<f32>;
@group(0) @binding(1) var currentTex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var dstTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: BloomParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dstSize = textureDimensions(dstTex);
  if (id.x >= dstSize.x || id.y >= dstSize.y) { return; }

  let texelSize = params.texelSize;  // texel size of the LOWER (smaller) mip
  let uv = (vec2f(id.xy) + 0.5) / vec2f(dstSize);

  // 9-tap tent filter on the lower (smaller) mip
  var upsampled = textureSampleLevel(lowerTex, samp, uv, 0.0).rgb * 4.0;
  upsampled += textureSampleLevel(lowerTex, samp, uv + vec2f(-texelSize.x, 0.0), 0.0).rgb * 2.0;
  upsampled += textureSampleLevel(lowerTex, samp, uv + vec2f( texelSize.x, 0.0), 0.0).rgb * 2.0;
  upsampled += textureSampleLevel(lowerTex, samp, uv + vec2f(0.0, -texelSize.y), 0.0).rgb * 2.0;
  upsampled += textureSampleLevel(lowerTex, samp, uv + vec2f(0.0,  texelSize.y), 0.0).rgb * 2.0;
  upsampled += textureSampleLevel(lowerTex, samp, uv + vec2f(-texelSize.x, -texelSize.y), 0.0).rgb;
  upsampled += textureSampleLevel(lowerTex, samp, uv + vec2f( texelSize.x, -texelSize.y), 0.0).rgb;
  upsampled += textureSampleLevel(lowerTex, samp, uv + vec2f(-texelSize.x,  texelSize.y), 0.0).rgb;
  upsampled += textureSampleLevel(lowerTex, samp, uv + vec2f( texelSize.x,  texelSize.y), 0.0).rgb;
  upsampled /= 16.0;

  // Blend with current mip's downsample data (skip connection)
  let current = textureSampleLevel(currentTex, samp, uv, 0.0).rgb;
  let result = current + upsampled * params.mipWeight;

  textureStore(dstTex, id.xy, vec4f(result, 1.0));
}
`;

const bloomCompositeShader = /* wgsl */`
@group(0) @binding(0) var sceneTex: texture_2d<f32>;
@group(0) @binding(1) var bloomTex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> intensity: f32;
const SCREEN_RADIUS = ${VISIBLE_SPHERE_RADIUS};

@vertex
fn vert(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32(i32(vi) / 2) * 4.0 - 1.0;
  let y = f32(i32(vi) % 2) * 4.0 - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn frag(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let texSize = vec2f(textureDimensions(sceneTex));
  let uv = pos.xy / texSize;
  let scene = textureSampleLevel(sceneTex, samp, uv, 0.0).rgb;
  let bloom = textureSampleLevel(bloomTex, samp, uv, 0.0).rgb;
  var color = scene + bloom * intensity;
  let aspect = texSize.x / texSize.y;
  let simUV = vec2f(
    (uv.x - 0.5) * max(aspect, 1.0) + 0.5,
    (uv.y - 0.5) * max(1.0 / aspect, 1.0) + 0.5
  );
  let dist = length(simUV - vec2f(0.5, 0.5));
  if (dist > SCREEN_RADIUS) { color = vec3f(0.0); }
  return vec4f(color, 1.0);
}
`;

function makeDisplayShaderFrag(hdr) {
  // Oklab→linear matrix: P3 for HDR, sRGB for SDR
  const M = hdr
    ? { r: '3.1272, -2.2566, 0.1294', g: '-1.0912, 2.4138, -0.3225', b: '-0.0260, -0.5082, 1.5341' }
    : { r: '4.0767416621, -3.3077115913, 0.2309699292', g: '-1.2684380046, 2.6097574011, -0.3413193965', b: '-0.0041960863, -0.7034186147, 1.7076147010' };

  return /* wgsl */`
@group(0) @binding(0) var dyeTex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct DisplayUniforms {
  screen: vec4f,      // xy=screenSize, z=time, w=sheenStrength
  baseColor: vec4f,   // xyz=baseColor RGB, w=hdrHeadroom
  accentColor: vec4f, // xyz=accentColor RGB, w=colorBlend
  sheenColor: vec4f,  // xyz=sheenColor RGB, w=metallic
  tipColor: vec4f,    // xyz=tipColor RGB, w=roughness
  _pad1: vec4f,
};
@group(0) @binding(2) var<uniform> du: DisplayUniforms;

${hdr ? `fn tonemap(x: vec3f) -> vec3f {
  let peak = 4.0;
  return x * (1.0 + x / (peak * peak)) / (1.0 + x / peak);
}` : `fn aces(x: vec3f) -> vec3f {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}`}

fn oklabToLinear(lab: vec3f) -> vec3f {
  let l = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  let m = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  let s = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
  return max(vec3f(
    ${M.r.split(',').map((c,i) => `${c.trim()} * ${['l','m','s'][i]}*${['l','m','s'][i]}*${['l','m','s'][i]}`).join(' + ')},
    ${M.g.split(',').map((c,i) => `${c.trim()} * ${['l','m','s'][i]}*${['l','m','s'][i]}*${['l','m','s'][i]}`).join(' + ')},
    ${M.b.split(',').map((c,i) => `${c.trim()} * ${['l','m','s'][i]}*${['l','m','s'][i]}*${['l','m','s'][i]}`).join(' + ')}
  ), vec3f(0.0));
}

@fragment
fn main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let screenSize = du.screen.xy;
  let time = du.screen.z;
  let sheenStrength = du.screen.w;
  let rawUV = vec2f(pos.x, screenSize.y - pos.y) / screenSize;

  // Aspect-correct UV so simulation renders as centered 1:1 square
  let aspect = screenSize.x / screenSize.y;
  let uv = vec2f(
    (rawUV.x - 0.5) * max(aspect, 1.0) + 0.5,
    (rawUV.y - 0.5) * max(1.0 / aspect, 1.0) + 0.5
  );

  // Sphere mask in simulation UV space (already 1:1)
  let centered = uv - vec2f(0.5, 0.5);
  let screenDist = length(centered);
  let screenRadius = ${VISIBLE_SPHERE_RADIUS};
  if (screenDist > screenRadius) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }

  // Sample fluid dye
  let raw = textureSampleLevel(dyeTex, samp, uv, 0.0).rgb;
  let intensity = dot(raw, vec3f(0.3, 0.6, 0.1));

  // Fluid base — gradient from accent (thin/wispy) to base (dense)
  // Colors pre-converted to Oklab on CPU
  let okBase = du.baseColor.rgb;
  let okAccent = du.accentColor.rgb;
  let blend = du.accentColor.w;
  let lo = mix(0.0, 0.35, blend);
  let hi = mix(1.0, 0.4, blend);
  let densityT = smoothstep(lo, hi, intensity);
  let okTip = du.tipColor.rgb;
  let t2 = min(densityT * 2.0, 1.0);
  let t3 = max(densityT * 2.0 - 1.0, 0.0);
  let fluidCol = oklabToLinear(mix(mix(okAccent, okBase, t2), okTip, t3));
  var color = fluidCol * intensity * ${hdr ? '0.5' : '0.25'};

  // Surface gradient for multi-lobe metallic sheen (clamp samples to sphere)
  let texel = vec2f(1.0 / 512.0);
  let uvL = uv - vec2f(texel.x * 2.0, 0.0);
  let uvR = uv + vec2f(texel.x * 2.0, 0.0);
  let uvB = uv - vec2f(0.0, texel.y * 2.0);
  let uvT = uv + vec2f(0.0, texel.y * 2.0);
  // If any sample is outside sphere, use center value instead
  let cVal = dot(raw, vec3f(0.3, 0.6, 0.1));
  let iL = select(cVal, dot(textureSampleLevel(dyeTex, samp, uvL, 0.0).rgb, vec3f(0.3, 0.6, 0.1)), length(uvL - vec2f(0.5)) < screenRadius);
  let iR = select(cVal, dot(textureSampleLevel(dyeTex, samp, uvR, 0.0).rgb, vec3f(0.3, 0.6, 0.1)), length(uvR - vec2f(0.5)) < screenRadius);
  let iB = select(cVal, dot(textureSampleLevel(dyeTex, samp, uvB, 0.0).rgb, vec3f(0.3, 0.6, 0.1)), length(uvB - vec2f(0.5)) < screenRadius);
  let iT = select(cVal, dot(textureSampleLevel(dyeTex, samp, uvT, 0.0).rgb, vec3f(0.3, 0.6, 0.1)), length(uvT - vec2f(0.5)) < screenRadius);
  let grad = vec2f(iR - iL, iT - iB);
  let gradLen = length(grad);
  let sheenDir = normalize(vec2f(0.4, 0.6));
  var spec = max(dot(normalize(grad + vec2f(0.001)), sheenDir), 0.0);
  if (du._pad1.x >= 0.5) {
    // Keep symmetry modes visually symmetric by removing directional bias.
    spec = smoothstep(0.002, 0.08, gradLen);
  }

  // Material properties
  let metallic = du.sheenColor.w;
  let roughness = du.tipColor.w;

  // Roughness controls specular exponent: smooth=tight, rough=broad
  let sharpExp = mix(16.0, 2.0, roughness);
  let broadExp = mix(6.0, 1.5, roughness);
  let sharpSheen = pow(spec, sharpExp) * smoothstep(0.003, 0.04, gradLen);
  let broadSpec = pow(spec, broadExp) * smoothstep(0.002, 0.08, gradLen);
  // Fresnel-like rim sheen (edges of sphere glow)
  let rimFactor = smoothstep(0.2, 0.40, screenDist);

  let headroom = du.baseColor.w;
  // Metallic: specular tinted by surface color vs sheen color
  let specColor = mix(du.sheenColor.rgb, color, metallic);
  let sheen = (sharpSheen * 0.6 + broadSpec * 0.3 + rimFactor * 0.15) * sheenStrength;
  color += color * sheen * specColor * headroom;

  // Tone mapping
  color = ${hdr ? 'tonemap(color * 1.6)' : 'aces(color * 1.6)'};

${hdr ? '  // HDR: browser expects linear values, handles transfer function' : `  // Fast gamma approximation (max error ~0.003 vs pow(x, 0.4545))
  { let sq = sqrt(clamp(color, vec3f(0.0), vec3f(1.0))); color = sq * (0.585 + sq * 0.415); }`}

  return vec4f(color, 1.0);
}
`;
}

// ─── Glass Shell Fragment Shader (commented out) ────────────────────────────
// ─── GPU Particle Init Shader ────────────────────────────────────────────────
function makeParticleInitShader(count) {
  return /* wgsl */`
struct Particle {
  posX: f32,
  posY: f32,
  normalX: f32,
  normalY: f32,
  normalZ: f32,
  angularVel: f32,
  life: f32,
  seed: f32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

fn pcg(inp: u32) -> u32 {
  var state = inp * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn randF(seed: u32) -> f32 {
  return f32(pcg(seed)) / 4294967295.0;
}

@compute @workgroup_size(${PARTICLE_WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${count}u) { return; }
  let idx = id.x;

  let h1 = pcg(idx * 7919u + 1234567u);
  let h2 = pcg(h1);
  let h3 = pcg(h2);
  let h4 = pcg(h3);
  let h5 = pcg(h4);
  let h6 = pcg(h5);
  let h7 = pcg(h6);

  let angle = randF(h1) * 6.2831853;
  let radius = sqrt(randF(h2)) * 0.43 * 0.9;

  var part: Particle;
  part.posX = 0.5 + cos(angle) * radius;
  part.posY = 0.5 + sin(angle) * radius;

  let phi = randF(h3) * 6.2831853;
  let cosTheta = randF(h4);
  let sinTheta = sqrt(1.0 - cosTheta * cosTheta);
  part.normalX = sinTheta * cos(phi);
  part.normalY = sinTheta * sin(phi);
  part.normalZ = max(cosTheta, 0.3);
  let nL = length(vec3f(part.normalX, part.normalY, part.normalZ));
  part.normalX /= nL;
  part.normalY /= nL;
  part.normalZ /= nL;

  part.angularVel = (randF(h5) - 0.5) * 2.0;
  part.life = randF(h6) * 0.5;
  part.seed = randF(h7);

  particles[idx] = part;
}
`;
}

// ─── CPU-side Oklab conversion (avoids per-particle GPU conversion) ──────────
function linearToOklabCPU(rgb) {
  const r = rgb[0], g = rgb[1], b = rgb[2];
  const l = Math.cbrt(Math.max(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b, 0));
  const m = Math.cbrt(Math.max(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b, 0));
  const s = Math.cbrt(Math.max(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b, 0));
  return [
    0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
  ];
}

// ─── Bayesian Optimization ───────────────────────────────────────────────────
import { BOController, normalizedToState, stateToNormalized, SLIDER_KEYS, SLIDER_SPACE, COLOR_KEYS, D } from './bo.js';

// ─── WebGPU Init ─────────────────────────────────────────────────────────────

const GPU_FAST_TIMEOUT_MS = 2500;
const GPU_ADAPTER_ATTEMPT_TIMEOUT_MS = 8000;
const GPU_ADAPTER_OVERALL_TIMEOUT_MS = 15000;
const GPU_DEVICE_FAST_TIMEOUT_MS = 5000;
const GPU_DEVICE_ATTEMPT_TIMEOUT_MS = 7000;
const GPU_DEVICE_RACE_TIMEOUT_MS = 9000;
const GPU_DEVICE_WAKE_RACE_TIMEOUT_MS = 7000;
const GPU_DEVICE_EXTENDED_WAIT_MS = 25000;
const GPU_WAKE_EVENT_TIMEOUT_MS = 8000;
const GPU_INIT_DIAG_SAMPLE_MS = 3000;
const MB = 1024 * 1024;
const GPU_DEBUG_STORAGE_KEY = 'fluid.gpu_debug';
const GPU_TRACE_LIMIT = 240;
const gpuInitSessionId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
const gpuInitTraceT0 = performance.now();
const gpuInitTrace = [];

const gpuDebugEnabled = (() => {
  try {
    const q = new URLSearchParams(location.search);
    const flag = q.get('gpu_debug');
    if (flag === '1') {
      sessionStorage.setItem(GPU_DEBUG_STORAGE_KEY, '1');
      return true;
    }
    if (flag === '0') {
      sessionStorage.removeItem(GPU_DEBUG_STORAGE_KEY);
      return false;
    }
    return sessionStorage.getItem(GPU_DEBUG_STORAGE_KEY) === '1';
  } catch {
    return false;
  }
})();

function lifecycleSnapshot() {
  let hasFocus = false;
  try { hasFocus = document.hasFocus(); } catch {}
  return {
    visibility: document.visibilityState,
    hidden: document.hidden,
    hasFocus,
    wasDiscarded: !!document.wasDiscarded,
    prerendering: !!document.prerendering,
  };
}

function normalizeTraceFields(fields) {
  if (!fields || typeof fields !== 'object') return null;
  const out = {};
  for (const [key, raw] of Object.entries(fields)) {
    if (raw == null || typeof raw === 'string' || typeof raw === 'number' || typeof raw === 'boolean') {
      out[key] = raw;
      continue;
    }
    if (Array.isArray(raw)) {
      out[key] = raw.slice(0, 8).map(v => (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') ? v : String(v));
      continue;
    }
    out[key] = String(raw);
  }
  return out;
}

function traceInit(event, fields = null) {
  const entry = {
    session: gpuInitSessionId,
    tMs: Number((performance.now() - gpuInitTraceT0).toFixed(1)),
    event,
    ...(normalizeTraceFields(fields) || {}),
  };
  gpuInitTrace.push(entry);
  if (gpuInitTrace.length > GPU_TRACE_LIMIT) gpuInitTrace.shift();
  if (gpuDebugEnabled) {
    console.debug(`InitTrace +${entry.tMs}ms ${event}`, entry);
  }
  return entry;
}

function buildGpuInitSummary(reason, extra = null) {
  return {
    reason,
    session: gpuInitSessionId,
    elapsedMs: Number((performance.now() - gpuInitTraceT0).toFixed(1)),
    lifecycle: lifecycleSnapshot(),
    extra: normalizeTraceFields(extra),
    recentTrace: gpuInitTrace.slice(-50),
  };
}

function dumpGpuInitSummary(reason, extra = null) {
  const summary = buildGpuInitSummary(reason, extra);
  console.warn('Init diagnostics summary:', summary);
  return summary;
}

if (typeof window !== 'undefined') {
  window.dumpGpuInitSummary = (reason = 'manual', extra = null) => buildGpuInitSummary(reason, extra);
  window.dumpGpuInitTrace = () => gpuInitTrace.slice();
}

traceInit('session-start', { href: location.href, ua: navigator.userAgent, ...lifecycleSnapshot() });
document.addEventListener('visibilitychange', () => traceInit('evt-visibilitychange', lifecycleSnapshot()), true);
window.addEventListener('focus', () => traceInit('evt-focus', lifecycleSnapshot()), true);
window.addEventListener('blur', () => traceInit('evt-blur', lifecycleSnapshot()), true);
window.addEventListener('pageshow', (e) => traceInit('evt-pageshow', { persisted: !!e.persisted, ...lifecycleSnapshot() }), true);
window.addEventListener('pagehide', (e) => traceInit('evt-pagehide', { persisted: !!e.persisted, ...lifecycleSnapshot() }), true);
document.addEventListener('freeze', () => traceInit('evt-freeze', lifecycleSnapshot()), true);
document.addEventListener('resume', () => traceInit('evt-resume', lifecycleSnapshot()), true);

function timeoutValue(ms, value = null) {
  return new Promise(resolve => setTimeout(() => resolve(value), ms));
}

function withTimeout(promise, ms, value = null) {
  return Promise.race([promise, timeoutValue(ms, value)]);
}

function waitForWakeEvent(timeoutMs = GPU_WAKE_EVENT_TIMEOUT_MS) {
  return new Promise(resolve => {
    let done = false;
    const cleanup = () => {
      window.removeEventListener('focus', onFocus, true);
      window.removeEventListener('resize', onResize, true);
      document.removeEventListener('visibilitychange', onVisibility, true);
      window.removeEventListener('pointerdown', onPointer, true);
      window.removeEventListener('keydown', onKey, true);
      if (timer) clearTimeout(timer);
    };
    const finish = (reason) => {
      if (done) return;
      done = true;
      cleanup();
      resolve(reason);
    };
    const onFocus = () => finish('focus');
    const onResize = () => finish('resize');
    const onPointer = () => finish('pointer');
    const onKey = () => finish('key');
    const onVisibility = () => {
      if (document.visibilityState === 'visible') finish('visible');
    };
    window.addEventListener('focus', onFocus, true);
    window.addEventListener('resize', onResize, true);
    document.addEventListener('visibilitychange', onVisibility, true);
    window.addEventListener('pointerdown', onPointer, true);
    window.addEventListener('keydown', onKey, true);
    const timer = timeoutMs > 0 ? setTimeout(() => finish(null), timeoutMs) : null;
  });
}

function pageIsInteractive() {
  let hasFocus = false;
  try { hasFocus = document.hasFocus(); } catch {}
  return document.visibilityState === 'visible' && hasFocus;
}

async function waitForInteractivePage(loadStatus, phase, timeoutMs = 8000) {
  if (pageIsInteractive()) {
    traceInit('interactive-ready', { phase, ...lifecycleSnapshot() });
    return true;
  }

  traceInit('interactive-wait-start', { phase, timeoutMs, ...lifecycleSnapshot() });
  if (loadStatus) loadStatus.textContent = `Waiting for active tab (${phase})...`;

  return new Promise(resolve => {
    let done = false;
    const started = performance.now();
    const cleanup = () => {
      window.removeEventListener('focus', onFocus, true);
      window.removeEventListener('pageshow', onPageShow, true);
      document.removeEventListener('visibilitychange', onVisibility, true);
      if (timer) clearTimeout(timer);
    };
    const finish = (ok, reason) => {
      if (done) return;
      done = true;
      cleanup();
      traceInit('interactive-wait-end', {
        phase,
        ok,
        reason,
        ms: Number((performance.now() - started).toFixed(1)),
        ...lifecycleSnapshot(),
      });
      resolve(ok);
    };
    const check = (reason) => {
      if (pageIsInteractive()) finish(true, reason);
    };
    const onFocus = () => check('focus');
    const onPageShow = () => check('pageshow');
    const onVisibility = () => check('visibilitychange');
    window.addEventListener('focus', onFocus, true);
    window.addEventListener('pageshow', onPageShow, true);
    document.addEventListener('visibilitychange', onVisibility, true);
    const timer = timeoutMs > 0 ? setTimeout(() => finish(false, 'timeout'), timeoutMs) : null;
    check('initial');
  });
}

async function requestAdapterAdaptive(loadStatus) {
  const q = new URLSearchParams(location.search);
  const gpuPref = q.get('gpu');
  const adapterT0 = performance.now();
  const attempts = gpuPref === 'high'
    ? [
      { label: 'high-performance', options: { powerPreference: 'high-performance' } },
      { label: 'default', options: {} },
      { label: 'low-power', options: { powerPreference: 'low-power' } },
    ]
    : gpuPref === 'low'
      ? [
        { label: 'low-power', options: { powerPreference: 'low-power' } },
        { label: 'default', options: {} },
        { label: 'high-performance', options: { powerPreference: 'high-performance' } },
      ]
      : [
        { label: 'default', options: {} },
        { label: 'low-power', options: { powerPreference: 'low-power' } },
        { label: 'high-performance', options: { powerPreference: 'high-performance' } },
      ];

  const fast = attempts[0];
  traceInit('adapter-request-start', { first: fast.label, gpuPref: gpuPref || 'default-order', ...lifecycleSnapshot() });
  if (loadStatus) loadStatus.textContent = `Requesting GPU adapter (${fast.label})...`;
  console.log(`Init: requesting GPU adapter (${fast.label} first)...`);
  const fastPromise = navigator.gpu.requestAdapter(fast.options).catch(() => null);
  const fastAdapter = await withTimeout(fastPromise, GPU_FAST_TIMEOUT_MS, null);
  if (fastAdapter) {
    traceInit('adapter-fast-success', {
      mode: fast.label,
      ms: Number((performance.now() - adapterT0).toFixed(1)),
      ...lifecycleSnapshot(),
    });
    return { adapter: fastAdapter, mode: fast.label, fallbackRace: false };
  }

  traceInit('adapter-fast-timeout', {
    mode: fast.label,
    timeoutMs: GPU_FAST_TIMEOUT_MS,
    ...lifecycleSnapshot(),
  });
  console.warn(`Init: adapter request exceeded ${GPU_FAST_TIMEOUT_MS}ms; racing adapter preferences...`);
  if (loadStatus) loadStatus.textContent = 'Waking GPU (trying fallbacks)...';

  const deadline = performance.now() + GPU_ADAPTER_OVERALL_TIMEOUT_MS;
  const remainingForFast = Math.max(0, deadline - performance.now());
  const lateFast = await withTimeout(fastPromise, remainingForFast, null);
  if (lateFast) {
    traceInit('adapter-race-success', {
      mode: fast.label,
      ms: Number((performance.now() - adapterT0).toFixed(1)),
      lateFast: true,
      ...lifecycleSnapshot(),
    });
    return { adapter: lateFast, mode: fast.label, fallbackRace: true };
  }

  for (const attempt of attempts.slice(1)) {
    const remaining = Math.max(0, deadline - performance.now());
    if (remaining <= 0) break;
    traceInit('adapter-fallback-start', { mode: attempt.label, remainingMs: Number(remaining.toFixed(1)) });
    const adapter = await withTimeout(
      navigator.gpu.requestAdapter(attempt.options).catch(() => null),
      Math.min(GPU_ADAPTER_ATTEMPT_TIMEOUT_MS, remaining),
      null
    );
    if (adapter) {
      traceInit('adapter-race-success', {
        mode: attempt.label,
        ms: Number((performance.now() - adapterT0).toFixed(1)),
        ...lifecycleSnapshot(),
      });
      return { adapter, mode: attempt.label, fallbackRace: true };
    }
  }

  const remainingForFallback = Math.max(0, deadline - performance.now());
  if (remainingForFallback > 0) {
    traceInit('adapter-fallback-start', { mode: 'fallback-software', remainingMs: Number(remainingForFallback.toFixed(1)) });
    const fallbackAdapter = await withTimeout(
      navigator.gpu.requestAdapter({ forceFallbackAdapter: true }).catch(() => null),
      Math.min(GPU_ADAPTER_ATTEMPT_TIMEOUT_MS, remainingForFallback),
      null
    );
    if (fallbackAdapter) {
      traceInit('adapter-race-success', {
        mode: 'fallback-software',
        ms: Number((performance.now() - adapterT0).toFixed(1)),
        ...lifecycleSnapshot(),
      });
      return { adapter: fallbackAdapter, mode: 'fallback-software', fallbackRace: true };
    }
  }

  traceInit('adapter-race-failed', {
    timeoutMs: GPU_ADAPTER_OVERALL_TIMEOUT_MS,
    ms: Number((performance.now() - adapterT0).toFixed(1)),
    ...lifecycleSnapshot(),
  });
  return null;
}

function requestDeviceOnAdapter(adapter, requestMaxLimits) {
  if (requestMaxLimits) {
    return adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1024 * MB),
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1024 * MB),
      },
    });
  }
  return adapter.requestDevice();
}

async function requestDeviceAdaptive(primaryAdapter, primaryMode, loadStatus, requestMaxLimits) {
  const started = performance.now();
  let statusTimer = null;
  let wakePokeTimer = null;
  let pendingDiagTimer = null;
  const pending = Symbol('pending');
  let statusStage = requestMaxLimits
    ? 'Acquiring GPU device (max limits)...'
    : 'Acquiring GPU device...';
  const setStatusStage = (text) => {
    statusStage = text;
    if (loadStatus) {
      const secs = Math.round((performance.now() - started) / 1000);
      loadStatus.textContent = `${statusStage} ${secs}s`;
    }
  };
  if (loadStatus) {
    loadStatus.textContent = statusStage;
    statusTimer = setInterval(() => {
      const secs = Math.round((performance.now() - started) / 1000);
      loadStatus.textContent = `${statusStage} ${secs}s`;
    }, 1000);
  }
  traceInit('device-request-start', {
    mode: primaryMode,
    requestMaxLimits,
    ...lifecycleSnapshot(),
  });

  let primaryRawPromise = null;
  try {
    console.log(`Init: invoking adapter.requestDevice() on ${primaryMode}...`);
    traceInit('device-primary-dispatch', { mode: primaryMode, requestMaxLimits });
    primaryRawPromise = requestDeviceOnAdapter(primaryAdapter, requestMaxLimits);
    console.log('Init: adapter.requestDevice() returned a pending promise.');
  } catch (err) {
    traceInit('device-primary-sync-throw', { mode: primaryMode, error: err?.message || String(err) });
    console.warn('Init: adapter.requestDevice() threw synchronously:', err?.message || err);
  }
  const primaryPromise = (primaryRawPromise || Promise.resolve(null))
    .then(device => {
      traceInit('device-primary-resolved', {
        mode: primaryMode,
        ok: !!device,
        ms: Number((performance.now() - started).toFixed(1)),
      });
      return device ? { device, mode: primaryMode, downgradedLimits: false } : null;
    })
    .catch(err => {
      traceInit('device-primary-error', {
        mode: primaryMode,
        error: err?.message || String(err),
        ms: Number((performance.now() - started).toFixed(1)),
      });
      console.warn('Init: primary device request failed:', err?.message || err);
      return null;
    });

  async function tryFallbackMode(mode, context = 'fallback', adapterOptions = null) {
    const fallbackT0 = performance.now();
    traceInit('device-fallback-start', {
      mode,
      context,
      forceFallbackAdapter: !!(adapterOptions && adapterOptions.forceFallbackAdapter),
      ...lifecycleSnapshot(),
    });
    const options = adapterOptions || (mode === 'default' ? {} : { powerPreference: mode });
    const adapter = await withTimeout(
      navigator.gpu.requestAdapter(options).catch(() => null),
      GPU_ADAPTER_ATTEMPT_TIMEOUT_MS,
      null
    );
    if (!adapter) {
      traceInit('device-fallback-no-adapter', {
        mode,
        context,
        ms: Number((performance.now() - fallbackT0).toFixed(1)),
      });
      return null;
    }
    const device = await withTimeout(
      requestDeviceOnAdapter(adapter, false).catch(() => null),
      GPU_DEVICE_ATTEMPT_TIMEOUT_MS,
      null
    );
    if (!device) {
      traceInit('device-fallback-no-device', {
        mode,
        context,
        ms: Number((performance.now() - fallbackT0).toFixed(1)),
      });
      return null;
    }
    console.warn(`Init: ${context} acquired device on ${mode}.`);
    traceInit('device-fallback-success', {
      mode,
      context,
      ms: Number((performance.now() - fallbackT0).toFixed(1)),
      ...lifecycleSnapshot(),
    });
    return { device, mode, downgradedLimits: requestMaxLimits };
  }

  async function tryFallbackModesSequential(modes, context = 'fallback', maxWaitMs = GPU_DEVICE_RACE_TIMEOUT_MS, includeSoftware = false) {
    const deadline = performance.now() + maxWaitMs;
    const ordered = includeSoftware ? [...modes, 'fallback-software'] : [...modes];
    for (const mode of ordered) {
      const remaining = Math.max(0, deadline - performance.now());
      if (remaining <= 0) break;
      const attemptPromise = mode === 'fallback-software'
        ? tryFallbackMode('fallback-software', context, { forceFallbackAdapter: true })
        : tryFallbackMode(mode, context);
      const got = await withTimeout(attemptPromise, remaining, null);
      if (got) return got;
      const maybePrimary = await withTimeout(primaryPromise, 0, pending);
      if (maybePrimary !== pending) return maybePrimary;
    }
    const maybePrimary = await withTimeout(primaryPromise, 0, pending);
    return maybePrimary === pending ? null : maybePrimary;
  }

  try {
    const fastDevice = await withTimeout(primaryPromise, GPU_DEVICE_FAST_TIMEOUT_MS, pending);
    if (fastDevice !== pending) {
      if (!fastDevice) return null;
      traceInit('device-fast-success', {
        mode: fastDevice.mode,
        ms: Number((performance.now() - started).toFixed(1)),
      });
      return {
        ...fastDevice,
        fallbackRace: false,
      };
    }

    traceInit('device-fast-timeout', {
      timeoutMs: GPU_DEVICE_FAST_TIMEOUT_MS,
      ...lifecycleSnapshot(),
    });
    console.warn(`Init: device request exceeded ${GPU_DEVICE_FAST_TIMEOUT_MS}ms; trying fallback adapters...`);
    setStatusStage('Acquiring GPU device (racing adapters)...');
    // Some browser/driver states recover after a synthetic resize + animation tick.
    wakePokeTimer = setInterval(() => {
      try { window.dispatchEvent(new Event('resize')); } catch {}
      try { requestAnimationFrame(() => {}); } catch {}
    }, 1500);
    pendingDiagTimer = setInterval(() => {
      const snap = lifecycleSnapshot();
      traceInit('device-pending', snap);
      console.warn('Init: device still pending...', snap);
    }, GPU_INIT_DIAG_SAMPLE_MS);

    const fallbackModes = ['default', 'low-power', 'high-performance'].filter(m => m !== primaryMode);
    const fallbackRace = await tryFallbackModesSequential(fallbackModes, 'fallback-race', GPU_DEVICE_RACE_TIMEOUT_MS, true);
    if (fallbackRace) {
      if (fallbackRace.mode === 'fallback-software') {
        console.warn('Init: using software fallback adapter/device.');
      }
      traceInit('device-race-success', {
        mode: fallbackRace.mode,
        ms: Number((performance.now() - started).toFixed(1)),
      });
      return { ...fallbackRace, fallbackRace: true };
    }
    traceInit('device-race-timeout', {
      timeoutMs: GPU_DEVICE_RACE_TIMEOUT_MS,
      ms: Number((performance.now() - started).toFixed(1)),
      ...lifecycleSnapshot(),
    });

    const wakeOrPrimary = await Promise.race([
      waitForWakeEvent(),
      withTimeout(primaryPromise, GPU_WAKE_EVENT_TIMEOUT_MS, pending),
    ]);
    if (wakeOrPrimary && typeof wakeOrPrimary === 'object' && wakeOrPrimary.device) {
      traceInit('device-primary-won-wake-race', {
        mode: wakeOrPrimary.mode,
        ms: Number((performance.now() - started).toFixed(1)),
      });
      return { ...wakeOrPrimary, fallbackRace: true };
    }
    const wakeReason = typeof wakeOrPrimary === 'string' ? wakeOrPrimary : null;
    if (wakeReason) {
      traceInit('device-wake-event', { wakeReason, ...lifecycleSnapshot() });
      console.warn(`Init: wake event '${wakeReason}' detected while device pending; retrying device acquisition...`);
      setStatusStage(`Retrying GPU device (${wakeReason})...`);
      const wakeRace = await tryFallbackModesSequential(
        ['default', 'low-power', 'high-performance'],
        `wake-${wakeReason}`,
        GPU_DEVICE_WAKE_RACE_TIMEOUT_MS,
        true
      );
      if (wakeRace) {
        if (wakeRace.mode === 'fallback-software') {
          console.warn(`Init: wake-${wakeReason} using software fallback adapter/device.`);
        }
        traceInit('device-wake-race-success', {
          wakeReason,
          mode: wakeRace.mode,
          ms: Number((performance.now() - started).toFixed(1)),
        });
        return { ...wakeRace, fallbackRace: true };
      }
      traceInit('device-wake-race-timeout', {
        wakeReason,
        timeoutMs: GPU_DEVICE_WAKE_RACE_TIMEOUT_MS,
        ms: Number((performance.now() - started).toFixed(1)),
      });
    }

    setStatusStage('Still acquiring GPU device...');
    const latePrimary = await withTimeout(primaryPromise, GPU_DEVICE_EXTENDED_WAIT_MS, pending);
    if (latePrimary !== pending) {
      if (!latePrimary) return null;
      traceInit('device-late-primary-success', {
        mode: latePrimary.mode,
        ms: Number((performance.now() - started).toFixed(1)),
      });
      return { ...latePrimary, fallbackRace: true };
    }
    traceInit('device-timeout', {
      ms: Number((performance.now() - started).toFixed(1)),
      ...lifecycleSnapshot(),
    });
    console.error(`Init: device acquisition timed out after ~${GPU_DEVICE_FAST_TIMEOUT_MS + GPU_DEVICE_RACE_TIMEOUT_MS + GPU_DEVICE_WAKE_RACE_TIMEOUT_MS + GPU_DEVICE_EXTENDED_WAIT_MS}ms.`);
    dumpGpuInitSummary('device-timeout', { primaryMode, requestMaxLimits });
    return null;
  } finally {
    if (statusTimer) clearInterval(statusTimer);
    if (wakePokeTimer) clearInterval(wakePokeTimer);
    if (pendingDiagTimer) clearInterval(pendingDiagTimer);
  }
}

async function main() {
  const t0 = performance.now();
  const loadStatus = document.getElementById('loadStatus');
  const canvas = document.getElementById('canvas');
  const errorDiv = document.getElementById('error');

  if (!navigator.gpu) {
    errorDiv.style.display = 'block';
    errorDiv.textContent = 'WebGPU not supported in this browser.';
    return;
  }

  const boT0 = performance.now();
  console.log('Init: creating BO controller...');
  const boPromise = BOController.create();

  const adapterInteractive = await waitForInteractivePage(loadStatus, 'before-adapter', 8000);
  if (!adapterInteractive) {
    console.warn('Init: tab did not become fully active before adapter request; proceeding anyway.');
  }

  const gpuT0 = performance.now();
  const adapterResult = await requestAdapterAdaptive(loadStatus);
  const adapter = adapterResult?.adapter || null;
  console.log(`Init: adapter acquired (${(performance.now() - gpuT0).toFixed(0)}ms, mode=${adapterResult?.mode || 'unknown'}${adapterResult?.fallbackRace ? ', raced' : ''})`);
  traceInit('adapter-acquired', {
    mode: adapterResult?.mode || 'unknown',
    raced: !!adapterResult?.fallbackRace,
    ms: Number((performance.now() - gpuT0).toFixed(1)),
    ...lifecycleSnapshot(),
  });
  if (!adapter) {
    errorDiv.style.display = 'block';
    errorDiv.textContent = 'No WebGPU adapter found.';
    console.error('No WebGPU adapter');
    dumpGpuInitSummary('no-adapter');
    return;
  }
  let adapterInfo = null;
  try {
    adapterInfo = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : null);
  } catch {}
  if (adapterInfo) {
    const infoLog = {
      vendor: adapterInfo.vendor || 'unknown',
      architecture: adapterInfo.architecture || 'unknown',
      description: adapterInfo.description || 'unknown',
      isFallbackAdapter: !!adapterInfo.isFallbackAdapter,
    };
    console.log('Init: adapter info', infoLog);
    traceInit('adapter-info', infoLog);
  } else {
    traceInit('adapter-info-unavailable');
  }

  const bo = await boPromise;
  console.log(`Init: BO controller ready (${(performance.now() - boT0).toFixed(0)}ms)`);

  const q = new URLSearchParams(location.search);
  const requestMaxLimits = q.get('gpu_limits') === 'max' || q.get('gpuLimits') === 'max';
  const deviceInteractive = await waitForInteractivePage(loadStatus, 'before-device', 10000);
  if (!deviceInteractive) {
    console.warn('Init: tab did not become fully active before device request; proceeding anyway.');
  }
  // Yield one paint before requestDevice; helps avoid hard-reload GPU wake stalls.
  await new Promise(resolve => requestAnimationFrame(resolve));
  await timeoutValue(0);
  traceInit('visibility-ready', lifecycleSnapshot());

  console.log(requestMaxLimits
    ? 'Init: requesting GPU device (max limits mode)...'
    : 'Init: requesting GPU device (default limits for fast startup)...');
  const deviceT0 = performance.now();
  const deviceResult = await requestDeviceAdaptive(adapter, adapterResult?.mode || 'default', loadStatus, requestMaxLimits);
  const device = deviceResult?.device || null;
  if (!device) {
    errorDiv.style.display = 'block';
    errorDiv.textContent = 'Failed to acquire WebGPU device.';
    console.error('No WebGPU device acquired.');
    dumpGpuInitSummary('no-device', { adapterMode: adapterResult?.mode || 'unknown' });
    return;
  }
  console.log(
    `Init: device acquired (${(performance.now() - deviceT0).toFixed(0)}ms, total GPU: ${(performance.now() - gpuT0).toFixed(0)}ms, mode=${deviceResult.mode}${deviceResult.fallbackRace ? ', raced' : ''}${deviceResult.downgradedLimits ? ', downgraded-limits' : ''})`
  );
  traceInit('device-acquired', {
    mode: deviceResult.mode,
    raced: !!deviceResult.fallbackRace,
    downgradedLimits: !!deviceResult.downgradedLimits,
    deviceMs: Number((performance.now() - deviceT0).toFixed(1)),
    totalGpuMs: Number((performance.now() - gpuT0).toFixed(1)),
    ...lifecycleSnapshot(),
  });
  if (deviceResult.downgradedLimits) {
    console.warn('Init: max-limit request downgraded to default limits to avoid long device stall.');
  }
  device.lost.then(info => {
    console.error('WebGPU device lost:', info.message);
    dumpGpuInitSummary('device-lost', {
      reason: info.reason || 'unknown',
      message: info.message || 'unknown',
    });
  });

  // GPU queue depth tracking — cap pending frames to prevent unbounded queue buildup
  let frameRunning = true;
  let gpuFramesPending = 0;
  let gpuFramesSkipped = 0;

  // Stop render loop + destroy device on unload. Queue cap limits pending GPU work
  // to ~2 frames so destroy() drains in <1ms instead of 17s.
  window.addEventListener('beforeunload', () => {
    frameRunning = false;
    console.log(`beforeunload: gpuFramesPending=${gpuFramesPending}, framesSkipped=${gpuFramesSkipped}`);
    device.destroy();
  });

  // Derive max particle count from granted limits
  const maxParticles = Math.min(16777216, Math.floor(device.limits.maxStorageBufferBindingSize / PARTICLE_STRIDE));
  state.particleCount = Math.min(state.particleCount, maxParticles);
  loadStatus.textContent = 'Compiling shaders...';
  device.pushErrorScope('validation');
  device.pushErrorScope('internal');
  console.log(`Init: WebGPU device acquired (${(performance.now() - t0).toFixed(0)}ms)`);
  console.log(`GPU limits: maxStorageBuffer=${(device.limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)}MB, maxParticles=${maxParticles}`);

  const ctx = canvas.getContext('webgpu');

  // Feature-detect HDR canvas support
  const hdrSupported = (() => {
    try {
      const tc = document.createElement('canvas');
      const tx = tc.getContext('webgpu');
      if (!tx) return false;
      tx.configure({
        device, format: 'rgba16float',
        colorSpace: 'display-p3',
        toneMapping: { mode: 'extended' },
        alphaMode: 'opaque',
      });
      tx.unconfigure();
      return true;
    } catch { return false; }
  })();

  const canvasFmt = hdrSupported ? 'rgba16float' : navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({
    device, format: canvasFmt, alphaMode: 'opaque',
    ...(hdrSupported && {
      colorSpace: 'display-p3',
      toneMapping: { mode: 'extended' },
    }),
  });
  console.log(`HDR=${hdrSupported}, format=${canvasFmt}${hdrSupported ? ', colorSpace=display-p3' : ''}`);

  // Expand BO color exploration range for HDR (exclude sheenColor)
  // Capped at 1.2 to avoid oversaturated/blown-out colors in ratings data
  if (hdrSupported) {
    for (const key of Object.keys(SLIDER_SPACE)) {
      if (key.match(/Color_[012]$|Accent_[012]$|Tip_[012]$/) && !key.startsWith('sheen')) {
        SLIDER_SPACE[key].max = 1.2;
      }
    }
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || window.innerWidth || 1;
    const cssH = canvas.clientHeight || window.innerHeight || 1;
    canvas.width = Math.max(1, Math.round(cssW * dpr));
    canvas.height = Math.max(1, Math.round(cssH * dpr));
  }
  resize();
  window.addEventListener('resize', resize);

  const dispatch = Math.ceil(SIM_RES / WORKGROUP);

  // Capture GPU errors
  device.addEventListener('uncapturederror', e => {
    console.error('GPU error:', e.error.message);
  });

  // ─── Textures ────────────────────────────────────────────────────────────
  function makeTex(label) {
    return device.createTexture({
      label,
      size: [SIM_RES, SIM_RES],
      format: TEX_FMT,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });
  }

  let velA = makeTex('velA'), velB = makeTex('velB');
  let pressA = makeTex('pressA'), pressB = makeTex('pressB');
  const divTex = makeTex('divergence');
  let dyeA = makeTex('dyeA'), dyeB = makeTex('dyeB');
  const curlTex = makeTex('curl');
  let tempA = makeTex('tempA'), tempB = makeTex('tempB');

  const linearSampler = device.createSampler({
    minFilter: 'linear', magFilter: 'linear',
    addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
  });

  // ─── Uniform buffer (simulation params) ────────────────────────────────
  const paramBuf = device.createBuffer({
    size: 64, // 16 floats × 4 bytes
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const paramData = new Float32Array(16);

  function writeParams(overrides = {}, buf = paramBuf) {
    const d = { dt: 0.016, dx: 1 / SIM_RES, simRes: SIM_RES, time: 0,
      splatX: 0, splatY: 0, splatDx: 0, splatDy: 0,
      splatR: 0, splatG: 0, splatB: 0,
      splatRadius: state.splatRadius, curlStrength: state.curlStrength,
      pressureDecay: state.pressureDecay,
      velDissipation: state.velDissipation, dyeDissipation: state.dyeDissipation,
      ...overrides };
    paramData[0]  = d.dt;
    paramData[1]  = d.dx;
    paramData[2]  = d.simRes;
    paramData[3]  = d.time;
    paramData[4]  = d.splatX;
    paramData[5]  = d.splatY;
    paramData[6]  = d.splatDx;
    paramData[7]  = d.splatDy;
    paramData[8]  = d.splatR;
    paramData[9]  = d.splatG;
    paramData[10] = d.splatB;
    paramData[11] = d.splatRadius;
    paramData[12] = d.curlStrength;
    paramData[13] = d.pressureDecay;
    paramData[14] = d.velDissipation;
    paramData[15] = d.dyeDissipation;
    device.queue.writeBuffer(buf, 0, paramData);
  }

  // ─── Split screen uniforms: particle UB + display UB ───────────────────
  // particleUB: [width, height, particleSize, glintBrightness, prismaticAmount, glitR, glitG, glitB, accentGlitR, accentGlitG, accentGlitB, colorBlend, sizeRandomness, pad, pad, pad]
  const particleUB = device.createBuffer({
    size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const particleUBData = new Float32Array(16);

  // displayUB: [width, height, time, sheenStrength, baseR, baseG, baseB, pad, accentR, accentG, accentB, colorBlend, sheenR, sheenG, sheenB, pad, tipR, tipG, tipB, pad, symmetryBehavior, pad, pad, pad]
  const displayUB = device.createBuffer({
    size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const displayUBData = new Float32Array(24);
  const bloomParamData = new Float32Array(8);
  const bloomCompositeData = new Float32Array(1);

  // ─── Pipeline helpers ───────────────────────────────────────────────────
  function checkShader(module, label) {
    if (module.getCompilationInfo) {
      module.getCompilationInfo().then(info => {
        for (const msg of info.messages) {
          console.error(`[WGSL ${msg.type}] ${label}: ${msg.message} (line ${msg.lineNum}:${msg.linePos})`);
        }
      });
    }
  }
  function buildPipeline(code, label, bindingDescs) {
    const module = device.createShaderModule({ code, label });
    checkShader(module, label);
    const entries = bindingDescs.map((desc, i) => {
      const e = { binding: i, visibility: GPUShaderStage.COMPUTE };
      if (desc === 'uniform') e.buffer = { type: 'uniform' };
      else if (desc === 'texture') e.texture = { sampleType: 'float' };
      else if (desc === 'storage') e.storageTexture = { access: 'write-only', format: TEX_FMT };
      else if (desc === 'sampler') e.sampler = {};
      return e;
    });
    const layout = device.createBindGroupLayout({ entries, label: label + '_bgl' });
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: 'main' },
    });
    return { pipeline, layout };
  }

  // Batch splat pipelines (shared layout: uniform, texture, storage-tex, storage-buf, uniform)
  const batchSplatBGL = device.createBindGroupLayout({
    label: 'batchSplat_bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: TEX_FMT } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const batchSplatLayout = device.createPipelineLayout({ bindGroupLayouts: [batchSplatBGL] });

  const batchSplatVelPipe = device.createComputePipeline({
    label: 'batchSplatVel',
    layout: batchSplatLayout,
    compute: {
      module: device.createShaderModule({ code: batchSplatShaderVel, label: 'batchSplatVel' }),
      entryPoint: 'main',
    },
  });

  const batchSplatDyePipe = device.createComputePipeline({
    label: 'batchSplatDye',
    layout: batchSplatLayout,
    compute: {
      module: device.createShaderModule({ code: batchSplatShaderDye, label: 'batchSplatDye' }),
      entryPoint: 'main',
    },
  });

  // Fused curl + vorticity: uniform, texture(vel), storage(curlDst), storage(velDst)
  const fusedCurlVortBGL = device.createBindGroupLayout({
    label: 'fusedCurlVort_bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: TEX_FMT } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: TEX_FMT } },
    ],
  });
  const fusedCurlVortModule = device.createShaderModule({ code: fusedCurlVortShader, label: 'fusedCurlVort' });
  checkShader(fusedCurlVortModule, 'fusedCurlVort');
  const fusedCurlVortPipe = device.createComputePipeline({
    label: 'fusedCurlVort',
    layout: device.createPipelineLayout({ bindGroupLayouts: [fusedCurlVortBGL] }),
    compute: { module: fusedCurlVortModule, entryPoint: 'main' },
  });

  // Fused divergence + clear pressure: uniform, texture(vel), texture(press), storage(divDst), storage(pressDst)
  const fusedDivClearPressBGL = device.createBindGroupLayout({
    label: 'fusedDivClearPress_bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: TEX_FMT } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: TEX_FMT } },
    ],
  });
  const fusedDivClearPressModule = device.createShaderModule({ code: fusedDivClearPressShader, label: 'fusedDivClearPress' });
  checkShader(fusedDivClearPressModule, 'fusedDivClearPress');
  const fusedDivClearPressPipe = device.createComputePipeline({
    label: 'fusedDivClearPress',
    layout: device.createPipelineLayout({ bindGroupLayouts: [fusedDivClearPressBGL] }),
    compute: { module: fusedDivClearPressModule, entryPoint: 'main' },
  });

  // Jacobi: uniform, texture(pressure), texture(div), storage(dst)
  const jacobiPipe = buildPipeline(jacobiShader, 'jacobi',
    ['uniform', 'texture', 'texture', 'storage']);

  // Gradient subtract: uniform, texture(vel), texture(pressure), storage(dst)
  const gradSubPipe = buildPipeline(gradSubShader, 'gradSub',
    ['uniform', 'texture', 'texture', 'storage']);

  // Advect velocity: uniform, texture(src), sampler, storage(dst)
  const advectVelPipe = buildPipeline(advectVelShader, 'advectVel',
    ['uniform', 'texture', 'sampler', 'storage']);

  // Advect dye: uniform, texture(vel), texture(dye), sampler, storage(dst)
  const advectDyePipe = buildPipeline(advectDyeShader, 'advectDye',
    ['uniform', 'texture', 'texture', 'sampler', 'storage']);

  // Curl noise pipeline
  const curlNoisePipe = buildPipeline(curlNoiseShader, 'curlNoise',
    ['uniform', 'texture', 'storage']);

  const noiseBuf = device.createBuffer({
    size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const noiseData = new Float32Array(16); // [time, amount, simRes, frequency, speed, type, symmetry, mapping, warp, sharpness, anisotropy, blend, pad...]

  // Dye noise pipeline: injects dye where flow converges
  const dyeNoisePipe = buildPipeline(dyeNoiseShader, 'dyeNoise',
    ['uniform', 'texture', 'texture', 'storage']);

  const dyeNoiseBuf = device.createBuffer({
    size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const dyeNoiseData = new Float32Array(8); // [time, amount, simRes, curlDyeAmount, r, g, b, symmetryBehavior]

  // ─── Sphere Cleanup pipeline (hard-zero outside sphere every frame) ─────
  const cleanupBGL = device.createBindGroupLayout({
    label: 'cleanup_bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: TEX_FMT } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: TEX_FMT } },
    ],
  });
  const cleanupModule = device.createShaderModule({ code: sphereCleanupShader, label: 'sphereCleanup' });
  const cleanupPipeline = device.createComputePipeline({
    label: 'sphereCleanup',
    layout: device.createPipelineLayout({ bindGroupLayouts: [cleanupBGL] }),
    compute: { module: cleanupModule, entryPoint: 'main' },
  });
  const cleanupBuf = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const cleanupData = new Float32Array([SIM_RES, 0, 0, 0]);
  device.queue.writeBuffer(cleanupBuf, 0, cleanupData);

  // ─── Temperature/Buoyancy pipelines ─────────────────────────────────────
  // Temperature advect: custom BGL (uniform, texture(vel), texture(temp), sampler, storage(tempDst))
  const tempAdvectPipe = buildPipeline(temperatureAdvectShader, 'tempAdvect',
    ['uniform', 'texture', 'texture', 'sampler', 'storage']);

  const tempParamBuf = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const tempParamData = new Float32Array(4); // [dt, dx, simRes, dissipation]

  // Buoyancy: custom BGL (uniform, texture(temp), texture(vel), storage(velDst))
  const buoyancyPipe = buildPipeline(buoyancyShader, 'buoyancy',
    ['uniform', 'texture', 'texture', 'storage']);

  const buoyancyBuf = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const buoyancyData = new Float32Array(4); // [simRes, dt, buoyancy, pad]

  // Temperature splat: reuses batchSplatBGL layout (uniform, texture, storage-tex, storage-buf, uniform)
  const tempSplatModule = device.createShaderModule({ code: tempSplatShader, label: 'tempSplat' });
  checkShader(tempSplatModule, 'tempSplat');
  const tempSplatPipe = device.createComputePipeline({
    label: 'tempSplat',
    layout: batchSplatLayout,
    compute: { module: tempSplatModule, entryPoint: 'main' },
  });

  // ─── Display render pipeline ────────────────────────────────────────────
  const displayBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  });

  const displayPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [displayBGL] }),
    vertex: {
      module: device.createShaderModule({ code: displayShaderVert }),
      entryPoint: 'main',
    },
    fragment: {
      module: device.createShaderModule({ code: makeDisplayShaderFrag(hdrSupported) }),
      entryPoint: 'main',
      targets: [{ format: canvasFmt }],
    },
    primitive: { topology: 'triangle-list' },
  });

  // ─── Bloom Pipelines ──────────────────────────────────────────────────────
  const BLOOM_MIPS = 5;

  const bloomDownBGL = device.createBindGroupLayout({
    label: 'bloomDown_bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: {} },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const bloomDownPipe = device.createComputePipeline({
    label: 'bloomDownsample',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bloomDownBGL] }),
    compute: {
      module: device.createShaderModule({ code: bloomDownsampleShader, label: 'bloomDown' }),
      entryPoint: 'main',
    },
  });

  const bloomUpBGL = device.createBindGroupLayout({
    label: 'bloomUp_bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: {} },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const bloomUpPipe = device.createComputePipeline({
    label: 'bloomUpsample',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bloomUpBGL] }),
    compute: {
      module: device.createShaderModule({ code: bloomUpsampleShader, label: 'bloomUp' }),
      entryPoint: 'main',
    },
  });

  const bloomCompositeBGL = device.createBindGroupLayout({
    label: 'bloomComposite_bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  });

  const bloomCompositeModule = device.createShaderModule({ code: bloomCompositeShader, label: 'bloomComposite' });
  const bloomCompositePipe = device.createRenderPipeline({
    label: 'bloomComposite',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bloomCompositeBGL] }),
    vertex: { module: bloomCompositeModule, entryPoint: 'vert' },
    fragment: {
      module: bloomCompositeModule,
      entryPoint: 'frag',
      targets: [{ format: canvasFmt }],
    },
    primitive: { topology: 'triangle-list' },
  });

  // Bloom uniform buffers (one per mip level pass)
  const bloomParamBufs = [];
  for (let i = 0; i < BLOOM_MIPS + (BLOOM_MIPS - 1); i++) {
    bloomParamBufs.push(device.createBuffer({
      label: `bloomParams_${i}`,
      size: 32, // 8 floats × 4 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }));
  }
  const bloomCompositeUB = device.createBuffer({
    label: 'bloomCompositeUB',
    size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Bloom resource lifecycle
  let bloomResources = null;
  let lastBloomW = 0, lastBloomH = 0;

  function ensureBloomResources(w, h) {
    if (bloomResources && lastBloomW === w && lastBloomH === h) return;
    destroyBloomResources();
    lastBloomW = w;
    lastBloomH = h;

    const sceneTex = device.createTexture({
      label: 'bloomScene',
      size: [w, h],
      format: canvasFmt,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    const bloomDown = [];
    const bloomUp = [];
    let mw = Math.max(1, w >> 1), mh = Math.max(1, h >> 1);
    for (let i = 0; i < BLOOM_MIPS; i++) {
      bloomDown.push(device.createTexture({
        label: `bloomDown_${i}`,
        size: [mw, mh],
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      }));
      if (i < BLOOM_MIPS - 1) {
        bloomUp.push(device.createTexture({
          label: `bloomUp_${i}`,
          size: [mw, mh],
          format: 'rgba16float',
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        }));
      }
      mw = Math.max(1, mw >> 1);
      mh = Math.max(1, mh >> 1);
    }

    const sceneView = sceneTex.createView();

    // Downsample bind groups: sceneTex→bloomDown[0], bloomDown[i]→bloomDown[i+1]
    const downBGs = [];
    for (let i = 0; i < BLOOM_MIPS; i++) {
      const srcView = i === 0 ? sceneView : bloomDown[i - 1].createView();
      downBGs.push(device.createBindGroup({
        layout: bloomDownBGL,
        entries: [
          { binding: 0, resource: srcView },
          { binding: 1, resource: linearSampler },
          { binding: 2, resource: bloomDown[i].createView() },
          { binding: 3, resource: { buffer: bloomParamBufs[i] } },
        ],
      }));
    }

    // Upsample bind groups: lower + current → bloomUp[i]
    const upBGs = [];
    for (let i = 0; i < BLOOM_MIPS - 1; i++) {
      const mipIdx = BLOOM_MIPS - 2 - i; // 3, 2, 1, 0
      const lowerView = i === 0
        ? bloomDown[BLOOM_MIPS - 1].createView()
        : bloomUp[BLOOM_MIPS - 2 - i + 1].createView();
      // Wait — the up array is indexed 0..3 for mip sizes matching bloomDown 0..3
      // Upsample pass i: lower(smaller) + bloomDown[mipIdx] → bloomUp[mipIdx]
      const lowerSrc = i === 0
        ? bloomDown[BLOOM_MIPS - 1].createView()
        : bloomUp[mipIdx + 1].createView();
      upBGs.push(device.createBindGroup({
        layout: bloomUpBGL,
        entries: [
          { binding: 0, resource: lowerSrc },
          { binding: 1, resource: bloomDown[mipIdx].createView() },
          { binding: 2, resource: linearSampler },
          { binding: 3, resource: bloomUp[mipIdx].createView() },
          { binding: 4, resource: { buffer: bloomParamBufs[BLOOM_MIPS + i] } },
        ],
      }));
    }

    // Composite bind group
    const compositeBG = device.createBindGroup({
      layout: bloomCompositeBGL,
      entries: [
        { binding: 0, resource: sceneView },
        { binding: 1, resource: bloomUp[0].createView() },
        { binding: 2, resource: linearSampler },
        { binding: 3, resource: { buffer: bloomCompositeUB } },
      ],
    });

    bloomResources = { sceneTex, sceneView, bloomDown, bloomUp, downBGs, upBGs, compositeBG };
  }

  function destroyBloomResources() {
    if (!bloomResources) return;
    bloomResources.sceneTex.destroy();
    for (const t of bloomResources.bloomDown) t.destroy();
    for (const t of bloomResources.bloomUp) t.destroy();
    bloomResources = null;
    lastBloomW = 0;
    lastBloomH = 0;
  }

  // ─── Bind group helpers ─────────────────────────────────────────────────
  function bg(layout, resources) {
    return device.createBindGroup({
      layout,
      entries: resources.map((r, i) => ({ binding: i, resource: r })),
    });
  }
  function ubuf(b) { return { buffer: b }; }
  const viewCache = new Map();
  function tview(t) {
    if (!viewCache.has(t)) viewCache.set(t, t.createView());
    return viewCache.get(t);
  }

  // ─── Particle GPU Resources ────────────────────────────────────────────
  let particleDispatches = Math.ceil(state.particleCount / PARTICLE_WG);

  let particleBuf = device.createBuffer({
    label: 'particles',
    size: state.particleCount * PARTICLE_STRIDE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  let colorBuf = device.createBuffer({
    label: 'particleColors',
    size: state.particleCount * 16,  // vec4f per particle
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // GPU indirect draw buffers
  let visibleIndexBuf = device.createBuffer({
    label: 'visibleIndices',
    size: state.particleCount * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const drawIndirectBuf = device.createBuffer({
    label: 'drawIndirect',
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
  });
  const atomicCounterBuf = device.createBuffer({
    label: 'atomicCounter',
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const zeroU32 = new Uint32Array([0]);

  // GPU-based particle initialization (avoids 512MB CPU allocation for 16M particles)
  function gpuInitParticles(buf, count) {
    const initCode = makeParticleInitShader(count);
    const initModule = device.createShaderModule({ code: initCode, label: 'particleInit' });
    const initBGL = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }],
    });
    const initPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [initBGL] }),
      compute: { module: initModule, entryPoint: 'main' },
    });
    const initBG = device.createBindGroup({
      layout: initBGL,
      entries: [{ binding: 0, resource: { buffer: buf } }],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(initPipeline);
    pass.setBindGroup(0, initBG);
    pass.dispatchWorkgroups(Math.ceil(count / PARTICLE_WG));
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  gpuInitParticles(particleBuf, state.particleCount);
  device.popErrorScope().then(err => { if (err) console.error('WebGPU internal error:', err.message); });
  device.popErrorScope().then(err => { if (err) console.error('WebGPU validation error:', err.message); });
  console.log(`Init: shaders compiled, buffers allocated (${(performance.now() - t0).toFixed(0)}ms)`);

  // Particle update compute pipeline
  const particleUpdateBGL = device.createBindGroupLayout({
    label: 'particleUpdate_bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: {} },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
    ],
  });

  let particleUpdatePipeline = device.createComputePipeline({
    label: 'particleUpdate',
    layout: device.createPipelineLayout({ bindGroupLayouts: [particleUpdateBGL] }),
    compute: {
      module: device.createShaderModule({ code: makeParticleUpdateShader(state.particleCount, hdrSupported), label: 'particleUpdate' }),
      entryPoint: 'main',
    },
  });

  // Particle compact pipeline (GPU indirect draw)
  const particleCompactBGL = device.createBindGroupLayout({
    label: 'particleCompact_bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const particleCompactLayout = device.createPipelineLayout({ bindGroupLayouts: [particleCompactBGL] });

  const compactModule = device.createShaderModule({ code: makeParticleCompactShader(state.particleCount), label: 'particleCompact' });
  checkShader(compactModule, 'particleCompact');
  let particleCompactPipeline = device.createComputePipeline({
    label: 'particleCompact',
    layout: particleCompactLayout,
    compute: { module: compactModule, entryPoint: 'main' },
  });

  const particleFinalizeBGL = device.createBindGroupLayout({
    label: 'particleFinalize_bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const finalizeModule = device.createShaderModule({ code: particleFinalizeShader, label: 'particleFinalize' });
  checkShader(finalizeModule, 'particleFinalize');
  const particleFinalizePipeline = device.createComputePipeline({
    label: 'particleFinalize',
    layout: device.createPipelineLayout({ bindGroupLayouts: [particleFinalizeBGL] }),
    compute: { module: finalizeModule, entryPoint: 'main' },
  });

  let particleCompactBG = device.createBindGroup({
    layout: particleCompactBGL,
    entries: [
      { binding: 0, resource: { buffer: colorBuf } },
      { binding: 1, resource: { buffer: visibleIndexBuf } },
      { binding: 2, resource: { buffer: atomicCounterBuf } },
    ],
  });
  const particleFinalizeBG = device.createBindGroup({
    layout: particleFinalizeBGL,
    entries: [
      { binding: 0, resource: { buffer: atomicCounterBuf } },
      { binding: 1, resource: { buffer: drawIndirectBuf } },
    ],
  });

  // Particle render pipeline
  const particleRenderBGL = device.createBindGroupLayout({
    label: 'particleRender_bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });

  const particleRenderPipeline = device.createRenderPipeline({
    label: 'particleRender',
    layout: device.createPipelineLayout({ bindGroupLayouts: [particleRenderBGL] }),
    vertex: {
      module: device.createShaderModule({ code: particleRenderVert, label: 'particleRenderVert' }),
      entryPoint: 'main',
    },
    fragment: {
      module: device.createShaderModule({ code: particleRenderFrag, label: 'particleRenderFrag' }),
      entryPoint: 'main',
      targets: [{
        format: canvasFmt,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-strip' },
  });

  let particleRenderBG = device.createBindGroup({
    layout: particleRenderBGL,
    entries: [
      { binding: 0, resource: { buffer: particleBuf } },
      { binding: 1, resource: { buffer: particleUB } },
      { binding: 2, resource: { buffer: colorBuf } },
      { binding: 3, resource: { buffer: visibleIndexBuf } },
    ],
  });

  // ─── Ping-pong flip state (0 = A is current, 1 = B is current) ─────────
  let velFlip = 0;   // 0: velA is current, 1: velB is current
  let dyeFlip = 0;   // 0: dyeA is current, 1: dyeB is current
  let pressFlip = 0; // 0: pressA is current, 1: pressB is current
  let tempFlip = 0;   // 0: tempA is current, 1: tempB is current

  const velTexs = [velA, velB];
  const dyeTexs = [dyeA, dyeB];
  const pressTexs = [pressA, pressB];
  const tempTexs = [tempA, tempB];

  // ─── Temperature texture init (0.5 = ambient everywhere) ──────────────
  {
    const pixels = new Float32Array(SIM_RES * SIM_RES * 4);
    for (let i = 0; i < SIM_RES * SIM_RES; i++) {
      pixels[i * 4] = 0.5;     // R = temperature = 0.5 (ambient)
      pixels[i * 4 + 1] = 0.0;
      pixels[i * 4 + 2] = 0.0;
      pixels[i * 4 + 3] = 1.0;
    }
    const u16 = new Uint16Array(SIM_RES * SIM_RES * 4);
    for (let i = 0; i < pixels.length; i++) {
      const f = pixels[i];
      const view = new DataView(new ArrayBuffer(4));
      view.setFloat32(0, f);
      const bits = view.getUint32(0);
      const sign = (bits >> 16) & 0x8000;
      const exp = ((bits >> 23) & 0xFF) - 127 + 15;
      const mant = (bits >> 13) & 0x3FF;
      if (exp <= 0) u16[i] = sign;
      else if (exp >= 31) u16[i] = sign | 0x7C00;
      else u16[i] = sign | (exp << 10) | mant;
    }
    device.queue.writeTexture(
      { texture: tempA },
      u16.buffer,
      { bytesPerRow: SIM_RES * 8, rowsPerImage: SIM_RES },
      { width: SIM_RES, height: SIM_RES }
    );
    device.queue.writeTexture(
      { texture: tempB },
      u16.buffer,
      { bytesPerRow: SIM_RES * 8, rowsPerImage: SIM_RES },
      { width: SIM_RES, height: SIM_RES }
    );
  }


  // ─── Batch splat GPU resources ──────────────────────────────────────────
  const splatBuf = device.createBuffer({
    label: 'splatData',
    size: MAX_SPLATS * 32, // 8 f32 per splat
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const splatCountBuf = device.createBuffer({
    label: 'splatCount',
    size: 16, // vec4u
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const splatArrayData = new Float32Array(MAX_SPLATS * 8);
  const splatCountUData = new Uint32Array(4);

  // ─── Pre-allocated simulation bind groups (ping-pong pairs) ────────
  // Each pass has 2 variants: [0] reads A writes B, [1] reads B writes A
  // batchSplatVel: reads vel[cur], writes vel[1-cur]
  const batchSplatVelBGs = [
    bg(batchSplatBGL, [ubuf(paramBuf), tview(velA), tview(velB), ubuf(splatBuf), ubuf(splatCountBuf)]),
    bg(batchSplatBGL, [ubuf(paramBuf), tview(velB), tview(velA), ubuf(splatBuf), ubuf(splatCountBuf)]),
  ];
  // batchSplatDye: reads dye[cur], writes dye[1-cur]
  const batchSplatDyeBGs = [
    bg(batchSplatBGL, [ubuf(paramBuf), tview(dyeA), tview(dyeB), ubuf(splatBuf), ubuf(splatCountBuf)]),
    bg(batchSplatBGL, [ubuf(paramBuf), tview(dyeB), tview(dyeA), ubuf(splatBuf), ubuf(splatCountBuf)]),
  ];
  // fusedCurlVort: reads vel[cur], writes curlTex + vel[1-cur]
  const fusedCurlVortBGs = [
    bg(fusedCurlVortBGL, [ubuf(paramBuf), tview(velA), tview(curlTex), tview(velB)]),
    bg(fusedCurlVortBGL, [ubuf(paramBuf), tview(velB), tview(curlTex), tview(velA)]),
  ];
  // fusedDivClearPress: reads vel[cur] + press[cur], writes divTex + press[1-cur]
  const fusedDivClearPressBGs = [
    [bg(fusedDivClearPressBGL, [ubuf(paramBuf), tview(velA), tview(pressA), tview(divTex), tview(pressB)]),
     bg(fusedDivClearPressBGL, [ubuf(paramBuf), tview(velA), tview(pressB), tview(divTex), tview(pressA)])],
    [bg(fusedDivClearPressBGL, [ubuf(paramBuf), tview(velB), tview(pressA), tview(divTex), tview(pressB)]),
     bg(fusedDivClearPressBGL, [ubuf(paramBuf), tview(velB), tview(pressB), tview(divTex), tview(pressA)])],
  ];
  // jacobi: reads press[cur] + divTex, writes press[1-cur]
  const jacobiBGs = [
    bg(jacobiPipe.layout, [ubuf(paramBuf), tview(pressA), tview(divTex), tview(pressB)]),
    bg(jacobiPipe.layout, [ubuf(paramBuf), tview(pressB), tview(divTex), tview(pressA)]),
  ];
  // gradSub: reads vel[cur] + press[cur], writes vel[1-cur] — 4 variants (vel × press)
  const gradSubBGs = [
    [bg(gradSubPipe.layout, [ubuf(paramBuf), tview(velA), tview(pressA), tview(velB)]),
     bg(gradSubPipe.layout, [ubuf(paramBuf), tview(velA), tview(pressB), tview(velB)])],
    [bg(gradSubPipe.layout, [ubuf(paramBuf), tview(velB), tview(pressA), tview(velA)]),
     bg(gradSubPipe.layout, [ubuf(paramBuf), tview(velB), tview(pressB), tview(velA)])],
  ];
  // advectVel: reads vel[cur] (both src + sampler), writes vel[1-cur]
  const advVelBGs = [
    bg(advectVelPipe.layout, [ubuf(paramBuf), tview(velA), linearSampler, tview(velB)]),
    bg(advectVelPipe.layout, [ubuf(paramBuf), tview(velB), linearSampler, tview(velA)]),
  ];
  // advectDye: reads vel[cur] + dye[cur], writes dye[1-cur] — 4 variants (vel × dye)
  const advDyeBGs = [
    [bg(advectDyePipe.layout, [ubuf(paramBuf), tview(velA), tview(dyeA), linearSampler, tview(dyeB)]),
     bg(advectDyePipe.layout, [ubuf(paramBuf), tview(velA), tview(dyeB), linearSampler, tview(dyeA)])],
    [bg(advectDyePipe.layout, [ubuf(paramBuf), tview(velB), tview(dyeA), linearSampler, tview(dyeB)]),
     bg(advectDyePipe.layout, [ubuf(paramBuf), tview(velB), tview(dyeB), linearSampler, tview(dyeA)])],
  ];
  // curlNoise: reads vel[cur], writes vel[1-cur]
  const curlNoiseBGs = [
    bg(curlNoisePipe.layout, [ubuf(noiseBuf), tview(velA), tview(velB)]),
    bg(curlNoisePipe.layout, [ubuf(noiseBuf), tview(velB), tview(velA)]),
  ];
  // dyeNoise: reads vel[cur] + dye[cur], writes dye[1-cur]
  const dyeNoiseBGs = [
    [bg(dyeNoisePipe.layout, [ubuf(dyeNoiseBuf), tview(velA), tview(dyeA), tview(dyeB)]),
     bg(dyeNoisePipe.layout, [ubuf(dyeNoiseBuf), tview(velA), tview(dyeB), tview(dyeA)])],
    [bg(dyeNoisePipe.layout, [ubuf(dyeNoiseBuf), tview(velB), tview(dyeA), tview(dyeB)]),
     bg(dyeNoisePipe.layout, [ubuf(dyeNoiseBuf), tview(velB), tview(dyeB), tview(dyeA)])],
  ];
  // sphereCleanup: reads vel[cur]+dye[cur], writes vel[1-cur]+dye[1-cur] — 4 variants
  const cleanupBGs = [
    [bg(cleanupBGL, [ubuf(cleanupBuf), tview(velA), tview(dyeA), tview(velB), tview(dyeB)]),
     bg(cleanupBGL, [ubuf(cleanupBuf), tview(velA), tview(dyeB), tview(velB), tview(dyeA)])],
    [bg(cleanupBGL, [ubuf(cleanupBuf), tview(velB), tview(dyeA), tview(velA), tview(dyeB)]),
     bg(cleanupBGL, [ubuf(cleanupBuf), tview(velB), tview(dyeB), tview(velA), tview(dyeA)])],
  ];
  // ─── Temperature/Buoyancy bind groups ──────────────────────────────────
  // tempSplat: uses batchSplatBGL (uniform, texture, storage-tex, storage-buf, uniform)
  const tempSplatBGs = [
    bg(batchSplatBGL, [ubuf(paramBuf), tview(tempA), tview(tempB), ubuf(splatBuf), ubuf(splatCountBuf)]),
    bg(batchSplatBGL, [ubuf(paramBuf), tview(tempB), tview(tempA), ubuf(splatBuf), ubuf(splatCountBuf)]),
  ];
  // tempAdvect: reads vel[cur] + temp[cur], writes temp[1-cur] — 4 variants [velFlip][tempFlip]
  const tempAdvectBGs = [
    [bg(tempAdvectPipe.layout, [ubuf(tempParamBuf), tview(velA), tview(tempA), linearSampler, tview(tempB)]),
     bg(tempAdvectPipe.layout, [ubuf(tempParamBuf), tview(velA), tview(tempB), linearSampler, tview(tempA)])],
    [bg(tempAdvectPipe.layout, [ubuf(tempParamBuf), tview(velB), tview(tempA), linearSampler, tview(tempB)]),
     bg(tempAdvectPipe.layout, [ubuf(tempParamBuf), tview(velB), tview(tempB), linearSampler, tview(tempA)])],
  ];
  // buoyancy: reads temp[cur] + vel[cur], writes vel[1-cur] — 4 variants [tempFlip][velFlip]
  const buoyancyBGs = [
    [bg(buoyancyPipe.layout, [ubuf(buoyancyBuf), tview(tempA), tview(velA), tview(velB)]),
     bg(buoyancyPipe.layout, [ubuf(buoyancyBuf), tview(tempA), tview(velB), tview(velA)])],
    [bg(buoyancyPipe.layout, [ubuf(buoyancyBuf), tview(tempB), tview(velA), tview(velB)]),
     bg(buoyancyPipe.layout, [ubuf(buoyancyBuf), tview(tempB), tview(velB), tview(velA)])],
  ];
  // display: reads dye[cur]
  const displayBGs = [
    bg(displayBGL, [tview(dyeA), linearSampler, ubuf(displayUB)]),
    bg(displayBGL, [tview(dyeB), linearSampler, ubuf(displayUB)]),
  ];
  // particleUpdate: reads vel[cur] + dye[cur] + curlTex — 4 variants [velFlip][dyeFlip]
  function makeParticleUpdateBGs(pBuf, cBuf) {
    return [
      [bg(particleUpdateBGL, [ubuf(paramBuf), tview(velA), linearSampler, {buffer: pBuf}, tview(dyeA), ubuf(particleUB), {buffer: cBuf}, tview(curlTex)]),
       bg(particleUpdateBGL, [ubuf(paramBuf), tview(velA), linearSampler, {buffer: pBuf}, tview(dyeB), ubuf(particleUB), {buffer: cBuf}, tview(curlTex)])],
      [bg(particleUpdateBGL, [ubuf(paramBuf), tview(velB), linearSampler, {buffer: pBuf}, tview(dyeA), ubuf(particleUB), {buffer: cBuf}, tview(curlTex)]),
       bg(particleUpdateBGL, [ubuf(paramBuf), tview(velB), linearSampler, {buffer: pBuf}, tview(dyeB), ubuf(particleUB), {buffer: cBuf}, tview(curlTex)])],
    ];
  }
  let particleUpdateBGs = makeParticleUpdateBGs(particleBuf, colorBuf);

  let splatCount = 0;
  let frameTimeScale = 1.0;
  function addSplat(x, y, dx, dy, r, g, b, radius) {
    // Containment at source: clamp splat center and remove outward force near wall.
    const WALL_R = SIM_SPHERE_RADIUS;
    const WALL_BAND = 0.04;
    let ox = x - 0.5;
    let oy = y - 0.5;
    let d = Math.hypot(ox, oy);
    if (d > WALL_R) {
      const inv = WALL_R / Math.max(d, 1e-6);
      ox *= inv;
      oy *= inv;
      x = 0.5 + ox;
      y = 0.5 + oy;
      d = WALL_R;
    }
    if (d > WALL_R - WALL_BAND) {
      const nx = ox / Math.max(d, 1e-6);
      const ny = oy / Math.max(d, 1e-6);
      const outward = dx * nx + dy * ny;
      if (outward > 0) {
        dx -= nx * outward;
        dy -= ny * outward;
      }
      const wallT = Math.min(1, Math.max(0, (d - (WALL_R - WALL_BAND)) / WALL_BAND));
      const damp = 1 - wallT * 0.35;
      dx *= damp;
      dy *= damp;
    }

    // Master-speed scaling for impulse-style effectors so slowdown is uniform.
    dx *= frameTimeScale;
    dy *= frameTimeScale;
    r *= frameTimeScale;
    g *= frameTimeScale;
    b *= frameTimeScale;

    if (splatCount >= MAX_SPLATS) return;
    const off = splatCount * 8;
    splatArrayData[off]     = x;
    splatArrayData[off + 1] = y;
    splatArrayData[off + 2] = dx;
    splatArrayData[off + 3] = dy;
    splatArrayData[off + 4] = r;
    splatArrayData[off + 5] = g;
    splatArrayData[off + 6] = b;
    splatArrayData[off + 7] = radius;
    splatCount++;
  }

  // ─── Face Tracking + Face Effector ------------------------------------------------
  const faceTrackingStatusEl = document.getElementById('faceTrackingStatus');
  const faceTrackingToggleBtn = document.getElementById('faceTrackingToggle');
  const faceEffectorModeSelect = document.getElementById('faceEffectorMode');
  const faceDebugModeSelect = document.getElementById('faceDebugMode');
  const faceDebugCanvas = document.getElementById('faceDebugCanvas');
  const faceDebugCtx = faceDebugCanvas?.getContext('2d', { alpha: true }) || null;

  const faceTracking = {
    enabled: false,
    ready: false,
    initializing: false,
    startToken: 0,
    engine: 'worker',
    worker: null,
    stream: null,
    video: null,
    mainLandmarker: null,
    frameInFlight: false,
    frameInFlightSince: 0,
    frameEveryMs: 1000 / 45,
    nextFrameAt: 0,
    frameTimeoutMs: 1200,
    face: null,
    rawLandmarks: null,
    rawLandmarkCount: 0,
    blendshapeScores: null,
    blendshapeCount: 0,
    transformMatrix: null,
    matrixMotion: 0,
    poseRoll: 0,
    poseScale: 1,
    prevMapped: null,
    prevCenterX: NaN,
    prevCenterY: NaN,
    centerVelX: 0,
    centerVelY: 0,
    smoothedMouth: 0,
    smoothedEyeLeft: 1,
    smoothedEyeRight: 1,
    prevMouth: 0,
    lastFaceSeenAt: -1,
    noFaceGraceMs: 1200,
    lastMouthBurstTime: -999,
    droppedFrames: 0,
    errorStreak: 0,
    inferenceMs: 0,
    initWatchdog: 0,
    mainFallbackAttempted: false,
  };

  function setFaceStatus(text, error = false) {
    if (!faceTrackingStatusEl) return;
    faceTrackingStatusEl.textContent = text;
    faceTrackingStatusEl.style.color = error ? '#cc6666' : '#666';
  }

  function syncFaceTrackingToggleButton() {
    if (!faceTrackingToggleBtn) return;
    if (faceTracking.initializing) {
      faceTrackingToggleBtn.textContent = 'Starting webcam...';
      faceTrackingToggleBtn.classList.add('active');
      faceTrackingToggleBtn.disabled = true;
      return;
    }
    faceTrackingToggleBtn.disabled = false;
    if (faceTracking.enabled) {
      faceTrackingToggleBtn.textContent = 'Stop Webcam Face Tracking';
      faceTrackingToggleBtn.classList.add('active');
    } else {
      faceTrackingToggleBtn.textContent = 'Start Webcam Face Tracking';
      faceTrackingToggleBtn.classList.remove('active');
    }
  }

  function normalizeBlendshapePayload(raw) {
    if (!raw) return null;
    const out = Object.create(null);
    let used = 0;
    for (const key of FACE_BLENDSHAPE_KEYS) {
      const v = Number(raw[key]);
      if (!Number.isFinite(v)) continue;
      out[key] = Math.max(0, Math.min(1, v));
      used++;
    }
    return used > 0 ? out : null;
  }

  function parseBlendshapeFromResult(res) {
    const categories = res?.faceBlendshapes?.[0]?.categories;
    if (!categories?.length) return { scores: null, count: 0 };
    const bag = Object.create(null);
    for (const cat of categories) {
      if (!cat?.categoryName) continue;
      bag[cat.categoryName] = cat.score;
    }
    return { scores: normalizeBlendshapePayload(bag), count: categories.length };
  }

  function parseMatrixPayload(raw) {
    if (!raw) return null;
    const src = Array.isArray(raw)
      ? raw
      : (raw.data || raw.matrix || raw.values || raw);
    if (!src || typeof src.length !== 'number' || src.length < 16) return null;
    const mat = new Float32Array(16);
    for (let i = 0; i < 16; i++) {
      const v = Number(src[i]);
      mat[i] = Number.isFinite(v) ? v : 0;
    }
    return mat;
  }

  function parseMatrixFromResult(res) {
    return parseMatrixPayload(res?.facialTransformationMatrixes?.[0] || null);
  }

  function blendScore(blend, key, fallback = 0) {
    if (!blend) return fallback;
    const v = blend[key];
    return Number.isFinite(v) ? v : fallback;
  }

  function syncFaceDebugCanvasSize() {
    if (!faceDebugCanvas || !faceDebugCtx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(window.innerWidth * dpr));
    const h = Math.max(1, Math.floor(window.innerHeight * dpr));
    if (faceDebugCanvas.width !== w || faceDebugCanvas.height !== h) {
      faceDebugCanvas.width = w;
      faceDebugCanvas.height = h;
      faceDebugCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }

  function simUVToScreen(uvx, uvy) {
    const rect = canvas.getBoundingClientRect();
    const aspect = rect.width / Math.max(rect.height, 1e-6);
    const rawX = (uvx - 0.5) / Math.max(aspect, 1.0) + 0.5;
    const rawY = (uvy - 0.5) / Math.max(1.0 / Math.max(aspect, 1e-6), 1.0) + 0.5;
    return [
      rect.left + rawX * rect.width,
      rect.top + (1.0 - rawY) * rect.height,
    ];
  }

  function buildLocalMeshEdges(face, sampleIdx) {
    const maxDist = face.radius * 0.35;
    const edgeSet = new Set();
    for (let i = 0; i < sampleIdx.length; i++) {
      const ai = sampleIdx[i];
      if (ai >= face.count) continue;
      const ax = face.mapped[ai * 2];
      const ay = face.mapped[ai * 2 + 1];
      let best1 = -1, best2 = -1;
      let d1 = 1e9, d2 = 1e9;
      for (let j = 0; j < sampleIdx.length; j++) {
        if (i === j) continue;
        const bi = sampleIdx[j];
        if (bi >= face.count) continue;
        const bx = face.mapped[bi * 2];
        const by = face.mapped[bi * 2 + 1];
        const d = Math.hypot(bx - ax, by - ay);
        if (d > maxDist) continue;
        if (d < d1) {
          d2 = d1; best2 = best1;
          d1 = d; best1 = bi;
        } else if (d < d2) {
          d2 = d; best2 = bi;
        }
      }
      if (best1 >= 0) {
        const a = Math.min(ai, best1);
        const b = Math.max(ai, best1);
        edgeSet.add(`${a}:${b}`);
      }
      if (best2 >= 0) {
        const a = Math.min(ai, best2);
        const b = Math.max(ai, best2);
        edgeSet.add(`${a}:${b}`);
      }
    }
    return edgeSet;
  }

  function drawFaceDebugOverlay(nowMs) {
    const debugMode = Math.round(state.faceDebugMode || 0);
    if (!faceDebugCanvas || !faceDebugCtx) return;
    if (debugMode <= 0) {
      faceDebugCanvas.style.display = 'none';
      return;
    }

    syncFaceDebugCanvasSize();
    faceDebugCanvas.style.display = 'block';

    const ctx2d = faceDebugCtx;
    const vw = faceDebugCanvas.width / (window.devicePixelRatio || 1);
    const vh = faceDebugCanvas.height / (window.devicePixelRatio || 1);
    ctx2d.clearRect(0, 0, vw, vh);

    // Show simulation boundary for alignment checks.
    const [cx, cy] = simUVToScreen(0.5, 0.5);
    const [ex, ey] = simUVToScreen(0.5 + SIM_SPHERE_RADIUS, 0.5);
    const boundaryR = Math.hypot(ex - cx, ey - cy);
    ctx2d.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx2d.lineWidth = 1;
    ctx2d.beginPath();
    ctx2d.arc(cx, cy, boundaryR, 0, Math.PI * 2);
    ctx2d.stroke();

    // Optional webcam preview pane for confidence checking.
    if (debugMode >= 2 && faceTracking.video) {
      const boxW = Math.min(260, vw * 0.28);
      const boxH = boxW * 0.75;
      const boxX = 18;
      const boxY = 18;

      ctx2d.fillStyle = 'rgba(0,0,0,0.5)';
      ctx2d.fillRect(boxX - 3, boxY - 3, boxW + 6, boxH + 6);

      ctx2d.save();
      ctx2d.translate(boxX + boxW, boxY);
      ctx2d.scale(-1, 1);
      ctx2d.drawImage(faceTracking.video, 0, 0, boxW, boxH);
      ctx2d.restore();

      if (faceTracking.rawLandmarks && faceTracking.rawLandmarkCount > 0) {
        ctx2d.fillStyle = 'rgba(32,255,160,0.72)';
        const lm = faceTracking.rawLandmarks;
        const count = faceTracking.rawLandmarkCount;
        for (let i = 0; i < count; i += 2) {
          const off = i * 3;
          const x = boxX + (1.0 - lm[off]) * boxW;
          const y = boxY + lm[off + 1] * boxH;
          ctx2d.fillRect(x - 1, y - 1, 2, 2);
        }
      }

      ctx2d.fillStyle = 'rgba(255,255,255,0.8)';
      ctx2d.font = '11px system-ui, sans-serif';
      ctx2d.fillText(`Face Debug (${faceTracking.inferenceMs.toFixed(1)}ms)`, boxX + 6, boxY + boxH + 16);
    }

    const face = faceTracking.face;
    const blend = faceTracking.blendshapeScores;
    const streamLabel = `L${faceTracking.rawLandmarkCount || 0}  B${faceTracking.blendshapeCount || 0}  M${faceTracking.transformMatrix ? 16 : 0}`;
    ctx2d.fillStyle = 'rgba(255,255,255,0.82)';
    ctx2d.font = '11px system-ui, sans-serif';
    ctx2d.fillText(streamLabel, 18, vh - 42);

    if (!face) {
      const age = faceTracking.lastFaceSeenAt > 0 ? nowMs - faceTracking.lastFaceSeenAt : Infinity;
      ctx2d.fillStyle = age < faceTracking.noFaceGraceMs ? 'rgba(255,220,120,0.9)' : 'rgba(255,80,80,0.9)';
      ctx2d.font = '12px system-ui, sans-serif';
      ctx2d.fillText(age < faceTracking.noFaceGraceMs ? 'Tracking hold...' : 'Searching for face...', 18, vh - 22);
      return;
    }

    const traceLoop = (indices) => {
      let started = false;
      let firstX = 0;
      let firstY = 0;
      ctx2d.beginPath();
      for (const idx of indices) {
        if (idx >= face.count) continue;
        const [sx, sy] = simUVToScreen(face.mapped[idx * 2], face.mapped[idx * 2 + 1]);
        if (!started) {
          ctx2d.moveTo(sx, sy);
          firstX = sx;
          firstY = sy;
          started = true;
        } else {
          ctx2d.lineTo(sx, sy);
        }
      }
      if (started) ctx2d.lineTo(firstX, firstY);
      return started;
    };

    // Filled face silhouette with carved eye/mouth holes (debug only).
    ctx2d.save();
    ctx2d.fillStyle = 'rgba(70,180,255,0.12)';
    if (traceLoop(FACE_IDX.contour)) ctx2d.fill();
    ctx2d.globalCompositeOperation = 'destination-out';
    ctx2d.fillStyle = 'rgba(0,0,0,1)';
    if (traceLoop(FACE_IDX.leftEye)) ctx2d.fill();
    if (traceLoop(FACE_IDX.rightEye)) ctx2d.fill();
    if (traceLoop(FACE_IDX.mouthHole)) ctx2d.fill();
    ctx2d.restore();

    const sample = [];
    for (let i = 0; i < FACE_DENSE_INDICES.length; i += 2) sample.push(FACE_DENSE_INDICES[i]);
    const localEdges = buildLocalMeshEdges(face, sample);

    // Dynamic local mesh network.
    ctx2d.strokeStyle = 'rgba(90, 220, 255, 0.42)';
    ctx2d.lineWidth = 1;
    ctx2d.beginPath();
    for (const key of localEdges) {
      const sep = key.indexOf(':');
      const a = parseInt(key.slice(0, sep), 10);
      const b = parseInt(key.slice(sep + 1), 10);
      if (a >= face.count || b >= face.count) continue;
      const [ax, ay] = simUVToScreen(face.mapped[a * 2], face.mapped[a * 2 + 1]);
      const [bx, by] = simUVToScreen(face.mapped[b * 2], face.mapped[b * 2 + 1]);
      ctx2d.moveTo(ax, ay);
      ctx2d.lineTo(bx, by);
    }
    ctx2d.stroke();

    // Structural loops for a clear, readable mesh.
    ctx2d.strokeStyle = 'rgba(255, 190, 70, 0.82)';
    ctx2d.lineWidth = 1.3;
    for (const path of FACE_DEBUG_PATHS) {
      let started = false;
      let firstX = 0, firstY = 0;
      ctx2d.beginPath();
      for (const idx of path.points) {
        if (idx >= face.count) continue;
        const [sx, sy] = simUVToScreen(face.mapped[idx * 2], face.mapped[idx * 2 + 1]);
        if (!started) {
          ctx2d.moveTo(sx, sy);
          firstX = sx;
          firstY = sy;
          started = true;
        } else {
          ctx2d.lineTo(sx, sy);
        }
      }
      if (started && path.closed) ctx2d.lineTo(firstX, firstY);
      ctx2d.stroke();
    }

    // Key points.
    ctx2d.fillStyle = 'rgba(255,255,255,0.84)';
    for (let i = 0; i < FACE_DENSE_INDICES.length; i += 1) {
      const idx = FACE_DENSE_INDICES[i];
      if (idx >= face.count) continue;
      const [sx, sy] = simUVToScreen(face.mapped[idx * 2], face.mapped[idx * 2 + 1]);
      ctx2d.fillRect(sx - 1, sy - 1, 2, 2);
    }

    // Pose axes from the facial transformation matrix.
    const [cx2, cy2] = simUVToScreen(face.centerX, face.centerY);
    const axisLen = Math.max(14, Math.min(56, boundaryR * (face.radius / Math.max(SIM_SPHERE_RADIUS, 1e-6)) * 0.55));
    const roll = Number.isFinite(face.roll) ? face.roll : (faceTracking.poseRoll || 0);
    const axX = Math.cos(roll);
    const axY = Math.sin(roll);
    const ayX = -axY;
    const ayY = axX;
    ctx2d.lineWidth = 2;
    ctx2d.strokeStyle = 'rgba(255,100,100,0.95)';
    ctx2d.beginPath();
    ctx2d.moveTo(cx2, cy2);
    ctx2d.lineTo(cx2 + axX * axisLen, cy2 + axY * axisLen);
    ctx2d.stroke();
    ctx2d.strokeStyle = 'rgba(100,255,140,0.95)';
    ctx2d.beginPath();
    ctx2d.moveTo(cx2, cy2);
    ctx2d.lineTo(cx2 + ayX * axisLen, cy2 + ayY * axisLen);
    ctx2d.stroke();

    // Mouth center highlight.
    const [mx, my] = simUVToScreen(face.mouthCenterX, face.mouthCenterY);
    ctx2d.strokeStyle = `rgba(255, 80, 80, ${0.45 + face.mouthOpen * 0.5})`;
    ctx2d.lineWidth = 2;
    ctx2d.beginPath();
    ctx2d.arc(mx, my, 8 + face.mouthOpen * 10, 0, Math.PI * 2);
    ctx2d.stroke();

    const jaw = blendScore(blend, 'jawOpen', face.mouthOpen);
    const blinkL = blendScore(blend, 'eyeBlinkLeft', 1 - (face.eyeLeftOpen ?? 0.5));
    const blinkR = blendScore(blend, 'eyeBlinkRight', 1 - (face.eyeRightOpen ?? 0.5));
    ctx2d.fillStyle = 'rgba(255,255,255,0.9)';
    ctx2d.font = '11px system-ui, sans-serif';
    ctx2d.fillText(`jaw ${jaw.toFixed(2)}  blinkL ${blinkL.toFixed(2)}  blinkR ${blinkR.toFixed(2)}  motion ${faceTracking.matrixMotion.toFixed(3)}`, 18, vh - 8);
  }

  if (faceDebugCanvas) {
    syncFaceDebugCanvasSize();
    window.addEventListener('resize', syncFaceDebugCanvasSize);
  }

  function clampFacePointToCircle(x, y, margin = 0.005) {
    const dx = x - 0.5;
    const dy = y - 0.5;
    const r = Math.hypot(dx, dy);
    const lim = Math.max(0.02, SIM_SPHERE_RADIUS - margin);
    if (r <= lim) return [x, y];
    const s = lim / Math.max(r, 1e-6);
    return [0.5 + dx * s, 0.5 + dy * s];
  }

  function getRawLandmark(landmarks, count, idx) {
    if (idx < 0 || idx >= count) return [0.5, 0.5, 0];
    const off = idx * 3;
    return [landmarks[off], landmarks[off + 1], landmarks[off + 2]];
  }

  function mapFaceLandmarksToSim(rawLandmarks, blendScores = null) {
    const count = Math.floor(rawLandmarks.length / 3);
    if (count < 200) return null;

    const nose = getRawLandmark(rawLandmarks, count, 1);
    const mouthTop = getRawLandmark(rawLandmarks, count, 13);
    const mouthBot = getRawLandmark(rawLandmarks, count, 14);
    const cheekL = getRawLandmark(rawLandmarks, count, 234);
    const cheekR = getRawLandmark(rawLandmarks, count, 454);
    const brow = getRawLandmark(rawLandmarks, count, 10);
    const chin = getRawLandmark(rawLandmarks, count, 152);
    const eyeL = getRawLandmark(rawLandmarks, count, 33);
    const eyeR = getRawLandmark(rawLandmarks, count, 263);

    const centerCamX = (nose[0] + mouthTop[0] + mouthBot[0]) / 3;
    const centerCamY = (nose[1] + mouthTop[1] + mouthBot[1]) / 3;
    const faceW = Math.hypot(cheekR[0] - cheekL[0], cheekR[1] - cheekL[1]);
    const faceH = Math.hypot(chin[0] - brow[0], chin[1] - brow[1]);
    const spread = Math.max(faceW, faceH, 0.08);

    const targetRadius = Math.max(0.1, Math.min(SIM_SPHERE_RADIUS - 0.045, 0.085 + spread * 0.86));
    let offsetX = (centerCamX - 0.5) * 0.52;
    let offsetY = -(centerCamY - 0.5) * 0.52;
    const offsetR = Math.hypot(offsetX, offsetY);
    const offsetLimit = Math.max(0, SIM_SPHERE_RADIUS - targetRadius - 0.01);
    if (offsetR > offsetLimit) {
      const s = offsetLimit / Math.max(offsetR, 1e-6);
      offsetX *= s;
      offsetY *= s;
    }
    const centerX = 0.5 + offsetX;
    const centerY = 0.5 + offsetY;

    let maxLocalR = 0;
    for (let i = 0; i < count; i++) {
      const off = i * 3;
      const lx = rawLandmarks[off] - centerCamX;
      const ly = rawLandmarks[off + 1] - centerCamY;
      maxLocalR = Math.max(maxLocalR, Math.hypot(lx, ly));
    }
    const scale = targetRadius / Math.max(maxLocalR, 1e-6);

    const mapped = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) {
      const off = i * 3;
      const lx = rawLandmarks[off] - centerCamX;
      const ly = rawLandmarks[off + 1] - centerCamY;
      const simX = centerX + lx * scale;
      const simY = centerY - ly * scale;
      const [cx, cy] = clampFacePointToCircle(simX, simY, 0.006);
      mapped[i * 2] = cx;
      mapped[i * 2 + 1] = cy;
    }

    const mouthW = Math.hypot(
      getRawLandmark(rawLandmarks, count, 291)[0] - getRawLandmark(rawLandmarks, count, 61)[0],
      getRawLandmark(rawLandmarks, count, 291)[1] - getRawLandmark(rawLandmarks, count, 61)[1]
    );
    const mouthGap = Math.hypot(mouthBot[0] - mouthTop[0], mouthBot[1] - mouthTop[1]);
    const mouthRatio = mouthGap / Math.max(mouthW, 1e-6);
    const mouthLandmark = Math.max(0, Math.min(1, (mouthRatio - 0.04) / 0.22));
    const jawBlend = blendScore(blendScores, 'jawOpen', mouthLandmark);
    const mouthOpen = Math.max(mouthLandmark, jawBlend * 0.96);

    const eyeLeftH = Math.hypot(
      getRawLandmark(rawLandmarks, count, 159)[0] - getRawLandmark(rawLandmarks, count, 145)[0],
      getRawLandmark(rawLandmarks, count, 159)[1] - getRawLandmark(rawLandmarks, count, 145)[1]
    );
    const eyeLeftW = Math.hypot(
      getRawLandmark(rawLandmarks, count, 33)[0] - getRawLandmark(rawLandmarks, count, 133)[0],
      getRawLandmark(rawLandmarks, count, 33)[1] - getRawLandmark(rawLandmarks, count, 133)[1]
    );
    const eyeRightH = Math.hypot(
      getRawLandmark(rawLandmarks, count, 386)[0] - getRawLandmark(rawLandmarks, count, 374)[0],
      getRawLandmark(rawLandmarks, count, 386)[1] - getRawLandmark(rawLandmarks, count, 374)[1]
    );
    const eyeRightW = Math.hypot(
      getRawLandmark(rawLandmarks, count, 263)[0] - getRawLandmark(rawLandmarks, count, 362)[0],
      getRawLandmark(rawLandmarks, count, 263)[1] - getRawLandmark(rawLandmarks, count, 362)[1]
    );
    const eyeOpenLandmarkL = Math.max(0, Math.min(1, (eyeLeftH / Math.max(eyeLeftW, 1e-6) - 0.1) / 0.26));
    const eyeOpenLandmarkR = Math.max(0, Math.min(1, (eyeRightH / Math.max(eyeRightW, 1e-6) - 0.1) / 0.26));
    const blinkL = blendScore(blendScores, 'eyeBlinkLeft', 1 - eyeOpenLandmarkL);
    const blinkR = blendScore(blendScores, 'eyeBlinkRight', 1 - eyeOpenLandmarkR);
    const eyeLeftOpen = Math.max(0, Math.min(1, eyeOpenLandmarkL * (1 - blinkL * 0.9)));
    const eyeRightOpen = Math.max(0, Math.min(1, eyeOpenLandmarkR * (1 - blinkR * 0.9)));

    const mouthTopMappedX = mapped[13 * 2];
    const mouthTopMappedY = mapped[13 * 2 + 1];
    const mouthBotMappedX = mapped[14 * 2];
    const mouthBotMappedY = mapped[14 * 2 + 1];
    const eyeMidX = (eyeL[0] + eyeR[0]) * 0.5;
    const eyeMidY = (eyeL[1] + eyeR[1]) * 0.5;
    const roll = Math.atan2(eyeR[1] - eyeL[1], eyeR[0] - eyeL[0]);
    const yaw = Math.max(-1, Math.min(1, (nose[0] - eyeMidX) * 7.5));
    const pitch = Math.max(-1, Math.min(1, (nose[1] - eyeMidY) * 7.0));

    return {
      count,
      mapped,
      centerX,
      centerY,
      radius: targetRadius,
      mouthOpen,
      eyeLeftOpen,
      eyeRightOpen,
      roll,
      yaw,
      pitch,
      mouthCenterX: (mouthTopMappedX + mouthBotMappedX) * 0.5,
      mouthCenterY: (mouthTopMappedY + mouthBotMappedY) * 0.5,
    };
  }

  function facePoint(face, idx) {
    if (!face || idx < 0 || idx >= face.count) return null;
    const off = idx * 2;
    return [face.mapped[off], face.mapped[off + 1]];
  }

  function direction(ax, ay, bx, by) {
    const dx = bx - ax;
    const dy = by - ay;
    const l = Math.hypot(dx, dy);
    if (l < 1e-6) return [0, 0];
    return [dx / l, dy / l];
  }

  function applyFaceLandmarkResult(landmarks, inferenceMs = 0, error = '', extras = null) {
    faceTracking.inferenceMs = inferenceMs || 0;
    const engineLabel = faceTracking.engine === 'main' ? 'main-cpu' : 'worker';
    const blendScores = normalizeBlendshapePayload(extras?.blendshapes || null);
    const blendCount = Number.isFinite(extras?.blendshapeCount)
      ? Math.max(0, Math.floor(extras.blendshapeCount))
      : (blendScores ? FACE_BLENDSHAPE_KEYS.length : 0);
    const matrix = parseMatrixPayload(extras?.matrix || null);

    if (blendScores) {
      faceTracking.blendshapeScores = blendScores;
      faceTracking.blendshapeCount = blendCount;
    } else if (!faceTracking.face) {
      faceTracking.blendshapeScores = null;
      faceTracking.blendshapeCount = 0;
    }

    if (matrix) {
      if (faceTracking.transformMatrix && faceTracking.transformMatrix.length === 16) {
        let d = 0;
        for (let i = 0; i < 16; i++) {
          d += Math.abs(matrix[i] - faceTracking.transformMatrix[i]);
        }
        faceTracking.matrixMotion = faceTracking.matrixMotion * 0.78 + d * 0.22;
      } else {
        faceTracking.matrixMotion = 0;
      }
      faceTracking.transformMatrix = matrix;
      const ax = Math.hypot(matrix[0], matrix[1], matrix[2]);
      const ay = Math.hypot(matrix[4], matrix[5], matrix[6]);
      faceTracking.poseScale = Math.max(0.001, (ax + ay) * 0.5);
      faceTracking.poseRoll = Math.atan2(matrix[1], matrix[0]) || faceTracking.poseRoll || 0;
    } else {
      faceTracking.matrixMotion *= 0.92;
      if (!faceTracking.face) {
        faceTracking.transformMatrix = null;
        faceTracking.poseScale = 1;
      }
    }

    if (error) {
      faceTracking.errorStreak++;
      if (faceTracking.errorStreak % 3 === 0) {
        console.warn('Face tracker inference warning:', error);
      }
      if (faceTracking.engine === 'worker' && faceTracking.errorStreak >= 9 && !faceTracking.mainFallbackAttempted) {
        switchToMainThreadTracker('worker-runtime-errors');
      }
    } else {
      faceTracking.errorStreak = 0;
    }

    if (!landmarks || landmarks.length < 3) {
      faceTracking.rawLandmarks = null;
      faceTracking.rawLandmarkCount = 0;
      const now = performance.now();
      if (faceTracking.lastFaceSeenAt < 0 || (now - faceTracking.lastFaceSeenAt) > faceTracking.noFaceGraceMs) {
        faceTracking.face = null;
        if (faceTracking.prevMapped) faceTracking.prevMapped.fill(NaN);
        faceTracking.prevCenterX = NaN;
        faceTracking.prevCenterY = NaN;
        faceTracking.centerVelX = 0;
        faceTracking.centerVelY = 0;
      }
      if (faceTracking.ready) {
        const ageMs = faceTracking.lastFaceSeenAt < 0 ? 9999 : performance.now() - faceTracking.lastFaceSeenAt;
        const jaw = blendScore(faceTracking.blendshapeScores, 'jawOpen', faceTracking.smoothedMouth || 0);
        const streamTag = `L0/B${faceTracking.blendshapeCount || 0}/M${faceTracking.transformMatrix ? 16 : 0}`;
        if (ageMs <= faceTracking.noFaceGraceMs) {
          setFaceStatus(`Face tracking active (${engineLabel}, ${faceTracking.inferenceMs.toFixed(1)} ms, holding lock, jaw ${(jaw * 100).toFixed(0)}%, ${streamTag})`);
        } else {
          setFaceStatus(`Face tracking active (${engineLabel}, ${faceTracking.inferenceMs.toFixed(1)} ms, searching, ${streamTag})`);
        }
      }
      return;
    }

    faceTracking.rawLandmarks = landmarks;
    faceTracking.rawLandmarkCount = landmarks.length / 3;
    const mappedFace = mapFaceLandmarksToSim(landmarks, faceTracking.blendshapeScores);
    if (!mappedFace) {
      faceTracking.face = null;
      if (faceTracking.prevMapped) faceTracking.prevMapped.fill(NaN);
      return;
    }
    faceTracking.lastFaceSeenAt = performance.now();
    faceTracking.smoothedMouth = faceTracking.smoothedMouth * 0.72 + mappedFace.mouthOpen * 0.28;
    faceTracking.smoothedEyeLeft = faceTracking.smoothedEyeLeft * 0.74 + mappedFace.eyeLeftOpen * 0.26;
    faceTracking.smoothedEyeRight = faceTracking.smoothedEyeRight * 0.74 + mappedFace.eyeRightOpen * 0.26;
    mappedFace.mouthOpen = faceTracking.smoothedMouth;
    mappedFace.eyeLeftOpen = faceTracking.smoothedEyeLeft;
    mappedFace.eyeRightOpen = faceTracking.smoothedEyeRight;
    if (faceTracking.transformMatrix && Number.isFinite(faceTracking.poseRoll)) {
      mappedFace.roll = faceTracking.poseRoll;
    }

    if (Number.isFinite(faceTracking.prevCenterX) && Number.isFinite(faceTracking.prevCenterY)) {
      const dtSec = Math.max(1e-3, faceTracking.frameEveryMs / 1000);
      faceTracking.centerVelX = (mappedFace.centerX - faceTracking.prevCenterX) / dtSec;
      faceTracking.centerVelY = (mappedFace.centerY - faceTracking.prevCenterY) / dtSec;
    } else {
      faceTracking.centerVelX = 0;
      faceTracking.centerVelY = 0;
    }
    faceTracking.prevCenterX = mappedFace.centerX;
    faceTracking.prevCenterY = mappedFace.centerY;
    faceTracking.face = mappedFace;
    const jaw = blendScore(faceTracking.blendshapeScores, 'jawOpen', mappedFace.mouthOpen);
    const streamTag = `L${faceTracking.rawLandmarkCount || 0}/B${faceTracking.blendshapeCount || 0}/M${faceTracking.transformMatrix ? 16 : 0}`;
    setFaceStatus(`Face tracking active (${engineLabel}, ${faceTracking.inferenceMs.toFixed(1)} ms, jaw ${(jaw * 100).toFixed(0)}%, ${streamTag})`);
  }

  async function initMainThreadLandmarker() {
    if (faceTracking.mainLandmarker) return faceTracking.mainLandmarker;
    if (!faceVisionModuleCache) {
      faceVisionModuleCache = await import(FACE_TRACKER_BUNDLE_URL);
    }
    const vision = faceVisionModuleCache;
    const fileset = await vision.FilesetResolver.forVisionTasks(FACE_TRACKER_WASM_URL);
    const lm = await vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: FACE_TRACKER_MODEL_URL,
        delegate: 'CPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.25,
      minFacePresenceConfidence: 0.25,
      minTrackingConfidence: 0.2,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    faceTracking.mainLandmarker = lm;
    return lm;
  }

  async function switchToMainThreadTracker(reason = 'fallback') {
    if (!faceTracking.enabled) return;
    if (faceTracking.engine === 'main') return;
    if (faceTracking.mainFallbackAttempted) return;
    faceTracking.mainFallbackAttempted = true;
    faceTracking.engine = 'main';
    faceTracking.ready = false;
    faceTracking.frameInFlight = false;
    faceTracking.frameInFlightSince = 0;
    if (faceTracking.worker) {
      try { faceTracking.worker.postMessage({ type: 'dispose' }); } catch {}
      try { faceTracking.worker.terminate(); } catch {}
      faceTracking.worker = null;
    }
    console.warn(`Face tracker: switching to main-thread compatibility mode (${reason}).`);
    setFaceStatus('Switching face tracker to compatibility mode...');
    try {
      await initMainThreadLandmarker();
      if (!faceTracking.enabled) return;
      faceTracking.ready = true;
      faceTracking.nextFrameAt = 0;
      setFaceStatus('Face tracking active (main-cpu)');
    } catch (err) {
      console.error('Face tracker main-thread fallback failed:', err?.message || err);
      setFaceStatus(`Face tracker failed: ${err?.message || err}`, true);
      stopFaceTracking(true);
    }
  }

  async function startFaceTracking() {
    if (faceTracking.initializing || faceTracking.enabled) return;
    const startToken = ++faceTracking.startToken;
    faceTracking.initializing = true;
    setFaceStatus('Starting webcam + face tracker...');
    syncFaceTrackingToggleButton();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 60, max: 120 },
        },
      });
      if (startToken !== faceTracking.startToken || !faceTracking.initializing) {
        for (const track of stream.getTracks()) track.stop();
        return;
      }

      const video = document.createElement('video');
      video.autoplay = true;
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      try { video.play(); } catch {}
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        await new Promise((resolve) => {
          let done = false;
          const finish = () => {
            if (done) return;
            done = true;
            video.removeEventListener('loadeddata', onLoaded);
            resolve();
          };
          const onLoaded = () => finish();
          video.addEventListener('loadeddata', onLoaded, { once: true });
          setTimeout(finish, 1200);
        });
      }
      if (startToken !== faceTracking.startToken || !faceTracking.initializing) {
        try { video.pause(); } catch {}
        video.srcObject = null;
        for (const track of stream.getTracks()) track.stop();
        return;
      }

      const worker = new Worker(new URL('./face-tracker-worker.js', import.meta.url));
      if (startToken !== faceTracking.startToken || !faceTracking.initializing) {
        try { worker.terminate(); } catch {}
        try { video.pause(); } catch {}
        video.srcObject = null;
        for (const track of stream.getTracks()) track.stop();
        return;
      }
      faceTracking.worker = worker;
      faceTracking.stream = stream;
      faceTracking.video = video;
      faceTracking.enabled = true;
      faceTracking.ready = false;
      faceTracking.engine = 'worker';
      faceTracking.frameInFlight = false;
      faceTracking.frameInFlightSince = 0;
      faceTracking.droppedFrames = 0;
      faceTracking.errorStreak = 0;
      faceTracking.lastFaceSeenAt = -1;
      faceTracking.rawLandmarks = null;
      faceTracking.rawLandmarkCount = 0;
      faceTracking.blendshapeScores = null;
      faceTracking.blendshapeCount = 0;
      faceTracking.transformMatrix = null;
      faceTracking.matrixMotion = 0;
      faceTracking.poseRoll = 0;
      faceTracking.poseScale = 1;
      faceTracking.prevCenterX = NaN;
      faceTracking.prevCenterY = NaN;
      faceTracking.centerVelX = 0;
      faceTracking.centerVelY = 0;
      faceTracking.smoothedMouth = 0;
      faceTracking.smoothedEyeLeft = 1;
      faceTracking.smoothedEyeRight = 1;
      faceTracking.mainFallbackAttempted = false;
      if (faceTracking.initWatchdog) clearTimeout(faceTracking.initWatchdog);
      faceTracking.initWatchdog = 0;
      syncFaceTrackingToggleButton();

      worker.onmessage = (ev) => {
        const msg = ev.data || {};
        if (msg.type === 'init-ok') {
          faceTracking.ready = true;
          if (faceTracking.initWatchdog) {
            clearTimeout(faceTracking.initWatchdog);
            faceTracking.initWatchdog = 0;
          }
          const delegate = msg.delegate || 'cpu';
          setFaceStatus(`Face tracking active (${delegate}, worker thread)`);
          return;
        }
        if (msg.type === 'delegate-fallback') {
          const delegate = msg.delegate || 'cpu';
          setFaceStatus(`Face tracking active (${delegate}, recovered)`);
          return;
        }
        if (msg.type === 'init-error') {
          console.error('Face tracker init error:', msg.error || 'unknown error');
          setFaceStatus(`Face tracker failed: ${msg.error || 'unknown error'}`, true);
          stopFaceTracking(true);
          return;
        }
        if (msg.type === 'result') {
          faceTracking.frameInFlight = false;
          faceTracking.frameInFlightSince = 0;
          const infMs = msg.inferenceMs || 0;
          if (infMs > 0) {
            const target = Math.max(1000 / 30, Math.min(1000 / 55, infMs * 1.2));
            faceTracking.frameEveryMs = faceTracking.frameEveryMs * 0.82 + target * 0.18;
          }
          const lm = msg.landmarks ? new Float32Array(msg.landmarks) : null;
          applyFaceLandmarkResult(lm, infMs, msg.error || '', {
            blendshapes: msg.blendshapes || null,
            blendshapeCount: msg.blendshapeCount || 0,
            matrix: msg.matrix || null,
          });
        }
      };

      worker.onerror = (err) => {
        console.error('Face tracker worker error:', err?.message || err);
        setFaceStatus('Face tracker worker crashed', true);
        stopFaceTracking(true);
      };

      worker.onmessageerror = (err) => {
        console.error('Face tracker message error:', err?.message || err);
      };

      worker.postMessage({
        type: 'init',
        bundleURL: FACE_TRACKER_BUNDLE_URL,
        wasmURL: FACE_TRACKER_WASM_URL,
        modelURL: FACE_TRACKER_MODEL_URL,
        preferGPU: false,
      });
      setFaceStatus('Initializing face tracker...');
      faceTracking.initWatchdog = setTimeout(() => {
        if (faceTracking.enabled && !faceTracking.ready) {
          setFaceStatus('Face tracker is still initializing...', true);
          console.warn('Face tracker init is taking longer than expected.');
          switchToMainThreadTracker('worker-init-timeout');
        }
      }, 6000);
    } catch (err) {
      if (startToken !== faceTracking.startToken) return;
      console.error('Face tracking start failed:', err);
      setFaceStatus(`Face tracking unavailable: ${err?.message || err}`, true);
      stopFaceTracking(true);
    } finally {
      if (startToken === faceTracking.startToken) {
        faceTracking.initializing = false;
      }
      syncFaceTrackingToggleButton();
    }
  }

  function stopFaceTracking(preserveStatus = false) {
    faceTracking.startToken++;
    faceTracking.initializing = false;
    faceTracking.enabled = false;
    faceTracking.ready = false;
    faceTracking.engine = 'worker';
    faceTracking.frameInFlight = false;
    faceTracking.frameInFlightSince = 0;
    faceTracking.face = null;
    faceTracking.rawLandmarks = null;
    faceTracking.rawLandmarkCount = 0;
    faceTracking.blendshapeScores = null;
    faceTracking.blendshapeCount = 0;
    faceTracking.transformMatrix = null;
    faceTracking.matrixMotion = 0;
    faceTracking.poseRoll = 0;
    faceTracking.poseScale = 1;
    faceTracking.prevMapped = null;
    faceTracking.prevCenterX = NaN;
    faceTracking.prevCenterY = NaN;
    faceTracking.centerVelX = 0;
    faceTracking.centerVelY = 0;
    faceTracking.smoothedMouth = 0;
    faceTracking.smoothedEyeLeft = 1;
    faceTracking.smoothedEyeRight = 1;
    faceTracking.prevMouth = 0;
    faceTracking.lastFaceSeenAt = -1;
    faceTracking.lastMouthBurstTime = -999;
    faceTracking.errorStreak = 0;
    faceTracking.mainFallbackAttempted = false;
    if (faceTracking.initWatchdog) {
      clearTimeout(faceTracking.initWatchdog);
      faceTracking.initWatchdog = 0;
    }
    if (faceTracking.worker) {
      try { faceTracking.worker.postMessage({ type: 'dispose' }); } catch {}
      try { faceTracking.worker.terminate(); } catch {}
    }
    if (faceTracking.stream) {
      for (const track of faceTracking.stream.getTracks()) track.stop();
    }
    if (faceTracking.video) {
      try { faceTracking.video.pause(); } catch {}
      faceTracking.video.srcObject = null;
    }
    if (faceTracking.mainLandmarker) {
      try { faceTracking.mainLandmarker.close?.(); } catch {}
    }
    faceTracking.worker = null;
    faceTracking.stream = null;
    faceTracking.video = null;
    faceTracking.mainLandmarker = null;
    syncFaceTrackingToggleButton();
    if (!preserveStatus) {
      setFaceStatus('Face tracking idle');
    }
  }

  function pumpFaceTracker(nowMs) {
    if (!faceTracking.enabled || !faceTracking.ready) return;
    const v = faceTracking.video;
    if (!v || v.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

    if (faceTracking.engine === 'main') {
      if (!faceTracking.mainLandmarker) return;
      if (nowMs < faceTracking.nextFrameAt) return;
      faceTracking.nextFrameAt = nowMs + faceTracking.frameEveryMs;
      const t0 = performance.now();
      try {
        const res = faceTracking.mainLandmarker.detectForVideo(v, nowMs);
        const blendPayload = parseBlendshapeFromResult(res);
        const matrixPayload = parseMatrixFromResult(res);
        const lms = res?.faceLandmarks?.[0];
        if (!lms || !lms.length) {
          applyFaceLandmarkResult(null, performance.now() - t0, '', {
            blendshapes: blendPayload.scores,
            blendshapeCount: blendPayload.count,
            matrix: matrixPayload,
          });
          return;
        }
        const packed = new Float32Array(lms.length * 3);
        for (let i = 0; i < lms.length; i++) {
          const off = i * 3;
          packed[off] = lms[i].x;
          packed[off + 1] = lms[i].y;
          packed[off + 2] = lms[i].z;
        }
        applyFaceLandmarkResult(packed, performance.now() - t0, '', {
          blendshapes: blendPayload.scores,
          blendshapeCount: blendPayload.count,
          matrix: matrixPayload,
        });
      } catch (err) {
        applyFaceLandmarkResult(null, performance.now() - t0, err?.message || String(err));
      }
      return;
    }

    if (faceTracking.frameInFlight) {
      if (faceTracking.frameInFlightSince > 0 && (nowMs - faceTracking.frameInFlightSince) > faceTracking.frameTimeoutMs) {
        faceTracking.frameInFlight = false;
        faceTracking.frameInFlightSince = 0;
        faceTracking.droppedFrames++;
        console.warn('Face tracker frame timed out; recovering frame pump.');
      } else {
        return;
      }
    }
    if (nowMs < faceTracking.nextFrameAt) return;

    faceTracking.frameInFlight = true;
    faceTracking.frameInFlightSince = nowMs;
    faceTracking.nextFrameAt = nowMs + faceTracking.frameEveryMs;
    createImageBitmap(v)
      .then((bitmap) => {
        if (!faceTracking.enabled || !faceTracking.worker) {
          bitmap.close();
          faceTracking.frameInFlight = false;
          faceTracking.frameInFlightSince = 0;
          return;
        }
        faceTracking.worker.postMessage({
          type: 'process',
          frame: bitmap,
          timestampMs: nowMs,
        }, [bitmap]);
      })
      .catch((err) => {
        faceTracking.frameInFlight = false;
        faceTracking.frameInFlightSince = 0;
        faceTracking.droppedFrames++;
        if (faceTracking.droppedFrames % 30 === 0) {
          console.warn('Face tracker frame copy failed:', err?.message || err);
        }
      });
  }

  function applyFaceEffectors(dt, modeTime) {
    if (Math.round(state.faceEffectorMode || 0) <= 0) return;
    const face = faceTracking.face;
    if (!face) return;

    if (!faceTracking.prevMapped || faceTracking.prevMapped.length !== face.count * 2) {
      faceTracking.prevMapped = new Float32Array(face.count * 2);
      faceTracking.prevMapped.fill(NaN);
    }

    const blend = faceTracking.blendshapeScores;
    const jaw = blendScore(blend, 'jawOpen', face.mouthOpen);
    const pucker = blendScore(blend, 'mouthPucker', 0);
    const funnel = blendScore(blend, 'mouthFunnel', 0);
    const smile = (blendScore(blend, 'mouthSmileLeft', 0) + blendScore(blend, 'mouthSmileRight', 0)) * 0.5;
    const blinkL = blendScore(blend, 'eyeBlinkLeft', 1 - (face.eyeLeftOpen ?? 0.5));
    const blinkR = blendScore(blend, 'eyeBlinkRight', 1 - (face.eyeRightOpen ?? 0.5));
    const browLift = blendScore(blend, 'browInnerUp', 0);
    const cheekPuff = blendScore(blend, 'cheekPuff', 0);

    const eyeLeftOpen = Math.max(0, Math.min(1, (face.eyeLeftOpen ?? 0.5) * (1 - blinkL * 0.9)));
    const eyeRightOpen = Math.max(0, Math.min(1, (face.eyeRightOpen ?? 0.5) * (1 - blinkR * 0.9)));
    const mouth = Math.max(face.mouthOpen, jaw * 0.95);
    const mouthDelta = mouth - faceTracking.prevMouth;
    const mouthBurst = mouth > 0.52 && mouthDelta > 0.06 && (modeTime - faceTracking.lastMouthBurstTime) > 0.11;
    if (mouthBurst) faceTracking.lastMouthBurstTime = modeTime;
    const poseMotion = Math.max(0, Math.min(1.25, faceTracking.matrixMotion * 1.85 + Math.hypot(faceTracking.centerVelX, faceTracking.centerVelY) * 6.5));
    const getPoint = (idx) => facePoint(face, idx);
    const mouthCenter = [face.mouthCenterX, face.mouthCenterY];
    const flowX = faceTracking.centerVelX * state.splatForce * 0.012;
    const flowY = faceTracking.centerVelY * state.splatForce * 0.012;
    const fillCol = palette(modeTime * (0.03 + smile * 0.015), 2);
    const rimCol = palette(modeTime * 0.09 + browLift * 0.07, 5);

    const carveLoop = (indices, openness, sizeMul = 1.0) => {
      const closed = 1.0 - Math.max(0, Math.min(1, openness));
      for (let i = 0; i < indices.length; i++) {
        const idx = indices[i];
        const pt = getPoint(idx);
        if (!pt) continue;
        const [ix, iy] = direction(pt[0], pt[1], face.centerX, face.centerY);
        const force = state.splatForce * (0.03 + closed * 0.2 + poseMotion * 0.08);
        addSplat(
          pt[0], pt[1],
          ix * force + flowX * 0.3,
          iy * force + flowY * 0.3,
          -0.06 * (0.8 + closed),
          -0.06 * (0.8 + closed),
          -0.06 * (0.8 + closed),
          state.splatRadius * (2.0 + sizeMul * (1.1 + closed * 1.2))
        );
      }
    };

    // 1) Rim silhouette: bright edge to make head outline unmistakable.
    for (let i = 0; i < FACE_IDX.contour.length; i += 2) {
      const idx = FACE_IDX.contour[i];
      const pt = getPoint(idx);
      if (!pt) continue;
      const [rx, ry] = direction(face.centerX, face.centerY, pt[0], pt[1]);
      const tx = -ry;
      const ty = rx;
      const tangential = Math.sin(modeTime * 5.0 + i * 0.37);
      const force = state.splatForce * (0.055 + mouth * 0.14 + poseMotion * 0.12);
      const tint = 0.045 + smile * 0.04;
      addSplat(
        pt[0], pt[1],
        rx * force + tx * force * 0.18 * tangential + flowX,
        ry * force + ty * force * 0.18 * tangential + flowY,
        rimCol[0] * tint, rimCol[1] * tint, rimCol[2] * tint,
        state.splatRadius * (2.5 + mouth * 1.2 + cheekPuff * 0.7)
      );
    }

    // 2) Interior fill: paint a recognizable face volume instead of sparse points.
    for (let i = 0; i < FACE_IDX.contour.length; i += 4) {
      const idx = FACE_IDX.contour[i];
      const pt = getPoint(idx);
      if (!pt) continue;
      for (const t of [0.34, 0.62]) {
        const x = face.centerX + (pt[0] - face.centerX) * t;
        const y = face.centerY + (pt[1] - face.centerY) * t;
        const g = 0.03 + mouth * 0.045 + poseMotion * 0.04;
        addSplat(
          x, y,
          flowX * (0.6 + t * 0.3), flowY * (0.6 + t * 0.3),
          fillCol[0] * g, fillCol[1] * g, fillCol[2] * g,
          state.splatRadius * (2.8 - t * 0.7 + smile * 0.35)
        );
      }
    }

    // 3) Eye and mouth holes.
    carveLoop(FACE_IDX.leftEye, eyeLeftOpen, 0.9);
    carveLoop(FACE_IDX.rightEye, eyeRightOpen, 0.9);
    carveLoop(FACE_IDX.mouthHole, 1.0 - Math.max(0, Math.min(1, mouth * (0.9 + pucker * 0.2))), 1.2 + mouth * 1.5);

    // 4) Mouth jets driven by jaw + funnel/pucker blendshapes.
    const mouthDrive = Math.max(0, Math.min(1, mouth * 0.72 + pucker * 0.18 + funnel * 0.15));
    if (mouthDrive > 0.16) {
      for (let i = 0; i < FACE_IDX.mouthHole.length; i += 2) {
        const idx = FACE_IDX.mouthHole[i];
        const pt = getPoint(idx);
        if (!pt) continue;
        const [mx, my] = direction(mouthCenter[0], mouthCenter[1], pt[0], pt[1]);
        const burst = state.splatForce * (0.06 + mouthDrive * 0.35);
        const g = 0.03 + mouthDrive * 0.08;
        addSplat(
          pt[0], pt[1],
          mx * burst + flowX * 0.5, my * burst + flowY * 0.5,
          fillCol[0] * g, fillCol[1] * g, fillCol[2] * g,
          state.splatRadius * (2.4 + mouthDrive * 2.6)
        );
      }
    }

    // 5) Matrix-driven motion wake.
    if (poseMotion > 0.05) {
      for (let i = 1; i < FACE_IDX.contour.length; i += 5) {
        const idx = FACE_IDX.contour[i];
        const pt = getPoint(idx);
        if (!pt) continue;
        const wake = state.splatForce * (0.045 + poseMotion * 0.18);
        addSplat(
          pt[0], pt[1],
          flowX * (1.1 + poseMotion * 0.7), flowY * (1.1 + poseMotion * 0.7),
          0, 0, 0,
          state.splatRadius * (2.1 + poseMotion * 1.6)
        );
        if (mouthBurst) {
          const [rx, ry] = direction(face.centerX, face.centerY, pt[0], pt[1]);
          addSplat(
            pt[0], pt[1],
            rx * wake * 0.7, ry * wake * 0.7,
            -0.05, -0.05, -0.05,
            state.splatRadius * 2.8
          );
        }
      }
    }

    for (let i = 0; i < face.count; i++) {
      const off = i * 2;
      faceTracking.prevMapped[off] = face.mapped[off];
      faceTracking.prevMapped[off + 1] = face.mapped[off + 1];
    }
    faceTracking.prevMouth = mouth;
  }

  // ─── Jet Emitters (flick-style moving injectors) ────────────────────────
  const burstEmitters = [];
  const burstShots = [];
  let burstEmitterStamp = '';
  let burstGlobalPhase = Math.random() * Math.PI * 2;

  function wrapAngle(v) {
    const TAU = Math.PI * 2;
    let a = v % TAU;
    if (a < 0) a += TAU;
    return a;
  }

  function ensureBurstEmitters(count, behavior) {
    const wanted = Math.max(0, Math.round(count));
    const stamp = `${wanted}:${behavior}`;
    if (stamp === burstEmitterStamp && burstEmitters.length === wanted) return;
    burstEmitterStamp = stamp;
    burstEmitters.length = 0;
    for (let i = 0; i < wanted; i++) {
      const angle = (i / Math.max(1, wanted)) * Math.PI * 2;
      burstEmitters.push({
        angle,
        cooldown: Math.random() * 0.25,
        drift: (Math.random() < 0.5 ? -1 : 1) * (0.7 + Math.random() * 0.8),
      });
    }
  }

  function clampPointToSphere(px, py, margin = 0.02) {
    const maxR = Math.max(0.02, SIM_SPHERE_RADIUS - margin);
    let dx = px - 0.5;
    let dy = py - 0.5;
    const d = Math.hypot(dx, dy);
    if (d > maxR) {
      const s = maxR / Math.max(d, 1e-6);
      dx *= s;
      dy *= s;
    }
    return { x: 0.5 + dx, y: 0.5 + dy };
  }

  function spawnBurstShot(e, behavior, idx, count, speedNorm, duration, width) {
    if (burstShots.length > 160) return;
    const TAU = Math.PI * 2;
    const slot = idx / Math.max(1, count);
    const edgeSourceR = SIM_SPHERE_RADIUS + 0.012;
    const centerSourceR = 0.02 + width * 0.05;
    const innerTargetMin = 0.03 + width * 0.05;
    const innerTargetMax = Math.max(innerTargetMin + 0.02, SIM_SPHERE_RADIUS - 0.09);

    let sourceAngle = e.angle;
    let sourceR = edgeSourceR;
    let targetAngle = sourceAngle + Math.PI;
    let targetR = 0.12;

    if (behavior === 1) {
      sourceAngle = slot * TAU + burstGlobalPhase * 0.15;
      sourceR = edgeSourceR;
      targetAngle = sourceAngle + Math.PI + (Math.random() * 2 - 1) * (0.16 + width * 0.3);
      targetR = 0.02 + Math.random() * (0.08 + width * 0.08);
    } else if (behavior === 2) {
      sourceAngle = slot * TAU + burstGlobalPhase * 0.3;
      sourceR = centerSourceR;
      targetAngle = sourceAngle + (Math.random() * 2 - 1) * (0.2 + width * 0.25);
      targetR = SIM_SPHERE_RADIUS - (0.045 + width * 0.05) - Math.random() * 0.03;
    } else if (behavior === 3) {
      e.angle = wrapAngle(e.angle + (0.2 + speedNorm * 0.9) * e.drift);
      sourceAngle = e.angle;
      sourceR = edgeSourceR;
      targetAngle = sourceAngle + Math.PI * (0.65 + Math.random() * 0.6) * e.drift;
      targetR = 0.08 + Math.random() * (SIM_SPHERE_RADIUS - 0.13);
    } else if (behavior === 4) {
      sourceAngle = slot * TAU + burstGlobalPhase * (0.8 + speedNorm * 1.0);
      sourceR = edgeSourceR;
      targetAngle = burstGlobalPhase * 1.2 + slot * TAU * 0.6 + (Math.random() * 2 - 1) * 0.2;
      targetR = 0.07 + Math.random() * 0.14;
    } else {
      e.angle = wrapAngle(e.angle + (Math.random() * 2 - 1) * 0.45 + e.drift * (0.12 + speedNorm * 0.35));
      sourceAngle = e.angle;
      sourceR = edgeSourceR;
      targetAngle = Math.random() * TAU;
      targetR = innerTargetMin + Math.random() * Math.max(0.02, innerTargetMax - innerTargetMin);
    }

    const sx = 0.5 + Math.cos(sourceAngle) * sourceR;
    const sy = 0.5 + Math.sin(sourceAngle) * sourceR;
    const tx0 = 0.5 + Math.cos(targetAngle) * targetR;
    const ty0 = 0.5 + Math.sin(targetAngle) * targetR;
    const t = clampPointToSphere(tx0, ty0, 0.03);
    const dx = t.x - sx;
    const dy = t.y - sy;
    const d = Math.hypot(dx, dy);
    if (d < 1e-5) return;

    const jitter = 1 + (Math.random() * 2 - 1) * state.burstForceRandomness;
    const speed = 0.9 + speedNorm * 2.1; // world units / second
    const life = Math.min(0.75, 0.05 + duration * 0.18);
    const forceScale = Math.max(0.15, (0.45 + state.burstForce * 1.5) * jitter);
    const dyeScale = forceScale * (1.8 + width * 0.8); // intentionally dye-heavy
    const radius = state.splatRadius * (0.95 + width * 3.4 + state.burstForce * 0.28);

    burstShots.push({
      x: sx,
      y: sy,
      vx: (dx / d) * speed,
      vy: (dy / d) * speed,
      life,
      maxLife: life,
      forceScale,
      dyeScale,
      radius,
      colorOffset: sourceAngle * 0.17 + Math.random() * 0.2,
    });
  }

  function updateBurstEmitters(dt) {
    const behavior = Math.max(0, Math.min(4, Math.round(state.burstBehavior)));
    const count = Math.max(0, Math.round(state.burstCount));
    if (count === 0 || state.burstForce <= 0.001) {
      burstShots.length = 0;
      return;
    }

    ensureBurstEmitters(count, behavior);
    const speedNorm = Math.max(0, Math.min(1, state.burstSpeed / 2.0));
    const duration = Math.max(0.05, Number.isFinite(state.burstDuration) ? state.burstDuration : 0.8);
    const width = Math.max(0, Math.min(1, Number.isFinite(state.burstWidth) ? state.burstWidth : 0.25));
    burstGlobalPhase = wrapAngle(burstGlobalPhase + dt * (0.15 + speedNorm * 1.5));

    const shotInterval = Math.max(0.16, (1.25 - speedNorm * 0.7) + Math.min(0.45, duration * 0.12));
    for (let i = 0; i < burstEmitters.length; i++) {
      const e = burstEmitters[i];
      e.cooldown -= dt;
      if (e.cooldown <= 0) {
        spawnBurstShot(e, behavior, i, count, speedNorm, duration, width);
        e.cooldown = shotInterval * (0.8 + Math.random() * 0.45);
      }
    }

    for (let i = burstShots.length - 1; i >= 0; i--) {
      const s = burstShots[i];
      const prevX = s.x;
      const prevY = s.y;
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      s.life -= dt;

      const distC = Math.hypot(s.x - 0.5, s.y - 0.5);
      if (s.life <= 0 || distC > SIM_SPHERE_RADIUS + 0.08) {
        burstShots.splice(i, 1);
        continue;
      }

      const stepX = s.x - prevX;
      const stepY = s.y - prevY;
      if (Math.abs(stepX) + Math.abs(stepY) < 1e-6) continue;

      const ageT = 1 - s.life / Math.max(1e-6, s.maxLife);
      const fade = Math.max(0.25, 1 - ageT * 0.7);
      const force = s.forceScale * fade;
      const dye = s.dyeScale * (0.72 + fade * 0.38);
      const col = palette(time * 0.13 + s.colorOffset + ageT * 0.3, 4);
      const samples = 2;
      for (let j = 1; j <= samples && splatCount < MAX_SPLATS; j++) {
        const t = j / samples;
        const px = prevX + stepX * t;
        const py = prevY + stepY * t;
        addSplat(
          px,
          py,
          stepX * state.splatForce * force,
          stepY * state.splatForce * force,
          col[0] * dye,
          col[1] * dye,
          col[2] * dye,
          s.radius * (1 - ageT * 0.22)
        );
      }

      const drag = Math.max(0, 1 - dt * 2.2);
      s.vx *= drag;
      s.vy *= drag;
    }
  }

  // ─── Rebuild particle system (called when count changes) ───────────────
  function rebuildParticleSystem(count) {
    count = Math.min(count, maxParticles);

    const newBuf = device.createBuffer({
      label: 'particles',
      size: count * PARTICLE_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const newColorBuf = device.createBuffer({
      label: 'particleColors',
      size: count * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    gpuInitParticles(newBuf, count);

    particleUpdatePipeline = device.createComputePipeline({
      label: 'particleUpdate',
      layout: device.createPipelineLayout({ bindGroupLayouts: [particleUpdateBGL] }),
      compute: {
        module: device.createShaderModule({ code: makeParticleUpdateShader(count, hdrSupported), label: 'particleUpdate' }),
        entryPoint: 'main',
      },
    });

    particleUpdateBGs = makeParticleUpdateBGs(newBuf, newColorBuf);

    // Rebuild indirect draw resources
    const newVisibleIndexBuf = device.createBuffer({
      label: 'visibleIndices',
      size: count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    particleCompactPipeline = device.createComputePipeline({
      label: 'particleCompact',
      layout: particleCompactLayout,
      compute: {
        module: device.createShaderModule({ code: makeParticleCompactShader(count), label: 'particleCompact' }),
        entryPoint: 'main',
      },
    });

    particleCompactBG = device.createBindGroup({
      layout: particleCompactBGL,
      entries: [
        { binding: 0, resource: { buffer: newColorBuf } },
        { binding: 1, resource: { buffer: newVisibleIndexBuf } },
        { binding: 2, resource: { buffer: atomicCounterBuf } },
      ],
    });

    particleRenderBG = device.createBindGroup({
      layout: particleRenderBGL,
      entries: [
        { binding: 0, resource: { buffer: newBuf } },
        { binding: 1, resource: { buffer: particleUB } },
        { binding: 2, resource: { buffer: newColorBuf } },
        { binding: 3, resource: { buffer: newVisibleIndexBuf } },
      ],
    });

    particleBuf = newBuf;
    colorBuf = newColorBuf;
    visibleIndexBuf = newVisibleIndexBuf;
    state.particleCount = count;
    particleDispatches = Math.ceil(count / PARTICLE_WG);

    console.log(`Rebuilt particle system: ${count} particles (${(count * PARTICLE_STRIDE / 1024 / 1024).toFixed(1)} MB)`);
  }

  // ─── Mouse / Pointer State ──────────────────────────────────────────────
  const pointer = { x: 0.5, y: 0.5, dx: 0, dy: 0, down: false, moved: false };

  // Convert screen coords to simulation UV (aspect-corrected 1:1 square)
  function screenToSimUV(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const rawX = (clientX - rect.left) / rect.width;
    const rawY = 1.0 - (clientY - rect.top) / rect.height;
    const aspect = rect.width / rect.height;
    const simX = (rawX - 0.5) * Math.max(aspect, 1.0) + 0.5;
    const simY = (rawY - 0.5) * Math.max(1.0 / aspect, 1.0) + 0.5;
    return [simX, simY];
  }

  canvas.addEventListener('pointerdown', e => {
    pointer.down = true;
    const [sx, sy] = screenToSimUV(e.clientX, e.clientY);
    pointer.x = sx;
    pointer.y = sy;
  });
  canvas.addEventListener('pointerup', () => {
    pointer.down = false;
  });
  canvas.addEventListener('pointermove', e => {
    const [nx, ny] = screenToSimUV(e.clientX, e.clientY);
    pointer.dx = nx - pointer.x;
    pointer.dy = ny - pointer.y;
    pointer.x = nx;
    pointer.y = ny;
    pointer.moved = true;
  });

  // ─── Scientific Colormaps (Inferno, Magma, Plasma, Viridis) ─────────
  // Polynomial approximations by Matt Zucker / Inigo Quilez
  function clamp01(x) { return Math.max(0, Math.min(1, x)); }

  function inferno(t) {
    t = clamp01(t);
    const c0 = [0.0002, 0.0016, -0.0194];
    const c1 = [0.1065, 0.5639, 3.9327];
    const c2 = [11.6024, -3.9728, -15.9423];
    const c3 = [-41.7039, 17.4363, 44.3541];
    const c4 = [77.1629, -33.4023, -81.8073];
    const c5 = [-73.6891, 32.6269, 73.2088];
    const c6 = [27.1632, -12.2461, -23.0702];
    return [0,1,2].map(i =>
      clamp01(c0[i]+t*(c1[i]+t*(c2[i]+t*(c3[i]+t*(c4[i]+t*(c5[i]+t*c6[i]))))))
    );
  }

  function magma(t) {
    t = clamp01(t);
    const c0 = [-0.0021, -0.0008, -0.0053];
    const c1 = [0.2516, 0.6775, 2.4946];
    const c2 = [8.3537, -3.5775, -8.6687];
    const c3 = [-27.6684, 14.2647, 27.1596];
    const c4 = [52.1761, -27.9436, -50.7682];
    const c5 = [-50.7685, 29.0467, 45.7131];
    const c6 = [18.6557, -11.4898, -15.8989];
    return [0,1,2].map(i =>
      clamp01(c0[i]+t*(c1[i]+t*(c2[i]+t*(c3[i]+t*(c4[i]+t*(c5[i]+t*c6[i]))))))
    );
  }

  function plasma(t) {
    t = clamp01(t);
    const c0 = [0.0587, 0.0234, 0.5433];
    const c1 = [2.1761, 0.2138, -2.6346];
    const c2 = [-6.8084, 6.2608, 12.6420];
    const c3 = [17.6953, -24.0146, -27.6687];
    const c4 = [-26.6811, 39.5587, 32.2827];
    const c5 = [20.5835, -31.7652, -20.0279];
    const c6 = [-6.0116, 10.5195, 5.7893];
    return [0,1,2].map(i =>
      clamp01(c0[i]+t*(c1[i]+t*(c2[i]+t*(c3[i]+t*(c4[i]+t*(c5[i]+t*c6[i]))))))
    );
  }

  function viridis(t) {
    t = clamp01(t);
    const c0 = [0.2777, 0.0054, 0.3340];
    const c1 = [0.1050, 1.4046, 1.3840];
    const c2 = [-0.3308, 0.2148, -4.7950];
    const c3 = [-4.6342, -5.7991, 12.2624];
    const c4 = [6.2282, 14.1799, -14.0464];
    const c5 = [4.7763, -13.7451, 6.0756];
    const c6 = [-5.4354, 4.6459, -0.6946];
    return [0,1,2].map(i =>
      clamp01(c0[i]+t*(c1[i]+t*(c2[i]+t*(c3[i]+t*(c4[i]+t*(c5[i]+t*c6[i]))))))
    );
  }

  // Liquid gold colormap: deep amber → rich gold → bright orange-gold
  function liquidGold(t) {
    t = clamp01(t);
    const r = clamp01(0.15 + t * 1.1 - t * t * 0.25);
    const g = clamp01(0.04 + t * 0.7 - t * t * 0.15);
    const b = clamp01(0.01 + t * 0.12 - t * t * 0.05);
    return [r, g, b];
  }

  const colormaps = [inferno, magma, plasma, viridis, liquidGold];

  const _palOut = [0, 0, 0];
  function palette(t, mapIndex) {
    const cm = colormaps[mapIndex % colormaps.length];
    const remapped = 0.15 + (((t % 1) + 1) % 1) * 0.8;
    const col = cm(remapped);
    _palOut[0] = col[0] * 2.0;
    _palOut[1] = col[1] * 2.0;
    _palOut[2] = col[2] * 2.0;
    return _palOut;
  }
  let time = 0;

  // ─── Frame loop ─────────────────────────────────────────────────────────
  function frame() {
    if (!frameRunning) return;
    requestAnimationFrame(frame);
    const nowMs = performance.now();

    if (canvas.width < 1 || canvas.height < 1) {
      resize();
      return;
    }

    pumpFaceTracker(nowMs);
    drawFaceDebugOverlay(nowMs);

    // Cap GPU queue depth: skip frame if GPU is >2 frames behind
    if (gpuFramesPending > 2) {
      gpuFramesSkipped++;
      return;
    }
    gpuFramesPending++;

    // Continuous time scaling: render every frame, but advance simulation with smaller dt.
    // This avoids the frame-skipping look while still slowing all calculations.
    const masterSpeed = Math.max(0, Math.min(1, Number.isFinite(state.masterSpeed) ? state.masterSpeed : 1.0));
    frameTimeScale = masterSpeed;
    const dt = 0.016 * state.simSpeed * masterSpeed;
    time += dt;

    updateAutoMorph();

    // Mood lighting: shift colors with slow sinusoidal cycle
    let moodBaseColor = state.baseColor;
    let moodAccentColor = state.accentColor;
    let moodTipColor = state.tipColor;
    if (state.moodAmount > 0.01) {
      const moodPhase = time * state.moodSpeed * 0.1;
      const warmShift = Math.sin(moodPhase) * state.moodAmount * 0.15;
      const shiftColor = (c) => [
        Math.max(0, Math.min(2, c[0] + warmShift)),
        Math.max(0, Math.min(2, c[1] + warmShift * 0.5)),
        Math.max(0, Math.min(2, c[2] - warmShift * 0.3)),
      ];
      moodBaseColor = shiftColor(state.baseColor);
      moodAccentColor = shiftColor(state.accentColor);
      moodTipColor = shiftColor(state.tipColor);
    }

    // Pre-convert colors to Oklab on CPU (avoids per-particle GPU conversion)
    const okGlitBase = linearToOklabCPU(state.glitterColor);
    const okGlitAccent = linearToOklabCPU(state.glitterAccent);
    const okGlitTip = linearToOklabCPU(state.glitterTip);
    const okBaseCol = linearToOklabCPU(moodBaseColor);
    const okAccentCol = linearToOklabCPU(moodAccentColor);
    const okTipCol = linearToOklabCPU(moodTipColor);

    // Update screen/particle/display uniform buffers
    particleUBData[0] = canvas.width;
    particleUBData[1] = canvas.height;
    particleUBData[2] = state.particleSize * (window.devicePixelRatio || 1);
    particleUBData[3] = state.glintBrightness;
    particleUBData[4] = state.prismaticAmount;
    particleUBData[5] = okGlitBase[0];
    particleUBData[6] = okGlitBase[1];
    particleUBData[7] = okGlitBase[2];
    particleUBData[8] = okGlitAccent[0];
    particleUBData[9] = okGlitAccent[1];
    particleUBData[10] = okGlitAccent[2];
    particleUBData[11] = state.colorBlend;
    particleUBData[12] = state.sizeRandomness;
    particleUBData[13] = okGlitTip[0];
    particleUBData[14] = okGlitTip[1];
    particleUBData[15] = okGlitTip[2];
    device.queue.writeBuffer(particleUB, 0, particleUBData);

    displayUBData[0] = canvas.width;
    displayUBData[1] = canvas.height;
    displayUBData[2] = time;
    displayUBData[3] = state.sheenStrength;
    displayUBData[4] = okBaseCol[0];
    displayUBData[5] = okBaseCol[1];
    displayUBData[6] = okBaseCol[2];
    displayUBData[7] = 1.0; // sheen headroom (no HDR boost — sheen stays SDR-range)
    displayUBData[8] = okAccentCol[0];
    displayUBData[9] = okAccentCol[1];
    displayUBData[10] = okAccentCol[2];
    displayUBData[11] = state.colorBlend;
    displayUBData[12] = state.sheenColor[0];
    displayUBData[13] = state.sheenColor[1];
    displayUBData[14] = state.sheenColor[2];
    displayUBData[15] = state.metallic;
    displayUBData[16] = okTipCol[0];
    displayUBData[17] = okTipCol[1];
    displayUBData[18] = okTipCol[2];
    displayUBData[19] = state.roughness;
    // slots 20-23: display extras (20 carries symmetry behavior)
    displayUBData[20] = Math.round(state.noiseBehavior || 0);
    displayUBData[21] = 0;
    displayUBData[22] = 0;
    displayUBData[23] = 0;
    device.queue.writeBuffer(displayUB, 0, displayUBData);

    // Collect splats into pre-allocated buffer
    splatCount = 0;
    updateBurstEmitters(dt);

    // Mouse splat — clamp to sphere (tighter than visual edge to keep splat radius inside)
    const INTERACT_R = 0.35;
    const pdx = pointer.x - 0.5, pdy = pointer.y - 0.5;
    const pDist = Math.sqrt(pdx * pdx + pdy * pdy);
    const inSphere = pDist < INTERACT_R;

    if (pointer.moved && pointer.down && inSphere) {
      const speed = Math.sqrt(pointer.dx * pointer.dx + pointer.dy * pointer.dy);
      const col = palette(time * 0.3, 4);
      const str = state.clickStrength * 6.0;
      const sz = 1.0 + state.clickSize * 8.0;
      addSplat(pointer.x, pointer.y,
        pointer.dx * state.splatForce * str,
        pointer.dy * state.splatForce * str,
        col[0] * str, col[1] * str, col[2] * str,
        state.splatRadius * sz * (1.0 + speed * 20));
    }
    applyFaceEffectors(dt, time);
    if (pointer.moved) { pointer.moved = false; }

    // ── Upload batch splat data + write simulation params ──
    splatCountUData[0] = splatCount;
    if (splatCount > 0) device.queue.writeBuffer(splatCountBuf, 0, splatCountUData);
    if (splatCount > 0) {
      device.queue.writeBuffer(splatBuf, 0, splatArrayData, 0, splatCount * 8);
    }
    writeParams({ dt, time });

    // ── Single encoder for all passes ──
    const enc = device.createCommandEncoder();

    // ── Batch splat pass (1 vel dispatch + 1 dye dispatch instead of ~20) ──
    if (splatCount > 0) {
      const p1 = enc.beginComputePass();
      p1.setPipeline(batchSplatVelPipe);
      p1.setBindGroup(0, batchSplatVelBGs[velFlip]);
      p1.dispatchWorkgroups(dispatch, dispatch);
      p1.end();
      velFlip ^= 1;

      const p2 = enc.beginComputePass();
      p2.setPipeline(batchSplatDyePipe);
      p2.setBindGroup(0, batchSplatDyeBGs[dyeFlip]);
      p2.dispatchWorkgroups(dispatch, dispatch);
      p2.end();
      dyeFlip ^= 1;
    }

    // ── Temperature Splat (inject heat where dye splats land) ──
    if (splatCount > 0 && state.tempAmount > 0.01) {
      const p = enc.beginComputePass();
      p.setPipeline(tempSplatPipe);
      p.setBindGroup(0, tempSplatBGs[tempFlip]);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      tempFlip ^= 1;
    }

    // ── Pass 2+3: Fused Curl + Vorticity (reads vel[cur], writes curlTex + vel[1-cur]) ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(fusedCurlVortPipe);
      p.setBindGroup(0, fusedCurlVortBGs[velFlip]);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      velFlip ^= 1;
    }

    // ── Pass 4+5: Fused Divergence + Clear Pressure (reads vel[cur] + press[cur], writes divTex + press[1-cur]) ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(fusedDivClearPressPipe);
      p.setBindGroup(0, fusedDivClearPressBGs[velFlip][pressFlip]);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      pressFlip ^= 1;
    }

    // ── Pass 6: Jacobi Pressure Solve ──
    for (let i = 0; i < state.pressureIters; i++) {
      const p = enc.beginComputePass();
      p.setPipeline(jacobiPipe.pipeline);
      p.setBindGroup(0, jacobiBGs[pressFlip]);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      pressFlip ^= 1;
    }

    // ── Pass 7: Gradient Subtraction (reads vel[cur] + press[cur], writes vel[1-cur]) ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(gradSubPipe.pipeline);
      p.setBindGroup(0, gradSubBGs[velFlip][pressFlip]);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      velFlip ^= 1;
    }

    // ── Pass 8: Advect Velocity (reads vel[cur], writes vel[1-cur]) ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(advectVelPipe.pipeline);
      p.setBindGroup(0, advVelBGs[velFlip]);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      velFlip ^= 1;
    }

    // ── Pass 9: Advect Dye (reads vel[cur] + dye[cur], writes dye[1-cur]) ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(advectDyePipe.pipeline);
      p.setBindGroup(0, advDyeBGs[velFlip][dyeFlip]);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      dyeFlip ^= 1;
    }

    // ── Temperature Advect (reads vel[cur] + temp[cur], writes temp[1-cur]) ──
    if (state.tempAmount > 0.01) {
      tempParamData[0] = dt;
      tempParamData[1] = 1.0 / SIM_RES;
      tempParamData[2] = SIM_RES;
      tempParamData[3] = state.tempDissipation;
      device.queue.writeBuffer(tempParamBuf, 0, tempParamData);
      const p = enc.beginComputePass();
      p.setPipeline(tempAdvectPipe.pipeline);
      p.setBindGroup(0, tempAdvectBGs[velFlip][tempFlip]);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      tempFlip ^= 1;
    }

    // ── Buoyancy (reads temp[cur] + vel[cur], writes vel[1-cur]) ──
    if (state.tempAmount > 0.01 && state.tempBuoyancy > 0.01) {
      buoyancyData[0] = SIM_RES;
      buoyancyData[1] = dt;
      buoyancyData[2] = state.tempBuoyancy * 50.0;
      buoyancyData[3] = 0;
      device.queue.writeBuffer(buoyancyBuf, 0, buoyancyData);
      const p = enc.beginComputePass();
      p.setPipeline(buoyancyPipe.pipeline);
      p.setBindGroup(0, buoyancyBGs[tempFlip][velFlip]);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      velFlip ^= 1;
    }

    // ── Noise Field (type-selectable curl perturbation) ──
    if (state.noiseAmount > 0.01) {
      noiseData[0] = time;
      // Keep low-end control but avoid over-damping the effect.
      const na = state.noiseAmount;
      noiseData[1] = na * na * frameTimeScale;
      noiseData[2] = SIM_RES;
      noiseData[3] = state.noiseFrequency;
      noiseData[4] = state.noiseSpeed;
      noiseData[5] = Math.round(state.noiseType || 0);
      noiseData[6] = Math.round(state.noiseBehavior || 0);
      noiseData[7] = Math.round(state.noiseMapping || 0);
      noiseData[8] = state.noiseWarp;
      noiseData[9] = state.noiseSharpness;
      noiseData[10] = state.noiseAnisotropy;
      noiseData[11] = state.noiseBlend;
      device.queue.writeBuffer(noiseBuf, 0, noiseData);
      const p = enc.beginComputePass();
      p.setPipeline(curlNoisePipe.pipeline);
      p.setBindGroup(0, curlNoiseBGs[velFlip]);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      velFlip ^= 1;
    }

    // ── Dye Noise (convergence dye + curl noise dye) ──
    const hasDyeNoise = state.dyeNoiseAmount > 0.001;
    const hasCurlDye = state.noiseDyeIntensity > 0.01 && state.noiseAmount > 0.01;
    const enforceNoiseSymmetry = state.noiseAmount > 0.01 && state.noiseBehavior > 0.5;
    if (hasDyeNoise || hasCurlDye || enforceNoiseSymmetry) {
      const col = palette(time * 0.05, 4);
      dyeNoiseData[0] = time;
      dyeNoiseData[1] = hasDyeNoise ? state.dyeNoiseAmount * frameTimeScale : 0;
      dyeNoiseData[2] = SIM_RES;
      // noiseDyeIntensity controls how much dye the curl noise injects
      const ndi = state.noiseDyeIntensity;
      dyeNoiseData[3] = hasCurlDye ? ndi * ndi * frameTimeScale : 0;
      dyeNoiseData[4] = col[0];
      dyeNoiseData[5] = col[1];
      dyeNoiseData[6] = col[2];
      dyeNoiseData[7] = Math.round(state.noiseBehavior || 0);
      device.queue.writeBuffer(dyeNoiseBuf, 0, dyeNoiseData);
      const p = enc.beginComputePass();
      p.setPipeline(dyeNoisePipe.pipeline);
      p.setBindGroup(0, dyeNoiseBGs[velFlip][dyeFlip]);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      dyeFlip ^= 1;
    }

    // Reaction-diffusion runtime pass is intentionally disabled.

    // ── Sphere Cleanup (hard-zero vel+dye outside sphere) ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(cleanupPipeline);
      p.setBindGroup(0, cleanupBGs[velFlip][dyeFlip]);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      velFlip ^= 1;
      dyeFlip ^= 1;
    }

    // ── Pass 10: Particle Update (compute) ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(particleUpdatePipeline);
      p.setBindGroup(0, particleUpdateBGs[velFlip][dyeFlip]);
      p.dispatchWorkgroups(particleDispatches);
      p.end();
    }

    // ── Pass 10b: Particle Compact (build visible index list for indirect draw) ──
    device.queue.writeBuffer(atomicCounterBuf, 0, zeroU32);
    {
      const p = enc.beginComputePass();
      p.setPipeline(particleCompactPipeline);
      p.setBindGroup(0, particleCompactBG);
      p.dispatchWorkgroups(particleDispatches);
      p.end();
    }
    {
      const p = enc.beginComputePass();
      p.setPipeline(particleFinalizePipeline);
      p.setBindGroup(0, particleFinalizeBG);
      p.dispatchWorkgroups(1);
      p.end();
    }

    // ── Bloom setup ──
    const bloomActive = state.bloomIntensity > 0;
    if (bloomActive) {
      ensureBloomResources(canvas.width, canvas.height);
    } else if (bloomResources) {
      destroyBloomResources();
    }
    let canvasView;
    try {
      canvasView = ctx.getCurrentTexture().createView();
    } catch (err) {
      console.error('GPU: failed to acquire current texture view:', err?.message || err);
      resize();
      gpuFramesPending--;
      return;
    }
    const renderTarget = bloomActive ? bloomResources.sceneView : canvasView;

    // ── Pass 11: Display Render (fluid base) ──
    {
      const rp = enc.beginRenderPass({
        colorAttachments: [{
          view: renderTarget,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      rp.setPipeline(displayPipeline);
      rp.setBindGroup(0, displayBGs[dyeFlip]);
      rp.draw(3);
      rp.end();
    }

    // ── Pass 12: Particle Render (additive glitter layer) ──
    {
      const rp = enc.beginRenderPass({
        colorAttachments: [{
          view: renderTarget,
          loadOp: 'load',
          storeOp: 'store',
        }],
      });
      rp.setPipeline(particleRenderPipeline);
      rp.setBindGroup(0, particleRenderBG);
      rp.drawIndirect(drawIndirectBuf, 0);
      rp.end();
    }

    // ── Pass 13: Bloom Post-Processing ──
    if (bloomActive) {
      const br = bloomResources;
      // Compute per-mip weights from size slider
      function smoothstepJS(edge0, edge1, x) {
        if (edge0 >= edge1) return x >= edge1 ? 1.0 : 0.0;
        const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
        return t * t * (3 - 2 * t);
      }

      // Downsample chain (5 passes)
      let srcW = canvas.width, srcH = canvas.height;
      for (let i = 0; i < BLOOM_MIPS; i++) {
        const dstW = Math.max(1, srcW >> 1);
        const dstH = Math.max(1, srcH >> 1);
        bloomParamData[0] = 1.0 / srcW;  // texelSize.x (of source)
        bloomParamData[1] = 1.0 / srcH;  // texelSize.y
        bloomParamData[2] = state.bloomThreshold;
        bloomParamData[3] = state.bloomThreshold * 0.5; // knee
        bloomParamData[4] = 0; // unused in downsample
        bloomParamData[5] = 0; // unused
        bloomParamData[6] = i === 0 ? 1.0 : 0.0; // flags: threshold on first pass only
        bloomParamData[7] = 0;
        device.queue.writeBuffer(bloomParamBufs[i], 0, bloomParamData);

        const p = enc.beginComputePass();
        p.setPipeline(bloomDownPipe);
        p.setBindGroup(0, br.downBGs[i]);
        p.dispatchWorkgroups(Math.ceil(dstW / 8), Math.ceil(dstH / 8));
        p.end();

        srcW = dstW;
        srcH = dstH;
      }

      // Upsample chain (4 passes)
      for (let i = 0; i < BLOOM_MIPS - 1; i++) {
        const mipIdx = BLOOM_MIPS - 2 - i; // 3, 2, 1, 0
        const dstW = br.bloomUp[mipIdx].width;
        const dstH = br.bloomUp[mipIdx].height;
        // texelSize of the lower (source) mip
        const lowerW = i === 0 ? br.bloomDown[BLOOM_MIPS - 1].width : br.bloomUp[mipIdx + 1].width;
        const lowerH = i === 0 ? br.bloomDown[BLOOM_MIPS - 1].height : br.bloomUp[mipIdx + 1].height;
        const radiusNorm = Math.min(state.bloomRadius / 10.0, 1.0);
        // Coarser passes (low i) need higher radius to activate
        // depth: 0 for finest pass, 1 for coarsest pass
        const depth = (BLOOM_MIPS - 2 - i) / (BLOOM_MIPS - 2);
        const mipWeight = Math.pow(radiusNorm, 1.0 + depth * 2.0);

        bloomParamData[0] = 1.0 / lowerW;
        bloomParamData[1] = 1.0 / lowerH;
        bloomParamData[2] = 0;
        bloomParamData[3] = 0;
        bloomParamData[4] = 0;
        bloomParamData[5] = mipWeight;
        bloomParamData[6] = 0;
        bloomParamData[7] = 0;
        device.queue.writeBuffer(bloomParamBufs[BLOOM_MIPS + i], 0, bloomParamData);

        const p = enc.beginComputePass();
        p.setPipeline(bloomUpPipe);
        p.setBindGroup(0, br.upBGs[i]);
        p.dispatchWorkgroups(Math.ceil(dstW / 8), Math.ceil(dstH / 8));
        p.end();
      }

      // Composite: sceneTex + bloomUp[0] → canvasView
      bloomCompositeData[0] = state.bloomIntensity;
      device.queue.writeBuffer(bloomCompositeUB, 0, bloomCompositeData);
      {
        const rp = enc.beginRenderPass({
          colorAttachments: [{
            view: canvasView,
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          }],
        });
        rp.setPipeline(bloomCompositePipe);
        rp.setBindGroup(0, br.compositeBG);
        rp.draw(3);
        rp.end();
      }
    }

    try {
      device.queue.submit([enc.finish()]);
    } catch (err) {
      gpuFramesPending--;
      console.error('GPU submit failed:', err?.message || err);
      frameRunning = false;
      if (errorDiv) {
        errorDiv.style.display = 'block';
        errorDiv.textContent = `GPU submit failed: ${err?.message || err}`;
      }
      return;
    }
    device.queue.onSubmittedWorkDone()
      .then(() => { gpuFramesPending--; })
      .catch(err => {
        gpuFramesPending--;
        console.error('GPU work completion failed:', err?.message || err);
      });
  }

  // ─── Wire Settings UI ──────────────────────────────────────────────────
  const settingsToggle = document.getElementById('settingsToggle');
  const settingsPanel = document.getElementById('settingsPanel');
  settingsToggle.addEventListener('click', () => {
    settingsPanel.classList.toggle('open');
  });

  // Auto-hide settings icon after 3s of no mouse movement
  let hideTimer = null;
  function showSettingsIcon() {
    settingsToggle.classList.remove('hidden');
    clearTimeout(hideTimer);
    if (!settingsPanel.classList.contains('open')) {
      hideTimer = setTimeout(() => settingsToggle.classList.add('hidden'), 3000);
    }
  }
  document.addEventListener('mousemove', showSettingsIcon);
  // Keep icon visible while panel is open
  settingsToggle.addEventListener('click', () => {
    if (settingsPanel.classList.contains('open')) {
      clearTimeout(hideTimer);
      settingsToggle.classList.remove('hidden');
    } else {
      showSettingsIcon();
    }
  });
  // Start the initial hide timer
  hideTimer = setTimeout(() => settingsToggle.classList.add('hidden'), 3000);

  // Particle count dropdown
  const particleCountSelect = document.getElementById('particleCountSelect');
  // Disable options exceeding GPU limit
  for (const opt of particleCountSelect.options) {
    if (parseInt(opt.value) > maxParticles) {
      opt.disabled = true;
      opt.textContent += ' (GPU limit)';
    }
  }
  // Set initial selection to match state
  particleCountSelect.value = String(state.particleCount);
  particleCountSelect.addEventListener('change', () => {
    const count = parseInt(particleCountSelect.value);
    rebuildParticleSystem(count);
  });

  // Wire sliders to state
  function wireSlider(id, stateKey, fmt) {
    const slider = document.getElementById(id);
    const valSpan = document.getElementById(id + 'Val');
    if (!slider) return;
    slider.value = state[stateKey];
    if (valSpan) valSpan.textContent = fmt ? fmt(state[stateKey]) : state[stateKey];
    let lastTextUpdate = 0;
    slider.addEventListener('input', () => {
      state[stateKey] = parseFloat(slider.value);
      if (valSpan) {
        const now = performance.now();
        if (now - lastTextUpdate > 100) { // ~10Hz text updates
          valSpan.textContent = fmt ? fmt(state[stateKey]) : state[stateKey];
          lastTextUpdate = now;
        }
      }
    });
    // Ensure final value shown on release
    if (valSpan) {
      slider.addEventListener('change', () => {
        valSpan.textContent = fmt ? fmt(state[stateKey]) : state[stateKey];
      });
    }
  }

  function wireSelect(id, stateKey, onChange) {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.value = String(Math.round(state[stateKey] ?? 0));
    sel.addEventListener('change', () => {
      state[stateKey] = parseFloat(sel.value);
      if (onChange) onChange(state[stateKey]);
    });
  }

  wireSlider('particleSize', 'particleSize');
  wireSlider('sizeRandomness', 'sizeRandomness');
  wireSlider('glintBrightness', 'glintBrightness');
  wireSlider('prismaticAmount', 'prismaticAmount');

  // Wire color pickers
  function hexToRGB(hex) {
    return [
      parseInt(hex.slice(1, 3), 16) / 255,
      parseInt(hex.slice(3, 5), 16) / 255,
      parseInt(hex.slice(5, 7), 16) / 255,
    ];
  }
  function rgbToHex(rgb) {
    return '#' + rgb.map(c => Math.round(Math.min(1, Math.max(0, c)) * 255).toString(16).padStart(2, '0')).join('');
  }
  function wireColor(id, stateKey) {
    const picker = document.getElementById(id);
    picker.value = rgbToHex(state[stateKey]);
    picker.addEventListener('input', () => {
      state[stateKey] = hexToRGB(picker.value);
    });
  }
  wireColor('baseColor', 'baseColor');
  wireColor('accentColor', 'accentColor');
  wireColor('glitterColor', 'glitterColor');
  wireColor('glitterAccent', 'glitterAccent');
  wireColor('tipColor', 'tipColor');
  wireColor('glitterTip', 'glitterTip');
  wireSlider('colorBlend', 'colorBlend', v => v.toFixed(2));

  const NOISE_TYPE_PROFILES = [
    // Each noise type ships with a curated preset so mode changes are immediately visible.
    { amount: 0.20, behavior: 0, mapping: 0, warp: 0.20, sharpness: 0.20, anisotropy: 0.10, blend: 0.20, frequency: 0.50, speed: 0.45 }, // Classic Curl
    { amount: 0.28, behavior: 0, mapping: 0, warp: 0.95, sharpness: 0.35, anisotropy: 0.30, blend: 0.40, frequency: 0.48, speed: 0.52 }, // Domain-Warped
    { amount: 0.24, behavior: 2, mapping: 0, warp: 0.40, sharpness: 1.00, anisotropy: 0.35, blend: 0.35, frequency: 0.56, speed: 0.46 }, // Ridged
    { amount: 0.30, behavior: 0, mapping: 0, warp: 0.35, sharpness: 0.55, anisotropy: 0.20, blend: 1.00, frequency: 0.62, speed: 0.42 }, // Voronoi
    { amount: 0.26, behavior: 1, mapping: 0, warp: 0.55, sharpness: 0.35, anisotropy: 0.95, blend: 0.35, frequency: 0.52, speed: 0.58 }, // Flow
    { amount: 0.23, behavior: 3, mapping: 0, warp: 0.25, sharpness: 0.75, anisotropy: 1.00, blend: 0.50, frequency: 0.60, speed: 0.52 }, // Gabor-like
    { amount: 0.32, behavior: 0, mapping: 0, warp: 0.80, sharpness: 0.70, anisotropy: 0.65, blend: 0.75, frequency: 0.55, speed: 0.55 }, // Hybrid
    { amount: 0.42, behavior: 0, mapping: 1, warp: 0.88, sharpness: 0.62, anisotropy: 0.95, blend: 0.58, frequency: 0.53, speed: 0.60 }, // Jupiter
  ];

  function formatNoiseBehavior(v) {
    const n = Math.round(v);
    if (n <= 0) return '0 None';
    if (n === 1) return '1 Mirror';
    return `${n} ${n}-Fold`;
  }

  function applyNoiseTypeProfile(type, syncUI = false) {
    const p = NOISE_TYPE_PROFILES[Math.max(0, Math.min(NOISE_TYPE_PROFILES.length - 1, Math.round(type)))];
    if (!p) return;
    state.noiseAmount = p.amount;
    state.noiseBehavior = p.behavior;
    state.noiseMapping = p.mapping;
    state.noiseWarp = p.warp;
    state.noiseSharpness = p.sharpness;
    state.noiseAnisotropy = p.anisotropy;
    state.noiseBlend = p.blend;
    state.noiseFrequency = p.frequency;
    state.noiseSpeed = p.speed;
    if (syncUI) {
      syncSlider('noiseAmount', 'noiseAmount');
      syncSlider('noiseBehavior', 'noiseBehavior', formatNoiseBehavior);
      {
        const noiseMappingSel = document.getElementById('noiseMapping');
        if (noiseMappingSel) noiseMappingSel.value = String(Math.round(state.noiseMapping || 0));
      }
      syncSlider('noiseWarp', 'noiseWarp');
      syncSlider('noiseSharpness', 'noiseSharpness');
      syncSlider('noiseAnisotropy', 'noiseAnisotropy');
      syncSlider('noiseBlend', 'noiseBlend');
      syncSlider('noiseFrequency', 'noiseFrequency');
      syncSlider('noiseSpeed', 'noiseSpeed');
    }
  }

  wireSlider('sheenStrength', 'sheenStrength');
  wireSlider('metallic', 'metallic');
  wireSlider('roughness', 'roughness');
  wireColor('sheenColor', 'sheenColor');
  wireSlider('clickSize', 'clickSize');
  wireSlider('clickStrength', 'clickStrength');
  wireSelect('burstBehavior', 'burstBehavior');
  wireSlider('burstCount', 'burstCount', v => Math.round(v));
  wireSlider('burstForce', 'burstForce');
  wireSlider('burstForceRandomness', 'burstForceRandomness');
  wireSlider('burstSpeed', 'burstSpeed');
  wireSlider('burstDuration', 'burstDuration');
  wireSlider('burstWidth', 'burstWidth');
  wireSlider('noiseAmount', 'noiseAmount', v => v.toFixed(2));
  wireSelect('noiseType', 'noiseType', v => applyNoiseTypeProfile(v, true));
  wireSelect('noiseMapping', 'noiseMapping');
  wireSelect('faceEffectorMode', 'faceEffectorMode', v => {
    const mode = Math.max(0, Math.min(1, Math.round(v)));
    state.faceEffectorMode = mode;
    if (faceEffectorModeSelect) faceEffectorModeSelect.value = String(mode);
    if (mode > 0 && !faceTracking.enabled) {
      startFaceTracking();
    }
    if (mode <= 0 && state.faceDebugMode <= 0 && faceTracking.enabled) {
      stopFaceTracking(true);
      setFaceStatus('Face tracking idle');
    }
  });
  wireSelect('faceDebugMode', 'faceDebugMode', v => {
    state.faceDebugMode = Math.max(0, Math.min(2, Math.round(v)));
  });
  wireSlider('noiseBehavior', 'noiseBehavior', formatNoiseBehavior);
  wireSlider('noiseFrequency', 'noiseFrequency');
  wireSlider('noiseSpeed', 'noiseSpeed');
  wireSlider('noiseWarp', 'noiseWarp');
  wireSlider('noiseSharpness', 'noiseSharpness');
  wireSlider('noiseAnisotropy', 'noiseAnisotropy');
  wireSlider('noiseBlend', 'noiseBlend');
  applyNoiseTypeProfile(state.noiseType, true);
  wireSlider('curlStrength', 'curlStrength', v => Math.round(v));
  wireSlider('splatForce', 'splatForce', v => Math.round(v));
  wireSlider('velDissipation', 'velDissipation', v => v.toFixed(3));
  wireSlider('dyeDissipation', 'dyeDissipation', v => v.toFixed(3));
  wireSlider('pressureIters', 'pressureIters', v => Math.round(v));
  wireSlider('pressureDecay', 'pressureDecay', v => v.toFixed(2));
  wireSlider('simSpeed', 'simSpeed');
  wireSlider('masterSpeed', 'masterSpeed', v => v.toFixed(2));
  wireSlider('noiseDyeIntensity', 'noiseDyeIntensity');
  wireSlider('dyeNoiseAmount', 'dyeNoiseAmount');
  // Temp
  wireSlider('tempAmount', 'tempAmount');
  wireSlider('tempBuoyancy', 'tempBuoyancy');
  wireSlider('tempDissipation', 'tempDissipation', v => v.toFixed(3));
  wireSlider('tempDyeTint', 'tempDyeTint');
  // Mood / Palette
  wireSlider('moodAmount', 'moodAmount');
  wireSlider('moodSpeed', 'moodSpeed');
  // Palette slider with special handling
  (() => {
    const slider = document.getElementById('paletteIndex');
    const valSpan = document.getElementById('paletteIndexVal');
    if (!slider) return;
    slider.addEventListener('input', () => {
      const idx = parseInt(slider.value);
      state.paletteIndex = idx;
      if (valSpan) valSpan.textContent = idx < 0 ? 'Manual' : PALETTES[idx]?.name || idx;
      if (idx >= 0) { applyPalette(idx); syncAllUI(); }
    });
  })();
  wireSlider('bloomIntensity', 'bloomIntensity', v => v.toFixed(2));
  wireSlider('bloomThreshold', 'bloomThreshold', v => v.toFixed(2));
  wireSlider('bloomRadius', 'bloomRadius', v => v.toFixed(2));
  if (faceTrackingToggleBtn) {
    faceTrackingToggleBtn.addEventListener('click', () => {
      if (faceTracking.initializing) {
        return;
      }
      if (faceTracking.enabled) {
        stopFaceTracking();
      } else {
        startFaceTracking();
      }
    });
  }
  syncFaceTrackingToggleButton();
  setFaceStatus('Face tracking idle');
  window.addEventListener('beforeunload', () => {
    if (faceTracking.enabled) stopFaceTracking();
  });

  // ─── Sync UI from state (for randomize) ─────────────────────────────
  function syncSlider(id, stateKey, fmt) {
    const slider = document.getElementById(id);
    const valSpan = document.getElementById(id + 'Val');
    if (slider) slider.value = state[stateKey];
    if (valSpan) valSpan.textContent = fmt ? fmt(state[stateKey]) : state[stateKey];
  }
  function syncColor(id, stateKey) {
    const picker = document.getElementById(id);
    if (picker) picker.value = rgbToHex(state[stateKey]);
  }
  function syncAllUI() {
    syncSlider('particleSize', 'particleSize');
    syncSlider('sizeRandomness', 'sizeRandomness');
    syncSlider('glintBrightness', 'glintBrightness');
    syncSlider('prismaticAmount', 'prismaticAmount');
    syncSlider('colorBlend', 'colorBlend', v => v.toFixed(2));
    syncSlider('sheenStrength', 'sheenStrength');
    syncSlider('metallic', 'metallic');
    syncSlider('roughness', 'roughness');
    syncSlider('clickSize', 'clickSize');
    syncSlider('clickStrength', 'clickStrength');
    {
      const burstBehaviorSel = document.getElementById('burstBehavior');
      if (burstBehaviorSel) burstBehaviorSel.value = String(Math.round(state.burstBehavior || 0));
    }
    syncSlider('burstCount', 'burstCount', v => Math.round(v));
    syncSlider('burstForce', 'burstForce');
    syncSlider('burstForceRandomness', 'burstForceRandomness');
    syncSlider('burstSpeed', 'burstSpeed');
    syncSlider('burstDuration', 'burstDuration');
    syncSlider('burstWidth', 'burstWidth');
    syncSlider('noiseAmount', 'noiseAmount', v => v.toFixed(2));
    syncSlider('noiseBehavior', 'noiseBehavior', formatNoiseBehavior);
    syncSlider('noiseFrequency', 'noiseFrequency');
    syncSlider('noiseSpeed', 'noiseSpeed');
    syncSlider('noiseWarp', 'noiseWarp');
    syncSlider('noiseSharpness', 'noiseSharpness');
    syncSlider('noiseAnisotropy', 'noiseAnisotropy');
    syncSlider('noiseBlend', 'noiseBlend');
    {
      const noiseTypeSel = document.getElementById('noiseType');
      if (noiseTypeSel) noiseTypeSel.value = String(Math.round(state.noiseType || 0));
    }
    {
      const noiseMappingSel = document.getElementById('noiseMapping');
      if (noiseMappingSel) noiseMappingSel.value = String(Math.round(state.noiseMapping || 0));
    }
    {
      const faceModeSel = document.getElementById('faceEffectorMode');
      const mode = Math.max(0, Math.min(1, Math.round(state.faceEffectorMode || 0)));
      state.faceEffectorMode = mode;
      if (faceModeSel) faceModeSel.value = String(mode);
    }
    {
      const faceDebugSel = document.getElementById('faceDebugMode');
      if (faceDebugSel) faceDebugSel.value = String(Math.round(state.faceDebugMode || 0));
    }
    syncSlider('curlStrength', 'curlStrength', v => Math.round(v));
    syncSlider('splatForce', 'splatForce', v => Math.round(v));
    syncSlider('velDissipation', 'velDissipation', v => v.toFixed(3));
    syncSlider('dyeDissipation', 'dyeDissipation', v => v.toFixed(3));
    syncSlider('pressureIters', 'pressureIters', v => Math.round(v));
    syncSlider('pressureDecay', 'pressureDecay', v => v.toFixed(2));
    syncSlider('noiseDyeIntensity', 'noiseDyeIntensity');
    syncSlider('dyeNoiseAmount', 'dyeNoiseAmount');
    // Temp
    syncSlider('tempAmount', 'tempAmount');
    syncSlider('tempBuoyancy', 'tempBuoyancy');
    syncSlider('tempDissipation', 'tempDissipation', v => v.toFixed(3));
    syncSlider('tempDyeTint', 'tempDyeTint');
    // Mood
    syncSlider('moodAmount', 'moodAmount');
    syncSlider('moodSpeed', 'moodSpeed');
    // Palette
    {
      const palSlider = document.getElementById('paletteIndex');
      const palVal = document.getElementById('paletteIndexVal');
      if (palSlider) palSlider.value = state.paletteIndex;
      if (palVal) palVal.textContent = state.paletteIndex < 0 ? 'Manual' : (PALETTES[state.paletteIndex]?.name || state.paletteIndex);
    }
    syncColor('baseColor', 'baseColor');
    syncColor('accentColor', 'accentColor');
    syncColor('glitterColor', 'glitterColor');
    syncColor('glitterAccent', 'glitterAccent');
    syncColor('tipColor', 'tipColor');
    syncColor('glitterTip', 'glitterTip');
    syncColor('sheenColor', 'sheenColor');
    syncSlider('simSpeed', 'simSpeed');
    syncSlider('masterSpeed', 'masterSpeed', v => v.toFixed(2));
    syncSlider('bloomIntensity', 'bloomIntensity', v => v.toFixed(2));
    syncSlider('bloomThreshold', 'bloomThreshold', v => v.toFixed(2));
    syncSlider('bloomRadius', 'bloomRadius', v => v.toFixed(2));
  }

  // ─── Randomize helpers ────────────────────────────────────────────────
  function randRange(lo, hi) { return lo + Math.random() * (hi - lo); }
  const hdrColorMax = hdrSupported ? 1.2 : 1.0;
  function randColor() { return [Math.random() * hdrColorMax, Math.random() * hdrColorMax, Math.random() * hdrColorMax]; }
  function snapTo(val, step) { return Math.round(val / step) * step; }

  document.getElementById('randomizeColors').addEventListener('click', () => {
    state.baseColor = randColor();
    state.accentColor = randColor();
    state.glitterColor = randColor();
    state.glitterAccent = randColor();
    state.tipColor = randColor();
    state.glitterTip = randColor();
    state.sheenColor = [Math.random(), Math.random(), Math.random()];
    state.colorBlend = randRange(0, 1);
    state.prismaticAmount = randRange(0, 20);
    syncAllUI();
  });

  document.getElementById('randomizeParams').addEventListener('click', () => {
    // Sim
    state.simSpeed = snapTo(randRange(0.3, 2.0), 0.01);
    // Particle appearance
    state.sizeRandomness = snapTo(randRange(0, 1), 0.01);
    state.sheenStrength = snapTo(randRange(0, 1), 0.01);
    state.metallic = Math.random();
    state.roughness = Math.random();
    // Interaction
    state.clickSize = snapTo(randRange(0, 1), 0.01);
    state.clickStrength = snapTo(randRange(0, 1), 0.01);
    // Burst emitters
    state.burstBehavior = Math.round(randRange(0, 4));
    state.burstCount = Math.round(randRange(0, 8));
    state.burstForce = snapTo(randRange(0, 3.2), 0.01);
    state.burstForceRandomness = snapTo(randRange(0, 1), 0.01);
    state.burstSpeed = snapTo(randRange(0, 2.0), 0.01);
    state.burstDuration = snapTo(randRange(0.1, 4.0), 0.01);
    state.burstWidth = snapTo(randRange(0.05, 0.9), 0.01);
    // Noise
    state.noiseAmount = snapTo(randRange(0, 1), 0.01);
    state.noiseType = Math.round(randRange(0, 7));
    applyNoiseTypeProfile(state.noiseType);
    state.noiseMapping = Math.random() < 0.35 ? 1 : 0;
    state.noiseBehavior = Math.random() < 0.45 ? 0 : (Math.random() < 0.7 ? 1 : Math.round(randRange(2, 8)));
    state.noiseWarp = snapTo(randRange(0, 1), 0.01);
    state.noiseSharpness = snapTo(randRange(0, 1), 0.01);
    state.noiseAnisotropy = snapTo(randRange(0, 1), 0.01);
    state.noiseBlend = snapTo(randRange(0, 1), 0.01);
    state.noiseFrequency = snapTo(randRange(0, 1), 0.01);
    state.noiseSpeed = snapTo(randRange(0, 1), 0.01);
    // Fluid sim
    state.curlStrength = Math.round(randRange(0, 50));
    state.splatForce = snapTo(randRange(1000, 20000), 100);
    state.velDissipation = snapTo(randRange(0.99, 1.0), 0.001);
    state.dyeDissipation = snapTo(randRange(0.98, 1.0), 0.001);
    state.pressureIters = Math.round(randRange(10, 60));
    state.pressureDecay = snapTo(randRange(0, 1), 0.01);
    // Dye noise
    state.noiseDyeIntensity = Math.random();
    state.dyeNoiseAmount = Math.random() * 0.15;
    // Temp
    state.tempAmount = Math.random();
    state.tempBuoyancy = Math.random();
    state.tempDissipation = 0.95 + Math.random() * 0.05;
    state.tempDyeTint = Math.random();
    // Mood
    state.moodAmount = Math.random();
    state.moodSpeed = Math.random();
    state.paletteIndex = Math.floor(Math.random() * 51) - 1; // -1 to 49
    if (state.paletteIndex >= 0) applyPalette(state.paletteIndex);
    // NOTE: particleCount is intentionally NOT randomized
    syncAllUI();
  });

  // ─── Auto-Morph ──────────────────────────────────────────────────────
  const morphSliders = {
    simSpeed: { min: 0, max: 3, step: 0.01 },
    sizeRandomness: { min: 0, max: 1, step: 0.01 },
    colorBlend: { min: 0, max: 1, step: 0.01 },
    sheenStrength: { min: 0, max: 1, step: 0.01 },
    metallic: { min: 0, max: 1, step: 0.01 },
    roughness: { min: 0, max: 1, step: 0.01 },
    clickSize: { min: 0, max: 1, step: 0.01 },
    clickStrength: { min: 0, max: 1, step: 0.01 },
    burstBehavior: { min: 0, max: 4, step: 1 },
    burstCount: { min: 0, max: 8, step: 1 },
    burstForce: { min: 0, max: 4, step: 0.01 },
    burstForceRandomness: { min: 0, max: 1, step: 0.01 },
    burstSpeed: { min: 0, max: 2.0, step: 0.01 },
    burstDuration: { min: 0.05, max: 4.0, step: 0.01 },
    burstWidth: { min: 0, max: 1, step: 0.01 },
    noiseAmount: { min: 0, max: 1, step: 0.01 },
    noiseMapping: { min: 0, max: 1, step: 1 },
    noiseFrequency: { min: 0, max: 1, step: 0.01 },
    noiseSpeed: { min: 0, max: 1, step: 0.01 },
    noiseWarp: { min: 0, max: 1, step: 0.01 },
    noiseSharpness: { min: 0, max: 1, step: 0.01 },
    noiseAnisotropy: { min: 0, max: 1, step: 0.01 },
    noiseBlend: { min: 0, max: 1, step: 0.01 },
    curlStrength: { min: 0, max: 50, step: 1 },
    splatForce: { min: 1000, max: 20000, step: 100 },
    velDissipation: { min: 0.99, max: 1.0, step: 0.001 },
    dyeDissipation: { min: 0.98, max: 1.0, step: 0.001 },
    pressureIters: { min: 10, max: 60, step: 1 },
    pressureDecay: { min: 0, max: 1, step: 0.01 },
    noiseDyeIntensity: { min: 0, max: 1, step: 0.01 },
    dyeNoiseAmount: { min: 0, max: 0.15, step: 0.001 },
    tempAmount: { min: 0, max: 1, step: 0.01 },
    tempBuoyancy: { min: 0, max: 1, step: 0.01 },
    tempDissipation: { min: 0.95, max: 1.0, step: 0.001 },
    tempDyeTint: { min: 0, max: 1, step: 0.01 },
    moodAmount: { min: 0, max: 1, step: 0.01 },
    moodSpeed: { min: 0, max: 1, step: 0.01 },
    paletteIndex: { min: -1, max: 49, step: 1 },
  };
  const morphColors = ['baseColor', 'accentColor', 'tipColor', 'glitterColor', 'glitterAccent', 'glitterTip', 'sheenColor'];

  const morphTargets = {};
  function pickMorphTarget(key) {
    if (morphColors.includes(key)) {
      const cMax = (key === 'sheenColor') ? 1.0 : hdrColorMax;
      morphTargets[key] = [Math.random() * cMax, Math.random() * cMax, Math.random() * cMax];
    } else {
      const s = morphSliders[key];
      morphTargets[key] = s.min + Math.random() * (s.max - s.min);
    }
  }
  // Initialize all targets
  for (const key of Object.keys(morphSliders)) pickMorphTarget(key);
  for (const key of morphColors) pickMorphTarget(key);

  let lastMorphSync = 0;
  let boMorphCooldown = 0;
  function updateAutoMorph() {
    if (!state.autoMorph) return;
    const rate = 0.002;
    const threshold = 0.005;

    // BO-guided morph: replace random targets with GP suggestions
    if (bo.boMorphMode && (bo.motionModel || bo.colorModel)) {
      boMorphCooldown--;
      // Check if all slider targets are reached
      let allReached = true;
      for (const [key, s] of Object.entries(morphSliders)) {
        const range = s.max - s.min;
        if (Math.abs(state[key] - morphTargets[key]) / range >= threshold) {
          allReached = false;
          break;
        }
      }
      if (allReached || boMorphCooldown <= 0) {
        boMorphCooldown = 300;
        const x = bo.getMorphTarget();
        // Decode normalized vector into morph targets (sliders + colors)
        for (let i = 0; i < SLIDER_KEYS.length; i++) {
          const key = SLIDER_KEYS[i];
          const cm = key.match(/^(.+)_([012])$/);
          if (cm && morphColors.includes(cm[1])) {
            // Color channel — update parent color's morph target
            if (!Array.isArray(morphTargets[cm[1]])) morphTargets[cm[1]] = [...state[cm[1]]];
            const s = SLIDER_SPACE[key];
            morphTargets[cm[1]][parseInt(cm[2])] = s.min + x[i] * (s.max - s.min);
          } else if (morphSliders[key]) {
            const s = SLIDER_SPACE[key];
            morphTargets[key] = s.min + x[i] * (s.max - s.min);
          }
        }
      }
    }

    // Lerp sliders
    for (const [key, s] of Object.entries(morphSliders)) {
      const target = morphTargets[key];
      const range = s.max - s.min;
      state[key] += (target - state[key]) * rate;
      if (Math.abs(state[key] - target) / range < threshold) {
        if (!(bo.boMorphMode && (bo.motionModel || bo.colorModel))) {
          pickMorphTarget(key);
        }
      }
    }

    // Lerp colors
    for (const key of morphColors) {
      const target = morphTargets[key];
      for (let i = 0; i < 3; i++) {
        state[key][i] += (target[i] - state[key][i]) * rate;
      }
      const dist = Math.abs(state[key][0] - target[0]) + Math.abs(state[key][1] - target[1]) + Math.abs(state[key][2] - target[2]);
      if (dist < threshold * 3) {
        if (bo.boMorphMode && (bo.motionModel || bo.colorModel)) {
          const cMax = (key === 'sheenColor') ? 1.0 : hdrColorMax;
          morphTargets[key] = [Math.random() * cMax, Math.random() * cMax, Math.random() * cMax];
        } else {
          pickMorphTarget(key);
        }
      }
    }

    // Sync UI at ~10Hz
    const now = performance.now();
    if (now - lastMorphSync > 100) {
      syncAllUI();
      lastMorphSync = now;
    }
  }

  // Auto-Morph button
  const autoMorphBtn = document.getElementById('autoMorph');
  autoMorphBtn.addEventListener('click', () => {
    state.autoMorph = !state.autoMorph;
    autoMorphBtn.classList.toggle('active', state.autoMorph);
  });

  // ─── Bayesian Optimization UI ────────────────────────────────────────
  const boOverlay = document.getElementById('boOverlay');
  const boRateBtn = document.getElementById('boRate');
  const boMorphBtn = document.getElementById('boMorph');
  const boBestBtn = document.getElementById('boBest');
  const boClearBtn = document.getElementById('boClear');

  function toggleRateMode() {
    bo.rateMode = !bo.rateMode;
    boRateBtn.classList.toggle('active', bo.rateMode);
    boOverlay.style.display = bo.rateMode ? 'block' : 'none';
    if (bo.rateMode) {
      // Reset pending ratings and flash for fresh generation
      bo._pendingMovement = null;
      bo._pendingColor = null;
      bo._resetFlash();
      bo.updateOverlay();
    }
  }

  boRateBtn.addEventListener('click', toggleRateMode);

  boMorphBtn.addEventListener('click', () => {
    bo.boMorphMode = !bo.boMorphMode;
    boMorphBtn.classList.toggle('active', bo.boMorphMode);
    if (bo.boMorphMode) {
      state.autoMorph = true;
      autoMorphBtn.classList.add('active');
    }
  });

  boBestBtn.addEventListener('click', () => {
    const x = bo.getBestParams();
    normalizedToState(x, state, bo.lockedKeys);
    syncAllUI();
  });

  boClearBtn.addEventListener('click', () => {
    if (confirm('Clear all BO ratings?')) {
      bo.clearData();
    }
  });

  // ─── Run Management UI ──────────────────────────────────────────
  document.getElementById('saveRun').addEventListener('click', async () => {
    const name = prompt('Run name (letters, numbers, - _ only):');
    if (!name) return;
    const ok = await bo.saveRun(name);
    if (!ok) alert('Save failed — check name or rate some configs first.');
  });

  document.getElementById('loadRun').addEventListener('click', async () => {
    const runs = await bo.listRuns();
    if (!runs.length) { alert('No saved runs.'); return; }
    const name = prompt('Available runs:\n' + runs.join('\n') + '\n\nEnter name to load:');
    if (!name) return;
    const ok = await bo.loadRun(name);
    if (ok) syncAllUI(); else alert('Load failed — run not found.');
  });

  document.getElementById('deleteRun').addEventListener('click', async () => {
    const runs = await bo.listRuns();
    if (!runs.length) { alert('No saved runs.'); return; }
    const name = prompt('Available runs:\n' + runs.join('\n') + '\n\nEnter name to delete:');
    if (!name) return;
    if (!confirm(`Delete run "${name}"?`)) return;
    await bo.deleteRun(name);
  });

  // ─── Example Management UI ─────────────────────────────────────────
  document.getElementById('saveExample').addEventListener('click', async () => {
    const name = prompt('Example name (letters, numbers, - _ only):');
    if (!name) return;
    await bo.saveExample(name, state);
    console.log(`Example "${name}" saved`);
  });

  document.getElementById('loadExample').addEventListener('click', async () => {
    const examples = await bo.listExamples();
    if (!examples.length) { alert('No saved examples.'); return; }
    const name = prompt('Available examples:\n' + examples.map(e => e.name).join('\n') + '\n\nEnter name to load:');
    if (!name) return;
    const ex = examples.find(e => e.name === name);
    if (!ex) { alert('Example not found.'); return; }
    bo.loadExample(ex, state, syncAllUI);
  });

  document.getElementById('deleteExample').addEventListener('click', async () => {
    const examples = await bo.listExamples();
    if (!examples.length) { alert('No saved examples.'); return; }
    const name = prompt('Available examples:\n' + examples.map(e => e.name).join('\n') + '\n\nEnter name to delete:');
    if (!name) return;
    if (!confirm(`Delete example "${name}"?`)) return;
    await bo.deleteExample(name);
  });

  // ─── Preset Browser ─────────────────────────────────────────────────
  let currentPresetIdx = -1;
  const presetNameEl = document.getElementById('presetName');
  document.getElementById('presetPrev')?.addEventListener('click', () => {
    if (bo.examples.length === 0) return;
    currentPresetIdx = (currentPresetIdx - 1 + bo.examples.length) % bo.examples.length;
    bo.loadExample(bo.examples[currentPresetIdx], state, syncAllUI);
    if (presetNameEl) presetNameEl.textContent = bo.examples[currentPresetIdx].name;
  });
  document.getElementById('presetNext')?.addEventListener('click', () => {
    if (bo.examples.length === 0) return;
    currentPresetIdx = (currentPresetIdx + 1) % bo.examples.length;
    bo.loadExample(bo.examples[currentPresetIdx], state, syncAllUI);
    if (presetNameEl) presetNameEl.textContent = bo.examples[currentPresetIdx].name;
  });

  // Keyboard handler
  document.addEventListener('keydown', (e) => {
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT') return;

    if (e.key === 'r' || e.key === 'R') {
      toggleRateMode();
      return;
    }

    if (bo.rateMode) {
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        bo.rateMovement(1, state, syncAllUI);
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        bo.rateMovement(-1, state, syncAllUI);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        bo.rateColor(1, state, syncAllUI);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        bo.rateColor(-1, state, syncAllUI);
      } else if (e.key === 'l' || e.key === 'L') {
        bo.toggleLockColors();
      } else if (e.key === 'm' || e.key === 'M') {
        bo.toggleLockMotion();
      }
    }
  });

  loadStatus.textContent = 'Starting simulation...';
  // Ensure we have a non-zero drawable size before first frame submission.
  resize();
  if (canvas.width < 2 || canvas.height < 2) {
    loadStatus.textContent = 'Waiting for drawable surface...';
    for (let i = 0; i < 120 && (canvas.width < 2 || canvas.height < 2); i++) {
      await new Promise(resolve => requestAnimationFrame(resolve));
      resize();
    }
    if (canvas.width < 2 || canvas.height < 2) {
      console.warn(`Init: drawable surface still tiny (${canvas.width}x${canvas.height}); continuing with clamped size.`);
    }
  }
  console.log(`Fluid simulation starting... (${(performance.now() - t0).toFixed(0)}ms)`);
  document.getElementById('loading').style.display = 'none';
  if (state.faceEffectorMode > 0) {
    startFaceTracking();
  }
  requestAnimationFrame(frame);

  // Trigger deferred BO retrain now that WebGPU is fully initialized
  if (bo.ratings.length >= 10 && !bo.motionModel && !bo.colorModel) {
    setTimeout(() => {
      try { bo.retrain(); } catch (e) { console.warn('BO: deferred retrain failed:', e); }
    }, 100);
  }
}

main().catch(err => {
  console.error('Fatal:', err);
  const errorDiv = document.getElementById('error');
  if (errorDiv) {
    errorDiv.style.display = 'block';
    errorDiv.textContent = err.message;
  }
});
