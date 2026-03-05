// ─── Configuration ───────────────────────────────────────────────────────────
const SIM_RES = 512;
const WORKGROUP = 8;
const TEX_FMT = 'rgba16float';
const PARTICLE_STRIDE = 32;       // 8 floats × 4 bytes
const PARTICLE_WG = 256;

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
  clickSize: 0.5,
  clickStrength: 0.5,
  injectorIntensity: 1.0,
  noiseAmount: 0.0,
  noiseFrequency: 0.5,
  noiseSpeed: 0.5,
  injectorSize: 0.5,
  injectorCount: 4,
  injectorSpeed: 0.5,
  burstCount: 0,
  sheenColor: [1.0, 0.9, 0.7],
  rimIntensity: 0.5,
  chromaticStrength: 1.0,
  causticIntensity: 0.2,
  splatForce: 6000,
  curlStrength: 15,
  pressureIters: 30,
  pressureDecay: 0.8,
  velDissipation: 0.998,
  dyeDissipation: 0.993,
  drawBotCount: 0,
  drawBotSpeed: 0.5,
  drawBotSize: 3.0,
  drawBotChaos: 0.3,
  drawBotDrift: 0.5,
  drawBotTurnRate: 0.5,
  drawBotSpeedVar: 0.5,
  drawBotRecMix: 0.5,
  // Blobs — large Brownian-motion agents
  blobCount: 0,
  blobSpeed: 0.3,
  blobSize: 0.7,
  blobWander: 0.5,
  // Flockers — small boids-style swarm
  flockCount: 0,
  flockSpeed: 0.5,
  flockSize: 0.3,
  flockSeparation: 0.6,
  flockAlignment: 0.4,
  flockCohesion: 0.5,
  flockBlobReact: 0.3,
  // Dye-coupled noise
  noiseDyeIntensity: 0.0,
  dyeNoiseAmount: 0.0,
  bloomIntensity: 0,
  bloomThreshold: 0.4,
  bloomRadius: 0.5,
  splatRadius: 0.0015,
  simSpeed: 1.0,
  autoMorph: false,
};

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
const SPHERE_RADIUS: f32 = 0.41;
`;

const MAX_SPLATS = 64;

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
  if (dist > SPHERE_RADIUS - 0.02) {
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
  let newVel = vel + force * p.dt;
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

  // Sphere boundary — hard kill matches particle visibility (0.42)
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

  // Hard kill dye outside sphere — matches particle visibility
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
  pad1: f32,
  pad2: f32,
  pad3: f32,
};

@group(0) @binding(0) var<uniform> np: NoiseParams;
@group(0) @binding(1) var velSrc: texture_2d<f32>;
@group(0) @binding(2) var velDst: texture_storage_2d<rgba16float, write>;

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

const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS = 0.41;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(np.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / np.simRes;

  let dist = length(uv - SPHERE_CENTER);
  if (dist > SPHERE_RADIUS - 0.03) {
    textureStore(velDst, id.xy, textureLoad(velSrc, id.xy, 0));
    return;
  }
  // Fade noise to zero near sphere edge to prevent dye bleed
  let edgeFade = smoothstep(SPHERE_RADIUS - 0.03, SPHERE_RADIUS - 0.08, dist);

  var vel = textureLoad(velSrc, id.xy, 0).xy;

  let baseScale = 2.0 + np.frequency * 14.0;
  let ts = np.time * (0.1 + np.speed * 2.0);
  var curl = vec2f(0.0);
  let eps = 1.0 / np.simRes;

  // Octave 1 — large swirls
  let s1 = baseScale;
  let p1 = uv * s1 + vec2f(ts * 0.3, ts * 0.2);
  let n1c = vnoise(p1);
  let n1x = vnoise(p1 + vec2f(eps * s1, 0.0));
  let n1y = vnoise(p1 + vec2f(0.0, eps * s1));
  curl += vec2f(n1y - n1c, -(n1x - n1c)) / eps * 0.5;

  // Octave 2 — medium detail
  let s2 = baseScale * 2.0;
  let p2 = uv * s2 + vec2f(-ts * 0.5, ts * 0.4);
  let n2c = vnoise(p2);
  let n2x = vnoise(p2 + vec2f(eps * s2, 0.0));
  let n2y = vnoise(p2 + vec2f(0.0, eps * s2));
  curl += vec2f(n2y - n2c, -(n2x - n2c)) / eps * 0.3;

  // Octave 3 — fine detail
  let s3 = baseScale * 4.0;
  let p3 = uv * s3 + vec2f(ts * 0.7, -ts * 0.3);
  let n3c = vnoise(p3);
  let n3x = vnoise(p3 + vec2f(eps * s3, 0.0));
  let n3y = vnoise(p3 + vec2f(0.0, eps * s3));
  curl += vec2f(n3y - n3c, -(n3x - n3c)) / eps * 0.2;

  vel += curl * np.amount * 8.0 * edgeFade;
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
  color: vec4f,
};

@group(0) @binding(0) var<uniform> dp: DyeNoiseParams;
@group(0) @binding(1) var velSrc: texture_2d<f32>;
@group(0) @binding(2) var dyeSrc: texture_2d<f32>;
@group(0) @binding(3) var dyeDst: texture_storage_2d<rgba16float, write>;

const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS = 0.41;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(dp.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / dp.simRes;

  var dye = textureLoad(dyeSrc, id.xy, 0);

  // Fade out near sphere edge to prevent bleed
  let dist = length(uv - SPHERE_CENTER);
  if (dist > SPHERE_RADIUS) {
    textureStore(dyeDst, id.xy, dye);
    return;
  }
  let edgeFade = smoothstep(SPHERE_RADIUS, SPHERE_RADIUS - 0.06, dist);

  // Compute velocity divergence (negative = convergent flow = density buildup)
  let vC = textureLoad(velSrc, id.xy, 0).xy;
  let vR = textureLoad(velSrc, vec2u(min(id.x + 1u, res - 1u), id.y), 0).xy;
  let vL = textureLoad(velSrc, vec2u(max(id.x, 1u) - 1u, id.y), 0).xy;
  let vU = textureLoad(velSrc, vec2u(id.x, min(id.y + 1u, res - 1u)), 0).xy;
  let vD = textureLoad(velSrc, vec2u(id.x, max(id.y, 1u) - 1u), 0).xy;
  let div = (vR.x - vL.x + vU.y - vD.y) * 0.5;

  // Don't inject into already-bright areas — preserves contrast
  let existingBrightness = max(dye.r, max(dye.g, dye.b));
  let headroom = max(1.0 - existingBrightness, 0.0);
  if (headroom < 0.05) {
    textureStore(dyeDst, id.xy, dye);
    return;
  }

  // Color varies spatially using velocity direction + position
  let velAngle = atan2(vC.y, vC.x);
  let colorPhase = velAngle * 0.5 + dp.time * 0.05 + dot(uv, vec2f(5.0, 3.0));
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
  let vel = textureSampleLevel(velTex, samp, vec2f(part.posX, part.posY), 0.0).xy;

  // ── Glass marble velocity fix (commented out) ─────────────────────────
  // var vel = ... (use var instead of let above to enable this)
  // let pToC = vec2f(part.posX, part.posY) - SPHERE_CENTER;
  // let pDst = length(pToC);
  // if (pDst > SPHERE_RADIUS - 0.08) {
  //   let pN = pToC / max(pDst, 0.001);
  //   let radial = dot(vel, pN);
  //   if (radial < 0.0) {
  //     let prox = smoothstep(SPHERE_RADIUS - 0.08, SPHERE_RADIUS, pDst);
  //     vel -= pN * radial * prox;
  //   }
  // }
  // ─────────────────────────────────────────────────────────────────────

  // Advect particle with fluid
  let dt = p.dt;
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
  let dist = length(toCenter);

  if (dist > SPHERE_RADIUS || part.life <= 0.0) {
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
  if (length(centered) > 0.42) {
    colors[idx] = vec4f(0.0);
    return;
  }

  // ── Glass marble dye warp + rimOverride (commented out) ───────────────
  // let pNorm = centered / 0.43;
  // let pR = min(length(pNorm), 0.999);
  // let pZ = sqrt(max(1.0 - pR * pR, 0.0));
  // let warpedPUV = particleUV - centered * (1.0 - pZ) * 0.3;
  // let dye = textureSampleLevel(dyeTex, samp, warpedPUV, 0.0).rgb;
  // let rimOverride = smoothstep(0.28, 0.43, pDist);
  // let fluidGate = max(smoothstep(0.0, 0.15, intensity), rimOverride);
  // ─────────────────────────────────────────────────────────────────────

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

  // ── Glass marble inverse warp (commented out) ─────────────────────────
  // let pCen = vec2f(part.posX, part.posY) - vec2f(0.5, 0.5);
  // let pDistN = min(length(pCen) / 0.43, 0.999);
  // let pZV = sqrt(max(1.0 - pDistN * pDistN, 0.0));
  // var warpedXY = vec2f(part.posX, part.posY) + pCen * (1.0 - pZV) * 0.3;
  // let wCen = warpedXY - vec2f(0.5, 0.5);
  // let wDist = length(wCen);
  // if (wDist > 0.43) { warpedXY = vec2f(0.5, 0.5) + wCen / wDist * 0.43; }
  // let rawClip = warpedXY * 2.0 - 1.0;
  // ─────────────────────────────────────────────────────────────────────

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
  if (sphereDist > 0.42) { discard; }
  let sphereFade = 1.0 - smoothstep(0.40, 0.42, sphereDist);

  // Circular cutout + soft edge
  let d = length(in.localUV - vec2f(0.5));
  if (d > 0.5) { discard; }
  let edge = 1.0 - smoothstep(0.3, 0.5, d);
  return vec4f(in.color * edge * sphereFade, in.alpha * edge * sphereFade);
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
  return vec4f(scene + bloom * intensity, 1.0);
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
  sheenColor: vec4f,  // xyz=sheenColor RGB
  tipColor: vec4f,    // xyz=tipColor RGB (dense end)
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
  let screenRadius = 0.40;
  let mask = 1.0 - smoothstep(screenRadius - 0.003, screenRadius, screenDist);

  // ── Glass marble (commented out) ──────────────────────────────────────
  // let normPos = centered / screenRadius;
  // let r = min(length(normPos), 0.999);
  // let z = sqrt(max(1.0 - r * r, 0.0));
  // let distortAmount = 0.3;
  // let warpedUV = uv - centered * (1.0 - z) * distortAmount;
  // let refractDir = normPos * (1.0 - z);
  // let chromaticStrength = du.baseColor.w;
  // let iorR = 0.01 * chromaticStrength;
  // let iorG = 0.025 * chromaticStrength;
  // let iorB = 0.04 * chromaticStrength;
  // let uvR = warpedUV + refractDir * iorR;
  // let uvG = warpedUV + refractDir * iorG;
  // let uvB = warpedUV + refractDir * iorB;
  // let raw = vec3f(
  //   textureSampleLevel(dyeTex, samp, uvR, 0.0).r,
  //   textureSampleLevel(dyeTex, samp, uvG, 0.0).g,
  //   textureSampleLevel(dyeTex, samp, uvB, 0.0).b
  // );
  // ─────────────────────────────────────────────────────────────────────

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

  // ── Glass marble depth darkening (commented out) ──────────────────────
  // let pathLength = 2.0 * z;
  // let absorptionColor = vec3f(0.04, 0.06, 0.02);
  // color *= exp(-absorptionColor * pathLength * 2.0);
  // ─────────────────────────────────────────────────────────────────────

  // Surface gradient for multi-lobe metallic sheen
  let texel = vec2f(1.0 / 512.0);
  let iL = dot(textureSampleLevel(dyeTex, samp, uv - vec2f(texel.x * 2.0, 0.0), 0.0).rgb, vec3f(0.3, 0.6, 0.1));
  let iR = dot(textureSampleLevel(dyeTex, samp, uv + vec2f(texel.x * 2.0, 0.0), 0.0).rgb, vec3f(0.3, 0.6, 0.1));
  let iB = dot(textureSampleLevel(dyeTex, samp, uv - vec2f(0.0, texel.y * 2.0), 0.0).rgb, vec3f(0.3, 0.6, 0.1));
  let iT = dot(textureSampleLevel(dyeTex, samp, uv + vec2f(0.0, texel.y * 2.0), 0.0).rgb, vec3f(0.3, 0.6, 0.1));
  let grad = vec2f(iR - iL, iT - iB);
  let gradLen = length(grad);
  let sheenDir = normalize(vec2f(0.4, 0.6));
  let spec = max(dot(normalize(grad + vec2f(0.001)), sheenDir), 0.0);

  // Sharp specular highlight
  let spec2 = spec * spec;
  let spec4 = spec2 * spec2;
  let spec8 = spec4 * spec4;
  let sharpSheen = spec8 * smoothstep(0.003, 0.04, gradLen);
  // Broad soft metallic glow
  let broadSpec = spec2 * smoothstep(0.002, 0.08, gradLen);
  // Fresnel-like rim sheen (edges of sphere glow)
  let rimFactor = smoothstep(0.2, 0.42, screenDist);

  let headroom = du.baseColor.w;
  let sheen = (sharpSheen * 0.6 + broadSpec * 0.3 + rimFactor * 0.15) * sheenStrength;
  color += color * sheen * du.sheenColor.rgb * headroom;

  // Tone mapping
  color = ${hdr ? 'tonemap(color * 1.6)' : 'aces(color * 1.6)'};

${hdr ? '  // HDR: browser expects linear values, handles transfer function' : `  // Fast gamma approximation (max error ~0.003 vs pow(x, 0.4545))
  { let sq = sqrt(clamp(color, vec3f(0.0), vec3f(1.0))); color = sq * (0.585 + sq * 0.415); }`}

  // Circular mask
  color *= mask;

  return vec4f(color, 1.0);
}
`;
}

// ─── Glass Shell Fragment Shader (commented out) ────────────────────────────
// const glassShellFrag = /* wgsl */`
// struct GlassUniforms {
//   screen: vec4f,
//   params: vec4f,
// };
// @group(0) @binding(0) var<uniform> gu: GlassUniforms;
// @fragment fn main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
//   ... Fresnel rim, specular highlight, caustic ring, env reflection ...
//   See git history for full implementation.
// }
// `;

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

async function main() {
  const t0 = performance.now();
  const loadStatus = document.getElementById('loadStatus');
  console.log('Init: creating BO controller...');
  const bo = await BOController.create();
  console.log(`Init: BO controller ready (${(performance.now() - t0).toFixed(0)}ms)`);
  const canvas = document.getElementById('canvas');
  const errorDiv = document.getElementById('error');

  if (!navigator.gpu) {
    errorDiv.style.display = 'block';
    errorDiv.textContent = 'WebGPU not supported in this browser.';
    return;
  }

  const gpuT0 = performance.now();
  console.log('Init: requesting GPU adapter...');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  console.log(`Init: adapter acquired (${(performance.now() - gpuT0).toFixed(0)}ms)`);
  if (!adapter) {
    errorDiv.style.display = 'block';
    errorDiv.textContent = 'No WebGPU adapter found.';
    console.error('No WebGPU adapter');
    return;
  }

  console.log('Init: requesting GPU device...');
  const deviceT0 = performance.now();
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1024 * 1024 * 1024),
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1024 * 1024 * 1024),
    },
  });
  console.log(`Init: device acquired (${(performance.now() - deviceT0).toFixed(0)}ms, total GPU: ${(performance.now() - gpuT0).toFixed(0)}ms)`);
  device.lost.then(info => console.error('WebGPU device lost:', info.message));

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
    canvas.width = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientHeight * dpr);
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

  // displayUB: [width, height, time, sheenStrength, baseR, baseG, baseB, pad, accentR, accentG, accentB, colorBlend, sheenR, sheenG, sheenB, pad, tipR, tipG, tipB, pad]
  const displayUB = device.createBuffer({
    size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const displayUBData = new Float32Array(20);
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
    size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const noiseData = new Float32Array(8); // [time, amount, simRes, frequency, speed, pad, pad, pad]

  // Dye noise pipeline: injects dye where flow converges
  const dyeNoisePipe = buildPipeline(dyeNoiseShader, 'dyeNoise',
    ['uniform', 'texture', 'texture', 'storage']);

  const dyeNoiseBuf = device.createBuffer({
    size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const dyeNoiseData = new Float32Array(8); // [time, amount, simRes, pad, r, g, b, a]

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

  // ─── Glass Shell render pipeline (commented out) ───────────────────────
  // const glassUB = device.createBuffer({
  //   size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  // });
  // const glassUBData = new Float32Array(8);
  // const glassShellBGL = device.createBindGroupLayout({ ... });
  // const glassShellPipeline = device.createRenderPipeline({ ... });
  // See git history for full glass shell pipeline setup.

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

  const velTexs = [velA, velB];
  const dyeTexs = [dyeA, dyeB];
  const pressTexs = [pressA, pressB];

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
  // const glassShellBG = bg(glassShellBGL, [ubuf(glassUB)]); // glass marble

  let splatCount = 0;
  function addSplat(x, y, dx, dy, r, g, b, radius) {
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

  // ─── Recording Controller ─────────────────────────────────────────────
  const recording = { active: false, points: [], startTime: 0, recordings: [], cHeld: false, fingerprint: null };
  state.useRecordings = false;

  function analyzeRecordings(recs) {
    if (!recs.length) return null;
    const speeds = [], turnRates = [], radii = [];
    for (const rec of recs) {
      const pts = rec.points;
      for (let i = 1; i < pts.length; i++) {
        const dx = pts[i].x - pts[i - 1].x;
        const dy = pts[i].y - pts[i - 1].y;
        const dt = pts[i].t - pts[i - 1].t;
        if (dt < 0.001) continue;
        speeds.push(Math.sqrt(dx * dx + dy * dy) / dt);
        radii.push(Math.sqrt((pts[i].x - 0.5) ** 2 + (pts[i].y - 0.5) ** 2));
        if (i >= 2) {
          const pdx = pts[i - 1].x - pts[i - 2].x;
          const pdy = pts[i - 1].y - pts[i - 2].y;
          const a1 = Math.atan2(pdy, pdx);
          const a2 = Math.atan2(dy, dx);
          let da = a2 - a1;
          if (da > Math.PI) da -= 2 * Math.PI;
          if (da < -Math.PI) da += 2 * Math.PI;
          turnRates.push(Math.abs(da / dt));
        }
      }
    }
    if (!speeds.length) return null;
    const sorted = a => [...a].sort((x, y) => x - y);
    const median = a => { const s = sorted(a); return s[Math.floor(s.length / 2)]; };
    const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
    const stddev = a => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
    const fp = {
      avgSpeed: median(speeds),
      speedVar: stddev(speeds) / (median(speeds) || 1),
      avgTurnRate: turnRates.length ? median(turnRates) : 0,
      turnRateVar: turnRates.length ? stddev(turnRates) / (median(turnRates) || 1) : 0,
      avgRadius: median(radii),
    };
    console.log('Recording fingerprint:', fp);
    return fp;
  }

  function seedStateFromFingerprint(fp) {
    if (!fp) return;
    // Map fingerprint to procedural params (normalized 0-1)
    state.drawBotSpeed = Math.min(1, fp.avgSpeed / 1.5);        // ~1.5 units/s = fast
    state.drawBotChaos = Math.min(1, fp.turnRateVar / 3.0);     // high variance = chaotic
    state.drawBotDrift = Math.min(1, fp.avgRadius / 0.35);      // 0.35 = sphere edge
    state.drawBotTurnRate = Math.min(1, fp.avgTurnRate / 15.0);  // ~15 rad/s = sharp turns
    state.drawBotSpeedVar = Math.min(1, fp.speedVar / 2.0);     // coefficient of variation
    console.log('Seeded bot params from recordings:', {
      speed: state.drawBotSpeed.toFixed(2),
      chaos: state.drawBotChaos.toFixed(2),
      drift: state.drawBotDrift.toFixed(2),
      turnRate: state.drawBotTurnRate.toFixed(2),
      speedVar: state.drawBotSpeedVar.toFixed(2),
    });
  }

  // Load existing recordings on startup
  fetch('/api/recordings').then(r => r.ok ? r.json() : []).then(recs => {
    recording.recordings = Array.isArray(recs) ? recs : [];
    console.log(`Recordings: loaded ${recording.recordings.length} gestures`);
    if (recording.recordings.length > 0) {
      recording.fingerprint = analyzeRecordings(recording.recordings);
    }
  }).catch(() => {});

  function finishRecording() {
    if (!recording.active) return;
    recording.active = false;
    const overlay = document.getElementById('recordingOverlay');
    if (overlay) overlay.style.display = 'none';
    if (recording.points.length >= 5) {
      const rec = { name: `gesture-${Date.now()}`, points: [...recording.points] };
      recording.recordings.push(rec);
      fetch('/api/recordings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(rec),
      }).catch(() => {});
      console.log(`Recording saved: ${rec.name} (${rec.points.length} points, ${recording.recordings.length} total)`);
      recording.fingerprint = analyzeRecordings(recording.recordings);
    } else {
      console.log(`Recording discarded (${recording.points.length} points, need >= 5)`);
    }
    recording.points = [];
  }

  canvas.addEventListener('pointerdown', e => {
    pointer.down = true;
    const [sx, sy] = screenToSimUV(e.clientX, e.clientY);
    pointer.x = sx;
    pointer.y = sy;
    // Start recording if C is held
    if (recording.cHeld && !recording.active) {
      recording.active = true;
      recording.startTime = performance.now();
      recording.points = [{ t: 0, x: sx, y: sy }];
      const overlay = document.getElementById('recordingOverlay');
      if (overlay) overlay.style.display = 'block';
    }
  });
  canvas.addEventListener('pointerup', () => {
    pointer.down = false;
    if (recording.active) finishRecording();
  });
  canvas.addEventListener('pointermove', e => {
    const [nx, ny] = screenToSimUV(e.clientX, e.clientY);
    pointer.dx = nx - pointer.x;
    pointer.dy = ny - pointer.y;
    pointer.x = nx;
    pointer.y = ny;
    pointer.moved = true;
    // Capture recording points
    if (recording.active) {
      const t = (performance.now() - recording.startTime) / 1000;
      recording.points.push({ t, x: nx, y: ny });
    }
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

  // ─── Auto-injectors ────────────────────────────────────────────────────
  const injectors = [
    { phase: 0.0, radius: 0.25, speed: 0.7, colorOffset: 0.0, cmIndex: 4 },
    { phase: 1.57, radius: 0.18, speed: -0.9, colorOffset: 0.08, cmIndex: 4 },
    { phase: 3.14, radius: 0.3, speed: 0.5, colorOffset: 0.15, cmIndex: 4 },
    { phase: 4.71, radius: 0.22, speed: -0.6, colorOffset: 0.22, cmIndex: 4 },
  ];

  let time = 0;
  let lastSplatTime = -1.0;

  // ─── Draw Bots ──────────────────────────────────────────────────────────
  function simplexNoise(t) {
    const i = Math.floor(t);
    const f = t - i;
    const smooth = f * f * (3 - 2 * f);
    const a = Math.sin(i * 127.1 + i * 311.7) * 43758.5453;
    const b = Math.sin((i+1) * 127.1 + (i+1) * 311.7) * 43758.5453;
    return (a - Math.floor(a)) * (1 - smooth) + (b - Math.floor(b)) * smooth;
  }

  const drawBots = [];
  let lastBotCount = 0;

  // Sacred geometry ratios for per-bot periodic steering
  const GOLDEN = (1 + Math.sqrt(5)) / 2;
  const SACRED_FREQS = [
    1.0,                    // circle
    GOLDEN,                 // golden spiral
    2.0,                    // figure-8 / lemniscate
    3.0,                    // trefoil
    GOLDEN * GOLDEN,        // nested golden
    Math.PI,                // irrational — never repeats
    Math.sqrt(2),           // diagonal harmony
    5 / 3,                  // pentatonic ratio
  ];

  function initDrawBots(count) {
    drawBots.length = 0;
    for (let i = 0; i < count; i++) {
      const startX = 0.3 + Math.random() * 0.4;
      const startY = 0.3 + Math.random() * 0.4;
      drawBots.push({
        x: startX, y: startY,
        _prevX: startX, _prevY: startY,
        vx: 0, vy: 0,
        angle: Math.random() * Math.PI * 2,
        turnRate: 0,
        colorPhase: i * 0.25,
        noiseT: Math.random() * 100,
        // Per-bot sacred geometry params
        baseFreq: SACRED_FREQS[i % SACRED_FREQS.length] * (0.3 + Math.random() * 0.4),
        secondFreq: SACRED_FREQS[(i + 3) % SACRED_FREQS.length] * (0.4 + Math.random() * 0.3),
        speedPhase: Math.random() * Math.PI * 2,
        orbitRadius: 0.08 + Math.random() * 0.22,  // Per-bot orbit scale
        symmetry: 3 + (i % 4),                      // 3-6 fold symmetry
        // Blend mode fields
        _mode: 'autonomous', _modeTimer: Math.random() * 3.0,
        // Recording playback fields
        _rec: null, _recIdx: 0, _recT: 0,
        _recRotation: 0, _recScale: 1, _recOffX: 0, _recOffY: 0,
        _recSpeed: 1, _recMirrorX: false, _recMirrorY: false, _recPause: 0,
        _recCentroidX: 0, _recCentroidY: 0,
      });
    }
  }

  // ─── Recording Playback ───────────────────────────────────────────────
  function pickRecordingForBot(bot) {
    const recs = recording.recordings;
    if (!recs.length) { bot._rec = null; return; }
    const rec = recs[Math.floor(Math.random() * recs.length)];
    bot._rec = rec;
    bot._recIdx = 0;
    bot._recT = 0;
    bot._recRotation = Math.random() * Math.PI * 2;
    bot._recScale = 0.6 + Math.random() * 0.8;
    bot._recSpeed = 0.7 + Math.random() * 0.6;
    bot._recMirrorX = Math.random() < 0.5;
    bot._recMirrorY = Math.random() < 0.5;
    bot._recPause = 0;
    // Compute gesture centroid
    let cx = 0, cy = 0;
    for (const p of rec.points) { cx += p.x; cy += p.y; }
    cx /= rec.points.length;
    cy /= rec.points.length;
    bot._recCentroidX = cx;
    bot._recCentroidY = cy;
    // Offset so gesture starts at bot's current position
    bot._recOffX = bot.x - cx;
    bot._recOffY = bot.y - cy;
  }

  function transformRecPoint(bot, px, py) {
    // Relative to gesture centroid
    let rx = px - bot._recCentroidX;
    let ry = py - bot._recCentroidY;
    // Mirror
    if (bot._recMirrorX) rx = -rx;
    if (bot._recMirrorY) ry = -ry;
    // Scale
    rx *= bot._recScale;
    ry *= bot._recScale;
    // Rotate
    const cos = Math.cos(bot._recRotation);
    const sin = Math.sin(bot._recRotation);
    const rotX = rx * cos - ry * sin;
    const rotY = rx * sin + ry * cos;
    // Offset to bot position
    return [rotX + bot._recCentroidX + bot._recOffX, rotY + bot._recCentroidY + bot._recOffY];
  }

  function updateDrawBotFromRecording(bot, dt) {
    if (!bot._rec) { pickRecordingForBot(bot); if (!bot._rec) return; }
    const rec = bot._rec;
    const pts = rec.points;

    // Pause between gestures
    if (bot._recPause > 0) {
      bot._recPause -= dt;
      bot.vx = 0; bot.vy = 0;
      return;
    }

    bot._recT += dt * bot._recSpeed;
    // Find the two points to interpolate between
    while (bot._recIdx < pts.length - 1 && pts[bot._recIdx + 1].t <= bot._recT) {
      bot._recIdx++;
    }

    if (bot._recIdx >= pts.length - 1) {
      // Gesture finished — switch to autonomous for a while
      bot._mode = 'autonomous';
      bot._modeTimer = (1 - state.drawBotRecMix) * 6.0 + Math.random() * 2.0;
      bot._rec = null;
      return;
    }

    const p0 = pts[bot._recIdx];
    const p1 = pts[bot._recIdx + 1];
    const segDt = p1.t - p0.t;
    const frac = segDt > 0 ? (bot._recT - p0.t) / segDt : 0;

    const rawX = p0.x + (p1.x - p0.x) * frac;
    const rawY = p0.y + (p1.y - p0.y) * frac;
    const [tx, ty] = transformRecPoint(bot, rawX, rawY);

    // Clamp to sphere boundary
    const SPHERE_R = 0.35;
    const dx = tx - 0.5, dy = ty - 0.5;
    const dist = Math.sqrt(dx * dx + dy * dy);
    let finalX = tx, finalY = ty;
    if (dist > SPHERE_R) {
      finalX = 0.5 + dx / dist * SPHERE_R * 0.95;
      finalY = 0.5 + dy / dist * SPHERE_R * 0.95;
    }

    bot.vx = (finalX - bot.x) / Math.max(dt, 0.001);
    bot.vy = (finalY - bot.y) / Math.max(dt, 0.001);
    bot.x = finalX;
    bot.y = finalY;
  }

  function updateBotAutonomous(bot, dt, speed, chaos, drift, turnRate, speedVar) {
    const SPHERE_R = 0.35;
    bot.noiseT += dt;
    const t = bot.noiseT * speed;
    const turnScale = 0.5 + turnRate * 3.0;

    // Sacred geometry: multi-frequency sinusoidal creates spirograph-like patterns
    const periodicSteer = (Math.sin(t * bot.baseFreq) * 1.5
                        + Math.sin(t * bot.secondFreq) * 0.8
                        + Math.sin(t * bot.baseFreq * bot.symmetry) * 0.3) * turnScale;

    // Noise steering for chaos
    const noiseSteer = (simplexNoise(bot.noiseT * (0.5 + chaos * 4)) * 2 - 1) * (1 + chaos * 10) * turnScale;

    // Blend: chaos=0 → pure sacred geometry, chaos=1 → pure noise
    const targetTurn = periodicSteer * (1 - chaos) + noiseSteer * chaos;
    bot.turnRate += (targetTurn - bot.turnRate) * (0.02 + chaos * 0.3);
    bot.angle += bot.turnRate * dt * speed;

    // Speed varies organically over time
    const sv = 1.0 + Math.sin(bot.noiseT * 0.7 + bot.speedPhase) * speedVar * 0.6;
    const spd = speed * 0.3 * sv;
    bot.vx = Math.cos(bot.angle) * spd;
    bot.vy = Math.sin(bot.angle) * spd;

    // Drift toward/away from center
    const toCenterX = 0.5 - bot.x;
    const toCenterY = 0.5 - bot.y;
    const distFromCenter = Math.sqrt(toCenterX * toCenterX + toCenterY * toCenterY);
    const centerPull = drift < 0.5
      ? 0.02 + distFromCenter * 0.1
      : 0.005 + distFromCenter * 0.03;
    bot.vx += toCenterX * centerPull;
    bot.vy += toCenterY * centerPull;

    // Hard boundary: reflect off sphere edge
    const newX = bot.x + bot.vx * dt;
    const newY = bot.y + bot.vy * dt;
    const newDist = Math.sqrt((newX - 0.5) ** 2 + (newY - 0.5) ** 2);
    if (newDist > SPHERE_R) {
      bot.angle = Math.atan2(0.5 - bot.y, 0.5 - bot.x) + (Math.random() - 0.5) * 1.0;
      bot.vx *= -0.3;
      bot.vy *= -0.3;
    }
    bot.x += bot.vx * dt;
    bot.y += bot.vy * dt;
    const d = Math.sqrt((bot.x - 0.5) ** 2 + (bot.y - 0.5) ** 2);
    if (d > SPHERE_R) {
      bot.x = 0.5 + (bot.x - 0.5) / d * SPHERE_R * 0.95;
      bot.y = 0.5 + (bot.y - 0.5) / d * SPHERE_R * 0.95;
    }
  }

  function updateDrawBots(dt, time) {
    const count = Math.round(state.drawBotCount);
    if (count !== lastBotCount) {
      initDrawBots(count);
      lastBotCount = count;
    }
    if (count === 0) return;

    // More bots → slightly slower so they don't overwhelm
    const botCountScale = 1 / Math.max(1, count);
    const speed = (0.1 + state.drawBotSpeed * 2.9) * (0.5 + botCountScale * 0.5);
    const chaos = state.drawBotChaos;
    const drift = state.drawBotDrift;
    const turnRate = state.drawBotTurnRate;
    const speedVar = state.drawBotSpeedVar;
    const hasRecordings = state.useRecordings && recording.recordings.length > 0;
    const recMix = hasRecordings ? state.drawBotRecMix : 0;

    for (const bot of drawBots) {
      if (recMix > 0 && bot._mode === 'recording') {
        updateDrawBotFromRecording(bot, dt);
      } else {
        updateBotAutonomous(bot, dt, speed, chaos, drift, turnRate, speedVar);
        if (recMix > 0) {
          bot._modeTimer -= dt;
          if (bot._modeTimer <= 0) {
            bot._mode = 'recording';
            bot._rec = null;
          }
        }
      }
    }
  }

  // ─── Blobs — large Brownian-motion agents ────────────────────────────────
  const blobs = [];
  let lastBlobCount = 0;

  function initBlobs(count) {
    blobs.length = 0;
    for (let i = 0; i < count; i++) {
      const sx = 0.3 + Math.random() * 0.4;
      const sy = 0.3 + Math.random() * 0.4;
      blobs.push({
        x: sx, y: sy, _prevX: sx, _prevY: sy,
        vx: 0, vy: 0,
        colorPhase: i * 0.4,
        wanderAngle: Math.random() * Math.PI * 2,
      });
    }
  }

  const SPHERE_R_AGENTS = 0.35;

  function clampToSphere(agent) {
    const dx = agent.x - 0.5, dy = agent.y - 0.5;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d > SPHERE_R_AGENTS) {
      agent.x = 0.5 + dx / d * SPHERE_R_AGENTS * 0.95;
      agent.y = 0.5 + dy / d * SPHERE_R_AGENTS * 0.95;
    }
  }

  function updateBlobs(dt) {
    const count = Math.round(state.blobCount);
    if (count !== lastBlobCount) { initBlobs(count); lastBlobCount = count; }
    if (count === 0) return;

    const speed = state.blobSpeed;
    const wander = state.blobWander;

    for (const blob of blobs) {
      // Brownian: smooth random walk (Ornstein-Uhlenbeck style)
      blob.wanderAngle += (Math.random() - 0.5) * wander * 6 * dt;
      const force = speed * 0.2;
      blob.vx += Math.cos(blob.wanderAngle) * force * dt;
      blob.vy += Math.sin(blob.wanderAngle) * force * dt;
      // Damping
      blob.vx *= 0.97;
      blob.vy *= 0.97;
      // Drift toward center
      const toCx = 0.5 - blob.x, toCy = 0.5 - blob.y;
      const dist = Math.sqrt(toCx * toCx + toCy * toCy);
      const pull = 0.01 + dist * 0.06;
      blob.vx += toCx * pull * dt;
      blob.vy += toCy * pull * dt;
      blob.x += blob.vx;
      blob.y += blob.vy;
      clampToSphere(blob);
    }
  }

  // ─── Flockers — Reynolds boids + blob interaction ──────────────────────
  const flockers = [];
  let lastFlockCount = 0;

  function initFlockers(count) {
    flockers.length = 0;
    for (let i = 0; i < count; i++) {
      const sx = 0.3 + Math.random() * 0.4;
      const sy = 0.3 + Math.random() * 0.4;
      flockers.push({
        x: sx, y: sy, _prevX: sx, _prevY: sy,
        vx: (Math.random() - 0.5) * 0.02,
        vy: (Math.random() - 0.5) * 0.02,
        colorPhase: i * 0.12,
      });
    }
  }

  function updateFlockers(dt) {
    const count = Math.round(state.flockCount);
    if (count !== lastFlockCount) { initFlockers(count); lastFlockCount = count; }
    if (count === 0) return;

    const sep = state.flockSeparation;
    const ali = state.flockAlignment;
    const coh = state.flockCohesion;
    const blobReact = state.flockBlobReact;
    const speed = state.flockSpeed;
    const maxSpd = (0.05 + speed * 0.3) * dt;

    for (const f of flockers) {
      let sx = 0, sy = 0, ax = 0, ay = 0, cx = 0, cy = 0, n = 0;

      // Flocking forces from neighbors
      for (const other of flockers) {
        if (other === f) continue;
        const dx = other.x - f.x, dy = other.y - f.y;
        const d2 = dx * dx + dy * dy;
        if (d2 > 0.02) continue; // perception radius ~0.14
        const d = Math.sqrt(d2);
        n++;
        if (d < 0.03 && d > 0.0001) { sx -= dx / d; sy -= dy / d; }
        ax += other.vx; ay += other.vy;
        cx += other.x; cy += other.y;
      }

      if (n > 0) {
        ax /= n; ay /= n;
        cx = cx / n - f.x; cy = cy / n - f.y;
        f.vx += sx * sep * 0.08 + (ax - f.vx) * ali * 0.04 + cx * coh * 0.015;
        f.vy += sy * sep * 0.08 + (ay - f.vy) * ali * 0.04 + cy * coh * 0.015;
      }

      // Blob interaction: positive = attract, negative = flee
      for (const blob of blobs) {
        const dx = blob.x - f.x, dy = blob.y - f.y;
        const d = Math.sqrt(dx * dx + dy * dy) + 0.001;
        const force = blobReact * 0.015 / (d * d + 0.01);
        f.vx += dx / d * force;
        f.vy += dy / d * force;
      }

      // Center pull
      f.vx += (0.5 - f.x) * 0.003;
      f.vy += (0.5 - f.y) * 0.003;

      // Speed limit
      const spd = Math.sqrt(f.vx * f.vx + f.vy * f.vy);
      if (spd > maxSpd) {
        f.vx *= maxSpd / spd;
        f.vy *= maxSpd / spd;
      }

      f.x += f.vx;
      f.y += f.vy;
      clampToSphere(f);
    }
  }

  // ─── Frame loop ─────────────────────────────────────────────────────────
  function frame() {
    if (!frameRunning) return;
    requestAnimationFrame(frame);

    // Cap GPU queue depth: skip frame if GPU is >2 frames behind
    if (gpuFramesPending > 2) {
      gpuFramesSkipped++;
      return;
    }
    gpuFramesPending++;

    const dt = 0.016 * state.simSpeed;
    time += dt;

    updateAutoMorph();
    updateDrawBots(dt, time);
    updateBlobs(dt);
    updateFlockers(dt);

    // Pre-convert colors to Oklab on CPU (avoids per-particle GPU conversion)
    const okGlitBase = linearToOklabCPU(state.glitterColor);
    const okGlitAccent = linearToOklabCPU(state.glitterAccent);
    const okGlitTip = linearToOklabCPU(state.glitterTip);
    const okBaseCol = linearToOklabCPU(state.baseColor);
    const okAccentCol = linearToOklabCPU(state.accentColor);
    const okTipCol = linearToOklabCPU(state.tipColor);

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
    displayUBData[16] = okTipCol[0];
    displayUBData[17] = okTipCol[1];
    displayUBData[18] = okTipCol[2];
    device.queue.writeBuffer(displayUB, 0, displayUBData);

    // Collect splats into pre-allocated buffer
    splatCount = 0;

    // Dye ramp: fade in over first 3 seconds (starting at t=1) to prevent initial blowout
    const dyeRamp = Math.min(1.0, (time - 1.0) / 3.0);

    // Auto-injectors (scaled by injectorIntensity; 0 = off)
    const injI = state.injectorIntensity;
    const injCount = Math.round(state.injectorCount);
    if (injI > 0.01 && injCount > 0) {
      // Build up to 8 injectors by duplicating base 4 with offset phases
      const allInjectors = [];
      for (let i = 0; i < Math.min(injCount, injectors.length); i++) {
        allInjectors.push(injectors[i]);
      }
      for (let i = injectors.length; i < injCount; i++) {
        const base = injectors[i % injectors.length];
        allInjectors.push({
          phase: base.phase + Math.PI * 0.5,
          radius: base.radius * 0.85,
          speed: base.speed * 1.2,
          colorOffset: base.colorOffset + 0.3,
          cmIndex: base.cmIndex,
        });
      }
      // More injectors → slightly slower so they don't overwhelm
      const injCountScale = 1 / Math.max(1, injCount);
      const injSplatRadius = state.splatRadius * 2.0 * (0.5 + state.injectorSize * 3.0);
      const spdMul = (0.2 + state.injectorSpeed * 3.6) * (0.5 + injCountScale * 0.5);
      for (const inj of allInjectors) {
        if (time < 1.0) { continue; }
        const spd = inj.speed * spdMul;
        const angle = time * spd + inj.phase;
        const cx = 0.5 + Math.cos(angle) * inj.radius;
        const cy = 0.5 + Math.sin(angle) * inj.radius;
        const vx = -Math.sin(angle) * spd * inj.radius * 0.5;
        const vy = Math.cos(angle) * spd * inj.radius * 0.5;
        const col = palette(time * 0.12 + inj.colorOffset, inj.cmIndex);
        addSplat(cx, cy,
          vx * state.splatForce * 0.1 * injI,
          vy * state.splatForce * 0.1 * injI,
          col[0] * dyeRamp * injI, col[1] * dyeRamp * injI, col[2] * dyeRamp * injI,
          injSplatRadius);
      }

      // Random splat burst every 3-5 seconds (controlled by burstCount)
      const bc = Math.round(state.burstCount);
      if (bc > 0 && time - lastSplatTime > 2.0 + Math.random() * 1.5) {
        lastSplatTime = time;
        const count = Math.max(1, bc - 1) + Math.floor(Math.random() * (bc + 1));
        for (let i = 0; i < count; i++) {
          const col = palette(Math.random(), 4);
          const angle = Math.random() * Math.PI * 2;
          const force = (500 + Math.random() * 1000) * injI;
          addSplat(
            0.15 + Math.random() * 0.7,
            0.15 + Math.random() * 0.7,
            Math.cos(angle) * force,
            Math.sin(angle) * force,
            col[0] * dyeRamp * injI, col[1] * dyeRamp * injI, col[2] * dyeRamp * injI,
            state.splatRadius * (2.0 + Math.random() * 3));
        }
      }
    }

    // Draw bot splats — gentle like injectors, not like mouse drags
    for (const bot of drawBots) {
      const dx = bot.x - bot._prevX;
      const dy = bot.y - bot._prevY;
      bot._prevX = bot.x;
      bot._prevY = bot.y;

      const speed = Math.sqrt(dx * dx + dy * dy);
      if (speed < 0.0001) continue;

      const col = palette(time * 0.15 + bot.colorPhase, 4);
      // Gentle but visible force — not mouse-slam, not invisible either
      const botForce = 0.4;
      const botDyeStr = 1.0;
      const botRadius = state.splatRadius * (1.0 + state.drawBotSize * 3.0);
      addSplat(
        bot.x, bot.y,
        dx * state.splatForce * botForce,
        dy * state.splatForce * botForce,
        col[0] * dyeRamp * botDyeStr, col[1] * dyeRamp * botDyeStr, col[2] * dyeRamp * botDyeStr,
        botRadius
      );
    }

    // Blob splats — big, slow, gentle
    for (const blob of blobs) {
      const dx = blob.x - blob._prevX;
      const dy = blob.y - blob._prevY;
      blob._prevX = blob.x;
      blob._prevY = blob.y;
      if (Math.sqrt(dx * dx + dy * dy) < 0.00005) continue;
      const col = palette(time * 0.08 + blob.colorPhase, 4);
      const radius = state.splatRadius * (2.0 + state.blobSize * 5.0);
      addSplat(blob.x, blob.y,
        dx * state.splatForce * 0.3, dy * state.splatForce * 0.3,
        col[0] * dyeRamp * 0.8, col[1] * dyeRamp * 0.8, col[2] * dyeRamp * 0.8,
        radius);
    }

    // Flocker splats — rotate subset each frame to stay under MAX_SPLATS
    {
      const maxFlockSplats = Math.min(flockers.length, 30);
      const startIdx = Math.floor(time * 60) % Math.max(1, flockers.length);
      let splatted = 0;
      for (let i = 0; i < flockers.length && splatted < maxFlockSplats; i++) {
        const f = flockers[(startIdx + i) % flockers.length];
        const dx = f.x - f._prevX;
        const dy = f.y - f._prevY;
        f._prevX = f.x;
        f._prevY = f.y;
        if (Math.sqrt(dx * dx + dy * dy) < 0.00005) continue;
        const col = palette(time * 0.2 + f.colorPhase, 4);
        const radius = state.splatRadius * (0.8 + state.flockSize * 3.0);
        addSplat(f.x, f.y,
          dx * state.splatForce * 0.3, dy * state.splatForce * 0.3,
          col[0] * dyeRamp * 0.9, col[1] * dyeRamp * 0.9, col[2] * dyeRamp * 0.9,
          radius);
        splatted++;
      }
    }

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

    // ── Curl Noise (field-wide velocity perturbation) ──
    if (state.noiseAmount > 0.01) {
      noiseData[0] = time;
      // Cubic curve so slider gives fine control at low values
      const na = state.noiseAmount;
      noiseData[1] = na * na * na;
      noiseData[2] = SIM_RES;
      noiseData[3] = state.noiseFrequency;
      noiseData[4] = state.noiseSpeed;
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
    if (hasDyeNoise || hasCurlDye) {
      const col = palette(time * 0.05, 4);
      dyeNoiseData[0] = time;
      dyeNoiseData[1] = hasDyeNoise ? state.dyeNoiseAmount : 0;
      dyeNoiseData[2] = SIM_RES;
      // noiseDyeIntensity controls how much dye the curl noise injects
      const ndi = state.noiseDyeIntensity;
      dyeNoiseData[3] = hasCurlDye ? ndi * ndi : 0;
      dyeNoiseData[4] = col[0];
      dyeNoiseData[5] = col[1];
      dyeNoiseData[6] = col[2];
      dyeNoiseData[7] = 1;
      device.queue.writeBuffer(dyeNoiseBuf, 0, dyeNoiseData);
      const p = enc.beginComputePass();
      p.setPipeline(dyeNoisePipe.pipeline);
      p.setBindGroup(0, dyeNoiseBGs[velFlip][dyeFlip]);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
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
    const canvasView = ctx.getCurrentTexture().createView();
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

    // ── Pass 14: Glass Shell (commented out) ──
    // glassUBData[0] = canvas.width;
    // glassUBData[1] = canvas.height;
    // glassUBData[2] = time;
    // glassUBData[4] = state.rimIntensity;
    // glassUBData[5] = state.causticIntensity;
    // device.queue.writeBuffer(glassUB, 0, glassUBData);
    // { const rp = enc.beginRenderPass({ ... });
    //   rp.setPipeline(glassShellPipeline);
    //   rp.setBindGroup(0, glassShellBG);
    //   rp.draw(3); rp.end(); }

    device.queue.submit([enc.finish()]);
    device.queue.onSubmittedWorkDone().then(() => { gpuFramesPending--; });
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

  wireSlider('sheenStrength', 'sheenStrength');
  wireColor('sheenColor', 'sheenColor');
  // wireSlider('rimIntensity', 'rimIntensity'); // glass marble
  // wireSlider('chromaticStrength', 'chromaticStrength'); // glass marble
  // wireSlider('causticIntensity', 'causticIntensity'); // glass marble
  wireSlider('clickSize', 'clickSize');
  wireSlider('clickStrength', 'clickStrength');
  wireSlider('injectorIntensity', 'injectorIntensity');
  wireSlider('noiseAmount', 'noiseAmount');
  wireSlider('noiseFrequency', 'noiseFrequency');
  wireSlider('noiseSpeed', 'noiseSpeed');
  wireSlider('injectorSize', 'injectorSize');
  wireSlider('injectorCount', 'injectorCount', v => Math.round(v));
  wireSlider('injectorSpeed', 'injectorSpeed');
  wireSlider('burstCount', 'burstCount', v => Math.round(v));
  wireSlider('curlStrength', 'curlStrength', v => Math.round(v));
  wireSlider('splatForce', 'splatForce', v => Math.round(v));
  wireSlider('velDissipation', 'velDissipation', v => v.toFixed(3));
  wireSlider('dyeDissipation', 'dyeDissipation', v => v.toFixed(3));
  wireSlider('pressureIters', 'pressureIters', v => Math.round(v));
  wireSlider('pressureDecay', 'pressureDecay', v => v.toFixed(2));
  wireSlider('simSpeed', 'simSpeed');
  wireSlider('drawBotCount', 'drawBotCount', v => Math.round(v));
  wireSlider('drawBotSpeed', 'drawBotSpeed');
  wireSlider('drawBotSize', 'drawBotSize');
  wireSlider('drawBotTurnRate', 'drawBotTurnRate');
  wireSlider('drawBotSpeedVar', 'drawBotSpeedVar');
  wireSlider('drawBotRecMix', 'drawBotRecMix');
  wireSlider('drawBotChaos', 'drawBotChaos');
  wireSlider('drawBotDrift', 'drawBotDrift');
  wireSlider('blobCount', 'blobCount', v => Math.round(v));
  wireSlider('blobSpeed', 'blobSpeed');
  wireSlider('blobSize', 'blobSize');
  wireSlider('blobWander', 'blobWander');
  wireSlider('flockCount', 'flockCount', v => Math.round(v));
  wireSlider('flockSpeed', 'flockSpeed');
  wireSlider('flockSize', 'flockSize');
  wireSlider('flockSeparation', 'flockSeparation');
  wireSlider('flockAlignment', 'flockAlignment');
  wireSlider('flockCohesion', 'flockCohesion');
  wireSlider('flockBlobReact', 'flockBlobReact');
  wireSlider('noiseDyeIntensity', 'noiseDyeIntensity');
  wireSlider('dyeNoiseAmount', 'dyeNoiseAmount');
  wireSlider('bloomIntensity', 'bloomIntensity', v => v.toFixed(2));
  wireSlider('bloomThreshold', 'bloomThreshold', v => v.toFixed(2));
  wireSlider('bloomRadius', 'bloomRadius', v => v.toFixed(2));

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
    syncSlider('clickSize', 'clickSize');
    syncSlider('clickStrength', 'clickStrength');
    syncSlider('injectorIntensity', 'injectorIntensity');
    syncSlider('noiseAmount', 'noiseAmount');
    syncSlider('noiseFrequency', 'noiseFrequency');
    syncSlider('noiseSpeed', 'noiseSpeed');
    syncSlider('injectorSize', 'injectorSize');
    syncSlider('injectorCount', 'injectorCount', v => Math.round(v));
    syncSlider('injectorSpeed', 'injectorSpeed');
    syncSlider('burstCount', 'burstCount', v => Math.round(v));
    syncSlider('curlStrength', 'curlStrength', v => Math.round(v));
    syncSlider('splatForce', 'splatForce', v => Math.round(v));
    syncSlider('velDissipation', 'velDissipation', v => v.toFixed(3));
    syncSlider('dyeDissipation', 'dyeDissipation', v => v.toFixed(3));
    syncSlider('pressureIters', 'pressureIters', v => Math.round(v));
    syncSlider('pressureDecay', 'pressureDecay', v => v.toFixed(2));
    syncSlider('drawBotCount', 'drawBotCount', v => Math.round(v));
    syncSlider('drawBotSpeed', 'drawBotSpeed');
    syncSlider('drawBotSize', 'drawBotSize');
    syncSlider('drawBotTurnRate', 'drawBotTurnRate');
    syncSlider('drawBotSpeedVar', 'drawBotSpeedVar');
    syncSlider('drawBotRecMix', 'drawBotRecMix');
    syncSlider('drawBotChaos', 'drawBotChaos');
    syncSlider('drawBotDrift', 'drawBotDrift');
    syncSlider('blobCount', 'blobCount', v => Math.round(v));
    syncSlider('blobSpeed', 'blobSpeed');
    syncSlider('blobSize', 'blobSize');
    syncSlider('blobWander', 'blobWander');
    syncSlider('flockCount', 'flockCount', v => Math.round(v));
    syncSlider('flockSpeed', 'flockSpeed');
    syncSlider('flockSize', 'flockSize');
    syncSlider('flockSeparation', 'flockSeparation');
    syncSlider('flockAlignment', 'flockAlignment');
    syncSlider('flockCohesion', 'flockCohesion');
    syncSlider('flockBlobReact', 'flockBlobReact');
    syncSlider('noiseDyeIntensity', 'noiseDyeIntensity');
    syncSlider('dyeNoiseAmount', 'dyeNoiseAmount');
    syncColor('baseColor', 'baseColor');
    syncColor('accentColor', 'accentColor');
    syncColor('glitterColor', 'glitterColor');
    syncColor('glitterAccent', 'glitterAccent');
    syncColor('tipColor', 'tipColor');
    syncColor('glitterTip', 'glitterTip');
    syncColor('sheenColor', 'sheenColor');
    syncSlider('simSpeed', 'simSpeed');
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
    // Interaction
    state.clickSize = snapTo(randRange(0, 1), 0.01);
    state.clickStrength = snapTo(randRange(0, 1), 0.01);
    // Injectors
    state.injectorIntensity = snapTo(randRange(0, 1), 0.01);
    state.injectorSize = snapTo(randRange(0, 1), 0.01);
    state.injectorCount = Math.round(randRange(0, 8));
    state.injectorSpeed = snapTo(randRange(0, 1), 0.01);
    state.burstCount = Math.round(randRange(0, 8));
    // Noise
    state.noiseAmount = snapTo(randRange(0, 1), 0.01);
    state.noiseFrequency = snapTo(randRange(0, 1), 0.01);
    state.noiseSpeed = snapTo(randRange(0, 1), 0.01);
    // Fluid sim
    state.curlStrength = Math.round(randRange(0, 50));
    state.splatForce = snapTo(randRange(1000, 20000), 100);
    state.velDissipation = snapTo(randRange(0.99, 1.0), 0.001);
    state.dyeDissipation = snapTo(randRange(0.98, 1.0), 0.001);
    state.pressureIters = Math.round(randRange(10, 60));
    state.pressureDecay = snapTo(randRange(0, 1), 0.01);
    // Draw bots
    state.drawBotCount = Math.round(Math.random() * 8);
    state.drawBotSpeed = Math.random();
    state.drawBotSize = 0.01 + Math.random() * 4.99;
    state.drawBotTurnRate = Math.random();
    state.drawBotSpeedVar = Math.random();
    state.drawBotRecMix = Math.random();
    state.drawBotChaos = Math.random();
    state.drawBotDrift = Math.random();
    // Blobs
    state.blobCount = Math.round(Math.random() * 4);
    state.blobSpeed = Math.random();
    state.blobSize = Math.random();
    state.blobWander = Math.random();
    // Flockers
    state.flockCount = Math.round(Math.pow(Math.random(), 0.5) * 50);
    state.flockSpeed = Math.random();
    state.flockSize = Math.random();
    state.flockSeparation = Math.random();
    state.flockAlignment = Math.random();
    state.flockCohesion = Math.random();
    state.flockBlobReact = Math.random() * 2 - 1;
    // Dye noise
    state.noiseDyeIntensity = Math.random();
    state.dyeNoiseAmount = Math.random() * 0.15;
    // NOTE: particleCount is intentionally NOT randomized
    syncAllUI();
  });

  // ─── Auto-Morph ──────────────────────────────────────────────────────
  const morphSliders = {
    simSpeed: { min: 0, max: 3, step: 0.01 },
    sizeRandomness: { min: 0, max: 1, step: 0.01 },
    colorBlend: { min: 0, max: 1, step: 0.01 },
    sheenStrength: { min: 0, max: 1, step: 0.01 },
    clickSize: { min: 0, max: 1, step: 0.01 },
    clickStrength: { min: 0, max: 1, step: 0.01 },
    injectorIntensity: { min: 0, max: 1, step: 0.01 },
    injectorSize: { min: 0.5, max: 1, step: 0.01 },
    injectorCount: { min: 0, max: 8, step: 1 },
    injectorSpeed: { min: 0, max: 1, step: 0.01 },
    burstCount: { min: 0, max: 8, step: 1 },
    noiseAmount: { min: 0, max: 1, step: 0.01 },
    noiseFrequency: { min: 0, max: 1, step: 0.01 },
    noiseSpeed: { min: 0, max: 1, step: 0.01 },
    curlStrength: { min: 0, max: 50, step: 1 },
    splatForce: { min: 1000, max: 20000, step: 100 },
    velDissipation: { min: 0.99, max: 1.0, step: 0.001 },
    dyeDissipation: { min: 0.98, max: 1.0, step: 0.001 },
    pressureIters: { min: 10, max: 60, step: 1 },
    pressureDecay: { min: 0, max: 1, step: 0.01 },
    drawBotCount: { min: 0, max: 8, step: 1 },
    drawBotSpeed: { min: 0, max: 1, step: 0.01 },
    drawBotSize: { min: 0.01, max: 5, step: 0.01 },
    drawBotTurnRate: { min: 0, max: 1, step: 0.01 },
    drawBotSpeedVar: { min: 0, max: 1, step: 0.01 },
    drawBotRecMix: { min: 0, max: 1, step: 0.01 },
    drawBotChaos: { min: 0, max: 1, step: 0.01 },
    drawBotDrift: { min: 0, max: 1, step: 0.01 },
    blobCount: { min: 0, max: 4, step: 1 },
    blobSpeed: { min: 0, max: 1, step: 0.01 },
    blobSize: { min: 0, max: 1, step: 0.01 },
    blobWander: { min: 0, max: 1, step: 0.01 },
    flockCount: { min: 0, max: 50, step: 1 },
    flockSpeed: { min: 0, max: 1, step: 0.01 },
    flockSize: { min: 0, max: 1, step: 0.01 },
    flockSeparation: { min: 0, max: 1, step: 0.01 },
    flockAlignment: { min: 0, max: 1, step: 0.01 },
    flockCohesion: { min: 0, max: 1, step: 0.01 },
    flockBlobReact: { min: -1, max: 1, step: 0.01 },
    noiseDyeIntensity: { min: 0, max: 1, step: 0.01 },
    dyeNoiseAmount: { min: 0, max: 0.15, step: 0.001 },
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

  // ─── Recording UI ────────────────────────────────────────────────────
  const useRecBtn = document.getElementById('useRecordingsBtn');
  const clearRecBtn = document.getElementById('clearRecordingsBtn');

  useRecBtn.addEventListener('click', () => {
    if (!state.useRecordings && recording.recordings.length === 0) {
      alert('No recordings yet. Hold C and draw to record gestures.');
      return;
    }
    state.useRecordings = !state.useRecordings;
    useRecBtn.classList.toggle('active', state.useRecordings);
    if (state.useRecordings) {
      // Seed procedural params from recording fingerprint + reset bot modes
      if (recording.fingerprint) seedStateFromFingerprint(recording.fingerprint);
      for (const bot of drawBots) {
        bot._rec = null; bot._recPause = 0;
        bot._mode = 'autonomous'; bot._modeTimer = Math.random() * 2.0;
      }
      syncAllUI();
    }
    console.log(`Use Recordings: ${state.useRecordings} (${recording.recordings.length} gestures)`);
  });

  clearRecBtn.addEventListener('click', async () => {
    if (recording.recordings.length === 0) { alert('No recordings to clear.'); return; }
    if (!confirm(`Clear all ${recording.recordings.length} recordings?`)) return;
    recording.recordings = [];
    state.useRecordings = false;
    useRecBtn.classList.remove('active');
    await fetch('/api/recordings', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ all: true }),
    });
    console.log('All recordings cleared');
  });

  // Keyboard handler
  document.addEventListener('keydown', (e) => {
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT') return;

    // Hold C to record
    if ((e.key === 'c' || e.key === 'C') && !e.repeat) {
      recording.cHeld = true;
      // If pointer is already down, start recording immediately
      if (pointer.down && !recording.active) {
        recording.active = true;
        recording.startTime = performance.now();
        recording.points = [{ t: 0, x: pointer.x, y: pointer.y }];
        const overlay = document.getElementById('recordingOverlay');
        if (overlay) overlay.style.display = 'block';
      }
      return;
    }

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

  document.addEventListener('keyup', (e) => {
    if (e.key === 'c' || e.key === 'C') {
      recording.cHeld = false;
      if (recording.active) finishRecording();
    }
  });

  loadStatus.textContent = 'Starting simulation...';
  console.log(`Fluid simulation starting... (${(performance.now() - t0).toFixed(0)}ms)`);
  document.getElementById('loading').style.display = 'none';
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
