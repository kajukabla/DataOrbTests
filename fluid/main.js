// ─── Configuration ───────────────────────────────────────────────────────────
const SIM_RES = 512;
const WORKGROUP = 8;
const TEX_FMT = 'rgba16float';
const PARTICLE_STRIDE = 32;       // 8 floats × 4 bytes
const PARTICLE_WG = 256;

const state = {
  particleCount: 4194304,   // 4M default
  particleSize: 0.6,
  sizeRandomness: 0.3,
  glintBrightness: 1.2,
  prismaticAmount: 5.0,
  baseColor: [1.0, 0.55, 0.1],
  accentColor: [0.15, 0.3, 0.8],
  glitterColor: [1.0, 1.0, 1.0],
  glitterAccent: [0.3, 0.5, 1.0],
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
  burstCount: 3,
  sheenColor: [1.0, 0.9, 0.7],
  splatForce: 6000,
  curlStrength: 15,
  pressureIters: 30,
  pressureDecay: 0.8,
  velDissipation: 0.998,
  dyeDissipation: 0.993,
  splatRadius: 0.0015,
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
`;

const MAX_SPLATS = 16;

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

const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS = 0.43;

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
  let boundaryFade = 1.0 - smoothstep(SPHERE_RADIUS - 0.06, SPHERE_RADIUS, dist);
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

const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS = 0.43;

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
  var dye = textureLoad(src, id.xy, 0);
  let boundaryFade = 1.0 - smoothstep(SPHERE_RADIUS - 0.06, SPHERE_RADIUS, dist);
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

const curlShader = /* wgsl */`
${commonHeader}
@group(0) @binding(1) var vel: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let L = textureLoad(vel, vec2u(u32(max(i32(id.x)-1, 0)), id.y), 0).xy;
  let R = textureLoad(vel, vec2u(min(id.x+1, res-1), id.y), 0).xy;
  let B = textureLoad(vel, vec2u(id.x, u32(max(i32(id.y)-1, 0))), 0).xy;
  let T = textureLoad(vel, vec2u(id.x, min(id.y+1, res-1)), 0).xy;
  let curl = 0.5 * ((R.y - L.y) - (T.x - B.x));
  textureStore(dst, id.xy, vec4f(curl, 0.0, 0.0, 1.0));
}
`;

const vorticityShader = /* wgsl */`
${commonHeader}
@group(0) @binding(1) var velTex: texture_2d<f32>;
@group(0) @binding(2) var curlTex: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS = 0.43;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / p.simRes;
  let dist = length(uv - SPHERE_CENTER);
  if (dist > SPHERE_RADIUS) {
    textureStore(dst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }
  let cL = abs(textureLoad(curlTex, vec2u(u32(max(i32(id.x)-1, 0)), id.y), 0).x);
  let cR = abs(textureLoad(curlTex, vec2u(min(id.x+1, res-1), id.y), 0).x);
  let cB = abs(textureLoad(curlTex, vec2u(id.x, u32(max(i32(id.y)-1, 0))), 0).x);
  let cT = abs(textureLoad(curlTex, vec2u(id.x, min(id.y+1, res-1)), 0).x);
  let c  = textureLoad(curlTex, id.xy, 0).x;
  var N = vec2f(cR - cL, cT - cB);
  let lenN = length(N);
  if (lenN < 1e-5) {
    textureStore(dst, id.xy, textureLoad(velTex, id.xy, 0));
    return;
  }
  N = N / lenN;
  let force = vec2f(N.y, -N.x) * c * p.curlStrength;
  var vel = textureLoad(velTex, id.xy, 0).xy;
  vel += force * p.dt;
  textureStore(dst, id.xy, vec4f(vel, 0.0, 1.0));
}
`;

const divergenceShader = /* wgsl */`
${commonHeader}
@group(0) @binding(1) var vel: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let L = textureLoad(vel, vec2u(u32(max(i32(id.x)-1, 0)), id.y), 0).x;
  let R = textureLoad(vel, vec2u(min(id.x+1, res-1), id.y), 0).x;
  let B = textureLoad(vel, vec2u(id.x, u32(max(i32(id.y)-1, 0))), 0).y;
  let T = textureLoad(vel, vec2u(id.x, min(id.y+1, res-1)), 0).y;
  let div = 0.5 * ((R - L) + (T - B));
  textureStore(dst, id.xy, vec4f(div, 0.0, 0.0, 1.0));
}
`;

const clearPressureShader = /* wgsl */`
${commonHeader}
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let pr = textureLoad(src, id.xy, 0).x * p.pressureDecay;
  textureStore(dst, id.xy, vec4f(pr, 0.0, 0.0, 1.0));
}
`;

const jacobiShader = /* wgsl */`
${commonHeader}
@group(0) @binding(1) var pressure: texture_2d<f32>;
@group(0) @binding(2) var divTex: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let pL = textureLoad(pressure, vec2u(u32(max(i32(id.x)-1, 0)), id.y), 0).x;
  let pR = textureLoad(pressure, vec2u(min(id.x+1, res-1), id.y), 0).x;
  let pB = textureLoad(pressure, vec2u(id.x, u32(max(i32(id.y)-1, 0))), 0).x;
  let pT = textureLoad(pressure, vec2u(id.x, min(id.y+1, res-1)), 0).x;
  let d  = textureLoad(divTex, id.xy, 0).x;
  let pNew = (pL + pR + pB + pT - d) * 0.25;
  textureStore(dst, id.xy, vec4f(pNew, 0.0, 0.0, 1.0));
}
`;

const gradSubShader = /* wgsl */`
${commonHeader}
@group(0) @binding(1) var velTex: texture_2d<f32>;
@group(0) @binding(2) var pressTex: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS = 0.43;

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

const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS = 0.43;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / p.simRes;

  // Sphere boundary check
  let toCenter = uv - SPHERE_CENTER;
  let dist = length(toCenter);
  if (dist > SPHERE_RADIUS) {
    textureStore(dst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }

  let vel = textureLoad(src, id.xy, 0).xy;
  let backUV = uv - p.dt * vel * p.dx;
  let clamped = clamp(backUV, vec2f(0.5 / p.simRes), vec2f(1.0 - 0.5 / p.simRes));
  var advected = textureSampleLevel(src, sampl, clamped, 0.0).xy;

  // Boundary interaction: hard reflect velocity off sphere wall
  let boundaryZone = SPHERE_RADIUS - 0.06;
  if (dist > boundaryZone) {
    let normal = toCenter / dist;
    let outward = dot(advected, normal);
    let proximity = (dist - boundaryZone) / (SPHERE_RADIUS - boundaryZone);
    if (outward > 0.0) {
      // Full reflection off wall — velocity reverses direction on the normal
      advected -= normal * outward * 2.0;
      // Slight damping scales with how close to boundary
      advected *= mix(1.0, 0.7, proximity);
    }
    // Strong repulsion force pushes fluid inward near the wall
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

const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS = 0.43;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(p.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / p.simRes;

  // Zero dye outside sphere boundary
  let toCenter = uv - SPHERE_CENTER;
  let dist = length(toCenter);
  if (dist > SPHERE_RADIUS) {
    textureStore(dst, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }

  let vel = textureLoad(velTex, id.xy, 0).xy;
  let backUV = uv - p.dt * vel * p.dx;
  let clamped = clamp(backUV, vec2f(0.5 / p.simRes), vec2f(1.0 - 0.5 / p.simRes));
  let advected = textureSampleLevel(dyeSrc, sampl, clamped, 0.0);
  // Ratio-preserving cap: scale all channels proportionally to keep hue intact
  var dye = advected.rgb * p.dyeDissipation;
  let maxC = max(dye.r, max(dye.g, dye.b));
  if (maxC > 2.0) {
    dye *= 2.0 / maxC;
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
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
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
const SPHERE_RADIUS = 0.43;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let res = u32(np.simRes);
  if (id.x >= res || id.y >= res) { return; }
  let uv = (vec2f(id.xy) + 0.5) / np.simRes;

  if (length(uv - SPHERE_CENTER) > SPHERE_RADIUS) {
    textureStore(velDst, id.xy, textureLoad(velSrc, id.xy, 0));
    return;
  }

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

  vel += curl * np.amount * 8.0;
  textureStore(velDst, id.xy, vec4f(vel, 0.0, 1.0));
}
`;

// ─── Particle Update Compute Shader (function — count baked in) ─────────────
function makeParticleUpdateShader(count) {
  return /* wgsl */`
${commonHeader}
@group(0) @binding(1) var velTex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

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
    4.0767416621 * l*l*l - 3.3077115913 * m*m*m + 0.2309699292 * s*s*s,
   -1.2684380046 * l*l*l + 2.6097574011 * m*m*m - 0.3413193965 * s*s*s,
   -0.0041960863 * l*l*l - 0.7034186147 * m*m*m + 1.7076147010 * s*s*s
  ), vec3f(0.0));
}

fn grainTint(h: f32) -> vec3f {
  if (h < 0.6) { return mix(vec3f(1.0, 0.75, 0.3), vec3f(1.0, 0.85, 0.45), h / 0.6); }
  if (h < 0.75) { return vec3f(1.0, 0.6, 0.25); }
  if (h < 0.85) { return vec3f(1.0, 0.88, 0.55); }
  if (h < 0.92) { return vec3f(0.85, 0.92, 1.0); }
  if (h < 0.96) { return vec3f(1.0, 0.75, 0.88); }
  return vec3f(0.75, 1.0, 0.82);
}

const SPHERE_CENTER = vec2f(0.5, 0.5);
const SPHERE_RADIUS: f32 = 0.43;

@compute @workgroup_size(${PARTICLE_WG})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  if (idx >= ${count}u) { return; }

  var part = particles[idx];

  // Sample velocity at particle position
  let vel = textureSampleLevel(velTex, samp, vec2f(part.posX, part.posY), 0.0).xy;

  // Advect particle with fluid
  let dt = p.dt;
  part.posX += vel.x * dt * p.dx;
  part.posY += vel.y * dt * p.dx;

  // Velocity gradient for 3D tumble (forward difference — 2 samples instead of 4)
  let eps = 2.0 / p.simRes;
  let pos = vec2f(part.posX, part.posY);
  let vR = textureSampleLevel(velTex, samp, pos + vec2f(eps, 0.0), 0.0).xy;
  let vT = textureSampleLevel(velTex, samp, pos + vec2f(0.0, eps), 0.0).xy;

  let curl = ((vR.y - vel.y) - (vT.x - vel.x)) * 2.0;
  let shearX = vR.x - vel.x;
  let shearY = vT.y - vel.y;

  // Update angular velocity with curl (damped for slower, smoother shimmer)
  part.angularVel = part.angularVel * 0.92 + curl * 1.0;

  // Rotate normal around Z-axis (spin)
  let spinAngle = part.angularVel * dt;
  let cosA = cos(spinAngle);
  let sinA = sin(spinAngle);
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
  let prismatic = vec3f(
    0.5 + 0.5 * cos(hueAngle),
    0.5 + 0.5 * cos(hueAngle + 2.094),
    0.5 + 0.5 * cos(hueAngle + 4.189)
  );
  let prisThreshold = max(1.0 - pa * 0.05, 0.0);
  let prisMix = clamp(pa * 0.06, 0.0, 1.0);
  if (ch4 > prisThreshold) {
    tint = mix(tint, prismatic, prisMix);
  }

  // Glitter color: blend base→accent in Oklab (pre-converted on CPU)
  let okBase = pp.extra.yzw;
  let okAccent = pp.extra2.xyz;
  let blend = pp.extra2.w;
  let gLo = mix(0.0, 0.35, blend);
  let gHi = mix(1.0, 0.4, blend);
  let densityT = smoothstep(gLo, gHi, intensity);
  let glitCol = oklabToLinear(mix(okAccent, okBase, densityT));
  tint *= glitCol;

  let brightness = glint * pp.screen.w + ambient;
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
  let col = colors[ii];

  var out: VSOut;

  // Early cull: pre-computed alpha <= 0
  if (col.a <= 0.0) {
    out.pos = vec4f(2.0, 2.0, 0.0, 1.0);
    out.color = vec3f(0.0);
    out.alpha = 0.0;
    out.localUV = vec2f(0.0);
    return out;
  }

  let part = particles[ii];

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
  if (length(centered) > 0.42) { discard; }

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

const displayShaderFrag = /* wgsl */`
@group(0) @binding(0) var dyeTex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct DisplayUniforms {
  screen: vec4f,      // xy=screenSize, z=time, w=sheenStrength
  baseColor: vec4f,   // xyz=baseColor RGB
  accentColor: vec4f, // xyz=accentColor RGB, w=colorBlend
  sheenColor: vec4f,  // xyz=sheenColor RGB
};
@group(0) @binding(2) var<uniform> du: DisplayUniforms;

fn aces(x: vec3f) -> vec3f {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

fn oklabToLinear(lab: vec3f) -> vec3f {
  let l = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  let m = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  let s = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
  return max(vec3f(
    4.0767416621 * l*l*l - 3.3077115913 * m*m*m + 0.2309699292 * s*s*s,
   -1.2684380046 * l*l*l + 2.6097574011 * m*m*m - 0.3413193965 * s*s*s,
   -0.0041960863 * l*l*l - 0.7034186147 * m*m*m + 1.7076147010 * s*s*s
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
  let screenRadius = 0.42;
  let mask = 1.0 - smoothstep(screenRadius - 0.005, screenRadius + 0.005, screenDist);

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
  let fluidCol = oklabToLinear(mix(okAccent, okBase, densityT));
  var color = fluidCol * intensity * 0.25;

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
  let sharpSheen = pow(spec, 8.0) * smoothstep(0.003, 0.04, gradLen);
  // Broad soft metallic glow
  let broadSpec = pow(spec, 2.0) * smoothstep(0.002, 0.08, gradLen);
  // Fresnel-like rim sheen (edges of sphere glow)
  let rimFactor = smoothstep(0.2, 0.42, screenDist);

  let sheen = (sharpSheen * 0.6 + broadSpec * 0.3 + rimFactor * 0.15) * sheenStrength;
  color += color * sheen * du.sheenColor.rgb;

  // Tone mapping
  color = aces(color * 1.6);

  // Gamma
  color = pow(clamp(color, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));

  // Circular mask
  color *= mask;

  return vec4f(color, 1.0);
}
`;

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

// ─── WebGPU Init ─────────────────────────────────────────────────────────────

async function main() {
  const canvas = document.getElementById('canvas');
  const errorDiv = document.getElementById('error');

  if (!navigator.gpu) {
    errorDiv.style.display = 'block';
    errorDiv.textContent = 'WebGPU not supported in this browser.';
    console.error('WebGPU not supported');
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    errorDiv.style.display = 'block';
    errorDiv.textContent = 'No WebGPU adapter found.';
    console.error('No WebGPU adapter');
    return;
  }

  // Request higher device limits for large particle buffers
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1024 * 1024 * 1024),
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1024 * 1024 * 1024),
    },
  });
  device.lost.then(info => console.error('WebGPU device lost:', info.message));

  // Derive max particle count from granted limits
  const maxParticles = Math.min(16777216, Math.floor(device.limits.maxStorageBufferBindingSize / PARTICLE_STRIDE));
  state.particleCount = Math.min(state.particleCount, maxParticles);
  console.log(`GPU limits: maxStorageBuffer=${(device.limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)}MB, maxParticles=${maxParticles}`);

  const ctx = canvas.getContext('webgpu');
  const canvasFmt = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: canvasFmt, alphaMode: 'opaque' });

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

  // displayUB: [width, height, time, sheenStrength, baseR, baseG, baseB, pad, accentR, accentG, accentB, colorBlend, sheenR, sheenG, sheenB, pad]
  const displayUB = device.createBuffer({
    size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const displayUBData = new Float32Array(16);

  // ─── Pipeline helpers ───────────────────────────────────────────────────
  function buildPipeline(code, label, bindingDescs) {
    const module = device.createShaderModule({ code, label });
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

  // Curl: uniform, texture(vel), storage(curl)
  const curlPipe = buildPipeline(curlShader, 'curl',
    ['uniform', 'texture', 'storage']);

  // Vorticity: uniform, texture(vel), texture(curl), storage(dst)
  const vortPipe = buildPipeline(vorticityShader, 'vorticity',
    ['uniform', 'texture', 'texture', 'storage']);

  // Divergence: uniform, texture(vel), storage(div)
  const divPipe = buildPipeline(divergenceShader, 'divergence',
    ['uniform', 'texture', 'storage']);

  // Clear pressure: uniform, texture(src), storage(dst)
  const clearPressPipe = buildPipeline(clearPressureShader, 'clearPressure',
    ['uniform', 'texture', 'storage']);

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
      module: device.createShaderModule({ code: displayShaderFrag }),
      entryPoint: 'main',
      targets: [{ format: canvasFmt }],
    },
    primitive: { topology: 'triangle-list' },
  });

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
    ],
  });

  let particleUpdatePipeline = device.createComputePipeline({
    label: 'particleUpdate',
    layout: device.createPipelineLayout({ bindGroupLayouts: [particleUpdateBGL] }),
    compute: {
      module: device.createShaderModule({ code: makeParticleUpdateShader(state.particleCount), label: 'particleUpdate' }),
      entryPoint: 'main',
    },
  });

  let particleUpdateBG = device.createBindGroup({
    layout: particleUpdateBGL,
    entries: [
      { binding: 0, resource: { buffer: paramBuf } },
      { binding: 1, resource: tview(velA) },
      { binding: 2, resource: linearSampler },
      { binding: 3, resource: { buffer: particleBuf } },
      { binding: 4, resource: tview(dyeA) },
      { binding: 5, resource: { buffer: particleUB } },
      { binding: 6, resource: { buffer: colorBuf } },
    ],
  });

  // Particle render pipeline
  const particleRenderBGL = device.createBindGroupLayout({
    label: 'particleRender_bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
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
    ],
  });

  // ─── Pre-allocated simulation bind groups (persistent textures) ────────
  const curlBGFixed = bg(curlPipe.layout, [ubuf(paramBuf), tview(velA), tview(curlTex)]);
  const vortBGFixed = bg(vortPipe.layout, [ubuf(paramBuf), tview(velA), tview(curlTex), tview(velB)]);
  const divBGFixed = bg(divPipe.layout, [ubuf(paramBuf), tview(velA), tview(divTex)]);
  const clearPressBGFixed = bg(clearPressPipe.layout, [ubuf(paramBuf), tview(pressA), tview(pressB)]);
  const jacobiBG_AtoB = bg(jacobiPipe.layout, [ubuf(paramBuf), tview(pressA), tview(divTex), tview(pressB)]);
  const jacobiBG_BtoA = bg(jacobiPipe.layout, [ubuf(paramBuf), tview(pressB), tview(divTex), tview(pressA)]);
  const gradSubBGFixed = bg(gradSubPipe.layout, [ubuf(paramBuf), tview(velA), tview(pressA), tview(velB)]);
  const advVelBGFixed = bg(advectVelPipe.layout, [ubuf(paramBuf), tview(velA), linearSampler, tview(velB)]);
  const advDyeBGFixed = bg(advectDyePipe.layout, [ubuf(paramBuf), tview(velA), tview(dyeA), linearSampler, tview(dyeB)]);
  const curlNoiseBGFixed = bg(curlNoisePipe.layout, [ubuf(noiseBuf), tview(velA), tview(velB)]);
  const displayBGFixed = bg(displayBGL, [tview(dyeA), linearSampler, ubuf(displayUB)]);
  // Batch splat GPU resources
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

  const batchSplatVelBG = bg(batchSplatBGL,
    [ubuf(paramBuf), tview(velA), tview(velB), ubuf(splatBuf), ubuf(splatCountBuf)]);
  const batchSplatDyeBG = bg(batchSplatBGL,
    [ubuf(paramBuf), tview(dyeA), tview(dyeB), ubuf(splatBuf), ubuf(splatCountBuf)]);

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
        module: device.createShaderModule({ code: makeParticleUpdateShader(count), label: 'particleUpdate' }),
        entryPoint: 'main',
      },
    });

    particleUpdateBG = device.createBindGroup({
      layout: particleUpdateBGL,
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: tview(velA) },
        { binding: 2, resource: linearSampler },
        { binding: 3, resource: { buffer: newBuf } },
        { binding: 4, resource: tview(dyeA) },
        { binding: 5, resource: { buffer: particleUB } },
        { binding: 6, resource: { buffer: newColorBuf } },
      ],
    });

    particleRenderBG = device.createBindGroup({
      layout: particleRenderBGL,
      entries: [
        { binding: 0, resource: { buffer: newBuf } },
        { binding: 1, resource: { buffer: particleUB } },
        { binding: 2, resource: { buffer: newColorBuf } },
      ],
    });

    particleBuf = newBuf;
    colorBuf = newColorBuf;
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
  canvas.addEventListener('pointerup', () => { pointer.down = false; });
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

  // ─── Auto-injectors ────────────────────────────────────────────────────
  const injectors = [
    { phase: 0.0, radius: 0.25, speed: 0.7, colorOffset: 0.0, cmIndex: 4 },
    { phase: 1.57, radius: 0.18, speed: -0.9, colorOffset: 0.08, cmIndex: 4 },
    { phase: 3.14, radius: 0.3, speed: 0.5, colorOffset: 0.15, cmIndex: 4 },
    { phase: 4.71, radius: 0.22, speed: -0.6, colorOffset: 0.22, cmIndex: 4 },
  ];

  let time = 0;
  let lastSplatTime = -1.0;

  // ─── Frame loop ─────────────────────────────────────────────────────────
  function frame() {
    requestAnimationFrame(frame);
    const dt = 0.016;
    time += dt;

    // Pre-convert colors to Oklab on CPU (avoids per-particle GPU conversion)
    const okGlitBase = linearToOklabCPU(state.glitterColor);
    const okGlitAccent = linearToOklabCPU(state.glitterAccent);
    const okBaseCol = linearToOklabCPU(state.baseColor);
    const okAccentCol = linearToOklabCPU(state.accentColor);

    // Update screen/particle/display uniform buffers
    particleUBData[0] = canvas.width;
    particleUBData[1] = canvas.height;
    particleUBData[2] = state.particleSize;
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
    device.queue.writeBuffer(particleUB, 0, particleUBData);

    displayUBData[0] = canvas.width;
    displayUBData[1] = canvas.height;
    displayUBData[2] = time;
    displayUBData[3] = state.sheenStrength;
    displayUBData[4] = okBaseCol[0];
    displayUBData[5] = okBaseCol[1];
    displayUBData[6] = okBaseCol[2];
    displayUBData[8] = okAccentCol[0];
    displayUBData[9] = okAccentCol[1];
    displayUBData[10] = okAccentCol[2];
    displayUBData[11] = state.colorBlend;
    displayUBData[12] = state.sheenColor[0];
    displayUBData[13] = state.sheenColor[1];
    displayUBData[14] = state.sheenColor[2];
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
      const injSplatRadius = state.splatRadius * 2.0 * (0.5 + state.injectorSize * 3.0);
      const spdMul = 0.2 + state.injectorSpeed * 3.6;
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
    device.queue.writeBuffer(splatCountBuf, 0, splatCountUData);
    if (splatCount > 0) {
      device.queue.writeBuffer(splatBuf, 0, splatArrayData);
    }
    writeParams({ dt, time });

    // ── Single encoder for all passes ──
    const enc = device.createCommandEncoder();

    // ── Batch splat pass (1 vel dispatch + 1 dye dispatch instead of ~20) ──
    if (splatCount > 0) {
      const p1 = enc.beginComputePass();
      p1.setPipeline(batchSplatVelPipe);
      p1.setBindGroup(0, batchSplatVelBG);
      p1.dispatchWorkgroups(dispatch, dispatch);
      p1.end();
      enc.copyTextureToTexture(
        { texture: velB }, { texture: velA }, [SIM_RES, SIM_RES]);

      const p2 = enc.beginComputePass();
      p2.setPipeline(batchSplatDyePipe);
      p2.setBindGroup(0, batchSplatDyeBG);
      p2.dispatchWorkgroups(dispatch, dispatch);
      p2.end();
      enc.copyTextureToTexture(
        { texture: dyeB }, { texture: dyeA }, [SIM_RES, SIM_RES]);
    }

    // ── Pass 2: Curl ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(curlPipe.pipeline);
      p.setBindGroup(0, curlBGFixed);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
    }

    // ── Pass 3: Vorticity Confinement ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(vortPipe.pipeline);
      p.setBindGroup(0, vortBGFixed);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      enc.copyTextureToTexture(
        { texture: velB }, { texture: velA }, [SIM_RES, SIM_RES]);
    }

    // ── Pass 4: Divergence ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(divPipe.pipeline);
      p.setBindGroup(0, divBGFixed);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
    }

    // ── Pass 5: Clear Pressure ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(clearPressPipe.pipeline);
      p.setBindGroup(0, clearPressBGFixed);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      enc.copyTextureToTexture(
        { texture: pressB }, { texture: pressA }, [SIM_RES, SIM_RES]);
    }

    // ── Pass 6: Jacobi Pressure Solve ──
    for (let i = 0; i < state.pressureIters; i++) {
      const jBG = (i % 2 === 0) ? jacobiBG_AtoB : jacobiBG_BtoA;
      const p = enc.beginComputePass();
      p.setPipeline(jacobiPipe.pipeline);
      p.setBindGroup(0, jBG);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
    }
    // After even iterations, result is in pressA; after odd, in pressB
    if (state.pressureIters % 2 !== 0) {
      enc.copyTextureToTexture(
        { texture: pressB }, { texture: pressA }, [SIM_RES, SIM_RES]);
    }

    // ── Pass 7: Gradient Subtraction ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(gradSubPipe.pipeline);
      p.setBindGroup(0, gradSubBGFixed);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      enc.copyTextureToTexture(
        { texture: velB }, { texture: velA }, [SIM_RES, SIM_RES]);
    }

    // ── Pass 8: Advect Velocity ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(advectVelPipe.pipeline);
      p.setBindGroup(0, advVelBGFixed);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      enc.copyTextureToTexture(
        { texture: velB }, { texture: velA }, [SIM_RES, SIM_RES]);
    }

    // ── Pass 9: Advect Dye ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(advectDyePipe.pipeline);
      p.setBindGroup(0, advDyeBGFixed);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      enc.copyTextureToTexture(
        { texture: dyeB }, { texture: dyeA }, [SIM_RES, SIM_RES]);
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
      p.setBindGroup(0, curlNoiseBGFixed);
      p.dispatchWorkgroups(dispatch, dispatch);
      p.end();
      enc.copyTextureToTexture(
        { texture: velB }, { texture: velA }, [SIM_RES, SIM_RES]);
    }

    // ── Pass 10: Particle Update (compute) ──
    {
      const p = enc.beginComputePass();
      p.setPipeline(particleUpdatePipeline);
      p.setBindGroup(0, particleUpdateBG);
      p.dispatchWorkgroups(particleDispatches);
      p.end();
    }

    // ── Pass 11: Display Render (fluid base) ──
    const canvasView = ctx.getCurrentTexture().createView();
    {
      const rp = enc.beginRenderPass({
        colorAttachments: [{
          view: canvasView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      rp.setPipeline(displayPipeline);
      rp.setBindGroup(0, displayBGFixed);
      rp.draw(3);
      rp.end();
    }

    // ── Pass 12: Particle Render (additive glitter layer) ──
    {
      const rp = enc.beginRenderPass({
        colorAttachments: [{
          view: canvasView,
          loadOp: 'load',
          storeOp: 'store',
        }],
      });
      rp.setPipeline(particleRenderPipeline);
      rp.setBindGroup(0, particleRenderBG);
      rp.draw(4, state.particleCount);
      rp.end();
    }

    device.queue.submit([enc.finish()]);
  }

  // ─── Wire Settings UI ──────────────────────────────────────────────────
  const settingsToggle = document.getElementById('settingsToggle');
  const settingsPanel = document.getElementById('settingsPanel');
  settingsToggle.addEventListener('click', () => {
    settingsPanel.classList.toggle('open');
  });

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
  wireSlider('glintBrightness', 'glintBrightness');
  wireSlider('sizeRandomness', 'sizeRandomness');
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
  wireSlider('colorBlend', 'colorBlend', v => v.toFixed(2));

  wireSlider('sheenStrength', 'sheenStrength');
  wireColor('sheenColor', 'sheenColor');
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

  console.log('Fluid simulation starting...');
  requestAnimationFrame(frame);
}

main().catch(err => {
  console.error('Fatal:', err);
  const errorDiv = document.getElementById('error');
  if (errorDiv) {
    errorDiv.style.display = 'block';
    errorDiv.textContent = err.message;
  }
});
