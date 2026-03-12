// ─── GLSL ES 3.00 Shader Sources for WebGL2 Fluid Simulation Platformer ─────
// All shaders use #version 300 es. Fragment shaders use precision highp float.

// ─── Shared Fullscreen Triangle Vertex Shader ────────────────────────────────
export const fullscreenVS = `#version 300 es
void main() {
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  gl_Position = vec4(x, y, 0, 1);
}
`;

// ─── 1. Splat Fragment Shader ────────────────────────────────────────────────
export const splatFS = `#version 300 es
precision highp float;

uniform sampler2D uTarget;
uniform vec2 uPoint;
uniform vec3 uColor;
uniform float uRadius;
uniform float uAspect;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(uTarget, 0));
  vec2 d = uv - uPoint;
  d.x *= uAspect;
  float w = exp(-dot(d, d) / (uRadius * uRadius));
  vec3 existing = texture(uTarget, uv).rgb;
  fragColor = vec4(existing + w * uColor, 1.0);
}
`;

// ─── 2. Curl (Vorticity) Fragment Shader ─────────────────────────────────────
export const curlFS = `#version 300 es
precision highp float;

uniform sampler2D uVelocity;
uniform vec2 uTexelSize;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy * uTexelSize;
  vec2 vL = texture(uVelocity, uv - vec2(uTexelSize.x, 0.0)).xy;
  vec2 vR = texture(uVelocity, uv + vec2(uTexelSize.x, 0.0)).xy;
  vec2 vB = texture(uVelocity, uv - vec2(0.0, uTexelSize.y)).xy;
  vec2 vT = texture(uVelocity, uv + vec2(0.0, uTexelSize.y)).xy;
  float curl = (vR.y - vL.y) - (vT.x - vB.x);
  fragColor = vec4(curl * 0.5, 0.0, 0.0, 1.0);
}
`;

// ─── 3. Vorticity Confinement Fragment Shader ────────────────────────────────
export const vorticityFS = `#version 300 es
precision highp float;

uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform vec2 uTexelSize;
uniform float uStrength;
uniform float uDt;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy * uTexelSize;
  float cL = texture(uCurl, uv - vec2(uTexelSize.x, 0.0)).x;
  float cR = texture(uCurl, uv + vec2(uTexelSize.x, 0.0)).x;
  float cB = texture(uCurl, uv - vec2(0.0, uTexelSize.y)).x;
  float cT = texture(uCurl, uv + vec2(0.0, uTexelSize.y)).x;
  float cC = texture(uCurl, uv).x;

  vec2 N = vec2(abs(cR) - abs(cL), abs(cT) - abs(cB));
  float lenN = length(N);
  if (lenN > 1e-5) {
    N /= lenN;
  }
  // cross(N, curl) in 2D: force = strength * vec2(N.y, -N.x) * curl
  vec2 force = uStrength * vec2(N.y, -N.x) * cC * uDt;
  vec2 vel = texture(uVelocity, uv).xy;
  fragColor = vec4(vel + force, 0.0, 1.0);
}
`;

// ─── 4. Divergence Fragment Shader (boundary-aware) ──────────────────────────
export const divergenceFS = `#version 300 es
precision highp float;

uniform sampler2D uVelocity;
uniform sampler2D uBoundary;
uniform vec2 uTexelSize;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy * uTexelSize;

  float bL = texture(uBoundary, uv - vec2(uTexelSize.x, 0.0)).x;
  float bR = texture(uBoundary, uv + vec2(uTexelSize.x, 0.0)).x;
  float bB = texture(uBoundary, uv - vec2(0.0, uTexelSize.y)).x;
  float bT = texture(uBoundary, uv + vec2(0.0, uTexelSize.y)).x;

  vec2 vL = bL > 0.5 ? vec2(0.0) : texture(uVelocity, uv - vec2(uTexelSize.x, 0.0)).xy;
  vec2 vR = bR > 0.5 ? vec2(0.0) : texture(uVelocity, uv + vec2(uTexelSize.x, 0.0)).xy;
  vec2 vB = bB > 0.5 ? vec2(0.0) : texture(uVelocity, uv - vec2(0.0, uTexelSize.y)).xy;
  vec2 vT = bT > 0.5 ? vec2(0.0) : texture(uVelocity, uv + vec2(0.0, uTexelSize.y)).xy;

  float div = 0.5 * ((vR.x - vL.x) + (vT.y - vB.y));
  fragColor = vec4(div, 0.0, 0.0, 1.0);
}
`;

// ─── 5. Pressure Jacobi Iteration Fragment Shader (boundary-aware) ───────────
export const pressureFS = `#version 300 es
precision highp float;

uniform sampler2D uPressure;
uniform sampler2D uDivergence;
uniform sampler2D uBoundary;
uniform vec2 uTexelSize;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy * uTexelSize;
  float pC = texture(uPressure, uv).x;

  float bL = texture(uBoundary, uv - vec2(uTexelSize.x, 0.0)).x;
  float bR = texture(uBoundary, uv + vec2(uTexelSize.x, 0.0)).x;
  float bB = texture(uBoundary, uv - vec2(0.0, uTexelSize.y)).x;
  float bT = texture(uBoundary, uv + vec2(0.0, uTexelSize.y)).x;

  float pL = bL > 0.5 ? pC : texture(uPressure, uv - vec2(uTexelSize.x, 0.0)).x;
  float pR = bR > 0.5 ? pC : texture(uPressure, uv + vec2(uTexelSize.x, 0.0)).x;
  float pB = bB > 0.5 ? pC : texture(uPressure, uv - vec2(0.0, uTexelSize.y)).x;
  float pT = bT > 0.5 ? pC : texture(uPressure, uv + vec2(0.0, uTexelSize.y)).x;

  float div = texture(uDivergence, uv).x;
  float p = (pL + pR + pB + pT - div) * 0.25;
  fragColor = vec4(p, 0.0, 0.0, 1.0);
}
`;

// ─── 6. Gradient Subtraction Fragment Shader (boundary-aware) ────────────────
export const gradSubFS = `#version 300 es
precision highp float;

uniform sampler2D uPressure;
uniform sampler2D uVelocity;
uniform sampler2D uBoundary;
uniform vec2 uTexelSize;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy * uTexelSize;

  float pL = texture(uPressure, uv - vec2(uTexelSize.x, 0.0)).x;
  float pR = texture(uPressure, uv + vec2(uTexelSize.x, 0.0)).x;
  float pB = texture(uPressure, uv - vec2(0.0, uTexelSize.y)).x;
  float pT = texture(uPressure, uv + vec2(0.0, uTexelSize.y)).x;

  vec2 grad = vec2(pR - pL, pT - pB) * 0.5;
  vec2 vel = texture(uVelocity, uv).xy;
  vel -= grad;

  float bC = texture(uBoundary, uv).x;
  if (bC > 0.5) {
    vel = vec2(0.0);
  }

  fragColor = vec4(vel, 0.0, 1.0);
}
`;

// ─── 7. Advection Fragment Shader (Semi-Lagrangian + MacCormack) ─────────────
export const advectFS = `#version 300 es
precision highp float;

uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform sampler2D uBoundary;
uniform vec2 uTexelSize;
uniform float uDt;
uniform float uDissipation;
uniform float uMaccormack;
uniform float uGravity;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy * uTexelSize;

  // Check if center is solid
  float bC = texture(uBoundary, uv).x;
  if (bC > 0.5) {
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  vec2 vel = texture(uVelocity, uv).xy;

  // Backtrace
  vec2 backPos = uv - uDt * vel * uTexelSize;

  vec4 result = texture(uSource, backPos);

  if (uMaccormack > 0.0) {
    // Forward trace result
    vec4 phi_hat_n1 = result;
    // Backward trace from forward result
    vec2 velBack = texture(uVelocity, backPos).xy;
    vec2 fwdPos = backPos + uDt * velBack * uTexelSize;
    vec4 phi_hat_n = texture(uSource, fwdPos);

    // Original value
    vec4 phi_n = texture(uSource, uv);

    // Error correction
    vec4 correction = (phi_n - phi_hat_n) * 0.5;

    // Clamp to neighbor min/max
    vec4 minVal = phi_hat_n1;
    vec4 maxVal = phi_hat_n1;
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        if (i == 0 && j == 0) continue;
        vec4 s = texture(uSource, backPos + vec2(float(i), float(j)) * uTexelSize);
        minVal = min(minVal, s);
        maxVal = max(maxVal, s);
      }
    }

    vec4 corrected = phi_hat_n1 + correction;
    corrected = clamp(corrected, minVal, maxVal);
    result = mix(phi_hat_n1, corrected, uMaccormack);
  }

  // Apply dissipation
  result *= uDissipation;

  // Add gravity to velocity Y component (when self-advecting velocity)
  // The caller sets uGravity > 0 only when advecting velocity
  result.y -= uGravity;

  fragColor = vec4(result.rgb, 1.0);
}
`;

// ─── 8. Buoyancy Fragment Shader ─────────────────────────────────────────────
export const buoyancyFS = `#version 300 es
precision highp float;

uniform sampler2D uVelocity;
uniform sampler2D uTemperature;
uniform float uBuoyancy;
uniform float uDt;
uniform float uRadialMix;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(uVelocity, 0));
  vec2 vel = texture(uVelocity, uv).xy;
  float temp = texture(uTemperature, uv).x;

  vec2 force = vec2(0.0);
  force.y += temp * uBuoyancy * uDt;

  if (uRadialMix > 0.0) {
    vec2 dir = uv - vec2(0.5);
    float r = length(dir);
    if (r > 1e-5) {
      dir /= r;
      force += dir * temp * uBuoyancy * uDt * uRadialMix;
    }
  }

  fragColor = vec4(vel + force, 0.0, 1.0);
}
`;

// ─── 9. Curl Noise Fragment Shader (8 noise types) ──────────────────────────
export const curlNoiseFS = `#version 300 es
precision highp float;

uniform sampler2D uVelocity;
uniform vec2 uTexelSize;
uniform float uTime;
uniform float uAmount;
uniform int uNoiseType;
uniform int uMapping;
uniform int uBehavior;
uniform float uFrequency;
uniform float uSpeed;
uniform float uWarp;
uniform float uSharpness;
uniform float uAnisotropy;
uniform float uBlend;

out vec4 fragColor;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;

// ── Hash function ──
float hash(vec2 p) {
  uint n = floatBitsToUint(p.x * 122.1 + p.y * 341.3);
  n = (n << 13u) ^ n;
  n = n * 0x45d9f3bu;
  n = n ^ (n >> 16u);
  return float(n & 0xFFFFu) / 65535.0;
}

// ── Value noise ──
float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
    mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
    u.y
  );
}

// ── 2D rotation ──
vec2 rot2(vec2 p, float a) {
  float c = cos(a), s = sin(a);
  return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

// ── Worley noise (lite: 4 cell offsets) ──
float worleyLite(vec2 p) {
  vec2 c = floor(p);
  vec2 f = fract(p);
  vec2 o0 = vec2(0.0, 0.0);
  vec2 o1 = vec2(1.0, 0.0);
  vec2 o2 = vec2(0.0, 1.0);
  vec2 o3 = vec2(1.0, 1.0);
  vec2 p0 = o0 + vec2(hash(c + o0), hash(c + o0 + vec2(17.0, 59.0))) - f;
  vec2 p1 = o1 + vec2(hash(c + o1), hash(c + o1 + vec2(17.0, 59.0))) - f;
  vec2 p2 = o2 + vec2(hash(c + o2), hash(c + o2 + vec2(17.0, 59.0))) - f;
  vec2 p3 = o3 + vec2(hash(c + o3), hash(c + o3 + vec2(17.0, 59.0))) - f;
  float d = min(min(length(p0), length(p1)), min(length(p2), length(p3)));
  return 1.0 - clamp(d, 0.0, 1.0);
}

// ── FBM 4 octaves ──
float fbm4(vec2 pIn, float t) {
  float n1 = vnoise(pIn + vec2(t * 0.31, t * 0.19));
  float n2 = vnoise(rot2(pIn * 2.03, 1.11) + vec2(-t * 0.47, t * 0.36));
  float n3 = vnoise(rot2(pIn * 4.02, -0.74) + vec2(t * 0.63, -t * 0.41));
  float n4 = vnoise(rot2(pIn * 8.01, 0.41) + vec2(-t * 0.85, t * 0.77));
  return n1 * 0.5 + n2 * 0.27 + n3 * 0.15 + n4 * 0.08;
}

// ── Noise mapping (radial outflow) ──
vec2 applyNoiseMapping(vec2 uvRaw, float t) {
  if (uMapping < 1) {
    return uvRaw;
  }
  vec2 p = uvRaw - vec2(0.5);
  float r = length(p) / 0.3665;
  float a = atan(p.y, p.x);
  float a01 = (a + PI) / TAU;
  float radialScroll = r * (3.2 + uFrequency * 2.6) - t * (0.45 + uSpeed * 1.8);
  float twist = (vnoise(vec2(a01 * 8.0, r * 4.0 + t * 0.2)) - 0.5) * (0.35 + uWarp * 0.75);
  return vec2(a01 * (1.0 + uAnisotropy * 3.0), radialScroll + twist);
}

// ── Symmetry frame ──
struct SymmetryFrame {
  vec2 uv;
  vec2 basisX;
  vec2 basisY;
  float handedness;
};

SymmetryFrame makeSymmetryFrame(vec2 uvRaw) {
  SymmetryFrame frame;
  frame.uv = uvRaw;
  frame.basisX = vec2(1.0, 0.0);
  frame.basisY = vec2(0.0, 1.0);
  frame.handedness = 1.0;

  if (uBehavior < 1) {
    return frame;
  }

  vec2 p = uvRaw - vec2(0.5);
  if (uBehavior == 1) {
    float sx = p.x >= 0.0 ? 1.0 : -1.0;
    frame.uv = vec2(0.5) + vec2(abs(p.x), p.y);
    frame.basisX = vec2(sx, 0.0);
    frame.basisY = vec2(0.0, 1.0);
    frame.handedness = sx;
    return frame;
  }

  // N-fold rotational symmetry
  float folds = max(2.0, min(8.0, float(uBehavior)));
  float sector = TAU / folds;
  float theta = atan(p.y, p.x);
  float k = floor(theta / sector + 0.5);
  float rotToCanonical = -k * sector;

  vec2 q = rot2(p, rotToCanonical);
  float reflectY = 1.0;
  if (q.y < 0.0) {
    q.y = -q.y;
    reflectY = -1.0;
  }
  frame.uv = vec2(0.5) + q;

  float rotBack = -rotToCanonical;
  float c2 = cos(rotBack);
  float s2 = sin(rotBack);
  frame.basisX = vec2(c2, s2);
  frame.basisY = vec2(-s2 * reflectY, c2 * reflectY);
  frame.handedness = reflectY;
  return frame;
}

// ── 0: Classic Curl ──
float noiseClassicCurl(vec2 uvRaw, float t) {
  vec2 uv = applyNoiseMapping(uvRaw, t);
  vec2 p = uv * (2.0 + uFrequency * 13.0);
  vec2 warp = (vec2(
    vnoise(p * 0.82 + vec2(t * 0.17, -t * 0.13)),
    vnoise(p * 0.82 + vec2(-t * 0.11, t * 0.19))
  ) * 2.0 - 1.0) * (0.2 + uWarp * 1.8);
  vec2 q = p + warp;
  float base = vnoise(q + vec2(t * 0.28, t * 0.18));
  float octave2 = vnoise(rot2(q * 2.08, 0.45 + uAnisotropy * 1.2) + vec2(-t * 0.43, t * 0.29));
  float octave3 = vnoise(rot2(q * 4.01, -0.75 - uAnisotropy * 0.6) + vec2(t * 0.67, -t * 0.38));
  float micro = (vnoise(q * 6.4 + vec2(-t * 0.75, t * 0.61)) - 0.5) * 2.0;
  float s = base * (0.55 - uSharpness * 0.25) + octave2 * 0.3 + octave3 * (0.15 + uSharpness * 0.22);
  s += micro * (0.07 + uBlend * 0.16);
  return s;
}

// ── 1: Domain-Warped Curl ──
float noiseDomainWarped(vec2 uvRaw, float t) {
  vec2 uv = applyNoiseMapping(uvRaw, t);
  vec2 p = uv * (2.4 + uFrequency * 11.5);
  vec2 warp1 = (vec2(
    vnoise(p * 0.58 + vec2(t * 0.24, -t * 0.17)),
    vnoise(p * 0.58 + vec2(-t * 0.22, t * 0.21))
  ) * 2.0 - 1.0) * (0.5 + uWarp * 3.3);
  vec2 p1 = p + warp1;
  vec2 warp2 = (vec2(
    vnoise(rot2(p1 * 1.62, 0.91) + vec2(-t * 0.36, t * 0.27)),
    vnoise(rot2(p1 * 1.62, -0.53) + vec2(t * 0.29, -t * 0.31))
  ) * 2.0 - 1.0) * (0.2 + uSharpness * 2.6);
  vec2 q = p1 + warp2;
  float base = fbm4(q, t);
  float ridged = 1.0 - abs(base * 2.0 - 1.0);
  float ribbons = 0.5 + 0.5 * sin((q.x + q.y * 0.6) * (4.0 + uAnisotropy * 9.0) + t * (0.8 + uSpeed * 2.4));
  float s = mix(base, ridged, 0.2 + uBlend * 0.5);
  s = mix(s, ribbons, 0.12 + uSharpness * 0.38);
  return s;
}

// ── 2: Ridged Fractal ──
float noiseRidged(vec2 uvRaw, float t) {
  vec2 uv = applyNoiseMapping(uvRaw, t);
  vec2 p = uv * (2.8 + uFrequency * 16.0);
  float base = fbm4(p, t);
  float ridge0 = 1.0 - abs(base * 2.0 - 1.0);
  float ridge1 = 1.0 - abs(vnoise(rot2(p * 2.7, 1.2) + vec2(t * 0.32, -t * 0.27)) * 2.0 - 1.0);
  float spine = pow(max(ridge0, 0.0), 1.1 + uSharpness * 3.2);
  float spikes = pow(max(ridge1, 0.0), 1.8 + uWarp * 2.5);
  float s = spine * (0.65 + uBlend * 0.45) + spikes * 0.35;
  float detail = (vnoise(p * 5.8 + vec2(-t * 0.66, t * 0.71)) - 0.5) * 2.0;
  s += detail * (0.05 + uBlend * 0.22);
  return s;
}

// ── 3: Voronoi / Cell ──
float noiseVoronoi(vec2 uvRaw, float t) {
  vec2 uv = applyNoiseMapping(uvRaw, t);
  vec2 p = uv * (3.5 + uFrequency * 21.0 + uWarp * 10.0);
  vec2 jitter = (vec2(
    vnoise(p * 0.7 + vec2(t * 0.2, -t * 0.18)),
    vnoise(p * 0.7 + vec2(-t * 0.21, t * 0.16))
  ) * 2.0 - 1.0);
  float cellA = worleyLite(p + jitter * (0.2 + uSharpness * 1.8));
  float cellB = worleyLite(rot2(p * 1.9, 0.9) + jitter.yx * (0.15 + uSharpness * 1.2));
  float plateau = smoothstep(0.2, 0.9, cellA);
  float edge = abs(cellA - cellB);
  float crack = pow(1.0 - clamp(edge * (1.4 + uBlend * 2.6), 0.0, 1.0), 0.9 + uBlend * 2.3);
  float debris = (vnoise(p * 4.6 + vec2(t * 0.27, t * 0.31)) - 0.5) * 2.0;
  float s = mix(plateau, crack, 0.35 + uBlend * 0.5);
  s += debris * (0.05 + uWarp * 0.18);
  return s;
}

// ── 4: Flow / Rotated ──
float noiseFlow(vec2 uvRaw, float t) {
  vec2 uv = applyNoiseMapping(uvRaw, t);
  vec2 p = uv * (2.2 + uFrequency * 15.0);
  float heading = (vnoise(p * 0.35 + vec2(t * 0.11, -t * 0.16)) * 2.0 - 1.0) * PI * (0.2 + uWarp * 1.7);
  vec2 dir = vec2(cos(heading), sin(heading));
  float shear = (vnoise(rot2(p * 0.85, 1.05) + vec2(-t * 0.28, t * 0.22)) - 0.5) * (0.4 + uSharpness * 2.6);
  float streamCoord = dot(p, dir) * (2.0 + uAnisotropy * 12.0) + shear + t * (0.65 + uSpeed * 2.6);
  float crossCoord = dot(p, vec2(-dir.y, dir.x)) * (1.4 + uBlend * 8.0) - t * 0.45;
  float stream = 0.5 + 0.5 * sin(streamCoord);
  float cross_ = 0.5 + 0.5 * sin(crossCoord);
  float eddy = fbm4(rot2(p * 1.1, 0.6 + heading * 0.12), t);
  return mix(stream, eddy, 0.22 + uBlend * 0.4) * mix(1.0, cross_, 0.2 + uSharpness * 0.45);
}

// ── 5: Gabor-like ──
float noiseGabor(vec2 uvRaw, float t) {
  vec2 uv = applyNoiseMapping(uvRaw, t);
  vec2 p = uv * (1.7 + uFrequency * 10.0);
  float orient = (vnoise(p * 0.42 + vec2(1.7, -2.9)) * 2.0 - 1.0) * PI * (0.15 + uSharpness * 0.95);
  vec2 dir = vec2(cos(orient), sin(orient));
  float freq = 8.0 + uFrequency * 22.0 + uWarp * 26.0;
  float carrier = sin(dot(p, dir) * freq + t * (0.45 + uSpeed * 2.2));
  float gateN = vnoise(rot2(p * (0.9 + uBlend * 1.5), 1.1) + vec2(-t * 0.3, t * 0.24));
  float envelope = pow(smoothstep(0.15, 0.95, gateN), 0.6 + uBlend * 2.5);
  float secondary = sin(dot(p, vec2(-dir.y, dir.x)) * (3.0 + uAnisotropy * 12.0) - t * 0.65);
  float grain = (vnoise(p * 5.5 + vec2(t * 0.7, -t * 0.62)) - 0.5) * 2.0;
  float s = 0.5 + 0.5 * carrier * envelope;
  s = mix(s, 0.5 + 0.5 * secondary, 0.16 + uAnisotropy * 0.34);
  s += grain * (0.05 + uBlend * 0.2);
  return s;
}

// ── 6: Hybrid ──
float noiseHybrid(vec2 uvRaw, float t) {
  vec2 uv = applyNoiseMapping(uvRaw, t);
  vec2 p = uv * (2.3 + uFrequency * 14.0);
  float fractal = fbm4(p, t);
  float ridge = pow(max(1.0 - abs(fractal * 2.0 - 1.0), 0.0), 1.0 + uAnisotropy * 2.8);
  float cell = worleyLite(rot2(p * (1.2 + uWarp * 1.6), 0.9) + vec2(t * 0.12, -t * 0.1));
  float flow = 0.5 + 0.5 * sin(dot(p, normalize(vec2(0.8, 0.6))) * (6.0 + uSharpness * 18.0) + t * (0.8 + uSpeed * 1.9));
  float s = mix(fractal, ridge, 0.2 + uAnisotropy * 0.55);
  s = mix(s, cell, 0.15 + uBlend * 0.5);
  s = mix(s, flow, 0.1 + uWarp * 0.45);
  s += (vnoise(p * 6.3 + vec2(-t * 0.81, t * 0.74)) - 0.5) * 2.0 * (0.04 + uBlend * 0.18);
  return s;
}

// ── Jupiter vortex helper ──
vec2 jupiterVortex(vec2 uvRaw, vec2 center, vec2 sigma, float sign) {
  vec2 delta = (uvRaw - center) / sigma;
  float r2 = dot(delta, delta);
  float bump = sign * exp(-r2);
  return vec2(bump, exp(-r2 * 0.5));
}

// ── 7: Jupiter ──
float noiseJupiter(vec2 uvRaw, float t) {
  vec2 pPlanet = uvRaw - vec2(0.5);
  float SPHERE_RADIUS = 0.3665;
  float latNorm = pPlanet.y / SPHERE_RADIUS;
  float lonNorm = pPlanet.x / SPHERE_RADIUS;

  float jetShear = uWarp;
  float stormDensity = uSharpness;
  float vortexStrength = uAnisotropy;
  float bandContrast = uBlend;

  float tSlow = t * 0.15;

  // Band structure
  float bandBase = 13.0 + uFrequency * 6.0;
  float dampPole = exp(-2.5 * latNorm * latNorm);
  float band1 = sin(latNorm * bandBase * 2.0 + tSlow * 0.2) * dampPole;
  float band2 = sin(latNorm * bandBase * 0.85 + 0.7 + tSlow * 0.1) * 0.4 * dampPole;
  float band3 = sin(latNorm * bandBase * 4.0 + 2.1) * 0.08 * dampPole;
  float rawBands = 0.5 + 0.42 * (band1 + band2 + band3);

  float eqSmooth = smoothstep(0.0, 0.12, abs(latNorm));
  float bands = mix(0.62, rawBands, eqSmooth);

  float shear = (vnoise(vec2(latNorm * 3.0 + 7.0, tSlow * 0.3)) - 0.5) * jetShear * 0.4;

  // Storm vortices
  vec2 SC = vec2(0.5);
  vec2 grs = jupiterVortex(uvRaw,
    vec2(SC.x + 0.06, SC.y - 0.37 * SPHERE_RADIUS),
    vec2(0.06 + vortexStrength * 0.02, 0.038 + vortexStrength * 0.012),
    0.6 + vortexStrength * 0.4);
  vec2 wo1 = jupiterVortex(uvRaw,
    vec2(SC.x - 0.08, SC.y - 0.55 * SPHERE_RADIUS),
    vec2(0.03, 0.02),
    0.2 + vortexStrength * 0.15);
  vec2 nts = jupiterVortex(uvRaw,
    vec2(SC.x + 0.1, SC.y + 0.38 * SPHERE_RADIUS),
    vec2(0.025, 0.018),
    -(0.18 + vortexStrength * 0.12));
  vec2 seb = jupiterVortex(uvRaw,
    vec2(SC.x - 0.12, SC.y - 0.18 * SPHERE_RADIUS),
    vec2(0.02, 0.015),
    0.15 + vortexStrength * 0.1);

  // Multi-scale turbulence
  float turb1 = (vnoise(vec2(lonNorm * 8.0 - tSlow * 0.12, latNorm * 6.0 + tSlow * 0.08)) - 0.5) * 2.0;
  float turb2 = (vnoise(vec2(lonNorm * 18.0 + tSlow * 0.2, latNorm * 14.0 - tSlow * 0.15)) - 0.5) * 2.0;
  float turb3 = (vnoise(vec2(lonNorm * 35.0 - tSlow * 0.3, latNorm * 28.0 + tSlow * 0.22)) - 0.5) * 2.0;

  // Festoons
  float festoon = (vnoise(vec2(lonNorm * 12.0 + latNorm * 6.0, tSlow * 0.1 + 3.7)) - 0.5) * 0.08 * dampPole;

  // Final mix
  float s = bands + shear * 0.15 + festoon;
  s += grs.x * (2.5 + stormDensity * 1.2);
  s += wo1.x * (0.8 + stormDensity * 0.4);
  s += nts.x * (0.8 + stormDensity * 0.4);
  s += seb.x * (0.6 + stormDensity * 0.3);
  float midLatBoost = 1.0 - abs(latNorm) * 0.5;
  s += turb1 * (0.012 + bandContrast * 0.03) * midLatBoost;
  s += turb2 * (0.008 + bandContrast * 0.02) * midLatBoost;
  s += turb3 * (0.004 + bandContrast * 0.01) * midLatBoost;
  s = mix(0.5, s, 0.6 + bandContrast * 0.35);
  return clamp(s, 0.0, 1.0);
}

// ── Scalar field core (selects noise type) ──
float scalarFieldCore(vec2 uvRaw, float t) {
  float s = 0.0;
  if (uNoiseType == 0) { s = noiseClassicCurl(uvRaw, t); }
  else if (uNoiseType == 1) { s = noiseDomainWarped(uvRaw, t); }
  else if (uNoiseType == 2) { s = noiseRidged(uvRaw, t); }
  else if (uNoiseType == 3) { s = noiseVoronoi(uvRaw, t); }
  else if (uNoiseType == 4) { s = noiseFlow(uvRaw, t); }
  else if (uNoiseType == 5) { s = noiseGabor(uvRaw, t); }
  else if (uNoiseType == 6) { s = noiseHybrid(uvRaw, t); }
  else { s = noiseJupiter(uvRaw, t); }

  // Radial pulse overlay for radial mapping mode
  if (uMapping >= 1) {
    vec2 pRad = uvRaw - vec2(0.5);
    float r = length(pRad) / 0.3665;
    float radialPulse = 0.5 + 0.5 * sin(r * (26.0 + uFrequency * 20.0) - t * (2.6 + uSpeed * 4.2));
    s = mix(s, radialPulse, 0.28 + 0.25 * uBlend);
  }
  return s;
}

void main() {
  vec2 simRes = 1.0 / uTexelSize;
  vec2 uv = gl_FragCoord.xy * uTexelSize;

  vec2 vel = texture(uVelocity, uv).xy;

  float t = uTime * (0.1 + uSpeed * 2.0);

  // Symmetry frame
  SymmetryFrame frame = makeSymmetryFrame(uv);
  vec2 sampleUV = frame.uv;

  float eps = uTexelSize.x;

  // Primary scale curl
  float sC = scalarFieldCore(sampleUV, t);
  float sX = scalarFieldCore(sampleUV + vec2(eps, 0.0), t);
  float sY = scalarFieldCore(sampleUV + vec2(0.0, eps), t);
  vec2 curlCanonical = vec2(sY - sC, -(sX - sC)) / eps;

  // Secondary scale (1.91x offset)
  vec2 uv2 = vec2(0.5) + (sampleUV - vec2(0.5)) * 1.91;
  vec2 uv2X = vec2(0.5) + (sampleUV + vec2(eps, 0.0) - vec2(0.5)) * 1.91;
  vec2 uv2Y = vec2(0.5) + (sampleUV + vec2(0.0, eps) - vec2(0.5)) * 1.91;
  float s2C = scalarFieldCore(uv2, t);
  float s2X = scalarFieldCore(uv2X, t);
  float s2Y = scalarFieldCore(uv2Y, t);
  curlCanonical += vec2(s2Y - s2C, -(s2X - s2C)) / eps * (0.35 + uBlend * 0.25);

  // Transform curl back through symmetry frame
  vec2 curl = (frame.basisX * curlCanonical.x + frame.basisY * curlCanonical.y) * frame.handedness;

  float layerBoost = 1.0 + (uSharpness + uWarp + uAnisotropy + uBlend) * 0.35;
  vel += curl * uAmount * (7.0 + 2.0 * layerBoost);

  fragColor = vec4(vel, 0.0, 1.0);
}
`;

// ─── 10. Boundary Rasterization Fragment Shader ──────────────────────────────
export const boundaryFS = `#version 300 es
precision highp float;

uniform vec4 uPlatforms[16];
uniform int uPlatformCount;
uniform vec2 uResolution;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy / uResolution;

  float solid = 0.0;
  for (int i = 0; i < 16; i++) {
    if (i >= uPlatformCount) break;
    vec4 rect = uPlatforms[i]; // x, y, w, h in UV coords
    if (uv.x >= rect.x && uv.x <= rect.x + rect.z &&
        uv.y >= rect.y && uv.y <= rect.y + rect.w) {
      solid = 1.0;
      break;
    }
  }

  fragColor = vec4(solid, 0.0, 0.0, 1.0);
}
`;

// ─── 11. Display Fragment Shader (17 colormaps + palette gradient) ───────────
export const displayFS = `#version 300 es
precision highp float;

uniform sampler2D uDye;
uniform sampler2D uVelocity;
uniform sampler2D uPressure;
uniform sampler2D uCurl;
uniform sampler2D uTemperature;
uniform sampler2D uDivergence;
uniform int uColormapMode;
uniform int uColorSource;
uniform float uColorGain;
uniform vec3 uBaseColor;
uniform vec3 uAccentColor;
uniform vec3 uTipColor;
uniform vec3 uGlitterColor;
uniform vec3 uSheenColor;
uniform float uColorBlend;
uniform float uTempColorShift;
uniform int uDyeSoftCap;
uniform float uDyeCeiling;
uniform vec2 uScreenSize;

out vec4 fragColor;

// ── Colormap functions (6th-degree polynomial approximations) ──

vec3 cmapViridis(float t) {
  vec3 c0 = vec3(0.2777, 0.0054, 0.3340);
  vec3 c1 = vec3(0.1050, 1.4046, 1.3840);
  vec3 c2 = vec3(-0.3308, 0.2148, -4.7950);
  vec3 c3 = vec3(-4.6342, -5.7991, 12.2624);
  vec3 c4 = vec3(6.2282, 14.1799, -14.0464);
  vec3 c5 = vec3(4.7763, -13.7451, 6.0756);
  vec3 c6 = vec3(-5.4354, 4.6459, -0.6946);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapInferno(float t) {
  vec3 c0 = vec3(0.0002, 0.0016, -0.0194);
  vec3 c1 = vec3(0.1065, 0.5639, 3.9327);
  vec3 c2 = vec3(11.6024, -3.9728, -15.9423);
  vec3 c3 = vec3(-41.7039, 17.4363, 44.3541);
  vec3 c4 = vec3(77.1629, -33.4023, -81.8073);
  vec3 c5 = vec3(-73.6891, 32.6269, 73.2088);
  vec3 c6 = vec3(27.1632, -12.2461, -23.0702);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapPlasma(float t) {
  vec3 c0 = vec3(0.0587, 0.0234, 0.5433);
  vec3 c1 = vec3(2.1761, 0.2138, -2.6346);
  vec3 c2 = vec3(-6.8084, 6.2608, 12.6420);
  vec3 c3 = vec3(17.6953, -24.0146, -27.6687);
  vec3 c4 = vec3(-26.6811, 39.5587, 32.2827);
  vec3 c5 = vec3(20.5835, -31.7652, -20.0279);
  vec3 c6 = vec3(-6.0116, 10.5195, 5.7893);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapMagma(float t) {
  vec3 c0 = vec3(-0.0021, -0.0008, -0.0053);
  vec3 c1 = vec3(0.2516, 0.6775, 2.4946);
  vec3 c2 = vec3(8.3537, -3.5775, -8.6687);
  vec3 c3 = vec3(-27.6684, 14.2647, 27.1596);
  vec3 c4 = vec3(52.1761, -27.9436, -50.7682);
  vec3 c5 = vec3(-50.7685, 29.0467, 45.7131);
  vec3 c6 = vec3(18.6557, -11.4898, -15.8989);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapRainbow(float t) {
  float h = t * 6.0;
  float r = clamp(abs(h - 3.0) - 1.0, 0.0, 1.0);
  float g = clamp(2.0 - abs(h - 2.0), 0.0, 1.0);
  float b = clamp(2.0 - abs(h - 4.0), 0.0, 1.0);
  return vec3(r, g, b);
}

vec3 cmapCividis(float t) {
  vec3 c0 = vec3(0.0118, 0.1365, 0.2845);
  vec3 c1 = vec3(-1.7675, 0.6084, 2.9457);
  vec3 c2 = vec3(27.2759, 0.7066, -20.8675);
  vec3 c3 = vec3(-99.6276, -2.7138, 66.5702);
  vec3 c4 = vec3(169.6920, 5.1714, -102.6734);
  vec3 c5 = vec3(-136.8732, -4.5348, 75.3209);
  vec3 c6 = vec3(42.3212, 1.5370, -21.3691);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapTurbo(float t) {
  vec3 c0 = vec3(0.1695, 0.0646, 0.1908);
  vec3 c1 = vec3(4.2981, 3.1627, 8.4247);
  vec3 c2 = vec3(-45.8051, -5.0894, -18.0147);
  vec3 c3 = vec3(161.2554, 26.3008, -56.4336);
  vec3 c4 = vec3(-228.9764, -72.0879, 214.5408);
  vec3 c5 = vec3(139.8151, 69.8321, -233.3300);
  vec3 c6 = vec3(-30.2413, -22.1441, 84.6455);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapTwilight(float t) {
  vec3 c0 = vec3(0.9030, 0.8515, 0.9412);
  vec3 c1 = vec3(-3.6605, -0.3096, -4.0021);
  vec3 c2 = vec3(16.9169, -5.9994, 40.1627);
  vec3 c3 = vec3(-71.4223, -7.5505, -177.5606);
  vec3 c4 = vec3(156.0521, 53.7958, 332.9832);
  vec3 c5 = vec3(-148.5457, -61.1882, -278.6118);
  vec3 c6 = vec3(50.6458, 21.2911, 86.9936);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapCoolWarm(float t) {
  vec3 c0 = vec3(0.2290, 0.2807, 0.7545);
  vec3 c1 = vec3(1.0634, 2.4719, 1.4381);
  vec3 c2 = vec3(1.6752, -8.8387, -0.4910);
  vec3 c3 = vec3(-4.1865, 38.5691, -7.1476);
  vec3 c4 = vec3(6.6425, -87.0970, 6.3481);
  vec3 c5 = vec3(-8.4946, 84.3136, 1.0286);
  vec3 c6 = vec3(3.7738, -29.6923, -1.7824);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapParula(float t) {
  vec3 c0 = vec3(0.3076, 0.1581, 0.6756);
  vec3 c1 = vec3(-2.4198, 0.1171, 2.1750);
  vec3 c2 = vec3(37.1713, 13.1990, 5.2631);
  vec3 c3 = vec3(-204.7459, -55.3360, -65.6626);
  vec3 c4 = vec3(454.5859, 111.6119, 158.8787);
  vec3 c5 = vec3(-430.4851, -112.2711, -160.8023);
  vec3 c6 = vec3(146.5745, 43.0000, 59.7521);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapOcean(float t) {
  vec3 c0 = vec3(0.0033, 0.3884, -0.0034);
  vec3 c1 = vec3(0.1223, 3.3475, 1.0784);
  vec3 c2 = vec3(-5.2406, -49.1775, -0.5923);
  vec3 c3 = vec3(40.9309, 181.9882, 1.6538);
  vec3 c4 = vec3(-119.0150, -287.3440, -1.8884);
  vec3 c5 = vec3(142.7142, 211.0243, 0.7553);
  vec3 c6 = vec3(-58.6215, -59.2210, -0.0000);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapHot(float t) {
  vec3 c0 = vec3(0.1307, -0.0113, -0.0688);
  vec3 c1 = vec3(-1.2313, 0.4351, 2.8980);
  vec3 c2 = vec3(35.8805, -1.3049, -31.5646);
  vec3 c3 = vec3(-119.7020, -23.6360, 140.5122);
  vec3 c4 = vec3(167.7675, 113.7873, -291.6687);
  vec3 c5 = vec3(-108.1641, -150.3936, 278.0787);
  vec3 c6 = vec3(26.3120, 62.1851, -97.1968);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapCopper(float t) {
  vec3 c0 = vec3(0.0010, -0.0027, -0.0017);
  vec3 c1 = vec3(0.9088, 0.8424, 0.5365);
  vec3 c2 = vec3(4.2640, -0.4627, -0.2947);
  vec3 c3 = vec3(-19.3395, 1.2919, 0.8228);
  vec3 c4 = vec3(38.2314, -1.4752, -0.9395);
  vec3 c5 = vec3(-33.1553, 0.5901, 0.3758);
  vec3 c6 = vec3(10.0547, -0.0000, -0.0000);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapCubehelix(float t) {
  vec3 c0 = vec3(-0.0155, 0.0049, 0.0007);
  vec3 c1 = vec3(4.1435, -0.4912, 0.6291);
  vec3 c2 = vec3(-44.9785, 17.2184, 27.5656);
  vec3 c3 = vec3(182.5358, -54.7812, -189.1731);
  vec3 c4 = vec3(-305.9171, 64.4403, 453.5747);
  vec3 c5 = vec3(225.5414, -23.3654, -452.3094);
  vec3 c6 = vec3(-60.2432, -2.0791, 160.7356);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapSpectral(float t) {
  vec3 c0 = vec3(0.5675, 0.0151, 0.2259);
  vec3 c1 = vec3(4.0665, 2.0707, 3.3389);
  vec3 c2 = vec3(-17.6618, -1.5484, -44.4454);
  vec3 c3 = vec3(41.8482, 19.3656, 219.8414);
  vec3 c4 = vec3(-45.6985, -57.9659, -453.0748);
  vec3 c5 = vec3(9.9559, 58.1645, 414.7922);
  vec3 c6 = vec3(7.3099, -19.7990, -140.0635);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3(0.0), vec3(1.0));
}

vec3 cmapJupiter(float t) {
  float s = clamp(t, 0.0, 1.0);
  if (s < 0.12) { return mix(vec3(0.60,0.55,0.44), vec3(0.62,0.57,0.43), s / 0.12); }
  else if (s < 0.25) { return mix(vec3(0.62,0.57,0.43), vec3(0.58,0.50,0.36), (s - 0.12) / 0.13); }
  else if (s < 0.38) { return mix(vec3(0.60,0.52,0.38), vec3(0.50,0.38,0.24), (s - 0.25) / 0.13); }
  else if (s < 0.50) { return mix(vec3(0.50,0.38,0.24), vec3(0.42,0.30,0.17), (s - 0.38) / 0.12); }
  else if (s < 0.65) { return mix(vec3(0.42,0.30,0.17), vec3(0.35,0.22,0.12), (s - 0.50) / 0.15); }
  else if (s < 0.80) { return mix(vec3(0.35,0.22,0.12), vec3(0.45,0.16,0.08), (s - 0.65) / 0.15); }
  else { return mix(vec3(0.45,0.16,0.08), vec3(0.38,0.12,0.06), (s - 0.80) / 0.20); }
}

vec3 evalColormap(float t, int mode) {
  if (mode == 1) { return cmapViridis(t); }
  else if (mode == 2) { return cmapInferno(t); }
  else if (mode == 3) { return cmapPlasma(t); }
  else if (mode == 4) { return cmapMagma(t); }
  else if (mode == 5) { return cmapRainbow(t); }
  else if (mode == 6) { return cmapRainbow(1.0 - t); }
  else if (mode == 7) { return cmapCividis(t); }
  else if (mode == 8) { return cmapTurbo(t); }
  else if (mode == 9) { return cmapTwilight(t); }
  else if (mode == 10) { return cmapCoolWarm(t); }
  else if (mode == 11) { return cmapParula(t); }
  else if (mode == 12) { return cmapOcean(t); }
  else if (mode == 13) { return cmapHot(t); }
  else if (mode == 14) { return cmapCopper(t); }
  else if (mode == 15) { return cmapCubehelix(t); }
  else if (mode == 16) { return cmapSpectral(t); }
  else if (mode == 17) { return cmapJupiter(t); }
  else { return cmapMagma(t); }
}

void main() {
  vec2 uv = gl_FragCoord.xy / uScreenSize;
  vec2 texSize = vec2(textureSize(uDye, 0));
  vec2 texel = 1.0 / texSize;

  vec3 bg = vec3(0.039, 0.039, 0.059); // #0a0a0f

  vec3 dye = texture(uDye, uv).rgb;

  // Soft cap
  if (uDyeSoftCap > 0) {
    float dyeLen = length(dye);
    if (dyeLen > 0.001) {
      dye = dye / (1.0 + dyeLen / uDyeCeiling);
    }
  }

  float intensity = dot(dye, vec3(0.3, 0.6, 0.1));
  float gain = pow(10.0, (uColorGain - 0.5) * 4.0);

  // Color source selection
  float sourceVal = intensity * gain;
  if (uColorSource == 1) {
    vec2 vel = texture(uVelocity, uv).rg;
    sourceVal = length(vel) * gain * 0.01;
  } else if (uColorSource == 2) {
    float press = texture(uPressure, uv).r;
    sourceVal = abs(press) * gain * 0.01;
  } else if (uColorSource == 3) {
    float c0 = texture(uCurl, uv).r;
    float c1 = texture(uCurl, uv + vec2(texel.x, 0.0)).r;
    float c2 = texture(uCurl, uv - vec2(texel.x, 0.0)).r;
    float c3 = texture(uCurl, uv + vec2(0.0, texel.y)).r;
    float c4 = texture(uCurl, uv - vec2(0.0, texel.y)).r;
    sourceVal = abs((c0 * 2.0 + c1 + c2 + c3 + c4) / 6.0) * gain * 0.02;
  } else if (uColorSource == 4) {
    float temp = texture(uTemperature, uv).r;
    sourceVal = abs(temp - 0.5) * 2.0 * gain;
  } else if (uColorSource == 5) {
    vec2 vR = texture(uVelocity, uv + vec2(texel.x, 0.0)).rg;
    vec2 vL = texture(uVelocity, uv - vec2(texel.x, 0.0)).rg;
    vec2 vU = texture(uVelocity, uv + vec2(0.0, texel.y)).rg;
    vec2 vD = texture(uVelocity, uv - vec2(0.0, texel.y)).rg;
    float divVal = (vR.x - vL.x + vU.y - vD.y) * 0.5;
    sourceVal = abs(divVal) * gain * 0.02;
  }

  float mappedVal = max(sourceVal, 0.0);
  vec3 color = vec3(0.0);

  if (uColormapMode >= 1) {
    float cmapT = min(mappedVal, 1.0);
    vec3 cmapColor = evalColormap(cmapT, uColormapMode);
    float bloom = 1.0 + max(mappedVal - 1.0, 0.0);
    float dataPresence = smoothstep(0.0, 0.015, intensity);
    color = cmapColor * bloom * dataPresence;
  } else {
    // Palette gradient mode: use dye RGB directly, blend with palette colors
    float lo = mix(0.0, 0.35, uColorBlend);
    float hi = mix(1.0, 0.4, uColorBlend);
    float densityT = smoothstep(lo, hi, mappedVal);
    float t2 = min(densityT * 2.0, 1.0);
    float t3 = max(densityT * 2.0 - 1.0, 0.0);
    vec3 fluidCol = mix(mix(uAccentColor, uBaseColor, t2), uTipColor, t3);
    color = fluidCol * mappedVal * 1.5;
  }

  // Temperature color shift
  if (uTempColorShift > 0.001) {
    float temp = texture(uTemperature, uv).r;
    float heatBias = (temp - 0.5) * 2.0;
    vec3 warmTint = vec3(0.5, 0.08, -0.35) * max(heatBias, 0.0);
    vec3 coolTint = vec3(-0.25, 0.05, 0.5) * max(-heatBias, 0.0);
    vec3 tint = (warmTint + coolTint) * uTempColorShift;
    color += tint * max(intensity, 0.05);
  }

  // Blend with background
  color = max(color, bg * (1.0 - smoothstep(0.0, 0.01, intensity)));

  fragColor = vec4(color, 1.0);
}
`;

// ─── 12. Game Object Vertex Shader ───────────────────────────────────────────
export const gameVS = `#version 300 es
uniform vec4 uRect; // x, y, w, h in pixels
uniform vec2 uResolution;

void main() {
  // 6 vertices = 2 triangles for a quad
  // Vertex order: 0,1,2, 2,1,3
  int idx = gl_VertexID;
  vec2 corner;
  if (idx == 0 || idx == 3) { corner = vec2(0.0, 0.0); }
  else if (idx == 1)        { corner = vec2(1.0, 0.0); }
  else if (idx == 2 || idx == 4) { corner = vec2(0.0, 1.0); }
  else                      { corner = vec2(1.0, 1.0); }

  // Pixel position
  vec2 pos = uRect.xy + corner * uRect.zw;

  // Convert to clip space (Y-down world to Y-up clip)
  float cx = pos.x / uResolution.x * 2.0 - 1.0;
  float cy = 1.0 - pos.y / uResolution.y * 2.0;

  gl_Position = vec4(cx, cy, 0.0, 1.0);
}
`;

// ─── 13. Game Object Fragment Shader ─────────────────────────────────────────
export const gameFS = `#version 300 es
precision highp float;

uniform vec4 uColor;

out vec4 fragColor;

void main() {
  fragColor = uColor;
}
`;

// ─── 14. Line Vertex Shader (gun arm) ────────────────────────────────────────
export const lineVS = `#version 300 es
uniform vec2 uStart;     // pixels
uniform vec2 uEnd;       // pixels
uniform float uWidth;    // pixels
uniform vec2 uResolution;

void main() {
  vec2 dir = uEnd - uStart;
  float len = length(dir);
  vec2 fwd = len > 0.001 ? dir / len : vec2(1.0, 0.0);
  vec2 perp = vec2(-fwd.y, fwd.x);

  // 6 vertices = 2 triangles forming a rectangle along the line
  int idx = gl_VertexID;
  float along, side;
  if (idx == 0 || idx == 3)      { along = 0.0; side = -1.0; }
  else if (idx == 1)             { along = 0.0; side =  1.0; }
  else if (idx == 2 || idx == 4) { along = 1.0; side = -1.0; }
  else                           { along = 1.0; side =  1.0; }

  vec2 pos = uStart + fwd * len * along + perp * uWidth * 0.5 * side;

  float cx = pos.x / uResolution.x * 2.0 - 1.0;
  float cy = 1.0 - pos.y / uResolution.y * 2.0;

  gl_Position = vec4(cx, cy, 0.0, 1.0);
}
`;

// ─── 15. Circle SDF Fragment Shader (player head / crosshair) ────────────────
export const circleFS = `#version 300 es
precision highp float;

uniform vec2 uCenter;    // pixels
uniform float uRadius;   // pixels
uniform vec4 uColor;
uniform vec2 uResolution;
uniform float uHollow;   // 0=filled, 1=ring only

out vec4 fragColor;

void main() {
  // Fragment position in pixel coords (Y-down)
  vec2 fragPos = vec2(gl_FragCoord.x, uResolution.y - gl_FragCoord.y);
  float dist = length(fragPos - uCenter);
  float alpha;
  if (uHollow > 0.5) {
    // Ring: visible only near the edge
    alpha = (1.0 - smoothstep(uRadius - 1.5, uRadius + 1.5, dist))
          * smoothstep(uRadius - 3.0, uRadius - 1.5, dist);
  } else {
    alpha = 1.0 - smoothstep(uRadius - 1.5, uRadius + 1.5, dist);
  }
  fragColor = vec4(uColor.rgb, uColor.a * alpha);
}
`;

// ─── 16. Bloom Extract Fragment Shader ───────────────────────────────────────
export const bloomExtractFS = `#version 300 es
precision highp float;

uniform sampler2D uSource;
uniform float uThreshold;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(uSource, 0));
  vec3 color = texture(uSource, uv).rgb;
  fragColor = vec4(max(color - vec3(uThreshold), vec3(0.0)), 1.0);
}
`;

// ─── 17. Bloom Gaussian Blur Fragment Shader ─────────────────────────────────
export const bloomBlurFS = `#version 300 es
precision highp float;

uniform sampler2D uSource;
uniform vec2 uDirection;
uniform vec2 uTexelSize;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy * uTexelSize;

  // 5-tap Gaussian blur
  float weights[3] = float[3](0.227027, 0.1945946, 0.1216216);
  float offsets[3] = float[3](0.0, 1.0, 2.0);

  vec3 result = texture(uSource, uv).rgb * weights[0];
  for (int i = 1; i < 3; i++) {
    vec2 offset = uDirection * offsets[i] * uTexelSize;
    result += texture(uSource, uv + offset).rgb * weights[i];
    result += texture(uSource, uv - offset).rgb * weights[i];
  }

  fragColor = vec4(result, 1.0);
}
`;

// ─── 18. Bloom Composite Fragment Shader ─────────────────────────────────────
export const bloomCompositeFS = `#version 300 es
precision highp float;

uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uIntensity;
uniform vec2 uScreenSize;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy / uScreenSize;
  vec3 scene = texture(uScene, uv).rgb;
  vec3 bloom = texture(uBloom, uv).rgb;
  fragColor = vec4(scene + bloom * uIntensity, 1.0);
}
`;
