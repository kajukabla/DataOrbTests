// ── Constants ──────────────────────────────────────────────────────────────────
const PARTICLE_COUNT = 2000;
const PARTICLE_STRIDE = 32;
const WORKGROUP_SIZE = 256;
const TRAIL_SCALE = 1; // Full resolution — sharp, thin filaments

function fatal(msg) {
  document.getElementById('error').style.display = 'block';
  document.getElementById('error').textContent = msg;
  throw new Error(msg);
}

// ── WGSL ───────────────────────────────────────────────────────────────────────

const PARTICLE_STRUCT_WGSL = `
struct Particle { pos: vec3f, life: f32, vel: vec3f, seed: f32, };
`;
const UNIFORMS_STRUCT_WGSL = `
struct Uniforms { time: f32, deltaTime: f32, resolution: vec2f, sphereRadius: f32, noiseScale: f32, curlStrength: f32, particleCount: u32, };
`;

const SIMULATE_WGSL = /* wgsl */`
${PARTICLE_STRUCT_WGSL}
${UNIFORMS_STRUCT_WGSL}
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

fn hash3(p: vec3f) -> vec3f {
  var q = vec3f(dot(p, vec3f(127.1,311.7,74.7)), dot(p, vec3f(269.5,183.3,246.1)), dot(p, vec3f(113.5,271.9,124.6)));
  return fract(sin(q) * 43758.5453123) * 2.0 - 1.0;
}
fn noise3D(p: vec3f) -> f32 {
  let i = floor(p); let f = fract(p); let u = f*f*(3.0-2.0*f);
  return mix(mix(mix(dot(hash3(i),f), dot(hash3(i+vec3f(1,0,0)),f-vec3f(1,0,0)),u.x),
    mix(dot(hash3(i+vec3f(0,1,0)),f-vec3f(0,1,0)), dot(hash3(i+vec3f(1,1,0)),f-vec3f(1,1,0)),u.x),u.y),
    mix(mix(dot(hash3(i+vec3f(0,0,1)),f-vec3f(0,0,1)), dot(hash3(i+vec3f(1,0,1)),f-vec3f(1,0,1)),u.x),
    mix(dot(hash3(i+vec3f(0,1,1)),f-vec3f(0,1,1)), dot(hash3(i+vec3f(1,1,1)),f-vec3f(1,1,1)),u.x),u.y),u.z);
}
fn curlNoise(p: vec3f) -> vec3f {
  let e=0.01; let dx=vec3f(e,0,0); let dy=vec3f(0,e,0); let dz=vec3f(0,0,e);
  let px=p; let py=p+vec3f(31.416,47.853,12.793); let pz=p+vec3f(93.146,17.352,68.235);
  return vec3f(
    (noise3D(pz+dy)-noise3D(pz-dy))-(noise3D(py+dz)-noise3D(py-dz)),
    (noise3D(px+dz)-noise3D(px-dz))-(noise3D(pz+dx)-noise3D(pz-dx)),
    (noise3D(py+dx)-noise3D(py-dx))-(noise3D(px+dy)-noise3D(px-dy)))/(2.0*e);
}
fn pcgHash(input: u32) -> u32 {
  var s = input*747796405u+2891336453u;
  let w = ((s>>((s>>28u)+4u))^s)*277803737u; return (w>>22u)^w;
}
fn randomF(seed: u32) -> f32 { return f32(pcgHash(seed))/4294967295.0; }

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= uniforms.particleCount) { return; }
  var p = particles[idx];
  let dt = uniforms.deltaTime; let time = uniforms.time; let R = uniforms.sphereRadius;
  p.life -= dt;
  let dist = length(p.pos);

  if (p.life <= 0.0 || dist > R * 1.05) {
    let s = pcgHash(u32(p.seed*1000.0)+u32(time*100.0)+idx);
    let s1=randomF(s); let s2=randomF(s+1u); let s3=randomF(s+2u); let s4=randomF(s+3u); let s5=randomF(s+4u);
    let angle = s1*6.2831853;
    let cosTheta = s2*2.0-1.0;
    let sinTheta = sqrt(max(1.0-cosTheta*cosTheta, 0.0));
    // 15% axis, 85% volume — biased toward sphere surface
    let useAxis = step(0.85, s5);
    let rVol = pow(s3, 0.2)*R*0.95;  // pow(0.2) biases toward surface
    let rAxis = s4*s4*0.12*R;
    let radial = mix(rVol*sinTheta, rAxis, useAxis);
    let yPos = mix(rVol*cosTheta, (s3*2.0-1.0)*R*0.9, useAxis);
    p.pos = vec3f(cos(angle)*radial, yPos, sin(angle)*radial);
    p.vel = vec3f(0.0); p.life = 5.0+s4*12.0; p.seed = f32(s)/4294967295.0;
    particles[idx] = p; return;
  }

  let rho = length(vec2f(p.pos.x, p.pos.z));
  let normDist = dist / R;
  // Curl noise — sweeping curves
  let curl = curlNoise(p.pos*uniforms.noiseScale + vec3f(time*0.12, time*0.08, time*0.1)) * uniforms.curlStrength;
  // Toroidal rotation — wrapping around Y-axis
  let toroidal = vec3f(-p.pos.z, 0.0, p.pos.x) * 0.5;
  // Poloidal circulation — dipole field lines, but weaker so particles spread out
  var rhoHat = vec3f(p.pos.x, 0.0, p.pos.z);
  let rhoLen = length(rhoHat);
  if (rhoLen > 0.001) { rhoHat = rhoHat / rhoLen; }
  let polRadial = p.pos.y * 0.8;
  let polVertical = (1.0 - 3.0*(rho/R)*(rho/R)) * 0.6;
  let poloidal = rhoHat * polRadial + vec3f(0.0, polVertical, 0.0);
  // Push particles outward from center to fill the sphere
  let outwardPush = normalize(p.pos + vec3f(0.0001)) * 0.15 * (1.0 - normDist);
  let axisForce = outwardPush;
  // Containment
  // Oscillating sphere boundary — breathes in and out
  let breathe = sin(time * 0.8) * 0.08 + sin(time * 1.3) * 0.04;
  let effectiveRadius = 1.0 + breathe;  // oscillates ~0.88 to ~1.12
  let normDistOsc = dist / (R * effectiveRadius);
  let containForce = -normalize(p.pos+vec3f(0.0001)) * smoothstep(0.85, 1.0, normDistOsc) * 6.0;

  p.vel = p.vel*0.96 + (curl + toroidal + poloidal + axisForce + containForce)*dt;
  let speed = length(p.vel);
  if (speed > 1.8) { p.vel *= 1.8/speed; }
  p.pos += p.vel * dt;
  particles[idx] = p;
}
`;

// Line splat at TRAIL resolution (lower than canvas → thicker filaments)
const SPLAT_WGSL = /* wgsl */`
${PARTICLE_STRUCT_WGSL}
${UNIFORMS_STRUCT_WGSL}
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read_write> accumGrid: array<atomic<u32>>;

fn projectToScreen(worldPos: vec3f, w: f32, h: f32, aspect: f32) -> vec2f {
  let eyeZ = 2.8 - worldPos.z;
  let ps = 1.0 / (0.85 * max(eyeZ, 0.1));
  return vec2f((worldPos.x*ps/aspect+1.0)*0.5*w, (-worldPos.y*ps+1.0)*0.5*h);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= uniforms.particleCount) { return; }
  let p = particles[idx];
  if (p.life <= 0.0 || length(p.vel) < 0.0001) { return; }

  // Trail resolution (passed via resolution uniform)
  let w = f32(u32(uniforms.resolution.x));
  let h = f32(u32(uniforms.resolution.y));
  let aspect = w / h;
  let iw = u32(w); let ih = u32(h);

  let curr = projectToScreen(p.pos, w, h, aspect);
  let prev = projectToScreen(p.pos - p.vel*uniforms.deltaTime, w, h, aspect);

  let diff = curr - prev;
  let lineLen = max(abs(diff.x), abs(diff.y));
  let steps = min(u32(lineLen)+1u, 40u);

  let radialDist = length(vec2f(p.pos.x, p.pos.z));
  let intensity = (1.0/(1.0+radialDist*5.0) + smoothstep(0.4,0.9,radialDist)*0.2) * min(p.life*0.2, 1.0);
  let fv = u32(clamp(intensity, 0.0, 2.0) * 512.0);
  if (fv == 0u) { return; }

  for (var i = 0u; i <= steps; i++) {
    let t = f32(i)/f32(max(steps,1u));
    let sp = mix(prev, curr, t);
    let px = i32(sp.x); let py = i32(sp.y);
    if (px >= 0 && px < i32(iw) && py >= 0 && py < i32(ih)) {
      atomicAdd(&accumGrid[u32(py)*iw+u32(px)], fv);
    }
  }
}
`;

const ACCUM_TO_TRAIL_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> accumGrid: array<atomic<u32>>;
@group(0) @binding(1) var trailOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var trailIn: texture_2d<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(trailOut);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let idx = gid.y * dims.x + gid.x;
  let raw = atomicExchange(&accumGrid[idx], 0u);
  let newVal = f32(raw) / 512.0;
  let prev = textureLoad(trailIn, vec2i(gid.xy), 0).r;
  let combined = min(prev * 0.992 + newVal * 0.5, 3.0);  // Short trails, bright per-particle, capped
  textureStore(trailOut, vec2i(gid.xy), vec4f(combined, combined, combined, 1.0));
}
`;

// Pure copy — NO blur. Cumulative blur destroys filament detail over hundreds of frames.
// The 3x upscale from trail→canvas with linear sampling provides natural softening.
const DIFFUSE_WGSL = /* wgsl */`
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(outputTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let val = textureLoad(inputTex, vec2i(gid.xy), 0).r;
  textureStore(outputTex, vec2i(gid.xy), vec4f(val, val, val, 1.0));
}
`;

const FULLSCREEN_VERT_WGSL = /* wgsl */`
struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, };
@vertex fn main(@builtin(vertex_index) vi: u32) -> VSOut {
  var pos = array<vec2f, 3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
  var out: VSOut; out.pos = vec4f(pos[vi], 0.0, 1.0);
  out.uv = pos[vi]*0.5+0.5; out.uv.y = 1.0-out.uv.y; return out;
}
`;

const COMPOSITE_FRAG_WGSL = /* wgsl */`
@group(0) @binding(0) var trailTex: texture_2d<f32>;
@group(0) @binding(1) var trailSampler: sampler;

fn gradientMap(t: f32) -> vec3f {
  let c0=vec3f(0.01,0.02,0.08); let c1=vec3f(0.06,0.12,0.4);
  let c2=vec3f(0.2,0.32,0.78); let c3=vec3f(0.5,0.28,0.65);
  let c4=vec3f(0.82,0.32,0.58); let c5=vec3f(1.0,0.78,0.9);
  if (t<0.1) { return mix(c0,c1,t/0.1); }
  if (t<0.28) { return mix(c1,c2,(t-0.1)/0.18); }
  if (t<0.48) { return mix(c2,c3,(t-0.28)/0.2); }
  if (t<0.7) { return mix(c3,c4,(t-0.48)/0.22); }
  return mix(c4,c5,clamp((t-0.7)/0.3,0.0,1.0));
}

@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(trailTex));
  let aspect = dims.x / dims.y;
  let delta = vec2f((uv.x-0.5)*aspect, uv.y-0.5);
  let sphereDist = length(delta) / 0.48;
  let sphereAlpha = 1.0 - smoothstep(0.95, 1.0, sphereDist);

  let sharp = textureSample(trailTex, trailSampler, uv).r;

  // Bloom: sample at multiple radii for soft glow around filaments
  let ts = 1.0 / dims;
  var bloom1 = 0.0;  // near glow (2px)
  bloom1 += textureSample(trailTex, trailSampler, uv + vec2f(ts.x*2.0, 0.0)).r;
  bloom1 += textureSample(trailTex, trailSampler, uv - vec2f(ts.x*2.0, 0.0)).r;
  bloom1 += textureSample(trailTex, trailSampler, uv + vec2f(0.0, ts.y*2.0)).r;
  bloom1 += textureSample(trailTex, trailSampler, uv - vec2f(0.0, ts.y*2.0)).r;
  bloom1 *= 0.25;
  var bloom2 = 0.0;  // far glow (6px)
  bloom2 += textureSample(trailTex, trailSampler, uv + vec2f(ts.x*6.0, 0.0)).r;
  bloom2 += textureSample(trailTex, trailSampler, uv - vec2f(ts.x*6.0, 0.0)).r;
  bloom2 += textureSample(trailTex, trailSampler, uv + vec2f(0.0, ts.y*6.0)).r;
  bloom2 += textureSample(trailTex, trailSampler, uv - vec2f(0.0, ts.y*6.0)).r;
  bloom2 += textureSample(trailTex, trailSampler, uv + vec2f(ts.x*4.0, ts.y*4.0)).r;
  bloom2 += textureSample(trailTex, trailSampler, uv - vec2f(ts.x*4.0, ts.y*4.0)).r;
  bloom2 += textureSample(trailTex, trailSampler, uv + vec2f(ts.x*4.0, -ts.y*4.0)).r;
  bloom2 += textureSample(trailTex, trailSampler, uv - vec2f(ts.x*4.0, -ts.y*4.0)).r;
  bloom2 *= 0.125;

  // Combine: sharp filament + near glow + far soft bloom
  let intensity = sharp + bloom1 * 0.4 + bloom2 * 0.25;

  let mapped = pow(max(intensity, 0.00001), 0.35) * 1.2;
  var color = gradientMap(clamp(mapped, 0.0, 1.0)) * min(mapped, 2.5) * 0.7;

  let rimGlow = pow(smoothstep(0.75, 0.98, sphereDist), 1.5) * 0.15;
  color += vec3f(0.1, 0.18, 0.45) * rimGlow;

  var finalColor = color * sphereAlpha;
  finalColor = finalColor / (1.0 + finalColor);
  finalColor = pow(finalColor, vec3f(1.0/2.2));
  return vec4f(finalColor, 1.0);
}
`;

// ── Main ───────────────────────────────────────────────────────────────────────

async function main() {
  console.log('Starting WebGPU init...');
  if (!navigator.gpu) fatal('WebGPU not supported.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) fatal('No adapter.');
  console.log('Adapter OK');
  const device = await adapter.requestDevice();
  device.lost.then(i => console.error('Device lost:', i.message));
  device.onuncapturederror = e => console.error('GPU ERROR:', e.error.message);
  console.log('Device created');

  const canvas = document.getElementById('canvas');
  const dpr = Math.max(devicePixelRatio, 2);  // Force at least 2x for retina sharpness
  canvas.width = window.innerWidth * dpr;
  canvas.height = window.innerHeight * dpr;
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'premultiplied' });
  const W = canvas.width, H = canvas.height;

  // Trail at lower resolution for thicker filaments
  const TW = Math.ceil(W / TRAIL_SCALE), TH = Math.ceil(H / TRAIL_SCALE);

  const particleBuf = device.createBuffer({ size: PARTICLE_COUNT * PARTICLE_STRIDE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  const init = new Float32Array(PARTICLE_COUNT * 8);
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const o = i * 8;
    const a = Math.random() * Math.PI * 2;
    const ct = Math.random() * 2 - 1, st = Math.sqrt(1 - ct * ct);
    const rs = Math.cbrt(Math.random()) * 0.95;
    if (Math.random() < 0.3) {
      const r = Math.pow(Math.random(), 2) * 0.12;
      init[o]=Math.cos(a)*r; init[o+1]=(Math.random()*2-1)*0.9; init[o+2]=Math.sin(a)*r;
    } else {
      init[o]=Math.cos(a)*st*rs; init[o+1]=ct*rs; init[o+2]=Math.sin(a)*st*rs;
    }
    init[o+3]=Math.random()*12+5; init[o+7]=Math.random();
  }
  device.queue.writeBuffer(particleBuf, 0, init);

  const uniformBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const accumBuf = device.createBuffer({ size: TW * TH * 4, usage: GPUBufferUsage.STORAGE });

  const makeTex = (w, h) => device.createTexture({ size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
  const trailA = makeTex(TW, TH), trailB = makeTex(TW, TH);
  const sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' });

  console.log(`Canvas: ${W}x${H}, Trail: ${TW}x${TH}`);

  const mk = (code, label) => {
    const m = device.createShaderModule({ code, label });
    m.getCompilationInfo().then(info => { for (const msg of info.messages) (msg.type==='error'?console.error:console.log)(`[${label}] ${msg.type}: ${msg.message} (line ${msg.lineNum})`); });
    return m;
  };

  const simPL = device.createComputePipeline({ layout:'auto', compute:{ module:mk(SIMULATE_WGSL,'sim'), entryPoint:'main' }});
  const splatPL = device.createComputePipeline({ layout:'auto', compute:{ module:mk(SPLAT_WGSL,'splat'), entryPoint:'main' }});
  const accumPL = device.createComputePipeline({ layout:'auto', compute:{ module:mk(ACCUM_TO_TRAIL_WGSL,'accum'), entryPoint:'main' }});
  const diffPL = device.createComputePipeline({ layout:'auto', compute:{ module:mk(DIFFUSE_WGSL,'diff'), entryPoint:'main' }});
  const renderPL = device.createRenderPipeline({ layout:'auto',
    vertex:{ module:mk(FULLSCREEN_VERT_WGSL,'vert'), entryPoint:'main' },
    fragment:{ module:mk(COMPOSITE_FRAG_WGSL,'frag'), entryPoint:'main', targets:[{format}] },
    primitive:{ topology:'triangle-list' }});

  const simBG = device.createBindGroup({ layout:simPL.getBindGroupLayout(0), entries:[
    {binding:0,resource:{buffer:particleBuf}},{binding:1,resource:{buffer:uniformBuf}}]});
  const splatBG = device.createBindGroup({ layout:splatPL.getBindGroupLayout(0), entries:[
    {binding:0,resource:{buffer:particleBuf}},{binding:1,resource:{buffer:uniformBuf}},{binding:2,resource:{buffer:accumBuf}}]});
  const accumBG = device.createBindGroup({ layout:accumPL.getBindGroupLayout(0), entries:[
    {binding:0,resource:{buffer:accumBuf}},{binding:1,resource:trailB.createView()},{binding:2,resource:trailA.createView()}]});
  const diffBG = device.createBindGroup({ layout:diffPL.getBindGroupLayout(0), entries:[
    {binding:0,resource:trailB.createView()},{binding:1,resource:trailA.createView()}]});
  const renderBG = device.createBindGroup({ layout:renderPL.getBindGroupLayout(0), entries:[
    {binding:0,resource:trailA.createView()},{binding:1,resource:sampler}]});

  console.log('All pipelines OK');
  const pD = Math.ceil(PARTICLE_COUNT/WORKGROUP_SIZE);
  const tDX = Math.ceil(TW/16), tDY = Math.ceil(TH/16);

  let lastTime = performance.now()/1000;
  const startTime = lastTime;

  function frame() {
    const now = performance.now()/1000;
    const dt = Math.min(now-lastTime, 0.05);
    lastTime = now;
    const time = now - startTime;

    const ub = new ArrayBuffer(32);
    const fv = new Float32Array(ub), uv = new Uint32Array(ub);
    fv[0]=time; fv[1]=dt;
    fv[2]=TW; fv[3]=TH;  // Trail resolution for splat shader
    fv[4]=1.0; fv[5]=1.8; fv[6]=1.5;  // stronger noise + more detail
    uv[7]=PARTICLE_COUNT;
    device.queue.writeBuffer(uniformBuf, 0, ub);

    const enc = device.createCommandEncoder();
    let p;
    p=enc.beginComputePass(); p.setPipeline(simPL); p.setBindGroup(0,simBG); p.dispatchWorkgroups(pD); p.end();
    p=enc.beginComputePass(); p.setPipeline(splatPL); p.setBindGroup(0,splatBG); p.dispatchWorkgroups(pD); p.end();
    p=enc.beginComputePass(); p.setPipeline(accumPL); p.setBindGroup(0,accumBG); p.dispatchWorkgroups(tDX,tDY); p.end();
    p=enc.beginComputePass(); p.setPipeline(diffPL); p.setBindGroup(0,diffBG); p.dispatchWorkgroups(tDX,tDY); p.end();
    const rp = enc.beginRenderPass({ colorAttachments:[{ view:ctx.getCurrentTexture().createView(), loadOp:'clear', storeOp:'store', clearValue:{r:0,g:0,b:0,a:1} }]});
    rp.setPipeline(renderPL); rp.setBindGroup(0,renderBG); rp.draw(3); rp.end();
    device.queue.submit([enc.finish()]);
    requestAnimationFrame(frame);
  }

  console.log('Starting render loop');
  requestAnimationFrame(frame);
}

main().catch(e => { console.error(e); fatal(e.message); });
