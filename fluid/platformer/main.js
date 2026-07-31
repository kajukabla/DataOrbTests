import { state, buildPlatforms, getPlatforms, setPlatforms } from './state.js';
import { initFluid } from './fluid.js';
import { initGame } from './game.js';
import { initUI } from './ui.js';

// --- Log Relay ---
(function setupLogRelay() {
  const origLog = console.log;
  const origWarn = console.warn;
  const origError = console.error;
  function relay(level, args) {
    const strs = args.map(a => typeof a === 'object' ? JSON.stringify(a) : String(a));
    fetch('/log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ level, args: strs }),
    }).catch(() => {});
  }
  console.log = (...args) => { origLog.apply(console, args); relay('log', args); };
  console.warn = (...args) => { origWarn.apply(console, args); relay('warn', args); };
  console.error = (...args) => { origError.apply(console, args); relay('error', args); };
  window.addEventListener('error', e => relay('error', [`Uncaught: ${e.message} at ${e.filename}:${e.lineno}`]));
  window.addEventListener('unhandledrejection', e => relay('error', [`Unhandled rejection: ${e.reason}`]));
})();

// --- Canvas Setup ---
const canvas = document.getElementById('canvas');
// --- Initialize Systems ---
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
setPlatforms(buildPlatforms(canvas.width, canvas.height));

const fluid = initFluid(canvas);
const game = initGame(canvas);
initUI();

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  setPlatforms(buildPlatforms(canvas.width, canvas.height));
  fluid.resize(canvas.width, canvas.height);
}
window.addEventListener('resize', resizeCanvas);

console.log('Fluid Platformer initialized');

// --- HUD ---
const hudEl = document.getElementById('hud');
let frameCount = 0;
let fpsTime = 0;
let fps = 0;

// Dye intensity tracking for diagnostics
let dyeIntensity = 0;
let velMagnitude = 0;
let diagSampleFrame = 0;

function sampleTextures(gl) {
  diagSampleFrame++;
  if (diagSampleFrame % 10 !== 0) return; // every 10 frames
  const textures = fluid.getTextures();
  const pixel = new Float32Array(4);

  // Sample dye at 9 grid points for a spatial average
  const pts = [[128,128],[256,128],[384,128],[128,256],[256,256],[384,256],[128,384],[256,384],[384,384]];
  let totalI = 0;
  gl.bindFramebuffer(gl.FRAMEBUFFER, textures.dye.read.fbo);
  for (const [x,y] of pts) {
    gl.readPixels(x, y, 1, 1, gl.RGBA, gl.FLOAT, pixel);
    totalI += pixel[0] * 0.3 + pixel[1] * 0.6 + pixel[2] * 0.1;
  }
  dyeIntensity = totalI / pts.length;

  // Sample velocity magnitude at center
  gl.bindFramebuffer(gl.FRAMEBUFFER, textures.velocity.read.fbo);
  gl.readPixels(256, 256, 1, 1, gl.RGBA, gl.FLOAT, pixel);
  velMagnitude = Math.sqrt(pixel[0] * pixel[0] + pixel[1] * pixel[1]);

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function updateHUD(dt) {
  frameCount++;
  fpsTime += dt;
  if (fpsTime >= 1.0) {
    fps = frameCount;
    frameCount = 0;
    fpsTime = 0;
  }
  hudEl.textContent = `FPS: ${fps}  DYE: ${dyeIntensity.toFixed(4)}  VEL: ${velMagnitude.toFixed(1)}`;
}

// --- Controls Hint Auto-fade ---
const hintEl = document.getElementById('controls-hint');
setTimeout(() => { if (hintEl) hintEl.style.opacity = '0'; }, 8000);

// --- Main Loop ---
let lastTime = 0;

function loop(time) {
  requestAnimationFrame(loop);
  const timeS = time * 0.001;
  const dt = Math.min(lastTime ? timeS - lastTime : 0.016, 0.05); // cap at 50ms
  lastTime = timeS;

  // Scale dt by sim speed
  const simDt = dt * state.simSpeed;

  // Update game physics, get splats to apply
  const { splats, repellerSplats } = game.update(dt, canvas.width, canvas.height);

  // Apply regular splats (shooting) before simulation
  for (const s of splats) {
    fluid.splat(s.x, s.y, s.dx, s.dy, s.color, s.radius);
  }

  // Run fluid simulation step — repeller splats applied after pressure projection
  fluid.step(simDt, getPlatforms(), repellerSplats);

  // Render fluid to screen
  fluid.render(null);

  // Render game objects on top
  game.render(fluid.gl, canvas.width, canvas.height);

  // Sample textures for HUD diagnostics
  sampleTextures(fluid.gl);

  // Update HUD
  updateHUD(dt);
}

requestAnimationFrame(loop);
