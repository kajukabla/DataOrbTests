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
function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  setPlatforms(buildPlatforms(canvas.width, canvas.height));
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// --- Initialize Systems ---
const fluid = initFluid(canvas);
const game = initGame(canvas);
initUI();

console.log('Fluid Platformer initialized');

// --- HUD ---
const hudEl = document.getElementById('hud');
let frameCount = 0;
let fpsTime = 0;
let fps = 0;

function updateHUD(dt) {
  frameCount++;
  fpsTime += dt;
  if (fpsTime >= 1.0) {
    fps = frameCount;
    frameCount = 0;
    fpsTime = 0;
  }
  hudEl.textContent = `FPS: ${fps}`;
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
  const splats = game.update(dt, canvas.width, canvas.height);

  // Apply splats to fluid
  for (const s of splats) {
    fluid.splat(s.x, s.y, s.dx, s.dy, s.color, s.radius);
  }

  // Run fluid simulation step
  fluid.step(simDt, getPlatforms());

  // Render fluid to screen
  fluid.render(null);

  // Render game objects on top
  game.render(fluid.gl, canvas.width, canvas.height);

  // Update HUD
  updateHUD(dt);
}

requestAnimationFrame(loop);
