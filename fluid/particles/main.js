// main.js — Game loop, initialization, settings

import { buildPlatforms } from './platforms.js';
import { initPlayer, updatePlayer, renderPlayer, getPlayer, getMouse, consumeShoot } from './player.js';
import { emitJet, updateParticles, renderParticles, getActiveCount } from './particles.js';

// ── Settings ───────────────────────────────────────────────────────────────

const settings = {
  firstSpeed: 400,
  particleSize: 4,
  particleCount: 30,
  curlNoise: 50,
  lifetime: 3,
  repellerSize: 60,
  repellerAmount: 500,
  emitterSize: 10,
};

// ── State ──────────────────────────────────────────────────────────────────

let canvas, ctx;
let platforms = [];
let shootQueued = false;

// ── Resize ─────────────────────────────────────────────────────────────────

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  platforms = buildPlatforms(canvas.width, canvas.height);
}

// ── HUD ────────────────────────────────────────────────────────────────────

let fpsAccum = 0, fpsFrames = 0, fpsDisplay = 60;
const hudEl = document.getElementById('hud');

function updateHUD(dt) {
  fpsAccum += dt;
  fpsFrames++;
  if (fpsAccum >= 0.5) {
    fpsDisplay = Math.round(fpsFrames / fpsAccum);
    fpsAccum = 0;
    fpsFrames = 0;
  }
  hudEl.textContent = `FPS: ${fpsDisplay} | Particles: ${getActiveCount()}`;
}

// ── Settings UI wiring ─────────────────────────────────────────────────────

function wireSlider(id, key) {
  const el = document.getElementById(id);
  const valEl = document.getElementById(id + 'Val');
  if (!el) return;
  el.value = settings[key];
  if (valEl) valEl.textContent = settings[key];
  el.addEventListener('input', () => {
    settings[key] = parseFloat(el.value);
    if (valEl) valEl.textContent = parseFloat(el.value).toFixed(
      el.step && el.step.includes('.') ? el.step.split('.')[1].length : 0
    );
  });
}

function initUI() {
  wireSlider('firstSpeed', 'firstSpeed');
  wireSlider('particleSize', 'particleSize');
  wireSlider('particleCount', 'particleCount');
  wireSlider('curlNoise', 'curlNoise');
  wireSlider('lifetime', 'lifetime');
  wireSlider('emitterSize', 'emitterSize');
  wireSlider('repellerSize', 'repellerSize');
  wireSlider('repellerAmount', 'repellerAmount');

  // Panel toggle
  const panel = document.getElementById('settingsPanel');
  const toggle = document.getElementById('settingsToggle');
  toggle.addEventListener('click', () => panel.classList.toggle('open'));
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') panel.classList.toggle('open');
  });
}

// ── Main Loop ──────────────────────────────────────────────────────────────

let lastTime = 0;

function loop(timestamp) {
  requestAnimationFrame(loop);

  const timeS = timestamp / 1000;
  const dt = Math.min(lastTime ? timeS - lastTime : 0.016, 0.05);
  lastTime = timeS;

  // Update player
  updatePlayer(dt, platforms, canvas.width, canvas.height);

  // Get player state
  const player = getPlayer();
  const playerCX = player.x + player.w / 2;
  const headTop = player.y - 12;       // head circle top (center y-4, radius 8)
  const bodyBottom = player.y + player.h; // feet
  const playerCY = (headTop + bodyBottom) / 2; // true visual center

  // Shooting — single press fires one jet
  if (consumeShoot()) {
    const mouse = getMouse();
    const gunX = playerCX;
    const gunY = player.y + player.h * 0.3;

    const dx = mouse.x - gunX;
    const dy = mouse.y - gunY;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const ax = dx / len, ay = dy / len;

    const tipX = gunX + ax * 15;
    const tipY = gunY + ay * 15;

    emitJet(tipX, tipY, ax, ay, settings.particleCount, settings.firstSpeed, settings.particleSize, settings.lifetime, settings.emitterSize);
  }
  updateParticles(dt, platforms, playerCX, playerCY, settings.curlNoise, settings.repellerSize, settings.repellerAmount);

  // Render
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Platforms
  ctx.fillStyle = 'rgba(20, 20, 30, 0.85)';
  for (const plat of platforms) {
    ctx.fillRect(plat.x, plat.y, plat.w, plat.h);
    // Top edge highlight
    ctx.fillStyle = 'rgba(50, 50, 70, 0.6)';
    ctx.fillRect(plat.x + 1, plat.y, plat.w - 2, 1);
    ctx.fillStyle = 'rgba(20, 20, 30, 0.85)';
  }

  // Particles (behind player)
  renderParticles(ctx);

  // Player (on top)
  renderPlayer(ctx);

  // Repeller radius visualization (faint circle around player)
  ctx.beginPath();
  ctx.arc(playerCX, playerCY, settings.repellerSize, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255, 150, 50, 0.25)';
  ctx.lineWidth = 1;
  ctx.stroke();


  updateHUD(dt);
}

// ── Init ───────────────────────────────────────────────────────────────────

canvas = document.getElementById('canvas');
ctx = canvas.getContext('2d');

resize();
window.addEventListener('resize', resize);

initPlayer(canvas);
initUI();

requestAnimationFrame(loop);
