// game.js — Platformer game engine (player physics, input, collision, shooting, rendering)

import { state, getPlatforms } from './state.js';
import { gameVS, gameFS, lineVS, circleFS, fullscreenVS } from './shaders.js';

// ── Player State ──────────────────────────────────────────────────────────────

const player = {
  x: 400, y: 500,
  vx: 0, vy: 0,
  w: 20, h: 32,
  grounded: false,
  facing: 1, // 1=right, -1=left
  dodgeCooldown: 0,
  dodging: false,
  dodgeTimer: 0,
  shootTimer: 0,
};

// ── Input State ───────────────────────────────────────────────────────────────

const keys = {};
const mouse = { x: 400, y: 300, down: false };
let shooting = false;

// ── Constants ─────────────────────────────────────────────────────────────────

const GRAVITY = 980;        // pixels/sec²
const MOVE_SPEED = 250;     // pixels/sec
const JUMP_SPEED = -420;    // pixels/sec (negative = up)
const DODGE_SPEED = 600;    // pixels/sec
const DODGE_DURATION = 0.15; // seconds
const DODGE_COOLDOWN = 0.5;  // seconds

// ── Shader Programs & Uniforms (populated in initGame) ───────────────────────

let gl = null;
let gameProgram = null;      // gameVS + gameFS
let lineProgram = null;      // lineVS + gameFS
let circleProgram = null;    // fullscreenVS + circleFS
let emptyVAO = null;

const uniforms = {
  game: {},
  line: {},
  circle: {},
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function compileShader(type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error('Shader compile error:', gl.getShaderInfoLog(s));
    gl.deleteShader(s);
    return null;
  }
  return s;
}

function createProgram(vsSrc, fsSrc) {
  const vs = compileShader(gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl.FRAGMENT_SHADER, fsSrc);
  if (!vs || !fs) return null;
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(p));
    return null;
  }
  return p;
}

function getUniforms(program, names) {
  const u = {};
  for (const n of names) {
    u[n] = gl.getUniformLocation(program, n);
  }
  return u;
}

function normalize(x, y) {
  const len = Math.sqrt(x * x + y * y) || 1;
  return [x / len, y / len];
}

// ── Exported: initGame ────────────────────────────────────────────────────────

export function initGame(canvas) {
  gl = canvas.getContext('webgl2');

  // ── Input listeners ──

  document.addEventListener('keydown', (e) => {
    keys[e.code] = true;
    if (e.code === 'Space') {
      shooting = true;
      e.preventDefault();
    }
  });

  document.addEventListener('keyup', (e) => {
    keys[e.code] = false;
    if (e.code === 'Space') {
      shooting = false;
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = (e.clientX - rect.left) * (canvas.width / rect.width);
    mouse.y = (e.clientY - rect.top) * (canvas.height / rect.height);
  });

  canvas.addEventListener('mousedown', () => { mouse.down = true; });
  canvas.addEventListener('mouseup', () => { mouse.down = false; });

  // ── Shader compilation ──

  gameProgram = createProgram(gameVS, gameFS);
  uniforms.game = getUniforms(gameProgram, [
    'uRect', 'uResolution', 'uColor',
  ]);

  lineProgram = createProgram(lineVS, gameFS);
  uniforms.line = getUniforms(lineProgram, [
    'uStart', 'uEnd', 'uWidth', 'uResolution', 'uColor',
  ]);

  circleProgram = createProgram(fullscreenVS, circleFS);
  uniforms.circle = getUniforms(circleProgram, [
    'uCenter', 'uRadius', 'uResolution', 'uColor', 'uHollow',
  ]);

  // Empty VAO for attribute-less rendering
  emptyVAO = gl.createVertexArray();

  return { update, render, getPlayer };
}

// ── Exported: update ──────────────────────────────────────────────────────────

function update(dt, canvasW, canvasH) {
  const splats = [];

  // ── Update facing based on mouse position ──
  const playerCX = player.x + player.w / 2;
  const playerCY = player.y + player.h / 2;
  if (mouse.x > playerCX) player.facing = 1;
  else if (mouse.x < playerCX) player.facing = -1;

  // ── 1. Horizontal movement ──
  if (keys['KeyA'] || keys['ArrowLeft']) {
    player.vx = -MOVE_SPEED;
    player.facing = -1;
  } else if (keys['KeyD'] || keys['ArrowRight']) {
    player.vx = MOVE_SPEED;
    player.facing = 1;
  } else {
    player.vx *= 0.85;
  }

  // ── 2. Jump ──
  if ((keys['KeyW'] || keys['ArrowUp']) && player.grounded) {
    player.vy = JUMP_SPEED;
    player.grounded = false;
  }

  // ── 3. Dodge ──
  if ((keys['ShiftLeft'] || keys['ShiftRight']) && player.dodgeCooldown <= 0 && !player.dodging) {
    player.dodging = true;
    player.dodgeTimer = DODGE_DURATION;
    player.vx = player.facing * DODGE_SPEED;
    player.vy = 0;
    player.dodgeCooldown = DODGE_COOLDOWN;
  }

  // ── 4. Dodge timer ──
  if (player.dodging) {
    player.dodgeTimer -= dt;
    if (player.dodgeTimer <= 0) {
      player.dodging = false;
    }
  }
  player.dodgeCooldown -= dt;

  // ── 5. Gravity ──
  if (!player.dodging) {
    player.vy += GRAVITY * dt;
  }

  // ── 6. Position update ──
  player.x += player.vx * dt;
  player.y += player.vy * dt;

  // ── 7. AABB collision ──
  player.grounded = false;

  for (const plat of getPlatforms()) {
    // Player rect
    const px = player.x, py = player.y, pw = player.w, ph = player.h;
    // Platform rect
    const bx = plat.x, by = plat.y, bw = plat.w, bh = plat.h;

    // Check overlap
    const overlapX = Math.min(px + pw, bx + bw) - Math.max(px, bx);
    const overlapY = Math.min(py + ph, by + bh) - Math.max(py, by);

    if (overlapX <= 0 || overlapY <= 0) continue; // no collision

    // Resolve along smallest overlap axis
    if (overlapX < overlapY) {
      // Push horizontally
      if (px + pw / 2 < bx + bw / 2) {
        player.x -= overlapX; // push left
      } else {
        player.x += overlapX; // push right
      }
      player.vx = 0;
    } else {
      // Push vertically
      if (py + ph / 2 < by + bh / 2) {
        // Player above platform — landed on top
        player.y -= overlapY;
        player.vy = 0;
        player.grounded = true;
      } else {
        // Player below platform — hit ceiling
        player.y += overlapY;
        player.vy = 0;
      }
    }
  }

  // ── 8. Clamp to canvas bounds ──
  if (player.x < 0) { player.x = 0; player.vx = 0; }
  if (player.x + player.w > canvasW) { player.x = canvasW - player.w; player.vx = 0; }
  if (player.y < 0) { player.y = 0; player.vy = 0; }
  if (player.y + player.h > canvasH) { player.y = canvasH - player.h; player.vy = 0; player.grounded = true; }

  // ── Shooting splats ──
  player.shootTimer -= dt;

  if (shooting && player.shootTimer <= 0) {
    player.shootTimer = 1.0 / state.shootRate;

    const [aimX, aimY] = normalize(
      mouse.x - (player.x + player.w / 2),
      mouse.y - (player.y + player.h * 0.3),
    );

    const dx = aimX * state.shootForce;
    const dy = -aimY * state.shootForce; // flip Y for UV space

    // Blend base and accent colors, scaled by dye amount
    const blend = state.colorBlend;
    const da = state.dyeAmount;
    const color = [
      (state.baseColor[0] * (1 - blend) + state.accentColor[0] * blend) * da,
      (state.baseColor[1] * (1 - blend) + state.accentColor[1] * blend) * da,
      (state.baseColor[2] * (1 - blend) + state.accentColor[2] * blend) * da,
    ];

    // Emit jet — multiple splats along aim direction from gun tip outward
    const jetSteps = 4;
    const startDist = 15;   // px from player center
    const stepDist = 20;    // px between each splat
    const baseRadius = state.shootRadius;

    for (let i = 0; i < jetSteps; i++) {
      const dist = startDist + i * stepDist;
      const px = player.x + player.w / 2 + aimX * dist;
      const py = player.y + player.h * 0.3 + aimY * dist;

      const su = px / canvasW;
      const sv = 1.0 - py / canvasH;

      // Taper: radius shrinks, force stays strong along the jet
      const t = i / (jetSteps - 1);
      const r = baseRadius * (1.0 - t * 0.5); // taper to 50% at tip

      splats.push({
        x: su, y: sv,
        dx, dy,
        color,
        radius: r,
        temp: state.tempAmount * 0.5,
      });
    }
  }

  // ── Movement effector — wake-style push ──
  // Single splat in movement direction, only when moving.
  // Pushes fluid ahead like a boat wake — no dye added.
  {
    const speed = Math.sqrt(player.vx * player.vx + player.vy * player.vy);

    if (speed > 20) {
      const cu = (player.x + player.w / 2) / canvasW;
      const cv = 1.0 - (player.y + player.h / 2) / canvasH;

      let force = state.repelForce * state.effectorStrength;
      if (!player.grounded) force *= 1.5;
      if (player.dodging) force *= 3;

      // Normalize movement direction, push fluid that way
      const dvx = (player.vx / speed) * force;
      const dvy = -(player.vy / speed) * force;

      splats.push({
        x: cu, y: cv,
        dx: dvx, dy: dvy,
        color: null,
        radius: state.repelRadius,
        temp: 0,
      });
    }
  }

  return splats;
}

// ── Exported: render ──────────────────────────────────────────────────────────

function render(glCtx, canvasW, canvasH) {
  const g = glCtx;
  g.enable(g.BLEND);
  g.blendFunc(g.SRC_ALPHA, g.ONE_MINUS_SRC_ALPHA);
  g.bindVertexArray(emptyVAO);

  // ── Draw platforms ──
  g.useProgram(gameProgram);
  g.uniform2f(uniforms.game.uResolution, canvasW, canvasH);

  for (const plat of getPlatforms()) {
    // Platform body
    g.uniform4f(uniforms.game.uRect, plat.x, plat.y, plat.w, plat.h);
    g.uniform4f(uniforms.game.uColor, 0.08, 0.08, 0.12, 0.85);
    g.drawArrays(g.TRIANGLES, 0, 6);

    // Top edge highlight
    g.uniform4f(uniforms.game.uRect, plat.x + 1, plat.y, plat.w - 2, 1);
    g.uniform4f(uniforms.game.uColor, 0.2, 0.2, 0.28, 0.6);
    g.drawArrays(g.TRIANGLES, 0, 6);
  }

  // ── Draw player body ──
  const alpha = player.dodging ? 0.4 : 0.9;

  g.useProgram(gameProgram);
  g.uniform2f(uniforms.game.uResolution, canvasW, canvasH);
  g.uniform4f(uniforms.game.uRect, player.x, player.y, player.w, player.h);
  g.uniform4f(uniforms.game.uColor, 0.1, 0.1, 0.15, alpha);
  g.drawArrays(g.TRIANGLES, 0, 6);

  // ── Draw player head (circle) — offset in facing direction ──
  const headCX = player.x + player.w / 2 + player.facing * 2;
  const headCY = player.y - 4;
  const headR = 8;

  g.useProgram(circleProgram);
  g.uniform2f(uniforms.circle.uResolution, canvasW, canvasH);
  g.uniform2f(uniforms.circle.uCenter, headCX, headCY);
  g.uniform1f(uniforms.circle.uRadius, headR);
  g.uniform4f(uniforms.circle.uColor, 0.15, 0.15, 0.22, alpha);
  g.uniform1f(uniforms.circle.uHollow, 0.0);
  g.drawArrays(g.TRIANGLES, 0, 6);

  // ── Draw gun arm ──
  const shoulderX = player.x + player.w / 2 + player.facing * 5;
  const shoulderY = player.y + player.h * 0.3 - 4;

  const [aimX, aimY] = normalize(
    mouse.x - (player.x + player.w / 2),
    mouse.y - (player.y + player.h * 0.3),
  );

  const gunTipX = player.x + player.w / 2 + aimX * 15;
  const gunTipY = player.y + player.h * 0.3 + aimY * 15;

  g.useProgram(lineProgram);
  g.uniform2f(uniforms.line.uResolution, canvasW, canvasH);
  g.uniform2f(uniforms.line.uStart, shoulderX, shoulderY);
  g.uniform2f(uniforms.line.uEnd, gunTipX, gunTipY);
  g.uniform1f(uniforms.line.uWidth, 3.0);
  g.uniform4f(uniforms.line.uColor, 0.2, 0.2, 0.3, alpha);
  g.drawArrays(g.TRIANGLES, 0, 6);

  // ── Draw crosshair ──
  g.useProgram(circleProgram);
  g.uniform2f(uniforms.circle.uResolution, canvasW, canvasH);

  // Outer ring
  g.uniform2f(uniforms.circle.uCenter, mouse.x, mouse.y);
  g.uniform1f(uniforms.circle.uRadius, 8.0);
  g.uniform4f(uniforms.circle.uColor, 1.0, 1.0, 1.0, 0.3);
  g.uniform1f(uniforms.circle.uHollow, 1.0);
  g.drawArrays(g.TRIANGLES, 0, 6);

  // Center dot
  g.uniform1f(uniforms.circle.uRadius, 2.0);
  g.uniform4f(uniforms.circle.uColor, 1.0, 1.0, 1.0, 0.5);
  g.uniform1f(uniforms.circle.uHollow, 0.0);
  g.drawArrays(g.TRIANGLES, 0, 6);

  g.bindVertexArray(null);
  g.disable(g.BLEND);
}

// ── Exported: getPlayer ───────────────────────────────────────────────────────

function getPlayer() {
  return player;
}
