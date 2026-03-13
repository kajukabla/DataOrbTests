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
};

// ── Input State ───────────────────────────────────────────────────────────────

const keys = {};
const mouse = { x: 400, y: 300, down: false };
let shootPressed = false;

// ── Jet Shots (travelling projectiles that splat along their path) ───────────

const shots = [];

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
    if (e.code === 'Space' && !e.repeat) {
      shootPressed = true;
      e.preventDefault();
    }
  });

  document.addEventListener('keyup', (e) => {
    keys[e.code] = false;
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
  const repellerSplats = [];

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

  // ── Shooting — single press spawns a travelling jet projectile ──
  if (shootPressed) {
    shootPressed = false;

    const [aimX, aimY] = normalize(
      mouse.x - (player.x + player.w / 2),
      mouse.y - (player.y + player.h * 0.3),
    );

    // Spawn position: gun tip in UV space
    const tipX = player.x + player.w / 2 + aimX * 15;
    const tipY = player.y + player.h * 0.3 + aimY * 15;

    // Blend color
    const blend = state.colorBlend;
    const dyeI = state.jetDyeIntensity;
    const travelSpeed = 0.35 + Math.max(0.25, state.jetSpeed) * 1.15;
    const life = 0.07 + state.jetDuration * 0.16;
    const forceScale = Math.max(0.15, 0.45 + state.jetForce * 1.5);

    shots.push({
      x: tipX / canvasW,
      y: 1.0 - tipY / canvasH,
      vx: aimX * travelSpeed,
      vy: -aimY * travelSpeed,
      life,
      maxLife: life,
      forceScale,
      radius: state.jetRadius,
      color: [
        (state.baseColor[0] * (1 - blend) + state.accentColor[0] * blend) * dyeI,
        (state.baseColor[1] * (1 - blend) + state.accentColor[1] * blend) * dyeI,
        (state.baseColor[2] * (1 - blend) + state.accentColor[2] * blend) * dyeI,
      ],
    });
  }

  // ── Update travelling jet shots — splat dye along their path each frame ──
  for (let i = shots.length - 1; i >= 0; i--) {
    const s = shots[i];
    const prevX = s.x;
    const prevY = s.y;

    s.x += s.vx * dt;
    s.y += s.vy * dt;
    s.life -= dt;

    // Remove dead or out-of-bounds shots
    if (s.life <= 0 || s.x < -0.05 || s.x > 1.05 || s.y < -0.05 || s.y > 1.05) {
      shots.splice(i, 1);
      continue;
    }

    // Platform collision — convert UV to pixel, check AABB, kill on hit
    const pixX = s.x * canvasW;
    const pixY = (1.0 - s.y) * canvasH;
    let hitPlatform = false;
    for (const plat of getPlatforms()) {
      if (pixX >= plat.x && pixX <= plat.x + plat.w &&
          pixY >= plat.y && pixY <= plat.y + plat.h) {
        hitPlatform = true;
        break;
      }
    }
    if (hitPlatform) {
      // Splat a final burst on impact then remove
      splats.push({
        x: s.x, y: s.y,
        dx: s.vx * s.forceScale * 30,
        dy: s.vy * s.forceScale * 30,
        color: s.color,
        radius: s.radius * 1.5,
        temp: state.tempAmount * 0.5,
      });
      shots.splice(i, 1);
      continue;
    }

    // Splat along the path this frame (matches main app jet behavior)
    const stepX = s.x - prevX;
    const stepY = s.y - prevY;
    if (Math.abs(stepX) + Math.abs(stepY) < 1e-6) continue;

    const ageT = 1 - s.life / Math.max(1e-6, s.maxLife);
    const fade = Math.max(0.25, 1 - ageT * 0.7);
    const force = s.forceScale * fade;
    const dyeScale = force * 2.0;  // dye stronger than force, like main app
    const radius = s.radius * (1 - ageT * 0.22);

    const FORCE_BASE = 6000;  // matches main app BURST_SPLAT_FORCE_BASE
    const samples = 2;
    for (let j = 1; j <= samples; j++) {
      const t = j / samples;
      const px = prevX + stepX * t;
      const py = prevY + stepY * t;

      splats.push({
        x: px, y: py,
        dx: stepX * FORCE_BASE * force,
        dy: stepY * FORCE_BASE * force,
        color: [s.color[0] * dyeScale, s.color[1] * dyeScale, s.color[2] * dyeScale],
        radius,
        temp: state.tempAmount * 0.5,
      });
    }

    // Drag (configurable, main app default: 2.2)
    const drag = Math.max(0, 1 - dt * state.jetDrag);
    s.vx *= drag;
    s.vy *= drag;
  }

  // ── Player repeller — ring of offset splats pushing outward ──
  // Applied AFTER pressure projection so they aren't canceled.
  {
    const cx = (player.x + player.w / 2) / canvasW;
    const headTop = player.y - 12;
    const bodyBottom = player.y + player.h;
    const cy = 1.0 - ((headTop + bodyBottom) / 2) / canvasH;

    const force = state.repelForce * state.effectorStrength;
    const ringRadius = state.repelRadius * 0.5;
    const splatRadius = state.repelRadius * 0.4;
    const numSplats = 8;

    for (let i = 0; i < numSplats; i++) {
      const angle = (i / numSplats) * Math.PI * 2;
      const nx = Math.cos(angle);
      const ny = Math.sin(angle);

      repellerSplats.push({
        x: cx + nx * ringRadius,
        y: cy + ny * ringRadius,
        dx: nx * force,
        dy: ny * force,
        color: null,
        radius: splatRadius,
      });
    }
  }

  return { splats, repellerSplats };
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
