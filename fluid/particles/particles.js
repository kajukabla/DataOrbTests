// particles.js — Pool-based particle system with curl noise

import { bounceParticle } from './platforms.js';

// ── Simplex-like 2D noise (hash-based) ─────────────────────────────────────

function hash(x, y) {
  let h = x * 374761393 + y * 668265263;
  h = (h ^ (h >> 13)) * 1274126177;
  return (h ^ (h >> 16)) / 2147483648; // [0, 2)
}

function smoothNoise(x, y) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const sx = fx * fx * (3 - 2 * fx);
  const sy = fy * fy * (3 - 2 * fy);

  const n00 = hash(ix, iy);
  const n10 = hash(ix + 1, iy);
  const n01 = hash(ix, iy + 1);
  const n11 = hash(ix + 1, iy + 1);

  return n00 * (1 - sx) * (1 - sy) + n10 * sx * (1 - sy) +
         n01 * (1 - sx) * sy + n11 * sx * sy;
}

function curlNoise2D(x, y, time) {
  const scale = 0.005; // spatial frequency
  const eps = 0.5;
  const px = x * scale + time * 0.3;
  const py = y * scale + time * 0.3;

  // Finite difference curl: (dN/dy, -dN/dx)
  const dNdy = smoothNoise(px, py + eps) - smoothNoise(px, py - eps);
  const dNdx = smoothNoise(px + eps, py) - smoothNoise(px - eps, py);

  return { cx: dNdy / (2 * eps), cy: -dNdx / (2 * eps) };
}

// ── Particle Pool ──────────────────────────────────────────────────────────

const MAX_PARTICLES = 10000;
const pool = [];
let activeCount = 0;
let time = 0;

function createParticle() {
  return {
    x: 0, y: 0,
    vx: 0, vy: 0,
    life: 0, maxLife: 0,
    size: 4,
    alive: false,
    r: 1, g: 0.8, b: 0.4, // warm orange default
  };
}

// Pre-allocate
for (let i = 0; i < MAX_PARTICLES; i++) {
  pool.push(createParticle());
}

function spawn() {
  for (let i = 0; i < MAX_PARTICLES; i++) {
    if (!pool[i].alive) return pool[i];
  }
  return null;
}

// ── Public API ─────────────────────────────────────────────────────────────

export function emitJet(x, y, aimX, aimY, count, speed, size, lifetime, emitterRadius) {
  // Jet: distribute particles along the aim direction in a line,
  // with emitterRadius controlling the perpendicular spread (tube thickness)
  const jetLength = 80; // total length of the jet in px
  const perp = { x: -aimY, y: aimX }; // perpendicular to aim

  for (let i = 0; i < count; i++) {
    const p = spawn();
    if (!p) break;

    // Position along jet line
    const t = i / (count - 1 || 1); // 0 at base, 1 at tip
    const dist = t * jetLength;

    // Perpendicular offset within emitter radius
    const offset = (Math.random() - 0.5) * 2 * emitterRadius;

    p.x = x + aimX * dist + perp.x * offset;
    p.y = y + aimY * dist + perp.y * offset;

    // All particles go same direction, speed tapers toward tip
    const spd = speed * (1.0 - t * 0.3);
    p.vx = aimX * spd;
    p.vy = aimY * spd;

    // Size tapers toward tip
    p.size = size * (1.0 - t * 0.5) * (0.7 + Math.random() * 0.6);
    p.life = lifetime * (0.7 + Math.random() * 0.6);
    p.maxLife = p.life;
    p.alive = true;

    // Slight color variation
    const hueShift = Math.random() * 0.2 - 0.1;
    p.r = Math.min(1, 1.0 + hueShift);
    p.g = Math.min(1, 0.7 + hueShift + Math.random() * 0.2);
    p.b = Math.min(1, 0.3 + Math.random() * 0.2);
  }
}

export function updateParticles(dt, platforms, playerCX, playerCY, curlAmount, repelSize, repelForce) {
  time += dt;
  activeCount = 0;
  const drag = 2.0;
  const gravity = 200; // lighter than player gravity

  for (let i = 0; i < MAX_PARTICLES; i++) {
    const p = pool[i];
    if (!p.alive) continue;

    p.life -= dt;
    if (p.life <= 0) {
      p.alive = false;
      continue;
    }

    activeCount++;

    // Drag
    p.vx *= (1 - drag * dt);
    p.vy *= (1 - drag * dt);

    // Gravity
    p.vy += gravity * dt;

    // Curl noise
    if (curlAmount > 0) {
      const { cx, cy } = curlNoise2D(p.x, p.y, time);
      p.vx += cx * curlAmount * dt;
      p.vy += cy * curlAmount * dt;
    }

    // Player repeller
    const dx = p.x - playerCX;
    const dy = p.y - playerCY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < repelSize && dist > 0.1) {
      const t = 1 - dist / repelSize;
      const nx = dx / dist, ny = dy / dist;

      // Force-based push
      const force = repelForce * t;
      p.vx += nx * force * dt;
      p.vy += ny * force * dt;

      // Hard push: if particle is deep inside (inner 60%), push it to the edge
      if (t > 0.4) {
        const pushDist = repelSize * 0.6;
        if (dist < pushDist) {
          p.x = playerCX + nx * pushDist;
          p.y = playerCY + ny * pushDist;
        }
      }
    }

    // Position
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Platform bounce
    bounceParticle(p, platforms, 0.3);
  }
}

export function renderParticles(ctx) {
  for (let i = 0; i < MAX_PARTICLES; i++) {
    const p = pool[i];
    if (!p.alive) continue;

    const t = p.life / p.maxLife; // 1 at birth, 0 at death
    const alpha = t * 0.9;
    const size = p.size * (0.3 + t * 0.7); // shrink to 30%

    ctx.beginPath();
    ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${Math.round(p.r * 255)}, ${Math.round(p.g * 255)}, ${Math.round(p.b * 255)}, ${alpha})`;
    ctx.fill();
  }
}

export function getActiveCount() { return activeCount; }
