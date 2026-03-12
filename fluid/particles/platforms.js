// platforms.js — Platform definitions and collision helpers

export function buildPlatforms(canvasW, canvasH) {
  const groundY = canvasH - 60;
  return [
    { x: 0, y: groundY, w: canvasW, h: 60 },                          // ground
    { x: canvasW * 0.10, y: canvasH * 0.60, w: canvasW * 0.14, h: 16 },
    { x: canvasW * 0.36, y: canvasH * 0.50, w: canvasW * 0.16, h: 16 },
    { x: canvasW * 0.60, y: canvasH * 0.42, w: canvasW * 0.12, h: 16 },
    { x: canvasW * 0.78, y: canvasH * 0.55, w: canvasW * 0.15, h: 16 },
    { x: canvasW * 0.20, y: canvasH * 0.32, w: canvasW * 0.10, h: 16 },
    { x: canvasW * 0.50, y: canvasH * 0.25, w: canvasW * 0.18, h: 16 },
    { x: canvasW * 0.80, y: canvasH * 0.30, w: canvasW * 0.12, h: 16 },
    { x: 0, y: 0, w: 10, h: canvasH },                                // left wall
    { x: canvasW - 10, y: 0, w: 10, h: canvasH },                    // right wall
  ];
}

// Resolve AABB collision for an object { x, y, w, h, vx, vy }
// Returns true if grounded (landed on top of something)
export function resolveAABB(obj, platforms) {
  let grounded = false;

  for (const plat of platforms) {
    const overlapX = Math.min(obj.x + obj.w, plat.x + plat.w) - Math.max(obj.x, plat.x);
    const overlapY = Math.min(obj.y + obj.h, plat.y + plat.h) - Math.max(obj.y, plat.y);

    if (overlapX <= 0 || overlapY <= 0) continue;

    if (overlapX < overlapY) {
      if (obj.x + obj.w / 2 < plat.x + plat.w / 2) {
        obj.x -= overlapX;
      } else {
        obj.x += overlapX;
      }
      obj.vx = 0;
    } else {
      if (obj.y + obj.h / 2 < plat.y + plat.h / 2) {
        obj.y -= overlapY;
        obj.vy = 0;
        grounded = true;
      } else {
        obj.y += overlapY;
        obj.vy = 0;
      }
    }
  }

  return grounded;
}

// Bounce a particle off platforms. Returns true if bounced.
export function bounceParticle(p, platforms, restitution = 0.3) {
  let bounced = false;

  for (const plat of platforms) {
    const r = p.size * 0.5;
    const overlapX = Math.min(p.x + r, plat.x + plat.w) - Math.max(p.x - r, plat.x);
    const overlapY = Math.min(p.y + r, plat.y + plat.h) - Math.max(p.y - r, plat.y);

    if (overlapX <= 0 || overlapY <= 0) continue;

    bounced = true;

    if (overlapX < overlapY) {
      if (p.x < plat.x + plat.w / 2) {
        p.x -= overlapX;
      } else {
        p.x += overlapX;
      }
      p.vx *= -restitution;
    } else {
      if (p.y < plat.y + plat.h / 2) {
        p.y -= overlapY;
      } else {
        p.y += overlapY;
      }
      p.vy *= -restitution;
    }
  }

  return bounced;
}
