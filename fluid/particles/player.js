// player.js — Player physics, input, collision, rendering

import { resolveAABB } from './platforms.js';

const GRAVITY = 980;
const MOVE_SPEED = 250;
const JUMP_SPEED = -420;
const DODGE_SPEED = 600;
const DODGE_DURATION = 0.15;
const DODGE_COOLDOWN = 0.5;

const player = {
  x: 400, y: 500,
  vx: 0, vy: 0,
  w: 20, h: 32,
  grounded: false,
  facing: 1,
  dodgeCooldown: 0,
  dodging: false,
  dodgeTimer: 0,
};

const keys = {};
const mouse = { x: 400, y: 300 };
let shootPressed = false;

export function initPlayer(canvas) {
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
}

export function getPlayer() { return player; }
export function getMouse() { return mouse; }
export function consumeShoot() {
  if (shootPressed) { shootPressed = false; return true; }
  return false;
}

export function updatePlayer(dt, platforms, canvasW, canvasH) {
  // Facing from mouse
  const cx = player.x + player.w / 2;
  if (mouse.x > cx) player.facing = 1;
  else if (mouse.x < cx) player.facing = -1;

  // Horizontal movement
  if (keys['KeyA'] || keys['ArrowLeft']) {
    player.vx = -MOVE_SPEED;
    player.facing = -1;
  } else if (keys['KeyD'] || keys['ArrowRight']) {
    player.vx = MOVE_SPEED;
    player.facing = 1;
  } else {
    player.vx *= 0.85;
  }

  // Jump
  if ((keys['KeyW'] || keys['ArrowUp']) && player.grounded) {
    player.vy = JUMP_SPEED;
    player.grounded = false;
  }

  // Dodge
  if ((keys['ShiftLeft'] || keys['ShiftRight']) && player.dodgeCooldown <= 0 && !player.dodging) {
    player.dodging = true;
    player.dodgeTimer = DODGE_DURATION;
    player.vx = player.facing * DODGE_SPEED;
    player.vy = 0;
    player.dodgeCooldown = DODGE_COOLDOWN;
  }

  if (player.dodging) {
    player.dodgeTimer -= dt;
    if (player.dodgeTimer <= 0) player.dodging = false;
  }
  player.dodgeCooldown -= dt;

  // Gravity
  if (!player.dodging) player.vy += GRAVITY * dt;

  // Position
  player.x += player.vx * dt;
  player.y += player.vy * dt;

  // Platform collision
  player.grounded = resolveAABB(player, platforms);

  // Canvas bounds
  if (player.x < 0) { player.x = 0; player.vx = 0; }
  if (player.x + player.w > canvasW) { player.x = canvasW - player.w; player.vx = 0; }
  if (player.y < 0) { player.y = 0; player.vy = 0; }
  if (player.y + player.h > canvasH) { player.y = canvasH - player.h; player.vy = 0; player.grounded = true; }
}

export function renderPlayer(ctx) {
  const alpha = player.dodging ? 0.4 : 0.9;

  // Body
  ctx.fillStyle = `rgba(25, 25, 38, ${alpha})`;
  ctx.fillRect(player.x, player.y, player.w, player.h);

  // Head
  const headCX = player.x + player.w / 2 + player.facing * 2;
  const headCY = player.y - 4;
  ctx.beginPath();
  ctx.arc(headCX, headCY, 8, 0, Math.PI * 2);
  ctx.fillStyle = `rgba(38, 38, 56, ${alpha})`;
  ctx.fill();

  // Gun arm
  const shoulderX = player.x + player.w / 2 + player.facing * 5;
  const shoulderY = player.y + player.h * 0.3 - 4;

  const dx = mouse.x - (player.x + player.w / 2);
  const dy = mouse.y - (player.y + player.h * 0.3);
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  const ax = dx / len, ay = dy / len;

  const gunTipX = player.x + player.w / 2 + ax * 15;
  const gunTipY = player.y + player.h * 0.3 + ay * 15;

  ctx.beginPath();
  ctx.moveTo(shoulderX, shoulderY);
  ctx.lineTo(gunTipX, gunTipY);
  ctx.strokeStyle = `rgba(50, 50, 75, ${alpha})`;
  ctx.lineWidth = 3;
  ctx.stroke();

  // Crosshair
  ctx.beginPath();
  ctx.arc(mouse.x, mouse.y, 8, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(mouse.x, mouse.y, 2, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
  ctx.fill();
}
