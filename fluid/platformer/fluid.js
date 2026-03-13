// fluid.js — WebGL2 fluid simulation engine (ES module)

import {
  fullscreenVS, splatFS, curlFS, vorticityFS, divergenceFS, pressureFS,
  gradSubFS, advectFS, buoyancyFS, curlNoiseFS, boundaryFS, displayFS,
  bloomExtractFS, bloomBlurFS, bloomCompositeFS
} from './shaders.js';
import { state } from './state.js';

const SIM_RES = 512;

// ── Shader helpers ──────────────────────────────────────────────────────────

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error('Shader compile error: ' + info);
  }
  return shader;
}

function createProgram(gl, vsSrc, fsSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error('Program link error: ' + info);
  }

  // Cache all uniform locations
  const uniforms = {};
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {
    const info = gl.getActiveUniform(program, i);
    // Handle array uniforms — strip [0] for the base name lookup
    const name = info.name.replace(/\[0\]$/, '');
    uniforms[name] = gl.getUniformLocation(program, info.name);
    // Also store indexed entries for array uniforms
    if (info.size > 1) {
      for (let j = 0; j < info.size; j++) {
        const arrName = name + '[' + j + ']';
        uniforms[arrName] = gl.getUniformLocation(program, arrName);
      }
    }
  }

  gl.deleteShader(vs);
  gl.deleteShader(fs);

  return { program, uniforms };
}

// ── FBO helpers ─────────────────────────────────────────────────────────────

function createFBO(gl, w, h, internalFormat, format, type, filter) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

  return { texture, fbo, width: w, height: h };
}

function createDoubleFBO(gl, w, h, internalFormat, format, type, filter) {
  let read = createFBO(gl, w, h, internalFormat, format, type, filter);
  let write = createFBO(gl, w, h, internalFormat, format, type, filter);
  return {
    get read() { return read; },
    get write() { return write; },
    swap() { const tmp = read; read = write; write = tmp; }
  };
}

// ── Main init ───────────────────────────────────────────────────────────────

export function initFluid(canvas) {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false
  });
  if (!gl) throw new Error('WebGL2 not supported');

  const extFloat = gl.getExtension('EXT_color_buffer_float');
  if (!extFloat) throw new Error('EXT_color_buffer_float required');

  const extLinear = gl.getExtension('OES_texture_float_linear');
  // optional — some devices don't support it

  // Empty VAO for fullscreen triangle draws
  const vao = gl.createVertexArray();

  // ── Compile programs ────────────────────────────────────────────────────

  const programs = {
    splat:          createProgram(gl, fullscreenVS, splatFS),
    curl:           createProgram(gl, fullscreenVS, curlFS),
    vorticity:      createProgram(gl, fullscreenVS, vorticityFS),
    divergence:     createProgram(gl, fullscreenVS, divergenceFS),
    pressure:       createProgram(gl, fullscreenVS, pressureFS),
    gradSub:        createProgram(gl, fullscreenVS, gradSubFS),
    advect:         createProgram(gl, fullscreenVS, advectFS),
    buoyancy:       createProgram(gl, fullscreenVS, buoyancyFS),
    curlNoise:      createProgram(gl, fullscreenVS, curlNoiseFS),
    boundary:       createProgram(gl, fullscreenVS, boundaryFS),
    display:        createProgram(gl, fullscreenVS, displayFS),
    bloomExtract:   createProgram(gl, fullscreenVS, bloomExtractFS),
    bloomBlur:      createProgram(gl, fullscreenVS, bloomBlurFS),
    bloomComposite: createProgram(gl, fullscreenVS, bloomCompositeFS),
  };

  // ── Create FBOs ─────────────────────────────────────────────────────────

  const velocity    = createDoubleFBO(gl, SIM_RES, SIM_RES, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);
  const dye         = createDoubleFBO(gl, SIM_RES, SIM_RES, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);
  const pressure    = createDoubleFBO(gl, SIM_RES, SIM_RES, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
  const temperature = createDoubleFBO(gl, SIM_RES, SIM_RES, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);
  const divergence  = createFBO(gl, SIM_RES, SIM_RES, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
  const curl        = createFBO(gl, SIM_RES, SIM_RES, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
  const boundary    = createFBO(gl, SIM_RES, SIM_RES, gl.R8, gl.RED, gl.UNSIGNED_BYTE, gl.NEAREST);

  // Full-resolution scene FBO for bloom composite
  let sceneFBO = createFBO(gl, canvas.width, canvas.height, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);

  // Bloom FBOs at quarter canvas resolution
  let bloomW = Math.floor(canvas.width / 4) || 1;
  let bloomH = Math.floor(canvas.height / 4) || 1;
  let bloomPing    = createFBO(gl, bloomW, bloomH, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);
  let bloomPong    = createFBO(gl, bloomW, bloomH, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);

  const texelSize = [1.0 / SIM_RES, 1.0 / SIM_RES];

  // Track canvas dimensions for aspect ratio
  let canvasW = canvas.width;
  let canvasH = canvas.height;

  // ── Blit helper ─────────────────────────────────────────────────────────

  function blit(target) {
    gl.bindVertexArray(vao);
    if (target) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      gl.viewport(0, 0, target.width, target.height);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // ── Bind texture to unit ────────────────────────────────────────────────

  function bindTex(unit, texture) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
  }

  // ── Use a program and return its uniforms shortcut ──────────────────────

  function use(prog) {
    gl.useProgram(prog.program);
    return prog.uniforms;
  }

  // ── Splat ───────────────────────────────────────────────────────────────

  function splat(x, y, dx, dy, color, radius) {
    const aspect = canvasW / canvasH;

    // Splat velocity
    let u = use(programs.splat);
    gl.uniform2f(u.uPoint, x, y);
    gl.uniform3f(u.uColor, dx, dy, 0.0);
    gl.uniform1f(u.uRadius, radius);
    gl.uniform1f(u.uAspect, aspect);
    bindTex(0, velocity.read.texture);
    gl.uniform1i(u.uTarget, 0);
    blit(velocity.write);
    velocity.swap();

    // Splat dye (skip if color is null — velocity-only effector)
    if (color) {
      u = use(programs.splat);
      gl.uniform2f(u.uPoint, x, y);
      gl.uniform3f(u.uColor, color[0], color[1], color[2]);
      gl.uniform1f(u.uRadius, radius);
      gl.uniform1f(u.uAspect, aspect);
      bindTex(0, dye.read.texture);
      gl.uniform1i(u.uTarget, 0);
      blit(dye.write);
      dye.swap();
    }

    // Splat temperature if enabled (skip for velocity-only splats)
    if (color && state.tempAmount > 0) {
      u = use(programs.splat);
      gl.uniform2f(u.uPoint, x, y);
      gl.uniform3f(u.uColor, state.tempAmount, 0.0, 0.0);
      gl.uniform1f(u.uRadius, radius);
      gl.uniform1f(u.uAspect, aspect);
      bindTex(0, temperature.read.texture);
      gl.uniform1i(u.uTarget, 0);
      blit(temperature.write);
      temperature.swap();
    }
  }

  // ── Simulation step ─────────────────────────────────────────────────────

  let noiseTime = 0;

  function step(dt, platforms, postProjectionSplats) {
    const scaledDt = dt * state.simSpeed;
    noiseTime += scaledDt;
    let u;

    // 1. Update boundary mask
    if (state.platformBoundaries && platforms && platforms.length > 0) {
      u = use(programs.boundary);
      gl.uniform2f(u.uResolution, SIM_RES, SIM_RES);
      gl.uniform1i(u.uPlatformCount, platforms.length);
      for (let i = 0; i < platforms.length; i++) {
        const p = platforms[i];
        const loc = u['uPlatforms[' + i + ']'];
        if (loc) {
          gl.uniform4f(loc,
            p.x / canvasW,
            1.0 - ((p.y + p.h) / canvasH),
            p.w / canvasW,
            p.h / canvasH
          );
        }
      }
      blit(boundary);
    }

    // 2. Curl noise
    if (state.noiseAmount > 0) {
      u = use(programs.curlNoise);
      gl.uniform2f(u.uTexelSize, texelSize[0], texelSize[1]);
      gl.uniform1f(u.uTime, noiseTime);
      gl.uniform1f(u.uAmount, state.noiseAmount);
      gl.uniform1i(u.uNoiseType, state.noiseType);
      gl.uniform1i(u.uBehavior, state.noiseBehavior);
      gl.uniform1i(u.uMapping, state.noiseMapping);
      gl.uniform1f(u.uFrequency, state.noiseFrequency);
      gl.uniform1f(u.uSpeed, state.noiseSpeed);
      gl.uniform1f(u.uWarp, state.noiseWarp);
      gl.uniform1f(u.uSharpness, state.noiseSharpness);
      gl.uniform1f(u.uAnisotropy, state.noiseAnisotropy);
      gl.uniform1f(u.uBlend, state.noiseBlend);
      bindTex(0, velocity.read.texture);
      gl.uniform1i(u.uVelocity, 0);
      blit(velocity.write);
      velocity.swap();
    }

    // 3. Compute curl
    u = use(programs.curl);
    gl.uniform2f(u.uTexelSize, texelSize[0], texelSize[1]);
    bindTex(0, velocity.read.texture);
    gl.uniform1i(u.uVelocity, 0);
    blit(curl);

    // 4. Vorticity confinement
    if (state.curlStrength > 0) {
      u = use(programs.vorticity);
      gl.uniform2f(u.uTexelSize, texelSize[0], texelSize[1]);
      gl.uniform1f(u.uDt, scaledDt);
      gl.uniform1f(u.uStrength, state.curlStrength);
      bindTex(0, velocity.read.texture);
      gl.uniform1i(u.uVelocity, 0);
      bindTex(1, curl.texture);
      gl.uniform1i(u.uCurl, 1);
      blit(velocity.write);
      velocity.swap();
    }

    // 5. Compute divergence
    u = use(programs.divergence);
    gl.uniform2f(u.uTexelSize, texelSize[0], texelSize[1]);
    bindTex(0, velocity.read.texture);
    gl.uniform1i(u.uVelocity, 0);
    bindTex(1, boundary.texture);
    gl.uniform1i(u.uBoundary, 1);
    blit(divergence);

    // 6-7. Jacobi pressure solve
    // Apply pressure decay before solving
    // (handled implicitly — pressure decays through iterations)
    u = use(programs.pressure);
    gl.uniform2f(u.uTexelSize, texelSize[0], texelSize[1]);
    for (let i = 0; i < state.pressureIters; i++) {
      gl.useProgram(programs.pressure.program);
      bindTex(0, pressure.read.texture);
      gl.uniform1i(programs.pressure.uniforms.uPressure, 0);
      bindTex(1, divergence.texture);
      gl.uniform1i(programs.pressure.uniforms.uDivergence, 1);
      bindTex(2, boundary.texture);
      gl.uniform1i(programs.pressure.uniforms.uBoundary, 2);
      blit(pressure.write);
      pressure.swap();
    }

    // 8. Gradient subtraction
    u = use(programs.gradSub);
    gl.uniform2f(u.uTexelSize, texelSize[0], texelSize[1]);
    bindTex(0, pressure.read.texture);
    gl.uniform1i(u.uPressure, 0);
    bindTex(1, velocity.read.texture);
    gl.uniform1i(u.uVelocity, 1);
    bindTex(2, boundary.texture);
    gl.uniform1i(u.uBoundary, 2);
    blit(velocity.write);
    velocity.swap();

    // 8b. Apply post-projection velocity splats (repeller)
    // These are injected AFTER pressure projection so they survive intact.
    if (postProjectionSplats) {
      for (const s of postProjectionSplats) {
        splat(s.x, s.y, s.dx, s.dy, s.color, s.radius);
      }
    }

    // 9. Advect velocity
    u = use(programs.advect);
    gl.uniform2f(u.uTexelSize, texelSize[0], texelSize[1]);
    gl.uniform1f(u.uDt, scaledDt);
    gl.uniform1f(u.uDissipation, state.velDissipation);
    gl.uniform1f(u.uGravity, state.fluidGravity);
    gl.uniform1f(u.uMaccormack, state.maccormack);
    bindTex(0, velocity.read.texture);
    gl.uniform1i(u.uVelocity, 0);
    bindTex(1, velocity.read.texture);
    gl.uniform1i(u.uSource, 1);
    bindTex(2, boundary.texture);
    gl.uniform1i(u.uBoundary, 2);
    blit(velocity.write);
    velocity.swap();

    // 10. Advect dye
    u = use(programs.advect);
    gl.uniform2f(u.uTexelSize, texelSize[0], texelSize[1]);
    gl.uniform1f(u.uDt, scaledDt);
    gl.uniform1f(u.uDissipation, state.dyeDissipation);
    gl.uniform1f(u.uGravity, 0.0);
    gl.uniform1f(u.uMaccormack, state.maccormack);
    bindTex(0, velocity.read.texture);
    gl.uniform1i(u.uVelocity, 0);
    bindTex(1, dye.read.texture);
    gl.uniform1i(u.uSource, 1);
    bindTex(2, boundary.texture);
    gl.uniform1i(u.uBoundary, 2);
    blit(dye.write);
    dye.swap();

    // 11. Advect temperature
    if (state.tempAmount > 0) {
      u = use(programs.advect);
      gl.uniform2f(u.uTexelSize, texelSize[0], texelSize[1]);
      gl.uniform1f(u.uDt, scaledDt);
      gl.uniform1f(u.uDissipation, state.tempDissipation);
      gl.uniform1f(u.uGravity, 0.0);
      gl.uniform1f(u.uMaccormack, 0);
      bindTex(0, velocity.read.texture);
      gl.uniform1i(u.uVelocity, 0);
      bindTex(1, temperature.read.texture);
      gl.uniform1i(u.uSource, 1);
      bindTex(2, boundary.texture);
      gl.uniform1i(u.uBoundary, 2);
      blit(temperature.write);
      temperature.swap();
    }

    // 12. Buoyancy
    if (state.tempAmount > 0 && state.tempBuoyancy > 0) {
      u = use(programs.buoyancy);
      gl.uniform2f(u.uTexelSize, texelSize[0], texelSize[1]);
      gl.uniform1f(u.uDt, scaledDt);
      gl.uniform1f(u.uBuoyancy, state.tempBuoyancy);
      gl.uniform1f(u.uRadialMix, state.tempRadialMix);
      bindTex(0, velocity.read.texture);
      gl.uniform1i(u.uVelocity, 0);
      bindTex(1, temperature.read.texture);
      gl.uniform1i(u.uTemperature, 1);
      blit(velocity.write);
      velocity.swap();
    }
  }

  // ── Render ──────────────────────────────────────────────────────────────

  function render(target) {
    // Main display pass
    let u = use(programs.display);
    gl.uniform2f(u.uTexelSize, texelSize[0], texelSize[1]);
    gl.uniform2f(u.uScreenSize, gl.canvas.width, gl.canvas.height);

    bindTex(0, dye.read.texture);
    gl.uniform1i(u.uDye, 0);
    bindTex(1, velocity.read.texture);
    gl.uniform1i(u.uVelocity, 1);
    bindTex(2, pressure.read.texture);
    gl.uniform1i(u.uPressure, 2);
    bindTex(3, curl.texture);
    gl.uniform1i(u.uCurl, 3);
    bindTex(4, temperature.read.texture);
    gl.uniform1i(u.uTemperature, 4);
    bindTex(5, divergence.texture);
    gl.uniform1i(u.uDivergence, 5);

    // Color uniforms
    gl.uniform3fv(u.uBaseColor, state.baseColor);
    gl.uniform3fv(u.uAccentColor, state.accentColor);
    gl.uniform3fv(u.uTipColor, state.tipColor);
    gl.uniform3fv(u.uGlitterColor, state.glitterColor);
    gl.uniform3fv(u.uSheenColor, state.sheenColor);
    gl.uniform1f(u.uColorBlend, state.colorBlend);

    // Colormap uniforms
    gl.uniform1i(u.uColormapMode, state.colormapMode);
    gl.uniform1i(u.uColorSource, state.colorSource);
    gl.uniform1f(u.uColorGain, state.colorGain);

    // Display options
    gl.uniform1f(u.uTempColorShift, state.tempColorShift);
    gl.uniform1i(u.uDyeSoftCap, state.dyeSoftCap);
    gl.uniform1f(u.uDyeCeiling, state.dyeCeiling);

    if (state.bloomIntensity <= 0) {
      // No bloom — render directly
      blit(target);
      return;
    }

    // Bloom enabled — render scene to full-res sceneFBO, then extract/blur/composite
    blit(sceneFBO);

    // Extract bright pixels from full-res scene into quarter-res bloom
    u = use(programs.bloomExtract);
    gl.uniform1f(u.uThreshold, state.bloomThreshold);
    bindTex(0, sceneFBO.texture);
    gl.uniform1i(u.uSource, 0);
    blit(bloomPing);

    // Gaussian blur passes (horizontal then vertical, repeated)
    const passes = 3;
    for (let i = 0; i < passes; i++) {
      // Horizontal
      u = use(programs.bloomBlur);
      gl.uniform2f(u.uTexelSize, 1.0 / bloomW, 1.0 / bloomH);
      gl.uniform2f(u.uDirection, state.bloomRadius, 0.0);
      bindTex(0, bloomPing.texture);
      gl.uniform1i(u.uSource, 0);
      blit(bloomPong);

      // Vertical
      u = use(programs.bloomBlur);
      gl.uniform2f(u.uTexelSize, 1.0 / bloomW, 1.0 / bloomH);
      gl.uniform2f(u.uDirection, 0.0, state.bloomRadius);
      bindTex(0, bloomPong.texture);
      gl.uniform1i(u.uSource, 0);
      blit(bloomPing);
    }

    // Composite bloom onto scene (full-res scene + quarter-res bloom)
    u = use(programs.bloomComposite);
    gl.uniform1f(u.uIntensity, state.bloomIntensity);
    gl.uniform2f(u.uScreenSize, gl.canvas.width, gl.canvas.height);
    bindTex(0, sceneFBO.texture);
    gl.uniform1i(u.uScene, 0);
    bindTex(1, bloomPing.texture);
    gl.uniform1i(u.uBloom, 1);
    blit(target);
  }

  // ── Resize ──────────────────────────────────────────────────────────────

  function resize(w, h) {
    canvasW = w;
    canvasH = h;

    // Rebuild scene FBO at full resolution
    sceneFBO = createFBO(gl, w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);

    // Rebuild bloom FBOs at new quarter resolution
    bloomW = Math.max(1, Math.floor(w / 4));
    bloomH = Math.max(1, Math.floor(h / 4));
    bloomPing = createFBO(gl, bloomW, bloomH, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);
    bloomPong = createFBO(gl, bloomW, bloomH, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);
  }

  // ── Get textures for external use ───────────────────────────────────────

  function getTextures() {
    return { velocity, dye, pressure, curl, temperature, divergence, boundary };
  }

  return { gl, step, render, splat, resize, getTextures };
}
