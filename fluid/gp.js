// ─── Gaussian Process Math Module ────────────────────────────────────────────
// Pure JS, no dependencies. Matérn 5/2 kernel, Cholesky solver, Nelder-Mead
// hyperparameter optimization, Expected Improvement acquisition.

// ─── Linear Algebra ──────────────────────────────────────────────────────────

/**
 * In-place Cholesky decomposition of N×N symmetric positive-definite matrix.
 * A is a flat Float64Array in row-major order. Returns L (lower triangular)
 * written into A. Adds jitter for numerical stability.
 */
export function cholesky(A, n, jitter = 1e-6) {
  // Add jitter to diagonal
  for (let i = 0; i < n; i++) A[i * n + i] += jitter;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i * n + j];
      for (let k = 0; k < j; k++) {
        sum -= A[i * n + k] * A[j * n + k];
      }
      if (i === j) {
        if (sum <= 0) {
          // Retry with larger jitter
          // Undo current jitter first
          for (let ii = 0; ii < n; ii++) A[ii * n + ii] -= jitter;
          // Reset lower triangle
          // We need to restore A — caller should handle retry
          return false;
        }
        A[i * n + j] = Math.sqrt(sum);
      } else {
        A[i * n + j] = sum / A[j * n + j];
      }
    }
    // Zero upper triangle for this row
    for (let j = i + 1; j < n; j++) {
      A[i * n + j] = 0;
    }
  }
  return true;
}

/**
 * Cholesky with automatic jitter escalation.
 * Returns the L factor (modifies A in-place) or null on failure.
 */
export function choleskyWithRetry(A, n) {
  const backup = new Float64Array(A);
  let jitter = 1e-6;
  for (let attempt = 0; attempt < 4; attempt++) {
    A.set(backup);
    if (cholesky(A, n, jitter)) return A;
    jitter *= 10;
  }
  return null;
}

/** Forward substitution: solve L * x = b. Returns x (new array). */
export function solveL(L, b, n) {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = b[i];
    for (let j = 0; j < i; j++) {
      sum -= L[i * n + j] * x[j];
    }
    x[i] = sum / L[i * n + i];
  }
  return x;
}

/** Back substitution: solve L^T * x = b. Returns x (new array). */
export function solveLT(L, b, n) {
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = b[i];
    for (let j = i + 1; j < n; j++) {
      sum -= L[j * n + i] * x[j];
    }
    x[i] = sum / L[i * n + i];
  }
  return x;
}

/** Solve (L L^T) x = b via forward then back substitution. */
export function cholSolve(L, b, n) {
  const z = solveL(L, b, n);
  return solveLT(L, z, n);
}

// ─── Kernel ──────────────────────────────────────────────────────────────────

/**
 * Matérn 5/2 kernel between two D-dimensional points.
 * Supports isotropic (lengthScales.length === 1) and ARD modes.
 * k(r) = σ_f² * (1 + √5·r + 5/3·r²) * exp(-√5·r)
 */
export function matern52(x1, x2, d, lengthScales, sigmaF) {
  const isotropic = lengthScales.length === 1;
  let r2 = 0;
  for (let i = 0; i < d; i++) {
    const ls = isotropic ? lengthScales[0] : lengthScales[i];
    const diff = (x1[i] - x2[i]) / ls;
    r2 += diff * diff;
  }
  const r = Math.sqrt(r2);
  const sqrt5r = Math.sqrt(5) * r;
  return sigmaF * sigmaF * (1 + sqrt5r + (5 / 3) * r2) * Math.exp(-sqrt5r);
}

/**
 * Build N×N kernel matrix K + σ_n²·I.
 * X is flat Float64Array of N*D values (row-major).
 */
export function buildKernelMatrix(X, N, D, lengthScales, sigmaF, sigmaN) {
  const K = new Float64Array(N * N);
  // Temp arrays for row extraction
  const xi = new Float64Array(D);
  const xj = new Float64Array(D);

  for (let i = 0; i < N; i++) {
    for (let k = 0; k < D; k++) xi[k] = X[i * D + k];
    for (let j = i; j < N; j++) {
      for (let k = 0; k < D; k++) xj[k] = X[j * D + k];
      const kij = matern52(xi, xj, D, lengthScales, sigmaF);
      K[i * N + j] = kij;
      K[j * N + i] = kij;
    }
    // Add noise to diagonal
    K[i * N + i] += sigmaN * sigmaN;
  }
  return K;
}

// ─── GP Core ─────────────────────────────────────────────────────────────────

/**
 * Fit a Gaussian Process model.
 * Returns { L, alpha, X, N, D, lengthScales, sigmaF, sigmaN, mu }
 * where L is the Cholesky factor of K and alpha = K⁻¹(y - μ).
 */
export function gpFit(X, y, N, D, lengthScales, sigmaF, sigmaN, mu) {
  const K = buildKernelMatrix(X, N, D, lengthScales, sigmaF, sigmaN);
  const L = choleskyWithRetry(K, N);
  if (!L) return null;

  // alpha = K⁻¹(y - μ)
  const yMinusMu = new Float64Array(N);
  for (let i = 0; i < N; i++) yMinusMu[i] = y[i] - mu;
  const alpha = cholSolve(L, yMinusMu, N);

  return { L, alpha, X: new Float64Array(X), N, D, lengthScales, sigmaF, sigmaN, mu };
}

/**
 * Predict mean and variance at a test point xStar (Float64Array of length D).
 */
export function gpPredict(model, xStar) {
  const { L, alpha, X, N, D, lengthScales, sigmaF } = model;
  const xi = new Float64Array(D);

  // k* vector: kernel between xStar and each training point
  const kStar = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < D; k++) xi[k] = X[i * D + k];
    kStar[i] = matern52(xi, xStar, D, lengthScales, sigmaF);
  }

  // Mean: μ + k*^T α
  let mean = model.mu;
  for (let i = 0; i < N; i++) mean += kStar[i] * alpha[i];

  // Variance: k(x*,x*) - k*^T K⁻¹ k* = k** - v^T v where L v = k*
  const kss = sigmaF * sigmaF;
  const v = solveL(L, kStar, N);
  let vTv = 0;
  for (let i = 0; i < N; i++) vTv += v[i] * v[i];
  let variance = kss - vTv;
  if (variance < 0) variance = 0;

  return { mean, variance };
}

// ─── Acquisition Functions ───────────────────────────────────────────────────

/** Standard normal PDF. */
export function normPDF(x) {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/** Standard normal CDF (Abramowitz & Stegun approximation). */
export function normCDF(x) {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const t = 1 / (1 + p * Math.abs(x));
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x / 2);
  return 0.5 * (1 + sign * y);
}

/**
 * Expected Improvement acquisition function.
 * EI = (μ - f* - ξ)Φ(Z) + σφ(Z) where Z = (μ - f* - ξ)/σ
 */
export function expectedImprovement(mean, variance, fBest, xi) {
  if (variance < 1e-12) return 0;
  const sigma = Math.sqrt(variance);
  const z = (mean - fBest - xi) / sigma;
  return (mean - fBest - xi) * normCDF(z) + sigma * normPDF(z);
}

// ─── Hyperparameter Optimization ─────────────────────────────────────────────

/**
 * Log marginal likelihood of the GP.
 * LML = -½ α^T(y-μ) - Σlog(diag(L)) - N/2·log(2π)
 */
export function logMarginalLikelihood(X, y, N, D, lengthScales, sigmaF, sigmaN, mu) {
  const K = buildKernelMatrix(X, N, D, lengthScales, sigmaF, sigmaN);
  const L = choleskyWithRetry(K, N);
  if (!L) return -1e10; // penalty for failed decomposition

  const yMinusMu = new Float64Array(N);
  for (let i = 0; i < N; i++) yMinusMu[i] = y[i] - mu;
  const alpha = cholSolve(L, yMinusMu, N);

  // -½ α^T(y-μ)
  let dataFit = 0;
  for (let i = 0; i < N; i++) dataFit += alpha[i] * yMinusMu[i];
  dataFit *= -0.5;

  // -Σlog(diag(L))
  let logDet = 0;
  for (let i = 0; i < N; i++) logDet += Math.log(L[i * N + i]);

  // -N/2 log(2π)
  const complexity = -0.5 * N * Math.log(2 * Math.PI);

  return dataFit - logDet + complexity;
}

/**
 * Nelder-Mead simplex optimizer (gradient-free).
 * Minimizes f(x). Works in the provided space directly.
 * @param {Function} f - objective function (array → number)
 * @param {number[]} x0 - initial point
 * @param {number} maxIter - max iterations
 * @param {number} tol - convergence tolerance
 * @returns {number[]} - best point found
 */
export function nelderMead(f, x0, maxIter = 200, tol = 1e-6) {
  const n = x0.length;
  const np1 = n + 1;

  // Initialize simplex
  const simplex = [];
  const fVals = [];
  simplex.push([...x0]);
  fVals.push(f(x0));

  for (let i = 0; i < n; i++) {
    const xi = [...x0];
    xi[i] += (Math.abs(xi[i]) < 1e-10 ? 0.5 : 0.5 * Math.abs(xi[i]));
    simplex.push(xi);
    fVals.push(f(xi));
  }

  const alpha = 1, gamma = 2, rho = 0.5, sigma = 0.5;

  for (let iter = 0; iter < maxIter; iter++) {
    // Sort by function value
    const indices = Array.from({ length: np1 }, (_, i) => i);
    indices.sort((a, b) => fVals[a] - fVals[b]);
    const sorted = indices.map(i => simplex[i]);
    const sortedF = indices.map(i => fVals[i]);
    for (let i = 0; i < np1; i++) {
      simplex[i] = sorted[i];
      fVals[i] = sortedF[i];
    }

    // Check convergence
    const fRange = Math.abs(fVals[np1 - 1] - fVals[0]);
    if (fRange < tol) break;

    // Centroid (excluding worst)
    const centroid = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) centroid[j] += simplex[i][j];
    }
    for (let j = 0; j < n; j++) centroid[j] /= n;

    // Reflection
    const xr = centroid.map((c, j) => c + alpha * (c - simplex[np1 - 1][j]));
    const fr = f(xr);

    if (fr < fVals[0]) {
      // Expansion
      const xe = centroid.map((c, j) => c + gamma * (xr[j] - c));
      const fe = f(xe);
      if (fe < fr) {
        simplex[np1 - 1] = xe;
        fVals[np1 - 1] = fe;
      } else {
        simplex[np1 - 1] = xr;
        fVals[np1 - 1] = fr;
      }
    } else if (fr < fVals[np1 - 2]) {
      simplex[np1 - 1] = xr;
      fVals[np1 - 1] = fr;
    } else {
      // Contraction
      const xc = centroid.map((c, j) => c + rho * (simplex[np1 - 1][j] - c));
      const fc = f(xc);
      if (fc < fVals[np1 - 1]) {
        simplex[np1 - 1] = xc;
        fVals[np1 - 1] = fc;
      } else {
        // Shrink
        for (let i = 1; i < np1; i++) {
          for (let j = 0; j < n; j++) {
            simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
          }
          fVals[i] = f(simplex[i]);
        }
      }
    }
  }

  // Return best
  let bestIdx = 0;
  for (let i = 1; i < np1; i++) {
    if (fVals[i] < fVals[bestIdx]) bestIdx = i;
  }
  return simplex[bestIdx];
}

/**
 * Optimize GP hyperparameters by maximizing log marginal likelihood.
 * Works in log-space for length scales, sigmaF, sigmaN. Soft penalty bounds.
 * @param {Float64Array} X - training inputs (N×D flat)
 * @param {Float64Array} y - training outputs (N)
 * @param {number} N - number of training points
 * @param {number} D - input dimension
 * @param {boolean} useARD - if true, optimize per-dimension length scales
 * @returns {{ lengthScales: Float64Array, sigmaF: number, sigmaN: number, mu: number }}
 */
export function optimizeHyperparams(X, y, N, D, useARD = false) {
  const nLS = useARD ? D : 1;

  // Compute mean of y as initial mu
  let yMean = 0;
  for (let i = 0; i < N; i++) yMean += y[i];
  yMean /= N;

  // Compute variance of y
  let yVar = 0;
  for (let i = 0; i < N; i++) yVar += (y[i] - yMean) * (y[i] - yMean);
  yVar /= N;
  const yStd = Math.sqrt(yVar) || 0.5;

  // Initial guess in log-space: [log(ls)..., log(sigmaF), log(sigmaN), mu]
  const x0 = new Array(nLS + 3);
  for (let i = 0; i < nLS; i++) x0[i] = Math.log(0.5); // length scales
  x0[nLS] = Math.log(yStd); // sigmaF
  x0[nLS + 1] = Math.log(0.1); // sigmaN
  x0[nLS + 2] = yMean; // mu (not log-space)

  // Objective: negative LML (we minimize)
  function objective(params) {
    const ls = new Float64Array(nLS);
    for (let i = 0; i < nLS; i++) ls[i] = Math.exp(params[i]);
    const sf = Math.exp(params[nLS]);
    const sn = Math.exp(params[nLS + 1]);
    const mu = params[nLS + 2];

    // Soft penalty for bounds: log-lengths in [-5, 3]
    let penalty = 0;
    for (let i = 0; i < nLS; i++) {
      if (params[i] < -5) penalty += 100 * (-5 - params[i]) * (-5 - params[i]);
      if (params[i] > 3) penalty += 100 * (params[i] - 3) * (params[i] - 3);
    }
    // sigmaF bounds
    if (params[nLS] < -5) penalty += 100 * (-5 - params[nLS]) * (-5 - params[nLS]);
    if (params[nLS] > 3) penalty += 100 * (params[nLS] - 3) * (params[nLS] - 3);
    // sigmaN bounds
    if (params[nLS + 1] < -1.2) penalty += 100 * (-1.2 - params[nLS + 1]) * (-1.2 - params[nLS + 1]);
    if (params[nLS + 1] > 1) penalty += 100 * (params[nLS + 1] - 1) * (params[nLS + 1] - 1);

    const lml = logMarginalLikelihood(X, y, N, D, ls, sf, sn, mu);
    return -lml + penalty;
  }

  const best = nelderMead(objective, x0, 100, 1e-4);

  const lengthScales = new Float64Array(nLS);
  for (let i = 0; i < nLS; i++) lengthScales[i] = Math.exp(best[i]);
  const sigmaF = Math.exp(best[nLS]);
  const sigmaN = Math.exp(best[nLS + 1]);
  const mu = best[nLS + 2];

  return { lengthScales, sigmaF, sigmaN, mu };
}

// ─── PCA (via kernel trick) ──────────────────────────────────────────────────

/**
 * Fit PCA on normalized data using the kernel trick (efficient when N < D).
 * @param {Float64Array} data - flat N×D matrix (row-major, values in [0,1])
 * @param {number} N - number of samples
 * @param {number} D - number of dimensions
 * @param {number} maxK - max components to keep
 * @returns {{ mean, components (K×D flat), eigenvalues, nK, D }}
 */
export function fitPCA(data, N, D, maxK = 8) {
  const nK = Math.min(maxK, N - 1, D);
  if (nK < 1) return null;

  // Mean
  const mean = new Float64Array(D);
  for (let i = 0; i < N; i++)
    for (let j = 0; j < D; j++)
      mean[j] += data[i * D + j];
  for (let j = 0; j < D; j++) mean[j] /= N;

  // Center
  const C = new Float64Array(N * D);
  for (let i = 0; i < N; i++)
    for (let j = 0; j < D; j++)
      C[i * D + j] = data[i * D + j] - mean[j];

  // Gram matrix N×N
  const G = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = i; j < N; j++) {
      let dot = 0;
      for (let k = 0; k < D; k++) dot += C[i * D + k] * C[j * D + k];
      G[i * N + j] = dot;
      G[j * N + i] = dot;
    }
  }

  // Power iteration with deflation
  const eigenvalues = [];
  const eigenvecs = [];
  const A = new Float64Array(G);

  for (let c = 0; c < nK; c++) {
    let v = new Float64Array(N);
    for (let i = 0; i < N; i++) v[i] = Math.random() - 0.5;
    let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    for (let i = 0; i < N; i++) v[i] /= norm;

    let ev = 0;
    for (let iter = 0; iter < 300; iter++) {
      const Av = new Float64Array(N);
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++)
          Av[i] += A[i * N + j] * v[j];
      norm = Math.sqrt(Av.reduce((s, x) => s + x * x, 0));
      if (norm < 1e-12) break;
      ev = norm;
      for (let i = 0; i < N; i++) v[i] = Av[i] / norm;
    }
    eigenvalues.push(ev);
    eigenvecs.push(v);
    // Deflate
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++)
        A[i * N + j] -= ev * v[i] * v[j];
  }

  // Convert gram eigenvecs → data-space principal components
  const components = new Float64Array(nK * D);
  for (let c = 0; c < nK; c++) {
    const u = eigenvecs[c];
    const scale = Math.sqrt(eigenvalues[c]);
    if (scale < 1e-10) continue;
    for (let j = 0; j < D; j++) {
      let val = 0;
      for (let i = 0; i < N; i++) val += C[i * D + j] * u[i];
      components[c * D + j] = val / scale;
    }
  }

  const totalVar = eigenvalues.reduce((s, v) => s + v, 0);
  console.log(`PCA: ${nK} components, explained variance: ${eigenvalues.map((v, i) => ((v / totalVar) * 100).toFixed(1) + '%').join(', ')}`);

  return { mean, components, eigenvalues, nK, D };
}

/** Project full normalized vector → PCA space (Float64Array of length nK). */
export function toPCA(pca, x) {
  const z = new Float64Array(pca.nK);
  for (let c = 0; c < pca.nK; c++)
    for (let j = 0; j < pca.D; j++)
      z[c] += (x[j] - pca.mean[j]) * pca.components[c * pca.D + j];
  return z;
}

/** Reconstruct from PCA space → full normalized vector, clamped to [0,1]. */
export function fromPCA(pca, z) {
  const x = new Float64Array(pca.D);
  for (let j = 0; j < pca.D; j++) {
    x[j] = pca.mean[j];
    for (let c = 0; c < pca.nK; c++)
      x[j] += z[c] * pca.components[c * pca.D + j];
    if (x[j] < 0) x[j] = 0;
    if (x[j] > 1) x[j] = 1;
  }
  return x;
}

// ─── Sampling ────────────────────────────────────────────────────────────────

/**
 * Latin Hypercube Sampling in [0,1]^D.
 * Returns flat Float64Array of M×D values.
 */
export function latinHypercube(M, D) {
  const result = new Float64Array(M * D);
  for (let j = 0; j < D; j++) {
    // Create permutation of 0..M-1
    const perm = Array.from({ length: M }, (_, i) => i);
    for (let i = M - 1; i > 0; i--) {
      const k = Math.floor(Math.random() * (i + 1));
      [perm[i], perm[k]] = [perm[k], perm[i]];
    }
    for (let i = 0; i < M; i++) {
      result[i * D + j] = (perm[i] + Math.random()) / M;
    }
  }
  return result;
}
