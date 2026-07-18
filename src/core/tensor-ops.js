// Deliberately dependency-free: at these sizes (d_model <= 64, seq_len <= 8)
// a hand-rolled implementation is simpler and smaller than pulling in a
// tensor library, and keeps Interactive Mode's compute fully transparent
// for teaching purposes (a student could read this file top to bottom).
//
// Phase 2 scope: stubs with correct signatures + core math only.
// Wiring into an actual live scene happens per-scene in Phase 3/4 when
// Interactive Mode is implemented for that scene.

export function matmul(a, b) {
  // a: [m, k], b: [k, n] -> [m, n]
  const m = a.length, k = a[0].length, n = b[0].length;
  const out = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) sum += a[i][p] * b[p][j];
      out[i][j] = sum;
    }
  }
  return out;
}

export function transpose(a) {
  const m = a.length, n = a[0].length;
  const out = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[j][i] = a[i][j];
  return out;
}

export function addMatrices(a, b) {
  return a.map((row, i) => row.map((v, j) => v + b[i][j]));
}

export function scale(a, s) {
  return a.map((row) => row.map((v) => v * s));
}

export function softmaxRow(row) {
  const max = Math.max(...row);
  const exps = row.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function softmaxRows(matrix) {
  return matrix.map(softmaxRow);
}

export function layerNorm(row, eps = 1e-5) {
  const mean = row.reduce((a, b) => a + b, 0) / row.length;
  const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
  const denom = Math.sqrt(variance + eps);
  return row.map((v) => (v - mean) / denom);
}

export function relu(a) {
  return a.map((row) => row.map((v) => Math.max(0, v)));
}

/**
 * Split [seqLen, dModel] into [numHeads, seqLen, dK].
 */
export function splitHeads(matrix, numHeads) {
  const seqLen = matrix.length;
  const dModel = matrix[0].length;
  const dK = dModel / numHeads;
  const heads = [];
  for (let h = 0; h < numHeads; h++) {
    const head = [];
    for (let t = 0; t < seqLen; t++) {
      head.push(matrix[t].slice(h * dK, (h + 1) * dK));
    }
    heads.push(head);
  }
  return heads;
}

export function concatHeads(heads) {
  const numHeads = heads.length;
  const seqLen = heads[0].length;
  const out = [];
  for (let t = 0; t < seqLen; t++) {
    let row = [];
    for (let h = 0; h < numHeads; h++) row = row.concat(heads[h][t]);
    out.push(row);
  }
  return out;
}

/**
 * Single-head scaled dot-product attention.
 * q, k, v: [seqLen, dK] -> { output: [seqLen, dK], scores: [seqLen, seqLen] (post-softmax) }
 */
export function attention(q, k, v, dK) {
  const scores = scale(matmul(q, transpose(k)), 1 / Math.sqrt(dK));
  const weights = softmaxRows(scores);
  const output = matmul(weights, v);
  return { output, weights };
}
