import { generateEmbeddingVector } from './embedding-utils.js';
import { sinusoidalPETable, addVectors } from './positional-encoding.js';

// Deliberately dependency-free: at these sizes (d_model <= 64, seq_len <= 8)
// a hand-rolled implementation is simpler and smaller than pulling in a
// tensor library, and keeps Interactive Mode's compute fully transparent
// for teaching purposes.

export function matmul(a, b) {
  if (!a || !a.length || !a[0] || !b || !b.length || !b[0]) return [];
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
  if (!a || !a.length || !a[0]) return [];
  const m = a.length, n = a[0].length;
  const out = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[j][i] = a[i][j];
  return out;
}

export function addMatrices(a, b) {
  if (!a || !a.length || !b || !b.length) return [];
  return a.map((row, i) => row.map((v, j) => v + b[i][j]));
}

export function scale(a, s) {
  if (!a || !a.length) return [];
  return a.map((row) => row.map((v) => v * s));
}

export function softmaxRow(row) {
  if (!row || !row.length) return [];
  const max = Math.max(...row);
  const exps = row.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function softmaxRows(matrix) {
  if (!matrix || !matrix.length) return [];
  return matrix.map(softmaxRow);
}

export function layerNorm(row, eps = 1e-5) {
  if (!row || !row.length) return [];
  const mean = row.reduce((a, b) => a + b, 0) / row.length;
  const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
  const denom = Math.sqrt(variance + eps);
  return row.map((v) => (v - mean) / denom);
}

export function relu(a) {
  if (!a || !a.length) return [];
  return a.map((row) => row.map((v) => Math.max(0, v)));
}

/**
 * Split [seqLen, dModel] into [numHeads, seqLen, dK].
 */
export function splitHeads(matrix, numHeads) {
  if (!matrix || !matrix.length || !matrix[0]) return [];
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
  if (!heads || !heads.length || !heads[0] || !heads[0].length) return [];
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
  if (!q || !q.length || !k || !k.length || !v || !v.length) {
    return { output: [], weights: [] };
  }
  const scores = scale(matmul(q, transpose(k)), 1 / Math.sqrt(dK));
  const weights = softmaxRows(scores);
  const output = matmul(weights, v);
  return { output, weights };
}

/**
 * Safe weight retrieval supporting dynamic scaling/resizing in Interactive Mode.
 */
export function getInteractiveWeights(weightKey, dModelVal, lectureWeights) {
  const baseW = lectureWeights?.[weightKey];
  if (!baseW || !baseW.length || !baseW[0].length) {
    // Return compatible identity fallback
    return Array.from({ length: dModelVal }, (_, i) =>
      Array.from({ length: dModelVal }, (_, j) => (i === j ? 1 : 0))
    );
  }
  const baseSize = baseW.length;
  if (baseSize === dModelVal) return baseW;
  
  const newW = Array.from({ length: dModelVal }, () => new Array(dModelVal).fill(0));
  for (let i = 0; i < dModelVal; i++) {
    for (let j = 0; j < dModelVal; j++) {
      if (i < baseSize && j < baseSize) {
        newW[i][j] = baseW[i][j];
      } else {
        newW[i][j] = (i === j) ? 1 : 0;
      }
    }
  }
  return newW;
}

/**
 * Safe bias retrieval supporting dynamic scaling/resizing in Interactive Mode.
 */
export function getInteractiveBias(biasKey, dModelVal, lectureWeights) {
  const baseB = lectureWeights?.[biasKey];
  if (!baseB || !baseB.length) {
    return new Array(dModelVal).fill(0);
  }
  const baseSize = baseB.length;
  if (baseSize === dModelVal) return baseB;

  const newB = new Array(dModelVal).fill(0);
  for (let i = 0; i < dModelVal; i++) {
    if (i < baseSize) {
      newB[i] = baseB[i];
    } else {
      newB[i] = 0;
    }
  }
  return newB;
}

/**
 * Shared central function computing the entire multi-head attention execution path in Interactive Mode.
 */
export function computeAttentionPipeline(sentence, dModelVal, numHeadsVal, dKVal, lectureWeights) {
  if (!sentence || !sentence.length) {
    return {
      embeds: [],
      pes: [],
      xPe: [],
      Q: [], K: [], V: [],
      Qh: [], Kh: [], Vh: [],
      Qh_trans: [], Kh_trans: [], Vh_trans: [],
      scores: [],
      weights: [],
      outputHeads: [],
      concatenatedOutput: [],
      outputProj: []
    };
  }

  // 1. Embeddings and Positional Encoding
  const embeds = sentence.map((w) => generateEmbeddingVector(w, dModelVal));
  const pes = sinusoidalPETable(sentence.length, dModelVal);
  const xPe = embeds.map((e, i) => addVectors(e, pes[i]));

  // 2. Weights & Biases (safely resized)
  const Wq = getInteractiveWeights('Wq', dModelVal, lectureWeights);
  const bq = getInteractiveBias('bq', dModelVal, lectureWeights);
  const Wk = getInteractiveWeights('Wk', dModelVal, lectureWeights);
  const bk = getInteractiveBias('bk', dModelVal, lectureWeights);
  const Wv = getInteractiveWeights('Wv', dModelVal, lectureWeights);
  const bv = getInteractiveBias('bv', dModelVal, lectureWeights);
  const Wo = getInteractiveWeights('Wo', dModelVal, lectureWeights);
  const bo = getInteractiveBias('bo', dModelVal, lectureWeights);

  // 3. Projections
  const Q = matmul(xPe, Wq).map((row) => row.map((v, c) => v + bq[c]));
  const K = matmul(xPe, Wk).map((row) => row.map((v, c) => v + bk[c]));
  const V = matmul(xPe, Wv).map((row) => row.map((v, c) => v + bv[c]));

  // 4. Split Heads (before transpose)
  const Qh = Array.from({ length: sentence.length }, (_, i) =>
    Array.from({ length: numHeadsVal }, (_, h) => Q[i].slice(h * dKVal, (h + 1) * dKVal))
  );
  const Kh = Array.from({ length: sentence.length }, (_, i) =>
    Array.from({ length: numHeadsVal }, (_, h) => K[i].slice(h * dKVal, (h + 1) * dKVal))
  );
  const Vh = Array.from({ length: sentence.length }, (_, i) =>
    Array.from({ length: numHeadsVal }, (_, h) => V[i].slice(h * dKVal, (h + 1) * dKVal))
  );

  // 5. Transpose Heads (head-first)
  const Qh_trans = splitHeads(Q, numHeadsVal);
  const Kh_trans = splitHeads(K, numHeadsVal);
  const Vh_trans = splitHeads(V, numHeadsVal);

  // 6. Attention operations per head
  const outScores = [];
  const outWeights = [];
  const outOutputs = [];
  const outConcatenated = Array.from({ length: sentence.length }, () => new Array(dModelVal).fill(0));

  for (let h = 0; h < numHeadsVal; h++) {
    const qHead = Qh_trans[h];
    const kHead = Kh_trans[h];
    const vHead = Vh_trans[h];

    // Scores
    const headScores = scale(matmul(qHead, transpose(kHead)), 1 / Math.sqrt(dKVal));
    outScores.push(headScores);

    // Weights
    const headWeights = softmaxRows(headScores);
    outWeights.push(headWeights);

    // Outputs
    const headOutput = matmul(headWeights, vHead);
    outOutputs.push(headOutput);

    // Concat
    for (let i = 0; i < sentence.length; i++) {
      for (let d = 0; d < dKVal; d++) {
        outConcatenated[i][h * dKVal + d] = headOutput[i][d];
      }
    }
  }

  // 7. Output Projection
  const outputProj = matmul(outConcatenated, Wo).map((row) => row.map((v, c) => v + bo[c]));

  // 8. Residual 1 Skip Connection
  const residual1 = addMatrices(xPe, outputProj);

  // 9. Layer Normalization 1
  const ln1_means = [];
  const ln1_vars = [];
  const ln1_norms = [];
  const ln1_outputs = [];
  const eps = 1e-5;

  const ln1_gamma = getInteractiveBias('ln1_gamma', dModelVal, lectureWeights).map(() => 1.0); // defaults to 1.0
  const ln1_beta = getInteractiveBias('ln1_beta', dModelVal, lectureWeights); // defaults to 0.0

  for (let i = 0; i < sentence.length; i++) {
    const row = residual1[i];
    if (!row || !row.length) {
      ln1_means.push(0);
      ln1_vars.push(0);
      ln1_norms.push([]);
      ln1_outputs.push([]);
      continue;
    }
    const mean = row.reduce((a, b) => a + b, 0) / row.length;
    const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
    const denom = Math.sqrt(variance + eps);
    const norm = row.map((v) => (v - mean) / denom);
    const outRow = norm.map((v, c) => v * (ln1_gamma[c] ?? 1) + (ln1_beta[c] ?? 0));

    ln1_means.push(mean);
    ln1_vars.push(variance);
    ln1_norms.push(norm);
    ln1_outputs.push(outRow);
  }

  return {
    embeds,
    pes,
    xPe,
    Q, K, V,
    Qh, Kh, Vh,
    Qh_trans, Kh_trans, Vh_trans,
    scores: outScores,
    weights: outWeights,
    outputHeads: outOutputs,
    concatenatedOutput: outConcatenated,
    outputProj,
    residual1,
    ln1_means,
    ln1_vars,
    ln1_norms,
    ln1_outputs
  };
}
