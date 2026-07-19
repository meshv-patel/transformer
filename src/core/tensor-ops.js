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
  const fallbackKey = weightKey.replace(/_dec|_cross/, '');
  const baseW = lectureWeights?.[weightKey] ?? lectureWeights?.[fallbackKey];
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
  const fallbackKey = biasKey.replace(/_dec|_cross/, '');
  const baseB = lectureWeights?.[biasKey] ?? lectureWeights?.[fallbackKey];
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
export function computeAttentionPipeline({
  encoderSentence = [],
  decoderSentence = [],
  dModel = 16,
  numHeads = 4,
  lectureWeights = {}
} = {}) {
  const sentence = encoderSentence;
  const dModelVal = dModel;
  const numHeadsVal = numHeads;
  if (dModelVal % numHeadsVal !== 0) {
    throw new Error(`dModel (${dModelVal}) must be divisible by numHeads (${numHeadsVal})`);
  }
  const safeDKVal = Math.max(1, Math.floor(dModelVal / (numHeadsVal || 1)));
  const eps = 1e-5;

  // Decoder Stage 1: Embeddings and Positional Encoding
  const dec_embeds = decoderSentence.map((w) => generateEmbeddingVector(w, dModelVal));
  const dec_pes = sinusoidalPETable(decoderSentence.length, dModelVal);
  const dec_xPe = dec_embeds.map((e, i) => addVectors(e, dec_pes[i]));

  // Decoder Stage 2: Masked Self-Attention
  const Wq_dec = getInteractiveWeights('Wq_dec', dModelVal, lectureWeights);
  const bq_dec = getInteractiveBias('bq_dec', dModelVal, lectureWeights);
  const Wk_dec = getInteractiveWeights('Wk_dec', dModelVal, lectureWeights);
  const bk_dec = getInteractiveBias('bk_dec', dModelVal, lectureWeights);
  const Wv_dec = getInteractiveWeights('Wv_dec', dModelVal, lectureWeights);
  const bv_dec = getInteractiveBias('bv_dec', dModelVal, lectureWeights);
  const Wo_dec = getInteractiveWeights('Wo_dec', dModelVal, lectureWeights);
  const bo_dec = getInteractiveBias('bo_dec', dModelVal, lectureWeights);

  const dec_Q = matmul(dec_xPe, Wq_dec).map((row) => row.map((v, c) => v + bq_dec[c]));
  const dec_K = matmul(dec_xPe, Wk_dec).map((row) => row.map((v, c) => v + bk_dec[c]));
  const dec_V = matmul(dec_xPe, Wv_dec).map((row) => row.map((v, c) => v + bv_dec[c]));

  const dec_Qh = Array.from({ length: decoderSentence.length }, (_, i) =>
    Array.from({ length: numHeadsVal }, (_, h) => dec_Q[i].slice(h * safeDKVal, (h + 1) * safeDKVal))
  );
  const dec_Kh = Array.from({ length: decoderSentence.length }, (_, i) =>
    Array.from({ length: numHeadsVal }, (_, h) => dec_K[i].slice(h * safeDKVal, (h + 1) * safeDKVal))
  );
  const dec_Vh = Array.from({ length: decoderSentence.length }, (_, i) =>
    Array.from({ length: numHeadsVal }, (_, h) => dec_V[i].slice(h * safeDKVal, (h + 1) * safeDKVal))
  );

  const dec_Qh_trans = splitHeads(dec_Q, numHeadsVal);
  const dec_Kh_trans = splitHeads(dec_K, numHeadsVal);
  const dec_Vh_trans = splitHeads(dec_V, numHeadsVal);

  const dec_rawScores = [];
  const dec_maskedScores = [];
  const dec_weights = [];
  const dec_outputHeads = [];
  const dec_concatenatedOutput = Array.from({ length: decoderSentence.length }, () => new Array(dModelVal).fill(0));

  for (let h = 0; h < numHeadsVal; h++) {
    const qHead = dec_Qh_trans[h];
    const kHead = dec_Kh_trans[h];
    const vHead = dec_Vh_trans[h];

    const rawHeadScores = scale(matmul(qHead, transpose(kHead)), 1 / Math.sqrt(safeDKVal));
    dec_rawScores.push(rawHeadScores);

    // Apply Causal Masking: for col j > row i, score is -Infinity
    const maskedHeadScores = rawHeadScores.map((row, i) =>
      row.map((val, j) => (j <= i ? val : -Infinity))
    );
    dec_maskedScores.push(maskedHeadScores);

    // Softmax
    const headWeights = softmaxRows(maskedHeadScores);
    dec_weights.push(headWeights);

    // Output heads
    const headOutput = matmul(headWeights, vHead);
    dec_outputHeads.push(headOutput);

    // Concatenate
    for (let i = 0; i < decoderSentence.length; i++) {
      for (let d = 0; d < safeDKVal; d++) {
        const destIdx = h * safeDKVal + d;
        if (destIdx < dModelVal && headOutput[i] && d < headOutput[i].length) {
          dec_concatenatedOutput[i][destIdx] = headOutput[i][d];
        }
      }
    }
  }

  const dec_outputProj = matmul(dec_concatenatedOutput, Wo_dec).map((row) => row.map((v, c) => v + bo_dec[c]));

  // Decoder Stage 3: Residual Connection 1 & LayerNorm 1
  const dec_residual1 = dec_xPe.map((xRow, i) =>
    xRow.map((v, c) => v + (dec_outputProj[i]?.[c] ?? 0))
  );

  const ln1_gamma_dec = getInteractiveBias('ln1_gamma_dec', dModelVal, lectureWeights).map((v, i) => (v === 0 ? 1 : v));
  const ln1_beta_dec = getInteractiveBias('ln1_beta_dec', dModelVal, lectureWeights);

  const dec_ln1_means = [];
  const dec_ln1_vars = [];
  const dec_ln1_norms = [];
  const dec_ln1_outputs = [];

  for (let i = 0; i < decoderSentence.length; i++) {
    const row = dec_residual1[i];
    if (!row || !row.length) {
      dec_ln1_means.push(0);
      dec_ln1_vars.push(0);
      dec_ln1_norms.push([]);
      dec_ln1_outputs.push([]);
      continue;
    }
    const mean = row.reduce((a, b) => a + b, 0) / row.length;
    const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
    const denom = Math.sqrt(variance + eps);
    const norm = row.map((v) => (v - mean) / denom);
    const outRow = norm.map((v, c) => v * (ln1_gamma_dec[c] ?? 1) + (ln1_beta_dec[c] ?? 0));

    dec_ln1_means.push(mean);
    dec_ln1_vars.push(variance);
    dec_ln1_norms.push(norm);
    dec_ln1_outputs.push(outRow);
  }

  const decoderState = {
    embeds: dec_embeds,
    pes: dec_pes,
    xPe: dec_xPe,
    Q: dec_Q,
    K: dec_K,
    V: dec_V,
    Qh: dec_Qh,
    Kh: dec_Kh,
    Vh: dec_Vh,
    Qh_trans: dec_Qh_trans,
    Kh_trans: dec_Kh_trans,
    Vh_trans: dec_Vh_trans,
    rawScores: dec_rawScores,
    maskedScores: dec_maskedScores,
    weights: dec_weights,
    outputHeads: dec_outputHeads,
    concatenatedOutput: dec_concatenatedOutput,
    outputProj: dec_outputProj,
    residual1: dec_residual1,
    ln1_means: dec_ln1_means,
    ln1_vars: dec_ln1_vars,
    ln1_norms: dec_ln1_norms,
    ln1_outputs: dec_ln1_outputs
  };

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
      outputProj: [],
      residual1: [],
      ln1_means: [],
      ln1_vars: [],
      ln1_norms: [],
      ln1_outputs: [],
      ffn_linear1: [],
      ffn_activation: [],
      ffn_outputs: [],
      residual2: [],
      ln2_means: [],
      ln2_vars: [],
      ln2_norms: [],
      ln2_outputs: [],
      decoder: decoderState
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
    Array.from({ length: numHeadsVal }, (_, h) => Q[i].slice(h * safeDKVal, (h + 1) * safeDKVal))
  );
  const Kh = Array.from({ length: sentence.length }, (_, i) =>
    Array.from({ length: numHeadsVal }, (_, h) => K[i].slice(h * safeDKVal, (h + 1) * safeDKVal))
  );
  const Vh = Array.from({ length: sentence.length }, (_, i) =>
    Array.from({ length: numHeadsVal }, (_, h) => V[i].slice(h * safeDKVal, (h + 1) * safeDKVal))
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
    const headScores = scale(matmul(qHead, transpose(kHead)), 1 / Math.sqrt(safeDKVal));
    outScores.push(headScores);

    // Weights
    const headWeights = softmaxRows(headScores);
    outWeights.push(headWeights);

    // Outputs
    const headOutput = matmul(headWeights, vHead);
    outOutputs.push(headOutput);

    // Concat
    for (let i = 0; i < sentence.length; i++) {
      for (let d = 0; d < safeDKVal; d++) {
        const destIdx = h * safeDKVal + d;
        if (destIdx < dModelVal && headOutput[i] && d < headOutput[i].length) {
          outConcatenated[i][destIdx] = headOutput[i][d];
        }
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

  // 10. Feed Forward Network (FFN)
  const dFFVal = dModelVal * 2;
  const W_ff1 = getInteractiveWeights2D('W_ff1', dModelVal, dFFVal, lectureWeights);
  const b_ff1 = getInteractiveBias1D('b_ff1', dFFVal, lectureWeights);
  const W_ff2 = getInteractiveWeights2D('W_ff2', dFFVal, dModelVal, lectureWeights);
  const b_ff2 = getInteractiveBias1D('b_ff2', dModelVal, lectureWeights);

  const ffn_linear1 = matmul(ln1_outputs, W_ff1).map((row) => row.map((v, c) => v + b_ff1[c]));
  const ffn_activation = relu(ffn_linear1);
  const ffn_outputs = matmul(ffn_activation, W_ff2).map((row) => row.map((v, c) => v + b_ff2[c]));

  // 11. Residual 2 Skip Connection
  const residual2 = addMatrices(ln1_outputs, ffn_outputs);

  // 12. Layer Normalization 2
  const ln2_means = [];
  const ln2_vars = [];
  const ln2_norms = [];
  const ln2_outputs = [];

  const ln2_gamma = getInteractiveBias('ln2_gamma', dModelVal, lectureWeights).map(() => 1.0); // defaults to 1.0
  const ln2_beta = getInteractiveBias('ln2_beta', dModelVal, lectureWeights); // defaults to 0.0

  for (let i = 0; i < sentence.length; i++) {
    const row = residual2[i];
    if (!row || !row.length) {
      ln2_means.push(0);
      ln2_vars.push(0);
      ln2_norms.push([]);
      ln2_outputs.push([]);
      continue;
    }
    const mean = row.reduce((a, b) => a + b, 0) / row.length;
    const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
    const denom = Math.sqrt(variance + eps);
    const norm = row.map((v) => (v - mean) / denom);
    const outRow = norm.map((v, c) => v * (ln2_gamma[c] ?? 1) + (ln2_beta[c] ?? 0));

    ln2_means.push(mean);
    ln2_vars.push(variance);
    ln2_norms.push(norm);
    ln2_outputs.push(outRow);
  }

  // --- Decoder Stage 4: Cross Attention ---
  const dec_cross_Qinput = dec_ln1_outputs;
  const dec_cross_KVinput = ln2_outputs;

  const Wq_cross = getInteractiveWeights('Wq_cross', dModelVal, lectureWeights);
  const bq_cross = getInteractiveBias('bq_cross', dModelVal, lectureWeights);
  const Wk_cross = getInteractiveWeights('Wk_cross', dModelVal, lectureWeights);
  const bk_cross = getInteractiveBias('bk_cross', dModelVal, lectureWeights);
  const Wv_cross = getInteractiveWeights('Wv_cross', dModelVal, lectureWeights);
  const bv_cross = getInteractiveBias('bv_cross', dModelVal, lectureWeights);
  const Wo_cross = getInteractiveWeights('Wo_cross', dModelVal, lectureWeights);
  const bo_cross = getInteractiveBias('bo_cross', dModelVal, lectureWeights);

  const dec_cross_Q = matmul(dec_cross_Qinput, Wq_cross).map((row) => row.map((v, c) => v + bq_cross[c]));
  const dec_cross_K = matmul(dec_cross_KVinput, Wk_cross).map((row) => row.map((v, c) => v + bk_cross[c]));
  const dec_cross_V = matmul(dec_cross_KVinput, Wv_cross).map((row) => row.map((v, c) => v + bv_cross[c]));

  const dec_cross_Qh = Array.from({ length: decoderSentence.length }, (_, i) =>
    Array.from({ length: numHeadsVal }, (_, h) => dec_cross_Q[i].slice(h * safeDKVal, (h + 1) * safeDKVal))
  );
  const dec_cross_Kh = Array.from({ length: sentence.length }, (_, i) =>
    Array.from({ length: numHeadsVal }, (_, h) => dec_cross_K[i].slice(h * safeDKVal, (h + 1) * safeDKVal))
  );
  const dec_cross_Vh = Array.from({ length: sentence.length }, (_, i) =>
    Array.from({ length: numHeadsVal }, (_, h) => dec_cross_V[i].slice(h * safeDKVal, (h + 1) * safeDKVal))
  );

  const dec_cross_Qh_trans = splitHeads(dec_cross_Q, numHeadsVal);
  const dec_cross_Kh_trans = splitHeads(dec_cross_K, numHeadsVal);
  const dec_cross_Vh_trans = splitHeads(dec_cross_V, numHeadsVal);

  const dec_cross_scores = [];
  const dec_cross_weights = [];
  const dec_cross_outputHeads = [];
  const dec_cross_concatenatedOutput = Array.from({ length: decoderSentence.length }, () => new Array(dModelVal).fill(0));

  for (let h = 0; h < numHeadsVal; h++) {
    const qHead = dec_cross_Qh_trans[h] ?? [];
    const kHead = dec_cross_Kh_trans[h] ?? [];
    const vHead = dec_cross_Vh_trans[h] ?? [];

    const rawHeadScores = scale(matmul(qHead, transpose(kHead)), 1 / Math.sqrt(safeDKVal));
    dec_cross_scores.push(rawHeadScores);

    const headWeights = softmaxRows(rawHeadScores);
    dec_cross_weights.push(headWeights);

    const headOutput = matmul(headWeights, vHead);
    dec_cross_outputHeads.push(headOutput);

    for (let i = 0; i < decoderSentence.length; i++) {
      for (let d = 0; d < safeDKVal; d++) {
        const destIdx = h * safeDKVal + d;
        if (destIdx < dModelVal && headOutput[i] && d < headOutput[i].length) {
          dec_cross_concatenatedOutput[i][destIdx] = headOutput[i][d];
        }
      }
    }
  }

  const dec_cross_outputProj = matmul(dec_cross_concatenatedOutput, Wo_cross).map((row) => row.map((v, c) => v + bo_cross[c]));

  // Decoder Stage 4: Residual 2 & LayerNorm 2
  const dec_cross_residual2 = dec_cross_Qinput.map((qRow, i) =>
    qRow.map((v, c) => v + (dec_cross_outputProj[i]?.[c] ?? 0))
  );

  const ln2_gamma_dec = getInteractiveBias('ln2_gamma_dec', dModelVal, lectureWeights).map((v, i) => (v === 0 ? 1 : v));
  const ln2_beta_dec = getInteractiveBias('ln2_beta_dec', dModelVal, lectureWeights);

  const dec_cross_ln2_means = [];
  const dec_cross_ln2_vars = [];
  const dec_cross_ln2_norms = [];
  const dec_cross_ln2_outputs = [];

  for (let i = 0; i < decoderSentence.length; i++) {
    const row = dec_cross_residual2[i];
    if (!row || !row.length) {
      dec_cross_ln2_means.push(0);
      dec_cross_ln2_vars.push(0);
      dec_cross_ln2_norms.push([]);
      dec_cross_ln2_outputs.push([]);
      continue;
    }
    const mean = row.reduce((a, b) => a + b, 0) / row.length;
    const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
    const denom = Math.sqrt(variance + eps);
    const norm = row.map((v) => (v - mean) / denom);
    const outRow = norm.map((v, c) => v * (ln2_gamma_dec[c] ?? 1) + (ln2_beta_dec[c] ?? 0));

    dec_cross_ln2_means.push(mean);
    dec_cross_ln2_vars.push(variance);
    dec_cross_ln2_norms.push(norm);
    dec_cross_ln2_outputs.push(outRow);
  }

  decoderState.crossAttention = {
    queryInput: dec_cross_Qinput,
    keyValueInput: dec_cross_KVinput,
    Q: dec_cross_Q,
    K: dec_cross_K,
    V: dec_cross_V,
    Qh: dec_cross_Qh,
    Kh: dec_cross_Kh,
    Vh: dec_cross_Vh,
    Qh_trans: dec_cross_Qh_trans,
    Kh_trans: dec_cross_Kh_trans,
    Vh_trans: dec_cross_Vh_trans,
    scores: dec_cross_scores,
    weights: dec_cross_weights,
    outputHeads: dec_cross_outputHeads,
    concatenatedOutput: dec_cross_concatenatedOutput,
    outputProj: dec_cross_outputProj,
    residual2: dec_cross_residual2,
    ln2_means: dec_cross_ln2_means,
    ln2_vars: dec_cross_ln2_vars,
    ln2_norms: dec_cross_ln2_norms,
    ln2_outputs: dec_cross_ln2_outputs
  };

  // --- Decoder Stage 5: Position-wise Feed Forward Network (FFN), Residual 3 & LayerNorm 3 ---
  const dec_ffn_input = dec_cross_ln2_outputs;
  const dec_dFFVal = 2 * dModelVal;

  const W_ffn1_dec = getInteractiveWeights2D('W_ffn1_dec', dModelVal, dec_dFFVal, lectureWeights);
  const b_ffn1_dec = getInteractiveBias1D('b_ffn1_dec', dec_dFFVal, lectureWeights);
  const W_ffn2_dec = getInteractiveWeights2D('W_ffn2_dec', dec_dFFVal, dModelVal, lectureWeights);
  const b_ffn2_dec = getInteractiveBias1D('b_ffn2_dec', dModelVal, lectureWeights);

  const dec_ffn_linear1 = matmul(dec_ffn_input, W_ffn1_dec).map((row) =>
    row.map((v, c) => v + (b_ffn1_dec[c] ?? 0))
  );

  const dec_ffn_activation = dec_ffn_linear1.map((row) =>
    row.map((v) => Math.max(0, v))
  );

  const dec_ffn_outputs = matmul(dec_ffn_activation, W_ffn2_dec).map((row) =>
    row.map((v, c) => v + (b_ffn2_dec[c] ?? 0))
  );

  const dec_ffn_residual3 = dec_ffn_input.map((xRow, i) =>
    xRow.map((v, c) => v + (dec_ffn_outputs[i]?.[c] ?? 0))
  );

  const ln3_gamma_dec = getInteractiveBias('ln3_gamma_dec', dModelVal, lectureWeights).map((v, i) => (v === 0 ? 1 : v));
  const ln3_beta_dec = getInteractiveBias('ln3_beta_dec', dModelVal, lectureWeights);

  const dec_ffn_ln3_means = [];
  const dec_ffn_ln3_vars = [];
  const dec_ffn_ln3_norms = [];
  const dec_ffn_ln3_outputs = [];

  for (let i = 0; i < decoderSentence.length; i++) {
    const row = dec_ffn_residual3[i];
    if (!row || !row.length) {
      dec_ffn_ln3_means.push(0);
      dec_ffn_ln3_vars.push(0);
      dec_ffn_ln3_norms.push([]);
      dec_ffn_ln3_outputs.push([]);
      continue;
    }
    const mean = row.reduce((a, b) => a + b, 0) / row.length;
    const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
    const denom = Math.sqrt(variance + eps);
    const norm = row.map((v) => (v - mean) / denom);
    const outRow = norm.map((v, c) => v * (ln3_gamma_dec[c] ?? 1) + (ln3_beta_dec[c] ?? 0));

    dec_ffn_ln3_means.push(mean);
    dec_ffn_ln3_vars.push(variance);
    dec_ffn_ln3_norms.push(norm);
    dec_ffn_ln3_outputs.push(outRow);
  }

  decoderState.ffn = {
    linear1: dec_ffn_linear1,
    activation: dec_ffn_activation,
    outputs: dec_ffn_outputs,
    residual3: dec_ffn_residual3,
    ln3_means: dec_ffn_ln3_means,
    ln3_vars: dec_ffn_ln3_vars,
    ln3_norms: dec_ffn_ln3_norms,
    ln3_outputs: dec_ffn_ln3_outputs
  };

  // --- Decoder Stage 6: Output Projection & Vocabulary Softmax ---
  const DEFAULT_VOCABULARY = ['cat', 'chased', 'dog', 'ran', 'slowly', 'the', '<eos>', 'mouse', 'barks', 'fast'];
  const vocabSize = DEFAULT_VOCABULARY.length;

  const W_vocab = getInteractiveWeights2D('W_vocab', dModelVal, vocabSize, lectureWeights);
  const b_vocab = getInteractiveBias1D('b_vocab', vocabSize, lectureWeights);

  const dec_finalOutput = dec_ffn_ln3_outputs;
  const dec_logits = matmul(dec_finalOutput, W_vocab).map((row) =>
    row.map((v, c) => v + (b_vocab[c] ?? 0))
  );

  const dec_probabilities = softmaxRows(dec_logits);

  const lastPos = Math.max(0, decoderSentence.length - 1);
  const lastProbs = dec_probabilities[lastPos] ?? new Array(vocabSize).fill(1 / vocabSize);

  let maxIdx = 0;
  for (let v = 1; v < vocabSize; v++) {
    if ((lastProbs[v] ?? 0) > (lastProbs[maxIdx] ?? 0)) {
      maxIdx = v;
    }
  }

  const dec_topK = DEFAULT_VOCABULARY.map((token, index) => ({
    token,
    prob: lastProbs[index] ?? 0,
    index
  }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5);

  decoderState.finalOutput = dec_finalOutput;
  decoderState.vocabProjection = {
    logits: dec_logits,
    probabilities: dec_probabilities,
    predictedIndex: maxIdx,
    predictedToken: DEFAULT_VOCABULARY[maxIdx] ?? '<unk>',
    topK: dec_topK,
    vocabulary: DEFAULT_VOCABULARY
  };

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
    ln1_outputs,
    ffn_linear1,
    ffn_activation,
    ffn_outputs,
    residual2,
    ln2_means,
    ln2_vars,
    ln2_norms,
    ln2_outputs,
    decoder: decoderState
  };
}

export function getInteractiveWeights2D(weightKey, inDim, outDim, lectureWeights) {
  const fallbackKey = weightKey.endsWith('_dec') ? weightKey.replace('_dec', '') : weightKey;
  const baseW = lectureWeights?.[weightKey] ?? lectureWeights?.[fallbackKey];
  if (!baseW || !baseW.length || !baseW[0].length) {
    return Array.from({ length: inDim }, (_, i) =>
      Array.from({ length: outDim }, (_, j) => (i === j ? 1.0 : 0.0))
    );
  }
  const baseIn = baseW.length;
  const baseOut = baseW[0].length;
  if (baseIn === inDim && baseOut === outDim) return baseW;

  const newW = Array.from({ length: inDim }, () => new Array(outDim).fill(0));
  for (let i = 0; i < inDim; i++) {
    for (let j = 0; j < outDim; j++) {
      if (i < baseIn && j < baseOut) {
        newW[i][j] = baseW[i][j];
      } else {
        newW[i][j] = (i === j) ? 1.0 : 0.0;
      }
    }
  }
  return newW;
}

export function getInteractiveBias1D(biasKey, outDim, lectureWeights) {
  const fallbackKey = biasKey.endsWith('_dec') ? biasKey.replace('_dec', '') : biasKey;
  const baseB = lectureWeights?.[biasKey] ?? lectureWeights?.[fallbackKey];
  if (!baseB || !baseB.length) return new Array(outDim).fill(0);
  const baseSize = baseB.length;
  if (baseSize === outDim) return baseB;

  const newB = new Array(outDim).fill(0);
  for (let i = 0; i < outDim; i++) {
    if (i < baseSize) {
      newB[i] = baseB[i];
    } else {
      newB[i] = 0;
    }
  }
  return newB;
}
