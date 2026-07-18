// Sinusoidal positional encoding — PE(pos, 2i) = sin(pos / 10000^(2i/d_model)),
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)). Deterministic, not learned,
// so — unlike embedding-utils.js's generateEmbeddingVector — this is the
// exact real formula in both Lecture and Interactive Mode. No illustrative
// disclaimer needed here; that contrast is itself worth pointing out to
// students (see scene-copy.js's positional-enc narration).

export function sinusoidalPositionEncoding(pos, dModel) {
  const vec = new Array(dModel);
  for (let i = 0; i < dModel; i += 2) {
    const freq = Math.pow(10000, i / dModel);
    vec[i] = Math.sin(pos / freq);
    if (i + 1 < dModel) vec[i + 1] = Math.cos(pos / freq);
  }
  return vec;
}

export function sinusoidalPETable(seqLen, dModel) {
  return Array.from({ length: seqLen }, (_, pos) => sinusoidalPositionEncoding(pos, dModel));
}

export function addVectors(a, b) {
  return a.map((v, i) => v + b[i]);
}
