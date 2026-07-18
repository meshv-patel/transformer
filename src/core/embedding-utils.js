// Interactive Mode needs an embedding for whatever word the student picks,
// at whatever d_model the config slider is set to. There's no trained model
// backing this (a toy model this small can't learn meaningful semantics from
// a hand-written corpus — see architecture Addendum §1). Instead we use a
// seeded hash -> PRNG so the *mechanism* (lookup produces a fixed-size
// vector; changing d_model changes its length) is real and reproducible,
// while the *values* are disclosed in-UI as illustrative, not learned.

function hashString(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed) {
  let a = seed;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Deterministic, reproducible per-word vector of length dModel, roughly in [-0.6, 0.6]. */
export function generateEmbeddingVector(word, dModel) {
  const rand = mulberry32(hashString(word.toLowerCase()));
  return Array.from({ length: dModel }, () => (rand() * 2 - 1) * 0.6);
}

/** Stable-ish token id for display purposes only (not a real vocabulary index). */
export function pseudoTokenId(word) {
  return hashString(word.toLowerCase()) % 5000;
}
