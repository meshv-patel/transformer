export function dot(a, b) {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

export function norm(a) {
  return Math.sqrt(dot(a, a));
}

export function cosineSimilarity(a, b) {
  const n = norm(a) * norm(b);
  return n === 0 ? 0 : dot(a, b) / n;
}

/** candidates: [{ word, vector }]. Returns the closest candidate to `vec` (excluding exact self-matches by word if `excludeWord` given). */
export function nearestNeighbor(vec, candidates, excludeWord = null) {
  let best = null;
  let bestSim = -Infinity;
  for (const c of candidates) {
    if (excludeWord && c.word === excludeWord) continue;
    const sim = cosineSimilarity(vec, c.vector);
    if (sim > bestSim) {
      bestSim = sim;
      best = c;
    }
  }
  return best ? { word: best.word, similarity: bestSim } : null;
}
