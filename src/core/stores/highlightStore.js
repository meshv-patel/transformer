import { writable } from 'svelte/store';

// A single "current term id" (e.g. 'matmul-q', 'head-2-softmax') broadcast by
// whichever panel (viz / equation / code) the user is interacting with, and
// consumed by all three at once. null = nothing highlighted.
export const highlightedTermId = writable(null);

export function setHighlight(termId) {
  highlightedTermId.set(termId);
}

export function clearHighlight() {
  highlightedTermId.set(null);
}
