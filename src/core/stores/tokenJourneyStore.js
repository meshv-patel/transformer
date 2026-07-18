import { writable, derived } from 'svelte/store';
import { forwardPassData } from '../data-loader.js';

// Which token (by sequence position) is currently being tracked across all
// stages. null = panel closed / nothing selected.
export const selectedTokenPos = writable(null);

// Derived per-stage trace for the selected token, pulled from the full
// precomputed (or live) pipeline rather than any single scene's data slice —
// this is why forward-pass data is loaded whole, up front (architecture §4.1).
export const tokenTrace = derived(
  [selectedTokenPos, forwardPassData],
  ([$pos, $data]) => {
    if ($pos === null || !$data) return null;
    return $data.stages.map((stage) => ({
      stageId: stage.id,
      stageLabel: stage.label,
      vector: stage.tokenVectors?.[$pos] ?? null,
      attendedTo: stage.attention?.[$pos] ?? null, // top attended positions, attention stages only
    }));
  }
);
