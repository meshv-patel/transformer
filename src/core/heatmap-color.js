// Shared cell-color logic for value-magnitude heatmaps. Extracted from
// VectorHeatmap.svelte (1D vector strips) so MatrixHeatmap.svelte (2D
// seq_len x seq_len grids, Self-Attention scenes) can reuse the identical
// color scale rather than re-implementing it. Behavior-preserving: output
// is byte-identical to the function this replaced.
//
// Convention (unchanged, do not repurpose): blue = negative value,
// orange = positive value. This is a value-sign scale only — it has
// nothing to do with Query/Key/Value role identity, which uses
// --accent/--accent-2/--accent-3 separately, on chrome (borders, labels,
// panel headers) only, never on cell fill. Reusing a role's accent color
// as a cell fill would silently collide with this scale for any
// negative-valued cell in that role's vectors.
export function cellColor(v, maxAbs = 0.6) {
  const intensity = Math.min(1, Math.abs(v) / maxAbs);
  return v < 0
    ? `rgba(122, 162, 255, ${0.15 + intensity * 0.75})`
    : `rgba(255, 184, 107, ${0.15 + intensity * 0.75})`;
}
