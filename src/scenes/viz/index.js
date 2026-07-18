import EmbeddingViz from './EmbeddingViz.svelte';
import PositionalEncodingViz from './PositionalEncodingViz.svelte';

export const SCENE_VIZ = {
  embedding: EmbeddingViz,
  'positional-enc': PositionalEncodingViz,
  // Phase 4 adds the rest here, one at a time, each following
  // EmbeddingViz.svelte's pattern.
};

export const IMPLEMENTED_SCENE_IDS = new Set(Object.keys(SCENE_VIZ));
