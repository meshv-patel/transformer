import EmbeddingViz from './EmbeddingViz.svelte';
import PositionalEncodingViz from './PositionalEncodingViz.svelte';
import LinearProjectionViz from '../../components/LinearProjectionViz.svelte';
import MatrixHeatmap from '../../components/MatrixHeatmap.svelte';

export const SCENE_VIZ = {
  embedding: EmbeddingViz,
  'positional-enc': PositionalEncodingViz,
  'proj-q': LinearProjectionViz,
  'proj-k': LinearProjectionViz,
  'proj-v': LinearProjectionViz,
  'qk-matmul': MatrixHeatmap,
  'scale-softmax': MatrixHeatmap,
  // Phase 4 adds the rest here, one at a time, each following
  // EmbeddingViz.svelte's pattern.
};

export const IMPLEMENTED_SCENE_IDS = new Set(Object.keys(SCENE_VIZ));
