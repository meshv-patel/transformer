import EmbeddingViz from './EmbeddingViz.svelte';
import PositionalEncodingViz from './PositionalEncodingViz.svelte';
import LinearProjectionViz from '../../components/LinearProjectionViz.svelte';
import MatrixHeatmap from '../../components/MatrixHeatmap.svelte';
import WeightedSumViz from '../../components/WeightedSumViz.svelte';
import HeadsSplitViz from '../../components/HeadsSplitViz.svelte';
import ResidualConnectionViz from '../../components/ResidualConnectionViz.svelte';
import LayerNormViz from '../../components/LayerNormViz.svelte';

export const SCENE_VIZ = {
  embedding: EmbeddingViz,
  'positional-enc': PositionalEncodingViz,
  'proj-q': LinearProjectionViz,
  'proj-k': LinearProjectionViz,
  'proj-v': LinearProjectionViz,
  'qk-matmul': MatrixHeatmap,
  'scale-softmax': MatrixHeatmap,
  'weighted-sum': WeightedSumViz,
  'split-heads': HeadsSplitViz,
  'heads-compare': MatrixHeatmap,
  'concat': HeadsSplitViz,
  'output-proj': LinearProjectionViz,
  'residual-1': ResidualConnectionViz,
  'residual-2': ResidualConnectionViz,
  'layer-norm-1': LayerNormViz,
  'layer-norm-2': LayerNormViz,
  // Phase 4 adds the rest here, one at a time, each following
  // EmbeddingViz.svelte's pattern.
};

export const IMPLEMENTED_SCENE_IDS = new Set(Object.keys(SCENE_VIZ));
