import EmbeddingViz from './EmbeddingViz.svelte';
import PositionalEncodingViz from './PositionalEncodingViz.svelte';
import LinearProjectionViz from '../../components/LinearProjectionViz.svelte';
import MatrixHeatmap from '../../components/MatrixHeatmap.svelte';
import WeightedSumViz from '../../components/WeightedSumViz.svelte';
import HeadsSplitViz from '../../components/HeadsSplitViz.svelte';
import ResidualConnectionViz from '../../components/ResidualConnectionViz.svelte';
import LayerNormViz from '../../components/LayerNormViz.svelte';
import FeedForwardViz from '../../components/FeedForwardViz.svelte';
import EncoderOutputViz from '../../components/EncoderOutputViz.svelte';
import VocabProjectionViz from '../../components/VocabProjectionViz.svelte';
import PredictionViz from '../../components/PredictionViz.svelte';
import InputSentenceViz from '../../components/InputSentenceViz.svelte';
import TokenizeViz from '../../components/TokenizeViz.svelte';
import ResidualStreamViz from '../../components/ResidualStreamViz.svelte';

export const SCENE_VIZ = {
  'input-sentence': InputSentenceViz,
  'tokenize': TokenizeViz,
  'residual-stream': ResidualStreamViz,
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
  'ffn': FeedForwardViz,
  'encoder-output': EncoderOutputViz,
  'dec-embedding': EmbeddingViz,
  'dec-positional-enc': PositionalEncodingViz,
  'dec-proj-q': LinearProjectionViz,
  'dec-proj-k': LinearProjectionViz,
  'dec-proj-v': LinearProjectionViz,
  'dec-split-heads': HeadsSplitViz,
  'dec-qk-matmul': MatrixHeatmap,
  'dec-causal-mask': MatrixHeatmap,
  'dec-scale-softmax': MatrixHeatmap,
  'dec-weighted-sum': WeightedSumViz,
  'dec-heads-compare': MatrixHeatmap,
  'dec-concat': HeadsSplitViz,
  'dec-output-proj': LinearProjectionViz,
  'dec-residual-1': ResidualConnectionViz,
  'dec-layer-norm-1': LayerNormViz,
  'dec-cross-proj-q': LinearProjectionViz,
  'dec-cross-proj-k': LinearProjectionViz,
  'dec-cross-proj-v': LinearProjectionViz,
  'dec-cross-split-heads': HeadsSplitViz,
  'dec-cross-qk-matmul': MatrixHeatmap,
  'dec-cross-scale-softmax': MatrixHeatmap,
  'dec-cross-weighted-sum': WeightedSumViz,
  'dec-cross-heads-compare': MatrixHeatmap,
  'dec-cross-concat': HeadsSplitViz,
  'dec-cross-output-proj': LinearProjectionViz,
  'dec-cross-residual-2': ResidualConnectionViz,
  'dec-cross-layernorm-2': LayerNormViz,
  'dec-ffn': FeedForwardViz,
  'dec-residual-3': ResidualConnectionViz,
  'dec-layernorm-3': LayerNormViz,
  'decoder-stack': PredictionViz,
  'vocab-projection': VocabProjectionViz,
  'softmax-output': VocabProjectionViz,
  'next-token': PredictionViz,
  'autoregressive-generation': PredictionViz,
};

export const IMPLEMENTED_SCENE_IDS = new Set(Object.keys(SCENE_VIZ));
