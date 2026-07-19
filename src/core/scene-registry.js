// Phase 2: registry only. Each scene is a placeholder until its Phase 3/4 slot.
// `subSteps` are declared now (per architecture §4.3) so the timeline scrubber
// has real nested tick marks to point at, even before animations exist.

export const SCENES = [
  { id: 'input-sentence',   group: 'Setup',     title: 'Input Sentence',            subSteps: ['reveal-sentence'] },
  { id: 'tokenize',         group: 'Setup',     title: 'Tokenization',              subSteps: ['split-words', 'assign-ids'] },
  { id: 'embedding',        group: 'Setup',     title: 'Word Embedding',            subSteps: ['lookup-table', 'vector-materialize', 'before-after', 'quick-check'] },
  { id: 'positional-enc',   group: 'Setup',     title: 'Positional Encoding',       subSteps: ['problem', 'table', 'combine', 'output', 'before-after', 'quick-check'] },
  { id: 'residual-stream',  group: 'Setup',     title: 'Residual Stream',           subSteps: ['introduce-stream'] },

  { id: 'proj-q',           group: 'Attention', title: 'Linear Projection to Q',    subSteps: ['weight-matrix', 'matmul', 'before-after', 'quick-check'] },
  { id: 'proj-k',           group: 'Attention', title: 'Linear Projection to K',    subSteps: ['weight-matrix', 'matmul', 'before-after', 'quick-check'] },
  { id: 'proj-v',           group: 'Attention', title: 'Linear Projection to V',    subSteps: ['weight-matrix', 'matmul', 'before-after', 'quick-check'] },
  { id: 'split-heads',      group: 'Attention', title: 'Split into Multiple Heads', subSteps: ['reshape', 'transpose', 'before-after', 'quick-check'] },
  { id: 'qk-matmul',        group: 'Attention', title: 'Q × Kᵀ',                    subSteps: ['attention-grid', 'before-after', 'quick-check'] },
  { id: 'scale-softmax',    group: 'Attention', title: 'Scaling & Softmax',         subSteps: ['divide-sqrt-dk', 'softmax-curve', 'before-after', 'quick-check'] },
  { id: 'weighted-sum',     group: 'Attention', title: 'Weighted Sum',              subSteps: ['weight-values', 'sum', 'before-after', 'quick-check'] },
  { id: 'heads-compare',    group: 'Attention', title: 'Individual Attention Heads',subSteps: ['per-head-heatmaps', 'before-after', 'quick-check'] },
  { id: 'concat',           group: 'Attention', title: 'Concatenation',             subSteps: ['stitch-heads', 'before-after', 'quick-check'] },
  { id: 'output-proj',      group: 'Attention', title: 'Output Projection (Wo)',    subSteps: ['weight-matrix', 'matmul', 'before-after', 'quick-check'] },
  { id: 'residual-1',       group: 'Attention', title: 'Residual Connection',       subSteps: ['main-path', 'skip-path', 'addition', 'before-after', 'quick-check'] },
  { id: 'layer-norm-1',     group: 'Attention', title: 'Layer Normalization',       subSteps: ['statistics', 'normalize', 'scale-shift', 'before-after', 'quick-check'] },

  { id: 'ffn',              group: 'FeedFwd',   title: 'Feed Forward Network',      subSteps: ['linear-1', 'activation', 'linear-2', 'before-after', 'quick-check'] },
  { id: 'residual-2',       group: 'FeedFwd',   title: 'Residual Connection',       subSteps: ['main-path', 'skip-path', 'addition', 'before-after', 'quick-check'] },
  { id: 'layer-norm-2',     group: 'FeedFwd',   title: 'Layer Normalization',       subSteps: ['statistics', 'normalize', 'scale-shift', 'before-after', 'quick-check'] },
  { id: 'encoder-output',   group: 'FeedFwd',   title: 'Encoder Output',            subSteps: ['final-representation', 'context', 'applications', 'before-after', 'quick-check'] },

  { id: 'dec-embedding',      group: 'DecoderSetup', title: 'Decoder Word Embedding',      subSteps: ['lookup-table', 'vector-materialize', 'before-after', 'quick-check'], config: { sentenceType: 'decoder' } },
  { id: 'dec-positional-enc', group: 'DecoderSetup', title: 'Decoder Positional Encoding', subSteps: ['problem', 'table', 'combine', 'output', 'before-after', 'quick-check'], config: { sentenceType: 'decoder' } },

  { id: 'dec-proj-q',         group: 'DecoderAttention', title: 'Decoder Projection to Q',    subSteps: ['weight-matrix', 'matmul', 'before-after', 'quick-check'], config: { stream: 'decoder', proj: 'q' } },
  { id: 'dec-proj-k',         group: 'DecoderAttention', title: 'Decoder Projection to K',    subSteps: ['weight-matrix', 'matmul', 'before-after', 'quick-check'], config: { stream: 'decoder', proj: 'k' } },
  { id: 'dec-proj-v',         group: 'DecoderAttention', title: 'Decoder Projection to V',    subSteps: ['weight-matrix', 'matmul', 'before-after', 'quick-check'], config: { stream: 'decoder', proj: 'v' } },
  { id: 'dec-split-heads',    group: 'DecoderAttention', title: 'Decoder Split Heads',        subSteps: ['reshape', 'transpose', 'before-after', 'quick-check'], config: { stream: 'decoder' } },
  { id: 'dec-qk-matmul',      group: 'DecoderAttention', title: 'Decoder Q × Kᵀ (Raw Scores)',subSteps: ['attention-grid', 'before-after', 'quick-check'], config: { stream: 'decoder', matrixType: 'scores' } },
  { id: 'dec-causal-mask',    group: 'DecoderAttention', title: 'Causal Masking (Lower Δ)',   subSteps: ['mask-matrix', 'before-after', 'quick-check'], config: { stream: 'decoder', matrixType: 'masked-scores', showCausalMask: true } },
  { id: 'dec-scale-softmax',  group: 'DecoderAttention', title: 'Decoder Masked Softmax',     subSteps: ['divide-sqrt-dk', 'softmax-curve', 'before-after', 'quick-check'], config: { stream: 'decoder', matrixType: 'weights', showCausalMask: true } },
  { id: 'dec-weighted-sum',   group: 'DecoderAttention', title: 'Decoder Weighted Sum',      subSteps: ['weight-values', 'sum', 'before-after', 'quick-check'], config: { stream: 'decoder' } },
  { id: 'dec-heads-compare',  group: 'DecoderAttention', title: 'Decoder Attention Heads',    subSteps: ['per-head-heatmaps', 'before-after', 'quick-check'], config: { stream: 'decoder', matrixType: 'per-head', showCausalMask: true } },
  { id: 'dec-concat',         group: 'DecoderAttention', title: 'Decoder Concatenation',      subSteps: ['stitch-heads', 'before-after', 'quick-check'], config: { stream: 'decoder', mode: 'concat' } },
  { id: 'dec-output-proj',    group: 'DecoderAttention', title: 'Decoder Output Projection',  subSteps: ['weight-matrix', 'matmul', 'before-after', 'quick-check'], config: { stream: 'decoder', proj: 'o' } },

  { id: 'dec-residual-1',     group: 'DecoderResidual',  title: 'Decoder Residual Connection ①', subSteps: ['main-path', 'skip-path', 'addition', 'before-after', 'quick-check'], config: { stream: 'decoder', stage: 'residual-1' } },
  { id: 'dec-layer-norm-1',   group: 'DecoderResidual',  title: 'Decoder Layer Normalization ①', subSteps: ['statistics', 'normalize', 'scale-shift', 'before-after', 'quick-check'], config: { stream: 'decoder', stage: 'layer-norm-1' } },
];

export const SCENE_INDEX = Object.fromEntries(SCENES.map((s, i) => [s.id, i]));
