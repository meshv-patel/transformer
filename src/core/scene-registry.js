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

  { id: 'ffn',              group: 'FeedFwd',   title: 'Feed Forward Network',      subSteps: ['expand', 'relu', 'project-back'] },
  { id: 'residual-2',       group: 'FeedFwd',   title: 'Residual Connection',       subSteps: ['main-path', 'skip-path', 'addition', 'before-after', 'quick-check'] },
  { id: 'layer-norm-2',     group: 'FeedFwd',   title: 'Layer Normalization',       subSteps: ['statistics', 'normalize', 'scale-shift', 'before-after', 'quick-check'] },
  { id: 'encoder-output',   group: 'FeedFwd',   title: 'Encoder Output',            subSteps: ['final-representation'] },
];

export const SCENE_INDEX = Object.fromEntries(SCENES.map((s, i) => [s.id, i]));
