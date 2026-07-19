<script>
  import { subStepIndex, dataMode, currentScene } from '../core/stores/sceneStore.js';
  import { dModel as configDModel } from '../core/stores/configStore.js';
  import { forwardPassData } from '../core/data-loader.js';
  import { computeAttentionPipeline } from '../core/tensor-ops.js';
  import { cellColor } from '../core/heatmap-color.js';

  export let sceneId = 'vocab-projection';

  $: stageConfig = $currentScene?.config?.stage ?? 'vocab-projection';
  $: isSoftmaxStage = stageConfig === 'softmax-output' || sceneId === 'softmax-output';

  const DEFAULT_INTERACTIVE_SENTENCE = ['cat', 'chased', 'dog'];
  const DEFAULT_TARGET_SENTENCE = ['the', 'dog', 'ran'];
  const MAX_INTERACTIVE_TOKENS = 6;

  let interactiveSentence = [...DEFAULT_INTERACTIVE_SENTENCE];
  let interactiveTargetSentence = [...DEFAULT_TARGET_SENTENCE];

  function addWord(word) {
    if (interactiveTargetSentence.length >= MAX_INTERACTIVE_TOKENS) return;
    interactiveTargetSentence = [...interactiveTargetSentence, word];
  }
  function removeWord(index) {
    interactiveTargetSentence = interactiveTargetSentence.filter((_, i) => i !== index);
  }
  function resetSentence() {
    interactiveTargetSentence = [...DEFAULT_TARGET_SENTENCE];
  }

  $: activeSentence = interactiveTargetSentence;
  $: currentDModel = $configDModel;
  $: seqLen = activeSentence.length;

  $: interactiveData = computeAttentionPipeline({
    encoderSentence: ['cat', 'chased', 'dog'],
    decoderSentence: activeSentence,
    dModel: currentDModel,
    numHeads: 4,
    lectureWeights: $forwardPassData?.weights ?? {}
  });

  $: vocabProj = interactiveData?.decoder?.vocabProjection ?? {
    logits: [],
    probabilities: [],
    predictedToken: '<unk>',
    topK: [],
    vocabulary: []
  };

  $: matrixData = isSoftmaxStage ? (vocabProj.probabilities ?? []) : (vocabProj.logits ?? []);
  $: vocabulary = vocabProj.vocabulary ?? [];

  let hoveredRow = null;
  let hoveredCol = null;

  const softmaxFormula = "P[i, v] = \\frac{\\exp(Z_{\\text{vocab}}[i, v])}{\\sum_{w} \\exp(Z_{\\text{vocab}}[i, w])}";
  const logitsFormula = "Z_{\\text{vocab}} = X_{\\text{dec\\_ln3}} W_{\\text{vocab}} + b_{\\text{vocab}} \\in \\mathbb{R}^{L_{\\text{dec}} \\times |V|}";
</script>

<div class="vocab-projection-container">
  <div class="step-header-box">
    <span class="eyebrow">{isSoftmaxStage ? "DECODER STAGE 6: SOFTMAX PROBABILITIES" : "DECODER STAGE 6: VOCABULARY LOGITS"}</span>
    <h3 class="step-title">{isSoftmaxStage ? "Vocabulary Softmax Distribution" : "Vocabulary Logit Projection"}</h3>
  </div>

  <div class="sentence-controls">
    <span class="label">Target Sentence (Decoder):</span>
    <div class="token-chips">
      {#each activeSentence as token, idx}
        <button class="chip" on:click={() => removeWord(idx)} title="Click to remove token">{token} &times;</button>
      {/each}
    </div>
    {#if activeSentence.length < MAX_INTERACTIVE_TOKENS}
      <div class="add-chip-group">
        <button class="add-btn" on:click={() => addWord('fast')}>+ fast</button>
        <button class="add-btn" on:click={() => addWord('barks')}>+ barks</button>
        <button class="add-btn" on:click={() => addWord('slowly')}>+ slowly</button>
      </div>
    {/if}
    <button class="reset-btn" on:click={resetSentence}>Reset</button>
  </div>

  <div class="matrix-card">
    <div class="card-header">
      <h4>{isSoftmaxStage ? 'Softmax Probabilities Matrix P [L_dec × |V|]' : 'Raw Logits Matrix Z_vocab [L_dec × |V|]'}</h4>
      <span class="badge">{isSoftmaxStage ? 'Row Sum = 1.00' : `Vocab Size = ${vocabulary.length}`}</span>
    </div>

    <div class="grid-table-wrapper">
      <table class="grid-table">
        <thead>
          <tr>
            <th class="corner-cell">Target Token</th>
            {#each vocabulary as vocabWord, c}
              <th class="col-header" class:active-col={hoveredCol === c}>{vocabWord}</th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each matrixData as row, r}
            <tr class="grid-row">
              <td class="row-header" class:active-row={hoveredRow === r}>{activeSentence[r] ?? `Token ${r}`}</td>
              {#each row as val, c}
                <td
                  class="val-cell"
                  class:highlight-top={isSoftmaxStage && val > 0.2}
                  style="background: {isSoftmaxStage ? `rgba(99, 102, 241, ${Math.min(1.0, val * 2.5)})` : cellColor(val, 3.0)};"
                  on:mouseenter={() => { hoveredRow = r; hoveredCol = c; }}
                  on:mouseleave={() => { hoveredRow = null; hoveredCol = null; }}
                  title={`Row: ${activeSentence[r]}, Vocab: ${vocabulary[c]}, Value: ${val.toFixed(4)}`}
                >
                  {isSoftmaxStage ? (val * 100).toFixed(1) + '%' : val.toFixed(2)}
                </td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>

  <div class="info-footer">
    <div class="formula-box">
      <code>{isSoftmaxStage ? 'P[i, v] = exp(Z[i, v]) / sum_w exp(Z[i, w])' : 'Z_vocab = X_dec_ln3 * W_vocab + b_vocab ∈ R^(L_dec × |V|)'}</code>
    </div>
    <div class="prediction-summary">
      <span class="pred-label">Predicted Next Token:</span>
      <span class="pred-value">"{vocabProj.predictedToken}"</span>
    </div>
  </div>
</div>

<style>
  .vocab-projection-container {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    padding: 1.5rem;
    background: #0f172a;
    color: #f8fafc;
    border-radius: 12px;
    border: 1px solid #334155;
  }
  .sentence-controls {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    background: #1e293b;
    padding: 0.75rem 1rem;
    border-radius: 8px;
  }
  .label { font-size: 0.85rem; color: #94a3b8; font-weight: 600; }
  .token-chips { display: flex; gap: 0.5rem; }
  .chip {
    background: #334155;
    color: #38bdf8;
    border: 1px solid #0284c7;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-size: 0.85rem;
    cursor: pointer;
  }
  .chip:hover { background: #0284c7; color: #fff; }
  .add-chip-group { display: flex; gap: 0.4rem; }
  .add-btn {
    background: #1e293b;
    color: #a7f3d0;
    border: 1px dashed #059669;
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    font-size: 0.8rem;
    cursor: pointer;
  }
  .add-btn:hover { background: #059669; color: #fff; }
  .reset-btn {
    margin-left: auto;
    background: #475569;
    color: #fff;
    border: none;
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    font-size: 0.8rem;
    cursor: pointer;
  }
  .matrix-card {
    background: #1e293b;
    border-radius: 10px;
    border: 1px solid #334155;
    padding: 1rem;
    overflow-x: auto;
  }
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }
  .card-header h4 { margin: 0; font-size: 1rem; color: #cbd5e1; }
  .badge { background: #3b82f6; color: #fff; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; }
  .grid-table-wrapper { overflow-x: auto; }
  .grid-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; text-align: center; }
  .grid-table th, .grid-table td { padding: 0.5rem 0.4rem; border: 1px solid #334155; }
  .corner-cell { background: #0f172a; color: #94a3b8; font-weight: 600; text-align: left; padding-left: 0.75rem; }
  .col-header { background: #1e293b; color: #e2e8f0; font-weight: 600; }
  .col-header.active-col { background: #3b82f6; color: #fff; }
  .row-header { background: #1e293b; color: #38bdf8; font-weight: 600; text-align: left; padding-left: 0.75rem; }
  .row-header.active-row { background: #0284c7; color: #fff; }
  .val-cell { font-family: monospace; transition: background 0.15s ease; }
  .val-cell.highlight-top { font-weight: bold; color: #fbbf24; border: 1px solid #f59e0b; }
  .info-footer { display: flex; justify-content: space-between; align-items: center; background: #1e293b; padding: 0.75rem 1rem; border-radius: 8px; }
  .prediction-summary { display: flex; align-items: center; gap: 0.5rem; }
  .pred-label { font-size: 0.9rem; color: #94a3b8; }
  .pred-value { font-size: 1.1rem; font-weight: bold; color: #a855f7; }
</style>
