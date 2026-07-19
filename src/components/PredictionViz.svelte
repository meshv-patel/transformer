<script>
  import { subStepIndex, dataMode, currentScene } from '../core/stores/sceneStore.js';
  import { dModel as configDModel, numHeads as configNumHeads } from '../core/stores/configStore.js';
  import { forwardPassData } from '../core/data-loader.js';
  import { computeAttentionPipeline } from '../core/tensor-ops.js';

  export let sceneId = 'next-token';

  $: stageConfig = $currentScene?.config?.stage ?? 'next-token';
  $: isAutoregressive = stageConfig === 'autoregressive' || sceneId === 'autoregressive-generation';
  $: isStackScene = stageConfig === 'stack' || sceneId === 'decoder-stack';

  let numDecoderLayers = 6;
  const AVAILABLE_LAYERS = [1, 2, 4, 6];

  const DEFAULT_TARGET_SENTENCE = ['the', 'dog', 'ran'];
  let activeSentence = [...DEFAULT_TARGET_SENTENCE];
  const MAX_TOKENS = 6;

  $: currentDModel = $configDModel;
  $: configNumHeadsVal = $dataMode === 'lecture' ? ($forwardPassData?.meta?.numHeads ?? 4) : $configNumHeads;
 
  $: interactiveData = computeAttentionPipeline({
    encoderSentence: ['cat', 'chased', 'dog'],
    decoderSentence: activeSentence,
    dModel: currentDModel,
    numHeads: configNumHeadsVal,
    lectureWeights: $forwardPassData?.weights ?? {}
  });

  $: vocabProj = interactiveData?.decoder?.vocabProjection ?? {
    logits: [],
    probabilities: [],
    predictedToken: '<unk>',
    topK: [],
    vocabulary: []
  };

  $: topK = vocabProj.topK ?? [];
  $: predictedToken = vocabProj.predictedToken ?? '<unk>';

  function stepNextToken() {
    if (activeSentence.length >= MAX_TOKENS) return;
    activeSentence = [...activeSentence, predictedToken];
  }

  function resetSequence() {
    activeSentence = [...DEFAULT_TARGET_SENTENCE];
  }
  const predFormula = "y_t = \\text{argmax}_{v \\in V} P(x_t = v \\mid x_1, \\dots, x_{t-1}, X_{\\text{encoder}})";
</script>

<div class="prediction-container">
  <div class="step-header-box">
    <span class="eyebrow">{isStackScene ? "FULL TRANSFORMER: DECODER STACK" : (isAutoregressive ? "AUTOREGRESSIVE GENERATION" : "NEXT-TOKEN PREDICTION")}</span>
    <h3 class="step-title">{isStackScene ? "Stacked Decoder Block Architecture" : (isAutoregressive ? "Step-by-Step Sequence Decoding" : "Top-5 Vocabulary Probability Distribution")}</h3>
  </div>

  <div class="controls-panel">
    <div class="control-group">
      <span class="ctrl-label">Decoder Layers (N):</span>
      <div class="layer-selector">
        {#each AVAILABLE_LAYERS as layerCount}
          <button
            class="layer-btn"
            class:active={numDecoderLayers === layerCount}
            on:click={() => numDecoderLayers = layerCount}
          >
            N = {layerCount}
          </button>
        {/each}
      </div>
    </div>

    <div class="control-group right">
      <button class="action-btn step" on:click={stepNextToken} disabled={activeSentence.length >= MAX_TOKENS}>
        + Append "{predictedToken}"
      </button>
      <button class="action-btn reset" on:click={resetSequence}>Reset</button>
    </div>
  </div>

  {#if isStackScene}
    <!-- Stack Visualization -->
    <div class="stack-architecture-card">
      <div class="enc-block">
        <span class="block-label">Encoder Block (N=6)</span>
        <span class="block-sub">Outputs encoder.ln2_outputs</span>
      </div>

      <div class="arrow-down">↓ (Encoder Keys & Values)</div>

      <div class="decoder-layers-stack">
        {#each Array(numDecoderLayers) as _, l}
          <div class="layer-box" class:top-layer={l === numDecoderLayers - 1}>
            <span class="layer-title">Decoder Layer {l + 1} / {numDecoderLayers}</span>
            <span class="layer-details">Masked Self-Attn → Cross-Attn → FFN</span>
          </div>
        {/each}
      </div>

      <div class="arrow-down">↓ (decoder.finalOutput)</div>

      <div class="head-block">
        <span class="block-label">Linear & Softmax Projection</span>
        <span class="block-sub">Logits → Probabilities P(x_t | x_&lt;t)</span>
      </div>
    </div>
  {:else}
    <!-- Top-K Prediction Chart -->
    <div class="prediction-card">
      <div class="target-sentence-strip">
        <span class="strip-label">Context Sequence x_&lt;t:</span>
        <div class="chip-sequence">
          {#each activeSentence as tok}
            <span class="tok-chip">{tok}</span>
          {/each}
          <span class="next-slot">? ({predictedToken})</span>
        </div>
      </div>

      <div class="chart-header">
        <h4>Top-5 Probabilities for Next Position t = {activeSentence.length + 1}</h4>
        <span class="badge">Argmax: "{predictedToken}"</span>
      </div>

      <div class="topk-bars">
        {#each topK as item, idx}
          <div class="bar-row" class:is-top={idx === 0}>
            <span class="tok-name">{item.token}</span>
            <div class="bar-track">
              <div class="bar-fill" style="width: {Math.min(100, Math.max(2, item.prob * 100))}%;"></div>
            </div>
            <span class="prob-num">{(item.prob * 100).toFixed(1)}%</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <div class="footer-formula">
    <code>y_t = argmax_(v ∈ V) P(x_t = v | x_1 ... x_(t-1), X_encoder)</code>
  </div>
</div>

<style>
  .prediction-container {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    padding: 1.5rem;
    background: #0f172a;
    color: #f8fafc;
    border-radius: 12px;
    border: 1px solid #334155;
  }
  .controls-panel {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #1e293b;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .control-group { display: flex; align-items: center; gap: 0.75rem; }
  .ctrl-label { font-size: 0.85rem; color: #94a3b8; font-weight: 600; }
  .layer-selector { display: flex; gap: 0.4rem; }
  .layer-btn {
    background: #334155;
    color: #cbd5e1;
    border: 1px solid #475569;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-size: 0.8rem;
    cursor: pointer;
  }
  .layer-btn.active { background: #3b82f6; color: #fff; border-color: #2563eb; font-weight: 600; }
  .action-btn {
    border: none;
    padding: 0.4rem 0.85rem;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
  }
  .action-btn.step { background: #10b981; color: #fff; }
  .action-btn.step:hover { background: #059669; }
  .action-btn.step:disabled { opacity: 0.5; cursor: not-allowed; }
  .action-btn.reset { background: #475569; color: #fff; }

  .stack-architecture-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    background: #1e293b;
    padding: 1.5rem;
    border-radius: 10px;
  }
  .enc-block, .head-block {
    background: #334155;
    border: 1px solid #475569;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    text-align: center;
    width: 100%;
    max-width: 400px;
  }
  .block-label { display: block; font-weight: bold; color: #38bdf8; font-size: 0.95rem; }
  .block-sub { font-size: 0.75rem; color: #94a3b8; }
  .arrow-down { color: #a7f3d0; font-size: 0.85rem; font-weight: 600; }
  .decoder-layers-stack { display: flex; flex-direction: column-reverse; gap: 0.5rem; width: 100%; max-width: 400px; }
  .layer-box {
    background: #0f172a;
    border: 1px solid #3b82f6;
    padding: 0.6rem 1rem;
    border-radius: 6px;
    text-align: center;
  }
  .layer-box.top-layer { border-color: #a855f7; background: #1e1b4b; }
  .layer-title { display: block; font-weight: 600; color: #f8fafc; font-size: 0.85rem; }
  .layer-details { font-size: 0.75rem; color: #94a3b8; }

  .prediction-card {
    background: #1e293b;
    padding: 1.25rem;
    border-radius: 10px;
    border: 1px solid #334155;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .target-sentence-strip { display: flex; align-items: center; gap: 0.75rem; }
  .strip-label { font-size: 0.85rem; color: #94a3b8; }
  .chip-sequence { display: flex; gap: 0.4rem; align-items: center; }
  .tok-chip { background: #334155; color: #e2e8f0; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.85rem; }
  .next-slot { background: #4c1d95; color: #c084fc; border: 1px dashed #a855f7; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.85rem; font-weight: bold; }
  .chart-header { display: flex; justify-content: space-between; align-items: center; }
  .chart-header h4 { margin: 0; font-size: 0.95rem; color: #cbd5e1; }
  .badge { background: #10b981; color: #fff; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }
  .topk-bars { display: flex; flex-direction: column; gap: 0.6rem; }
  .bar-row { display: flex; align-items: center; gap: 0.75rem; }
  .bar-row.is-top .tok-name { color: #10b981; font-weight: bold; }
  .bar-row.is-top .bar-fill { background: #10b981; }
  .tok-name { width: 80px; font-family: monospace; font-size: 0.85rem; color: #cbd5e1; }
  .bar-track { flex: 1; height: 16px; background: #0f172a; border-radius: 4px; overflow: hidden; border: 1px solid #334155; }
  .bar-fill { height: 100%; background: #3b82f6; transition: width 0.3s ease; }
  .prob-num { width: 50px; font-family: monospace; font-size: 0.85rem; color: #94a3b8; text-align: right; }
  .footer-formula { background: #1e293b; padding: 0.75rem 1rem; border-radius: 8px; text-align: center; }
</style>
