<script>
  import { fade } from 'svelte/transition';
  import { subStepIndex, replayTick, dataMode, currentScene } from '../core/stores/sceneStore.js';
  import { dModel as configDModel } from '../core/stores/configStore.js';
  import { forwardPassData } from '../core/data-loader.js';
  import { highlightedTermId, setHighlight, clearHighlight } from '../core/stores/highlightStore.js';
  import { computeAttentionPipeline } from '../core/tensor-ops.js';
  import { getSceneCopy } from '../data/scene-copy.js';
  import { focusShot } from '../core/camera/cameraStore.js';
  import { motionMs } from '../core/motion.js';
  import { VOCAB_WORDS } from '../data/vocab.js';
  import TensorShapeTrace from './TensorShapeTrace.svelte';
  import CameraStage from './CameraStage.svelte';
  import VectorHeatmap from './VectorHeatmap.svelte';
  import BeforeAfterSummary from './BeforeAfterSummary.svelte';
  import QuickCheck from './QuickCheck.svelte';

  $: sceneId = $currentScene?.id;
  $: copy = getSceneCopy(sceneId);

  // Sub-step index definitions
  const STEP = { STATS: 0, NORMALIZE: 1, SCALESHIFT: 2, SUMMARY: 3, QUIZ: 4 };

  // Expose configuration variables
  export let inputStageId = undefined;
  export let outputStageId = undefined;
  export let gammaKey = undefined;
  export let betaKey = undefined;
  export let highlightId = undefined;
  export let title = '';

  $: activeInputStageId = inputStageId || (sceneId === 'layer-norm-2' ? 'residual-2' : 'residual-1');
  $: activeOutputStageId = outputStageId || sceneId || 'layer-norm-1';
  $: activeHighlightId = highlightId || (sceneId === 'layer-norm-2' ? 'eq-layernorm-2' : 'eq-layernorm-1');
  $: activeGammaKey = gammaKey || (sceneId === 'layer-norm-2' ? 'ln2_gamma' : 'ln1_gamma');
  $: activeBetaKey = betaKey || (sceneId === 'layer-norm-2' ? 'ln2_beta' : 'ln1_beta');

  $: activeTitle = title || (sceneId === 'layer-norm-2' ? 'Layer Normalization ②' : 'Layer Normalization ①');

  const DEFAULT_INTERACTIVE_SENTENCE = ['cat', 'chased', 'dog'];
  const MAX_INTERACTIVE_TOKENS = 6;
  let interactiveSentence = [...DEFAULT_INTERACTIVE_SENTENCE];

  function addWord(word) {
    if (interactiveSentence.length >= MAX_INTERACTIVE_TOKENS) return;
    interactiveSentence = [...interactiveSentence, word];
  }
  function removeWord(index) {
    interactiveSentence = interactiveSentence.filter((_, i) => i !== index);
  }
  function resetSentence() {
    interactiveSentence = [...DEFAULT_INTERACTIVE_SENTENCE];
  }

  $: activeSentence = $dataMode === 'lecture'
    ? ($forwardPassData?.meta?.sentence ?? [])
    : interactiveSentence;
  $: currentDModel = $dataMode === 'lecture' ? ($forwardPassData?.meta?.dModel ?? 16) : $configDModel;
  $: seqLen = activeSentence.length;

  // --- Dynamic calculations using centralized pipeline ---
  $: interactiveData = computeAttentionPipeline(
    activeSentence,
    currentDModel,
    4,
    currentDModel / 4,
    $forwardPassData?.weights ?? {}
  );

  $: precomputedInput = $forwardPassData?.stages?.find((s) => s.id === activeInputStageId)?.tokenVectors ?? [];
  $: precomputedOutput = $forwardPassData?.stages?.find((s) => s.id === activeOutputStageId)?.tokenVectors ?? [];

  // Input representation (Residual Output)
  $: inputMatrix = $dataMode === 'lecture'
    ? precomputedInput
    : (sceneId === 'layer-norm-2'
        ? (precomputedInput.length ? precomputedInput.map(r => r.slice(0, currentDModel)) : Array.from({ length: seqLen }, () => new Array(currentDModel).fill(0))) // fallback
        : (interactiveData?.residual1 ?? [])
      );

  // Computing live Mean and Variance
  $: statistics = (() => {
    const stats = [];
    const eps = 1e-5;
    for (let i = 0; i < seqLen; i++) {
      const row = inputMatrix[i] ?? [];
      if (!row.length) {
        stats.push({ mean: 0, variance: 0, stdDev: 0, normRow: [] });
        continue;
      }
      const mean = row.reduce((a, b) => a + b, 0) / row.length;
      const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
      const stdDev = Math.sqrt(variance + eps);
      const normRow = row.map((v) => (v - mean) / stdDev);
      stats.push({ mean, variance, stdDev, normRow });
    }
    return stats;
  })();

  // Output representation
  $: outputMatrix = $dataMode === 'lecture'
    ? precomputedOutput
    : (sceneId === 'layer-norm-2'
        ? (precomputedOutput.length ? precomputedOutput.map(r => r.slice(0, currentDModel)) : Array.from({ length: seqLen }, () => new Array(currentDModel).fill(0)))
        : (interactiveData?.ln1_outputs ?? [])
      );

  // Gamma and Beta parameters
  $: gamma = $forwardPassData?.weights?.[activeGammaKey] ?? Array(currentDModel).fill(1.0);
  $: beta = $forwardPassData?.weights?.[activeBetaKey] ?? Array(currentDModel).fill(0.0);

  // Active statistics for hovered token
  let activeHoverIdx = 0;

  function handleHover(i) {
    activeHoverIdx = i;
    setHighlight(activeHighlightId);
  }

  function handleUnhover() {
    clearHighlight();
  }

  // --- Camera Choreography ---
  const WIDE = { x: 0, y: 0, scale: 1 };
  $: {
    $replayTick;
    focusShot(WIDE);
  }

  $: sentenceKey = [$dataMode, activeSentence.join('|'), $replayTick].join('::');
  $: stageKey = [$dataMode, activeSentence.join('|'), $subStepIndex, $replayTick].join('::');

  // Trace steps setup
  $: traceSteps = [
    { label: 'Residual Output', shape: [seqLen, currentDModel], highlightId: activeHighlightId },
    { label: 'LayerNorm', shape: [seqLen, currentDModel], highlightId: activeHighlightId },
    { label: 'Output', shape: [seqLen, currentDModel], highlightId: activeHighlightId }
  ];
</script>

<CameraStage>
  <div class="layernorm-scene-wrap">
    {#if $dataMode === 'interactive'}
      <div class="word-picker" transition:fade={{ duration: motionMs(150) }}>
        <span class="hint">Build a sentence:</span>
        {#each VOCAB_WORDS as w}
          <button class="chip" on:click={() => addWord(w)} disabled={interactiveSentence.length >= MAX_INTERACTIVE_TOKENS}>
            + {w}
          </button>
        {/each}
        <button class="chip reset" on:click={resetSentence}>Reset</button>
      </div>
    {/if}

    {#key stageKey}
      <div class="main-content">
        {#if $subStepIndex < STEP.SUMMARY}
          <div class="layout-container" in:fade={{ duration: motionMs(300) }}>
            
            <!-- Left: Input Matrix (Residual Output) -->
            <div class="matrix-card" class:dimmed={$subStepIndex > STEP.STATS}>
              <h4>Residual Output (Input X)</h4>
              <div class="matrix-grid">
                {#each inputMatrix as row, rIdx}
                  <div
                    class="matrix-row interactive-row"
                    role="presentation"
                    class:hovered={activeHoverIdx === rIdx}
                    on:mouseenter={() => handleHover(rIdx)}
                    on:mouseleave={handleUnhover}
                  >
                    <span class="token-label" role="button" tabindex="0" on:click={() => removeWord(rIdx)} on:keydown={(e) => e.key === 'Enter' && removeWord(rIdx)}>
                      {activeSentence[rIdx] ?? ''}
                      {#if $dataMode === 'interactive'}<span class="delete-icon">×</span>{/if}
                    </span>
                    <VectorHeatmap vector={row} />
                  </div>
                {/each}
              </div>
              <div class="matrix-shape">[{seqLen}, {currentDModel}]</div>
            </div>

            <!-- Arrow connector -->
            <div class="arrow-connector">➔</div>

            <!-- Middle: Statistics Panel -->
            <div class="stats-panel" class:highlighted={$subStepIndex === STEP.STATS}>
              <h4>Row-wise Statistics</h4>
              {#if statistics[activeHoverIdx]}
                {@const s = statistics[activeHoverIdx]}
                <div class="stats-box">
                  <div class="token-display">Hovered Token: <strong>{activeSentence[activeHoverIdx] ?? 'None'}</strong></div>
                  <div class="stat-item">
                    <span class="symbol">μ (Mean):</span>
                    <span class="value">{s.mean.toFixed(4)}</span>
                  </div>
                  <div class="stat-item">
                    <span class="symbol">σ² (Variance):</span>
                    <span class="value">{s.variance.toFixed(4)}</span>
                  </div>
                  <div class="stat-item">
                    <span class="symbol">σ (Std Dev):</span>
                    <span class="value">{s.stdDev.toFixed(4)}</span>
                  </div>

                  <div class="mini-vector-wrap">
                    <div class="mini-label">Normalized Row Vector:</div>
                    <VectorHeatmap vector={s.normRow} size="small" />
                  </div>
                </div>
              {:else}
                <div class="stats-box empty">Hover a token row to view parameters</div>
              {/if}
              <div class="caption">Computed independently per row</div>
            </div>

            <!-- Arrow connector -->
            <div class="arrow-connector">➔</div>

            <!-- Right: Normalized or Scale-shifted Output Matrix -->
            <div class="matrix-card output-card" class:highlighted={$subStepIndex >= STEP.NORMALIZE}>
              {#if $subStepIndex === STEP.STATS}
                <h4>Calculating parameters...</h4>
                <div class="placeholder-grid">
                  {#each inputMatrix as row}
                    <div class="placeholder-row"></div>
                  {/each}
                </div>
              {:else}
                <h4>
                  {$subStepIndex === STEP.NORMALIZE ? 'Normalized Matrix (X̂)' : 'Scale & Shifted Output (Y)'}
                </h4>
                <div class="matrix-grid">
                  {#each ($subStepIndex === STEP.NORMALIZE ? statistics.map(s => s.normRow) : outputMatrix) as row, rIdx}
                    <div class="matrix-row" class:hovered={activeHoverIdx === rIdx}>
                      <span class="token-label">{activeSentence[rIdx] ?? ''}</span>
                      <VectorHeatmap vector={row} />
                    </div>
                  {/each}
                </div>
              {/if}
              <div class="matrix-shape">[{seqLen}, {currentDModel}]</div>
            </div>

          </div>
        {/if}

        {#if $subStepIndex === STEP.SUMMARY}
          <div class="beat" in:fade={{ duration: motionMs(300) }}>
            {#if copy.beforeAfter}
              <BeforeAfterSummary
                before={{ label: copy.beforeAfter.before.label, shape: [seqLen, currentDModel] }}
                after={{ label: copy.beforeAfter.after.label, shape: [seqLen, currentDModel] }}
                whatChanged={copy.beforeAfter.whatChanged}
                structured={copy.beforeAfter.structured}
              />
            {/if}
          </div>
        {/if}

        {#if $subStepIndex === STEP.QUIZ}
          <div class="beat" in:fade={{ duration: motionMs(300) }}>
            {#if copy.quickCheck}
              <QuickCheck
                question={copy.quickCheck.question}
                choices={copy.quickCheck.choices}
                correctIndex={copy.quickCheck.correctIndex}
                explanation={copy.quickCheck.explanation}
                transition={copy.quickCheck.transition}
                distractorNotes={copy.quickCheck.distractorNotes}
              />
            {/if}
          </div>
        {/if}

        <!-- Tensor shape trace panel -->
        <div class="shape-trace-wrap">
          <TensorShapeTrace
            steps={traceSteps}
            activeIndex={$subStepIndex === STEP.STATS ? 0 : ($subStepIndex === STEP.NORMALIZE ? 1 : 2)}
          />
        </div>
      </div>
    {/key}
  </div>
</CameraStage>

<style>
  .layernorm-scene-wrap {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    padding: 1rem;
  }

  .main-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    position: relative;
  }

  .layout-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    width: 100%;
    margin-bottom: 2rem;
  }

  .matrix-card {
    background: rgba(25d, 25d, 25d, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
    transition: opacity 0.3s, border-color 0.3s, transform 0.3s;
  }

  .matrix-card.dimmed {
    opacity: 0.25;
  }

  .matrix-card.highlighted {
    border-color: var(--accent, #7aa2ff);
  }

  .output-card.highlighted {
    border-color: var(--accent-3, #7ee787);
    box-shadow: 0 0 12px rgba(126, 231, 135, 0.12);
  }

  h4 {
    margin: 0 0 0.6rem 0;
    font-size: 0.85rem;
    color: var(--text-muted, #8a8a8a);
    font-weight: 500;
  }

  .matrix-grid {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .matrix-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 2px 4px;
    border-radius: 4px;
    transition: background-color 0.15s;
  }

  .matrix-row.hovered {
    background: rgba(255, 255, 255, 0.06);
  }

  .matrix-row.interactive-row {
    cursor: pointer;
  }

  .token-label {
    min-width: 4rem;
    text-align: right;
    font-family: var(--font-mono, monospace);
    font-size: 0.75rem;
    color: var(--text-main, #e0e0e0);
    position: relative;
    cursor: pointer;
  }

  .token-label:hover .delete-icon {
    opacity: 1;
  }

  .delete-icon {
    position: absolute;
    right: -10px;
    top: 50%;
    transform: translateY(-50%);
    color: #ff5555;
    font-weight: bold;
    opacity: 0;
    transition: opacity 0.15s;
  }

  .matrix-shape {
    font-size: 0.7rem;
    color: var(--text-muted, #8a8a8a);
    margin-top: 0.5rem;
    font-family: var(--font-mono, monospace);
  }

  .arrow-connector {
    font-size: 1.5rem;
    color: var(--text-muted, #8a8a8a);
    user-select: none;
  }

  .stats-panel {
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 0.8rem;
    width: 260px;
    text-align: left;
    transition: border-color 0.3s, box-shadow 0.3s;
  }

  .stats-panel.highlighted {
    border-color: var(--accent-2, #ffb86b);
    box-shadow: 0 0 12px rgba(255, 184, 107, 0.12);
  }

  .stats-box {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 6px;
    padding: 0.6rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .stats-box.empty {
    color: var(--text-muted, #8a8a8a);
    font-size: 0.8rem;
    text-align: center;
    padding: 2rem 0;
  }

  .token-display {
    font-size: 0.8rem;
    color: var(--text-main, #e0e0e0);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 0.3rem;
    margin-bottom: 0.2rem;
  }

  .stat-item {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
  }

  .stat-item .symbol {
    color: var(--text-muted, #8a8a8a);
    font-family: var(--font-mono, monospace);
  }

  .stat-item .value {
    color: var(--accent-2, #ffb86b);
    font-weight: 600;
    font-family: var(--font-mono, monospace);
  }

  .mini-vector-wrap {
    margin-top: 0.4rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-top: 0.4rem;
  }

  .mini-label {
    font-size: 0.7rem;
    color: var(--text-muted, #8a8a8a);
    margin-bottom: 0.3rem;
  }

  .stats-panel .caption {
    font-size: 0.65rem;
    color: var(--text-muted, #8a8a8a);
    margin-top: 0.6rem;
    text-align: center;
  }

  .placeholder-grid {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .placeholder-row {
    height: 1.4rem;
    width: 9rem;
    background: rgba(255, 255, 255, 0.02);
    border: 1px dashed rgba(255, 255, 255, 0.08);
    border-radius: 2px;
  }

  .word-picker {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
    background: rgba(0, 0, 0, 0.2);
    padding: 0.5rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 0.5rem;
    width: 100%;
    max-width: 700px;
  }

  .word-picker .hint {
    font-size: 0.8rem;
    color: var(--text-muted, #8a8a8a);
  }

  .chip {
    background: rgba(255, 255, 255, 0.08);
    border: none;
    border-radius: 4px;
    color: var(--text-main, #e0e0e0);
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    cursor: pointer;
    transition: background 0.15s;
  }

  .chip:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.15);
  }

  .chip.reset {
    background: rgba(255, 85, 85, 0.15);
    color: #ff5555;
  }

  .chip.reset:hover {
    background: rgba(255, 85, 85, 0.25);
  }

  .beat {
    width: 100%;
    max-width: 600px;
    margin-bottom: 2rem;
  }

  .shape-trace-wrap {
    margin-top: 1rem;
    width: 100%;
    display: flex;
    justify-content: center;
  }
</style>
