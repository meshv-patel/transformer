<script>
  import { fade } from 'svelte/transition';
  import { subStepIndex, replayTick, dataMode, currentScene } from '../core/stores/sceneStore.js';
  import { dModel as configDModel } from '../core/stores/configStore.js';
  import { forwardPassData } from '../core/data-loader.js';
  import { highlightedTermId, setHighlight, clearHighlight } from '../core/stores/highlightStore.js';
  import { computeAttentionPipeline, getInteractiveWeights2D, getInteractiveBias1D } from '../core/tensor-ops.js';
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
  const STEP = { LINEAR1: 0, ACTIVATION: 1, LINEAR2: 2, SUMMARY: 3, QUIZ: 4 };

  // Expose configuration variables
  export let inputStageId = undefined;
  export let outputStageId = undefined;
  export let weight1Key = undefined;
  export let bias1Key = undefined;
  export let weight2Key = undefined;
  export let bias2Key = undefined;
  export let activationType = 'relu';
  export let highlightId = undefined;
  export let title = '';

  // Reactively resolve parameters based on props or sceneId
  $: activeInputStageId = inputStageId || 'layer-norm-1';
  $: activeOutputStageId = outputStageId || 'ffn';
  $: activeWeight1Key = weight1Key || 'W_ff1';
  $: activeBias1Key = bias1Key || 'b_ff1';
  $: activeWeight2Key = weight2Key || 'W_ff2';
  $: activeBias2Key = bias2Key || 'b_ff2';
  $: activeHighlightId = highlightId || 'eq-ffn';
  $: activeTitle = title || 'Feed Forward Network (FFN)';

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
  $: dFF = currentDModel * 2;

  // --- Dynamic calculations using centralized pipeline ---
  $: interactiveData = computeAttentionPipeline({
    encoderSentence: activeSentence,
    dModel: currentDModel,
    numHeads: 4,
    lectureWeights: $forwardPassData?.weights ?? {}
  });

  $: precomputedInput = $forwardPassData?.stages?.find((s) => s.id === activeInputStageId)?.tokenVectors ?? [];
  $: precomputedOutput = $forwardPassData?.stages?.find((s) => s.id === activeOutputStageId)?.tokenVectors ?? [];

  // FFN Inputs
  $: inputMatrix = $dataMode === 'lecture'
    ? precomputedInput
    : (interactiveData?.ln1_outputs ?? []);

  // FFN intermediate: Linear 1 Output
  $: linear1Matrix = $dataMode === 'lecture'
    ? ($forwardPassData?.stages?.find((s) => s.id === activeOutputStageId)?.linear1 ?? [])
    : (interactiveData?.ffn_linear1 ?? []);

  // FFN intermediate: Activated Output
  $: activationMatrix = $dataMode === 'lecture'
    ? ($forwardPassData?.stages?.find((s) => s.id === activeOutputStageId)?.activated ?? [])
    : (interactiveData?.ffn_activation ?? []);

  // FFN outputs
  $: outputMatrix = $dataMode === 'lecture'
    ? precomputedOutput
    : (interactiveData?.ffn_outputs ?? []);

  // Active hover highlights
  let activeHoverIdx = 0;
  let hoveredColIdx = null;

  function handleHover(r, c = null) {
    activeHoverIdx = r;
    hoveredColIdx = c;
    setHighlight(activeHighlightId);
  }

  function handleUnhover() {
    hoveredColIdx = null;
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
    { label: 'Input to FFN', shape: [seqLen, currentDModel], highlightId: activeHighlightId },
    { label: `Expansion (d_ff)`, shape: [seqLen, dFF], highlightId: activeHighlightId },
    { label: 'Output (d_model)', shape: [seqLen, currentDModel], highlightId: activeHighlightId }
  ];
</script>

<CameraStage>
  <div class="ffn-scene-wrap">
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
            
            <!-- Matrix 1: Input to FFN (d_model) -->
            <div class="matrix-card" class:dimmed={$subStepIndex > STEP.LINEAR1}>
              <h4>Input ({currentDModel}d)</h4>
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

            <!-- Matrix 2: Intermediate Grid (d_ff) -->
            <div
              class="matrix-card intermediate-card"
              class:highlighted={$subStepIndex === STEP.LINEAR1 || $subStepIndex === STEP.ACTIVATION}
              class:dimmed={$subStepIndex > STEP.ACTIVATION}
            >
              <h4>
                {#if $subStepIndex === STEP.LINEAR1}
                  Expanded Linear 1 ({dFF}d)
                {:else}
                  Activated {activationType === 'relu' ? 'ReLU' : 'GELU'} ({dFF}d)
                {/if}
              </h4>
              <div class="matrix-grid">
                {#each ($subStepIndex === STEP.LINEAR1 ? linear1Matrix : activationMatrix) as row, rIdx}
                  <div
                    class="matrix-row interactive-row"
                    role="presentation"
                    class:hovered={activeHoverIdx === rIdx}
                    on:mouseenter={() => handleHover(rIdx)}
                    on:mouseleave={handleUnhover}
                  >
                    <span class="token-label">{activeSentence[rIdx] ?? ''}</span>
                    <VectorHeatmap vector={row} size="small" />
                  </div>
                {/each}
              </div>
              <div class="matrix-shape">[{seqLen}, {dFF}]</div>
            </div>

            <!-- Arrow connector -->
            <div class="arrow-connector">➔</div>

            <!-- Matrix 3: Final Output Grid (d_model) -->
            <div class="matrix-card output-card" class:highlighted={$subStepIndex === STEP.LINEAR2}>
              <h4>Output Projection ({currentDModel}d)</h4>
              <div class="matrix-grid">
                {#each outputMatrix as row, rIdx}
                  <div
                    class="matrix-row interactive-row"
                    role="presentation"
                    class:hovered={activeHoverIdx === rIdx}
                    on:mouseenter={() => handleHover(rIdx)}
                    on:mouseleave={handleUnhover}
                  >
                    <span class="token-label">{activeSentence[rIdx] ?? ''}</span>
                    <VectorHeatmap vector={row} />
                  </div>
                {/each}
              </div>
              <div class="matrix-shape">[{seqLen}, {currentDModel}]</div>
            </div>

            <!-- Side panel for showing specific value activations -->
            {#if $subStepIndex === STEP.ACTIVATION}
              <div class="activation-side-panel" transition:fade={{ duration: motionMs(200) }}>
                <h4>Activation details</h4>
                <div class="detail-box">
                  <div class="token-header">Token: <strong>{activeSentence[activeHoverIdx] ?? 'None'}</strong></div>
                  
                  <!-- Display value mapping detail -->
                  {#if linear1Matrix[activeHoverIdx] && activationMatrix[activeHoverIdx]}
                    {@const rawVal = linear1Matrix[activeHoverIdx][0] ?? 0}
                    {@const actVal = activationMatrix[activeHoverIdx][0] ?? 0}
                    <div class="demo-math">
                      <div class="formula">{activationType === 'relu' ? 'ReLU' : 'GELU'}(x) = {activationType === 'relu' ? 'max(0, x)' : 'x * cdf(x)'}</div>
                      <div class="example-row">
                        <span class="label">Raw value x:</span>
                        <span class="val" class:negative={rawVal < 0}>{rawVal.toFixed(4)}</span>
                      </div>
                      <div class="example-row">
                        <span class="label">Activated {activationType === 'relu' ? 'ReLU' : 'GELU'}(x):</span>
                        <span class="val activated" class:zeroed={actVal === 0}>{actVal.toFixed(4)}</span>
                      </div>
                    </div>
                    <div class="suppression-info">
                      {#if rawVal < 0}
                        <span class="alert negative">✖ Negative value suppressed to 0.0</span>
                      {:else}
                        <span class="alert positive">✔ Positive value preserved</span>
                      {/if}
                    </div>
                  {/if}
                </div>
              </div>
            {/if}

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
            activeIndex={$subStepIndex === STEP.LINEAR1 ? 1 : ($subStepIndex === STEP.LINEAR2 ? 2 : 1)}
          />
        </div>
      </div>
    {/key}
  </div>
</CameraStage>

<style>
  .ffn-scene-wrap {
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

  .intermediate-card.highlighted {
    border-color: var(--accent-2, #ffb86b);
    box-shadow: 0 0 12px rgba(255, 184, 107, 0.12);
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

  .activation-side-panel {
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 0.8rem;
    width: 240px;
    text-align: left;
  }

  .detail-box {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 6px;
    padding: 0.6rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .token-header {
    font-size: 0.8rem;
    color: var(--text-main, #e0e0e0);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 0.3rem;
  }

  .demo-math {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    font-size: 0.75rem;
  }

  .formula {
    font-family: var(--font-mono, monospace);
    color: var(--accent, #7aa2ff);
    margin-bottom: 0.2rem;
  }

  .example-row {
    display: flex;
    justify-content: space-between;
  }

  .example-row .label {
    color: var(--text-muted, #8a8a8a);
  }

  .example-row .val {
    font-family: var(--font-mono, monospace);
    font-weight: 600;
  }

  .example-row .val.negative {
    color: #ff5555;
  }

  .example-row .val.activated {
    color: var(--accent-3, #7ee787);
  }

  .example-row .val.activated.zeroed {
    color: #ff5555;
  }

  .suppression-info {
    font-size: 0.7rem;
    margin-top: 0.4rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-top: 0.4rem;
    text-align: center;
  }

  .suppression-info .alert.negative {
    color: #ff5555;
  }

  .suppression-info .alert.positive {
    color: var(--accent-3, #7ee787);
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
