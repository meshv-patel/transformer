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
  const STEP = { REPRESENTATION: 0, CONTEXT: 1, APPLICATIONS: 2, SUMMARY: 3, QUIZ: 4 };

  // Expose configuration variables
  export let inputStageId = 'layer-norm-2';
  export let title = 'Encoder Output';
  export let highlightId = 'eq-encoder-output';
  export let destinationLabels = ['Decoder', 'Classification Head', 'Next Transformer Block'];

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
  $: interactiveData = computeAttentionPipeline({
    encoderSentence: activeSentence,
    dModel: currentDModel,
    numHeads: 4,
    lectureWeights: $forwardPassData?.weights ?? {}
  });

  $: precomputedInput = $forwardPassData?.stages?.find((s) => s.id === inputStageId)?.tokenVectors ?? [];

  // Final representations (LayerNorm-2 output)
  $: outputMatrix = $dataMode === 'lecture'
    ? precomputedInput
    : (interactiveData?.ln2_outputs ?? []);

  // Active hover highlights
  let activeHoverIdx = 0;

  function handleHover(i) {
    activeHoverIdx = i;
    setHighlight(highlightId);
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
    { label: 'LayerNorm-2 Output', shape: [seqLen, currentDModel], highlightId: highlightId },
    { label: title, shape: [seqLen, currentDModel], highlightId: highlightId }
  ];
</script>

<CameraStage>
  <div class="encoder-output-scene-wrap">
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
            
            <!-- Left: Encoder Output Matrix -->
            <div class="matrix-card" class:highlighted={$subStepIndex === STEP.REPRESENTATION || $subStepIndex === STEP.CONTEXT}>
              <h4>{title} Matrix (LayerNorm ② Output)</h4>
              <div class="matrix-grid">
                {#each outputMatrix as row, rIdx}
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

            <!-- Middle/Right: Application Destinations -->
            {#if $subStepIndex >= STEP.CONTEXT}
              <div class="arrow-connector" transition:fade={{ duration: motionMs(200) }}>➔</div>

              <div class="destinations-panel" class:highlighted={$subStepIndex === STEP.APPLICATIONS} transition:fade={{ duration: motionMs(200) }}>
                {#if $subStepIndex === STEP.CONTEXT}
                  <h4>Contextual Integration</h4>
                  <div class="info-box">
                    <p>Every token vector has now aggregated context from all other tokens in the sentence.</p>
                    <p>For example, the representation for <strong>{activeSentence[activeHoverIdx] ?? 'None'}</strong> contains information mixed from all positions via Self-Attention.</p>
                  </div>
                {:else}
                  <h4>Downstream Applications</h4>
                  <div class="destinations-list">
                    {#each destinationLabels as dest}
                      <div class="destination-item">
                        <span class="bullet">✦</span>
                        <span class="name">{dest}</span>
                      </div>
                    {/each}
                  </div>
                {/if}
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
            activeIndex={$subStepIndex === STEP.REPRESENTATION ? 0 : 1}
          />
        </div>
      </div>
    {/key}
  </div>
</CameraStage>

<style>
  .encoder-output-scene-wrap {
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

  .matrix-card.highlighted {
    border-color: var(--accent, #7aa2ff);
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

  .destinations-panel {
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 0.8rem;
    width: 260px;
    text-align: left;
    transition: border-color 0.3s, box-shadow 0.3s;
  }

  .destinations-panel.highlighted {
    border-color: var(--accent-3, #7ee787);
    box-shadow: 0 0 12px rgba(126, 231, 135, 0.12);
  }

  .info-box {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 6px;
    padding: 0.6rem;
    font-size: 0.75rem;
    color: var(--text-main, #e0e0e0);
    line-height: 1.35;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .info-box p {
    margin: 0;
  }

  .destinations-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .destination-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    color: var(--text-main, #e0e0e0);
  }

  .destination-item .bullet {
    color: var(--accent-3, #7ee787);
    font-size: 0.85rem;
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
